"""Optional real-model coverage using redistributable LibriSpeech clips."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sortformer_2spk"
FIXTURE_SHA256 = {
    "6930-75918-0000.flac": "9ce35224156f071ab58eb7feb8a5ceae600f6f9f353da2a6cbf797b6b1ac8a23",
    "7902-96591-0000.flac": "955d0cc57211f03986124307ade29fbab521e26133caa2d2831b9a1b38c7a9b6",
}
VALIDATED_MODEL_REVISION = "5240a64075176943f677d30fa2171c780229f341"
MODEL_REPO_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--nvidia--diar_streaming_sortformer_4spk-v2"
)


def _local_sortformer_model() -> Path | None:
    override = os.environ.get("WLK_SORTFORMER_MODEL_PATH")
    if override:
        return Path(override).expanduser()
    validated_model = (
        MODEL_REPO_CACHE
        / "snapshots"
        / VALIDATED_MODEL_REVISION
        / "diar_streaming_sortformer_4spk-v2.nemo"
    )
    return validated_model if validated_model.is_file() else None


def _build_two_speaker_signal() -> tuple[np.ndarray, int, dict[str, tuple[float, float]]]:
    speaker_a, sample_rate_a = sf.read(
        FIXTURE_DIR / "6930-75918-0000.flac",
        dtype="float32",
    )
    speaker_b, sample_rate_b = sf.read(
        FIXTURE_DIR / "7902-96591-0000.flac",
        dtype="float32",
    )
    assert sample_rate_a == sample_rate_b == 16_000
    assert speaker_a.ndim == speaker_b.ndim == 1

    silence = np.zeros(sample_rate_a // 2, dtype=np.float32)
    overlap = 0.5 * speaker_a[: len(speaker_b)] + 0.5 * speaker_b
    signal = np.concatenate(
        [
            speaker_a,
            silence,
            speaker_b,
            silence,
            speaker_a,
            silence,
            overlap,
        ]
    )
    signal = np.pad(signal, (0, (-len(signal)) % sample_rate_a))

    windows = {
        "speaker_a_first": (0.25, 3.25),
        "speaker_b": (4.25, 5.85),
        "speaker_a_return": (7.0, 9.85),
        "overlap": (10.85, 12.45),
    }
    return signal, sample_rate_a, windows


def _prediction_window(
    predictions: np.ndarray,
    frames_per_second: float,
    bounds: tuple[float, float],
) -> np.ndarray:
    start, end = bounds
    first = math.ceil(start * frames_per_second)
    last = math.floor(end * frames_per_second)
    return predictions[first:last]


def _speaker_with_most_overlap(segments, bounds: tuple[float, float]) -> int:
    start, end = bounds
    overlap_by_speaker = {}
    for segment in segments:
        overlap = max(0.0, min(end, segment.end) - max(start, segment.start))
        if overlap:
            speaker = int(segment.speaker)
            overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + overlap
    assert overlap_by_speaker
    return max(overlap_by_speaker, key=overlap_by_speaker.get)


async def _run_session(online, signal: np.ndarray, sample_rate: int):
    segments = []
    prediction_chunks = []
    for start in range(0, len(signal), sample_rate):
        online.insert_audio_chunk(signal[start : start + sample_rate])
        segments.extend(await online.diarize())
        prediction_length = online._len_prediction
        prediction_chunks.append(
            online.total_preds[0, -prediction_length:, :].detach().cpu().numpy()
        )
    return segments, np.concatenate(prediction_chunks)


@pytest.mark.parametrize(("filename", "expected_sha256"), FIXTURE_SHA256.items())
def test_sortformer_fixture_matches_official_librispeech_bytes(
    filename: str,
    expected_sha256: str,
):
    path = FIXTURE_DIR / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256

    info = sf.info(path)
    assert info.samplerate == 16_000
    assert info.channels == 1
    assert info.subtype == "PCM_16"


_MODEL_PATH = _local_sortformer_model()
_NEMO_AVAILABLE = importlib.util.find_spec("nemo") is not None


@pytest.mark.skipif(
    not _NEMO_AVAILABLE or _MODEL_PATH is None,
    reason=(
        "requires NeMo and a cached Sortformer 4spk model, or set "
        "WLK_SORTFORMER_MODEL_PATH"
    ),
)
@pytest.mark.asyncio
async def test_real_two_speaker_cap_is_stable_through_chunks_and_overlap():
    from whisperlivekit.diarization.sortformer_backend import (
        SortformerDiarization,
        SortformerDiarizationOnline,
    )

    signal, sample_rate, windows = _build_two_speaker_signal()
    shared = SortformerDiarization(model_path=str(_MODEL_PATH))
    capped = SortformerDiarizationOnline(shared_model=shared, max_speakers=2)
    default = SortformerDiarizationOnline(shared_model=shared)

    capped_segments, capped_predictions = await _run_session(
        capped,
        signal,
        sample_rate,
    )

    assert capped.max_speakers == 2
    assert default.max_speakers == 4

    # Use the same real prediction tensor for a bit-exact comparison. Separate
    # inference passes include random preprocessor dither that is unrelated to
    # max_speakers. The legacy implementation used argmax over every channel.
    legacy_ids = np.argmax(capped_predictions, axis=1)
    default_ids = np.argmax(
        capped_predictions[:, :default.max_speakers],
        axis=1,
    )
    capped_ids = np.argmax(
        capped_predictions[:, :capped.max_speakers],
        axis=1,
    )
    np.testing.assert_array_equal(default_ids, legacy_ids)
    np.testing.assert_array_equal(capped_ids, default_ids)

    emitted_speakers = {int(segment.speaker) for segment in capped_segments}
    assert emitted_speakers == {0, 1}
    assert [
        _speaker_with_most_overlap(capped_segments, windows[name])
        for name in ("speaker_a_first", "speaker_b", "speaker_a_return")
    ] == [0, 1, 0]

    duration = len(signal) / sample_rate
    frames_per_second = len(capped_predictions) / duration
    assert set(legacy_ids).issubset({0, 1})
    assert not np.any(capped_predictions[:, 2:] > 0.5)
    assert capped.streaming_state.spk_perm is None

    speaker_a_first = _prediction_window(
        capped_predictions,
        frames_per_second,
        windows["speaker_a_first"],
    )
    speaker_b = _prediction_window(
        capped_predictions,
        frames_per_second,
        windows["speaker_b"],
    )
    speaker_a_return = _prediction_window(
        capped_predictions,
        frames_per_second,
        windows["speaker_a_return"],
    )
    overlap = _prediction_window(
        capped_predictions,
        frames_per_second,
        windows["overlap"],
    )

    assert np.mean(np.argmax(speaker_a_first, axis=1) == 0) >= 0.95
    assert np.mean(np.argmax(speaker_b, axis=1) == 1) >= 0.95
    assert np.mean(np.argmax(speaker_a_return, axis=1) == 0) >= 0.90

    # Sortformer exposes overlap as independent sigmoid activity. WLK's
    # existing argmax output remains one speaker per frame and token.
    assert np.mean(overlap[:, 0] > 0.5) >= 0.50
    assert np.mean(overlap[:, 1] > 0.5) >= 0.50
