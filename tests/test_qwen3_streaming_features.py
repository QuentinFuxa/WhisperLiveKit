"""StreamingMelExtractor parity with one-shot WhisperFeatureExtractor output."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from whisperlivekit.qwen3_streaming.features import StreamingMelExtractor  # noqa: E402


@pytest.fixture(scope="module")
def feature_extractor():
    # Constructed locally (no network); matches the Qwen3-ASR extractor shape.
    return transformers.WhisperFeatureExtractor(feature_size=128)


def one_shot_mel(feature_extractor, audio):
    features = feature_extractor(
        audio,
        sampling_rate=16_000,
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )["input_features"][0]
    return features.transpose(0, 1).contiguous()


def speech_like_audio(n_samples: int, seed: int = 0) -> np.ndarray:
    """Noise plus periodic loud bursts, so energy is spread like real speech.

    The window-local dynamic-range clamp only matches the file-global one when
    every featurized window contains a near-max frame — true for speech, not
    for a single isolated onset (that divergence is the documented
    approximation of StreamingMelExtractor).
    """
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(n_samples) * 0.05).astype(np.float32)
    burst = np.sin(np.linspace(0, 440 * np.pi, 1600)).astype(np.float32) * 0.8
    for start in range(0, n_samples - 1600, 8000):  # one burst every 0.5s
        audio[start : start + 1600] += burst
    return audio


def collect_streaming_frames(extractor, audio, chunk_sizes):
    frames = []
    cursor = 0
    idx = 0
    while cursor < len(audio):
        size = chunk_sizes[idx % len(chunk_sizes)]
        idx += 1
        out = extractor.append(audio[cursor : cursor + size])
        if out is not None:
            frames.append(out[0])
        cursor += size
    tail = extractor.flush()
    if tail is not None:
        frames.append(tail[0])
    return torch.cat(frames, dim=0) if frames else torch.zeros(0, 128)


def test_streaming_matches_one_shot_on_irregular_chunks(feature_extractor):
    audio = speech_like_audio(16_000 * 3 + 731)
    reference = one_shot_mel(feature_extractor, audio)

    extractor = StreamingMelExtractor(feature_extractor)
    streamed = collect_streaming_frames(
        extractor, audio, chunk_sizes=[2_771, 16_000, 5_000, 160, 36_800]
    )

    assert streamed.shape == reference.shape
    # Frame positions must be sample-exact: virtually all frames identical.
    per_frame_diff = (streamed - reference).abs().amax(dim=1)
    exact_ratio = float((per_frame_diff <= 1e-4).float().mean())
    assert exact_ratio >= 0.99, f"only {exact_ratio:.3f} of frames are sample-exact"
    # The rest may drift by the window-local vs file-global clamp delta, which
    # stays tiny on speech-like audio (documented approximation).
    assert float(per_frame_diff.max()) < 2e-2


def test_tail_frames_held_back_until_flush(feature_extractor):
    extractor = StreamingMelExtractor(feature_extractor)
    audio = speech_like_audio(16_000)

    out = extractor.append(audio)
    emitted_before_flush = int(out.shape[1]) if out is not None else 0
    total_frames = len(audio) // 160
    assert emitted_before_flush < total_frames

    tail = extractor.flush()
    emitted_total = emitted_before_flush + (int(tail.shape[1]) if tail is not None else 0)
    assert emitted_total == total_frames


def test_tiny_audio_emits_nothing_until_flush(feature_extractor):
    extractor = StreamingMelExtractor(feature_extractor)
    assert extractor.append(np.zeros(100, dtype=np.float32)) is None
    tail = extractor.flush()
    # 100 samples < one hop: no frames at all
    assert tail is None


def test_reset_clears_state(feature_extractor):
    extractor = StreamingMelExtractor(feature_extractor)
    extractor.append(speech_like_audio(16_000))
    extractor.reset()
    assert extractor.emitted_frames == 0
    audio = speech_like_audio(8_000, seed=1)
    reference = one_shot_mel(feature_extractor, audio)
    streamed = collect_streaming_frames(extractor, audio, chunk_sizes=[3_000])
    assert streamed.shape == reference.shape
