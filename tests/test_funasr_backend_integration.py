import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from whisperlivekit.funasr_backend import FunASRASR
from whisperlivekit.local_agreement.online_asr import OnlineASRProcessor
from whisperlivekit.session_asr_proxy import SessionASRProxy

RUN_INTEGRATION = os.environ.get("WLK_RUN_FUNASR_INTEGRATION") == "1"
MODEL_DIR = os.environ.get("WLK_FUNASR_MODEL_DIR")
SENSEVOICE_REPO = "FunAudioLLM/SenseVoiceSmall"
SENSEVOICE_REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"

EXPECTED = {
    "en": "tribal",
    "zh": "开饭时间",
    "yue": "表达",
    "ja": "中学",
    "ko": "생각",
}


def test_test_harness_audio_loader_falls_back_without_ffmpeg(tmp_path, monkeypatch):
    from whisperlivekit import test_harness

    sample_rate = 16000
    audio = np.sin(np.linspace(0, np.pi * 8, sample_rate, dtype=np.float32)) * 0.2
    sample = tmp_path / "sample.wav"
    sf.write(sample, audio, sample_rate)

    def missing_ffmpeg(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", missing_ffmpeg)

    pcm = test_harness.load_audio_pcm(str(sample), sample_rate=sample_rate)

    assert pcm
    assert len(pcm) == sample_rate * 2


def _load_16khz(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        scipy_signal = pytest.importorskip(
            "scipy.signal",
            reason="resampling non-16 kHz FunASR fixtures requires scipy",
        )
        divisor = int(np.gcd(sample_rate, 16000))
        audio = scipy_signal.resample_poly(
            audio, 16000 // divisor, sample_rate // divisor
        )
    return np.asarray(audio, dtype=np.float32)


@pytest.fixture(scope="module")
def sensevoice():
    if not RUN_INTEGRATION or not MODEL_DIR:
        pytest.skip("set WLK_RUN_FUNASR_INTEGRATION=1 and WLK_FUNASR_MODEL_DIR")
    return FunASRASR(lan="auto", model_dir=MODEL_DIR)


@pytest.mark.parametrize("language", ["en", "zh", "yue", "ja", "ko"])
def test_real_sensevoice_multilingual_timestamp_contract(sensevoice, language):
    sample = Path(MODEL_DIR) / "example" / f"{language}.mp3"
    audio = _load_16khz(sample)
    result = sensevoice.transcribe(audio)
    tokens = sensevoice.ts_words(result)
    text = "".join(token.text for token in tokens)

    assert tokens
    assert EXPECTED[language] in text
    assert text == sensevoice._postprocess(result[0]["text"])
    assert len(result[0]["words"]) == len(result[0]["timestamp"]) == len(tokens)
    assert all(np.isfinite([token.start, token.end]).all() for token in tokens)
    assert all(token.start >= 0 and token.end >= token.start for token in tokens)
    assert all(
        left.start <= right.start and left.end <= right.end
        for left, right in zip(tokens, tokens[1:])
    )
    assert tokens[-1].end <= len(audio) / 16000 + 0.75
    assert {token.detected_language for token in tokens} == {language}


def test_real_sensevoice_session_language_survives_proxy(sensevoice):
    audio = _load_16khz(Path(MODEL_DIR) / "example" / "ko.mp3")
    session = SessionASRProxy(sensevoice, "ko")

    result = session.transcribe(audio)
    tokens = session.ts_words(result)

    assert sensevoice.original_language is None
    assert EXPECTED["ko"] in "".join(token.text for token in tokens)
    assert {token.detected_language for token in tokens} == {"ko"}


def test_real_sensevoice_progressive_english_is_stable(sensevoice):
    audio = _load_16khz(Path(MODEL_DIR) / "example" / "en.mp3")
    hypotheses = []
    for seconds in (3, 5, len(audio) / 16000):
        result = sensevoice.transcribe(audio[: int(seconds * 16000)])
        hypotheses.append([token.text for token in sensevoice.ts_words(result)])

    shared_prefixes = []
    for earlier, later in zip(hypotheses, hypotheses[1:]):
        shared = 0
        for left, right in zip(earlier, later):
            if left != right:
                break
            shared += 1
        shared_prefixes.append(shared)
        assert shared >= min(5, max(1, len(earlier) - 1)), hypotheses

    assert shared_prefixes == sorted(shared_prefixes)


def test_real_sensevoice_streams_through_local_agreement(sensevoice):
    audio = _load_16khz(Path(MODEL_DIR) / "example" / "en.mp3")
    expected_result = sensevoice.transcribe(audio)
    expected_text = "".join(
        token.text for token in sensevoice.ts_words(expected_result)
    )

    sensevoice.tokenizer = None
    sensevoice.confidence_validation = False
    sensevoice.buffer_trimming = "segment"
    sensevoice.buffer_trimming_sec = 15
    online = OnlineASRProcessor(sensevoice)

    boundaries = (3 * 16000, 5 * 16000, len(audio))
    emitted = []
    previous = 0
    processed_upto = []
    for boundary in boundaries:
        online.insert_audio_chunk(audio[previous:boundary])
        tokens, end_time = online.process_iter()
        emitted.append(tokens)
        processed_upto.append(end_time)
        previous = boundary

    remaining, final_time = online.finish()
    output = [token for chunk in emitted for token in chunk] + remaining

    assert not emitted[0]
    assert len(emitted[1]) >= 5
    assert len(emitted[2]) >= 2
    assert "".join(token.text for token in output) == expected_text
    assert all(
        left.start <= right.start and left.end <= right.end
        for left, right in zip(output, output[1:])
    )
    assert processed_upto == pytest.approx([3, 5, len(audio) / 16000])
    assert final_time == pytest.approx(len(audio) / 16000)


@pytest.mark.asyncio
async def test_real_sensevoice_runs_through_test_harness():
    pytest.importorskip("funasr")
    from huggingface_hub import snapshot_download

    from whisperlivekit.test_harness import TestHarness

    model_dir = (
        Path(MODEL_DIR)
        if MODEL_DIR
        else Path(
            snapshot_download(
                repo_id=SENSEVOICE_REPO,
                revision=SENSEVOICE_REVISION,
            )
        )
    )
    sample = model_dir / "example" / "en.mp3"
    async with TestHarness(
        backend="funasr",
        backend_policy="localagreement",
        model_dir=str(model_dir),
        lan="en",
        vac=False,
        warmup_file=None,
    ) as harness:
        await harness.feed(str(sample), speed=0, chunk_duration=1.0)
        result = await harness.finish(timeout=90)

    assert not result.error
    assert "tribal" in result.committed_text
    assert result.buffer_transcription == ""
    assert result.text == result.committed_text
    assert not result.timing_errors()
