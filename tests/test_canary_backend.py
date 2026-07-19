from argparse import Namespace

import numpy as np


def test_parse_args_accepts_canary_options(monkeypatch):
    from whisperlivekit.parse_args import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "whisperlivekit-server",
            "--backend", "canary",
            "--canary-model", "nvidia/canary-1b-v2",
            "--canary-default-lang", "de",
            "--canary-lid-model", "langid_ambernet",
            "--canary-lid-min-sec", "3.0",
            "--canary-lid-min-conf", "0.6",
        ],
    )
    cfg = parse_args()
    assert cfg.backend == "canary"
    assert cfg.canary_model == "nvidia/canary-1b-v2"
    assert cfg.canary_default_lang == "de"
    assert cfg.canary_lid_model == "langid_ambernet"
    assert cfg.canary_lid_min_sec == 3.0
    assert cfg.canary_lid_min_conf == 0.6


def test_canary_config_defaults():
    from whisperlivekit.config import WhisperLiveKitConfig

    cfg = WhisperLiveKitConfig.from_kwargs(backend="canary")
    assert cfg.canary_model == "nvidia/canary-1b-v2"
    assert cfg.canary_default_lang == "en"
    assert cfg.canary_lid_model == "langid_ambernet"
    assert cfg.canary_lid_min_sec == 2.0
    assert cfg.canary_lid_min_conf == 0.5


def test_canary_words_to_tokens_maps_word_timestamps():
    from whisperlivekit.canary_backend import canary_words_to_tokens

    word_stamps = [
        {"word": "hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.4, "end": 0.9},
    ]
    tokens = canary_words_to_tokens(word_stamps)
    assert [t.text for t in tokens] == ["hello", "world"]
    assert tokens[0].start == 0.0 and tokens[0].end == 0.4
    assert tokens[1].start == 0.4 and tokens[1].end == 0.9


def test_canary_words_to_tokens_handles_missing_stamps():
    from whisperlivekit.canary_backend import canary_words_to_tokens

    assert canary_words_to_tokens(None) == []
    assert canary_words_to_tokens([]) == []


def test_canary_segment_end_ts():
    from whisperlivekit.canary_backend import canary_segment_end_ts

    seg_stamps = [
        {"segment": "hello world.", "start": 0.0, "end": 0.9},
        {"segment": "bye.", "start": 1.0, "end": 1.5},
    ]
    assert canary_segment_end_ts(seg_stamps) == [0.9, 1.5]
    assert canary_segment_end_ts(None) == []


def test_map_voxlingua_to_canary_supported():
    from whisperlivekit.canary_backend import map_voxlingua_to_canary

    assert map_voxlingua_to_canary("en") == "en"
    assert map_voxlingua_to_canary("de") == "de"
    assert map_voxlingua_to_canary("uk") == "uk"


def test_map_voxlingua_to_canary_unsupported_returns_none():
    from whisperlivekit.canary_backend import map_voxlingua_to_canary

    assert map_voxlingua_to_canary("zh") is None   # Chinese not in Canary's 25
    assert map_voxlingua_to_canary("") is None
    assert map_voxlingua_to_canary(None) is None


class _RecordingCanaryASR:
    """Minimal stand-in for CanaryASR that records the source language used."""
    sep = " "

    def __init__(self):
        self.original_language = None
        self.calls = []

    def transcribe(self, audio, init_prompt=""):
        self.calls.append(self.original_language)
        return f"decoded:{self.original_language}"


class _StubLID:
    def __init__(self, code="de", conf=0.9):
        self._code, self._conf = code, conf
        self.n_calls = 0

    def detect(self, audio):
        self.n_calls += 1
        return self._code, self._conf


def _audio(seconds):
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def test_explicit_language_bypasses_lid():
    from whisperlivekit.canary_backend import CanarySessionASR

    asr = _RecordingCanaryASR()
    lid = _StubLID()
    session = CanarySessionASR(asr, "fr", lid=lid, default_lang="en",
                               lid_min_sec=2.0, lid_min_conf=0.5)
    session.transcribe(_audio(5))
    assert asr.calls == ["fr"]
    assert lid.n_calls == 0


def test_auto_uses_default_until_enough_audio():
    from whisperlivekit.canary_backend import CanarySessionASR

    asr = _RecordingCanaryASR()
    lid = _StubLID(code="de", conf=0.9)
    session = CanarySessionASR(asr, "auto", lid=lid, default_lang="en",
                               lid_min_sec=2.0, lid_min_conf=0.5)
    session.transcribe(_audio(1.0))          # below lid_min_sec -> default, no LID
    assert asr.calls == ["en"]
    assert lid.n_calls == 0


def test_auto_detects_once_then_locks():
    from whisperlivekit.canary_backend import CanarySessionASR

    asr = _RecordingCanaryASR()
    lid = _StubLID(code="de", conf=0.9)
    session = CanarySessionASR(asr, "auto", lid=lid, default_lang="en",
                               lid_min_sec=2.0, lid_min_conf=0.5)
    session.transcribe(_audio(3.0))          # detects -> "de"
    session.transcribe(_audio(4.0))          # locked -> "de", no second detect
    assert asr.calls == ["de", "de"]
    assert lid.n_calls == 1


def test_auto_low_confidence_stays_on_default():
    from whisperlivekit.canary_backend import CanarySessionASR

    asr = _RecordingCanaryASR()
    lid = _StubLID(code="de", conf=0.2)       # below lid_min_conf
    session = CanarySessionASR(asr, "auto", lid=lid, default_lang="en",
                               lid_min_sec=2.0, lid_min_conf=0.5)
    session.transcribe(_audio(3.0))
    assert asr.calls == ["en"]                # not locked, retried later


def test_auto_lid_exception_stays_on_default_and_retries():
    from whisperlivekit.canary_backend import CanarySessionASR

    class _RaisingLID:
        def __init__(self):
            self.n_calls = 0
        def detect(self, audio):
            self.n_calls += 1
            raise RuntimeError("boom")

    asr = _RecordingCanaryASR()
    lid = _RaisingLID()
    session = CanarySessionASR(asr, "auto", lid=lid, default_lang="en",
                               lid_min_sec=2.0, lid_min_conf=0.5)
    session.transcribe(_audio(3.0))          # detect raises -> default, not locked
    session.transcribe(_audio(3.0))          # retries (still not locked)
    assert asr.calls == ["en", "en"]
    assert lid.n_calls == 2                   # retried, never locked


def test_auto_with_no_lid_uses_default():
    from whisperlivekit.canary_backend import CanarySessionASR

    asr = _RecordingCanaryASR()
    session = CanarySessionASR(asr, "auto", lid=None, default_lang="en",
                               lid_min_sec=2.0, lid_min_conf=0.5)
    session.transcribe(_audio(5.0))
    assert asr.calls == ["en"]


import importlib.util

import pytest

_NEMO_AVAILABLE = importlib.util.find_spec("nemo") is not None
requires_nemo = pytest.mark.skipif(not _NEMO_AVAILABLE, reason="NeMo not installed")


@requires_nemo
def test_canary_asr_transcribes_with_word_timestamps():
    import soundfile as sf

    from whisperlivekit.canary_backend import CanaryASR
    from whisperlivekit.test_data import get_sample  # real 16kHz mono clip + reference text

    asr = CanaryASR(
        lan="en",
        canary_model="nvidia/canary-1b-v2",
        buffer_trimming="segment",
        buffer_trimming_sec=15.0,
        confidence_validation=False,
    )
    sample = get_sample("librispeech_short")
    audio, _sr = sf.read(sample.path, dtype="float32")
    res = asr.transcribe(audio, source_lang="en")
    tokens = asr.ts_words(res)
    assert tokens, "expected at least one word token"
    assert all(t.end >= t.start for t in tokens)
    ends = asr.segments_end_ts(res)
    assert ends and ends[-1] > 0


class _FakeCanaryASR:
    """Stands in for a loaded CanaryASR so online_factory needs no NeMo."""
    sep = " "

    def __init__(self):
        self.original_language = None
        self.lid_model = _StubLID()
        self.canary_default_lang = "en"
        self.confidence_validation = False
        self.tokenizer = None
        self.buffer_trimming = "segment"
        self.buffer_trimming_sec = 15.0

    def transcribe(self, audio, init_prompt=""):
        return "x"


def test_online_factory_routes_canary_to_localagreement():
    from whisperlivekit.canary_backend import CanarySessionASR
    from whisperlivekit.core import online_factory
    from whisperlivekit.local_agreement.online_asr import OnlineASRProcessor

    args = Namespace(
        backend="canary", backend_policy="simulstreaming", lan="auto",
        canary_default_lang="en", canary_lid_min_sec=2.0, canary_lid_min_conf=0.5,
    )
    asr = _FakeCanaryASR()
    processor = online_factory(args, asr, language="auto")
    assert isinstance(processor, OnlineASRProcessor)
    assert isinstance(processor.asr, CanarySessionASR)
    assert processor.asr._lid is asr.lid_model

    # language=None falls back to args.lan ("auto") -> auto-detect mode.
    processor_none = online_factory(args, _FakeCanaryASR(), language=None)
    assert isinstance(processor_none.asr, CanarySessionASR)
    assert processor_none.asr._is_auto is True


@requires_nemo
def test_canary_end_to_end_via_testharness():
    """Full pipeline smoke test: FFmpeg decode -> VAD -> Canary ASR -> LocalAgreement.

    Uses the real TestHarness (whisperlivekit/test_harness.py), which forwards
    **kwargs straight to TranscriptionEngine/WhisperLiveKitConfig, so `backend`
    and `lan` are the same fields exercised by test_canary_config_defaults()
    above. Audio comes from whisperlivekit/test_data.py's cached LibriSpeech
    sample (same fixture used by test_canary_asr_transcribes_with_word_timestamps
    and tests/test_pipeline.py).
    """
    import asyncio

    from whisperlivekit import TestHarness
    from whisperlivekit.test_data import get_sample

    sample = get_sample("librispeech_short")

    async def _run():
        async with TestHarness(backend="canary", lan="en") as h:
            await h.feed(sample.path, speed=1.0)
            await h.drain(3.0)
            result = await h.finish()
            assert result.text.strip(), "expected non-empty transcription"

    asyncio.run(_run())
