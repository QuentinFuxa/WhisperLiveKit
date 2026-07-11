"""Guards against silently-empty ASR output.

Two real incidents motivated these: the torch 2.13 MLX device mismatch
(#383) and CTranslate2 wheels shipping PTX newer than the GPU driver. In
both cases every chunk failed, the exceptions were logged as warnings,
and sessions looked healthy while producing empty captions forever.
"""

from __future__ import annotations

import logging
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from whisperlivekit.audio_processor import AudioProcessor
from whisperlivekit.warmup import warmup_asr


def _write_wav(path, seconds=0.5, sr=16000):
    samples = (np.sin(np.linspace(0, 440 * seconds * 2 * np.pi, int(sr * seconds)))
               * 0.1 * 32767).astype(np.int16)
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(samples.tobytes())


class _BrokenASR:
    def transcribe(self, audio):
        raise RuntimeError("cudaErrorInvalidDevice: invalid device ordinal")


class _WorkingASR:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1


def test_warmup_asr_raises_when_inference_is_broken(tmp_path):
    wav = tmp_path / "warmup.wav"
    _write_wav(wav)
    with pytest.raises(RuntimeError, match="refusing to serve"):
        warmup_asr(_BrokenASR(), warmup_file=str(wav))


def test_warmup_asr_passes_and_skips(tmp_path):
    wav = tmp_path / "warmup.wav"
    _write_wav(wav)
    asr = _WorkingASR()
    warmup_asr(asr, warmup_file=str(wav))
    assert asr.calls == 1
    # explicit skip stays a skip
    warmup_asr(_BrokenASR(), warmup_file="")


def test_alignatt_warmup_raises_when_inference_is_broken():
    align_att_base = pytest.importorskip(
        "whisperlivekit.simul_whisper.align_att_base"
    )

    class Broken(align_att_base.AlignAttBase):
        def __init__(self):  # skip the heavy base init
            pass

        def insert_audio(self, audio):
            raise RuntimeError("Tensor for argument weight is on cpu but expected on mps")

        def infer(self, is_last=False):
            pass

        def refresh_segment(self, complete=False):
            pass

    Broken.__abstractmethods__ = frozenset()
    with pytest.raises(RuntimeError, match="refusing to serve"):
        Broken().warmup(np.zeros(16000, dtype=np.float32))


def _watchdog_self():
    return SimpleNamespace(
        _any_asr_output=False,
        _silent_backend_warned=False,
        _SILENT_BACKEND_WARN_SECONDS=AudioProcessor._SILENT_BACKEND_WARN_SECONDS,
    )


def test_silent_backend_watchdog_fires_once(caplog):
    fake = _watchdog_self()
    with caplog.at_level(logging.ERROR, logger="whisperlivekit.audio_processor"):
        AudioProcessor._warn_if_backend_silent(fake, 5.0)      # too early
        AudioProcessor._warn_if_backend_silent(fake, 25.0)     # fires
        AudioProcessor._warn_if_backend_silent(fake, 60.0)     # already warned
    errors = [r for r in caplog.records if "produced no output" in r.message]
    assert len(errors) == 1
    assert fake._silent_backend_warned is True


def test_silent_backend_watchdog_respects_real_output(caplog):
    fake = _watchdog_self()
    fake._any_asr_output = True
    with caplog.at_level(logging.ERROR, logger="whisperlivekit.audio_processor"):
        AudioProcessor._warn_if_backend_silent(fake, 120.0)
    assert not [r for r in caplog.records if "produced no output" in r.message]
