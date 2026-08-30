"""Pipeline failure reporting and guards against silently-empty ASR output.

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


@pytest.mark.asyncio
@pytest.mark.parametrize('failure_phase', ['chunk', 'eof'])
async def test_asr_failure_is_reported_and_keeps_the_last_confirmed_text(monkeypatch, failure_phase):
    import asyncio
    from argparse import Namespace
    from dataclasses import asdict

    import whisperlivekit.audio_processor as processing
    from whisperlivekit.config import WhisperLiveKitConfig
    from whisperlivekit.core import TranscriptionEngine
    from whisperlivekit.timed_objects import ASRToken, Transcript

    class Online:
        SAMPLING_RATE = 16000
        asr = SimpleNamespace(sep='')
        calls = 0
        end = 0.0

        def insert_audio_chunk(self, audio, end):
            self.end = end

        def process_iter(self, **kwargs):
            self.calls += 1
            if self.calls > 1 and failure_phase == 'chunk':
                raise RuntimeError('decoder fixture failed')
            return [ASRToken(start=0, end=self.end, text='Confirmed.')], self.end

        def get_buffer(self):
            return Transcript(start=self.end, end=self.end, text='')

        def start_silence(self):
            return [], self.end

        def end_silence(self, duration, offset):
            self.end += duration

        def finish(self):
            raise RuntimeError('decoder fixture failed')

    monkeypatch.setattr(processing, 'online_factory', lambda *args, **kwargs: Online())
    engine = object.__new__(TranscriptionEngine)
    engine.args = Namespace(**asdict(WhisperLiveKitConfig(vac=False, pcm_input=True, min_chunk_size=0.5, asr_coalesce_min_s=0)))
    engine.asr = SimpleNamespace()
    engine.translation_model = None
    processor = AudioProcessor(transcription_engine=engine)
    records = []
    confirmed = asyncio.Event()

    async def collect(generator):
        async for message in generator:
            records.append(message)
            if any(line.text == 'Confirmed.' for line in message.lines):
                confirmed.set()

    try:
        consumer = asyncio.create_task(collect(await processor.create_tasks()))
        await processor.process_audio(b'\x01\x00' * 8000)
        await asyncio.wait_for(confirmed.wait(), 3)
        if failure_phase == 'chunk':
            await processor.process_audio(b'\x01\x00' * 8000)
        else:
            await processor.process_audio(b'')
        await asyncio.wait_for(consumer, 3)
        assert records[-1].status == 'error'
        assert 'decoder fixture failed' in records[-1].error
        assert any(any(line.text == 'Confirmed.' for line in r.lines) for r in records[:-1])
        assert processor.transcription_queue.closed
        with pytest.raises(RuntimeError, match='decoder fixture failed'):
            await processor.process_audio(b'\x01\x00' * 8000)
        await asyncio.wait_for(processor.transcription_task, 1)
    finally:
        await processor.cleanup()
        if 'consumer' in locals():
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)


@pytest.mark.asyncio
async def test_formatter_drains_results_that_finish_during_a_client_send():
    import asyncio
    from argparse import Namespace
    from dataclasses import asdict

    from whisperlivekit.config import WhisperLiveKitConfig
    from whisperlivekit.core import TranscriptionEngine
    from whisperlivekit.timed_objects import ASRToken

    engine = object.__new__(TranscriptionEngine)
    engine.args = Namespace(**asdict(WhisperLiveKitConfig(transcription=False, vac=False, pcm_input=True)))
    engine.asr = None
    engine.translation_model = None
    processor = AudioProcessor(transcription_engine=engine)
    processor.transcription_task = asyncio.get_running_loop().create_future()
    processor.state.new_tokens = [ASRToken(start=0, end=1, text='First.')]
    output = processor.results_formatter()
    try:
        first = await anext(output)
        assert ''.join(line.text for line in first.lines) == 'First.'
        # A WebSocket send can suspend the consumer while the final ASR call
        # finishes. That new output must be formatted before the generator ends.
        processor.state.new_tokens = [ASRToken(start=1, end=2, text=' Last words')]
        processor.is_stopping = True
        processor.transcription_task.set_result(None)
        final = await anext(output)
        assert 'Last words' in ''.join(line.text for line in final.lines)
        with pytest.raises(StopAsyncIteration):
            await anext(output)
    finally:
        await output.aclose()
        await processor.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["chunk", "eof"])
async def test_diarization_failure_reaches_the_client_and_stops_input(monkeypatch, failure_phase):
    import asyncio
    from argparse import Namespace
    from dataclasses import asdict

    from whisperlivekit.config import WhisperLiveKitConfig
    from whisperlivekit.core import TranscriptionEngine

    class Diarizer:
        buffer_audio = []

        def __init__(self):
            self.calls = 0
            self.called = asyncio.Event()
            self.closed = False

        def insert_audio_chunk(self, audio):
            pass

        def insert_silence(self, duration):
            pass

        async def diarize(self):
            self.calls += 1
            self.called.set()
            if failure_phase == "chunk" or self.calls > 1:
                raise RuntimeError("speaker decoder fixture failed")
            return []

        def close(self):
            self.closed = True

    diarizer = Diarizer()
    monkeypatch.setattr("whisperlivekit.audio_processor.online_diarization_factory", lambda *args: diarizer)
    engine = object.__new__(TranscriptionEngine)
    engine.args = Namespace(**asdict(WhisperLiveKitConfig(
        transcription=False, diarization=True, vac=False, pcm_input=True, min_chunk_size=.5,
    )))
    engine.asr = None
    engine.translation_model = None
    engine.diarization_model = SimpleNamespace()
    processor = AudioProcessor(transcription_engine=engine)
    records = []

    async def collect(generator):
        async for response in generator:
            records.append(response)

    try:
        consumer = asyncio.create_task(collect(await processor.create_tasks()))
        await processor.process_audio(b"\x01\x00" * 8000)
        await asyncio.wait_for(diarizer.called.wait(), 1)
        if failure_phase == "eof":
            await processor.process_audio(b"")
        await asyncio.wait_for(consumer, 1)
        assert records[-1].status == "error"
        assert "speaker decoder fixture failed" in records[-1].error
        assert processor.diarization_queue.closed
        with pytest.raises(RuntimeError, match="speaker decoder fixture failed"):
            await processor.process_audio(b"\x01\x00" * 8000)
        await asyncio.wait_for(processor.diarization_task, 1)
    finally:
        await processor.cleanup()
        if "consumer" in locals():
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
    assert diarizer.closed
