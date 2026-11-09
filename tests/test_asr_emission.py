"""Emission wiring: the ASR side of the pipeline emits caption events.

``tests/test_caption_events.py`` proves the ``EventTap`` forwards what it is
handed; it cannot go red when a pipeline call site stops emitting, because the
test calls the tap itself. This drives the real
``AudioProcessor.transcription_processor`` loop with a tap wired to a
recording sink and asserts the transcription events the four documented call
sites in ``whisperlivekit/audio_processor.py`` produce:

  - the committed segment (``transcription_final``) and the rolling unstable
    tail (``transcription_provisional``) of a processed audio chunk;
  - the periodic refresh of a streaming decoder's buffer on the loop's queue
    timeout (text that arrived between chunks);
  - the terminal flush of the remaining tail when the stream ends.

Run against a scripted streaming backend — no live model, no network.
"""
from __future__ import annotations

import asyncio
from time import time
from types import SimpleNamespace

import numpy as np

from whisperlivekit.audio_processor import SENTINEL, AudioProcessor
from whisperlivekit.caption_events import EventTap
from whisperlivekit.metrics_collector import SessionMetrics
from whisperlivekit.timed_objects import ASRToken, State, Transcript


class _Recorder:
    """In-memory sink for the emitted events (no file, no display)."""

    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


class _ScriptedStreamingASR:
    """Streaming backend: one committed segment plus a revising tail.

    Like LocalAgreement, the buffer holds only the unvalidated tail:
    ``process_iter()`` commits "今天我們" and leaves "討論" behind;
    ``decode_between_chunks()`` models a decoder that keeps producing text
    while no audio is arriving — the case the loop's periodic refresh exists
    for.
    """

    SAMPLING_RATE = 16_000

    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.end = 0.0
        self.tail = ""

    def insert_audio_chunk(self, audio, audio_stream_end_time):
        self.audio_buffer = np.append(self.audio_buffer, audio)
        self.end = audio_stream_end_time

    def process_iter(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.tail = "討論"
        return [ASRToken(0.0, 0.25, "今天我們")], self.end

    def get_buffer(self):
        return Transcript(0.25, self.end, self.tail)

    def decode_between_chunks(self, text, audio_end):
        self.tail += text
        self.end = audio_end

    def finish(self):
        return [ASRToken(0.55, 0.75, "皮膚科。")], 0.75


def _processor_for(backend, tap):
    processor = object.__new__(AudioProcessor)
    processor.args = SimpleNamespace(asr_coalesce_min_s=0.0)
    processor.pause_segmentation_seconds = 2.0
    processor.transcription = backend
    processor.transcription_queue = asyncio.Queue()
    processor.translation_queue = None
    processor.translation = None
    processor.translate_on_complete = False
    processor._pending_translation_tokens = []
    processor.diarization_queue = None
    processor.metrics = SessionMetrics()
    processor.state = State()
    processor.lock = asyncio.Lock()
    processor.sample_rate = 16_000
    processor.beg_loop = time()
    processor.sep = " "
    processor.is_stopping = False
    processor._any_asr_output = False
    processor._silent_backend_warned = False
    processor.event_tap = tap
    processor._last_asr_prov = ""
    processor.processing_error = None
    processor.tokens_alignment = SimpleNamespace(_retention_seconds=300.0)
    return processor


async def _wait_for_events(recorder, count, timeout=5.0):
    """Poll until the recorder holds `count` events.

    The periodic refresh happens on the loop's own 0.5 s queue timeout, so the
    test cannot await it directly.
    """
    deadline = time() + timeout
    while len(recorder.events) < count and time() < deadline:
        await asyncio.sleep(0.01)
    return list(recorder.events)


def test_one_utterance_emits_the_transcription_sequence():
    """One utterance through the pipeline: the transcription events a
    consumer of the tap sees, with their text and audio position.

    Fails if a call site in `transcription_processor` /
    `_finish_transcription` stops emitting, if the emission order changes, or
    if an event stops carrying the text or the audio position it was produced
    against.
    """

    async def driver():
        recorder = _Recorder()
        backend = _ScriptedStreamingASR()
        processor = _processor_for(backend, EventTap(sink=recorder))
        task = asyncio.create_task(processor.transcription_processor())

        # 0.25 s of audio: the chunk path commits a segment and publishes the
        # unstable tail it stripped from the buffer.
        await processor.transcription_queue.put(np.zeros(4_000, dtype=np.float32))
        observed = await _wait_for_events(recorder, 2)
        assert len(observed) == 2, f"chunk path emitted {observed}"

        # The decoder keeps producing text with no audio arriving; the loop
        # republishes the buffer on its queue timeout.
        backend.decode_between_chunks("皮膚科", audio_end=0.4)
        observed = await _wait_for_events(recorder, 3)
        assert len(observed) == 3, f"periodic refresh emitted {observed}"

        # End of stream flushes the remaining tail.
        await processor.transcription_queue.put(SENTINEL)
        await asyncio.wait_for(task, timeout=5.0)
        return recorder

    recorder = asyncio.run(driver())

    assert [(e.type, e.text, e.audio_t) for e in recorder.events] == [
        ("transcription_final", "今天我們", 0.25),
        ("transcription_provisional", "討論", 0.25),
        ("transcription_provisional", "討論皮膚科", 0.4),
        ("transcription_final", "皮膚科。", 0.75),
    ]
