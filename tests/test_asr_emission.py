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
  - the terminal flush of the remaining tail when the stream ends;
  - a provisional that repeats the pre-final tail: the final reset the
    dedupe, so the tail after it is a new event;
  - an inert tap (no sink): the loop builds no event payloads at all;
  - a hand-built processor whose ``__init__`` never ran: the class-level
    default tap is inert, so the loop transcribes and emits nothing.

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


_NO_TAP = object()


def _processor_for(backend, tap=_NO_TAP):
    """Build a processor by hand, bypassing ``__init__``.

    ``tap`` defaults to not being set at all, which is how a double stays
    tap-less; every attribute below is what the transcription loop itself and
    the emission sites read.
    """
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
    if tap is not _NO_TAP:
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


class _RepeatingTailASR(_ScriptedStreamingASR):
    """Commits one segment per chunk while leaving the same text in the tail.

    The tail a chunk publishes is identical to the pre-final provisional, so
    the provisional after the commit only survives if the final reset the
    dedupe state.
    """

    def __init__(self):
        super().__init__()
        self.steps = [
            ([ASRToken(0.0, 0.25, "今天我們")], "討論"),
            ([ASRToken(0.25, 0.5, "皮膚科")], "討論"),
        ]

    def process_iter(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        tokens, self.tail = self.steps.pop(0)
        return tokens, self.end

    def finish(self):
        return [ASRToken(0.5, 0.75, "完成。")], 0.75


class _InertCountingTap:
    """A tap with no sink: ``active`` is False, and every call is recorded so
    the test can see whether the loop skipped it."""

    active = False

    def __init__(self):
        self.calls = []

    def transcription_provisional(self, audio_t, text):
        self.calls.append(("transcription_provisional", text))

    def transcription_final(self, audio_t, text):
        self.calls.append(("transcription_final", text))


def test_provisional_after_a_final_is_not_suppressed():
    """A committed segment consumes the tail the previous provisional
    described, so the next provisional emits even when its text repeats.

    Fails if `_last_asr_prov` survives a final: the second provisional (same
    text as the pre-final one) would be swallowed as a duplicate.
    """

    async def driver():
        recorder = _Recorder()
        backend = _RepeatingTailASR()
        processor = _processor_for(backend, EventTap(sink=recorder))
        task = asyncio.create_task(processor.transcription_processor())

        await processor.transcription_queue.put(np.zeros(4_000, dtype=np.float32))
        observed = await _wait_for_events(recorder, 2)
        assert [(e.type, e.text) for e in observed] == [
            ("transcription_final", "今天我們"),
            ("transcription_provisional", "討論"),
        ]

        # second chunk: same tail text, another committed segment
        await processor.transcription_queue.put(np.zeros(4_000, dtype=np.float32))
        observed = await _wait_for_events(recorder, 4)
        assert len(observed) == 4, f"repeated tail after a final emitted {observed}"

        await processor.transcription_queue.put(SENTINEL)
        await asyncio.wait_for(task, timeout=5.0)
        return recorder

    recorder = asyncio.run(driver())
    assert [(e.type, e.text) for e in recorder.events][:4] == [
        ("transcription_final", "今天我們"),
        ("transcription_provisional", "討論"),
        ("transcription_final", "皮膚科"),
        ("transcription_provisional", "討論"),
    ]


def test_inert_tap_is_never_called():
    """With no sink the loop must not build event payloads (each call site
    joins token text before the tap can discard it).

    Fails if a call site emits unconditionally: the counting tap records the
    call even though nothing consumes the stream.
    """

    async def driver():
        tap = _InertCountingTap()
        backend = _ScriptedStreamingASR()
        processor = _processor_for(backend, tap)
        task = asyncio.create_task(processor.transcription_processor())

        await processor.transcription_queue.put(np.zeros(4_000, dtype=np.float32))
        # The tap cannot tell the test that the chunk path ran (it is inert),
        # so wait on the committed state instead.
        deadline = time() + 5.0
        while not processor.state.tokens and time() < deadline:
            await asyncio.sleep(0.01)
        # new tail text for the periodic refresh, which runs on the loop's
        # 0.5 s queue timeout
        backend.tail = "討論皮膚科"
        await asyncio.sleep(0.6)

        await processor.transcription_queue.put(SENTINEL)
        await asyncio.wait_for(task, timeout=5.0)
        return tap

    tap = asyncio.run(driver())
    assert tap.calls == []


def test_hand_built_processor_without_a_tap_transcribes_and_emits_nothing():
    """A processor built with ``object.__new__`` never runs ``__init__``, so it
    carries no session tap and never sets one.

    The class-level default must absorb that: the loop still commits tokens and
    reaches the emission sites, and the shared inert default reports ``active``
    False so nothing is emitted. Fails with ``AttributeError: 'AudioProcessor'
    object has no attribute 'event_tap'`` — recorded as ``processing_error`` —
    if the class-level default is removed.
    """

    async def driver():
        backend = _ScriptedStreamingASR()
        processor = _processor_for(backend)  # no tap argument: event_tap unset
        task = asyncio.create_task(processor.transcription_processor())

        await processor.transcription_queue.put(np.zeros(4_000, dtype=np.float32))
        deadline = time() + 5.0
        while not processor.state.tokens and time() < deadline:
            await asyncio.sleep(0.01)

        await processor.transcription_queue.put(SENTINEL)
        await asyncio.wait_for(task, timeout=5.0)
        return processor

    processor = asyncio.run(driver())

    assert processor.processing_error is None
    # The loop ran to the emission sites: both the committed segment and the
    # terminal flush landed.
    assert [t.text for t in processor.state.tokens] == ["今天我們", "皮膚科。"]
    # The default is the one shared inert instance; a sink-less tap cannot have
    # emitted anything.
    assert processor.event_tap is AudioProcessor.event_tap
    assert processor.event_tap.active is False
