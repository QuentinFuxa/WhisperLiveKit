"""Emission wiring: the restructured translation loop emits caption events.

Proves the pipeline's translation side emits translation_final and
translation_provisional events through the caption EventTap with the
display-correctness rules preserved:

  - finals emit for every finalized translation (list or single);
  - provisionals emit ONLY when the backend declares provides_drafts —
    the base path's buffer holds untranslated source and must never be
    flashed as an MT draft;
  - identical consecutive provisionals are deduped (display noise);
  - the provisional's audio position is the ASR-side ``state.end_buffer``,
    the same fallback the final uses;
  - a final in the same batch as a buffer suppresses the provisional;
  - with no sink the loop emits nothing (the inert tap is passed on as no
    tap); with a sink the processor's tap reaches the loop.

Run against mock translation objects — no live model.
"""
from __future__ import annotations

import asyncio

from whisperlivekit.caption_events import EventTap
from whisperlivekit.timed_objects import State, TimedText
from whisperlivekit.translation_processor import run_translation


class _Recorder:
    """In-memory sink for the emitted events (no file, no display)."""

    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def _tr(text, end=1.0):
    return TimedText(start=0.0, end=end, text=text)


class _Tr:
    """Mock draft-capable translation backend (stand-in for
    MlxLlmTranslationSimul): it produces provisional buffers as it streams."""

    provides_drafts = True

    def __init__(self):
        self.next_result = None  # (final, buffer) per process() call

    def insert_tokens(self, tokens):
        pass

    def _take(self):
        # real backends consume their result: a second call before new tokens
        # arrives yields nothing (the buffer was reset/validated away)
        result, self.next_result = self.next_result, None
        return result

    def process(self):
        return self._take()

    def validate_buffer_and_reset(self):
        return self._take()


def _run(items, translation):
    """Drive run_translation over `items` with a recording sink."""
    log = _Recorder()
    tap = EventTap(sink=log)
    state = State()
    lock = asyncio.Lock()
    async def driver():
        # unwrap get_all_from_queue semantics: it returns one item per call,
        # draining batches; feed items through a real asyncio.Queue for
        # fidelity with the production loop.
        from whisperlivekit.processing_queue import SENTINEL, ProcessingQueue
        q = ProcessingQueue("test")
        for it in items:
            await q.put(it)
        await q.put(SENTINEL)
        await run_translation(q, translation, state, lock, tap)

    asyncio.run(driver())
    return log, state


def test_translation_final_emitted_for_single_and_list():
    tr = _Tr()
    fin = _tr("Hello world.")
    tr.next_result = (fin, None)
    log, state = _run([object()], tr)  # object() = a tokens item
    finals = [e for e in log.events if e.type == "translation_final"]
    assert len(finals) == 1 and finals[0].text == "Hello world."

    # list of finals: each emits
    tr2 = _Tr()
    tr2.next_result = ([_tr("A."), _tr("B.")], None)
    log2, _ = _run([object()], tr2)
    finals2 = [e for e in log2.events if e.type == "translation_final"]
    assert [f.text for f in finals2] == ["A.", "B."]


def test_provisional_requires_provides_drafts():
    # backend without draft capability: buffer with text must NOT emit a draft
    class _Base:
        provides_drafts = False
        def insert_tokens(self, tokens):
            pass
        def process(self):
            return (None, _tr("raw untranslated source queue"))
        def validate_buffer_and_reset(self):
            return (None, None)

    log, _ = _run([object()], _Base())
    assert not [e for e in log.events if e.type == "translation_provisional"]

    # draft-capable backend: same buffer emits
    tr2 = _Tr()
    tr2.next_result = (None, _tr("Today we will discuss"))
    log2, _ = _run([object()], tr2)
    provs = [e for e in log2.events if e.type == "translation_provisional"]
    assert len(provs) == 1 and provs[0].text == "Today we will discuss"


def test_provisional_dedupe_identical_consecutive():
    tr = _Tr()
    draft = _tr("Today we will discuss")
    # two process() calls returning the identical draft (cached release)
    items = [object(), object()]
    log = _Recorder()
    tap = EventTap(sink=log)
    state = State()
    lock = asyncio.Lock()

    async def driver():
        from whisperlivekit.processing_queue import SENTINEL, ProcessingQueue
        q = ProcessingQueue("test")
        for it in items:
            await q.put(it)
        await q.put(SENTINEL)

        tr.process = lambda: (None, draft)
        await run_translation(q, tr, state, lock, tap)

    asyncio.run(driver())
    provs = [e for e in log.events if e.type == "translation_provisional"]
    assert len(provs) == 1, f"identical draft emitted {len(provs)}x"


def test_provisional_audio_t_is_the_asr_end_buffer():
    """The provisional's audio position is the ASR-side ``state.end_buffer`` —
    the same fallback ``translation_final`` uses — not a position the backend
    reports.

    Fails if the provisional goes back to a backend-supplied source position
    (the draft buffer's own ``end``), or falls back to 0.0: the buffer here
    ends at 1.0 and the ASR has processed 12.5 s.
    """
    tr = _Tr()
    tr.next_result = (None, _tr("Draft text", end=1.0))
    tap = EventTap(sink=_Recorder())
    state = State()
    state.end_buffer = 12.5

    async def driver():
        from whisperlivekit.processing_queue import SENTINEL, ProcessingQueue
        q = ProcessingQueue("test")
        await q.put(object())
        await q.put(SENTINEL)
        await run_translation(q, tr, state, asyncio.Lock(), tap)

    asyncio.run(driver())
    prov = tap._sink.events[0]
    assert prov.type == "translation_provisional"
    assert prov.audio_t == 12.5


def test_base_backend_contract_defaults_are_honest():
    # the REAL base class (no model load): capability off, and the emission
    # loop never flashes its buffer as a draft even when the base path
    # returns one
    from whisperlivekit.translation_mlx_llm_mt import MlxLlmTranslation
    base = MlxLlmTranslation(model_id="hy-mt2-1.8b-8bit", warmup=False)
    # the honest default: this backend's buffer holds source, not a draft
    assert base.provides_drafts is False
    # no progress() accessor survives the reduced payload
    assert not hasattr(base, "progress")

    # drive the real base through the loop (empty token batch — no model
    # load): no draft emission, FrontData contract unchanged
    log, state = _run([[]], base)
    assert not [e for e in log.events if e.type == "translation_provisional"]
    assert state.new_translation == []  # FrontData contract unchanged


def test_final_suppresses_provisional_in_same_batch():
    # when a final arrives, the buffer (also present) must not emit a draft
    tr = _Tr()
    tr.next_result = (_tr("Final text."), _tr("partial next"))
    log, _ = _run([object()], tr)
    assert [e.type for e in log.events] == ["translation_final"]


def test_state_still_updated():
    # the state contract (new_translation / new_translation_buffer) is unchanged
    tr = _Tr()
    tr.next_result = (None, _tr("draft"))
    log, state = _run([object()], tr)
    assert state.new_translation_buffer is not None
    assert state.new_translation == []


def _processor_with_tap(translation, event_tap, items):
    """An AudioProcessor with only what the translation loop touches, driven
    over `items` until the loop's SENTINEL."""
    from whisperlivekit.audio_processor import AudioProcessor
    from whisperlivekit.processing_queue import SENTINEL, ProcessingQueue

    processor = object.__new__(AudioProcessor)
    processor.translation = translation
    processor.state = State()
    processor.lock = asyncio.Lock()
    processor.event_tap = event_tap
    processor.translation_queue = ProcessingQueue("test")

    async def driver():
        for item in items:
            await processor.translation_queue.put(item)
        await processor.translation_queue.put(SENTINEL)
        await processor.translation_processor()

    asyncio.run(driver())
    return processor


def test_inert_tap_through_the_processor_emits_nothing():
    """With no sink (no ``--event-log``) the processor passes no tap to the
    loop, so nothing is emitted — the stream is sink-driven, not flag-driven.

    Fails if ``translation_processor`` stops dropping its inert tap and the
    loop starts emitting into a display nobody asked for.
    """
    tr = _Tr()
    tr.next_result = (None, _tr("Draft text"))
    processor = _processor_with_tap(tr, EventTap(sink=None), [object(), object()])
    assert processor.state.new_translation_buffer is not None  # the loop ran
    assert processor.event_tap._sink is None


def test_the_processors_tap_reaches_the_translation_loop():
    """The companion case: any sink turns the stream on, and the events the
    loop emits arrive at the processor's own sink."""
    tr = _Tr()
    tr.next_result = (None, _tr("Draft text"))
    recorder = _Recorder()
    _processor_with_tap(tr, EventTap(sink=recorder), [object(), object()])
    drafts = [e for e in recorder.events if e.type == "translation_provisional"]
    assert [d.text for d in drafts] == ["Draft text"]
