"""Emission wiring: the restructured translation loop emits caption events.

Proves the pipeline's translation side emits translation_final and
translation_provisional events through the caption EventTap with the
display-correctness rules preserved:

  - finals emit for every finalized translation (list or single);
  - provisionals emit ONLY when the backend declares provides_drafts —
    the base path's buffer holds untranslated source and must never be
    flashed as an MT draft;
  - identical consecutive provisionals are deduped (display noise);
  - fresh=True only when the MT call counter advanced;
  - the provisional payload (committed/source/source_end) comes from the
    backend's progress() contract, never from private-attribute probing;
  - a progress() that raises disables draft display for the session with
    one visible warning (never a silent empty string).

Run against mock translation objects — no live model.
"""
from __future__ import annotations

import asyncio

from whisperlivekit.caption_events import EventLog, EventTap
from whisperlivekit.timed_objects import State, TimedText, TranslationProgress
from whisperlivekit.translation_processor import run_translation


def _tr(text, end=1.0):
    return TimedText(start=0.0, end=end, text=text)


class _Tr:
    """Mock translation backend implementing the progress display contract
    (stand-in for MlxLlmTranslationSimul)."""

    provides_drafts = True

    def __init__(self):
        self._mt_call_count = 0
        self._committed = "committed source"
        self._source = "committed source + tail"
        self._source_end = None
        self.next_result = None  # (final, buffer) per process() call

    def progress(self):
        return TranslationProgress(
            mt_call_count=self._mt_call_count,
            committed_text=self._committed,
            source_text=self._source,
            source_end=self._source_end,
        )

    def insert_tokens(self, tokens):
        pass

    def _take(self):
        # real backends consume their result: a second call before new tokens
        # arrives yields nothing (the buffer was reset/validated away)
        result, self.next_result = self.next_result, None
        return result

    def process(self):
        self._mt_call_count += 1
        return self._take()

    def validate_buffer_and_reset(self):
        return self._take()


def _run(items, translation):
    """Drive run_translation over `items` with a real EventLog sink."""
    log = EventLog()
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
        _mt_call_count = 0
        def progress(self):
            return TranslationProgress()
        def insert_tokens(self, tokens):
            pass
        def process(self):
            return (None, _tr("raw untranslated source queue"))

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
    log = EventLog()
    tap = EventTap(sink=log)
    state = State()
    lock = asyncio.Lock()

    async def driver():
        from whisperlivekit.processing_queue import SENTINEL, ProcessingQueue
        q = ProcessingQueue("test")
        for it in items:
            await q.put(it)
        await q.put(SENTINEL)

        def two_results():
            tr._mt_call_count += 1
            return (None, draft)

        tr.process = two_results
        await run_translation(q, tr, state, lock, tap)

    asyncio.run(driver())
    provs = [e for e in log.events if e.type == "translation_provisional"]
    assert len(provs) == 1, f"identical draft emitted {len(provs)}x"


def test_provisional_carries_committed_source_fresh():
    tr = _Tr()
    tr.next_result = (None, _tr("Draft text"))
    log, _ = _run([object()], tr)
    prov = [e for e in log.events if e.type == "translation_provisional"][0]
    assert prov.committed == "committed source"
    assert prov.source == "committed source + tail"
    assert prov.fresh is True  # the MT call counter advanced

    # fresh=False when the call counter does not advance (cached release)
    tr2 = _Tr()
    tr2.next_result = (None, _tr("Draft one"))
    tr2.process = lambda: (None, _tr("Draft two"))  # no counter bump
    log2, _ = _run([object()], tr2)
    prov2 = [e for e in log2.events if e.type == "translation_provisional"][0]
    assert prov2.fresh is False


def test_provisional_timestamp_prefers_progress_source_end():
    # the provisional's audio clock comes from progress().source_end when
    # the backend reports it (real source-tail position), falling back to
    # the ASR-side end_buffer otherwise
    tr = _Tr()
    tr._source_end = 5.5
    tr.next_result = (None, _tr("Draft text"))
    log, _ = _run([object()], tr)
    prov = [e for e in log.events if e.type == "translation_provisional"][0]
    assert prov.audio_t == 5.5

    tr2 = _Tr()  # source_end None → end_buffer fallback (0.0 default state)
    tr2.next_result = (None, _tr("Draft text"))
    log2, _ = _run([object()], tr2)
    prov2 = [e for e in log2.events if e.type == "translation_provisional"][0]
    assert prov2.audio_t == 0.0


def test_base_backend_contract_defaults_are_honest():
    # the REAL base class (no model load): capability off, progress() empty,
    # and the emission loop never flashes its buffer as a draft even when
    # the base path returns one
    from whisperlivekit.translation_mlx_llm_mt import MlxLlmTranslation
    base = MlxLlmTranslation(model_id="hy-mt2-1.8b-8bit", warmup=False)
    assert base.provides_drafts is False
    d = base.progress()
    assert d.committed_text == "" and d.source_text == ""
    assert d.source_end is None and d.mt_call_count == 0

    # drive the real base through the loop (empty token batch — no model
    # load): no draft emission, FrontData contract unchanged
    log, state = _run([[]], base)
    assert not [e for e in log.events if e.type == "translation_provisional"]
    assert state.new_translation == []  # FrontData contract unchanged


def test_progress_raise_disables_drafts_with_one_warning(caplog):
    # a progress() that raises is a backend bug: one warning, draft display
    # disabled for the session, finals and the loop itself keep working
    class _Broken(_Tr):
        def progress(self):
            raise RuntimeError("backend state exploded")

        def process(self):
            return (None, _tr("Draft text"))

        def validate_buffer_and_reset(self):
            return (None, None)

    tr = _Broken()
    import logging
    with caplog.at_level(logging.WARNING):
        log, state = _run([object(), object()], tr)  # two items — loop continues
    assert not [e for e in log.events if e.type == "translation_provisional"]
    warnings = [r for r in caplog.records if "progress()" in r.message]
    assert len(warnings) == 1, "exactly one warning per session"
    assert "_Broken" in warnings[0].message  # names the backend class


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
