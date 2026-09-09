"""Emission wiring: the restructured translation loop emits caption events.

Proves the pipeline's translation side emits translation_final and
translation_provisional events through the caption EventTap with the
display-correctness rules preserved:

  - finals emit for every finalized translation (list or single);
  - provisionals emit ONLY when the simul layer is active (_simul_active)
    — the base path's buffer holds untranslated source and must never be
    flashed as an MT draft;
  - identical consecutive provisionals are deduped (display noise);
  - fresh=True only when the MT call counter advanced.

Run against a mock translation object — no live model.
"""
from __future__ import annotations

import asyncio

import pytest

from whisperlivekit.caption_events import EventLog, EventTap
from whisperlivekit.timed_objects import State, TimedText
from whisperlivekit.translation_processor import run_translation


def _tr(text, end=1.0):
    return TimedText(start=0.0, end=end, text=text)


class _Tr:
    """Mock translation backend simulating the simul-MT lifecycle."""

    def __init__(self):
        self._mt_call_count = 0
        self._simul_active = True
        self._committed_text = lambda: "committed source"
        self._source_text = lambda: "committed source + tail"
        self.next_result = None  # (final, buffer) per process() call

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
        from whisperlivekit.processing_queue import ProcessingQueue, SENTINEL
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


def test_provisional_requires_simul_active():
    # base path (no _simul_active): buffer with text must NOT emit a draft
    tr = _Tr()
    tr._simul_active = False
    tr.next_result = (None, _tr("raw untranslated source queue"))
    log, _ = _run([object()], tr)
    assert not [e for e in log.events if e.type == "translation_provisional"]

    # simul active: same buffer emits
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
        from whisperlivekit.processing_queue import ProcessingQueue, SENTINEL
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
    assert prov.fresh is True  # _mt_call_count advanced

    # fresh=False when the call counter does not advance (cached release)
    tr2 = _Tr()
    tr2.next_result = (None, _tr("Draft one"))
    tr2.process = lambda: (None, _tr("Draft two"))  # no counter bump
    log2, _ = _run([object()], tr2)
    prov2 = [e for e in log2.events if e.type == "translation_provisional"][0]
    assert prov2.fresh is False


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
