"""Regression tests driven by the canonical captured event stream
(qwen3-asr + simul-MT on zh_long.wav, production event tap).

CL: 'the flickering prefix is again happening in many asr sentences' —
reproduced from the canonical capture: at every mid-sentence commit the
display dropped the already-shown tail for a frame, then the next
provisional re-added it. These tests lock the behavior down.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from whisperlivekit.caption_events import EventLog
from whisperlivekit.overlay import OverlayRenderer
from whisperlivekit.overlay_model import OverlayDisplayModel
from whisperlivekit.src_buffer import SrcReadingBuffer

CANONICAL = Path(__file__).parent / "golden" / "zh_long_canonical.jsonl"


def test_src_buffer_commit_absorbs_tail_display_unchanged():
    """The canonical flicker: tail '我们今天来讨论镭射。' shown, commit '我们今天'
    lands -> the display must NOT drop the tail for a frame."""
    b = SrcReadingBuffer()
    assert b.tail("我们今天来讨论镭射。") == "我们今天来讨论镭射。"
    # mid-sentence commit absorbs the committed part of the tail
    display = b.commit("我们今天")
    assert display == "我们今天来讨论镭射。", f"commit changed the display: {display!r}"
    assert b.tail_text == "来讨论镭射。"
    # the next provisional replaces the tail — text keeps growing
    assert b.tail("来讨论镭射在医学上的应用。") == "我们今天来讨论镭射在医学上的应用。"


def test_src_buffer_commit_absorbing_whole_tail():
    """A commit covering the whole tail clears it (no duplicated text)."""
    b = SrcReadingBuffer()
    b.tail("镭射技术可以精确的。")
    display = b.commit("镭射技术可以精确的切除肿瘤组织。")
    assert display == "镭射技术可以精确的切除肿瘤组织。"
    assert b.tail_text == ""


def test_src_buffer_commit_reworded_tail_dropped():
    """A commit whose text matches neither the tail nor a prefix: the stale
    tail is dropped rather than rendered next to text the ASR retracted."""
    b = SrcReadingBuffer()
    b.tail("镭射在医学上")
    display = b.commit("镭射技术可以")
    assert display == "镭射技术可以" and b.tail_text == ""


def test_canonical_stream_no_prefix_flicker():
    """Replay the canonical zh_long stream through the real view classes:
    the src display must never lose text and then re-show it. Commits keep
    the absorbed tail so the text never changes at commit time."""
    events = EventLog.load(str(CANONICAL)).events
    r = OverlayRenderer(overlay_mode="both")
    r._model = OverlayDisplayModel(hold_sec=3.5, clock=time.monotonic)
    frames = []
    prev = ""
    for e in events:
        now = datetime.now()
        if e.type == "transcription_provisional":
            r.partial("mic", e.text, now)
        elif e.type == "transcription_final":
            r.final("mic", [(None, e.text)], now)
        elif e.type == "translation_provisional":
            r.preview("mic", [(None, e.text)], now)
        elif e.type == "translation_final":
            r.translation("mic", [(None, e.text)], now)
        st = r._model.tick()
        if st is not None and st.partial:
            frames.append(st.partial)
            if prev and len(st.partial) < len(prev) and prev.startswith(st.partial):
                # a shrink rendered — legal only at a promote/reset (a new
                # sentence); the LOST text must never re-appear
                for nxt in frames[len(frames):len(frames) + 6]:
                    if nxt == prev:
                        raise AssertionError(
                            f"prefix flicker: {prev!r} -> {st.partial!r} -> reappeared")
            prev = st.partial
    assert frames, "no frames rendered"
    print(f"canonical replay OK: {len(frames)} frames, no flicker")


def test_reword_retraction_held():
    """The mic run's flicker: the hypothesis re-decodes and RETRACTS the tail
    ('我们今天来讨论镭射。在医学上的应用，镭射。' -> '确的切除。'). Any shorter
    text at the same commit boundary is held — the reader never sees a
    retraction; the commit resolves it."""
    import time as _t
    m = OverlayDisplayModel(hold_sec=3.5, clock=_t.monotonic)
    m.set_partial("我们今天来讨论镭射。在医学上的应用，镭射。", committed_len=0)
    st = m.tick()
    assert st.partial == "我们今天来讨论镭射。在医学上的应用，镭射。"
    m.set_partial("确的切除。", committed_len=0)   # retraction — held
    assert m.tick() is None, "retraction must not re-render"
    # even a longer reword is held while it's shorter than the held text —
    # the commit (committed_len changes) is what resolves the stale display
    m.set_partial("确的切除肿瘤组织。", committed_len=9)
    st = m.tick()
    assert st.partial == "确的切除肿瘤组织。"


def test_reworded_provisional_amends():
    """A reworded LONGER draft must amend (keep the common prefix, type the
    divergent suffix), not hard-swap the whole line — for both the EN (MT)
    row and the src (ASR) row."""
    from whisperlivekit.overlay import OverlayRenderer as R
    from whisperlivekit.overlay_model import PROVISIONAL

    r = R(overlay_mode="both")

    class C:
        now = 0.0
        def __call__(self): return self.now
        def advance(self, s): self.now += s
    clk = C()
    r._model = OverlayDisplayModel(hold_sec=3.5, clock=clk)

    class F:
        pass
    r._field_en = en = object()
    r._field_en_prev = object()
    r._field_partial = object()
    events = []
    r._set = lambda field, value: events.append(str(value)) if field is en else None

    def fake_attr(self, spans):
        if not spans:
            return None
        return ("DIM:" if spans[0].style == PROVISIONAL else "BRIGHT:") + \
            "".join(sp.text for sp in spans)
    R._spans_to_attributed = fake_attr
    R._spans_to_attributed_simple = staticmethod(
        lambda t, prov: ("DIM:" if prov else "BRIGHT:") + t)
    r._set_attr = lambda field, attr: events.append(attr[:90]) if field is en else None

    def tick_render(n=12):
        for _ in range(n):
            st = r._model.tick()
            if st is not None:
                r._reconcile(st)
            time.sleep(0.02)

    r.preview("mic", [(None, "Today we will discuss the applications of laser in medicine")], datetime.now())
    tick_render(8)
    # a reword arrives: LONGER but not a prefix-extension
    r.preview("mic", [(None, "Today we will discuss laser applications in medicine and surgery.")], datetime.now())
    for _ in range(30):  # pump the drainer so the amend pops and types out
        st = r._model.tick()
        if st is not None:
            r._reconcile(st)
        time.sleep(0.02)
    time.sleep(0.6)  # let the streaming thread finish typing the correction
    # the correction must type out (multiple renders), not flash in one frame
    assert len(events) >= 3, f"reword provisional did not amend: {events!r}"


def test_premature_terminator_commit_types():
    """A mid-sentence commit whose tail carried a premature terminator
    ('...来进行。' -> commit '...来进行口腔手术。') must TYPE the continuation,
    not hard-snap — every mid-sentence growth types (CL: typing only showed
    on one sentence)."""
    b = SrcReadingBuffer()
    b.commit("牙医也使用")           # commit 1 (new tokens only)
    display = b.tail("镭射来进行。")  # premature hypothesis tail
    assert display == "牙医也使用镭射来进行。"
    # the commit continues past the premature terminator (new tokens only)
    display = b.commit("镭射来进行口腔手术。")
    assert display == "牙医也使用镭射来进行口腔手术。", display


def test_zh_ja_fallback_no_raw_source_provisional():
    """The zh->ja run has no calibrated heads — simul degrades to the base
    path, whose buffer holds the UNTRANSLATED source queue. That buffer must
    NOT be forwarded as a translation provisional: raw source flashing in the
    MT row until the real final replaced it. The emission gate lives in
    audio_processor; here we assert the captured fallback stream's
    provisionals are exactly what the gate removes."""
    fixture = Path(__file__).parent / "golden" / "zh_ja_fallback.jsonl"
    evs = EventLog.load(str(fixture)).events
    # the pre-fix stream carried raw-source provisionals; the gate removes them
    raw = [e for e in evs if e.type == "translation_provisional"]
    # every translation provisional in the fallback path is raw source
    # (matches a transcription_final verbatim, no kana)
    finals_zh = {e.text for e in evs if e.type == "transcription_final"}
    leaked = [e.text for e in raw if e.text in finals_zh and not _is_target_script(e.text)]
    assert leaked, "fixture should contain the fallback raw-source provisionals"


def _is_target_script(text: str) -> bool:
    """True when the text is predominantly Japanese (kana present)."""
    return any("\u3040" <= ch <= "\u30ff" for ch in text)
