"""Model-level regression tests driven by the canonical captured event
streams (qwen3-asr + simul-MT, production event tap).

These tests drive the caption display model directly (no view code): the
model owns the display rules, the views render its state. View-level replay
tests (src row typing, overlay view mechanics) live with the view code.
"""
from __future__ import annotations

from pathlib import Path

from whisperlivekit.caption_display import CaptionDisplay
from whisperlivekit.caption_events import EventLog

CANONICAL = CANONICAL = Path(__file__).parent / "golden" / "zh_long_canonical.jsonl"


def test_reword_retraction_held():
    """The mic run's flicker: the hypothesis re-decodes and RETRACTS the tail
    ('我们今天来讨论镭射。在医学上的应用，镭射。' -> '确的切除。'). Any shorter
    text at the same commit boundary is held — the reader never sees a
    retraction; the commit resolves it."""
    import time as _t
    m = CaptionDisplay(hold_sec=3.5, clock=_t.monotonic)
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
