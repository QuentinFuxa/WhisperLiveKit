#!/usr/bin/env python3
"""Replay a captured canonical caption event stream through the REAL overlay
view, capturing what the src row actually RENDERS (including the typing
thread's intermediate frames), then run the flicker detector over the
RENDERED sequence — not the model state.

Flicker = the rendered text loses a suffix and that text re-appears soon
after. A healthy stream only grows, flips style at commits, and resets at
promotes.

Usage: .venv/bin/python scripts/replay_canonical_overlay.py <events.jsonl>
"""
from __future__ import annotations
import sys, time
from datetime import datetime

from whisperlivekit.overlay import OverlayRenderer
from whisperlivekit.overlay_model import OverlayDisplayModel
from whisperlivekit.caption_events import EventLog


def replay_canonical(path: str, hold: float = 3.5, pace: float = 0.25) -> None:
    events = EventLog.load(path).events
    t0 = events[0].t

    r = OverlayRenderer(overlay_mode="both")

    class C:
        now = 0.0
        def __call__(self): return self.now
        def advance(self, s): self.now += s
    clk = C()
    r._model = OverlayDisplayModel(hold_sec=hold, clock=clk)

    # capture what actually renders (typing thread + hard-swaps both funnel
    # through _set / _render_src_text; AppKit is None here so both go to _set)
    rendered: list[tuple[float, str, str]] = []  # (elapsed, kind, text)
    # target-row (EN) rendered line per caption event, for the reader-visible
    # sentence-sequence trace (--target)
    target_trace: list[tuple[float, str, str]] = []

    class F: pass
    r._field_partial = fp = object()

    def traced_set(field, value):
        if field is fp:
            rendered.append((time.monotonic() - t0, "render", str(value)))
    r._set = traced_set

    import sys as _sys
    trace_target = "--target" in _sys.argv

    def snap_target(elapsed, etype):
        if not trace_target:
            return
        m = r._model
        cur = "".join(sp.text for sp in m._en_spans)
        prev = "".join(sp.text for sp in m._en_prev_spans)
        tag = "BRIGHT" if m._en_is_final else "dim   "
        target_trace.append((elapsed, etype, f"[{tag}] {cur!r}" + (f"  (prev {prev!r})" if prev else "")))

    for e in events:
        clk.advance(pace)
        now = datetime.now()
        if e.type == "transcription_provisional":
            r.partial("mic", e.text, now)
        elif e.type == "transcription_final":
            r.final("mic", [(None, e.text)], now)
        elif e.type == "translation_provisional":
            r.preview("mic", [(None, e.text)], now)
        elif e.type == "translation_final":
            r.translation("mic", [(None, e.text)], now)
        if e.type.startswith("translation"):
            snap_target(time.monotonic() - t0, e.type.replace("translation_", ""))
        # pump the drainer across the event's pacing window so the typing
        # thread's intermediate frames are captured
        end = time.monotonic() + pace
        while time.monotonic() < end:
            st = r._model.tick()
            if st is not None:
                r._reconcile(st)
            time.sleep(0.02)

    # pump the drainer past the last event so queued sentences drain as they
    # would in real time (the reader's clock keeps running after the last event)
    if trace_target:
        for _ in range(int(12 / pace)):
            st = r._model.tick()
            if st is not None:
                r._reconcile(st)
            snap_target(time.monotonic() - t0, "drain")
            time.sleep(0.02)
    if trace_target:
        print(f"=== target-row reader-visible sequence: {path} ===")
        for elapsed, etype, line in target_trace:
            print(f"  [+{elapsed:7.2f}] {etype:10s} {line}")
        return

    print(f"=== {len(rendered)} rendered src frames from {path} ===")
    flickers = 0
    prev = ""
    i = 0
    while i < len(rendered):
        at, _ty, full = rendered[i]
        if prev and full and len(full) < len(prev) and prev.startswith(full):
            vanished = prev[len(full):].lstrip()
            for j in range(i + 1, min(i + 8, len(rendered))):
                full2 = rendered[j][2]
                if len(full2) > len(full) and (
                        full2 == prev or full2[len(full):].startswith(vanished)):
                    print(f"  [+{at:6.2f}] {prev!r} -> {full!r} -> reappeared {full2!r}")
                    flickers += 1
                    break
        if full:
            prev = full
        i += 1
    print("src flicker events (rendered frames):", flickers)


if __name__ == "__main__":
    replay_canonical(sys.argv[1] if len(sys.argv) > 1 else "/tmp/canonical_zh_long_qwen3.jsonl", pace=float(__import__("os").environ.get("REPLAY_PACE", "0.25")))
