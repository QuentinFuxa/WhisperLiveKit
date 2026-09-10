#!/usr/bin/env python3
"""Replay a captured caption event stream through the display model and
print the reader-visible target-row sequence.

This is the model-level replay instrument: it drives OverlayDisplayModel
directly with the recorded events (no view code) and traces the committed
caption line as the hold-drain releases queued sentences. Use it to check
what a viewer would see: sentence pacing, dim/bright transitions, and
whether completed sentences display before the stream ends.

Usage:
  .venv/bin/python scripts/replay_canonical_overlay.py <events.jsonl>
  REPLAY_PACE=0.8 .venv/bin/python scripts/replay_canonical_overlay.py <events.jsonl> --target

A captured stream and the display model tests (tests/test_overlay_replay.py)
are the regression pair: the tests assert the rules, the script shows the
sequence.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta

from whisperlivekit.caption_events import EventLog
from whisperlivekit.overlay_model import OverlayDisplayModel


def replay_canonical(path: str, hold: float = 3.5, pace: float = 0.25) -> None:
    events = EventLog.load(path).events
    t0 = events[0].t

    class C:
        now = 0.0

        def __call__(self): return self.now

        def advance(self, s): self.now += s
    clk = C()
    m = OverlayDisplayModel(hold_sec=hold, clock=clk)

    trace: list[tuple[float, str, str]] = []
    base = datetime.now()

    def snap(elapsed, etype):
        cur = m._en_plain
        prev = m._en_prev_plain
        tag = "BRIGHT" if m._en_is_final else "dim   "
        line = f"[{tag}] {cur!r}" + (f"  (prev {prev!r})" if prev else "")
        trace.append((elapsed, etype, line))

    for e in events:
        clk.advance(pace)
        started = base + timedelta(seconds=e.t - t0)
        if e.type == "translation_provisional":
            m.preview([(None, e.text)], started)
        elif e.type == "translation_final":
            m.translation([(None, e.text)], started)
        if e.type.startswith("translation"):
            snap(time.monotonic() - t0, e.type.replace("translation_", ""))
        # pump the drain across the event's pacing window
        end = time.monotonic() + pace
        while time.monotonic() < end:
            m.tick()
            time.sleep(0.02)

    # pump the drain past the last event: the reader's clock keeps running
    # after the stream ends, so queued sentences must still surface
    for _ in range(int(12 / pace)):
        m.tick()
        snap(time.monotonic() - t0, "drain")
        time.sleep(0.02)

    print(f"=== target-row reader-visible sequence: {path} ===")
    for elapsed, etype, line in trace:
        print(f"  [+{elapsed:7.2f}] {etype:10s} {line}")


if __name__ == "__main__":
    replay_canonical(
        sys.argv[1] if len(sys.argv) > 1 else "/tmp/canonical_zh_long_qwen3.jsonl",
        pace=float(os.environ.get("REPLAY_PACE", "0.25")),
    )
