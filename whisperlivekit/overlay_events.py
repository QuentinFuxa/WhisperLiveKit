"""Timestamped event log for the overlay display (decouples capture from display).

Records what the MT produces and when: each provisional/final becomes a
timestamped ``OverlayEvent`` (wall-clock, utterance id, kind, plain text,
segments-with-diff). The display (overlay/terminal/test) consumes the log
and can be optimized/replayed against it deterministically — decoupling
"what the MT produced and when" from "how to display it".

This is the foundation for tuning display timing (streaming-append delays,
hold timers, expiry) from a deterministic event stream rather than from the
live callbacks. The overlay records events as they arrive; tests replay a
recorded log and assert on the rendered DisplayState.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class OverlayEvent:
    """One MT output event (provisional or final) at a wall-clock time."""
    wallclock: float              # monotonic seconds (or injected clock)
    utt_id: float                 # started_at.timestamp() (utterance identity)
    kind: str                     # "provisional" | "final"
    plain: str                    # the plain text (for comparison)
    segments: list                # the segments (with optional diff for finals)


class OverlayEventLog:
    """Append-only log of MT events. The overlay records into it; the display
    and tests read from it. Replaying a log deterministically reproduces the
    exact display sequence (modulo the display's own timing choices)."""

    def __init__(self, clock=time.monotonic) -> None:  # type: ignore[name-defined]
        self._clock = clock
        self._events: List[OverlayEvent] = []

    def record_preview(self, segments: list, started_at: Optional[datetime]) -> None:
        utt = started_at.timestamp() if started_at is not None else 0.0
        from .overlay import _segments_text  # local import to avoid cycle
        self._events.append(OverlayEvent(
            wallclock=self._clock(), utt_id=utt, kind="provisional",
            plain=_segments_text(segments), segments=segments,
        ))

    def record_final(self, segments: list, started_at: Optional[datetime]) -> None:
        utt = started_at.timestamp() if started_at is not None else 0.0
        from .overlay import _segments_text
        self._events.append(OverlayEvent(
            wallclock=self._clock(), utt_id=utt, kind="final",
            plain=_segments_text(segments), segments=segments,
        ))

    def events(self) -> List[OverlayEvent]:
        return list(self._events)

    def save(self, path: str) -> None:
        """Save the log as JSONL for offline replay/tuning."""
        import json
        with open(path, "w", encoding="utf-8") as f:
            for e in self._events:
                # segments may carry diff spans; serialize plainly
                segs = []
                for seg in e.segments:
                    segs.append(list(seg))  # tuples -> lists for JSON
                f.write(json.dumps({
                    "wallclock": e.wallclock, "utt_id": e.utt_id, "kind": e.kind,
                    "plain": e.plain, "segments": segs,
                }, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str) -> "OverlayEventLog":
        """Load a JSONL log for deterministic replay."""
        import json
        log = cls(clock=lambda: 0.0)
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                log._events.append(OverlayEvent(
                    wallclock=d["wallclock"], utt_id=d["utt_id"], kind=d["kind"],
                    plain=d["plain"], segments=[tuple(s) for s in d["segments"]],
                ))
        return log


# import time for the default arg (kept here so the module imports cleanly)
import time  # noqa: E402
