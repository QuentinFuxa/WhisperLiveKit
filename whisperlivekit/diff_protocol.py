"""Diff-based WebSocket output protocol for WhisperLiveKit.

Instead of sending the full FrontData state on every update, the DiffTracker
computes incremental diffs — only sending new/changed lines and volatile fields.

Protocol
--------
Opt-in via query parameter: ``ws://host:port/asr?mode=diff``

First message from server:
    ``{"type": "snapshot", "seq": 1, ...full state...}``

Subsequent messages:
    ``{"type": "diff", "seq": N, "new_lines": [...], ...}``

The client reconstructs state by:
1. On ``"snapshot"``: replace all state.
2. On ``"diff"``:
   - If ``lines_pruned`` > 0: drop that many lines from the front.
   - Append ``new_lines`` to the end.
   - Replace ``buffer_*`` and ``remaining_time_*`` fields.
   - Use ``n_lines`` to verify sync (total expected line count).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from whisperlivekit.timed_objects import FrontData


@dataclass
class DiffTracker:
    """Tracks FrontData state and computes incremental diffs."""

    seq: int = 0
    _prev_lines: List[Dict[str, Any]] = field(default_factory=list)
    _sent_snapshot: bool = False

    def to_message(self, front_data: FrontData) -> Dict[str, Any]:
        """Convert a FrontData into a diff or snapshot message.

        First call returns a full snapshot. Subsequent calls return diffs
        containing only changed/new data.
        """
        self.seq += 1
        full = front_data.to_dict()
        current_lines = full["lines"]

        if not self._sent_snapshot:
            self._sent_snapshot = True
            self._prev_lines = current_lines[:]
            return {"type": "snapshot", "seq": self.seq, **full}

        # Compute diff
        msg: Dict[str, Any] = {
            "type": "diff",
            "seq": self.seq,
            "status": full["status"],
            "n_lines": len(current_lines),
            "buffer_transcription": full["buffer_transcription"],
            "buffer_diarization": full["buffer_diarization"],
            "buffer_translation": full["buffer_translation"],
            "remaining_time_transcription": full["remaining_time_transcription"],
            "remaining_time_diarization": full["remaining_time_diarization"],
        }
        if full.get("error"):
            msg["error"] = full["error"]

        # Detect front-pruning: find where current[0] appears in prev
        prune_offset = 0
        if current_lines and self._prev_lines:
            first_current = current_lines[0]
            for i, prev_line in enumerate(self._prev_lines):
                if prev_line == first_current:
                    prune_offset = i
                    break
            else:
                # current[0] not found in prev — treat all prev as pruned
                prune_offset = len(self._prev_lines)
        elif not current_lines:
            prune_offset = len(self._prev_lines)

        if prune_offset > 0:
            msg["lines_pruned"] = prune_offset

        # Find common prefix starting after pruned lines
        common = 0
        remaining_prev = len(self._prev_lines) - prune_offset
        min_len = min(remaining_prev, len(current_lines))
        while common < min_len and self._prev_lines[prune_offset + common] == current_lines[common]:
            common += 1

        # New or changed lines after the common prefix
        new_lines = current_lines[common:]
        if new_lines:
            msg["new_lines"] = new_lines

        self._prev_lines = current_lines[:]
        return msg

    def reset(self) -> None:
        """Reset state so the next call produces a fresh snapshot."""
        self.seq = 0
        self._prev_lines = []
        self._sent_snapshot = False
