"""Job 2: timestamp manufacture for ASR backends that emit text forward
without per-word timestamps.

Factored from ``voxtral_mlx_asr.py``'s ``_word_time_range``,
``_audio_pos_to_time``, and ``word_audio_starts`` tracking. Provides a
helper that assigns start and end times to forward-emit tokens based on
the decoder position where each word started and ended.

Usage:
    tracker = WordTimestampTracker(secs_per_token=0.08, delay_secs=0.0)
    tracker.record_word_start(audio_pos, "hello")
    tracker.record_word_end(audio_pos)
    t0, t1 = tracker.word_time_range(word_idx)
"""
from __future__ import annotations

from typing import List, Optional, Tuple


class WordTimestampTracker:
    """Track per-word audio positions and convert them to wall-clock times.

    A "word" is a whitespace-delimited token in the decoded text. The
    tracker records the decoder position (relative to the prefix/prompt
    end) where each word's first and last token was emitted, then
    converts positions to seconds via ``secs_per_token`` with an optional
    ``delay_secs`` compensation and a running ``time_offset``.
    """

    def __init__(self, secs_per_token: float, delay_secs: float = 0.0):
        self._secs_per_token = secs_per_token
        self._delay_secs = delay_secs
        self._time_offset = 0.0
        self._word_audio_starts: List[int] = []
        self._word_audio_ends: List[int] = []
        self._current_word_pos: Optional[int] = None

    @property
    def time_offset(self) -> float:
        return self._time_offset

    @time_offset.setter
    def time_offset(self, value: float) -> None:
        self._time_offset = value

    @property
    def word_audio_starts(self) -> List[int]:
        return self._word_audio_starts

    @property
    def word_audio_ends(self) -> List[int]:
        return self._word_audio_ends

    @property
    def has_current_word(self) -> bool:
        return self._current_word_pos is not None

    def reset(self) -> None:
        """Clear all tracked positions and reset the current-word state."""
        self._word_audio_starts = []
        self._word_audio_ends = []
        self._current_word_pos = None

    def record_word_start(self, audio_pos: int) -> None:
        """Record the start of a new word at ``audio_pos``.

        Closes the previous word (if any) at ``audio_pos`` first.
        """
        if self._current_word_pos is not None:
            self._word_audio_ends.append(audio_pos)
        self._word_audio_starts.append(audio_pos)
        self._current_word_pos = audio_pos

    def record_word_end(self, audio_pos: int) -> None:
        """Record the end of the current word at ``audio_pos``."""
        if self._current_word_pos is not None:
            self._word_audio_ends.append(audio_pos)
            self._current_word_pos = None

    def close_current_word(self, audio_pos: int) -> None:
        """Close the current word if one is being built (alias for record_word_end)."""
        if self._current_word_pos is not None:
            self._word_audio_ends.append(audio_pos)
            self._current_word_pos = None

    def audio_pos_to_time(self, pos: int) -> float:
        """Convert an audio position (relative to prefix end) to seconds."""
        return max(0.0, pos * self._secs_per_token - self._delay_secs + self._time_offset)

    def word_time_range(self, word_idx: int) -> Tuple[float, float]:
        """Compute (start, end) time for a word using tracked positions.

        Falls back to estimating from neighboring positions when exact
        data is unavailable (e.g. the last word is still being built).
        """
        starts = self._word_audio_starts
        ends = self._word_audio_ends

        if not starts:
            return self._time_offset, self._time_offset

        # Start position for this word
        if word_idx < len(starts):
            t0 = self.audio_pos_to_time(starts[word_idx])
        else:
            last_pos = ends[-1] if ends else starts[-1]
            t0 = self.audio_pos_to_time(last_pos + 1)

        # End position: use the start of the next word, or the end of this word
        if word_idx + 1 < len(starts):
            t1 = self.audio_pos_to_time(starts[word_idx + 1])
        elif word_idx < len(ends):
            t1 = self.audio_pos_to_time(ends[word_idx] + 1)
        else:
            last_pos = starts[word_idx] if word_idx < len(starts) else (ends[-1] if ends else 0)
            t1 = self.audio_pos_to_time(last_pos + 1)

        return t0, t1
