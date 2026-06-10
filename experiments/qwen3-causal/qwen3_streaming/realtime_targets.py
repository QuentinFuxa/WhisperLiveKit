"""Word-alignment data structures used by manifest preparation.

The frame-synchronous ``[P]``/``[W]`` target builder that used to live here
belonged to the abandoned frame-synchronous training line and was removed
(see RUNS.md 2026-06-10 cleanup). What remains is the alignment record shared
by the data-prep scripts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WordAlignment:
    text: str
    start_sec: float
    end_sec: float


def heuristic_word_alignments(text: str, duration_sec: float) -> list[WordAlignment]:
    """Approximate word timings when a dataset has only utterance-level text.

    Not good enough for training targets; useful as a fallback when a forced
    aligner fails on a row. Durations are distributed by word character length
    so long words get a little more time than short words.
    """

    words = [word.strip() for word in text.split() if word.strip()]
    if not words or duration_sec <= 0.0:
        return []
    weights = [max(1, len(word)) for word in words]
    total = float(sum(weights))
    cursor = 0.0
    alignments: list[WordAlignment] = []
    for word, weight in zip(words, weights, strict=True):
        end = min(duration_sec, cursor + duration_sec * (float(weight) / total))
        alignments.append(WordAlignment(text=word, start_sec=cursor, end_sec=end))
        cursor = end
    if alignments:
        last = alignments[-1]
        alignments[-1] = WordAlignment(
            text=last.text,
            start_sec=last.start_sec,
            end_sec=duration_sec,
        )
    return alignments
