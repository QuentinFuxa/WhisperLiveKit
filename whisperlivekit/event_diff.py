"""Diff a captured caption event stream against a golden sequence.

Scores how far a harnessed run is from the ideal event stream, decoupled from
display. The score is a coarse, interpretable number, not a rigorous metric:
it counts structural mismatches at the event-type level (ignoring exact text),
so it flags the failure modes we observed — fragment finals (too many
translation_finals), empty provisionals (translation_provisional with committed=""), dead paths (no
translation_provisional before translation_final) — without requiring exact text match.

Usage:
  golden = load_golden("tests/golden/zh_long.jsonl")
  captured = EventLog.load("/tmp/zh_en_events.jsonl")
  report = diff_event_streams(captured.events, golden)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from whisperlivekit.caption_events import CaptionEvent


@dataclass
class DiffReport:
    golden_finals: int = 0
    captured_finals: int = 0
    golden_drafts: int = 0
    captured_drafts: int = 0
    empty_committed_drafts: int = 0   # drafts with committed="" (informational: sentence-initial tail drafts are legitimate)
    starved_sentences: int = 0  # sentences where EVERY draft had committed="" (AlignAtt released nothing)
    finals_without_preceding_draft: int = 0  # translation_final with no translation_provisional before it (provisional starved)
    fragment_finals: bool = False  # captured_finals >> golden_finals (fragmentation)
    verdict: str = ""  # matches | partial | diverges

    def summary(self) -> str:
        lines = [
            f"translation_final: golden={self.golden_finals} captured={self.captured_finals}",
            f"translation_provisional: golden={self.golden_drafts} captured={self.captured_drafts}",
            f"empty-committed drafts: {self.empty_committed_drafts}",
            f"starved sentences (no release ever): {self.starved_sentences}",
            f"finals without preceding draft: {self.finals_without_preceding_draft}",
            f"fragment finals: {self.fragment_finals}",
            f"verdict: {self.verdict}",
        ]
        return "\n".join(lines)


def diff_event_streams(captured: List[CaptionEvent], golden: List[CaptionEvent]) -> DiffReport:
    r = DiffReport()
    r.golden_finals = sum(1 for e in golden if e.type == "translation_final")
    r.captured_finals = sum(1 for e in captured if e.type == "translation_final")
    r.golden_drafts = sum(1 for e in golden if e.type == "translation_provisional")
    r.captured_drafts = sum(1 for e in captured if e.type == "translation_provisional")
    r.empty_committed_drafts = sum(
        1 for e in captured if e.type == "translation_provisional" and not e.committed.strip()
    )

    # starved sentences: every draft in a sentence had committed="" — AlignAtt
    # never released anything for that sentence. A few empty-committed drafts
    # are expected (sentence-initial tail drafts precede any release); the
    # pathology is a sentence where NOTHING was ever released.
    r.starved_sentences = 0
    saw_draft_in_sentence = False
    saw_release = False
    for e in captured:
        if e.type == "translation_provisional":
            saw_draft_in_sentence = True
            if e.committed.strip():
                saw_release = True
        elif e.type == "translation_final":
            if saw_draft_in_sentence and not saw_release:
                r.starved_sentences += 1
            saw_draft_in_sentence = False
            saw_release = False

    # finals without a preceding translation_provisional (provisional starved)
    saw_draft = False
    for e in captured:
        if e.type == "translation_provisional":
            saw_draft = True
        elif e.type == "translation_final":
            if not saw_draft:
                r.finals_without_preceding_draft += 1
            saw_draft = False

    # fragmentation: captured has 2x or more finals than golden
    if r.golden_finals > 0 and r.captured_finals >= r.golden_finals * 2:
        r.fragment_finals = True

    if r.captured_finals == r.golden_finals and r.starved_sentences == 0 and not r.fragment_finals:
        r.verdict = "matches"
    elif r.fragment_finals or r.starved_sentences > 0 or r.finals_without_preceding_draft > 0:
        r.verdict = "diverges"
    else:
        r.verdict = "partial"
    return r
