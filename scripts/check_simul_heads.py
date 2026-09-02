#!/usr/bin/env python3
"""Simul-MT health check: ONE number that fails on mis-calibrated heads.

Draft coverage — for each finalized translation, how much of its content was
already visible as a provisional draft before the final landed (word-level
recall: the fraction of the final's words that appeared in any earlier
provisional). The commit policy's whole job is releasing the committed
prefix of the draft early; with a working head the draft covers most of the
final's text before it lands. With a dead head (today: the calibrated (9,5)
has decode-step source mass 0.044) the drafts are stuck fragments:
'Today, we will' for 8s, 'Future' for the whole last sentence.

    coverage = mean over finals of |final_words ∩ union(pre-final provs)| / |final_words|

Fails today  (~0.2-0.4).  Passes after the heads are rebuilt (~0.7+).

Usage:
  .venv/bin/python scripts/check_simul_heads.py /path/to/events.jsonl
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

PASS_COVERAGE = 0.6


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def check(path: str) -> float:
    evs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    t0 = evs[0]["t"]
    provs: list[tuple[float, str]] = []
    finals: list[tuple[float, str]] = []
    for e in evs:
        if e["type"] == "translation_provisional":
            provs.append((e["t"], e["text"]))
        elif e["type"] == "translation_final":
            finals.append((e["t"], e["text"]))
    if not finals:
        print("no translation finals in the stream")
        return 0.0

    ratios = []
    for ft, ftext in finals:
        seen_words: set[str] = set()
        for pt, ptext in provs:
            if pt < ft:
                seen_words |= _words(ptext)
        fw = _words(ftext)
        r = (len(fw & seen_words) / len(fw)) if fw else 0.0
        ratios.append(r)
        mark = "OK " if r >= PASS_COVERAGE else "LOW"
        print(f"  [{mark}] final@+{ft-t0:6.2f}s coverage={r:.2f}  {ftext[:52]!r}")
    cov = sum(ratios) / len(ratios)
    print(f"\ndraft coverage: {cov:.2f}  ({len(finals)} finals)  pass >= {PASS_COVERAGE}")
    print("RESULT:", "PASS — the commit policy releases drafts" if cov >= PASS_COVERAGE
          else "FAIL — drafts are held/stuck (mis-calibrated head?)")
    return cov


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/final5.jsonl"
    check(path)
