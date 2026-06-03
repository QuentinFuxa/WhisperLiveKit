#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a teacher-annotated ASR manifest.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-teacher-wer", type=float, default=0.35)
    parser.add_argument(
        "--allow-missing-wer",
        action="store_true",
        help="Keep records without teacher_wer when teacher succeeded.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def teacher_filter_reason(
    record: dict[str, object],
    *,
    max_teacher_wer: float,
    allow_missing_wer: bool = False,
) -> str | None:
    if record.get("teacher_error") is not None:
        return "teacher_error"
    teacher_wer = record.get("teacher_wer")
    if teacher_wer is None:
        return None if allow_missing_wer else "missing_teacher_wer"
    if float(teacher_wer) > max_teacher_wer:
        return "teacher_wer"
    return None


def filter_records(
    records: list[dict[str, object]],
    *,
    max_teacher_wer: float,
    allow_missing_wer: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    kept: list[dict[str, object]] = []
    rejected = Counter()
    sources = Counter()
    alignment_sources = Counter()
    for record in records:
        reason = teacher_filter_reason(
            record,
            max_teacher_wer=max_teacher_wer,
            allow_missing_wer=allow_missing_wer,
        )
        if reason is not None:
            rejected[reason] += 1
            continue
        kept.append(record)
        sources[str(record.get("source", ""))] += 1
        alignment_sources[str(record.get("alignment_source", ""))] += 1
    summary = {
        "input": len(records),
        "kept": len(kept),
        "rejected": len(records) - len(kept),
        "max_teacher_wer": max_teacher_wer,
        "reject_reasons": dict(rejected),
        "sources": dict(sources),
        "alignment_sources": dict(alignment_sources),
    }
    return kept, summary


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input_jsonl)
    kept, summary = filter_records(
        records,
        max_teacher_wer=args.max_teacher_wer,
        allow_missing_wer=args.allow_missing_wer,
    )
    write_jsonl(args.output_jsonl, kept)
    summary_path = args.output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
