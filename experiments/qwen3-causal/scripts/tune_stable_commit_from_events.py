#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen3_streaming.metrics import (
    stable_prefix_stats,
    streaming_text_event_stats,
    word_error_rate,
)
from qwen3_streaming.stable_commit import (
    StableTextCommitState,
    join_text_units,
    update_stable_text_commit,
)


def parse_int_grid(value: str) -> list[int]:
    values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = [part.strip() for part in item.split(":")]
            if len(parts) not in (2, 3):
                raise ValueError(f"invalid range item: {item!r}")
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            if step <= 0:
                raise ValueError("range step must be > 0")
            values.extend(range(start, stop + 1, step))
        else:
            values.append(int(item))
    if not values:
        raise ValueError("grid cannot be empty")
    deduped = sorted(set(values))
    if deduped[0] < 0:
        raise ValueError("grid values must be >= 0")
    return deduped


def parse_float_grid(value: str) -> list[float]:
    values: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = [part.strip() for part in item.split(":")]
            if len(parts) not in (2, 3):
                raise ValueError(f"invalid range item: {item!r}")
            start = float(parts[0])
            stop = float(parts[1])
            step = float(parts[2]) if len(parts) == 3 else 1.0
            if step <= 0.0:
                raise ValueError("range step must be > 0")
            current = start
            while current <= stop + 1e-9:
                values.append(round(current, 6))
                current += step
        else:
            values.append(float(item))
    if not values:
        raise ValueError("grid cannot be empty")
    deduped = sorted(set(values))
    if deduped[0] < 0.0:
        raise ValueError("grid values must be >= 0")
    return deduped


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def replay_text_commit(
    events: Iterable[dict[str, Any]],
    *,
    hold_back_words: int,
    stable_iterations: int,
    min_commit_audio_sec: float = 0.0,
    normalize_commit_match: bool = False,
    final_revises_committed: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    if hold_back_words < 0:
        raise ValueError("hold_back_words must be >= 0")
    if stable_iterations <= 0:
        raise ValueError("stable_iterations must be > 0")

    state = StableTextCommitState()
    replayed: list[dict[str, Any]] = []
    last_hypothesis = ""
    for event in events:
        last_hypothesis = str(event.get("hypothesis", ""))
        update = update_stable_text_commit(
            state,
            last_hypothesis,
            hold_back_units=hold_back_words,
            stable_iterations=stable_iterations,
            normalize_for_match=normalize_commit_match,
            allow_commit=float(event.get("audio_sec", 0.0)) >= min_commit_audio_sec,
        )
        replay_event = dict(event)
        replay_event["committed"] = update.committed_text
        replay_event["display"] = update.display_text
        replay_event["unstable"] = update.unstable_text
        replay_event["committed_units"] = update.committed_unit_count
        replayed.append(replay_event)

    if final_revises_committed and replayed:
        final_update = update_stable_text_commit(
            state,
            last_hypothesis,
            hold_back_units=hold_back_words,
            stable_iterations=stable_iterations,
            normalize_for_match=normalize_commit_match,
            final=True,
            final_revises_committed=True,
        )
        replayed[-1]["committed"] = final_update.committed_text
        replayed[-1]["display"] = final_update.display_text
        replayed[-1]["unstable"] = final_update.unstable_text
        replayed[-1]["committed_units"] = final_update.committed_unit_count

    return replayed, join_text_units(state.committed_units)


def _mean(values: Iterable[float | int | None]) -> float | None:
    kept = [float(value) for value in values if value is not None]
    return statistics.mean(kept) if kept else None


def _score(row: dict[str, Any]) -> float:
    coverage = float(row.get("stable_coverage_ratio_mean") or 0.0)
    first_commit = float(row.get("first_commit_sec_mean") or 60.0)
    wer = float(row.get("wer_stable_mean") if row.get("wer_stable_mean") is not None else 1.0)
    revisions = float(row.get("committed_revision_events_total") or 0.0)
    mismatch_rate = float(row.get("stable_final_prefix_mismatch_rate") or 0.0)
    return coverage - 0.03 * first_commit - 0.5 * wer - 0.1 * revisions - mismatch_rate


def evaluate_policy(
    *,
    prediction_rows: list[dict[str, Any]],
    events_dir: Path,
    hold_back_words: int,
    stable_iterations: int,
    min_commit_audio_sec: float = 0.0,
    normalize_commit_match: bool = False,
    final_revises_committed: bool = False,
) -> dict[str, Any]:
    item_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        if row.get("error") is not None:
            continue
        item_id = str(row["id"])
        event_path = events_dir / f"{item_id}.jsonl"
        if not event_path.exists():
            raise FileNotFoundError(event_path)
        events = load_jsonl(event_path)
        replayed, committed_text = replay_text_commit(
            events,
            hold_back_words=hold_back_words,
            stable_iterations=stable_iterations,
            min_commit_audio_sec=min_commit_audio_sec,
            normalize_commit_match=normalize_commit_match,
            final_revises_committed=final_revises_committed,
        )
        final_text = str(row.get("last_hypothesis_text") or row.get("final_text") or "")
        reference = row.get("reference")
        streaming = streaming_text_event_stats(
            replayed,
            final_text=final_text,
            stable_text=committed_text,
        )
        final_prefix = stable_prefix_stats(final_text, committed_text)
        committed_words = int(final_prefix.hypothesis_words)
        consistency_ratio = (
            float(final_prefix.common_prefix_words / committed_words)
            if committed_words
            else 1.0
        )
        item_rows.append(
            {
                "id": item_id,
                "reference": reference,
                "final_text": final_text,
                "stable_committed_text": committed_text,
                "wer_stable": word_error_rate(reference, committed_text)
                if reference
                else None,
                "streaming": streaming,
                "stable_final_prefix_revision_words": final_prefix.revision_words,
                "stable_final_prefix_mismatch": final_prefix.revision_words > 0,
                "stable_final_consistency_ratio": consistency_ratio,
            }
        )

    result: dict[str, Any] = {
        "hold_back_words": hold_back_words,
        "stable_iterations": stable_iterations,
        "min_commit_audio_sec": min_commit_audio_sec,
        "normalize_commit_match": normalize_commit_match,
        "final_revises_committed": final_revises_committed,
        "count": len(item_rows),
        "wer_stable_mean": _mean(item.get("wer_stable") for item in item_rows),
        "first_commit_sec_mean": _mean(
            item["streaming"].get("first_commit_sec") for item in item_rows
        ),
        "stable_coverage_ratio_mean": _mean(
            item["streaming"].get("stable_coverage_ratio") for item in item_rows
        ),
        "stable_word_count_mean": _mean(
            item["streaming"].get("stable_word_count") for item in item_rows
        ),
        "committed_revision_events_total": sum(
            int(item["streaming"].get("committed_revision_events", 0))
            for item in item_rows
        ),
        "committed_revision_words_total": sum(
            int(item["streaming"].get("committed_revision_words", 0))
            for item in item_rows
        ),
        "stable_final_prefix_mismatch_count": sum(
            1 for item in item_rows if item["stable_final_prefix_mismatch"]
        ),
        "stable_final_prefix_mismatch_rate": (
            sum(1 for item in item_rows if item["stable_final_prefix_mismatch"])
            / len(item_rows)
            if item_rows
            else 0.0
        ),
        "stable_final_prefix_revision_words_mean": _mean(
            item["stable_final_prefix_revision_words"] for item in item_rows
        ),
        "stable_final_consistency_ratio_mean": _mean(
            item["stable_final_consistency_ratio"] for item in item_rows
        ),
    }
    result["score"] = _score(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay saved full-hypothesis streaming events and tune the stable text "
            "commit policy without rerunning ASR inference."
        )
    )
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--hold-back-words", default="0:8")
    parser.add_argument("--stable-iterations", default="1:4")
    parser.add_argument("--min-commit-audio-sec", default="0")
    parser.add_argument("--normalize-commit-match", action="store_true")
    parser.add_argument("--final-revises-committed", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_rows = load_jsonl(args.predictions_jsonl)
    hold_back_grid = parse_int_grid(args.hold_back_words)
    stable_iterations_grid = parse_int_grid(args.stable_iterations)
    min_commit_audio_sec_grid = parse_float_grid(args.min_commit_audio_sec)

    results: list[dict[str, Any]] = []
    for hold_back_words in hold_back_grid:
        for stable_iterations in stable_iterations_grid:
            for min_commit_audio_sec in min_commit_audio_sec_grid:
                results.append(
                    evaluate_policy(
                        prediction_rows=prediction_rows,
                        events_dir=args.events_dir,
                        hold_back_words=hold_back_words,
                        stable_iterations=stable_iterations,
                        min_commit_audio_sec=min_commit_audio_sec,
                        normalize_commit_match=args.normalize_commit_match,
                        final_revises_committed=args.final_revises_committed,
                    )
                )

    results.sort(key=lambda row: float(row["score"]), reverse=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"top": results[: args.top_k]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
