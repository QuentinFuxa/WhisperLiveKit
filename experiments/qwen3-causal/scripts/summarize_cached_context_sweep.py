#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    data = dict(data)
    data["summary_path"] = str(path)
    return data


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def summarize_context_sweep(
    summaries: list[dict[str, Any]],
    *,
    mel_hop_ms: int = 10,
) -> list[dict[str, Any]]:
    if mel_hop_ms <= 0:
        raise ValueError("mel_hop_ms must be > 0")
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        left_frames = _as_int(summary.get("qwen_audio_left_context_frames"))
        right_frames = _as_int(summary.get("qwen_audio_right_context_frames"))
        rows.append(
            {
                "summary_path": summary.get("summary_path", ""),
                "count": _as_int(summary.get("count")),
                "ok": _as_int(summary.get("ok")),
                "left_context_sec": left_frames * mel_hop_ms / 1000.0,
                "right_context_ms": right_frames * mel_hop_ms,
                "left_context_frames": left_frames,
                "right_context_frames": right_frames,
                "cache_bound_frames": _as_int(summary.get("cache_bound_frames")),
                "max_last_recomputed_frames": _as_int(
                    summary.get("max_last_recomputed_frames")
                ),
                "max_recomputed_context_frames": _as_int(
                    summary.get("max_recomputed_context_frames")
                ),
                "cache_bound_violations": _as_int(
                    summary.get("cache_bound_violations")
                ),
                "wer_final_mean": _as_float(summary.get("wer_final_mean")),
                "wer_latest_mean": _as_float(summary.get("wer_latest_mean")),
                "wer_stable_mean": _as_float(summary.get("wer_stable_mean")),
                "latency_mean_sec": _as_float(summary.get("latency_mean_sec")),
                "first_display_sec_mean": _as_float(
                    summary.get("first_display_sec_mean")
                ),
                "first_commit_sec_mean": _as_float(
                    summary.get("first_commit_sec_mean")
                ),
                "stable_coverage_ratio_mean": _as_float(
                    summary.get("stable_coverage_ratio_mean")
                ),
            }
        )

    baseline = max(rows, key=lambda row: row["left_context_frames"], default=None)
    baseline_wer = None if baseline is None else baseline.get("wer_final_mean")
    for row in rows:
        row_wer = row.get("wer_final_mean")
        if row_wer is None or baseline_wer is None:
            row["wer_final_delta_vs_max_left"] = None
        else:
            row["wer_final_delta_vs_max_left"] = float(row_wer - baseline_wer)
    rows.sort(key=lambda row: (row["left_context_frames"], row["right_context_frames"]))
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "left_s",
        "right_ms",
        "recompute",
        "ctx_recompute",
        "WER",
        "WER_delta",
        "first_display",
        "first_commit",
        "stable_cov",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            f"{row['left_context_sec']:.2f}",
            f"{row['right_context_ms']:.0f}",
            str(row["max_last_recomputed_frames"]),
            str(row["max_recomputed_context_frames"]),
            _fmt(row.get("wer_final_mean")),
            _fmt(row.get("wer_final_delta_vs_max_left"), signed=True),
            _fmt(row.get("first_display_sec_mean")),
            _fmt(row.get("first_commit_sec_mean")),
            _fmt(row.get("stable_coverage_ratio_mean")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _fmt(value: Any, *, signed: bool = False) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return ""
    if signed:
        return f"{numeric:+.4f}"
    return f"{numeric:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize cached full-hypothesis audio context sweep summaries."
    )
    parser.add_argument("summary_json", nargs="+", type=Path)
    parser.add_argument("--mel-hop-ms", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize_context_sweep(
        [load_summary(path) for path in args.summary_json],
        mel_hop_ms=args.mel_hop_ms,
    )
    payload = {"count": len(rows), "rows": rows}
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    table = markdown_table(rows)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
