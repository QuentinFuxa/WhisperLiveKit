#!/usr/bin/env python3
"""Summarize the fixed M5 screening reports without rerunning inference.

Usage: python analyze.py REPORT_DIRECTORY --output OUTPUT_DIRECTORY
Plain schema-3.1 JSON and the same files compressed as .json.gz are accepted.
Missing runs remain pending. Numerical gates are not an integration decision.
"""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

BACKENDS = {
    "whisper-small-la": "Whisper small MLX / LocalAgreement",
    "qwen-metal": "Qwen3 0.6B / Metal",
    "qwen-native": "Qwen3 0.6B / native MLX",
    "nemotron-native": "Nemotron 0.6B / native MLX",
}
LANGUAGES = ("en", "fr", "zh")
LATENCY = ("finalization_time_s", "first_visible_time_s", "source_end_lag_s")
MEMORY = ("rss_peak_bytes", "mlx_peak_bytes")
PACING = "speed>0: absolute chunk-end deadlines; no post-feed sleep. speed=0: immediate"


def percentile(values, q=0.95):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def summarize(rows, language):
    successful = [row for row in rows if row["status"] == "ok"]
    score_key, units = ("cer_details", "ref_chars") if language == "zh" else ("wer_details", "ref_words")
    reference_units = sum(row[score_key].get(units, 0) for row in successful)
    errors = sum(sum(row[score_key].get(key, 0) for key in ("substitutions", "insertions", "deletions"))
                 for row in successful)
    result = {
        "n": len(rows), "completed": len(successful),
        "invalid_timestamps": sum(not (row["timing_valid"] and row["timing_monotonic"]) for row in successful),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "quality": errors / reference_units if reference_units else None,
        "reference_units": reference_units, "errors": errors,
    }
    for key in LATENCY:
        values = [row[key] for row in successful if row[key] is not None]
        result[key + "_p95"] = percentile(values)
        result[key + "_count"] = len(values)
    for key in MEMORY:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = max(values, default=None)
        result[key + "_count"] = len(values)
    return result


def identities(report):
    return Counter((row["sample"], row["repeat"], row["audio_sha256"], row["reference"], row["duration_s"])
                   for row in report["results"])


def readiness(report, language, *, continuous=False):
    reasons = []
    if report["benchmark_version"] != "3.1" or report["measurement"].get("audio_pacing") != PACING:
        reasons.append("missing corrected chunk-end pacing")
    if report["config"]["feed_speed"] != 1:
        reasons.append("not paced at speed 1")
    if report["system_info"].get("git_dirty") is not False:
        reasons.append("source tree was dirty or not recorded")
    if not report.get("corpus_sha256") or not report.get("model_artifacts"):
        reasons.append("missing corpus or model provenance")
    rows = report["results"]
    if continuous:
        if len(rows) != 1 or abs(rows[0]["duration_s"] - 600) > .001:
            reasons.append("expected one ten-minute continuous stream")
        if any(row["reference"] for row in rows):
            reasons.append("truncated continuous stream must not have an aggregate reference")
    else:
        if Counter(row["repeat"] for row in rows) != {1: 30, 2: 30, 3: 30}:
            reasons.append("expected three passes of 30 clips")
        for repeat in (1, 2, 3):
            if len({row["sample"] for row in rows if row["repeat"] == repeat}) != 30:
                reasons.append(f"pass {repeat} does not contain 30 distinct clips")
    if any(row["language"] != language for row in rows):
        reasons.append("mixed languages")
    if any(row["status"] != "ok" or row["translation_errors"] for row in rows):
        reasons.append("failed or skipped clip")
    if any(not row["timing_valid"] or not row["timing_monotonic"] for row in rows):
        reasons.append("invalid timestamp ordering")
    if not continuous and (len(report["warmup_results"]) != 1 or report["warmup_results"][0]["status"] != "ok"):
        reasons.append("missing or failed warmup")
    return reasons


def compare(candidate, baseline, language):
    issues = readiness(candidate, language) + readiness(baseline, language)
    if identities(candidate) != identities(baseline):
        issues.append("clip identities, repetitions, audio hashes or references differ")
    if candidate["corpus_sha256"] != baseline["corpus_sha256"]:
        issues.append("corpus hashes differ")
    if candidate["measurement"]["quality"] != baseline["measurement"]["quality"]:
        issues.append("text normalization differs")
    for key in ("cpu", "ram_gb", "machine"):
        if candidate["system_info"].get(key) != baseline["system_info"].get(key):
            issues.append(f"hardware differs: {key}")
    if issues:
        return {"numerical_gate": "incomplete", "issues": sorted(set(issues))}
    passes = []
    for repeat in (1, 2, 3):
        left, right = [summarize([row for row in report["results"] if row["repeat"] == repeat], language)
                       for report in (candidate, baseline)]
        entry = {"repeat": repeat, "quality_delta_pp": None, "gains": {}}
        if left["quality"] is not None and right["quality"] is not None:
            entry["quality_delta_pp"] = 100 * (left["quality"] - right["quality"])
        for key in ("finalization_time_s_p95", *MEMORY):
            a, b = left[key], right[key]
            count_key = key + "_count" if key in MEMORY else "finalization_time_s_count"
            complete = left[count_key] == right[count_key] == 30
            entry["gains"][key] = 1 - a / b if complete and a is not None and b is not None and b > 0 else None
        passes.append(entry)
    quality_ok = all(row["quality_delta_pp"] is not None and row["quality_delta_pp"] <= 1 for row in passes)
    # The same metric must improve by 20% on every pass; do not mix RSS and MLX.
    qualifying_metrics = [key for key in passes[0]["gains"]
                          if all(row["gains"][key] is not None and row["gains"][key] >= .2 for row in passes)]
    return {"numerical_gate": "pass" if quality_ok and qualifying_metrics else "fail",
            "quality_within_one_point_all_passes": quality_ok,
            "qualifying_metrics": qualifying_metrics, "passes": passes,
            "streaming_review": "required; inspect the continuous run and retained hypotheses"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = {}
    result = {"reports": {}, "comparisons": {}, "pending": [],
              "environment_review": "Power conditions changed during this series; see METHODS.md before interpreting latency gates."}
    table = ["# M5 backend screening", "",
             "Generated from the retained schema-3.1 reports. Incomplete runs are not ranked.", "",
             result["environment_review"], "",
             "| Backend | Language | Completed | WER/CER % | EOF p95 s | First visible p95 s | Source-end lag p95 s | RSS GiB | MLX GiB |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]

    def fmt(value, factor=1):
        return f"{value * factor:.3f}" if value is not None else "—"

    for backend, label in BACKENDS.items():
        for language in LANGUAGES:
            for continuous in (False, True) if language == "zh" else (False,):
                key = f"{backend}-{language}" + ("-continuous" if continuous else "")
                path = args.reports / (key + ".json")
                if not path.exists():
                    path = args.reports / (key + ".json.gz")
                if not path.exists():
                    result["pending"].append(key)
                    continue
                raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
                report = json.loads(raw)
                if report["benchmark_version"] != "3.1":
                    raise ValueError(f"Unsupported report schema: {path}")
                loaded[key] = report
                summary = summarize(report["results"], language)
                result["reports"][key] = {
                    "file": path.name, "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
                    "source": report["system_info"], "summary": summary,
                    "readiness_issues": readiness(report, language, continuous=continuous),
                    "warmup": [{k: row[k] for k in ("status", "startup_time_s", *MEMORY)}
                               for row in report["warmup_results"]],
                    "passes": {str(repeat): summarize([row for row in report["results"] if row["repeat"] == repeat], language)
                               for repeat in sorted({row["repeat"] for row in report["results"]})},
                }
                if continuous:
                    continue
                values = [fmt(summary["quality"], 100),
                          *(fmt(summary[k + "_p95"]) for k in LATENCY),
                          *(fmt(summary[k], 1 / 1024**3) for k in MEMORY)]
                table.append(f"| {label} | {language} | {summary['completed']}/{summary['n']} | " + " | ".join(values) + " |")
    table += ["", "Quality pools edit counts; latency p95 pools successful clips. Memory columns describe separate measured quantities and must not be added.",
              "", "## Per-pass numerical gate", "",
              "The same latency or memory metric must improve by at least 20% in each pass, with no more than one percentage point worse quality in any pass. Streaming review is separate.", ""]
    for candidate in ("qwen-native", "nemotron-native"):
        for baseline in ("whisper-small-la", "qwen-metal"):
            for language in LANGUAGES:
                a, b = f"{candidate}-{language}", f"{baseline}-{language}"
                if a not in loaded or b not in loaded:
                    continue
                comparison = compare(loaded[a], loaded[b], language)
                result["comparisons"][f"{a}-vs-{baseline}"] = comparison
                table.append(f"- {candidate} / {baseline}, {language}: **{comparison['numerical_gate']}**.")
    if result["pending"]:
        table += ["", "Pending: " + ", ".join(result["pending"]) + "."]
    table += ["", "The original JSON retains failed hypotheses, configuration, package versions and model/audio hashes. Passing this gate alone does not establish a backend's suitability.", ""]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output / "summary.md").write_text("\n".join(table))
    print(f"{len(loaded)}/16 reports; {len(result['pending'])} pending; output: {args.output}")


if __name__ == "__main__":
    main()
