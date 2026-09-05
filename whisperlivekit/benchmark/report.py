"""Terminal summaries and versioned JSON export of diagnostic benchmarks."""

import json
import sys
from pathlib import Path
from typing import TextIO

from whisperlivekit.benchmark.metrics import BenchmarkReport


def _number(value, decimals=2):
    return f"{value:.{decimals}f}" if value is not None else "n/a"


def print_report(report: BenchmarkReport, out: TextIO = sys.stderr) -> None:
    out.write(f"\nWhisperLiveKit benchmark: {report.backend} / {report.model_size}\n")
    out.write(f"Feed speed: {report.feed_speed} (0 = immediate, 1 = real time)\n")
    out.write(f"{'Sample':<26} {'State':<8} {'WER %':>7} {'ASR RTF':>8} {'Wall s':>8} {'First s':>8}\n")
    for result in report.results:
        out.write(
            f"{result.sample_name:<26} {result.status:<8} "
            f"{_number(result.wer * 100 if result.wer is not None else None):>7} "
            f"{_number(result.rtf, 3):>8} {_number(result.wall_time_s):>8} "
            f"{_number(result.first_text_time_s):>8}\n"
        )
        if result.error:
            out.write(f"  {result.error}\n")
        if result.status == "ok" and not (result.timing_valid and result.timing_monotonic):
            out.write("  Timestamp ordering/bounds require inspection.\n")
    out.write(f"\n{len(report.successful_results)}/{report.n_samples} completed; {report.n_failed} failed.\n")
    out.write(f"Successful samples: weighted WER {_number(report.weighted_wer * 100 if report.weighted_wer is not None else None)}%; ASR RTF {_number(report.overall_rtf, 3)}\n")
    out.write("ASR RTF measures inference calls. Wall time includes audio pacing and EOF drain.\n")
    out.write("First text is measured only at speed=1; it is not per-word latency.\n")
    out.write("Failed samples are excluded from scores; inspect their records before comparing runs.\n\n")


def print_transcriptions(report: BenchmarkReport, out: TextIO = sys.stderr) -> None:
    for result in report.results:
        out.write(f"\n{result.sample_name} ({result.language}, {result.status})\n")
        out.write(f"  ref: {result.reference}\n  hyp: {result.hypothesis}\n")


def write_json(report: BenchmarkReport, path: str) -> None:
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
