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
    if report.translation_config:
        out.write(f"Translation: {_translation_label(report.translation_config)}\n")
    out.write(f"Feed speed: {report.feed_speed} (0 = immediate, 1 = real time)\n")
    translated = any(r.translation_reference for r in report.results)
    header = (f"{'Sample':<26} {'State':<8} {'WER/CER%':>8} {'ASR RTF':>8} {'Wall s':>8} "
              f"{'First s':>8} {'EOF s':>8}")
    if translated:
        header += f" {'chrF':>8}"
    out.write(header + "\n")
    for result in report.results:
        out.write(
            f"{result.sample_name:<26} {result.status:<8} "
            f"{_number((result.cer if result.cer is not None else result.wer) * 100 if result.cer is not None or result.wer is not None else None):>8} "
            f"{_number(result.rtf, 3):>8} {_number(result.wall_time_s):>8} "
            f"{_number(result.first_visible_time_s):>8} {_number(result.finalization_time_s):>8}"
        )
        if translated:
            out.write(f" {_number(result.chrf):>8}")
        out.write("\n")
        if result.error:
            out.write(f"  {result.error}\n")
        if result.status == "ok" and not (result.timing_valid and result.timing_monotonic):
            out.write("  Timestamp ordering/bounds require inspection.\n")
    out.write(f"\n{len(report.successful_results)}/{report.n_samples} completed; {report.n_failed} failed.\n")
    out.write(f"Successful samples: weighted WER {_number(report.weighted_wer * 100 if report.weighted_wer is not None else None)}%; ASR RTF {_number(report.overall_rtf, 3)}\n")
    if report.weighted_cer is not None:
        out.write(f"Chinese/Japanese character error rate: {report.weighted_cer * 100:.2f}%\n")
    _print_translation_summary(report, out)
    out.write("ASR RTF measures inference calls. Wall time includes audio pacing and EOF drain.\n")
    out.write("First visible text is measured only at speed=1; it is not per-word latency. EOF time starts after actual feeding ends.\n")
    out.write("Failed samples are excluded from scores; inspect their records before comparing runs.\n\n")


def _translation_label(config) -> str:
    parts = [str(config.get("target_language", ""))]
    backend = config.get("translation_backend", "")
    if backend:
        parts.append(backend)
    model = config.get("mlx_llm_mt_model") or config.get("nllb_size")
    if model:
        parts.append(str(model))
    return " / ".join(part for part in parts if part)


def _print_translation_summary(report: BenchmarkReport, out: TextIO) -> None:
    """Corpus chrF per language pair, plus the translation latencies that exist."""
    from whisperlivekit.benchmark.chrf import corpus_chrf_by_language

    summary = corpus_chrf_by_language(report.translation_scored_results)
    if not summary:
        return
    out.write("Translation quality (corpus chrF from summed n-gram statistics; "
              "ASR errors flow into it):\n")
    for language, values in sorted(summary.items()):
        out.write(f"  {language} -> {_translation_label(report.translation_config)}: "
                  f"{_number(values['corpus_chrf'])}% over {values['n_rows']} rows\n")
    finals = [r.final_translation_time_s for r in report.translation_scored_results
              if r.final_translation_time_s is not None]
    if finals:
        out.write(f"Complete translation: p95 {_number(report.percentile('final_translation_time_s'))}s "
                  f"after feed start; first visible translation p95 "
                  f"{_number(report.percentile('first_translation_time_s'))}s\n")
    leads = [r.provisional_lead_s for r in report.translation_scored_results
             if r.provisional_lead_s is not None]
    if leads:
        out.write(f"Draft visible {_number(report.percentile('provisional_lead_s'))}s "
                  f"(p95) before the complete translation\n")


def print_transcriptions(report: BenchmarkReport, out: TextIO = sys.stderr) -> None:
    for result in report.results:
        out.write(f"\n{result.sample_name} ({result.language}, {result.status})\n")
        out.write(f"  ref: {result.reference}\n  hyp: {result.hypothesis}\n")
        if result.translation_reference or result.translation_hypothesis:
            out.write(f"  translation ref: {result.translation_reference}\n")
            out.write(f"  translation hyp: {result.translation_hypothesis}\n")


def write_json(report: BenchmarkReport, path: str) -> None:
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
