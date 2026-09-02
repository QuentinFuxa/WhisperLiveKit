"""Benchmark report formatting — terminal tables and JSON export."""

import json
import sys
from pathlib import Path
from typing import TextIO

from whisperlivekit.benchmark.metrics import BenchmarkReport

# ANSI color codes
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _wer_color(wer: float) -> str:
    if wer < 0.15:
        return GREEN
    elif wer < 0.30:
        return YELLOW
    return RED


def _rtf_color(rtf: float) -> str:
    if rtf < 0.5:
        return GREEN
    elif rtf < 1.0:
        return YELLOW
    return RED


def _lat_color(ms: float) -> str:
    if ms < 500:
        return GREEN
    elif ms < 1000:
        return YELLOW
    return RED


def print_report(report: BenchmarkReport, out: TextIO = sys.stderr) -> None:
    """Print a comprehensive benchmark report to the terminal."""
    w = out.write

    # Header
    w(f"\n{BOLD}  WhisperLiveKit Benchmark Report{RESET}\n")
    w(f"  {'─' * 72}\n")

    si = report.system_info
    w(f"  Backend:      {CYAN}{report.backend}{RESET}\n")
    w(f"  Model:        {report.model_size}\n")
    if report.translation_backend:
        variant = " (simultaneous)" if report.simultaneous else ""
        w(f"  Translation:  {CYAN}{report.translation_backend}{variant}{RESET}"
          f"  → {report.target_language or '?'}\n")
    w(f"  Accelerator:  {si.get('accelerator', 'unknown')}\n")
    w(f"  CPU:          {si.get('cpu', 'unknown')}\n")
    w(f"  RAM:          {si.get('ram_gb', '?')} GB\n")
    w(f"  Timestamp:    {report.timestamp}\n")
    w(f"  {'─' * 72}\n\n")

    # Per-sample table
    w(f"  {BOLD}{'Sample':<20} {'Lang':>4} {'Dur':>5} {'WER':>7} "
      f"{'RTF':>6} {'Lat(avg)':>8} {'Lat(p95)':>8} {'Calls':>5} {'Lines':>5}{RESET}\n")
    w(f"  {'─' * 72}\n")

    for r in report.results:
        wc = _wer_color(r.wer)
        rc = _rtf_color(r.rtf)
        lc = _lat_color(r.avg_latency_ms)

        name = r.sample_name[:20]
        wer_str = f"{r.wer * 100:>6.1f}%" if r.wer_applicable else "   N/A"
        w(f"  {name:<20} {r.language:>4} {r.duration_s:>4.1f}s "
          f"{wc}{wer_str}{RESET} "
          f"{rc}{r.rtf:>5.2f}x{RESET} "
          f"{lc}{r.avg_latency_ms:>7.0f}ms{RESET} "
          f"{lc}{r.p95_latency_ms:>7.0f}ms{RESET} "
          f"{r.n_transcription_calls:>5} {r.n_lines:>5}\n")

        # Timing warnings
        if not r.timing_valid:
            w(f"  {' ' * 20} {RED}⚠ invalid timestamps{RESET}\n")
        if not r.timing_monotonic:
            w(f"  {' ' * 20} {YELLOW}⚠ non-monotonic timestamps{RESET}\n")

    w(f"  {'─' * 72}\n\n")

    # Translation per-sample table (only when a translation backend was run)
    if report.has_translation:
        w(f"  {BOLD}Translation{RESET}\n")
        w(f"  {'─' * 92}\n")
        w(f"  {BOLD}{'Sample':<20} {'MT-RTF':>7} {'1st-tr':>8} {'Provis?':>7} "
          f"{'MTcalls':>7} {'MT-time':>8} {'Accuracy':>9}{RESET}\n")
        w(f"  {'─' * 92}\n")
        for r in report.results:
            name = r.sample_name[:20]
            trtf = f"{r.translation_rtf:>5.2f}x" if r.translation_rtf is not None else "    -"
            trc = _rtf_color(r.translation_rtf) if r.translation_rtf is not None else ""
            ftt = (
                f"{r.first_translation_time_s:>6.2f}s"
                if r.first_translation_time_s is not None else "      -"
            )
            prov = "yes" if r.provisional_before_final else " no"
            calls = f"{r.mt_call_count}" if r.mt_call_count is not None else "-"
            mttime = (
                f"{r.translation_time_s:>6.2f}s"
                if r.translation_time_s is not None else "      -"
            )
            acc = (
                f"{r.translation_accuracy:>7.1f}" if r.translation_accuracy is not None else "      -"
            )
            w(f"  {name:<20} {trc}{trtf:>7}{RESET} {ftt:>8} {prov:>7} "
              f"{calls:>7} {mttime:>8} {acc:>9}\n")
            if r.translation_metric_name and r.translation_accuracy is not None:
                w(f"  {' ' * 20} {DIM}metric: {r.translation_metric_name}{RESET}\n")
        w(f"  {'─' * 92}\n\n")
    w(f"  {BOLD}Summary{RESET} ({report.n_samples} samples, "
      f"{report.total_audio_s:.1f}s total audio)\n\n")

    wc = _wer_color(report.avg_wer)
    rc = _rtf_color(report.overall_rtf)
    lc = _lat_color(report.avg_latency_ms)

    if report.has_wer:
        w(f"    Avg WER (macro):   {wc}{report.avg_wer * 100:>6.1f}%{RESET}\n")
        w(f"    Weighted WER:      {_wer_color(report.weighted_wer)}"
          f"{report.weighted_wer * 100:>6.1f}%{RESET}\n")
    else:
        w("    Avg WER (macro):       N/A (no applicable reference)\n")
        w("    Weighted WER:          N/A\n")
    w(f"    Overall RTF:       {rc}{report.overall_rtf:>6.3f}x{RESET}  "
      f"({report.total_processing_s:.1f}s for {report.total_audio_s:.1f}s audio)\n")
    w(f"    Avg latency:       {lc}{report.avg_latency_ms:>6.0f}ms{RESET}\n")
    w(f"    P95 latency:       {_lat_color(report.p95_latency_ms)}"
      f"{report.p95_latency_ms:>6.0f}ms{RESET}\n")

    if report.has_translation:
        w(f"\n  {BOLD}Translation Summary{RESET}\n")
        w(f"    Overall MT-RTF:     {_rtf_color(report.overall_translation_rtf)}"
          f"{report.overall_translation_rtf:>6.3f}x{RESET}  "
          f"({report.total_translation_time_s:.1f}s for {report.total_audio_s:.1f}s audio)\n")
        w(f"    Avg 1st-translation:"
          f" {report.avg_first_translation_time_s:>6.2f}s{RESET}\n")
        w(f"    Total MT calls:    {report.total_mt_calls}\n")
        w(f"    Provisional before final: {report.n_provisional_before_final}"
          f"/{report.n_samples}\n")
        if report.avg_translation_accuracy is not None:
            w(f"    Avg translation acc:"
              f" {report.avg_translation_accuracy:>6.1f}\n")


    # Per-language breakdown
    wer_by_lang = report.wer_by_language()
    if len(wer_by_lang) > 1:
        w(f"\n  {BOLD}By Language{RESET}\n")
        w(f"  {'─' * 40}\n")
        w(f"    {'Lang':>4}  {'WER':>7}  {'RTF':>6}  {'Samples':>7}\n")
        w(f"    {'─' * 34}\n")
        lang_groups = {}
        for r in report.results:
            lang_groups.setdefault(r.language, []).append(r)
        for lang in sorted(lang_groups):
            group = lang_groups[lang]
            avg_wer = sum(r.wer for r in group) / len(group)
            avg_rtf = sum(r.rtf for r in group) / len(group)
            wc = _wer_color(avg_wer)
            rc = _rtf_color(avg_rtf)
            w(f"    {lang:>4}  {wc}{avg_wer * 100:>6.1f}%{RESET}  "
              f"{rc}{avg_rtf:>5.2f}x{RESET}  {len(group):>7}\n")

    # Per-category breakdown
    wer_by_cat = report.wer_by_category()
    if len(wer_by_cat) > 1:
        w(f"\n  {BOLD}By Category{RESET}\n")
        w(f"  {'─' * 40}\n")
        w(f"    {'Category':>12}  {'WER':>7}  {'Samples':>7}\n")
        w(f"    {'─' * 30}\n")
        cat_groups = {}
        for r in report.results:
            cat_groups.setdefault(r.category, []).append(r)
        for cat in sorted(cat_groups):
            group = cat_groups[cat]
            avg_wer = sum(r.wer for r in group) / len(group)
            wc = _wer_color(avg_wer)
            w(f"    {cat:>12}  {wc}{avg_wer * 100:>6.1f}%{RESET}  {len(group):>7}\n")

    w(f"\n  {'─' * 72}\n\n")


def print_transcriptions(report: BenchmarkReport, out: TextIO = sys.stderr) -> None:
    """Print hypothesis vs reference for each sample."""
    w = out.write
    w(f"\n  {BOLD}Transcriptions{RESET}\n")
    w(f"  {'─' * 72}\n")
    for r in report.results:
        wc = _wer_color(r.wer)
        w(f"\n  {BOLD}{r.sample_name}{RESET} ({r.language}, {r.category}) "
          f"WER={wc}{r.wer * 100:.1f}%{RESET}\n")
        ref = r.reference[:120] + "..." if len(r.reference) > 120 else r.reference
        hyp = r.hypothesis[:120] + "..." if len(r.hypothesis) > 120 else r.hypothesis
        w(f"    {DIM}ref: {ref}{RESET}\n")
        w(f"    hyp: {hyp}\n")
        if r.hypothesis_translation:
            tr = r.hypothesis_translation[:120] + "..." if len(r.hypothesis_translation) > 120 else r.hypothesis_translation
            w(f"    {DIM}mt: {tr}{RESET}\n")
            if r.reference_translation:
                rt = r.reference_translation[:120] + "..." if len(r.reference_translation) > 120 else r.reference_translation
                w(f"    {DIM}ref-mt: {rt}{RESET}\n")
    w(f"\n  {'─' * 72}\n\n")


def write_json(report: BenchmarkReport, path: str) -> None:
    """Export the full report as JSON."""
    Path(path).write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
