#!/usr/bin/env python3
"""Run benchmark across all backend x model x policy combos for scatter plot.

Tests each configuration on long audio samples in two modes:
  - Compute-unaware (speed=0): all audio dumped instantly, measures pure model accuracy
  - Compute-aware  (speed=1.0): real-time simulation, slow models lose audio

Usage:
    python scripts/run_scatter_benchmark.py
    python scripts/run_scatter_benchmark.py --aware          # only compute-aware
    python scripts/run_scatter_benchmark.py --unaware        # only compute-unaware
    python scripts/run_scatter_benchmark.py --plot-only results.json
"""

import argparse
import asyncio
import gc
import json
import logging
import platform
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
for name in [
    "whisperlivekit", "transformers", "torch", "httpx", "datasets",
    "numexpr", "faster_whisper",
]:
    logging.getLogger(name).setLevel(logging.ERROR)


LONG_SAMPLES_PATH = "~/.cache/whisperlivekit/benchmark_data/long_samples.json"

# ── All configurations to benchmark ──

COMBOS = [
    # faster-whisper x LocalAgreement
    {"backend": "faster-whisper", "model_size": "base", "policy": "localagreement",
     "label": "fw LA base", "color": "#4a9eff", "marker": "o", "size": 100},
    {"backend": "faster-whisper", "model_size": "small", "policy": "localagreement",
     "label": "fw LA small", "color": "#4a9eff", "marker": "o", "size": 220},
    # faster-whisper x SimulStreaming
    {"backend": "faster-whisper", "model_size": "base", "policy": "simulstreaming",
     "label": "fw SS base", "color": "#4a9eff", "marker": "s", "size": 100},
    {"backend": "faster-whisper", "model_size": "small", "policy": "simulstreaming",
     "label": "fw SS small", "color": "#4a9eff", "marker": "s", "size": 220},
    # mlx-whisper x LocalAgreement
    {"backend": "mlx-whisper", "model_size": "base", "policy": "localagreement",
     "label": "mlx LA base", "color": "#4ecca3", "marker": "o", "size": 100},
    {"backend": "mlx-whisper", "model_size": "small", "policy": "localagreement",
     "label": "mlx LA small", "color": "#4ecca3", "marker": "o", "size": 220},
    # mlx-whisper x SimulStreaming
    {"backend": "mlx-whisper", "model_size": "base", "policy": "simulstreaming",
     "label": "mlx SS base", "color": "#4ecca3", "marker": "s", "size": 100},
    {"backend": "mlx-whisper", "model_size": "small", "policy": "simulstreaming",
     "label": "mlx SS small", "color": "#4ecca3", "marker": "s", "size": 220},
    # voxtral-mlx (4B, native streaming)
    {"backend": "voxtral-mlx", "model_size": "", "policy": "",
     "label": "voxtral mlx", "color": "#f5a623", "marker": "D", "size": 250},
]


def is_backend_available(backend):
    try:
        if backend == "faster-whisper":
            import faster_whisper; return True  # noqa
        elif backend == "mlx-whisper":
            import mlx_whisper; return True  # noqa
        elif backend == "whisper":
            import whisper; return True  # noqa
        elif backend == "voxtral-mlx":
            import mlx.core  # noqa
            from whisperlivekit.voxtral_mlx.loader import load_voxtral_model; return True  # noqa
        elif backend == "voxtral":
            from transformers import VoxtralRealtimeForConditionalGeneration; return True  # noqa
        elif backend in ("qwen3", "qwen3-simul"):
            from whisperlivekit.qwen3_asr import _patch_transformers_compat
            _patch_transformers_compat()
            from qwen_asr import Qwen3ASRModel; return True  # noqa
    except (ImportError, Exception):
        pass
    return False


def get_system_info():
    info = {"platform": platform.platform(), "machine": platform.machine()}
    try:
        info["cpu"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        info["cpu"] = platform.processor()
    try:
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        info["ram_gb"] = round(mem / (1024**3))
    except Exception:
        info["ram_gb"] = None
    return info


async def run_combo_on_samples(combo, samples, lang="en", speed=0):
    """Run one config on all samples, return averaged result.

    Args:
        speed: 0 = compute-unaware (instant dump), 1.0 = compute-aware (real-time)
    """
    from whisperlivekit.core import TranscriptionEngine
    from whisperlivekit.metrics import compute_wer
    from whisperlivekit.test_harness import TestHarness, _engine_cache

    kwargs = {"lan": lang, "pcm_input": True}
    if combo["backend"]:
        kwargs["backend"] = combo["backend"]
    if combo["model_size"]:
        kwargs["model_size"] = combo["model_size"]
    if combo.get("policy"):
        kwargs["backend_policy"] = combo["policy"]

    TranscriptionEngine.reset()
    _engine_cache.clear()
    gc.collect()

    total_ref_words, total_errors = 0, 0
    total_infer_time, total_audio_time = 0.0, 0.0
    n_ok = 0

    for sample in samples:
        try:
            async with TestHarness(**kwargs) as h:
                await h.feed(sample["path"], speed=speed)
                await h.drain(max(5.0, sample["duration"] * 0.5))
                state = await h.finish(timeout=120)
                metrics = h.metrics

            hypothesis = state.committed_text or state.text
            wer_result = compute_wer(sample["reference"], hypothesis)

            total_ref_words += wer_result["ref_words"]
            total_errors += (wer_result["substitutions"] +
                             wer_result["insertions"] +
                             wer_result["deletions"])

            # Use actual inference time from metrics, not wall clock
            if metrics and metrics.transcription_durations:
                total_infer_time += sum(metrics.transcription_durations)
            total_audio_time += sample["duration"]
            n_ok += 1
        except Exception as e:
            print(f" [WARN: {sample['name']} failed: {e}]", end="")

    if n_ok == 0:
        return None

    weighted_wer = total_errors / max(total_ref_words, 1)
    # Real RTF = actual inference time / audio duration
    real_rtf = total_infer_time / total_audio_time if total_audio_time > 0 else 0

    return {
        "label": combo["label"],
        "backend": combo["backend"],
        "model_size": combo.get("model_size", ""),
        "policy": combo.get("policy", ""),
        "color": combo["color"],
        "marker": combo["marker"],
        "size": combo["size"],
        "rtf": round(real_rtf, 4),
        "wer_pct": round(weighted_wer * 100, 1),
        "n_samples": n_ok,
    }


async def run_all(combos, samples, lang="en", speed=0):
    mode_label = "compute-aware" if speed > 0 else "compute-unaware"
    results = []
    for i, combo in enumerate(combos):
        if not is_backend_available(combo["backend"]):
            print(f"  [{i+1}/{len(combos)}] SKIP {combo['label']} (not installed)")
            continue
        print(f"  [{i+1}/{len(combos)}] {combo['label']} ({mode_label})...", end="", flush=True)
        result = await run_combo_on_samples(combo, samples, lang, speed=speed)
        if result:
            results.append(result)
            print(f" RTF={result['rtf']:.2f}x WER={result['wer_pct']:.1f}% ({result['n_samples']} samples)")
        else:
            print(" FAILED (no results)")
    return results


def get_long_samples_for_lang(lang="en"):
    """Load long benchmark samples from long_samples.json, filtered by language."""
    import os
    path = os.path.expanduser(LONG_SAMPLES_PATH)
    if not os.path.exists(path):
        print(f"ERROR: Long samples file not found: {path}")
        print("Please generate it first (see benchmark_data/README).")
        sys.exit(1)
    with open(path) as f:
        all_samples = json.load(f)
    samples = [s for s in all_samples if s["language"] == lang]
    return [{"name": s["name"], "path": s["path"], "reference": s["reference"],
             "duration": s["duration"]} for s in samples]


LANG_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "pt": "Portuguese", "it": "Italian", "nl": "Dutch", "pl": "Polish",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ru": "Russian",
}


def generate_scatter(results, system_info, output_path, n_samples, lang="en",
                     mode="unaware", sample_duration=0.0):
    """Generate scatter plot.

    Args:
        mode: "unaware" or "aware" -- shown in title
        sample_duration: total audio duration in seconds -- shown in title
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("#fafafa")

    # Separate main cluster from outliers (RTF > 1.0)
    main = [r for r in results if r["rtf"] <= 1.0]
    slow = [r for r in results if r["rtf"] > 1.0]

    # Axis limits: tight around main data
    if main:
        xmax = max(r["rtf"] for r in main) * 1.6
        ymax = max(r["wer_pct"] for r in main) * 1.5 + 1
    else:
        xmax, ymax = 0.5, 10
    xmax = max(xmax, 0.45)
    ymax = max(ymax, 8)

    # Sweet spot zone
    sweet_x = xmax * 0.85
    sweet_y = ymax * 0.55
    rect = plt.Rectangle((0, 0), sweet_x, sweet_y, alpha=0.07, color="#4ecca3",
                          zorder=0, linewidth=0)
    ax.add_patch(rect)
    ax.text(sweet_x - 0.005, sweet_y - 0.15, "sweet spot", ha="right", va="top",
            fontsize=10, color="#2ecc71", fontstyle="italic", fontweight="bold", alpha=0.5)

    # Manual label offsets keyed by label name — hand-tuned
    OFFSETS = {
        "fw LA base":     (8, 8),
        "fw LA small":    (8, 8),
        "fw SS base":     (-55, -14),
        "fw SS small":    (8, 8),
        "mlx LA base":    (8, 10),
        "mlx LA small":   (8, 8),
        "mlx SS base":    (-55, 8),
        "mlx SS small":   (-55, -5),
        "voxtral mlx":    (10, -14),
        "qwen3 0.6B":     (10, 8),
        "fw LA large-v3": (8, -5),
        "fw SS large-v3": (8, 5),
    }

    # Plot main points
    for r in main:
        ax.scatter(r["rtf"], r["wer_pct"], c=r["color"], marker=r["marker"],
                   s=r["size"], edgecolors="white", linewidths=1.0, zorder=5, alpha=0.85)
        ox, oy = OFFSETS.get(r["label"], (8, -4))
        ax.annotate(r["label"], (r["rtf"], r["wer_pct"]),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=8.5, color="#333333", fontweight="medium")

    # Note slow backends outside main view
    if slow:
        lines = []
        for r in slow:
            lines.append(f"{r['label']}: RTF={r['rtf']:.1f}x, WER={r['wer_pct']:.1f}%")
        note = "Beyond real-time:\n" + "\n".join(lines)
        ax.text(xmax * 0.97, ymax * 0.97, note, ha="right", va="top",
                fontsize=7.5, color="#777777", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                          edgecolor="#dddddd", alpha=0.9))

    # Axes
    ax.set_xlim(left=-0.01, right=xmax)
    ax.set_ylim(bottom=0, top=ymax)
    ax.set_xlabel("RTF (lower = faster)", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("WER % (lower = more accurate)", fontsize=13, fontweight="bold", labelpad=8)
    ax.grid(True, alpha=0.15, linestyle="-", color="#cccccc")
    ax.tick_params(labelsize=10)

    # Title
    cpu = system_info.get("cpu", "unknown").replace("Apple ", "")
    lang_name = LANG_NAMES.get(lang, lang.upper())
    mode_label = "compute-unaware" if mode == "unaware" else "compute-aware"
    dur_str = f"{sample_duration / 60:.0f}min" if sample_duration >= 60 else f"{sample_duration:.0f}s"
    ax.set_title(
        f"Speed vs Accuracy ({mode_label}) — {n_samples} {lang_name} samples, {dur_str} ({cpu})",
        fontsize=14, fontweight="bold", pad=12)

    # Legend — backends
    backend_handles = []
    seen = set()
    for r in results:
        if r["backend"] not in seen:
            seen.add(r["backend"])
            backend_handles.append(mpatches.Patch(color=r["color"], label=r["backend"]))

    # Legend — shapes
    marker_map = {"o": "LocalAgreement", "s": "SimulStreaming", "D": "Native streaming",
                  "h": "Batch + aligner"}
    active = set(r["marker"] for r in results)
    shape_handles = [
        Line2D([0], [0], marker=m, color="#888", label=lbl,
               markerfacecolor="#888", markersize=8, linestyle="None")
        for m, lbl in marker_map.items() if m in active
    ]
    # sizes
    shape_handles += [
        Line2D([0], [0], marker="o", color="#888", label="base",
               markerfacecolor="#888", markersize=5, linestyle="None"),
        Line2D([0], [0], marker="o", color="#888", label="small / 4B",
               markerfacecolor="#888", markersize=9, linestyle="None"),
    ]

    leg1 = ax.legend(handles=backend_handles, loc="upper left", fontsize=9,
                     framealpha=0.95, edgecolor="#ddd", title="Backend", title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, loc="lower right", fontsize=8,
              framealpha=0.95, edgecolor="#ddd", ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", default=None)
    parser.add_argument("--lang", default="en", help="Language code (en, fr, es, de, ...)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path prefix (mode suffix added automatically)")
    parser.add_argument("--json-output", default=None,
                        help="JSON output path prefix (mode suffix added automatically)")
    parser.add_argument("--aware", action="store_true",
                        help="Run only compute-aware mode (speed=1.0)")
    parser.add_argument("--unaware", action="store_true",
                        help="Run only compute-unaware mode (speed=0)")
    args = parser.parse_args()

    lang = args.lang

    # Determine which modes to run
    if args.aware and args.unaware:
        modes = ["unaware", "aware"]
    elif args.aware:
        modes = ["aware"]
    elif args.unaware:
        modes = ["unaware"]
    else:
        # Default: run both
        modes = ["unaware", "aware"]

    if args.plot_only:
        data = json.load(open(args.plot_only))
        mode = data.get("mode", "unaware")
        output_path = args.output or f"benchmark_scatter_{lang}_{mode}.png"
        generate_scatter(data["results"], data["system_info"], output_path,
                         data["n_samples"], data.get("lang", "en"),
                         mode=mode,
                         sample_duration=data.get("total_audio_s", 0))
        return

    print(f"Loading long {lang} samples from {LONG_SAMPLES_PATH}...")
    samples = get_long_samples_for_lang(lang)
    if not samples:
        print(f"ERROR: No long samples for language '{lang}'")
        sys.exit(1)
    print(f"Using {len(samples)} samples: {[s['name'] for s in samples]}")
    total_dur = sum(s["duration"] for s in samples)
    print(f"Total audio: {total_dur:.0f}s ({total_dur / 60:.1f}min)\n")

    # Filter combos to backends that support this language
    from whisperlivekit.benchmark.compat import backend_supports_language
    combos = [c for c in COMBOS if backend_supports_language(c["backend"], lang)]

    system_info = get_system_info()

    for mode in modes:
        speed = 1.0 if mode == "aware" else 0
        mode_label = "compute-aware" if mode == "aware" else "compute-unaware"
        print(f"\n{'='*60}")
        print(f" Running {mode_label} (speed={speed})")
        print(f"{'='*60}\n")

        t0 = time.time()
        results = asyncio.run(run_all(combos, samples, lang, speed=speed))
        total = time.time() - t0

        # Save JSON
        json_path = args.json_output or f"/tmp/bench_scatter_{lang}"
        json_file = f"{json_path}_{mode}.json"
        output_data = {
            "system_info": system_info,
            "lang": lang,
            "mode": mode,
            "speed": speed,
            "n_samples": len(samples),
            "sample_names": [s["name"] for s in samples],
            "total_audio_s": round(total_dur, 1),
            "total_benchmark_time_s": round(total, 1),
            "results": results,
        }
        with open(json_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nJSON: {json_file} ({total:.0f}s total)")

        # Generate scatter plot
        output_base = args.output or f"benchmark_scatter_{lang}"
        output_path = f"{output_base}_{mode}.png"
        generate_scatter(results, system_info, output_path, len(samples), lang,
                         mode=mode, sample_duration=total_dur)


if __name__ == "__main__":
    main()
