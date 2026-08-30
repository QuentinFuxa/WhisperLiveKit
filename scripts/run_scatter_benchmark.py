#!/usr/bin/env python3
"""Compare backends using the same per-sample report as `wlk bench`.

`--aware`, `--unaware`, `--lang`, `--output`, `--json-output` and `--plot-only`
remain available. New measurements and figures default to benchmarks/runs/.
Each configuration runs in its own process, with a separate startup sample.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

COMBOS_DARWIN = [
    # faster-whisper x LocalAgreement
    {"backend": "faster-whisper", "model_size": "base", "policy": "localagreement",
     "label": "fw LA base", "color": "#4a9eff", "marker": "o"},
    {"backend": "faster-whisper", "model_size": "small", "policy": "localagreement",
     "label": "fw LA small", "color": "#4a9eff", "marker": "o"},
    # faster-whisper x SimulStreaming
    {"backend": "faster-whisper", "model_size": "base", "policy": "simulstreaming",
     "label": "fw SS base", "color": "#4a9eff", "marker": "s"},
    {"backend": "faster-whisper", "model_size": "small", "policy": "simulstreaming",
     "label": "fw SS small", "color": "#4a9eff", "marker": "s"},
    # mlx-whisper x LocalAgreement
    {"backend": "mlx-whisper", "model_size": "base", "policy": "localagreement",
     "label": "mlx LA base", "color": "#4ecca3", "marker": "o"},
    {"backend": "mlx-whisper", "model_size": "small", "policy": "localagreement",
     "label": "mlx LA small", "color": "#4ecca3", "marker": "o"},
    # mlx-whisper x SimulStreaming
    {"backend": "mlx-whisper", "model_size": "base", "policy": "simulstreaming",
     "label": "mlx SS base", "color": "#4ecca3", "marker": "s"},
    {"backend": "mlx-whisper", "model_size": "small", "policy": "simulstreaming",
     "label": "mlx SS small", "color": "#4ecca3", "marker": "s"},
    # voxtral-mlx (4B, native streaming)
    {"backend": "voxtral-mlx", "model_size": "", "policy": "",
     "label": "voxtral mlx", "color": "#f5a623", "marker": "D"},
]

COMBOS_CUDA = [
    # faster-whisper x LocalAgreement
    {"backend": "faster-whisper", "model_size": "base", "policy": "localagreement",
     "label": "fw LA base", "color": "#4a9eff", "marker": "o"},
    {"backend": "faster-whisper", "model_size": "small", "policy": "localagreement",
     "label": "fw LA small", "color": "#4a9eff", "marker": "o"},
    {"backend": "faster-whisper", "model_size": "large-v3-turbo", "policy": "localagreement",
     "label": "fw LA turbo", "color": "#4a9eff", "marker": "o"},
    # faster-whisper x SimulStreaming
    {"backend": "faster-whisper", "model_size": "base", "policy": "simulstreaming",
     "label": "fw SS base", "color": "#4a9eff", "marker": "s"},
    {"backend": "faster-whisper", "model_size": "small", "policy": "simulstreaming",
     "label": "fw SS small", "color": "#4a9eff", "marker": "s"},
    {"backend": "faster-whisper", "model_size": "large-v3-turbo", "policy": "simulstreaming",
     "label": "fw SS turbo", "color": "#4a9eff", "marker": "s"},
    # qwen3-streaming (0.6B): causal tower (append-only, English-only) and
    # windowed re-compute (multilingual)
    {"backend": "qwen3-streaming", "model_size": "0.6b", "policy": "",
     "label": "qwen3 causal", "color": "#b06ee8", "marker": "D",
     "kwargs": {"qwen3_streaming_audio_backend": "causal"}, "languages": ["en"]},
    {"backend": "qwen3-streaming", "model_size": "0.6b", "policy": "",
     "label": "qwen3 windowed", "color": "#8e44ad", "marker": "D",
     "kwargs": {"qwen3_streaming_audio_backend": "windowed"}},
]

COMBOS = COMBOS_DARWIN if sys.platform == "darwin" else COMBOS_CUDA



def get_long_samples_for_lang(lang):
    path = Path("~/.cache/whisperlivekit/benchmark_data/long_samples.json").expanduser()
    rows = json.loads(path.read_text())
    return [row for row in rows if row["language"] == lang]


async def run_combo_on_samples(combo, samples, lang="en", speed=0, repeats=3):
    from whisperlivekit.benchmark.datasets import BenchmarkSample
    from whisperlivekit.benchmark.runner import BenchmarkRunner

    kwargs = dict(combo.get("kwargs") or {})
    if combo.get("policy"):
        kwargs["backend_policy"] = combo["policy"]
    selected = [BenchmarkSample(
        name=s["name"], path=s["path"], reference=s["reference"], duration=s["duration"],
        language=lang, category=s.get("category", "legacy"), source=s.get("source", "legacy manifest"),
        expected_sha256=s.get("audio_sha256", ""),
    ) for s in samples]
    report = await BenchmarkRunner(
        backend=combo["backend"], model_size=combo.get("model_size", "base"),
        speed=speed, samples=selected, repeats=repeats, warmup=True, engine_kwargs=kwargs,
    ).run()
    return {"label": combo["label"], "color": combo.get("color"),
            "marker": combo.get("marker", "o"), "report": report.to_dict()}


def generate_scatter(data, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_path)
    published = Path(__file__).resolve().parents[1]
    if destination.exists() and (
        destination.resolve().parent == published
        or (published / "benchmarks/archive") in destination.resolve().parents
    ):
        raise ValueError("Choose a new output path; published root figures are preserved")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    legacy = "comparison_version" not in data
    for index, entry in enumerate(data["results"]):
        if legacy:
            rtf, score = entry["rtf"], entry["wer_pct"]
            label = entry["label"] + " (historical aggregate)"
        else:
            report = entry["report"]
            summary = report["summary"]
            rtf = summary["overall_rtf"]
            quality = summary["weighted_cer"] if data.get("lang") == "zh" else summary["weighted_wer"]
            score = quality * 100 if quality is not None else None
            label = f"{entry['label']}: {summary['n_successful']}/{summary['n_samples']} completed"
            if summary["n_failed"] or summary["n_skipped"]:
                label += " (incomplete)"
        if rtf is None or score is None:
            ax.plot([], [], "x", label=label + "; no score")
            continue
        ax.scatter(rtf, score, s=65, marker=entry.get("marker", "o"),
                   color=entry.get("color") or f"C{index % 10}", label=label)
    ax.set(xlabel="ASR inference time / audio duration", ylabel="CER (%)" if data.get("lang") == "zh" else "WER (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.2)
    ax.set_title(f"{data.get('lang', 'en')} · feed speed {data.get('speed', 'unknown')} · "
                 + ("historical aggregates" if legacy else "diagnostic comparison"))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), fontsize=8)
    fig.text(.02, .01, "ASR compute ratio is not end-to-end latency. Inspect sample identities and failures in the JSON.", fontsize=8)
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--output", "-o", help="Figure prefix (mode suffix added)")
    parser.add_argument("--json-output", help="JSON prefix (mode suffix added)")
    parser.add_argument("--aware", action="store_true", help="Paced audio, speed=1")
    parser.add_argument("--unaware", action="store_true", help="Immediate audio, speed=0")
    parser.add_argument("--manifest", help="Fixed corpus manifest; defaults to existing long_samples.json")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--combos", help="JSON array of configurations (backend, model_size, policy, label, kwargs)")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        request = json.loads(Path(args.worker).read_text())
        result = asyncio.run(run_combo_on_samples(**request["run"]))
        Path(request["output"]).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return
    output_dir = Path("benchmarks/runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        generate_scatter(json.loads(Path(args.plot_only).read_text()), args.output or output_dir/"scatter.png")
        return
    if args.manifest:
        from whisperlivekit.benchmark.datasets import load_manifest
        samples = [dict(name=s.name, path=s.path, reference=s.reference, duration=s.duration,
                        language=s.language, category=s.category, source=s.source, audio_sha256=s.expected_sha256)
                   for s in load_manifest(args.manifest, continuous=args.continuous) if s.language == args.lang]
    else:
        samples = get_long_samples_for_lang(args.lang)
    if not samples:
        parser.error("No samples for the selected language")
    from whisperlivekit.benchmark.metrics import BenchmarkReport, SampleResult, get_system_info
    combos = json.loads(Path(args.combos).read_text()) if args.combos else COMBOS
    modes = ["aware"] if args.aware and not args.unaware else ["unaware"] if args.unaware and not args.aware else ["unaware", "aware"]
    failed = False
    for mode in modes:
        data = {"comparison_version": 1, "lang": args.lang, "mode": mode,
                "speed": 1 if mode == "aware" else 0, "system_info": get_system_info(), "results": []}
        if args.manifest:
            import hashlib
            data["corpus_sha256"] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
        if not combos:
            parser.error("No backend configurations supplied")
        for i, combo in enumerate(combos):
            output = output_dir/f"{args.lang}_{mode}_{i}.json"
            print(f"{combo['label']}: {len(samples)} samples, {args.repeats} passes, speed={data['speed']}", flush=True)
            with tempfile.TemporaryDirectory(prefix="wlk-benchmark-") as tmp:
                request = Path(tmp)/"request.json"
                request.write_text(json.dumps({"output": str(output.resolve()), "run": {
                    "combo": combo, "samples": samples, "lang": args.lang, "speed": data["speed"], "repeats": args.repeats}}))
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(request)])
            if output.exists():
                entry = json.loads(output.read_text())
            else:
                report = BenchmarkReport(combo["backend"], combo.get("model_size", ""), feed_speed=data["speed"])
                report.results = [SampleResult(s["name"], args.lang, s.get("category", "legacy"), s["duration"],
                    status="error", error=f"Worker exited with code {process.returncode}", reference=s["reference"])
                    for s in samples]
                entry = {"label": combo["label"], "report": report.to_dict()}
            summary = entry["report"]["summary"]
            entry["report"]["corpus_sha256"] = data.get("corpus_sha256", "")
            output.write_text(json.dumps(entry, indent=2, ensure_ascii=False, allow_nan=False))
            failed |= process.returncode != 0 or summary["n_failed"] > 0 or summary["n_warmup_failed"] > 0 or summary["n_successful"] == 0
            data["results"].append(entry)
            json_path = Path(f"{args.json_output}_{mode}.json") if args.json_output else output_dir/f"{args.lang}_{mode}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
        generate_scatter(data, f"{args.output}_{mode}.png" if args.output else output_dir/f"{args.lang}_{mode}.png")
        print(f"Report: {json_path}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
