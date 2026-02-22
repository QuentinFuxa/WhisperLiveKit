#!/usr/bin/env python3
"""
Comprehensive benchmark runner for WhisperLiveKit.

Tests all available backend+policy combinations across multiple audio files,
model sizes, and VAC on/off configurations. Outputs structured JSON that
is consumed by the report generator.

Usage:
    python run_benchmark.py                    # full benchmark
    python run_benchmark.py --quick            # subset (tiny models, fewer combos)
    python run_benchmark.py --json results.json # custom output path
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
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)

# Re-use harness functions
sys.path.insert(0, str(Path(__file__).parent))
from test_backend_offline import (
    AUDIO_TESTS_DIR,
    SAMPLE_RATE,
    TestResult,
    create_engine,
    discover_audio_files,
    download_sample_audio,
    load_audio,
    run_test,
)

CACHE_DIR = Path(__file__).parent / ".test_cache"


def get_system_info() -> dict:
    """Collect system metadata for the report."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    # macOS: get chip info
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        info["cpu"] = chip
    except Exception:
        info["cpu"] = platform.processor()

    # RAM
    try:
        mem_bytes = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
        info["ram_gb"] = round(mem_bytes / (1024**3))
    except Exception:
        info["ram_gb"] = None

    # Backend versions
    versions = {}
    try:
        import faster_whisper
        versions["faster-whisper"] = faster_whisper.__version__
    except ImportError:
        pass
    try:
        import mlx_whisper  # noqa: F401
        versions["mlx-whisper"] = "installed"
    except ImportError:
        pass
    try:
        import mlx.core as mx
        versions["mlx"] = mx.__version__
    except ImportError:
        pass
    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        pass

    info["backend_versions"] = versions
    return info


def detect_combos(quick: bool = False) -> list:
    """Build list of (backend, policy, model_size) combos to test."""
    combos = []

    # Model sizes to test
    model_sizes = ["tiny", "base", "small"] if not quick else ["tiny", "base"]

    # faster-whisper
    try:
        import faster_whisper  # noqa: F401
        for model in model_sizes:
            combos.append({"backend": "faster-whisper", "policy": "localagreement", "model": model})
            combos.append({"backend": "faster-whisper", "policy": "simulstreaming", "model": model})
    except ImportError:
        pass

    # mlx-whisper
    try:
        import mlx_whisper  # noqa: F401
        for model in model_sizes:
            combos.append({"backend": "mlx-whisper", "policy": "localagreement", "model": model})
            combos.append({"backend": "mlx-whisper", "policy": "simulstreaming", "model": model})
    except ImportError:
        pass

    # voxtral-mlx (single model, single policy)
    try:
        from whisperlivekit.voxtral_mlx import VoxtralMLXModel  # noqa: F401
        combos.append({"backend": "voxtral-mlx", "policy": "voxtral", "model": ""})
    except ImportError:
        pass

    # voxtral HF (single model, single policy)
    try:
        from transformers import AutoModelForSpeechSeq2Seq  # noqa: F401
        combos.append({"backend": "voxtral", "policy": "voxtral", "model": ""})
    except ImportError:
        pass

    return combos


def collect_audio_files() -> list:
    """Collect all benchmark audio files."""
    files = []

    # audio_tests/ directory
    if AUDIO_TESTS_DIR.is_dir():
        files.extend(discover_audio_files(str(AUDIO_TESTS_DIR)))

    # JFK sample
    jfk = CACHE_DIR / "jfk.wav"
    if not jfk.exists():
        jfk = download_sample_audio()
    if jfk.exists():
        files.append(jfk)

    return files


async def run_single_combo(
    combo: dict, audio_files: list, vac: bool, lan: str, max_duration: float,
) -> list:
    """Run one backend+policy+model combo across all audio files."""
    backend = combo["backend"]
    policy = combo["policy"]
    model = combo["model"]

    results = []
    try:
        engine = create_engine(
            backend=backend,
            model_size=model,
            lan=lan,
            vac=vac,
            policy=policy,
        )

        # Quiet noisy loggers
        for mod in (
            "whisperlivekit.audio_processor",
            "whisperlivekit.simul_whisper",
            "whisperlivekit.tokens_alignment",
            "whisperlivekit.simul_whisper.align_att_base",
            "whisperlivekit.simul_whisper.simul_whisper",
        ):
            logging.getLogger(mod).setLevel(logging.WARNING)

        for audio_path in audio_files:
            duration = len(load_audio(str(audio_path))) / SAMPLE_RATE
            if duration > max_duration:
                logger.info(f"  Skipping {audio_path.name} ({duration:.0f}s > {max_duration:.0f}s)")
                continue

            file_lan = lan
            if "french" in audio_path.name.lower() and lan == "en":
                file_lan = "fr"

            audio = load_audio(str(audio_path))
            result = await run_test(
                engine, audio, chunk_ms=100, realtime=False,
                audio_file=audio_path.name, backend=backend,
                policy=policy, lan=file_lan,
            )
            # Tag with extra metadata
            result_dict = asdict(result)
            result_dict["model_size"] = model
            result_dict["vac"] = vac
            results.append(result_dict)

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

    return results


async def run_full_benchmark(combos, audio_files, max_duration=60.0):
    """Run all combos with VAC on and off."""
    all_results = []
    total = len(combos) * 2  # x2 for VAC on/off
    idx = 0

    for combo in combos:
        for vac in [True, False]:
            idx += 1
            vac_str = "VAC=on" if vac else "VAC=off"
            desc = f"{combo['backend']} / {combo['policy']}"
            if combo["model"]:
                desc += f" / {combo['model']}"
            desc += f" / {vac_str}"

            print(f"\n{'='*70}")
            print(f"[{idx}/{total}] {desc}")
            print(f"{'='*70}")

            results = await run_single_combo(
                combo, audio_files, vac=vac, lan="en", max_duration=max_duration,
            )
            all_results.extend(results)

            # Free memory between combos
            gc.collect()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive WhisperLiveKit benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer models and combos")
    parser.add_argument("--json", default="benchmark_results.json", dest="json_output", help="Output JSON path")
    parser.add_argument("--max-duration", type=float, default=60.0, help="Max audio duration in seconds")
    args = parser.parse_args()

    system_info = get_system_info()
    combos = detect_combos(quick=args.quick)
    audio_files = collect_audio_files()

    print(f"System: {system_info.get('cpu', 'unknown')}, {system_info.get('ram_gb', '?')}GB RAM")
    print(f"Backends: {list(system_info['backend_versions'].keys())}")
    print(f"Combos to test: {len(combos)} x 2 (VAC on/off) = {len(combos)*2}")
    print(f"Audio files: {[f.name for f in audio_files]}")
    print()

    t0 = time.time()
    all_results = asyncio.run(
        run_full_benchmark(combos, audio_files, max_duration=args.max_duration)
    )
    total_time = time.time() - t0

    output = {
        "system_info": system_info,
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M"),
        "total_benchmark_time_s": round(total_time, 1),
        "n_combos": len(combos) * 2,
        "n_audio_files": len(audio_files),
        "results": all_results,
    }

    Path(args.json_output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nBenchmark complete in {total_time:.0f}s. Results: {args.json_output}")


if __name__ == "__main__":
    main()
