"""CLI entry point for WhisperLiveKit.

Provides subcommands:
  wlk serve       — Start the transcription server (default when no args)
  wlk listen      — Live microphone transcription
  wlk run         — Auto-pull model and start server
  wlk transcribe  — Transcribe audio files offline
  wlk bench       — Benchmark speed and accuracy on standard test audio
  wlk models      — List available and installed backends/models
  wlk pull        — Download a model for offline use
  wlk rm          — Delete downloaded models
  wlk check       — Verify system dependencies (ffmpeg, etc.)
  wlk diagnose    — Run pipeline diagnostics on audio file
"""

import importlib.util
import logging
import platform
import sys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _gpu_info() -> str:
    """Return a short string describing available accelerators."""
    parts = []
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            parts.append(f"CUDA ({name})")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            parts.append("MPS (Apple Silicon)")
    except ImportError:
        pass

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if _module_available("mlx"):
            parts.append("MLX")

    return ", ".join(parts) if parts else "CPU only"


BACKENDS = [
    {
        "id": "faster-whisper",
        "name": "Faster Whisper",
        "module": "faster_whisper",
        "install": "pip install faster-whisper",
        "description": "CTranslate2-based Whisper (fast, CPU/CUDA)",
        "policy": "localagreement",
    },
    {
        "id": "whisper",
        "name": "OpenAI Whisper",
        "module": "whisper",
        "install": "pip install openai-whisper",
        "description": "Original OpenAI Whisper (PyTorch)",
        "policy": "simulstreaming",
    },
    {
        "id": "mlx-whisper",
        "name": "MLX Whisper",
        "module": "mlx_whisper",
        "install": "pip install mlx-whisper",
        "description": "Apple Silicon native Whisper (MLX)",
        "policy": "localagreement",
        "platform": "darwin-arm64",
    },
    {
        "id": "voxtral-mlx",
        "name": "Voxtral MLX",
        "module": "mlx",
        "install": "pip install whisperlivekit[voxtral-mlx]",
        "description": "Mistral Voxtral Mini on Apple Silicon (MLX)",
        "platform": "darwin-arm64",
    },
    {
        "id": "voxtral",
        "name": "Voxtral HF",
        "module": "transformers",
        "install": "pip install whisperlivekit[voxtral-hf]",
        "description": "Mistral Voxtral Mini (HF Transformers, CUDA/CPU/MPS)",
    },
    {
        "id": "qwen3",
        "name": "Qwen3 ASR",
        "module": "qwen_asr",
        "install": "pip install qwen-asr",
        "description": "Qwen3-ASR with ForcedAligner timestamps",
    },
    {
        "id": "openai-api",
        "name": "OpenAI API",
        "module": "openai",
        "install": "pip install openai",
        "description": "Cloud-based transcription via OpenAI API",
    },
]


# ---------------------------------------------------------------------------
# Model catalog — maps "wlk pull <name>" to download actions
# ---------------------------------------------------------------------------

# Whisper model sizes available across backends
WHISPER_SIZES = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo",
]

# Faster-Whisper uses Systran HuggingFace repos
FASTER_WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "Systran/faster-distil-whisper-large-v3",
}

# MLX Whisper repos from model_mapping.py
MLX_WHISPER_REPOS = {
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v1": "mlx-community/whisper-large-v1-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large": "mlx-community/whisper-large-mlx",
}

# Voxtral/Qwen3 model repos
VOXTRAL_HF_REPO = "mistralai/Voxtral-Mini-4B-Realtime-2602"
VOXTRAL_MLX_REPO = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"
QWEN3_REPOS = {
    "1.7b": "Qwen/Qwen3-ASR-1.7B",
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
}
QWEN3_ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"


def _check_platform(backend: dict) -> bool:
    """Check if backend is compatible with current platform."""
    req = backend.get("platform")
    if req is None:
        return True
    if req == "darwin-arm64":
        return platform.system() == "Darwin" and platform.machine() == "arm64"
    return True


def _is_installed(backend: dict) -> bool:
    return _module_available(backend["module"])


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    import shutil
    return shutil.which("ffmpeg") is not None


def _scan_downloaded_models() -> dict:
    """Scan HuggingFace and Whisper caches to find downloaded models.

    Returns:
        dict mapping repo_id → cached path (or True if found).
    """
    found = {}

    # 1. Scan HuggingFace hub cache
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            found[repo.repo_id] = str(repo.repo_path)
    except Exception:
        pass

    # 2. Scan native Whisper cache (~/.cache/whisper)
    import os
    whisper_cache = os.path.join(os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")), "whisper")
    if os.path.isdir(whisper_cache):
        for f in os.listdir(whisper_cache):
            if f.endswith(".pt"):
                # e.g. "base.pt" or "large-v3.pt"
                size = f.rsplit(".", 1)[0]
                found[f"openai/whisper-{size}"] = os.path.join(whisper_cache, f)

    return found


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

def print_banner(config, host: str, port: int, ssl: bool = False):
    """Print a clean startup banner with server info."""
    protocol = "https" if ssl else "http"
    ws_protocol = "wss" if ssl else "ws"

    # Resolve display host
    display_host = host if host not in ("0.0.0.0", "::") else "localhost"
    base_url = f"{protocol}://{display_host}:{port}"
    ws_url = f"{ws_protocol}://{display_host}:{port}"

    backend = getattr(config, "backend", "auto")
    model = getattr(config, "model_size", "base")
    language = getattr(config, "lan", "auto")

    # Resolve actual backend name
    backend_label = backend
    if backend == "auto":
        backend_label = "auto (will resolve on first request)"

    lines = [
        "",
        "  WhisperLiveKit",
        f"  Backend: {backend_label} | Model: {model} | Language: {language}",
        f"  Accelerator: {_gpu_info()}",
        "",
        f"  Web UI:       {base_url}/",
        f"  WebSocket:    {ws_url}/asr",
        f"  Deepgram:     {ws_url}/v1/listen",
        f"  REST API:     {base_url}/v1/audio/transcriptions",
        f"  Models:       {base_url}/v1/models",
        f"  Health:       {base_url}/health",
        "",
    ]
    print("\n".join(lines), file=sys.stderr)


# ---------------------------------------------------------------------------
# `wlk models` subcommand
# ---------------------------------------------------------------------------

def cmd_models():
    """List available backends and their installation status."""
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

    print("\nAvailable backends:\n")

    max_name = max(len(b["name"]) for b in BACKENDS)

    for b in BACKENDS:
        compatible = _check_platform(b)
        installed = _is_installed(b)

        if installed:
            status = "\033[32m installed\033[0m"
        elif not compatible:
            status = "\033[90m n/a (wrong platform)\033[0m"
        else:
            status = "\033[33m not installed\033[0m"

        name_pad = b["name"].ljust(max_name)
        print(f"  {name_pad}  [{status}]  {b['description']}")

        if not installed and compatible:
            print(f"  {''.ljust(max_name)}  └─ {b['install']}")

    # System info
    print(f"\n  Platform:     {platform.system()} {platform.machine()}")
    print(f"  Python:       {platform.python_version()}")
    print(f"  Accelerator:  {_gpu_info()}")
    print(f"  ffmpeg:       {'found' if _check_ffmpeg() else 'NOT FOUND (required)'}")

    if is_apple_silicon:
        print("\n  Tip: On Apple Silicon, mlx-whisper and voxtral-mlx offer the best performance.")

    # Scan for downloaded models
    downloaded = _scan_downloaded_models()

    print("\n  Downloaded models:\n")
    found_any = False

    # Check Whisper-family models
    all_repos = {
        "faster-whisper": FASTER_WHISPER_REPOS,
        "mlx-whisper": MLX_WHISPER_REPOS,
    }
    for backend_name, repos in all_repos.items():
        for size, repo_id in repos.items():
            if repo_id in downloaded:
                found_any = True
                print(f"    \033[32m*\033[0m {backend_name}:{size}  ({repo_id})")

    # Check native whisper
    for size in WHISPER_SIZES:
        key = f"openai/whisper-{size}"
        if key in downloaded:
            found_any = True
            print(f"    \033[32m*\033[0m whisper:{size}")

    # Check voxtral / qwen3
    if VOXTRAL_HF_REPO in downloaded:
        found_any = True
        print(f"    \033[32m*\033[0m voxtral  ({VOXTRAL_HF_REPO})")
    if VOXTRAL_MLX_REPO in downloaded:
        found_any = True
        print(f"    \033[32m*\033[0m voxtral-mlx  ({VOXTRAL_MLX_REPO})")
    for qsize, repo_id in QWEN3_REPOS.items():
        if repo_id in downloaded:
            found_any = True
            print(f"    \033[32m*\033[0m qwen3:{qsize}  ({repo_id})")
    if QWEN3_ALIGNER_REPO in downloaded:
        found_any = True
        print(f"    \033[32m*\033[0m qwen3-aligner  ({QWEN3_ALIGNER_REPO})")

    if not found_any:
        print("    (none — models download automatically on first use, or use 'wlk pull')")

    # Show pullable models
    print("\n  Available models (use 'wlk pull <name>'):\n")
    print("    Whisper sizes: " + ", ".join(WHISPER_SIZES))
    print("    Voxtral:       voxtral, voxtral-mlx")
    print("    Qwen3:         qwen3:1.7b, qwen3:0.6b")
    print()
    print("  Examples:")
    print("    wlk pull base                     # Download for best available backend")
    print("    wlk pull faster-whisper:large-v3   # Specific backend + model")
    print("    wlk pull voxtral                   # Voxtral HF model")
    print("    wlk pull qwen3:1.7b                # Qwen3-ASR 1.7B")
    print()


# ---------------------------------------------------------------------------
# `wlk pull` subcommand
# ---------------------------------------------------------------------------

def _hf_download(repo_id: str, label: str):
    """Download a HuggingFace model repo to the local cache."""
    from huggingface_hub import snapshot_download
    print(f"  Downloading {label} ({repo_id})...")
    path = snapshot_download(repo_id)
    print(f"  Saved to: {path}")
    return path


def _resolve_pull_target(spec: str):
    """Parse a pull spec like 'faster-whisper:large-v3' or 'base' into (backend, size/repo).

    Returns: list of (backend_id, repo_id, label) tuples to download.
    """
    targets = []

    # Check for backend:size format
    if ":" in spec:
        backend_part, size_part = spec.split(":", 1)
    else:
        backend_part = None
        size_part = spec

    # Handle voxtral
    if size_part == "voxtral" or backend_part == "voxtral":
        targets.append(("voxtral", VOXTRAL_HF_REPO, "Voxtral Mini (HF)"))
        return targets

    if size_part == "voxtral-mlx" or backend_part == "voxtral-mlx":
        targets.append(("voxtral-mlx", VOXTRAL_MLX_REPO, "Voxtral Mini (MLX)"))
        return targets

    # Handle qwen3
    if backend_part == "qwen3" or size_part.startswith("qwen3"):
        qwen_size = size_part.split(":")[-1] if ":" in spec else "1.7b"
        if qwen_size.startswith("qwen3"):
            qwen_size = "1.7b"  # default
        repo = QWEN3_REPOS.get(qwen_size)
        if not repo:
            print(f"  Unknown Qwen3 size: {qwen_size}. Available: {', '.join(QWEN3_REPOS.keys())}")
            return []
        targets.append(("qwen3", repo, f"Qwen3-ASR {qwen_size}"))
        targets.append(("qwen3-aligner", QWEN3_ALIGNER_REPO, "Qwen3 ForcedAligner"))
        return targets

    # Handle whisper-family models with optional backend prefix
    if backend_part:
        # Specific backend requested
        if backend_part == "faster-whisper":
            repo = FASTER_WHISPER_REPOS.get(size_part)
            if not repo:
                print(f"  Unknown size: {size_part}. Available: {', '.join(FASTER_WHISPER_REPOS.keys())}")
                return []
            targets.append(("faster-whisper", repo, f"Faster Whisper {size_part}"))
        elif backend_part == "mlx-whisper":
            repo = MLX_WHISPER_REPOS.get(size_part)
            if not repo:
                print(f"  Unknown size: {size_part}. Available: {', '.join(MLX_WHISPER_REPOS.keys())}")
                return []
            targets.append(("mlx-whisper", repo, f"MLX Whisper {size_part}"))
        elif backend_part == "whisper":
            # OpenAI whisper downloads on first use; we can at least pull HF version
            targets.append(("whisper", f"openai/whisper-{size_part}", f"Whisper {size_part}"))
        else:
            print(f"  Unknown backend: {backend_part}")
            return []
    else:
        # No backend specified — download for the best available backend
        is_apple = platform.system() == "Darwin" and platform.machine() == "arm64"

        if size_part in WHISPER_SIZES:
            if is_apple and _module_available("mlx_whisper"):
                repo = MLX_WHISPER_REPOS.get(size_part)
                if repo:
                    targets.append(("mlx-whisper", repo, f"MLX Whisper {size_part}"))
            if _module_available("faster_whisper"):
                repo = FASTER_WHISPER_REPOS.get(size_part)
                if repo:
                    targets.append(("faster-whisper", repo, f"Faster Whisper {size_part}"))

            if not targets:
                # Fallback: download for any available backend
                repo = FASTER_WHISPER_REPOS.get(size_part)
                if repo:
                    targets.append(("faster-whisper", repo, f"Faster Whisper {size_part}"))
        else:
            print(f"  Unknown model: {spec}")
            print(f"  Available sizes: {', '.join(WHISPER_SIZES)}")
            print("  Other models: voxtral, voxtral-mlx, qwen3:1.7b, qwen3:0.6b")
            return []

    return targets


def cmd_pull(spec: str):
    """Download a model for offline use."""
    targets = _resolve_pull_target(spec)
    if not targets:
        return 1

    print(f"\n  Pulling model: {spec}\n")

    for backend_id, repo_id, label in targets:
        try:
            _hf_download(repo_id, label)
        except Exception as e:
            print(f"  Failed to download {label}: {e}")
            return 1

    print("\n  Done. Model ready for offline use.")
    print()
    return 0


# ---------------------------------------------------------------------------
# `wlk transcribe` subcommand
# ---------------------------------------------------------------------------

def cmd_transcribe(args: list):
    """Transcribe audio files using the full pipeline, no server needed.

    Usage: wlk transcribe [options] <audio_file> [audio_file ...]
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="wlk transcribe",
        description="Transcribe audio files offline using WhisperLiveKit.",
    )
    parser.add_argument("files", nargs="+", help="Audio files to transcribe")
    parser.add_argument("--backend", default="auto", help="ASR backend (default: auto)")
    parser.add_argument("--model", default="base", dest="model_size", help="Model size (default: base)")
    parser.add_argument("--language", "--lan", default="auto", dest="lan", help="Language code (default: auto)")
    parser.add_argument("--format", default="text", choices=["text", "json", "srt", "vtt", "verbose_json"],
                        help="Output format (default: text)")
    parser.add_argument("--output", "-o", default=None, help="Output file (default: stdout)")
    parser.add_argument("--diarization", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed processing logs")

    parsed = parser.parse_args(args)

    import asyncio

    # Suppress noisy logging unless --verbose.
    # Must happen AFTER importing (some modules set levels at import time)
    # so we use a wrapper that silences after import.
    if not parsed.verbose:
        asyncio.run(_transcribe_files_quiet(parsed))
    else:
        asyncio.run(_transcribe_files(parsed))


async def _transcribe_files_quiet(parsed):
    """Wrapper that silences logging after imports are done."""
    import warnings
    warnings.filterwarnings("ignore")

    # Force root logger to ERROR — overrides any per-module settings
    logging.root.setLevel(logging.ERROR)
    for handler in logging.root.handlers:
        handler.setLevel(logging.ERROR)
    # Silence all known noisy loggers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logging.getLogger(name).setLevel(logging.ERROR)

    await _transcribe_files(parsed)


async def _transcribe_files(parsed):
    """Run transcription on one or more audio files."""
    import json as json_module

    from whisperlivekit.test_harness import TestHarness, load_audio_pcm

    results = []

    for audio_path in parsed.files:
        print(f"  Transcribing: {audio_path}", file=sys.stderr)

        kwargs = {
            "model_size": parsed.model_size,
            "lan": parsed.lan,
            "pcm_input": True,
        }
        if parsed.backend != "auto":
            kwargs["backend"] = parsed.backend
        if parsed.diarization:
            kwargs["diarization"] = True

        async with TestHarness(**kwargs) as h:
            await h.feed(audio_path, speed=0)
            await h.drain(5.0)
            result = await h.finish(timeout=120)

        duration = len(load_audio_pcm(audio_path)) / (16000 * 2)

        if parsed.format == "text":
            results.append(result.committed_text or result.text)
        elif parsed.format == "json":
            results.append(json_module.dumps({"text": result.committed_text or result.text}))
        elif parsed.format == "verbose_json":
            results.append(json_module.dumps({
                "text": result.committed_text or result.text,
                "duration": round(duration, 2),
                "language": parsed.lan,
                "segments": [
                    {
                        "text": line.get("text", ""),
                        "start": line.get("start", "0:00:00"),
                        "end": line.get("end", "0:00:00"),
                        "speaker": line.get("speaker", 0),
                    }
                    for line in result.lines
                    if line.get("text") and line.get("speaker", 0) != -2
                ],
            }, indent=2))
        elif parsed.format in ("srt", "vtt"):
            results.append(_format_subtitle(result, parsed.format))

    # Output
    output_text = "\n".join(results)
    if parsed.output:
        with open(parsed.output, "w") as f:
            f.write(output_text)
        print(f"  Output written to: {parsed.output}", file=sys.stderr)
    else:
        print(output_text)


def _format_subtitle(result, fmt: str) -> str:
    """Format result as SRT or VTT subtitles."""
    from whisperlivekit.test_harness import _parse_time

    lines_out = []
    if fmt == "vtt":
        lines_out.append("WEBVTT\n")

    idx = 0
    for line in result.lines:
        if line.get("speaker") == -2 or not line.get("text"):
            continue
        idx += 1
        start = line.get("start", "0:00:00")
        end = line.get("end", "0:00:00")

        start_s = _parse_time(start)
        end_s = _parse_time(end)

        start_ts = _subtitle_timestamp(start_s, fmt)
        end_ts = _subtitle_timestamp(end_s, fmt)

        if fmt == "srt":
            lines_out.append(str(idx))
        lines_out.append(f"{start_ts} --> {end_ts}")
        lines_out.append(line["text"])
        lines_out.append("")

    return "\n".join(lines_out)


def _subtitle_timestamp(seconds: float, fmt: str) -> str:
    """Format seconds as SRT or VTT timestamp."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    sep = "," if fmt == "srt" else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


# ---------------------------------------------------------------------------
# `wlk bench` subcommand
# ---------------------------------------------------------------------------

def cmd_bench(args: list):
    """Benchmark the transcription pipeline on standard test audio.

    Usage: wlk bench [options]
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="wlk bench",
        description="Benchmark WhisperLiveKit on standard test audio.",
    )
    parser.add_argument("--backend", default="auto", help="ASR backend (default: auto)")
    parser.add_argument("--model", default="base", dest="model_size", help="Model size (default: base)")
    parser.add_argument("--language", "--lan", default="en", dest="lan", help="Language code (default: en)")
    parser.add_argument("--samples", default="all", help="Sample name or 'all' (default: all)")
    parser.add_argument("--json", default=None, dest="json_out", help="Export results to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed logs")

    parsed = parser.parse_args(args)

    import asyncio

    if not parsed.verbose:
        asyncio.run(_run_bench_quiet(parsed))
    else:
        asyncio.run(_run_bench(parsed))


async def _run_bench_quiet(parsed):
    """Run benchmark with suppressed logging."""
    import warnings
    warnings.filterwarnings("ignore")
    logging.root.setLevel(logging.ERROR)
    for handler in logging.root.handlers:
        handler.setLevel(logging.ERROR)
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logging.getLogger(name).setLevel(logging.ERROR)
    await _run_bench(parsed)


async def _run_bench(parsed):
    """Run the benchmark."""
    import json as json_module
    import time

    from whisperlivekit.metrics import compute_wer
    from whisperlivekit.test_data import get_sample, get_samples
    from whisperlivekit.test_harness import TestHarness

    # Determine samples to run
    if parsed.samples == "all":
        print("  Downloading test samples (first run only)...", file=sys.stderr)
        samples = get_samples()
        # Filter to matching language
        samples = [s for s in samples if s.language == parsed.lan]
        if not samples:
            # Fall back to all samples if none match the language
            samples = get_samples()
    else:
        samples = [get_sample(parsed.samples)]

    backend_label = parsed.backend
    if backend_label == "auto":
        backend_label = "auto-detect"

    print(file=sys.stderr)
    print("  WhisperLiveKit Benchmark", file=sys.stderr)
    print(f"  Backend: {backend_label} | Model: {parsed.model_size} | Language: {parsed.lan}", file=sys.stderr)
    print(f"  Samples: {len(samples)}", file=sys.stderr)
    print(f"  {'─' * 70}", file=sys.stderr)

    results = []

    kwargs = {
        "model_size": parsed.model_size,
        "lan": parsed.lan,
        "pcm_input": True,
    }
    if parsed.backend != "auto":
        kwargs["backend"] = parsed.backend

    for sample in samples:
        print(f"\n  {sample.name} ({sample.duration:.1f}s, {sample.language})", file=sys.stderr)

        t_start = time.perf_counter()

        async with TestHarness(**kwargs) as h:
            await h.feed(sample.path, speed=0)
            await h.drain(5.0)
            state = await h.finish(timeout=120)

        t_elapsed = time.perf_counter() - t_start
        rtf = t_elapsed / sample.duration if sample.duration > 0 else 0

        # Compute WER
        hypothesis = state.committed_text or state.text
        wer_result = compute_wer(sample.reference, hypothesis)

        n_lines = len(state.speech_lines)

        result_entry = {
            "sample": sample.name,
            "duration_s": round(sample.duration, 2),
            "processing_time_s": round(t_elapsed, 2),
            "rtf": round(rtf, 3),
            "wer": round(wer_result["wer"], 4),
            "wer_details": {
                "substitutions": wer_result["substitutions"],
                "insertions": wer_result["insertions"],
                "deletions": wer_result["deletions"],
                "ref_words": wer_result["ref_words"],
                "hyp_words": wer_result["hyp_words"],
            },
            "n_lines": n_lines,
            "transcription": hypothesis,
        }
        results.append(result_entry)

        # Print per-sample result
        wer_pct = wer_result["wer"] * 100
        wer_color = "\033[32m" if wer_pct < 15 else "\033[33m" if wer_pct < 30 else "\033[31m"
        rtf_color = "\033[32m" if rtf < 0.5 else "\033[33m" if rtf < 1.0 else "\033[31m"

        print(f"    WER:  {wer_color}{wer_pct:5.1f}%\033[0m  "
              f"(S:{wer_result['substitutions']} I:{wer_result['insertions']} D:{wer_result['deletions']})",
              file=sys.stderr)
        print(f"    RTF:  {rtf_color}{rtf:.3f}x\033[0m  "
              f"({t_elapsed:.1f}s for {sample.duration:.1f}s audio)",
              file=sys.stderr)
        print(f"    Lines: {n_lines}",
              file=sys.stderr)

    # Summary
    if len(results) > 1:
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_rtf = sum(r["rtf"] for r in results) / len(results)
        total_audio = sum(r["duration_s"] for r in results)
        total_proc = sum(r["processing_time_s"] for r in results)

        print(f"\n  {'─' * 70}", file=sys.stderr)
        print(f"  Summary ({len(results)} samples, {total_audio:.1f}s total audio)", file=sys.stderr)
        wer_color = "\033[32m" if avg_wer * 100 < 15 else "\033[33m" if avg_wer * 100 < 30 else "\033[31m"
        rtf_color = "\033[32m" if avg_rtf < 0.5 else "\033[33m" if avg_rtf < 1.0 else "\033[31m"
        print(f"    Avg WER:  {wer_color}{avg_wer * 100:5.1f}%\033[0m", file=sys.stderr)
        print(f"    Avg RTF:  {rtf_color}{avg_rtf:.3f}x\033[0m  "
              f"({total_proc:.1f}s for {total_audio:.1f}s audio)", file=sys.stderr)

    print(file=sys.stderr)

    # JSON export
    if parsed.json_out:
        export = {
            "backend": parsed.backend,
            "model_size": parsed.model_size,
            "language": parsed.lan,
            "accelerator": _gpu_info(),
            "results": results,
        }
        with open(parsed.json_out, "w") as f:
            json_module.dump(export, f, indent=2)
        print(f"  Results exported to: {parsed.json_out}", file=sys.stderr)


# ---------------------------------------------------------------------------
# `wlk listen` subcommand
# ---------------------------------------------------------------------------

def cmd_listen(args: list):
    """Live microphone transcription.

    Usage: wlk listen [options]
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="wlk listen",
        description="Transcribe live microphone input in real-time.",
    )
    parser.add_argument("--backend", default="auto", help="ASR backend (default: auto)")
    parser.add_argument("--model", default="base", dest="model_size", help="Model size (default: base)")
    parser.add_argument("--language", "--lan", default="auto", dest="lan", help="Language code (default: auto)")
    parser.add_argument("--diarization", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--output", "-o", default=None, help="Save transcription to file on exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed logs")

    parsed = parser.parse_args(args)

    try:
        import sounddevice  # noqa: F401
    except ImportError:
        print("\n  sounddevice is required for microphone input.", file=sys.stderr)
        print("  Install it with:  pip install sounddevice\n", file=sys.stderr)
        sys.exit(1)

    import asyncio

    if not parsed.verbose:
        asyncio.run(_listen_quiet(parsed))
    else:
        asyncio.run(_listen_main(parsed))


async def _listen_quiet(parsed):
    """Run listen with suppressed logging."""
    import warnings
    warnings.filterwarnings("ignore")
    logging.root.setLevel(logging.ERROR)
    for handler in logging.root.handlers:
        handler.setLevel(logging.ERROR)
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logging.getLogger(name).setLevel(logging.ERROR)
    await _listen_main(parsed)


async def _listen_main(parsed):
    """Live microphone transcription loop."""
    import numpy as np
    import sounddevice as sd

    from whisperlivekit.test_harness import TestHarness

    SAMPLE_RATE = 16000
    BLOCK_SIZE = int(SAMPLE_RATE * 0.5)  # 500ms chunks

    kwargs = {
        "model_size": parsed.model_size,
        "lan": parsed.lan,
        "pcm_input": True,
    }
    if parsed.backend != "auto":
        kwargs["backend"] = parsed.backend
    if parsed.diarization:
        kwargs["diarization"] = True

    out = sys.stderr

    out.write("\n  Loading model...")
    out.flush()

    async with TestHarness(**kwargs) as h:
        out.write(" done.\n")
        out.write("  Listening (Ctrl+C to stop)\n\n")
        out.flush()

        n_lines_printed = 0
        pipe_stdout = not sys.stdout.isatty()

        def on_state_update(state):
            nonlocal n_lines_printed
            speech = state.speech_lines
            buf = state.buffer_transcription.strip()

            # Clear the buffer line
            out.write("\r\033[K")

            # Print new committed lines
            while n_lines_printed < len(speech):
                text = speech[n_lines_printed].get("text", "")
                out.write(f"  {text}\n")
                if pipe_stdout:
                    sys.stdout.write(f"{text}\n")
                    sys.stdout.flush()
                n_lines_printed += 1

            # Show buffer (ephemeral, overwritten next update)
            if buf:
                out.write(f"  \033[90m| {buf}\033[0m")
            out.flush()

        h.on_update(on_state_update)

        # Bridge sounddevice thread -> async event loop
        import asyncio
        feed_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def audio_callback(indata, frames, time_info, status):
            pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            loop.call_soon_threadsafe(feed_queue.put_nowait, pcm)

        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=BLOCK_SIZE,
                callback=audio_callback,
            )
            stream.start()
        except Exception as e:
            out.write(f"\n  Could not open microphone: {e}\n")
            out.write("  Check that a microphone is connected and permissions are granted.\n\n")
            return

        try:
            while True:
                try:
                    pcm_data = await asyncio.wait_for(feed_queue.get(), timeout=0.1)
                    await h.feed_pcm(pcm_data, speed=0)
                except asyncio.TimeoutError:
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()

            out.write("\r\033[K\n  Finishing...\n")
            out.flush()

            result = await h.finish(timeout=30)

            # Print any remaining committed lines
            speech = result.speech_lines
            while n_lines_printed < len(speech):
                text = speech[n_lines_printed].get("text", "")
                out.write(f"  {text}\n")
                if pipe_stdout:
                    sys.stdout.write(f"{text}\n")
                    sys.stdout.flush()
                n_lines_printed += 1

            # Print remaining buffer
            buf = result.buffer_transcription.strip()
            if buf:
                out.write(f"  {buf}\n")
                if pipe_stdout:
                    sys.stdout.write(f"{buf}\n")
                    sys.stdout.flush()

            out.write("\n")
            out.flush()

            if parsed.output:
                with open(parsed.output, "w") as f:
                    f.write(result.text + "\n")
                out.write(f"  Saved to: {parsed.output}\n\n")
                out.flush()


# ---------------------------------------------------------------------------
# `wlk run` subcommand
# ---------------------------------------------------------------------------

def _resolve_run_spec(spec: str):
    """Map a model spec to (backend, model_size).

    Returns (backend_id_or_None, model_size_or_None).
    """
    if ":" in spec:
        backend_part, model_part = spec.split(":", 1)
        return backend_part, model_part

    backend_ids = {b["id"] for b in BACKENDS}
    if spec in backend_ids:
        return spec, None

    if spec == "voxtral-mlx":
        return "voxtral-mlx", None

    if spec in WHISPER_SIZES:
        return None, spec

    return None, spec


def cmd_run(args: list):
    """Auto-pull model if needed and start the server.

    Usage: wlk run [model] [server options]
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="wlk run",
        description="Download model (if needed) and start the transcription server.",
    )
    parser.add_argument("model", nargs="?", default=None,
                        help="Model spec (e.g., voxtral, large-v3, faster-whisper:base)")

    parsed, extra_args = parser.parse_known_args(args)

    backend_flag = None
    model_flag = None

    if parsed.model:
        backend_flag, model_flag = _resolve_run_spec(parsed.model)

        # Auto-pull if needed
        downloaded = _scan_downloaded_models()
        targets = _resolve_pull_target(parsed.model)
        need_pull = any(repo_id not in downloaded for _, repo_id, _ in targets)

        if need_pull and targets:
            print("\n  Model not found locally. Downloading...\n", file=sys.stderr)
            result = cmd_pull(parsed.model)
            if result != 0:
                sys.exit(1)
            print(file=sys.stderr)

    # Build server argv
    server_argv = [sys.argv[0]]
    if backend_flag:
        server_argv.extend(["--backend", backend_flag])
    if model_flag:
        server_argv.extend(["--model", model_flag])
    server_argv.extend(extra_args)

    sys.argv = server_argv
    from whisperlivekit.basic_server import main as serve_main
    serve_main()


# ---------------------------------------------------------------------------
# `wlk rm` subcommand
# ---------------------------------------------------------------------------

def cmd_rm(spec: str):
    """Delete a downloaded model from the cache."""
    targets = _resolve_pull_target(spec)
    if not targets:
        return 1

    downloaded = _scan_downloaded_models()
    found_any = any(repo_id in downloaded for _, repo_id, _ in targets)

    if not found_any:
        print(f"\n  Model '{spec}' is not downloaded.\n", file=sys.stderr)
        return 1

    print(file=sys.stderr)

    for _, repo_id, label in targets:
        if repo_id not in downloaded:
            continue

        try:
            # Try HuggingFace cache first
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir()
            deleted = False

            for repo in cache_info.repos:
                if repo.repo_id == repo_id:
                    size_bytes = repo.size_on_disk
                    size_str = f"{size_bytes / 1e9:.1f} GB" if size_bytes > 1e9 else f"{size_bytes / 1e6:.0f} MB"
                    hashes = [rev.commit_hash for rev in repo.revisions]
                    strategy = cache_info.delete_revisions(*hashes)
                    print(f"  Deleting {label} ({repo_id})...", file=sys.stderr)
                    strategy.execute()
                    print(f"  Freed {size_str}", file=sys.stderr)
                    deleted = True
                    break

            if not deleted:
                # Native whisper cache — plain file
                import os
                path = downloaded.get(repo_id)
                if path and os.path.isfile(path):
                    size = os.path.getsize(path)
                    size_str = f"{size / 1e6:.0f} MB"
                    os.remove(path)
                    print(f"  Deleted {label} ({path})", file=sys.stderr)
                    print(f"  Freed {size_str}", file=sys.stderr)

        except Exception as e:
            print(f"  Failed to delete {label}: {e}", file=sys.stderr)
            return 1

    print(file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# `wlk check` subcommand
# ---------------------------------------------------------------------------

def cmd_check():
    """Verify system dependencies."""
    print("\nSystem check:\n")

    checks = [
        ("Python >= 3.11", sys.version_info >= (3, 11)),
        ("ffmpeg", _check_ffmpeg()),
        ("torch", _module_available("torch")),
        ("torchaudio", _module_available("torchaudio")),
        ("faster-whisper", _module_available("faster_whisper")),
        ("uvicorn", _module_available("uvicorn")),
        ("fastapi", _module_available("fastapi")),
    ]

    all_ok = True
    for name, ok in checks:
        icon = "\033[32m OK\033[0m" if ok else "\033[31m MISSING\033[0m"
        print(f"  {icon}  {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  All dependencies OK. Ready to serve.")
    else:
        print("  Some dependencies are missing. Install them before running the server.")
    print()
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# `wlk diagnose` subcommand
# ---------------------------------------------------------------------------

def cmd_diagnose(args: list):
    """Run pipeline diagnostics on an audio file.

    Feeds audio through the full pipeline while probing internal backend state
    at regular intervals. Produces a timeline of what happened inside the
    pipeline, flags anomalies (stuck tokens, generate thread errors, etc.),
    and prints a pass/fail summary.

    Usage: wlk diagnose [audio_file] [options]
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="wlk diagnose",
        description="Run pipeline diagnostics to debug transcription issues.",
    )
    parser.add_argument("file", nargs="?", default=None,
                        help="Audio file to diagnose (default: built-in test sample)")
    parser.add_argument("--backend", default="auto", help="ASR backend (default: auto)")
    parser.add_argument("--model", default="base", dest="model_size", help="Model size (default: base)")
    parser.add_argument("--language", "--lan", default="auto", dest="lan", help="Language code (default: auto)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (1.0=realtime, 0=instant, default: 1.0)")
    parser.add_argument("--probe-interval", type=float, default=2.0,
                        help="Seconds between state probes (default: 2.0)")
    parser.add_argument("--diarization", action="store_true", help="Enable speaker diarization")

    parsed = parser.parse_args(args)

    import asyncio
    asyncio.run(_diagnose_main(parsed))


def _probe_backend_state(processor) -> dict:
    """Probe internal state of whatever ASR backend is running.

    Returns a dict of diagnostic key-value pairs specific to the backend.
    """
    info = {}
    transcription = processor.transcription
    if transcription is None:
        info["error"] = "no transcription processor"
        return info

    # Common: audio buffer size
    audio_buf = getattr(transcription, "audio_buffer", None)
    if audio_buf is not None:
        info["audio_buffer_samples"] = len(audio_buf)
        info["audio_buffer_sec"] = round(len(audio_buf) / 16000, 2)

    # Common: get_buffer result
    try:
        buf = transcription.get_buffer()
        info["buffer_text"] = buf.text if buf else ""
    except Exception as e:
        info["buffer_error"] = str(e)

    # Voxtral HF streaming specifics
    if hasattr(transcription, "_generate_started"):
        info["backend_type"] = "voxtral-hf-streaming"
        info["generate_started"] = transcription._generate_started
        info["generate_finished"] = transcription._generate_finished
        info["n_audio_tokens_fed"] = transcription._n_audio_tokens_fed
        info["n_text_tokens_received"] = transcription._n_text_tokens_received
        info["n_committed_words"] = transcription._n_committed_words
        info["pending_audio_samples"] = len(transcription._pending_audio)
        with transcription._text_lock:
            info["accumulated_text"] = transcription._accumulated_text
        if transcription._generate_error:
            info["generate_error"] = str(transcription._generate_error)
        # Audio queue depth
        info["audio_queue_depth"] = transcription._audio_queue.qsize()

    # Voxtral MLX specifics
    elif hasattr(transcription, "_mlx_processor"):
        info["backend_type"] = "voxtral-mlx"

    # SimulStreaming specifics
    elif hasattr(transcription, "prev_output"):
        info["backend_type"] = "simulstreaming"
        info["prev_output_len"] = len(getattr(transcription, "prev_output", "") or "")

    # LocalAgreement (OnlineASRProcessor) specifics
    elif hasattr(transcription, "hypothesis_buffer"):
        info["backend_type"] = "localagreement"
        hb = transcription.hypothesis_buffer
        if hasattr(hb, "committed"):
            info["committed_words"] = len(hb.committed)
        if hasattr(hb, "buffer"):
            info["hypothesis_buffer_words"] = len(hb.buffer)

    else:
        info["backend_type"] = "unknown"

    return info


def _probe_pipeline_state(processor) -> dict:
    """Probe pipeline-level state (queues, tasks, ffmpeg)."""
    info = {}
    if processor.transcription_queue:
        info["transcription_queue_size"] = processor.transcription_queue.qsize()
    if processor.diarization_queue:
        info["diarization_queue_size"] = processor.diarization_queue.qsize()
    if processor.translation_queue:
        info["translation_queue_size"] = processor.translation_queue.qsize()
    info["total_pcm_samples"] = processor.total_pcm_samples
    info["total_audio_sec"] = round(processor.total_pcm_samples / 16000, 2)
    info["is_stopping"] = processor.is_stopping
    info["in_silence"] = processor.current_silence is not None
    info["n_state_lines"] = len(processor.state.tokens)
    info["n_state_updates"] = len(getattr(processor.state, "new_tokens", []))
    return info


async def _diagnose_main(parsed):
    """Run the full diagnostic pipeline."""
    import asyncio
    import time as time_module

    from whisperlivekit.test_harness import TestHarness, load_audio_pcm

    out = sys.stderr

    # Resolve audio file
    audio_path = parsed.file
    if audio_path is None:
        try:
            from whisperlivekit.test_data import get_samples
            samples = get_samples()
            # Prefer a sample matching the requested language
            lang_match = [s for s in samples if s.language == parsed.lan]
            sample = lang_match[0] if lang_match else samples[0]
            audio_path = sample.path
            out.write(f"\n  Using test sample: {sample.name} ({sample.duration:.1f}s)\n")
        except Exception as e:
            out.write(f"\n  No audio file provided and couldn't load test sample: {e}\n")
            out.write("  Usage: wlk diagnose <audio_file> [options]\n\n")
            sys.exit(1)

    # Load audio
    try:
        pcm = load_audio_pcm(audio_path)
    except Exception as e:
        out.write(f"\n  Failed to load audio: {e}\n\n")
        sys.exit(1)

    audio_duration = len(pcm) / (16000 * 2)

    # Print header
    out.write(f"\n  {'━' * 70}\n")
    out.write("  WhisperLiveKit Pipeline Diagnostic\n")
    out.write(f"  {'━' * 70}\n\n")
    out.write(f"  Audio:        {audio_path}\n")
    out.write(f"  Duration:     {audio_duration:.1f}s\n")
    out.write(f"  Backend:      {parsed.backend}\n")
    out.write(f"  Model:        {parsed.model_size}\n")
    out.write(f"  Language:     {parsed.lan}\n")
    out.write(f"  Speed:        {parsed.speed}x\n")
    out.write(f"  Probe every:  {parsed.probe_interval}s\n")
    out.write(f"  Platform:     {platform.system()} {platform.machine()}\n")
    out.write(f"  Accelerator:  {_gpu_info()}\n")
    out.write(f"\n  {'─' * 70}\n")
    out.write("  Loading model...\n")
    out.flush()

    kwargs = {
        "model_size": parsed.model_size,
        "lan": parsed.lan,
        "pcm_input": True,
    }
    if parsed.backend != "auto":
        kwargs["backend"] = parsed.backend
    if parsed.diarization:
        kwargs["diarization"] = True

    t_load_start = time_module.perf_counter()

    probes = []
    anomalies = []

    async with TestHarness(**kwargs) as h:
        t_load = time_module.perf_counter() - t_load_start
        out.write(f"  Model loaded in {t_load:.1f}s\n")
        out.write(f"  {'─' * 70}\n")
        out.write("  Feeding audio...\n\n")
        out.flush()

        processor = h._processor
        chunk_duration = 0.5  # seconds per chunk
        chunk_bytes = int(chunk_duration * 16000 * 2)
        offset = 0
        t_start = time_module.perf_counter()
        last_probe = t_start
        probe_idx = 0

        # Feed audio with periodic probes
        while offset < len(pcm):
            end = min(offset + chunk_bytes, len(pcm))
            await processor.process_audio(pcm[offset:end])
            chunk_seconds = (end - offset) / (16000 * 2)
            h._audio_position += chunk_seconds
            offset = end

            if parsed.speed > 0:
                await asyncio.sleep(chunk_duration / parsed.speed)

            # Probe at intervals
            now = time_module.perf_counter()
            if now - last_probe >= parsed.probe_interval:
                probe_idx += 1
                elapsed = now - t_start
                audio_pos = h._audio_position

                backend_state = _probe_backend_state(processor)
                pipeline_state = _probe_pipeline_state(processor)
                harness_state = {
                    "n_history": len(h.history),
                    "state_text_len": len(h.state.text),
                    "state_lines": len(h.state.lines),
                    "state_speech_lines": len(h.state.speech_lines),
                    "buffer": h.state.buffer_transcription[:80] if h.state.buffer_transcription else "",
                }

                probe = {
                    "idx": probe_idx,
                    "wall_time": round(elapsed, 1),
                    "audio_pos": round(audio_pos, 1),
                    "backend": backend_state,
                    "pipeline": pipeline_state,
                    "harness": harness_state,
                }
                probes.append(probe)

                # Print probe
                out.write(f"  [{probe_idx:3d}] wall={elapsed:5.1f}s  audio={audio_pos:5.1f}s")

                bt = backend_state.get("backend_type", "?")
                if bt == "voxtral-hf-streaming":
                    out.write(
                        f"  | gen={'Y' if backend_state.get('generate_started') else 'N'}"
                        f" fin={'Y' if backend_state.get('generate_finished') else 'N'}"
                        f" audio_tok={backend_state.get('n_audio_tokens_fed', 0)}"
                        f" text_tok={backend_state.get('n_text_tokens_received', 0)}"
                        f" words={backend_state.get('n_committed_words', 0)}"
                        f" q={backend_state.get('audio_queue_depth', 0)}"
                    )
                    if backend_state.get("generate_error"):
                        out.write(f" \033[31mERROR: {backend_state['generate_error']}\033[0m")
                elif bt == "localagreement":
                    out.write(
                        f"  | committed={backend_state.get('committed_words', 0)}"
                        f" buf_words={backend_state.get('hypothesis_buffer_words', 0)}"
                    )
                elif bt == "simulstreaming":
                    out.write(
                        f"  | prev_out_len={backend_state.get('prev_output_len', 0)}"
                    )

                buf_text = backend_state.get("buffer_text", "")
                if buf_text:
                    display = buf_text[:50] + ("..." if len(buf_text) > 50 else "")
                    out.write(f'\n        buf="{display}"')

                out.write("\n")
                out.flush()

                # Anomaly detection
                if bt == "voxtral-hf-streaming":
                    if backend_state.get("generate_started") and not backend_state.get("generate_finished"):
                        if backend_state.get("n_audio_tokens_fed", 0) > 10 and backend_state.get("n_text_tokens_received", 0) == 0:
                            anomalies.append(f"[probe {probe_idx}] {backend_state['n_audio_tokens_fed']} audio tokens fed but 0 text tokens received — model may be stalled")
                    if backend_state.get("generate_error"):
                        anomalies.append(f"[probe {probe_idx}] Generate thread error: {backend_state['generate_error']}")

                if harness_state["n_history"] == 0 and elapsed > 5:
                    anomalies.append(f"[probe {probe_idx}] No state updates after {elapsed:.0f}s — pipeline may be stuck")

                last_probe = now

        # Done feeding — drain and finish
        out.write(f"\n  {'─' * 70}\n")
        out.write("  Audio feeding complete. Draining pipeline...\n")
        out.flush()

        await h.drain(3.0)

        # One more probe after drain
        backend_state = _probe_backend_state(processor)
        pipeline_state = _probe_pipeline_state(processor)
        probe_idx += 1
        elapsed = time_module.perf_counter() - t_start
        out.write(f"  [{probe_idx:3d}] wall={elapsed:5.1f}s  audio={h._audio_position:5.1f}s  (post-drain)\n")

        bt = backend_state.get("backend_type", "?")
        if bt == "voxtral-hf-streaming":
            out.write(
                f"        text_tok={backend_state.get('n_text_tokens_received', 0)}"
                f" words={backend_state.get('n_committed_words', 0)}"
                f" accumulated_text_len={len(backend_state.get('accumulated_text', ''))}\n"
            )

        result = await h.finish(timeout=60)
        t_total = time_module.perf_counter() - t_start

    # === Summary ===
    out.write(f"\n  {'━' * 70}\n")
    out.write("  Diagnostic Summary\n")
    out.write(f"  {'━' * 70}\n\n")

    out.write(f"  Wall time:        {t_total:.1f}s\n")
    out.write(f"  Audio duration:   {audio_duration:.1f}s\n")
    rtf = t_total / audio_duration if audio_duration > 0 else 0
    out.write(f"  RTF:              {rtf:.3f}x\n")
    out.write(f"  Model load:       {t_load:.1f}s\n")
    out.write(f"  Probes taken:     {probe_idx}\n\n")

    # Text output summary
    text = result.committed_text or result.text
    n_words = len(text.split()) if text.strip() else 0
    n_lines = len(result.speech_lines)
    has_silence = result.has_silence

    out.write(f"  Output words:     {n_words}\n")
    out.write(f"  Output lines:     {n_lines}\n")
    out.write(f"  Has silence:      {has_silence}\n")
    out.write(f"  Timing valid:     {result.timing_valid}\n")
    out.write(f"  Timing monotonic: {result.timing_monotonic}\n")

    timing_errors = result.timing_errors()
    if timing_errors:
        out.write("\n  Timing errors:\n")
        for err in timing_errors[:10]:
            out.write(f"    - {err}\n")

    # Transcription preview
    if text:
        preview = text[:200] + ("..." if len(text) > 200 else "")
        out.write(f'\n  Transcription:\n    "{preview}"\n')
    else:
        out.write("\n  \033[31mNo transcription output!\033[0m\n")

    # Anomalies
    out.write(f"\n  {'─' * 70}\n")
    if anomalies:
        out.write(f"  \033[33mAnomalies detected ({len(anomalies)}):\033[0m\n")
        for a in anomalies:
            out.write(f"    ⚠ {a}\n")
    else:
        out.write("  \033[32mNo anomalies detected.\033[0m\n")

    # Pass/fail checks
    out.write(f"\n  {'─' * 70}\n")
    out.write("  Health checks:\n\n")

    checks = [
        ("Model loaded successfully", t_load < 300),
        ("Audio processed without errors", not anomalies),
        ("Transcription produced output", n_words > 0),
        ("At least one committed line", n_lines > 0),
        ("Timestamps are valid", result.timing_valid),
        ("Timestamps are monotonic", result.timing_monotonic),
        ("RTF < 2.0x (faster than half real-time)", rtf < 2.0),
    ]

    all_pass = True
    for label, ok in checks:
        icon = "\033[32m PASS\033[0m" if ok else "\033[31m FAIL\033[0m"
        out.write(f"    {icon}  {label}\n")
        if not ok:
            all_pass = False

    out.write(f"\n  {'━' * 70}\n")
    if all_pass:
        out.write("  \033[32mAll checks passed.\033[0m\n")
    else:
        out.write("  \033[31mSome checks failed. Review the timeline above for details.\033[0m\n")
    out.write(f"  {'━' * 70}\n\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _print_version():
    """Print version."""
    from importlib.metadata import version
    try:
        v = version("whisperlivekit")
    except Exception:
        v = "dev"
    print(f"WhisperLiveKit {v}")


def _print_help():
    """Print top-level help."""
    print("""
WhisperLiveKit — Local speech-to-text toolkit

Usage: wlk <command> [options]

Commands:
  serve         Start the transcription server (default)
  listen        Live microphone transcription
  run           Auto-pull model and start server
  transcribe    Transcribe audio files offline
  bench         Benchmark speed and accuracy
  diagnose      Run pipeline diagnostics on audio
  models        List available backends and models
  pull          Download models for offline use
  rm            Delete downloaded models
  check         Verify system dependencies

Examples:
  wlk                                    # Start server with defaults
  wlk listen                             # Transcribe from microphone
  wlk listen --backend voxtral           # Listen with specific backend
  wlk run voxtral                        # Auto-pull + start server
  wlk run large-v3                       # Auto-pull + start server
  wlk transcribe audio.wav               # Transcribe a file
  wlk transcribe --format srt audio.wav  # Generate SRT subtitles
  wlk bench                              # Benchmark current backend
  wlk diagnose audio.wav --backend voxtral  # Diagnose pipeline issues
  wlk models                             # List backends + models
  wlk pull large-v3                      # Download model
  wlk rm large-v3                        # Delete downloaded model
  wlk check                              # Check dependencies

Run 'wlk <command> --help' for command-specific help.
""")


def main():
    """CLI entry point: routes to subcommands or defaults to 'serve'."""
    # Quick subcommand routing before argparse (so `wlk models` works
    # without loading the full server stack)
    if len(sys.argv) >= 2:
        subcmd = sys.argv[1]
        if subcmd == "models":
            cmd_models()
            return
        if subcmd == "check":
            sys.exit(cmd_check())
        if subcmd == "pull":
            if len(sys.argv) < 3:
                print("Usage: wlk pull <model>")
                print("  e.g.: wlk pull base, wlk pull faster-whisper:large-v3, wlk pull voxtral")
                sys.exit(1)
            sys.exit(cmd_pull(sys.argv[2]))
        if subcmd == "rm":
            if len(sys.argv) < 3:
                print("Usage: wlk rm <model>")
                print("  e.g.: wlk rm base, wlk rm voxtral")
                sys.exit(1)
            sys.exit(cmd_rm(sys.argv[2]))
        if subcmd == "transcribe":
            cmd_transcribe(sys.argv[2:])
            return
        if subcmd == "bench":
            cmd_bench(sys.argv[2:])
            return
        if subcmd == "listen":
            cmd_listen(sys.argv[2:])
            return
        if subcmd == "diagnose":
            cmd_diagnose(sys.argv[2:])
            return
        if subcmd == "run":
            cmd_run(sys.argv[2:])
            return
        if subcmd in ("-h", "--help", "help"):
            _print_help()
            return
        if subcmd in ("version", "--version", "-V"):
            _print_version()
            return
        if subcmd == "serve":
            # Strip "serve" and pass remaining args to the server
            sys.argv = [sys.argv[0]] + sys.argv[2:]

    # Default: serve
    from whisperlivekit.basic_server import main as serve_main
    serve_main()
