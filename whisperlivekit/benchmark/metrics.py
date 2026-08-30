"""Benchmark records. Failed and skipped samples remain in the report."""

import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SampleResult:
    sample_name: str
    language: str
    category: str
    duration_s: float
    status: str = "ok"
    error: str = ""
    wer: Optional[float] = None
    wer_details: Dict[str, int] = field(default_factory=dict)
    # Successful ASR calls only; excludes model loading and audio pacing.
    processing_time_s: Optional[float] = None
    rtf: Optional[float] = None
    wall_time_s: Optional[float] = None
    startup_time_s: Optional[float] = None
    first_text_time_s: Optional[float] = None
    # These are inference-call durations, not audio-to-text latencies.
    avg_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    n_transcription_calls: int = 0
    n_lines: int = 0
    n_tokens: int = 0
    timing_valid: bool = True
    timing_monotonic: bool = True
    hypothesis: str = ""
    reference: str = ""
    source: str = ""
    tags: List[str] = field(default_factory=list)
    audio_sha256: str = ""
    effective_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["sample"] = result.pop("sample_name")
        result["asr_call_mean_ms"] = result.pop("avg_latency_ms")
        result["asr_call_p95_ms"] = result.pop("p95_latency_ms")
        return result


@dataclass
class BenchmarkReport:
    backend: str
    model_size: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    system_info: Dict[str, Any] = field(default_factory=dict)
    results: List[SampleResult] = field(default_factory=list)
    feed_speed: float = 0

    @property
    def successful_results(self) -> List[SampleResult]:
        return [r for r in self.results if r.status == "ok"]

    @property
    def n_samples(self) -> int:
        return len(self.results)

    @property
    def n_failed(self) -> int:
        return sum(r.status in ("error", "timeout") for r in self.results)

    @property
    def total_audio_s(self) -> float:
        return sum(r.duration_s for r in self.successful_results)

    @property
    def total_processing_s(self) -> float:
        return sum(r.processing_time_s or 0 for r in self.successful_results)

    @property
    def avg_wer(self) -> Optional[float]:
        values = [r.wer for r in self.successful_results if r.wer is not None]
        return sum(values) / len(values) if values else None

    @property
    def weighted_wer(self) -> Optional[float]:
        scored = [r for r in self.successful_results if r.wer is not None]
        errors = sum(sum(r.wer_details.get(k, 0) for k in
                         ("substitutions", "insertions", "deletions")) for r in scored)
        words = sum(r.wer_details.get("ref_words", 0) for r in scored)
        return errors / words if words else None

    @property
    def overall_rtf(self) -> Optional[float]:
        return self.total_processing_s / self.total_audio_s if self.total_audio_s else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_version": "2.0",
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "config": {"backend": self.backend, "model_size": self.model_size,
                       "feed_speed": self.feed_speed},
            "measurement": {
                "rtf": "successful ASR call time / audio duration",
                "wall_time_s": "audio feed and EOF drain, excluding startup and cleanup",
                "first_text_time_s": "first committed text since feed start; paced runs only",
                "asr_call_p95_ms": "last 4096 inference calls per sample, not word latency",
                "memory": "not measured",
                "model_revisions": "not resolved; pin model artifacts for published comparisons",
            },
            "summary": {
                "n_samples": self.n_samples,
                "n_successful": len(self.successful_results),
                "n_failed": self.n_failed,
                "n_skipped": sum(r.status == "skipped" for r in self.results),
                "total_audio_s": self.total_audio_s,
                "total_processing_s": self.total_processing_s,
                "avg_wer": self.avg_wer,
                "weighted_wer": self.weighted_wer,
                "overall_rtf": self.overall_rtf,
            },
            "results": [r.to_dict() for r in self.results],
        }


def get_system_info() -> Dict[str, Any]:
    """Collect system metadata for the benchmark report."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }
    from importlib.metadata import PackageNotFoundError, version
    from pathlib import Path

    try:
        info["whisperlivekit_version"] = version("whisperlivekit")
    except PackageNotFoundError:
        info["whisperlivekit_version"] = None
    root = Path(__file__).resolve().parents[2]
    if (root / ".git").exists():
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        ).strip()
        info["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True,
        ).strip())

    # CPU info
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
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
        try:
            import os
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            info["ram_gb"] = round(pages * page_size / (1024**3))
        except Exception:
            info["ram_gb"] = None

    # Accelerator
    try:
        import torch
        if torch.cuda.is_available():
            info["accelerator"] = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["accelerator"] = "Apple Silicon (MPS)"
        else:
            info["accelerator"] = "CPU"
    except ImportError:
        info["accelerator"] = "CPU"

    # Backend versions
    versions = {}
    for pkg, name in [
        ("faster_whisper", "faster-whisper"),
        ("whisper", "openai-whisper"),
        ("mlx_whisper", "mlx-whisper"),
        ("transformers", "transformers"),
        ("torch", "torch"),
    ]:
        try:
            mod = __import__(pkg)
            versions[name] = getattr(mod, "__version__", "installed")
        except ImportError:
            pass
    try:
        import mlx.core as mx
        versions["mlx"] = mx.__version__
    except ImportError:
        pass

    info["backend_versions"] = versions
    return info
