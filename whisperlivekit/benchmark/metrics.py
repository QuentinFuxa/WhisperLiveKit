"""Benchmark result data structures and aggregation."""

import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SampleResult:
    """Result from benchmarking one audio sample."""

    sample_name: str
    language: str
    category: str
    duration_s: float

    # Quality
    wer: float
    wer_details: Dict[str, int]

    # Speed
    processing_time_s: float
    rtf: float

    # Latency (from SessionMetrics)
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    n_transcription_calls: int = 0

    # Pipeline stats
    n_lines: int = 0
    n_tokens: int = 0

    # Timing quality
    timing_valid: bool = True
    timing_monotonic: bool = True

    # Memory
    peak_memory_mb: Optional[float] = None

    # Texts
    hypothesis: str = ""
    reference: str = ""

    # Source
    source: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample": self.sample_name,
            "language": self.language,
            "category": self.category,
            "duration_s": round(self.duration_s, 2),
            "wer": round(self.wer, 4),
            "wer_details": self.wer_details,
            "processing_time_s": round(self.processing_time_s, 2),
            "rtf": round(self.rtf, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "n_transcription_calls": self.n_transcription_calls,
            "n_lines": self.n_lines,
            "n_tokens": self.n_tokens,
            "timing_valid": self.timing_valid,
            "timing_monotonic": self.timing_monotonic,
            "peak_memory_mb": round(self.peak_memory_mb, 1) if self.peak_memory_mb else None,
            "hypothesis": self.hypothesis,
            "reference": self.reference,
            "source": self.source,
            "tags": self.tags,
        }


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report with system info and per-sample results."""

    backend: str
    model_size: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    system_info: Dict[str, Any] = field(default_factory=dict)
    results: List[SampleResult] = field(default_factory=list)

    # --- Aggregate properties ---

    @property
    def n_samples(self) -> int:
        return len(self.results)

    @property
    def total_audio_s(self) -> float:
        return sum(r.duration_s for r in self.results)

    @property
    def total_processing_s(self) -> float:
        return sum(r.processing_time_s for r in self.results)

    @property
    def avg_wer(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.wer for r in self.results) / len(self.results)

    @property
    def weighted_wer(self) -> float:
        """Micro-averaged WER: total errors / total reference words."""
        total_errors = sum(
            r.wer_details.get("substitutions", 0) +
            r.wer_details.get("insertions", 0) +
            r.wer_details.get("deletions", 0)
            for r in self.results
        )
        total_ref = sum(r.wer_details.get("ref_words", 0) for r in self.results)
        return total_errors / max(total_ref, 1)

    @property
    def avg_rtf(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.rtf for r in self.results) / len(self.results)

    @property
    def overall_rtf(self) -> float:
        if self.total_audio_s <= 0:
            return 0.0
        return self.total_processing_s / self.total_audio_s

    @property
    def avg_latency_ms(self) -> float:
        vals = [r.avg_latency_ms for r in self.results if r.avg_latency_ms > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def p95_latency_ms(self) -> float:
        vals = [r.p95_latency_ms for r in self.results if r.p95_latency_ms > 0]
        return sum(vals) / len(vals) if vals else 0.0

    # --- Per-dimension breakdowns ---

    def _group_by(self, key: str) -> Dict[str, List[SampleResult]]:
        groups: Dict[str, List[SampleResult]] = {}
        for r in self.results:
            k = getattr(r, key, "unknown")
            groups.setdefault(k, []).append(r)
        return groups

    def wer_by_language(self) -> Dict[str, float]:
        return {
            lang: sum(r.wer for r in group) / len(group)
            for lang, group in sorted(self._group_by("language").items())
        }

    def rtf_by_language(self) -> Dict[str, float]:
        return {
            lang: sum(r.rtf for r in group) / len(group)
            for lang, group in sorted(self._group_by("language").items())
        }

    def wer_by_category(self) -> Dict[str, float]:
        return {
            cat: sum(r.wer for r in group) / len(group)
            for cat, group in sorted(self._group_by("category").items())
        }

    @property
    def languages(self) -> List[str]:
        return sorted(set(r.language for r in self.results))

    @property
    def categories(self) -> List[str]:
        return sorted(set(r.category for r in self.results))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_version": "1.0",
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "config": {
                "backend": self.backend,
                "model_size": self.model_size,
            },
            "summary": {
                "n_samples": self.n_samples,
                "total_audio_s": round(self.total_audio_s, 1),
                "total_processing_s": round(self.total_processing_s, 1),
                "avg_wer": round(self.avg_wer, 4),
                "weighted_wer": round(self.weighted_wer, 4),
                "avg_rtf": round(self.avg_rtf, 3),
                "overall_rtf": round(self.overall_rtf, 3),
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "p95_latency_ms": round(self.p95_latency_ms, 1),
                "wer_by_language": {
                    k: round(v, 4) for k, v in self.wer_by_language().items()
                },
                "rtf_by_language": {
                    k: round(v, 3) for k, v in self.rtf_by_language().items()
                },
                "wer_by_category": {
                    k: round(v, 4) for k, v in self.wer_by_category().items()
                },
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
