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
    # WER is N/A when the sample reference is empty (the zh_long sample has no
    # matching reference; it is for MT-RTF and translation quality, not ASR WER).
    wer_applicable: bool = True

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

    # Translation metrics (populated when a translation backend is configured)
    first_translation_time_s: Optional[float] = None
    provisional_before_final: bool = False
    mt_call_count: Optional[int] = None
    translation_time_s: Optional[float] = None
    translation_rtf: Optional[float] = None
    translation_accuracy: Optional[float] = None
    translation_metric_name: Optional[str] = None
    reference_translation: str = ""
    hypothesis_translation: str = ""

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
            "wer_applicable": self.wer_applicable,
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
            "first_translation_time_s": round(self.first_translation_time_s, 3) if self.first_translation_time_s is not None else None,
            "provisional_before_final": self.provisional_before_final,
            "mt_call_count": self.mt_call_count,
            "translation_time_s": round(self.translation_time_s, 3) if self.translation_time_s is not None else None,
            "translation_rtf": round(self.translation_rtf, 3) if self.translation_rtf is not None else None,
            "translation_accuracy": round(self.translation_accuracy, 2) if self.translation_accuracy is not None else None,
            "translation_metric_name": self.translation_metric_name,
            "reference_translation": self.reference_translation,
            "hypothesis_translation": self.hypothesis_translation,
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
    translation_backend: Optional[str] = None
    target_language: Optional[str] = None
    simultaneous: bool = False

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

    # --- Translation aggregates (populated when a translation backend ran) ---

    @property
    def has_translation(self) -> bool:
        return any(
            r.mt_call_count is not None
            or r.translation_rtf is not None
            or (r.hypothesis_translation or "")
            for r in self.results
        )

    @property
    def has_wer(self) -> bool:
        return any(r.wer_applicable for r in self.results)

    @property
    def total_translation_time_s(self) -> float:
        return sum(r.translation_time_s for r in self.results if r.translation_time_s)

    @property
    def overall_translation_rtf(self) -> float:
        if self.total_audio_s <= 0:
            return 0.0
        return self.total_translation_time_s / self.total_audio_s

    @property
    def avg_translation_time_s(self) -> float:
        vals = [r.translation_time_s for r in self.results if r.translation_time_s]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_first_translation_time_s(self) -> float:
        vals = [r.first_translation_time_s for r in self.results if r.first_translation_time_s is not None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def total_mt_calls(self) -> int:
        return sum(r.mt_call_count for r in self.results if r.mt_call_count)

    @property
    def n_provisional_before_final(self) -> int:
        return sum(1 for r in self.results if r.provisional_before_final)

    @property
    def avg_translation_accuracy(self) -> Optional[float]:
        vals = [r.translation_accuracy for r in self.results if r.translation_accuracy is not None]
        return sum(vals) / len(vals) if vals else None

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
            "translation": {
                "has_translation": self.has_translation,
                "total_translation_time_s": round(self.total_translation_time_s, 2),
                "overall_translation_rtf": round(self.overall_translation_rtf, 3),
                "avg_first_translation_time_s": round(self.avg_first_translation_time_s, 3),
                "total_mt_calls": self.total_mt_calls,
                "n_provisional_before_final": self.n_provisional_before_final,
                "avg_translation_accuracy": round(self.avg_translation_accuracy, 2)
                if self.avg_translation_accuracy is not None
                else None,
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


# ---------------------------------------------------------------------------
# Translation accuracy scoring
# ---------------------------------------------------------------------------

def compute_translation_accuracy(
    hypothesis: str, reference: str
) -> tuple:
    """Score a translation hypothesis against a reference.

    Returns ``(score, metric_name)`` where ``score`` is in 0–100.

    Tiered fallback:
      1. sacrebleu BLEU (if sacrebleu is importable)
      2. chrF (compact in-house implementation, char n-gram F-beta)
      3. exact word-overlap ratio (last resort)

    A ``hypothesis`` that is empty or whitespace scores 0.
    """
    hyp = (hypothesis or "").strip()
    ref = (reference or "").strip()
    if not hyp or not ref:
        return 0.0, "none"

    # Tier 1: sacrebleu BLEU.
    try:
        import sacrebleu

        bleu = sacrebleu.corpus_bleu([hyp], [[ref]]).score
        return float(bleu), "bleu"
    except ImportError:
        pass
    except Exception:
        pass

    # Tier 2: chrF (compact implementation).
    try:
        chrf = _chrf_score(hyp, ref)
        return chrf, "chrf"
    except Exception:
        pass

    # Tier 3: exact word-overlap ratio.
    hyp_words = hyp.split()
    ref_words = ref.split()
    if not hyp_words or not ref_words:
        return 0.0, "word_overlap"
    hyp_set = set(hyp_words)
    ref_set = set(ref_words)
    overlap = len(hyp_set & ref_set)
    ratio = overlap / max(len(hyp_set | ref_set), 1)
    return ratio * 100.0, "word_overlap"


def _chrf_score(hypothesis: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Compact chrF (character n-gram F-beta) score in 0–100.

    Averages char n-gram precision/recall over orders 1..max_n with word
    boundary spaces preserved (sacrebleu default). No smoothing; matches
    sacrebleu's chrF (non-++) closely enough for a fallback metric.
    """
    def _ngrams(text: str, n: int):
        if len(text) < n:
            return {}
        counts = {}
        for i in range(len(text) - n + 1):
            g = text[i : i + n]
            counts[g] = counts.get(g, 0) + 1
        return counts

    precisions = []
    recalls = []
    for n in range(1, max_n + 1):
        hg = _ngrams(hypothesis, n)
        rg = _ngrams(reference, n)
        if not hg or not rg:
            continue
        overlap = 0
        for g, c in hg.items():
            overlap += min(c, rg.get(g, 0))
        p = overlap / sum(hg.values())
        r = overlap / sum(rg.values())
        precisions.append(p)
        recalls.append(r)
    if not precisions:
        return 0.0
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    f = (1 + beta * beta) * (p * r) / (r + beta * beta * p)
    return f * 100.0
