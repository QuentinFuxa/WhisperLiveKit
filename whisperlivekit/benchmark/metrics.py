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
    cer: Optional[float] = None
    cer_details: Dict[str, int] = field(default_factory=dict)
    repeat: int = 1
    translation_errors: List[str] = field(default_factory=list)
    first_visible_time_s: Optional[float] = None
    first_translation_time_s: Optional[float] = None
    feed_time_s: Optional[float] = None
    finalization_time_s: Optional[float] = None
    source_end_lag_s: Optional[float] = None
    rss_peak_bytes: Optional[int] = None
    rss_samples: int = 0
    mlx_peak_bytes: Optional[int] = None
    mlx_active_bytes: Optional[int] = None
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
    # Translation half of an audio-to-translation run; empty without a target language.
    target_language: str = ""
    translation: str = ""
    translation_reference: str = ""
    chrf: Optional[float] = None
    final_translation_time_s: Optional[float] = None
    provisional_lead_s: Optional[float] = None
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
    warmup_results: List[SampleResult] = field(default_factory=list)
    corpus_sha256: str = ""
    model_artifacts: Dict[str, Any] = field(default_factory=dict)
    translation: Dict[str, Any] = field(default_factory=dict)

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
    def weighted_cer(self) -> Optional[float]:
        scored = [r for r in self.successful_results if r.cer is not None]
        chars = sum(r.cer_details.get("ref_chars", 0) for r in scored)
        errors = sum(sum(r.cer_details.get(k, 0) for k in
                         ("substitutions", "insertions", "deletions")) for r in scored)
        return errors / chars if chars else None

    @property
    def overall_rtf(self) -> Optional[float]:
        return self.total_processing_s / self.total_audio_s if self.total_audio_s else None

    @property
    def avg_chrf(self) -> Optional[float]:
        """Unweighted mean of the per-row chrF values, for inspection.

        Not the report's chrF: averaging per-sentence scores overweights short
        rows. Read :attr:`corpus_chrf` for the score to quote.
        """
        values = [r.chrf for r in self.successful_results if r.chrf is not None]
        return sum(values) / len(values) if values else None

    @property
    def corpus_chrf(self) -> Optional[float]:
        """chrF over every scored row, from character n-gram counts summed over the pair.

        This is sacrebleu's corpus score for the language pair: the per-order
        match counts of each row are added up first and the F-score is taken
        over the totals. Taking the mean of the per-row scores instead would
        weight a four-character row like a forty-character one.
        """
        from whisperlivekit.metrics import CHRF_CHAR_ORDER, chrf_score, chrf_statistics

        scored = [r for r in self.successful_results if r.chrf is not None]
        if not scored:
            return None
        totals = [0] * (3 * CHRF_CHAR_ORDER)
        for result in scored:
            for index, count in enumerate(chrf_statistics(result.translation_reference,
                                                          result.translation)):
                totals[index] += count
        return chrf_score(totals)

    def percentile(self, field_name, quantile=.95):
        values = sorted(value for r in self.successful_results
                        if (value := getattr(r, field_name)) is not None)
        if not values:
            return None
        index = (len(values) - 1) * quantile
        lower = int(index)
        return values[lower] + (values[min(lower+1, len(values)-1)] - values[lower]) * (index-lower)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_version": "3.2",
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "config": {"asr": describe_asr_config(self.backend, self.model_size),
                       "translation": self.translation,
                       "feed_speed": self.feed_speed},
            "measurement": {
                "audio_pacing": "speed>0: absolute chunk-end deadlines; no post-feed sleep. speed=0: immediate",
                "rtf": "successful ASR call time / audio duration",
                "wall_time_s": "audio feed and EOF drain, excluding startup and cleanup",
                "first_text_time_s": "first committed text since feed start; paced runs only",
                "first_visible_time_s": "first committed or provisional ASR text since feed start; paced runs only",
                "first_translation_time_s": "first committed or provisional translation since feed start; paced runs only",
                "finalization_time_s": "EOF completion minus actual end of audio feeding",
                "source_end_lag_s": "EOF completion minus nominal source duration / feed speed; includes feed backlog",
                "rss_peak_bytes": "largest process RSS sampled every 50 ms, including startup; not an exact high-water mark",
                "mlx_peak_bytes": "MLX allocator peak since per-sample reset; separate from RSS, do not add them",
                "quality": "NFC, lowercase, punctuation-normalized WER; Chinese uses CER with whitespace removed",
                "chrf": "per-row chrF: sacrebleu's CHRF(char_order=6, word_order=0, beta=2) for the translation against the same sentence_id's reference in the target language, whitespace removed and case and punctuation kept; null without a translation hypothesis and reference",
                "corpus_chrf": "the report's chrF: character n-gram counts summed over every scored row of the language pair, scored once as sacrebleu scores a corpus; the headline quality figure",
                "avg_chrf": "unweighted mean of the per-row chrf values, kept for inspection only; it overweights short rows and is not corpus chrF",
                "translation_reference": "reference of the same sentence_id in the target language, joined from the corpus manifest",
                "final_translation_time_s": "last committed line that first showed a translation, since feed start; paced runs only",
                "provisional_lead_s": "final_translation_time_s minus first_translation_time_s, so how long drafting ran before the translation settled; null unless the backend drafted provisional text",
                "asr_call_p95_ms": "last 4096 inference calls per sample, not word latency",
                "model_revisions": "explicit local artifacts are hashed; floating Hub aliases are not resolved",
            },
            "corpus_sha256": self.corpus_sha256,
            "model_artifacts": self.model_artifacts,
            "warmup_results": [r.to_dict() for r in self.warmup_results],
            "summary": {
                "n_samples": self.n_samples,
                "n_successful": len(self.successful_results),
                "n_failed": self.n_failed,
                "n_skipped": sum(r.status == "skipped" for r in self.results),
                "total_audio_s": self.total_audio_s,
                "total_processing_s": self.total_processing_s,
                "avg_wer": self.avg_wer,
                "weighted_wer": self.weighted_wer,
                "weighted_cer": self.weighted_cer,
                "corpus_chrf": self.corpus_chrf,
                "avg_chrf": self.avg_chrf,
                "first_translation_p95_s": self.percentile("first_translation_time_s"),
                "final_translation_p95_s": self.percentile("final_translation_time_s"),
                "provisional_lead_p95_s": self.percentile("provisional_lead_s"),
                "finalization_p95_s": self.percentile("finalization_time_s"),
                "first_visible_p95_s": self.percentile("first_visible_time_s"),
                "source_end_lag_p95_s": self.percentile("source_end_lag_s"),
                "n_warmup_failed": sum(r.status in ("error", "timeout") for r in self.warmup_results),
                "n_invalid_timestamps": sum(not (r.timing_valid and r.timing_monotonic) for r in self.successful_results),
                "rss_peak_bytes": max((r.rss_peak_bytes for r in self.results if r.rss_peak_bytes is not None), default=None),
                "mlx_peak_bytes": max((r.mlx_peak_bytes for r in self.results if r.mlx_peak_bytes is not None), default=None),
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

    # Metadata lookup avoids importing unrelated model runtimes.
    versions = {}
    for name in ("faster-whisper", "openai-whisper", "mlx-whisper", "transformers",
                 "torch", "mlx", "mlx-lm", "mlx-metal", "mlx-audio", "mlx-qwen3-asr",
                 "vllm", "vllm-metal", "qwen3-asr-causal", "numpy", "psutil"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    info["backend_versions"] = versions
    return info


def describe_asr_config(backend: str, model: str) -> Dict[str, Any]:
    """Record the ASR half of a run: the resolved backend and the model it loaded.

    An audio-to-translation report has to state both halves, because the same
    MT settings score differently behind a different recognizer.
    """
    return {"backend": backend, "model": model}


# MT backends that always draft: those whose ``process()`` returns the target
# text they cannot commit yet as its second element. ``nllw`` is the library
# behind the ``nllb`` backend, accepted as the same name.
_ALWAYS_DRAFTING_BACKENDS = ("nllb", "nllw", "alignatt")
# MT backends with an in-process simultaneous variant selected by
# ``mlx_llm_mt_simultaneous``; those are the ones whose drafting is a switch.
_SIMULTANEOUS_MLX_BACKENDS = ("mlx-llm-mt", "hunyuan-mlx")


def drafts_provisional_text(config: Dict[str, Any]) -> bool:
    """Whether the configured MT backend publishes provisional target text.

    Two arrangements put a partial translation in the streaming buffer: the
    alignatt server and nllw's ``OnlineTranslation`` (the ``nllb`` backend),
    which both return the target tokens outside the committed prefix as their
    buffer, and the MLX backend with ``mlx_llm_mt_simultaneous``. That covers
    every backend ``--translation-backend`` accepts, so the segment MLX path is
    the only non-drafting one: it translates completed segments and returns an
    empty buffer, and a non-empty one there holds untranslated source text,
    which read as a draft would report a translation lead the run never had.
    """
    backend = config.get("translation_backend", "")
    if backend in _ALWAYS_DRAFTING_BACKENDS:
        return True
    return backend in _SIMULTANEOUS_MLX_BACKENDS and bool(config.get("mlx_llm_mt_simultaneous", False))


def describe_translation_config(config: Dict[str, Any],
                                source_languages=()) -> Dict[str, Any]:
    """Record the MT half of an audio-to-translation run.

    An audio-to-translation benchmark has to state what produced the target
    text, not only the ASR backend: the direction (the source languages that
    were fed against the target language), the backend and the loaded model.
    ``simultaneous`` is the MLX drafting switch and stays null where the backend
    has no such switch. Two backends cannot name a model from the client side,
    so they say what they can see instead of leaving ``model`` to be read as an
    unknown: the alignatt model runs in a server whose endpoint is recorded, and
    nllw resolves NLLB's checkpoint from its own settings.
    """
    backend = config.get("translation_backend", "")
    record = {
        "source_languages": sorted({language for language in source_languages if language}),
        "target_language": config.get("target_language", ""),
        "backend": backend,
        "model": None,
        "simultaneous": None,
    }
    if backend in _SIMULTANEOUS_MLX_BACKENDS:
        record["model"] = config.get("mlx_llm_mt_model")
        record["simultaneous"] = bool(config.get("mlx_llm_mt_simultaneous", False))
    elif backend == "alignatt":
        # The MT model is loaded by the translation server, so the run records
        # which server it called; the checkpoint behind it is not visible here.
        record["model_note"] = ("the translation model runs in the alignatt server; this run "
                                "records the endpoint it called, not the served checkpoint")
        record["server"] = {"url": config.get("alignatt_url"),
                            "preset": config.get("alignatt_preset"),
                            "latency": config.get("alignatt_latency")}
    elif backend:
        record["model_note"] = ("nllw selects the NLLB checkpoint from nllb_backend and "
                                "nllb_size; both are recorded in each row's effective_config")
    return record


def describe_model_artifacts(config):
    """Identify explicitly supplied model artifacts without resolving floating Hub refs."""
    import hashlib
    from pathlib import Path

    artifacts = {}
    for key, value in config.items():
        if not isinstance(value, str) or not value or "model" not in key or "cache" in key:
            continue
        path = Path(value).expanduser()
        if not path.exists():
            continue
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        records = []
        for file in files:
            with file.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            records.append({"file": str(file.relative_to(path)) if path.is_dir() else file.name,
                            "bytes": file.stat().st_size, "sha256": digest})
        revision = None
        if "snapshots" in path.parts:
            revision = path.parts[path.parts.index("snapshots") + 1]
        artifacts[key] = {"path": str(path), "snapshot_revision": revision, "files": records}
    return artifacts
