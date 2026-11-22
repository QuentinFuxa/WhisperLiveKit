"""Run diagnostic benchmarks through the same audio pipeline as clients."""

import hashlib
import logging
import math
import time
from dataclasses import fields, replace
from typing import Callable, List, Optional

from whisperlivekit.benchmark.compat import backend_supports_language, resolve_backend
from whisperlivekit.benchmark.datasets import BenchmarkSample, get_benchmark_samples
from whisperlivekit.benchmark.metrics import (
    BenchmarkReport,
    SampleResult,
    describe_translation_config,
    drafts_provisional_text,
    get_system_info,
)
from whisperlivekit.benchmark.resources import ResourceMonitor

logger = logging.getLogger(__name__)


def _line_translation(line) -> str:
    """Translation text carried by one front-end line, or "" when it has none."""
    translation = line.get("translation")
    if not translation:
        return ""
    text = translation if isinstance(translation, str) else getattr(translation, "text", "")
    return (text or "").strip()


def _committed_translation(lines) -> str:
    """Join the translations attached to the committed lines."""
    return " ".join(text for text in (_line_translation(line) for line in lines) if text)


class BenchmarkRunner:
    def __init__(
        self, backend: str = "auto", model_size: str = "base",
        languages: Optional[List[str]] = None, categories: Optional[List[str]] = None,
        quick: bool = False, speed: float = 0, on_progress: Optional[Callable] = None,
        samples=None, repeats=1, warmup=False, engine_kwargs=None,
    ):
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("Benchmark speed must be finite and non-negative")
        if not isinstance(repeats, int) or repeats < 1:
            raise ValueError("Benchmark repeats must be a positive integer")
        self.backend = resolve_backend(backend)
        self.model_size, self.languages, self.categories = model_size, languages, categories
        self.quick, self.speed, self.on_progress = quick, speed, on_progress
        self.samples, self.repeats, self.warmup = samples, repeats, warmup
        self.engine_kwargs = engine_kwargs or {}
        from whisperlivekit.config import WhisperLiveKitConfig
        unknown = self.engine_kwargs.keys() - {field.name for field in fields(WhisperLiveKitConfig)}
        if unknown:
            raise ValueError(f"Unknown benchmark engine options: {sorted(unknown)}")
        if {"backend", "model_size", "lan"} & self.engine_kwargs.keys():
            raise ValueError("Set backend, model_size and language using the benchmark options")

    async def run(self) -> BenchmarkReport:
        from whisperlivekit.metrics import compute_wer

        samples = self.samples if self.samples is not None else get_benchmark_samples(
            languages=self.languages, categories=self.categories, quick=self.quick,
        )
        samples = [sample for sample in samples
                   if (not self.languages or sample.language in self.languages)
                   and (not self.categories or sample.category in self.categories)]
        if not samples:
            raise RuntimeError("No benchmark samples available for the selected languages/categories")
        kwargs = {"model_size": self.model_size, "pcm_input": True, "backend": self.backend,
                  **self.engine_kwargs}
        report = BenchmarkReport(backend=self.backend, model_size=self.model_size,
                                 system_info=get_system_info(), feed_speed=self.speed)
        if self.warmup:
            for language in dict.fromkeys(sample.language for sample in samples):
                sample = next(sample for sample in samples if sample.language == language)
                result = await self._run_sample(sample, kwargs, compute_wer)
                report.warmup_results.append(replace(result, repeat=0))
        for repeat in range(1, self.repeats + 1):
            for i, sample in enumerate(samples):
                if self.on_progress:
                    self.on_progress(sample.name, (repeat-1)*len(samples)+i, self.repeats*len(samples))
                result = await self._run_sample(sample, kwargs, compute_wer)
                report.results.append(replace(result, repeat=repeat))
        if self.on_progress:
            self.on_progress("done", self.repeats*len(samples), self.repeats*len(samples))
        # Hash after timing: reading weights beforehand would warm the OS cache.
        from whisperlivekit.benchmark.metrics import describe_model_artifacts
        report.model_artifacts = describe_model_artifacts(kwargs)
        # Record what the engine ran with, not only what was asked for: the
        # configuration fills in defaults engine_kwargs never mentions, such as
        # the alignatt endpoint or the MT model id.
        effective = next((r.effective_config for r in report.results if r.effective_config), None)
        report.translation = describe_translation_config(
            effective or self.engine_kwargs, (sample.language for sample in samples))
        return report

    async def _run_sample(self, sample: BenchmarkSample, harness_kwargs: dict, compute_wer) -> SampleResult:
        from whisperlivekit.metrics import compute_chrf, normalize_text
        from whisperlivekit.test_harness import TestHarness

        target_language = harness_kwargs.get("target_language", "")
        result = SampleResult(sample_name=sample.name, language=sample.language, category=sample.category,
                              duration_s=sample.duration, reference=sample.reference,
                              source=sample.source, tags=sorted(sample.tags),
                              target_language=target_language)
        if not backend_supports_language(self.backend, sample.language):
            result.status = "skipped"
            result.error = f"Language {sample.language} unsupported by benchmark backend {self.backend}"
            return result
        kwargs = {**harness_kwargs, "lan": sample.language}
        feed_started = None
        # First moment each committed line showed a translation: one half of the
        # translation latency, taken with whether the backend drafts above.
        line_translation_times = {}

        def observe(state):
            if state.translation_error and state.translation_error not in result.translation_errors:
                result.translation_errors.append(state.translation_error)
            if self.speed != 1:
                return
            elapsed = time.perf_counter() - feed_started
            if state.committed_text and result.first_text_time_s is None:
                result.first_text_time_s = elapsed
            if state.text and result.first_visible_time_s is None:
                result.first_visible_time_s = elapsed
            if result.first_translation_time_s is None and (
                state.buffer_translation or any(line.get("translation") for line in state.lines)
            ):
                result.first_translation_time_s = elapsed
            for line in state.lines:
                if _line_translation(line):
                    line_translation_times.setdefault(str(line.get("start")), elapsed)

        monitor = ResourceMonitor()
        try:
            with open(sample.path, "rb") as audio:
                result.audio_sha256 = hashlib.file_digest(audio, "sha256").hexdigest()
            if sample.expected_sha256 and result.audio_sha256 != sample.expected_sha256:
                raise ValueError("Audio SHA-256 differs from the corpus manifest")
            with monitor:
                startup = time.perf_counter()
                async with TestHarness(**kwargs) as harness:
                    harness.on_update(observe)
                    result.startup_time_s = time.perf_counter() - startup
                    result.effective_config = {
                        key: value for key, value in vars(harness._processor.args).items()
                        if key != "api_token"
                    }
                    # Drafting is a property of the MT backend the engine
                    # resolved, not of the flags the caller passed: the bench
                    # CLI leaves --translation-backend unset by default and the
                    # configuration fills in nllb, which drafts. It is not a
                    # property of the buffer either: a segment backend can hold
                    # untranslated source text there, and a lead computed from
                    # that would invent a provisional translation.
                    drafting_backend = drafts_provisional_text(result.effective_config)
                    feed_started = time.perf_counter()
                    try:
                        # Decode before starting the audio clock; conversion is not streaming latency.
                        player = harness.load_audio(sample.path)
                        feed_started = time.perf_counter()
                        await player.play(speed=self.speed)
                        feed_finished = time.perf_counter()
                        result.feed_time_s = feed_finished - feed_started
                        state = await harness.finish(timeout=max(120, sample.duration * 2.5))
                        finished = time.perf_counter()
                        result.finalization_time_s = finished - feed_finished
                        if self.speed:
                            result.source_end_lag_s = finished - feed_started - sample.duration/self.speed
                    finally:
                        result.wall_time_s = time.perf_counter() - feed_started
                        result.hypothesis = harness.state.committed_text or harness.state.text
                        result.translation = (_committed_translation(harness.state.lines)
                                              or harness.state.buffer_translation)
                        if line_translation_times:
                            result.final_translation_time_s = max(line_translation_times.values())
                        if drafting_backend and result.first_translation_time_s is not None \
                                and result.final_translation_time_s is not None:
                            result.provisional_lead_s = (result.final_translation_time_s
                                                         - result.first_translation_time_s)
                    metrics = harness.metrics
                    result.processing_time_s = metrics.total_processing_time_s
                    result.rtf = result.processing_time_s / sample.duration if sample.duration > 0 else None
                    result.avg_latency_ms, result.p95_latency_ms = metrics.avg_latency_ms, metrics.p95_latency_ms
                    result.n_transcription_calls, result.n_tokens = metrics.n_transcription_calls, metrics.n_tokens_produced
                    result.n_lines = len(state.speech_lines)
                    result.timing_valid, result.timing_monotonic = state.timing_valid, state.timing_monotonic
            if result.translation_errors:
                raise RuntimeError("Translation failed: " + "; ".join(result.translation_errors))
            if result.target_language:
                result.translation_reference = sample.translation_references.get(result.target_language, "")
                if result.translation.strip() and result.translation_reference.strip():
                    result.chrf = compute_chrf(result.translation_reference, result.translation)
            if sample.reference.strip():
                if sample.language in {"zh", "cmn", "ja", "yue"}:
                    ref = " ".join("".join(normalize_text(sample.reference).split()))
                    hyp = " ".join("".join(normalize_text(result.hypothesis).split()))
                    scores = compute_wer(ref, hyp)
                    result.cer = scores["wer"]
                    result.cer_details = {key: scores[key] for key in ("substitutions", "insertions", "deletions")}
                    result.cer_details.update(ref_chars=scores["ref_words"], hyp_chars=scores["hyp_words"])
                else:
                    scores = compute_wer(sample.reference, result.hypothesis)
                    result.wer = scores["wer"]
                    result.wer_details = {key: scores[key] for key in (
                        "substitutions", "insertions", "deletions", "ref_words", "hyp_words")}
        except Exception as exc:
            result.status = "timeout" if isinstance(exc, TimeoutError) else "error"
            result.error = f"{type(exc).__name__}: {exc}"
            result.wer = result.cer = result.rtf = result.processing_time_s = None
            logger.warning("Benchmark %s failed: %s", sample.name, result.error)
        finally:
            result.rss_peak_bytes, result.rss_samples = monitor.rss_peak_bytes, monitor.rss_samples
            result.mlx_peak_bytes, result.mlx_active_bytes = monitor.mlx_peak_bytes, monitor.mlx_active_bytes
        return result
