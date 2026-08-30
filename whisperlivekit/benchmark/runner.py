"""Run diagnostic benchmarks through the same audio pipeline as clients."""

import hashlib
import logging
import math
import time
from typing import Callable, List, Optional

from whisperlivekit.benchmark.compat import backend_supports_language, resolve_backend
from whisperlivekit.benchmark.datasets import BenchmarkSample, get_benchmark_samples
from whisperlivekit.benchmark.metrics import BenchmarkReport, SampleResult, get_system_info

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    def __init__(
        self,
        backend: str = "auto",
        model_size: str = "base",
        languages: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        quick: bool = False,
        speed: float = 0,
        on_progress: Optional[Callable] = None,
    ):
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("Benchmark speed must be finite and non-negative")
        self.backend = resolve_backend(backend)
        self.model_size = model_size
        self.languages = languages
        self.categories = categories
        self.quick = quick
        self.speed = speed
        self.on_progress = on_progress

    async def run(self) -> BenchmarkReport:
        from whisperlivekit.metrics import compute_wer

        samples = get_benchmark_samples(
            languages=self.languages, categories=self.categories, quick=self.quick,
        )
        if not samples:
            raise RuntimeError("No benchmark samples available for the selected languages/categories")
        kwargs = {"model_size": self.model_size, "pcm_input": True, "backend": self.backend}
        report = BenchmarkReport(
            backend=self.backend, model_size=self.model_size,
            system_info=get_system_info(), feed_speed=self.speed,
        )
        for i, sample in enumerate(samples):
            if self.on_progress:
                self.on_progress(sample.name, i, len(samples))
            report.results.append(await self._run_sample(sample, kwargs, compute_wer))
        if self.on_progress:
            self.on_progress("done", len(samples), len(samples))
        return report

    async def _run_sample(self, sample: BenchmarkSample, harness_kwargs: dict, compute_wer) -> SampleResult:
        from whisperlivekit.test_harness import TestHarness

        result = SampleResult(
            sample_name=sample.name, language=sample.language, category=sample.category,
            duration_s=sample.duration, reference=sample.reference,
            source=sample.source, tags=list(sample.tags),
        )
        if not backend_supports_language(self.backend, sample.language):
            result.status = "skipped"
            result.error = f"Language {sample.language} unsupported by benchmark backend {self.backend}"
            return result
        kwargs = {**harness_kwargs, "lan": sample.language}
        feed_started = None
        def observe(state):
            if self.speed == 1 and state.committed_text and result.first_text_time_s is None:
                result.first_text_time_s = time.perf_counter() - feed_started
        try:
            with open(sample.path, "rb") as audio:
                result.audio_sha256 = hashlib.file_digest(audio, "sha256").hexdigest()
            startup = time.perf_counter()
            async with TestHarness(**kwargs) as harness:
                harness.on_update(observe)
                result.startup_time_s = time.perf_counter() - startup
                result.effective_config = {
                    key: value for key, value in vars(harness._processor.args).items()
                    if key != "api_token"
                }
                feed_started = time.perf_counter()
                try:
                    await harness.feed(sample.path, speed=self.speed)
                    state = await harness.finish(timeout=max(120, sample.duration * 2.5))
                finally:
                    result.wall_time_s = time.perf_counter() - feed_started
                    result.hypothesis = harness.state.committed_text or harness.state.text
                metrics = harness.metrics
                result.processing_time_s = metrics.total_processing_time_s
                result.rtf = result.processing_time_s / sample.duration if sample.duration > 0 else None
                result.avg_latency_ms = metrics.avg_latency_ms
                result.p95_latency_ms = metrics.p95_latency_ms
                result.n_transcription_calls = metrics.n_transcription_calls
                result.n_tokens = metrics.n_tokens_produced
                result.n_lines = len(state.speech_lines)
                result.timing_valid = state.timing_valid
                result.timing_monotonic = state.timing_monotonic
            if sample.reference.strip():
                scores = compute_wer(sample.reference, result.hypothesis)
                result.wer = scores["wer"]
                result.wer_details = {key: scores[key] for key in (
                    "substitutions", "insertions", "deletions", "ref_words", "hyp_words",
                )}
        except Exception as exc:
            result.status = "timeout" if isinstance(exc, TimeoutError) else "error"
            result.error = f"{type(exc).__name__}: {exc}"
            result.wer = result.rtf = result.processing_time_s = None
            logger.warning("Benchmark %s failed: %s", sample.name, result.error)
        return result
