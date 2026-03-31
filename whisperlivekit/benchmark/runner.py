"""Benchmark runner — orchestrates runs through TestHarness."""

import logging
import resource
import time
from typing import Callable, List, Optional

from whisperlivekit.benchmark.compat import backend_supports_language, resolve_backend
from whisperlivekit.benchmark.datasets import BenchmarkSample, get_benchmark_samples
from whisperlivekit.benchmark.metrics import BenchmarkReport, SampleResult, get_system_info

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates benchmark runs through TestHarness.

    Args:
        backend: ASR backend name or "auto".
        model_size: Model size (e.g. "base", "large-v3").
        languages: Language codes to benchmark (None = all available).
        categories: Categories to benchmark (None = all).
        quick: Use a small subset for fast smoke tests.
        speed: Feed speed (0 = instant, 1.0 = real-time).
        on_progress: Callback(sample_name, i, total) for progress updates.
    """

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
        self.backend = resolve_backend(backend)
        self.model_size = model_size
        self.languages = languages
        self.categories = categories
        self.quick = quick
        self.speed = speed
        self.on_progress = on_progress

    async def run(self) -> BenchmarkReport:
        """Run the full benchmark suite and return a report."""
        from whisperlivekit.metrics import compute_wer

        # Get samples
        samples = get_benchmark_samples(
            languages=self.languages,
            categories=self.categories,
            quick=self.quick,
        )

        # Filter by backend language support
        compatible = []
        for s in samples:
            if backend_supports_language(self.backend, s.language):
                compatible.append(s)
            else:
                logger.info(
                    "Skipping %s (%s) — backend %s does not support %s",
                    s.name, s.language, self.backend, s.language,
                )
        samples = compatible

        if not samples:
            raise RuntimeError(
                f"No benchmark samples available for backend={self.backend}, "
                f"languages={self.languages}, categories={self.categories}"
            )

        # Build harness kwargs
        harness_kwargs = {
            "model_size": self.model_size,
            "lan": "auto",  # let the model auto-detect for multilingual
            "pcm_input": True,
        }
        if self.backend not in ("auto",):
            harness_kwargs["backend"] = self.backend

        report = BenchmarkReport(
            backend=self.backend,
            model_size=self.model_size,
            system_info=get_system_info(),
        )

        for i, sample in enumerate(samples):
            if self.on_progress:
                self.on_progress(sample.name, i, len(samples))

            result = await self._run_sample(
                sample, harness_kwargs, compute_wer,
            )
            report.results.append(result)

        if self.on_progress:
            self.on_progress("done", len(samples), len(samples))

        return report

    async def _run_sample(
        self,
        sample: BenchmarkSample,
        harness_kwargs: dict,
        compute_wer,
    ) -> SampleResult:
        """Benchmark a single sample through TestHarness."""
        from whisperlivekit.test_harness import TestHarness

        # Override language for the specific sample
        kwargs = {**harness_kwargs, "lan": sample.language}

        # Memory before
        mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        t_start = time.perf_counter()

        async with TestHarness(**kwargs) as h:
            await h.feed(sample.path, speed=self.speed)
            # Drain time scales with audio duration for slow backends
            drain = max(5.0, sample.duration * 0.5)
            await h.drain(drain)
            state = await h.finish(timeout=120)

            # Extract metrics from the pipeline
            metrics = h.metrics

        t_elapsed = time.perf_counter() - t_start

        # Memory after
        mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS ru_maxrss is bytes, on Linux it's KB
        import sys
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        mem_delta = (mem_after - mem_before) / divisor

        # RTF
        rtf = t_elapsed / sample.duration if sample.duration > 0 else 0

        # WER
        hypothesis = state.committed_text or state.text
        wer_result = compute_wer(sample.reference, hypothesis)

        # Latency from SessionMetrics
        avg_lat = metrics.avg_latency_ms if metrics else 0
        p95_lat = metrics.p95_latency_ms if metrics else 0
        n_calls = metrics.n_transcription_calls if metrics else 0
        n_tokens = metrics.n_tokens_produced if metrics else 0

        return SampleResult(
            sample_name=sample.name,
            language=sample.language,
            category=sample.category,
            duration_s=sample.duration,
            wer=wer_result["wer"],
            wer_details={
                "substitutions": wer_result["substitutions"],
                "insertions": wer_result["insertions"],
                "deletions": wer_result["deletions"],
                "ref_words": wer_result["ref_words"],
                "hyp_words": wer_result["hyp_words"],
            },
            processing_time_s=round(t_elapsed, 2),
            rtf=round(rtf, 3),
            avg_latency_ms=round(avg_lat, 1),
            p95_latency_ms=round(p95_lat, 1),
            n_transcription_calls=n_calls,
            n_lines=len(state.speech_lines),
            n_tokens=n_tokens,
            timing_valid=state.timing_valid,
            timing_monotonic=state.timing_monotonic,
            peak_memory_mb=round(mem_delta, 1) if mem_delta > 0 else None,
            hypothesis=hypothesis,
            reference=sample.reference,
            source=sample.source,
            tags=list(sample.tags),
        )
