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
        translation_backend: Translation backend name (None = no translation).
        target_language: Target language for translation (e.g. "en").
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
        translation_backend: Optional[str] = None,
        target_language: Optional[str] = None,
        simultaneous: bool = False,
        reference_translation: Optional[str] = None,
        on_progress: Optional[Callable] = None,
        mlx_llm_mt_model: Optional[str] = None,
    ):
        self.backend = resolve_backend(backend)
        self.model_size = model_size
        self.languages = languages
        self.categories = categories
        self.quick = quick
        self.speed = speed
        self.translation_backend = translation_backend
        self.target_language = target_language
        self.simultaneous = simultaneous
        self.reference_translation = reference_translation
        self.on_progress = on_progress
        self.mlx_llm_mt_model = mlx_llm_mt_model

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
        if self.translation_backend:
            harness_kwargs["translation_backend"] = self.translation_backend
        if self.target_language:
            harness_kwargs["target_language"] = self.target_language
        if self.simultaneous:
            harness_kwargs["mlx_llm_mt_simultaneous"] = True
        if self.mlx_llm_mt_model:
            harness_kwargs["mlx_llm_mt_model"] = self.mlx_llm_mt_model

        report = BenchmarkReport(
            backend=self.backend,
            model_size=self.model_size,
            system_info=get_system_info(),
            translation_backend=self.translation_backend,
            target_language=self.target_language,
            simultaneous=self.simultaneous,
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

        # Translation tracking state (populated via on_update callback)
        first_provisional_time: Optional[float] = None
        first_final_time: Optional[float] = None

        def _track_translation(state) -> None:
            nonlocal first_provisional_time, first_final_time
            now = time.perf_counter() - t_start
            # Provisional: buffer_translation is non-empty during speech.
            if state.buffer_translation and first_provisional_time is None:
                first_provisional_time = now
            # Final: a committed line carries a translation field.
            if first_final_time is None:
                for line in state.lines:
                    if line.get("translation"):
                        first_final_time = now
                        break

        # first_translation_time is whichever came first.
        first_translation_time: Optional[float] = None

        # Memory before
        mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        t_start = time.perf_counter()

        async with TestHarness(**kwargs) as h:
            if self.translation_backend:
                h.on_update(_track_translation)
            await h.feed(sample.path, speed=self.speed)
            # Drain time scales with audio duration for slow backends
            drain = max(5.0, sample.duration * 0.5)
            await h.drain(drain)
            state = await h.finish(timeout=120)
            # Final sweep: the last state update may carry the final translation.
            if self.translation_backend:
                _track_translation(h.state)

            # Extract metrics from the pipeline
            metrics = h.metrics

            # Read MT-call count and translation wall-time from the translation
            # backend if available (instrumented in the backend).
            mt_call_count = None
            mt_total_time_s = None
            if self.translation_backend and h._processor:
                translation = getattr(h._processor, "translation", None)
                if translation is not None:
                    mt_call_count = getattr(translation, "_mt_call_count", None)
                    mt_total_time_s = getattr(translation, "_mt_total_time_s", None)

        t_elapsed = time.perf_counter() - t_start

        # Memory after
        mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS ru_maxrss is bytes, on Linux it's KB
        import sys
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        mem_delta = (mem_after - mem_before) / divisor

        # RTF
        rtf = t_elapsed / sample.duration if sample.duration > 0 else 0

        # WER — skip when the reference is empty (not applicable for this sample).
        hypothesis = state.committed_text or state.text
        if sample.reference and sample.reference.strip():
            wer_result = compute_wer(sample.reference, hypothesis)
            wer = wer_result["wer"]
            wer_details = {
                "substitutions": wer_result["substitutions"],
                "insertions": wer_result["insertions"],
                "deletions": wer_result["deletions"],
                "ref_words": wer_result["ref_words"],
                "hyp_words": wer_result["hyp_words"],
            }
        else:
            wer = 0.0  # not applicable; report shows N/A
            wer_details = {
                "substitutions": 0,
                "insertions": 0,
                "deletions": 0,
                "ref_words": 0,
                "hyp_words": 0,
            }

        # Compute translation timing metrics.
        times = [t for t in (first_provisional_time, first_final_time) if t is not None]
        first_translation_time = min(times) if times else None
        provisional_before_final = (
            first_provisional_time is not None
            and (first_final_time is None or first_provisional_time < first_final_time)
        )

        # Latency from SessionMetrics
        avg_lat = metrics.avg_latency_ms if metrics else 0
        p95_lat = metrics.p95_latency_ms if metrics else 0
        n_calls = metrics.n_transcription_calls if metrics else 0
        n_tokens = metrics.n_tokens_produced if metrics else 0

        # Translation accuracy (only when a reference translation is available).
        translation_accuracy = None
        translation_metric_name = None
        hyp_translation = ""
        if self.translation_backend:
            # Concatenate committed line translations (silence/empty lines have
            # no translation field and are naturally excluded).
            parts = [
                line.get("translation", "")
                for line in state.lines
                if line.get("translation")
            ]
            # Fall back to the live buffer translation if no committed lines.
            if not parts and state.buffer_translation:
                parts = [state.buffer_translation]
            hyp_translation = " ".join(p.strip() for p in parts if p and p.strip())
            if self.reference_translation:
                from whisperlivekit.benchmark.metrics import compute_translation_accuracy
                translation_accuracy, translation_metric_name = compute_translation_accuracy(
                    hyp_translation, self.reference_translation
                )
        translation_rtf = (
            mt_total_time_s / sample.duration
            if mt_total_time_s is not None and sample.duration > 0
            else None
        )

        return SampleResult(
            sample_name=sample.name,
            language=sample.language,
            category=sample.category,
            duration_s=sample.duration,
            wer=wer,
            wer_details=wer_details,
            wer_applicable=bool(sample.reference and sample.reference.strip()),
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
            first_translation_time_s=round(first_translation_time, 3) if first_translation_time is not None else None,
            provisional_before_final=provisional_before_final,
            mt_call_count=mt_call_count,
            translation_time_s=round(mt_total_time_s, 3) if mt_total_time_s is not None else None,
            translation_rtf=round(translation_rtf, 3) if translation_rtf is not None else None,
            translation_accuracy=round(translation_accuracy, 2) if translation_accuracy is not None else None,
            translation_metric_name=translation_metric_name,
            reference_translation=self.reference_translation or "",
            hypothesis_translation=hyp_translation,
            hypothesis=hypothesis,
            reference=sample.reference,
            source=sample.source,
            tags=list(sample.tags),
        )
