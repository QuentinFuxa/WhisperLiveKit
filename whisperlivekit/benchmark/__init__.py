"""WhisperLiveKit benchmark suite.

Comprehensive benchmarking of ASR backends using public datasets,
run through the same pipeline as real-time streaming.

Usage:
    wlk bench                           # benchmark current backend
    wlk bench --backend whisper --json results.json
    wlk bench --languages en,fr,es      # multilingual
    wlk bench --quick                   # fast subset

Programmatic:
    from whisperlivekit.benchmark import BenchmarkRunner
    import asyncio

    runner = BenchmarkRunner(backend="whisper", model_size="base")
    report = asyncio.run(runner.run())
    print(report.summary_table())
"""

from whisperlivekit.benchmark.datasets import (
    BENCHMARK_CATALOG,
    get_benchmark_samples,
)
from whisperlivekit.benchmark.metrics import BenchmarkReport, SampleResult
from whisperlivekit.benchmark.runner import BenchmarkRunner

__all__ = [
    "BENCHMARK_CATALOG",
    "BenchmarkReport",
    "BenchmarkRunner",
    "SampleResult",
    "get_benchmark_samples",
]
