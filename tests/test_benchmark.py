"""Benchmark completion and failure semantics through the real PCM harness."""

import asyncio
import io
import json
import wave
from dataclasses import replace

import pytest


@pytest.mark.asyncio
async def test_benchmark_retains_failures_without_scoring_partial_results(tmp_path, monkeypatch):
    from whisperlivekit.benchmark.datasets import BenchmarkSample
    from whisperlivekit.benchmark.metrics import BenchmarkReport
    from whisperlivekit.benchmark.report import print_report, write_json
    from whisperlivekit.benchmark.runner import BenchmarkRunner
    from whisperlivekit.metrics import compute_wer
    from whisperlivekit.test_harness import TestHarness

    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(3200))
    sample = BenchmarkSample("silence", str(path), "", .1, "en", "clean")
    runner = BenchmarkRunner(backend="whisper")
    # Real ingestion/EOF/collector, with inference disabled: this scenario
    # checks measurement plumbing, not the accuracy or speed of a model.
    config = dict(transcription=False, vac=False, pcm_input=True)
    async def no_artificial_drain(*args):
        raise AssertionError("A benchmark must finish at EOF, not sleep for an arbitrary drain")
    monkeypatch.setattr(TestHarness, "drain", no_artificial_drain)
    good = await runner._run_sample(sample, config, compute_wer)
    assert good.status == "ok"
    assert len(good.audio_sha256) == 64
    assert good.effective_config["pcm_input"] is True
    assert good.wer is None  # no reference is not 0% WER
    assert good.wall_time_s is not None and good.startup_time_s is not None

    async def timeout(*args, **kwargs):
        raise TimeoutError("EOF did not complete")
    monkeypatch.setattr(TestHarness, "finish", timeout)
    timed_out = await runner._run_sample(replace(sample, name="timeout"), config, compute_wer)
    assert timed_out.status == "timeout"
    assert timed_out.rtf is None and timed_out.wer is None
    missing = await runner._run_sample(replace(sample, path=str(tmp_path / "missing.wav")), config, compute_wer)
    assert missing.status == "error"
    report = BenchmarkReport("whisper", "tiny", results=[good, timed_out, missing])
    destination = tmp_path / "report.json"
    write_json(report, str(destination))
    data = json.loads(destination.read_text())
    assert data["summary"]["n_failed"] == 2
    assert data["summary"]["n_successful"] == 1
    assert data["summary"]["weighted_wer"] is None
    assert data["summary"]["total_audio_s"] == .1
    assert len(data["results"]) == 3
    out = io.StringIO()
    print_report(report, out)
    assert "timeout" in out.getvalue() and "1/3 completed" in out.getvalue()


@pytest.mark.asyncio
async def test_harness_finish_propagates_timeout_and_collector_errors():
    from whisperlivekit.test_harness import TestHarness
    from whisperlivekit.timed_objects import FrontData

    for failure in ("timeout", "error"):
        async with TestHarness(transcription=False, vac=False, pcm_input=True) as harness:
            harness._collect_task.cancel()
            await asyncio.gather(harness._collect_task, return_exceptions=True)
            async def results():
                if failure == "timeout":
                    await asyncio.Event().wait()
                yield FrontData(status="error", error="decoder failed")
            harness._results_gen = results()
            harness._collect_task = asyncio.create_task(harness._collect_results())
            with pytest.raises(TimeoutError if failure == "timeout" else RuntimeError):
                await harness.finish(timeout=.05)
