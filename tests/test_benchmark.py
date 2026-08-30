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
    assert good.finalization_time_s is not None and good.feed_time_s is not None
    tampered = await runner._run_sample(replace(sample, expected_sha256="0"*64), config, compute_wer)
    assert tampered.status == "error" and "SHA-256" in tampered.error
    chinese = await runner._run_sample(replace(sample, language="zh", reference="你好。"), config, compute_wer)
    assert chinese.cer == 1 and chinese.wer is None
    assert chinese.cer_details["ref_chars"] == 2

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

    for failure in ("timeout", "error", "translation"):
        async with TestHarness(transcription=False, vac=False, pcm_input=True) as harness:
            harness._collect_task.cancel()
            await asyncio.gather(harness._collect_task, return_exceptions=True)
            async def results():
                if failure == "timeout":
                    await asyncio.Event().wait()
                if failure == "translation":
                    yield FrontData(translation_error="capture failed")
                else:
                    yield FrontData(status="error", error="decoder failed")
            harness._results_gen = results()
            harness._collect_task = asyncio.create_task(harness._collect_results())
            with pytest.raises(TimeoutError if failure == "timeout" else RuntimeError):
                await harness.finish(timeout=.05)


def test_float_wav_preserves_audio_amplitude(tmp_path):
    import numpy as np
    import soundfile as sf

    from whisperlivekit.test_harness import load_audio_pcm

    path = tmp_path / "float.wav"
    values = np.array([0, .5, -.5, 1, -1], dtype=np.float32)
    sf.write(path, values, 16000, subtype="FLOAT")
    np.testing.assert_array_equal(np.frombuffer(load_audio_pcm(path), dtype="<i2"), [0, 16384, -16384, 32767, -32768])


@pytest.mark.asyncio
async def test_pcm_pacing_uses_chunk_end_deadlines_without_an_eof_delay(monkeypatch):
    from types import SimpleNamespace

    import whisperlivekit.test_harness as testing

    class Clock:
        now = 0.0

        def time(self):
            return self.now

        async def sleep(self, delay):
            assert delay > 0
            self.now += delay

    # Include a short final packet, temporary backpressure and a slow last
    # write. Processing time must not be added to every subsequent deadline.
    for costs, expected_calls, expected_end in [
        ([0, 0, 0], [.5, 1, 1.25], 1.25),
        ([.7, 0, 0], [.5, 1.2, 1.25], 1.25),
        ([0, 0, .8], [.5, 1, 1.25], 2.05),
    ]:
        clock = Clock()
        calls, chunks = [], []

        async def receive(pcm):
            calls.append(clock.now)
            chunks.append(pcm)
            clock.now += costs[len(calls) - 1]

        monkeypatch.setattr(testing, "asyncio", SimpleNamespace(
            get_running_loop=lambda: clock, sleep=clock.sleep,
        ))
        harness = object.__new__(testing.TestHarness)
        harness._processor = SimpleNamespace(process_audio=receive)
        harness._audio_position = 0.0
        audio = bytes(40000)  # 1.25 s at 16 kHz, including a 0.25 s last packet.
        await harness.feed_pcm(audio)
        assert calls == pytest.approx(expected_calls)
        assert clock.now == pytest.approx(expected_end)
        assert [len(chunk) for chunk in chunks] == [16000, 16000, 8000]
        assert b"".join(chunks) == audio
        assert harness._audio_position == 1.25

    calls.clear()
    chunks.clear()
    costs = [0, 0, 0]
    clock.now = 0
    await harness.feed_pcm(audio, speed=0)
    assert calls == [0, 0, 0]
    for options in ({"speed": -1}, {"speed": float("nan")},
                    {"chunk_duration": 0}, {"chunk_duration": -1}):
        with pytest.raises(ValueError):
            await harness.feed_pcm(audio, **options)
