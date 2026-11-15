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


def test_chrf_scores_a_translation_without_a_model():
    from whisperlivekit.metrics import compute_chrf

    assert compute_chrf("Hello world", "Hello world") == 100.0
    assert compute_chrf("Hello world", "") == 0.0
    assert compute_chrf("", "Hello world") == 0.0
    # Orders are averaged over the n-grams a three-character text has: orders
    # 1 and 2 match once of twice each, order 3 never; F-beta is 38.89 at beta=2.
    assert compute_chrf("abc", "abd") == pytest.approx(38.8889, abs=1e-4)
    # Precision equal to recall is the same value for any beta, so that pair
    # cannot pin the exponent. This one separates them: a two-character
    # reference is wholly contained in "abc" (recall 1) while precision
    # averages to 7/12 over its two orders, so F-2 is 87.5, and 80.77 if beta
    # stands in for beta squared.
    assert compute_chrf("ab", "abc") == pytest.approx(87.5, abs=1e-4)
    # Formatting cannot move the score, and Chinese needs no tokenizer.
    assert compute_chrf("Hello  world", "hello world") == 100.0
    assert compute_chrf("你好世界。", "你好，世界！") == 100.0
    # A metric that cannot rank two translations would pass every check above.
    assert compute_chrf("你好世界", "你好") < compute_chrf("你好世界", "你好世界")


def test_manifest_sentence_id_join_supplies_the_translation_reference(tmp_path):
    from whisperlivekit.benchmark.datasets import load_manifest

    def row(name, language, reference, sentence_id=None):
        sample = {"name": name, "file": f"{name}.wav", "reference": reference, "duration": 1.0,
                  "language": language, "category": "fleurs", "sample_rate": 16000,
                  "source": "test", "audio_sha256": "0" * 64}
        if sentence_id is not None:
            sample["sentence_id"] = sentence_id
        return sample

    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({"cache_subdir": "test", "samples": [
        row("en_1", "en", "Hello.", 1), row("zh_1", "zh", "你好。", 1),
        row("en_2", "en", "Orphan.", 2), row("unkeyed", "en", "No id."),
    ]}))
    samples = {sample.name: sample for sample in load_manifest(manifest)}
    assert samples["en_1"].sentence_id == 1
    assert samples["en_1"].translation_references == {"en": "Hello.", "zh": "你好。"}
    # No target-language sibling and no key at all must stay empty, not raise.
    assert samples["en_2"].translation_references == {"en": "Orphan."}
    assert samples["unkeyed"].translation_references == {}


@pytest.mark.asyncio
async def test_translated_rows_carry_chrf_and_derived_translation_latency(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import whisperlivekit.test_harness as testing
    from whisperlivekit.benchmark.datasets import BenchmarkSample
    from whisperlivekit.benchmark.metrics import BenchmarkReport, describe_translation_config
    from whisperlivekit.benchmark.runner import BenchmarkRunner
    from whisperlivekit.metrics import compute_wer

    def state(lines, buffer_translation=""):
        return SimpleNamespace(
            lines=lines, buffer_translation=buffer_translation,
            committed_text=" ".join(line["text"] for line in lines),
            text=" ".join(line["text"] for line in lines),
            translation_error="", timing_valid=True, timing_monotonic=True,
            speech_lines=lines,
        )

    # Each state is delivered while audio is still being fed, so the runner has
    # to timestamp the first and the last committed line itself. A run without a
    # target language never receives provisional or committed translation text.
    hello, world = {"start": "0:00:00.00", "text": "Hello"}, {"start": "0:00:02.00", "text": "World"}

    def observed_states(translating):
        if not translating:
            return [state([hello]), state([hello, world])]
        return [
            state([{**hello, "translation": ""}], buffer_translation="你"),
            state([{**hello, "translation": "你好"}]),
            state([{**hello, "translation": "你好"}, {**world, "translation": "世界"}]),
        ]

    class FakeHarness:
        def __init__(self, **kwargs):
            self._states = observed_states(bool(kwargs.get("target_language")))
            self.state = self._states[0]
            self.metrics = SimpleNamespace(total_processing_time_s=.1, avg_latency_ms=1., p95_latency_ms=2.,
                                           n_transcription_calls=1, n_tokens_produced=1)
            self._processor = SimpleNamespace(args=SimpleNamespace(pcm_input=True))
            self._on_update = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exception):
            return False

        def on_update(self, callback):
            self._on_update = callback

        def load_audio(self, path):
            harness = self

            class Player:
                async def play(self, speed=0):
                    for observed in harness._states:
                        harness.state = observed
                        harness._on_update(observed)

            return Player()

        async def finish(self, timeout=0):
            self.state = self._states[-1]
            return self.state

    monkeypatch.setattr(testing, "TestHarness", FakeHarness)
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(3200))
    sample = BenchmarkSample("fleurs_en_1", str(path), "Hello World.", .1, "en", "fleurs",
                             sentence_id=1, translation_references={"en": "Hello World.", "zh": "你好世界。"})
    config = dict(transcription=False, pcm_input=True, target_language="zh",
                  translation_backend="mlx-llm-mt", mlx_llm_mt_model="hy-mt2-1.8b-8bit")
    runner = BenchmarkRunner(backend="whisper", speed=1)
    result = await runner._run_sample(sample, config, compute_wer)
    assert result.status == "ok"
    assert result.translation == "你好 世界"
    assert result.translation_reference == "你好世界。"
    assert result.chrf == pytest.approx(100.0)
    assert result.first_translation_time_s is not None and result.final_translation_time_s is not None
    assert result.final_translation_time_s >= result.first_translation_time_s
    # The drafting state carried provisional text, so the lead is measurable.
    assert result.provisional_lead_s is not None

    # Without a target language the pair is not joined and nothing is scored.
    asr_only = await runner._run_sample(sample, dict(transcription=False, pcm_input=True), compute_wer)
    assert asr_only.target_language == "" and asr_only.translation_reference == ""
    assert asr_only.chrf is None and asr_only.provisional_lead_s is None

    report = BenchmarkReport("whisper", "tiny", results=[result],
                             translation=describe_translation_config(config))
    data = report.to_dict()
    assert data["benchmark_version"] == "3.2"
    assert data["config"]["translation"] == {"target_language": "zh", "backend": "mlx-llm-mt",
                                             "model": "hy-mt2-1.8b-8bit", "simultaneous": False}
    assert data["summary"]["avg_chrf"] == pytest.approx(100.0)
    assert describe_translation_config({})["backend"] == ""
