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


def test_chrf_matches_sacrebleu_on_pinned_inputs():
    """chrF is sacrebleu's CHRF(char_order=6, word_order=0, beta=2), value for value.

    Each expected value below was computed once with sacrebleu 2.6.0 and
    hard-coded, so the suite checks against the reference implementation
    without importing it.
    """
    from whisperlivekit.metrics import compute_chrf

    assert compute_chrf("Hello world", "Hello world") == pytest.approx(100.0)
    assert compute_chrf("Hello world", "") == 0.0
    assert compute_chrf("", "Hello world") == 0.0
    assert compute_chrf("abc", "abd") == pytest.approx(38.888889)
    # Precision equal to recall is the same value for any beta, so that pair
    # cannot pin the exponent. This one separates them: a two-character
    # reference is wholly contained in "abc" (recall 1) while precision
    # averages to 7/12 over its two orders, so F-2 is 87.5, and 80.77 if beta
    # stands in for beta squared.
    assert compute_chrf("ab", "abc") == pytest.approx(87.5)
    # sacrebleu removes whitespace and nothing else. Case and punctuation are
    # part of the score, which is what makes the number comparable; the old
    # WER-normalized variant scored both of these inputs 100.0.
    assert compute_chrf("The Cat Sat", "the cat sat") == pytest.approx(17.361111)
    assert compute_chrf("你好世界。", "你好，世界！") == pytest.approx(24.910180)
    # Whitespace is the one thing removed, so layout cannot move the score and a
    # character language needs no tokenizer.
    assert compute_chrf("Hello world", "Hello  world") == 100.0
    assert compute_chrf("你好世界。", "你好世界") == pytest.approx(72.573464)
    # A metric that cannot rank two translations would pass every check above.
    assert compute_chrf("你好世界", "你好") < compute_chrf("你好世界", "你好世界")


def test_corpus_chrf_sums_ngram_counts_rather_than_averaging_rows():
    """The report's chrF is sacrebleu's corpus score, not the mean of per-row scores."""
    from whisperlivekit.benchmark.metrics import BenchmarkReport, SampleResult
    from whisperlivekit.metrics import compute_chrf

    pairs = [("Hello world.", "Hello word."),
             ("你好世界。", "你好，世界！"),
             ("The cat sat on the mat.", "the cat sat on a mat")]
    report = BenchmarkReport("whisper", "tiny", translation={"target_language": "zh"}, results=[
        SampleResult(sample_name=f"row{index}", language="en", category="fleurs", duration_s=1.0,
                     target_language="zh", translation=hypothesis, translation_reference=reference,
                     chrf=compute_chrf(reference, hypothesis))
        for index, (reference, hypothesis) in enumerate(pairs)])
    # sacrebleu 2.6.0 corpus score over the same three segments, and the mean of
    # the same three per-sentence scores: 59.094 against 51.049.
    assert report.corpus_chrf == pytest.approx(59.093996)
    assert report.avg_chrf == pytest.approx(51.048780)
    assert report.to_dict()["summary"]["corpus_chrf"] == pytest.approx(59.093996)
    # A row with no translation is not part of the pair and must not dilute it.
    report.results.append(SampleResult(sample_name="asr only", language="en", category="fleurs",
                                       duration_s=1.0))
    assert report.corpus_chrf == pytest.approx(59.093996)
    # Nothing scored at all is null rather than zero.
    assert BenchmarkReport("whisper", "tiny").corpus_chrf is None


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
    sample, runner, compute_wer = _cascade_run(monkeypatch, tmp_path)
    from whisperlivekit.benchmark.metrics import BenchmarkReport, describe_translation_config

    config = dict(transcription=False, pcm_input=True, target_language="zh",
                  translation_backend="mlx-llm-mt", mlx_llm_mt_model="hy-mt2-1.8b-8bit",
                  mlx_llm_mt_simultaneous=True)
    result = await runner._run_sample(sample, config, compute_wer)
    assert result.status == "ok"
    assert result.translation == "你好 世界"
    assert result.translation_reference == "你好世界"
    # Whitespace is all that separates them, and chrF removes whitespace.
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
                             translation=describe_translation_config(config, ["en"]))
    data = report.to_dict()
    assert data["benchmark_version"] == "3.2"
    # Both halves of the run are named: the recognizer and the translator.
    assert data["config"]["asr"] == {"backend": "whisper", "model": "tiny"}
    assert data["config"]["translation"] == {"source_languages": ["en"], "target_language": "zh",
                                             "backend": "mlx-llm-mt",
                                             "model": "hy-mt2-1.8b-8bit", "simultaneous": True}
    assert data["summary"]["corpus_chrf"] == pytest.approx(100.0)
    assert data["summary"]["avg_chrf"] == pytest.approx(100.0)
    assert describe_translation_config({})["backend"] == ""


@pytest.mark.asyncio
async def test_a_non_drafting_backend_leaves_the_provisional_lead_null(tmp_path, monkeypatch):
    """Untranslated source text in the buffer is not a draft, so there is no lead."""
    sample, runner, compute_wer = _cascade_run(monkeypatch, tmp_path)
    from whisperlivekit.benchmark.metrics import drafts_provisional_text

    # The fake run carries source text in the buffer while the backend returns
    # completed segments only, which is what mlx-llm-mt does without the
    # simultaneous switch. Both timing endpoints exist, so a lead would be
    # computable: null has to come from the backend, not from missing times.
    config = dict(transcription=False, pcm_input=True, target_language="zh",
                  translation_backend="mlx-llm-mt", mlx_llm_mt_model="hy-mt2-1.8b-8bit")
    assert not drafts_provisional_text(config)
    result = await runner._run_sample(sample, config, compute_wer)
    assert result.status == "ok"
    assert result.first_translation_time_s is not None and result.final_translation_time_s is not None
    assert result.final_translation_time_s > result.first_translation_time_s
    assert result.provisional_lead_s is None
    # The sidecar backend drafts, which is the answer the lead is gated on.
    assert drafts_provisional_text({**config, "translation_backend": "alignatt"})


@pytest.mark.asyncio
async def test_the_nllb_backend_reports_a_measured_provisional_lead(tmp_path, monkeypatch):
    """The default backend drafts, so its buffer yields a lead rather than null."""
    sample, runner, compute_wer = _cascade_run(monkeypatch, tmp_path)
    from whisperlivekit.benchmark.metrics import drafts_provisional_text

    config = dict(transcription=False, pcm_input=True, target_language="zh",
                  translation_backend="nllb", nllb_backend="transformers", nllb_size="600M")
    assert drafts_provisional_text(config)
    result = await runner._run_sample(sample, config, compute_wer)
    assert result.status == "ok"
    assert result.first_translation_time_s is not None and result.final_translation_time_s is not None
    # Same two endpoints as the non-drafting run above, so the difference is the backend.
    assert result.provisional_lead_s == pytest.approx(
        result.final_translation_time_s - result.first_translation_time_s)
    assert result.provisional_lead_s > 0


@pytest.mark.asyncio
async def test_a_run_that_names_no_backend_reports_the_default_ones_lead(tmp_path, monkeypatch):
    """`wlk bench --target-language zh` names no backend, and the default drafts.

    The requested kwargs cannot answer whether the run drafted: the backend
    comes from WhisperLiveKitConfig, which fills in nllb. Reading the request
    instead of the resolved configuration reported null beside a recorded
    backend of nllb, which the README reads as a run that did not draft.
    """
    sample, runner, compute_wer = _cascade_run(monkeypatch, tmp_path)

    config = dict(transcription=False, pcm_input=True, target_language="zh")
    assert "translation_backend" not in config
    result = await runner._run_sample(sample, config, compute_wer)
    assert result.status == "ok"
    assert result.effective_config["translation_backend"] == "nllb"
    assert result.first_translation_time_s is not None and result.final_translation_time_s is not None
    assert result.provisional_lead_s == pytest.approx(
        result.final_translation_time_s - result.first_translation_time_s)
    assert result.provisional_lead_s > 0


def test_the_bench_translation_flag_rejects_a_name_the_engine_would_not_route(capsys):
    """A misspelt `wlk bench --translation-backend` must not run nllb silently.

    The engine routes only the names in `TRANSLATION_BACKENDS` and sends every
    other one down the nllb branch, so before the bench flag had choices an
    unlisted name ran nllw and reported that translator's lead.
    """
    from whisperlivekit.cli import cmd_bench

    with pytest.raises(SystemExit) as exit_info:
        cmd_bench(["--target-language", "zh", "--translation-backend", "nllw"])
    assert exit_info.value.code == 2
    assert "nllw" in capsys.readouterr().err


def test_translation_config_records_what_an_alignatt_run_can_know():
    """AC-4 beyond one backend: the alignatt record names the endpoint it called."""
    from whisperlivekit.benchmark.metrics import describe_translation_config, drafts_provisional_text

    config = {"translation_backend": "alignatt", "target_language": "zh",
              "alignatt_url": "ws://localhost:8765", "alignatt_preset": "quality",
              "alignatt_latency": "balanced"}
    record = describe_translation_config(config, ["en", "en"])
    assert record["backend"] == "alignatt"
    assert record["source_languages"] == ["en"] and record["target_language"] == "zh"
    # The model lives in the server; the record says so rather than claiming one.
    assert record["model"] is None
    assert "server" in record["model_note"]
    assert record["server"] == {"url": "ws://localhost:8765", "preset": "quality",
                                "latency": "balanced"}
    # The same drafting answer the runner uses for the lead, per backend.
    assert drafts_provisional_text(config)
    assert drafts_provisional_text({**config, "translation_backend": "nllb"})
    nllb = describe_translation_config({"translation_backend": "nllb", "target_language": "zh",
                                        "nllb_size": "600M"}, ["en"])
    assert nllb["model"] is None and "nllb_size" in nllb["model_note"]


def _cascade_run(monkeypatch, tmp_path):
    """Fake TestHarness whose observed states follow the configured MT backend.

    A drafting backend puts a partial translation in the buffer while audio is
    still arriving; a segment backend returns completed segments and still
    leaves untranslated source text in the buffer. Both cases are what the real
    backends do, so the runner is exercised against the buffer contents it will
    actually see rather than against an empty one.
    """
    from types import SimpleNamespace

    import whisperlivekit.test_harness as testing
    from whisperlivekit.benchmark.datasets import BenchmarkSample
    from whisperlivekit.benchmark.runner import BenchmarkRunner
    from whisperlivekit.config import WhisperLiveKitConfig
    from whisperlivekit.metrics import compute_wer

    def state(lines, buffer_translation=""):
        return SimpleNamespace(
            lines=lines, buffer_translation=buffer_translation,
            committed_text=" ".join(line["text"] for line in lines),
            text=" ".join(line["text"] for line in lines),
            translation_error="", timing_valid=True, timing_monotonic=True,
            speech_lines=lines,
        )

    hello, world = {"start": "0:00:00.00", "text": "Hello"}, {"start": "0:00:02.00", "text": "World"}

    def observed_states(config):
        if not config.get("target_language"):
            return [state([hello]), state([hello, world])]
        # Branch on the backend's own configuration — its simultaneous switch or
        # its name — not on the predicate under test.
        if config.get("mlx_llm_mt_simultaneous") or config.get("translation_backend") in ("nllb", "nllw"):
            return [
                state([{**hello, "translation": ""}], buffer_translation="你"),
                state([{**hello, "translation": "你好"}]),
                state([{**hello, "translation": "你好"}, {**world, "translation": "世界"}]),
            ]
        return [
            state([{**hello, "translation": ""}], buffer_translation="Hello"),
            state([{**hello, "translation": "你好"}, {**world, "translation": "世界"}]),
        ]

    class FakeHarness:
        def __init__(self, **kwargs):
            # The real harness records `asdict(WhisperLiveKitConfig)`, which
            # names an MT backend even when the caller never asked for one, so
            # the fake resolves those two keys instead of echoing kwargs only.
            resolved = {"translation_backend": WhisperLiveKitConfig.translation_backend,
                        "mlx_llm_mt_simultaneous": False, **kwargs}
            self._states = observed_states(resolved)
            self.state = self._states[0]
            self.metrics = SimpleNamespace(total_processing_time_s=.1, avg_latency_ms=1., p95_latency_ms=2.,
                                           n_transcription_calls=1, n_tokens_produced=1)
            self._processor = SimpleNamespace(args=SimpleNamespace(**resolved))
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
                             sentence_id=1, translation_references={"en": "Hello World.", "zh": "你好世界"})
    return sample, BenchmarkRunner(backend="whisper", speed=1), compute_wer
