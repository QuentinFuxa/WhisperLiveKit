"""End-to-end check of the translation half through the benchmark's own code.

A fake harness stands in for the ASR/MT stack so the scoring and latency paths
in ``BenchmarkRunner._run_sample`` run unmodified: the real audio file is
hashed and validated against the manifest, and the harness produces the
``FrontData`` snapshots the pipeline would. The fake only replaces the model,
never the measurement code.
"""

import asyncio
import hashlib
import json
import wave
from types import SimpleNamespace

import pytest


def _write_silence(path, seconds=0.2):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(int(32000 * seconds)))


class FakeHarness:
    """Minimal stand-in for ``TestHarness`` with injected snapshots."""

    def __init__(self, snapshots, draft=False, translation_error="", target_language="fr"):
        self._snapshots = snapshots
        self._draft = draft
        self._translation_error = translation_error
        self.state = SimpleNamespace(committed_text="said", text="said")
        self.metrics = SimpleNamespace(total_processing_time_s=0.0, n_transcription_calls=0,
                                       n_tokens_produced=0, avg_latency_ms=None,
                                       p95_latency_ms=None)
        self._on_update = None
        # The runner records the engine options it ran with; a Namespace keeps
        # that path exercised rather than skipped.
        self._processor = SimpleNamespace(args=SimpleNamespace(
            target_language=target_language, lan="en", pcm_input=True,
        ))

    def on_update(self, callback):
        self._on_update = callback

    def load_audio(self, path):
        return SimpleNamespace(play=self._play)

    async def _play(self, speed=0):
        for snapshot in self._snapshots:
            self.state = snapshot
            if self._on_update:
                self._on_update(snapshot)
            await asyncio.sleep(0)

    async def finish(self, timeout=30.0):
        if self._translation_error:
            raise RuntimeError(self._translation_error)
        return self.state

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _state(lines, buffer_translation="", translation_error=""):
    return SimpleNamespace(
        lines=lines, buffer_translation=buffer_translation, text="said",
        committed_text="said", timing_valid=True, timing_monotonic=True,
        speech_lines=[line for line in lines if (line.get("translation") or "").strip()],
        translation_error=translation_error,
    )


def _run(monkeypatch, tmp_path, snapshots, draft, target_language="fr",
         translation_reference="un mot", expected_sha=None):
    from whisperlivekit.benchmark import runner as runner_module
    from whisperlivekit.benchmark.datasets import BenchmarkSample
    from whisperlivekit.metrics import compute_wer

    audio = tmp_path / "sample.wav"
    _write_silence(audio)
    sample = BenchmarkSample(
        name="en_1", path=str(audio), reference="said", duration=0.2, language="en",
        category="fleurs", expected_sha256=expected_sha or hashlib.sha256(
            audio.read_bytes()).hexdigest(),
        translation_reference=translation_reference, sentence_id=1,
    )

    harness = FakeHarness(snapshots, draft=draft, target_language=target_language)
    monkeypatch.setattr("whisperlivekit.test_harness.TestHarness", lambda **kwargs: harness)
    instance = runner_module.BenchmarkRunner(backend="whisper", speed=1,
                                             engine_kwargs={"target_language": target_language})
    return asyncio.run(instance._run_sample(
        sample, {"transcription": False, "vac": False, "pcm_input": True}, compute_wer))


def test_scoring_and_latency_with_a_drafting_backend(monkeypatch, tmp_path):
    from whisperlivekit.benchmark.chrf import compute_chrf

    draft_state = _state([{"text": "said", "translation": ""}], buffer_translation="un")
    final_state = _state([{"text": "said", "translation": "un mot"}])
    result = _run(monkeypatch, tmp_path, [draft_state, final_state], draft=True)

    assert result.status == "ok"
    assert result.translation_hypothesis == "un mot"
    assert result.chrf == pytest.approx(compute_chrf("un mot", "un mot"))
    # The complete translation is measured after feed start, and the draft was
    # visible before it, so a positive lead exists.
    assert result.final_translation_time_s is not None
    assert result.provisional_lead_s is not None and result.provisional_lead_s >= 0
    assert result.first_translation_time_s <= result.final_translation_time_s


def test_no_lead_is_claimed_when_the_backend_never_drafts(monkeypatch, tmp_path):
    """Only committed translations: no provisional lead to report."""
    from whisperlivekit.benchmark.chrf import compute_chrf

    committed_only = _state([{"text": "said", "translation": "un mot"}])
    result = _run(monkeypatch, tmp_path, [committed_only], draft=False)

    assert result.status == "ok"
    assert result.chrf == pytest.approx(compute_chrf("un mot", "un mot"))
    assert result.provisional_lead_s is None
    assert result.final_translation_time_s is not None


def test_rows_without_a_reference_stay_unscored(monkeypatch, tmp_path):
    committed = _state([{"text": "said", "translation": "un mot"}])
    result = _run(monkeypatch, tmp_path, [committed], draft=True, translation_reference="")

    assert result.status == "ok"
    assert result.translation_hypothesis == "un mot"
    assert result.chrf is None


def test_translation_errors_still_fail_the_sample(monkeypatch, tmp_path):
    broken = _state([{"text": "said", "translation": "un mot"}],
                    translation_error="capture failed")
    result = _run(monkeypatch, tmp_path, [broken], draft=False)

    assert result.status == "error"
    assert result.chrf is None


def test_a_tampered_audio_file_is_rejected_before_scoring(monkeypatch, tmp_path):
    committed = _state([{"text": "said", "translation": "un mot"}])
    result = _run(monkeypatch, tmp_path, [committed], draft=False,
                  expected_sha="0" * 64)

    assert result.status == "error"
    assert "SHA-256" in result.error
    assert result.chrf is None


def test_report_from_these_rows_carries_the_mt_configuration_and_corpus_chrf(
        monkeypatch, tmp_path):
    from whisperlivekit.benchmark.metrics import BenchmarkReport

    committed = _state([{"text": "said", "translation": "un mot"}])
    result = _run(monkeypatch, tmp_path, [committed], draft=False)
    report = BenchmarkReport("whisper", "base", results=[result],
                             translation_config={"target_language": "fr"})
    data = report.to_dict()

    assert data["benchmark_version"] == "3.2"
    assert data["translation"]["config"] == {"target_language": "fr"}
    assert data["translation"]["corpus_chrf_by_language"]["en"]["corpus_chrf"] == pytest.approx(100.0)
    assert data["summary"]["n_translation_scored"] == 1
    # The whole report must stay JSON-serialisable.
    json.dumps(data, allow_nan=False)
