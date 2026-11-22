"""Translation scoring and latency reporting for the benchmark.

The chrF values pinned here were computed once with sacrebleu 2.6.0 for
``sacrebleu.CHRF()`` (char order 6, word order 0, beta 2, no whitespace,
effective-order smoothing). sacrebleu stays a test-time reference only: it is
not a runtime dependency, so the implementation is checked against those
numbers instead of importing it.
"""

import io
import json

import pytest

# (hypothesis, reference, sacrebleu 2.6.0 sentence chrF)
SACREBLEU_VALUES = [
    ("the cat sat on a mat", "the cat sat on the mat", 65.97965990995549),
    ("hello world this is test", "hello world, this is a test.", 62.004049534810605),
    ("虽然没有人能确定作者是谁但众所周知这份大型羊皮纸文件在其生命早期曾被卷起存放",
     "虽然没有人能确定作者是谁，但众所周知，这份大型羊皮纸文件在其生命早期曾被卷起存放。",
     80.02777735753219),
]
# sacrebleu.CHRF().corpus_score(...) over the three rows above.
SACREBLEU_CORPUS = 72.35151511816518


@pytest.mark.parametrize("hypothesis,reference,expected", SACREBLEU_VALUES)
def test_chrf_matches_sacrebleu_per_row(hypothesis, reference, expected):
    from whisperlivekit.benchmark.chrf import compute_chrf

    assert compute_chrf(hypothesis, reference) == pytest.approx(expected)


def test_chrf_matches_sacrebleu_at_corpus_level():
    from whisperlivekit.benchmark.chrf import corpus_chrf

    pairs = [(hypothesis, reference) for hypothesis, reference, _ in SACREBLEU_VALUES]
    assert corpus_chrf(pairs) == pytest.approx(SACREBLEU_CORPUS)


def test_whitespace_is_removed_before_counting():
    from whisperlivekit.benchmark.chrf import compute_chrf

    assert compute_chrf("a b c", "abc") == 100.0


def test_empty_text_is_not_charged_as_a_translation_error():
    from whisperlivekit.benchmark.chrf import compute_chrf, corpus_chrf

    assert compute_chrf("", "abc") == 0.0
    # A corpus with no references at all has nothing to score.
    assert corpus_chrf([("abc", "")]) is None
    # An empty hypothesis against a reference is a real 0%, not a missing score.
    assert corpus_chrf([("", "abc")]) == 0.0


def test_corpus_chrf_weights_rows_by_their_statistics_not_by_row_average():
    """Summed statistics differ from a mean of row scores; sacrebleu sums."""
    from whisperlivekit.benchmark.chrf import compute_chrf, corpus_chrf

    pairs = [
        ("a", "a"),                       # a per-row 100% that covers one character
        ("zzzzzzzzzz", "yyyyyyyyyy"),      # a per-row 0% that covers ten
    ]
    mean_of_rows = sum(compute_chrf(*pair) for pair in pairs) / len(pairs)
    assert corpus_chrf(pairs) != pytest.approx(mean_of_rows)


def test_manifest_join_supplies_translation_references(tmp_path):
    from whisperlivekit.benchmark import datasets

    rows = [
        {"name": "en_1", "file": "en_1.wav", "reference": "one", "duration": 1,
         "language": "en", "category": "fleurs", "sample_rate": 16000,
         "source": "src", "audio_sha256": "A" * 64, "sentence_id": 1},
        {"name": "en_2", "file": "en_2.wav", "reference": "two", "duration": 1,
         "language": "en", "category": "fleurs", "sample_rate": 16000,
         "source": "src", "audio_sha256": "B" * 64, "sentence_id": 2},
        {"name": "fr_1", "file": "fr_1.wav", "reference": "un", "duration": 1,
         "language": "fr", "category": "fleurs", "sample_rate": 16000,
         "source": "src", "audio_sha256": "C" * 64, "sentence_id": 1},
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cache_subdir": "corpus", "samples": rows}))
    dataset_bytes = manifest.read_bytes()

    samples = datasets.load_manifest(str(manifest), target_language="fr")
    joined = {sample.name: sample.translation_reference for sample in samples}
    # The French row speaks the target language already, so it is left unscored
    # rather than scored against itself.
    assert joined == {"en_1": "un", "en_2": "", "fr_1": ""}
    # The join must not rewrite the pinned corpus.
    assert manifest.read_bytes() == dataset_bytes

    without_target = datasets.load_manifest(str(manifest))
    assert not any(sample.translation_reference for sample in without_target)


def test_manifest_join_requires_a_parallel_id_on_both_rows(tmp_path):
    from whisperlivekit.benchmark import datasets

    rows = [
        {"name": "en_1", "file": "en_1.wav", "reference": "one", "duration": 1,
         "language": "en", "category": "fleurs", "sample_rate": 16000,
         "source": "src", "audio_sha256": "A" * 64},
        {"name": "fr_1", "file": "fr_1.wav", "reference": "un", "duration": 1,
         "language": "fr", "category": "fleurs", "sample_rate": 16000,
         "source": "src", "audio_sha256": "B" * 64},
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cache_subdir": "corpus", "samples": rows}))
    samples = datasets.load_manifest(str(manifest), target_language="fr")
    assert not any(sample.translation_reference for sample in samples)


def _sample_result(**overrides):
    from whisperlivekit.benchmark.metrics import SampleResult

    values = dict(sample_name="en_1", language="en", category="fleurs", duration_s=1.0,
                  translation_hypothesis="un", translation_reference="un",
                  first_translation_time_s=1.0, final_translation_time_s=3.0,
                  provisional_lead_s=2.0)
    values.update(overrides)
    return SampleResult(**values)


def test_report_records_corpus_chrf_and_the_translation_config():
    from whisperlivekit.benchmark.metrics import BenchmarkReport, SampleResult

    results = [
        _sample_result(),
        _sample_result(sample_name="en_2", translation_hypothesis="deux",
                       translation_reference="deux"),
        # No reference: unscored, and it must not quietly count as 0%.
        SampleResult(sample_name="en_3", language="en", category="fleurs",
                     duration_s=1.0, translation_hypothesis="trois"),
    ]
    report = BenchmarkReport("whisper", "base", results=results,
                             translation_config={"target_language": "fr",
                                                 "translation_backend": "alignatt"})
    data = report.to_dict()
    assert data["benchmark_version"] == "3.2"
    assert data["translation"]["config"]["translation_backend"] == "alignatt"
    summary = data["summary"]
    assert summary["n_translation_scored"] == 2
    assert summary["corpus_chrf_by_language"]["en"]["n_rows"] == 2
    assert summary["corpus_chrf_by_language"]["en"]["corpus_chrf"] == pytest.approx(100.0)
    assert summary["final_translation_p95_s"] == pytest.approx(3.0)
    assert data["results"][2]["chrf"] is None


def test_report_prints_translation_rows_only_when_a_reference_exists():
    from whisperlivekit.benchmark.metrics import BenchmarkReport, SampleResult
    from whisperlivekit.benchmark.report import print_report

    asr_only = BenchmarkReport("whisper", "base",
                               results=[SampleResult("en_1", "en", "clean", 1.0,
                                                     hypothesis="one", reference="one")])
    out = io.StringIO()
    print_report(asr_only, out)
    assert "chrF" not in out.getvalue()

    translated = BenchmarkReport(
        "whisper", "base",
        results=[_sample_result()],
        translation_config={"target_language": "fr", "translation_backend": "alignatt"},
    )
    out = io.StringIO()
    print_report(translated, out)
    text = out.getvalue()
    assert "chrF" in text
    assert "Translation: fr / alignatt" in text
    assert "100.00% over 1 rows" in text
    assert "Complete translation: p95 3.00s" in text


def test_translation_hypothesis_is_the_validated_text_in_reading_order():
    from types import SimpleNamespace

    from whisperlivekit.benchmark.runner import translated_text, translation_config

    state = SimpleNamespace(speech_lines=[
        {"text": "hello", "translation": "bonjour"},
        {"text": "  ", "translation": None},
        {"text": "world", "translation": "monde "},
    ])
    assert translated_text(state) == "bonjour monde"
    # The buffer holds uncommitted text the reader never got as a translation.
    assert translation_config({"target_language": "fr", "translation_backend": "nllb"}) == {
        "target_language": "fr", "translation_backend": "nllb"}
    assert translation_config({"model_size": "base"}) == {}


@pytest.mark.asyncio
async def test_provisional_lead_is_null_unless_the_backend_drafts(tmp_path, monkeypatch):
    """A backend without drafts must not be credited with a provisional lead."""
    import wave

    from whisperlivekit.benchmark.datasets import BenchmarkSample
    from whisperlivekit.benchmark.runner import BenchmarkRunner
    from whisperlivekit.metrics import compute_wer
    from whisperlivekit.test_harness import TestHarness

    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(3200))
    sample = BenchmarkSample("silence", str(path), "", .1, "en", "clean",
                             translation_reference="ref")

    runner = BenchmarkRunner(backend="whisper", speed=1)
    config = dict(transcription=False, vac=False, pcm_input=True)

    async def no_artificial_drain(*args):
        raise AssertionError("A benchmark must finish at EOF, not sleep for an arbitrary drain")

    monkeypatch.setattr(TestHarness, "drain", no_artificial_drain)
    result = await runner._run_sample(sample, config, compute_wer)
    assert result.status == "ok"
    # No translation was produced, so nothing is scored and no lead is claimed.
    assert result.chrf is None
    assert result.provisional_lead_s is None
