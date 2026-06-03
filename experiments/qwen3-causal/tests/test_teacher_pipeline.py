from pathlib import Path

from scripts.annotate_teacher_transcripts import annotate_record
from scripts.filter_teacher_manifest import filter_records, teacher_filter_reason
from scripts.make_audio_manifest import audio_manifest_records


def test_filter_teacher_manifest_keeps_only_clean_low_wer_records():
    records = [
        {
            "audio": "a.wav",
            "source": "fleurs_en",
            "teacher_error": None,
            "teacher_wer": 0.1,
            "word_alignments": [{"text": "hello", "start_sec": 0.0, "end_sec": 0.2}],
        },
        {"audio": "b.wav", "teacher_error": None, "teacher_wer": 0.5},
        {"audio": "c.wav", "teacher_error": "boom", "teacher_wer": None},
        {"audio": "d.wav", "teacher_error": None, "teacher_wer": None},
    ]

    kept, summary = filter_records(records, max_teacher_wer=0.35)

    assert [record["audio"] for record in kept] == ["a.wav"]
    assert kept[0]["word_alignments"] == records[0]["word_alignments"]
    assert summary["kept"] == 1
    assert summary["reject_reasons"] == {
        "teacher_wer": 1,
        "teacher_error": 1,
        "missing_teacher_wer": 1,
    }


def test_teacher_filter_allows_missing_wer_for_reference_free_eval_records():
    record = {"audio": "wlk.wav", "teacher_error": None, "teacher_wer": None}

    assert teacher_filter_reason(record, max_teacher_wer=0.35) == "missing_teacher_wer"
    assert (
        teacher_filter_reason(record, max_teacher_wer=0.35, allow_missing_wer=True)
        is None
    )


def test_annotate_record_adds_teacher_fields_with_fake_transcriber(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake")

    def fake_transcribe(path: Path, language: str | None):
        assert path == audio
        assert language == "en"
        return "hello world", {"text": "hello world"}

    record = {
        "audio": str(audio),
        "text": "hello brave world",
        "language_code": "en",
    }

    annotated = annotate_record(
        record,
        model="teacher",
        transcribe_fn=fake_transcribe,
    )

    assert annotated["teacher_text"] == "hello world"
    assert annotated["teacher_model"] == "teacher"
    assert annotated["teacher_error"] is None
    assert annotated["teacher_wer"] is not None
    assert "text" in annotated


def test_audio_manifest_records_are_absolute_and_eval_only(tmp_path):
    (tmp_path / "b.wav").write_bytes(b"")
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "skip.txt").write_text("x", encoding="utf-8")

    records = audio_manifest_records(tmp_path, glob="*.wav", source="wlk_audio")

    assert [record["id"] for record in records] == ["a", "b"]
    assert all(Path(str(record["audio"])).is_absolute() for record in records)
    assert all(record["source"] == "wlk_audio" for record in records)
    assert all("text" not in record for record in records)
