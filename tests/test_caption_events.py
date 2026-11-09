"""Tests for the caption event stream contract.

Covered here: the ``CaptionEvent`` payload, the ``EventTap`` no-op and
forwarding behaviour, and the ``EventLog`` JSONL sink (a line per event,
appended and flushed as it is emitted). Display policy is out of scope —
it is a separate component with its own suite.
"""
import json
import logging

from whisperlivekit.caption_events import CaptionEvent, EventLog, EventTap


class _Recorder:
    """Minimal sink that records events in memory (test double)."""

    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestEventLog:
    def test_appends_one_line_per_event(self, tmp_path):
        path = tmp_path / "ev.jsonl"
        log = EventLog(str(path))
        log.emit(CaptionEvent(1.0, 2.0, "transcription_provisional", "hello"))
        log.emit(CaptionEvent(1.1, 2.1, "translation_final", "你好"))
        log.close()

        rows = _lines(path)
        assert len(rows) == 2
        assert rows[0]["type"] == "transcription_provisional"
        assert rows[1]["text"] == "你好"  # ensure_ascii=False keeps CJK readable

    def test_two_events_are_two_lines_before_close(self, tmp_path):
        """Lines are written as the events are emitted, not flushed at session
        cleanup: the file is complete while the sink is still open."""
        path = tmp_path / "ev.jsonl"
        log = EventLog(str(path))
        log.emit(CaptionEvent(0.0, 1.0, "transcription_final", "one"))
        assert [r["text"] for r in _lines(path)] == ["one"]
        log.emit(CaptionEvent(0.0, 2.0, "transcription_final", "two"))

        # read the file while the sink is still open
        raw = path.read_text(encoding="utf-8")
        assert raw.endswith("\n")
        assert len(raw.splitlines()) == 2
        assert [r["text"] for r in _lines(path)] == ["one", "two"]
        log.close()

    def test_appends_to_an_existing_file(self, tmp_path):
        path = tmp_path / "ev.jsonl"
        first = EventLog(str(path))
        first.emit(CaptionEvent(0.0, 1.0, "transcription_final", "first session"))
        first.close()

        second = EventLog(str(path))
        second.emit(CaptionEvent(0.0, 1.0, "transcription_final", "second session"))
        second.close()

        assert [r["text"] for r in _lines(path)] == ["first session", "second session"]

    def test_emit_after_close_is_ignored(self, tmp_path):
        """Session cleanup closes the log; a late emit must not raise."""
        log = EventLog(str(tmp_path / "ev.jsonl"))
        log.close()
        log.emit(CaptionEvent(0.0, 1.0, "transcription_final", "late"))

    def test_write_error_disables_log_without_raising(self, tmp_path, caplog):
        """A recording instrument must not end a live session."""

        class _BrokenFile:
            def write(self, _line):
                raise OSError("No space left on device")

            def close(self):
                pass

        log = EventLog(str(tmp_path / "ev.jsonl"))
        log._fh = _BrokenFile()
        with caplog.at_level(logging.WARNING):
            log.emit(CaptionEvent(0.0, 1.0, "transcription_final", "lost"))
        assert any("disabled" in r.message for r in caplog.records)
        # the handle is dropped, so later events do not retry the broken write
        assert log._fh is None


class TestEventTap:
    def test_noop_when_no_sink(self):
        """A tap with no sink is zero-cost: nothing emitted, no crash."""
        tap = EventTap()
        tap.transcription_provisional(1.0, "hello")
        tap.translation_final(2.0, "你好")

    def test_forwards_to_sink(self):
        rec = _Recorder()
        tap = EventTap(sink=rec, clock=lambda: 0.0)
        tap.transcription_provisional(1.0, "rolling")
        tap.translation_provisional(2.0, "prov", committed="c", source="s")
        assert [e.type for e in rec.events] == [
            "transcription_provisional", "translation_provisional",
        ]
        assert rec.events[1].committed == "c"
        assert rec.events[1].source == "s"

    def test_one_utterance_emits_the_typed_sequence_in_order(self):
        """One utterance driven through the tap: the caption sequence a
        consumer sees, in emission order, with each event's text, audio
        position and the source columns a provisional was drafted against.

        Fails if an emit method stops forwarding its text or audio position,
        if the emitted order changes, if `committed`/`source`/`fresh` are
        mis-threaded on a provisional, or if the blank-text filter starts
        swallowing a real event.
        """
        rec = _Recorder()
        tap = EventTap(sink=rec, clock=lambda: 42.0)

        # rolling source -> committed segment -> MT drafts against the
        # committed source -> finalized translation
        tap.transcription_provisional(0.8, "今天我們")
        tap.transcription_provisional(1.4, "今天我們討論皮膚科。")
        tap.transcription_final(1.4, "今天我們討論皮膚科。")
        tap.translation_provisional(
            1.5,
            "Today we",
            committed="今天我們",
            source="今天我們討論皮膚科。",
            fresh=True,
        )
        tap.translation_provisional(
            1.6,
            "Today we discuss",
            committed="今天我們討論",
            source="今天我們討論皮膚科。",
        )
        tap.translation_final(1.9, "Today we discuss dermatology.")

        assert [(e.type, e.text, e.audio_t) for e in rec.events] == [
            ("transcription_provisional", "今天我們", 0.8),
            ("transcription_provisional", "今天我們討論皮膚科。", 1.4),
            ("transcription_final", "今天我們討論皮膚科。", 1.4),
            ("translation_provisional", "Today we", 1.5),
            ("translation_provisional", "Today we discuss", 1.6),
            ("translation_final", "Today we discuss dermatology.", 1.9),
        ]
        assert [
            (e.committed, e.source, e.fresh)
            for e in rec.events
            if e.type == "translation_provisional"
        ] == [
            ("今天我們", "今天我們討論皮膚科。", True),
            ("今天我們討論", "今天我們討論皮膚科。", False),
        ]
        # one clock reading per event, stamped at emit time
        assert [e.t for e in rec.events] == [42.0] * 6

    def test_blank_text_is_not_emitted(self):
        rec = _Recorder()
        tap = EventTap(sink=rec)
        tap.transcription_provisional(1.0, "")
        tap.transcription_final(1.0, "   ")
        assert rec.events == []
