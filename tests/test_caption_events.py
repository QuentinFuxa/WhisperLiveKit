"""Tests for the caption event stream contract.

Covered here: the ``CaptionEvent`` payload, the ``EventTap`` no-op and
forwarding behaviour, the ``EventLog`` JSONL sink (a line per event, appended
and flushed as it is emitted, tagged with the writing session's id), and the
startup check that rejects an unwritable ``--event-log`` path. Display policy
is out of scope — it is a separate component with its own suite.
"""
import asyncio
import json
import logging
import sys
from types import SimpleNamespace

import pytest

from whisperlivekit.caption_events import (
    CaptionEvent,
    EventLog,
    EventTap,
    validate_event_log_path,
)


class _Recorder:
    """Minimal sink that records events in memory (test double)."""

    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class _SteppingClock:
    """Monotonic test clock: every read advances by ``step`` seconds."""

    def __init__(self, start=100.0, step=0.5):
        self.value = start
        self.step = step

    def __call__(self):
        reading = self.value
        self.value += self.step
        return reading


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

    def test_unwritable_path_raises_value_error_naming_the_path(self, tmp_path):
        """A bad path is a configuration error, not a raw FileNotFoundError
        escaping through the per-session factory."""
        missing = tmp_path / "missing-dir" / "ev.jsonl"
        with pytest.raises(ValueError) as excinfo:
            EventLog(str(missing))
        message = str(excinfo.value)
        assert "not writable" in message
        assert str(missing) in message

    def test_line_carries_its_session_id(self, tmp_path):
        """Two sessions sharing one path interleave, and each row names the
        session that wrote it."""
        path = tmp_path / "ev.jsonl"
        first = EventLog(str(path))
        second = EventLog(str(path))
        first.emit(CaptionEvent(0.0, 1.0, "transcription_final", "one"))
        second.emit(CaptionEvent(0.0, 1.0, "transcription_final", "two"))
        first.close()
        second.close()

        rows = _lines(path)
        assert rows[0]["session"] == first.session
        assert rows[1]["session"] == second.session
        assert rows[0]["session"] != rows[1]["session"]
        assert len(rows[0]["session"]) == 8

    def test_source_columns_only_on_translation_provisional(self, tmp_path):
        """``committed``/``source``/``fresh`` describe a draft's source gating;
        the other three event types must not carry empty columns."""
        path = tmp_path / "ev.jsonl"
        log = EventLog(str(path))
        log.emit(CaptionEvent(0.0, 1.0, "transcription_provisional", "今天我們"))
        log.emit(CaptionEvent(0.1, 1.1, "transcription_final", "今天我們討論皮膚科。"))
        log.emit(CaptionEvent(0.2, 1.2, "translation_provisional", "Today we",
                              "今天我們", "今天我們討論皮膚科。", True))
        log.emit(CaptionEvent(0.3, 1.3, "translation_final", "Today we discuss dermatology."))
        log.close()

        rows = _lines(path)
        base = {"session", "t", "audio_t", "type", "text"}
        assert [set(row) for row in rows[:2]] == [base, base]
        assert set(rows[2]) == base | {"committed", "source", "fresh"}
        assert (rows[2]["committed"], rows[2]["source"], rows[2]["fresh"]) == (
            "今天我們", "今天我們討論皮膚科。", True,
        )
        assert set(rows[3]) == base

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

    def test_inert_tap_reports_itself_inactive(self):
        """``active`` is what call sites check before building a payload, so
        it must follow the sink, not a configuration flag."""
        assert EventTap().active is False
        assert EventTap(sink=_Recorder()).active is True

    def test_t_is_seconds_since_the_stream_started(self):
        """``t`` is measured from the tap's creation, not the raw process
        clock: a stream starts at zero."""
        rec = _Recorder()
        clock = _SteppingClock(start=100.0, step=0.5)
        tap = EventTap(sink=rec, clock=clock)
        tap.transcription_final(1.0, "one")
        tap.transcription_final(2.0, "two")
        assert [e.t for e in rec.events] == [0.5, 1.0]

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
        tap = EventTap(sink=rec, clock=_SteppingClock(start=100.0, step=0.5))

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
        # one clock reading per event, stamped at emit time as the offset
        # from the stream's start (the tap's creation)
        assert [e.t for e in rec.events] == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    def test_blank_text_is_not_emitted(self):
        rec = _Recorder()
        tap = EventTap(sink=rec)
        tap.transcription_provisional(1.0, "")
        tap.transcription_final(1.0, "   ")
        assert rec.events == []


class TestEventLogStartupCheck:
    def test_accepts_a_writable_path(self, tmp_path):
        path = tmp_path / "ev.jsonl"
        validate_event_log_path(str(path))
        assert path.exists()

    def test_no_path_configured_is_not_an_error(self):
        validate_event_log_path(None)

    def test_unwritable_path_raises_value_error_naming_the_path(self, tmp_path):
        missing = tmp_path / "missing-dir" / "ev.jsonl"
        with pytest.raises(ValueError) as excinfo:
            validate_event_log_path(str(missing))
        message = str(excinfo.value)
        assert "--event-log path" in message
        assert str(missing) in message

    def test_server_startup_rejects_an_unwritable_path(self, tmp_path, monkeypatch):
        """The check runs in the app's lifespan, so a bad path stops the server
        at startup instead of failing every connection as it opens the log."""
        monkeypatch.setattr(sys, "argv", ["whisperlivekit-server"])
        from whisperlivekit import basic_server
        from whisperlivekit.config import WhisperLiveKitConfig

        # the engine load is not what this test is about
        monkeypatch.setattr(
            basic_server, "TranscriptionEngine",
            lambda **kwargs: SimpleNamespace(args=kwargs["config"], asr=None),
        )
        app = basic_server.create_app(
            WhisperLiveKitConfig(event_log=str(tmp_path / "missing-dir" / "ev.jsonl"))
        )

        async def startup():
            async with basic_server.lifespan(app):
                pass

        with pytest.raises(ValueError, match="not writable"):
            asyncio.run(startup())
