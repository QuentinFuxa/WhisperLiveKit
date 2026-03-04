"""Tests for whisperlivekit.timed_objects data classes."""

import pytest

from whisperlivekit.timed_objects import (
    ASRToken,
    FrontData,
    Segment,
    Silence,
    TimedText,
    Transcript,
    format_time,
)


class TestFormatTime:
    def test_zero(self):
        assert format_time(0) == "0:00:00"

    def test_one_minute(self):
        assert format_time(60) == "0:01:00"

    def test_one_hour(self):
        assert format_time(3600) == "1:00:00"

    def test_fractional_truncated(self):
        assert format_time(61.9) == "0:01:01"


class TestASRToken:
    def test_with_offset(self):
        t = ASRToken(start=1.0, end=2.0, text="hello")
        shifted = t.with_offset(0.5)
        assert shifted.start == pytest.approx(1.5)
        assert shifted.end == pytest.approx(2.5)
        assert shifted.text == "hello"

    def test_with_offset_preserves_fields(self):
        t = ASRToken(start=0.0, end=1.0, text="hi", speaker=2, probability=0.95)
        shifted = t.with_offset(1.0)
        assert shifted.speaker == 2
        assert shifted.probability == 0.95

    def test_is_silence_false(self):
        t = ASRToken(start=0.0, end=1.0, text="hello")
        assert t.is_silence() is False

    def test_bool_truthy(self):
        t = ASRToken(start=0.0, end=1.0, text="hello")
        assert bool(t) is True

    def test_bool_falsy(self):
        t = ASRToken(start=0.0, end=1.0, text="")
        assert bool(t) is False


class TestTimedText:
    def test_has_punctuation_period(self):
        t = TimedText(text="hello.")
        assert t.has_punctuation() is True

    def test_has_punctuation_exclamation(self):
        t = TimedText(text="wow!")
        assert t.has_punctuation() is True

    def test_has_punctuation_question(self):
        t = TimedText(text="really?")
        assert t.has_punctuation() is True

    def test_has_punctuation_cjk(self):
        t = TimedText(text="hello。")
        assert t.has_punctuation() is True

    def test_no_punctuation(self):
        t = TimedText(text="hello world")
        assert t.has_punctuation() is False

    def test_duration(self):
        t = TimedText(start=1.0, end=3.5)
        assert t.duration() == pytest.approx(2.5)

    def test_contains_timespan(self):
        outer = TimedText(start=0.0, end=5.0)
        inner = TimedText(start=1.0, end=3.0)
        assert outer.contains_timespan(inner) is True
        assert inner.contains_timespan(outer) is False


class TestSilence:
    def test_compute_duration(self):
        s = Silence(start=1.0, end=3.5)
        d = s.compute_duration()
        assert d == pytest.approx(2.5)
        assert s.duration == pytest.approx(2.5)

    def test_compute_duration_none_start(self):
        s = Silence(start=None, end=3.5)
        d = s.compute_duration()
        assert d is None

    def test_compute_duration_none_end(self):
        s = Silence(start=1.0, end=None)
        d = s.compute_duration()
        assert d is None

    def test_is_silence_true(self):
        s = Silence()
        assert s.is_silence() is True


class TestTranscript:
    def test_from_tokens(self, sample_tokens):
        t = Transcript.from_tokens(sample_tokens, sep="")
        assert t.text == "Hello world test."
        assert t.start == pytest.approx(0.0)
        assert t.end == pytest.approx(1.5)

    def test_from_tokens_with_sep(self, sample_tokens):
        t = Transcript.from_tokens(sample_tokens, sep="|")
        assert t.text == "Hello| world| test."

    def test_from_empty_tokens(self):
        t = Transcript.from_tokens([])
        assert t.text == ""
        assert t.start is None
        assert t.end is None

    def test_from_tokens_with_offset(self, sample_tokens):
        t = Transcript.from_tokens(sample_tokens, offset=10.0)
        assert t.start == pytest.approx(10.0)
        assert t.end == pytest.approx(11.5)


class TestSegment:
    def test_from_tokens(self, sample_tokens):
        seg = Segment.from_tokens(sample_tokens)
        assert seg is not None
        assert seg.text == "Hello world test."
        assert seg.start == pytest.approx(0.0)
        assert seg.end == pytest.approx(1.5)
        assert seg.speaker == -1

    def test_from_silence_tokens(self):
        silences = [
            Silence(start=1.0, end=2.0),
            Silence(start=2.0, end=3.0),
        ]
        seg = Segment.from_tokens(silences, is_silence=True)
        assert seg is not None
        assert seg.speaker == -2
        assert seg.is_silence() is True
        assert seg.text is None

    def test_from_empty_tokens(self):
        seg = Segment.from_tokens([])
        assert seg is None

    def test_to_dict(self, sample_tokens):
        seg = Segment.from_tokens(sample_tokens)
        d = seg.to_dict()
        assert "text" in d
        assert "speaker" in d
        assert "start" in d
        assert "end" in d


class TestFrontData:
    def test_to_dict_empty(self):
        fd = FrontData()
        d = fd.to_dict()
        assert d["lines"] == []
        assert d["buffer_transcription"] == ""
        assert "error" not in d

    def test_to_dict_with_error(self):
        fd = FrontData(error="something broke")
        d = fd.to_dict()
        assert d["error"] == "something broke"

    def test_to_dict_with_lines(self, sample_tokens):
        seg = Segment.from_tokens(sample_tokens)
        fd = FrontData(lines=[seg])
        d = fd.to_dict()
        assert len(d["lines"]) == 1
        assert d["lines"][0]["text"] == "Hello world test."
