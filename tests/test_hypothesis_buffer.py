"""Tests for HypothesisBuffer — the core of LocalAgreement policy."""

import pytest

from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.local_agreement.online_asr import HypothesisBuffer


def make_tokens(words, start=0.0, step=0.5):
    """Helper: create ASRToken list from word strings."""
    tokens = []
    t = start
    for w in words:
        tokens.append(ASRToken(start=t, end=t + step, text=w, probability=0.9))
        t += step
    return tokens


class TestInsert:
    def test_basic_insert(self):
        buf = HypothesisBuffer()
        tokens = make_tokens(["hello", "world"])
        buf.insert(tokens, offset=0.0)
        assert len(buf.new) == 2
        assert buf.new[0].text == "hello"

    def test_insert_with_offset(self):
        buf = HypothesisBuffer()
        tokens = make_tokens(["hello"], start=0.0)
        buf.insert(tokens, offset=5.0)
        assert buf.new[0].start == pytest.approx(5.0)

    def test_insert_filters_old_tokens(self):
        buf = HypothesisBuffer()
        buf.last_committed_time = 10.0
        tokens = make_tokens(["old", "new"], start=5.0, step=3.0)
        buf.insert(tokens, offset=0.0)
        # "old" at 5.0 is before last_committed_time - 0.1 = 9.9 → filtered
        # "new" at 8.0 is also before 9.9 → filtered
        assert len(buf.new) == 0

    def test_insert_deduplicates_committed(self):
        buf = HypothesisBuffer()
        # Commit "hello"
        tokens1 = make_tokens(["hello", "world"])
        buf.insert(tokens1, offset=0.0)
        buf.flush()  # commits "hello" (buffer was empty, so nothing matches)
        # Actually with empty buffer, flush won't commit anything
        # Let's do it properly: two rounds
        buf2 = HypothesisBuffer()
        first = make_tokens(["hello", "world"])
        buf2.insert(first, offset=0.0)
        buf2.flush()  # buffer was empty → no commits, buffer = ["hello", "world"]

        second = make_tokens(["hello", "world", "test"])
        buf2.insert(second, offset=0.0)
        committed = buf2.flush()
        # LCP of ["hello", "world"] and ["hello", "world", "test"] = ["hello", "world"]
        assert len(committed) == 2
        assert committed[0].text == "hello"
        assert committed[1].text == "world"


class TestFlush:
    def test_flush_empty(self):
        buf = HypothesisBuffer()
        committed = buf.flush()
        assert committed == []

    def test_flush_lcp_matching(self):
        buf = HypothesisBuffer()
        # Round 1: establish buffer
        buf.insert(make_tokens(["hello", "world"]), offset=0.0)
        buf.flush()  # buffer = ["hello", "world"], committed = []

        # Round 2: same prefix, new suffix
        buf.insert(make_tokens(["hello", "world", "test"]), offset=0.0)
        committed = buf.flush()
        assert [t.text for t in committed] == ["hello", "world"]

    def test_flush_no_match(self):
        buf = HypothesisBuffer()
        # Round 1
        buf.insert(make_tokens(["hello", "world"]), offset=0.0)
        buf.flush()

        # Round 2: completely different
        buf.insert(make_tokens(["foo", "bar"]), offset=0.0)
        committed = buf.flush()
        assert committed == []

    def test_flush_partial_match(self):
        buf = HypothesisBuffer()
        buf.insert(make_tokens(["hello", "world", "test"]), offset=0.0)
        buf.flush()

        buf.insert(make_tokens(["hello", "earth", "again"]), offset=0.0)
        committed = buf.flush()
        assert len(committed) == 1
        assert committed[0].text == "hello"

    def test_flush_updates_last_committed(self):
        buf = HypothesisBuffer()
        buf.insert(make_tokens(["hello", "world"]), offset=0.0)
        buf.flush()

        buf.insert(make_tokens(["hello", "world", "test"]), offset=0.0)
        buf.flush()
        assert buf.last_committed_word == "world"
        assert buf.last_committed_time > 0

    def test_flush_with_confidence_validation(self):
        buf = HypothesisBuffer(confidence_validation=True)
        high_conf = [
            ASRToken(start=0.0, end=0.5, text="sure", probability=0.99),
            ASRToken(start=0.5, end=1.0, text="maybe", probability=0.5),
        ]
        buf.insert(high_conf, offset=0.0)
        committed = buf.flush()
        # "sure" has p>0.95 → committed immediately
        assert len(committed) == 1
        assert committed[0].text == "sure"


class TestPopCommitted:
    def test_pop_removes_old(self):
        buf = HypothesisBuffer()
        buf.committed_in_buffer = make_tokens(["a", "b", "c"], start=0.0, step=1.0)
        # "a": end=1.0, "b": end=2.0, "c": end=3.0
        # pop_committed removes tokens with end <= time
        buf.pop_committed(2.0)
        # "a" (end=1.0) and "b" (end=2.0) removed, "c" (end=3.0) remains
        assert len(buf.committed_in_buffer) == 1
        assert buf.committed_in_buffer[0].text == "c"

    def test_pop_nothing(self):
        buf = HypothesisBuffer()
        buf.committed_in_buffer = make_tokens(["a", "b"], start=5.0)
        buf.pop_committed(0.0)
        assert len(buf.committed_in_buffer) == 2

    def test_pop_all(self):
        buf = HypothesisBuffer()
        buf.committed_in_buffer = make_tokens(["a", "b"], start=0.0, step=0.5)
        buf.pop_committed(100.0)
        assert len(buf.committed_in_buffer) == 0


class TestStreamingSimulation:
    """Multi-round insert/flush simulating real streaming behavior."""

    def test_three_rounds(self):
        buf = HypothesisBuffer()
        all_committed = []

        # Round 1: "this is"
        buf.insert(make_tokens(["this", "is"]), offset=0.0)
        all_committed.extend(buf.flush())

        # Round 2: "this is a test"
        buf.insert(make_tokens(["this", "is", "a", "test"]), offset=0.0)
        all_committed.extend(buf.flush())

        # Round 3: "this is a test today"
        buf.insert(make_tokens(["this", "is", "a", "test", "today"]), offset=0.0)
        all_committed.extend(buf.flush())

        words = [t.text for t in all_committed]
        assert "this" in words
        assert "is" in words
        assert "a" in words
        assert "test" in words
