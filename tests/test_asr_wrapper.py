"""Tests for the generalized ASR wrapper layer.

These tests exercise the factored modules (asr_commit, asr_timestamps,
asr_wrapper) without requiring mlx, torch, or model downloads. They
validate:

- Job 1 (stable/unstable split): cross-pass agreement + hold-back
- Job 2 (timestamp manufacture): position-to-time conversion
- AsrWrapper: transform chain composition and boundary reset
- Token normalization: foreign token → WLK ASRToken conversion
"""
from __future__ import annotations

import pytest

from whisperlivekit.asr_commit import (
    StableCommitTransform,
    StablePrefixCommitState,
    StableTextCommitState,
    join_text_units,
    longest_common_text_prefix_length,
    split_text_units,
    text_prefix_matches,
    update_stable_prefix_commit,
    update_stable_text_commit,
)
from whisperlivekit.asr_timestamps import WordTimestampTracker
from whisperlivekit.asr_wrapper import (
    AsrWrapper,
    _ASRTokenNormalizer,
    _to_wlk_token,
)
from whisperlivekit.timed_objects import ASRToken

# ---------------------------------------------------------------------------
# asr_commit: text-unit splitting
# ---------------------------------------------------------------------------

class TestSplitJoinUnits:
    def test_split_basic(self):
        assert split_text_units("hello world") == ["hello ", "world"]

    def test_split_empty(self):
        assert split_text_units("") == []

    def test_split_cjk(self):
        # CJK characters have no spaces between them
        units = split_text_units("你好世界")
        assert len(units) == 1
        assert units[0].strip() == "你好世界"

    def test_join_roundtrip(self):
        text = "hello world foo"
        assert join_text_units(split_text_units(text)).strip() == text


class TestLongestCommonTextPrefix:
    def test_identical(self):
        assert longest_common_text_prefix_length(["a", "b"], ["a", "b"]) == 2

    def test_divergent(self):
        assert longest_common_text_prefix_length(["a", "b"], ["a", "c"]) == 1

    def test_empty_left(self):
        assert longest_common_text_prefix_length([], ["a", "b"]) == 0

    def test_case_insensitive(self):
        assert longest_common_text_prefix_length(["Hello"], ["hello"], case_sensitive=False) == 1


class TestTextPrefixMatches:
    def test_matches(self):
        assert text_prefix_matches(["a", "b"], ["a", "b", "c"]) is True

    def test_no_match(self):
        assert text_prefix_matches(["a", "x"], ["a", "b"]) is False

    def test_prefix_longer_than_units(self):
        assert text_prefix_matches(["a", "b", "c"], ["a", "b"]) is False


# ---------------------------------------------------------------------------
# asr_commit: update_stable_text_commit
# ---------------------------------------------------------------------------

class TestUpdateStableTextCommit:
    def test_first_pass_commits_nothing(self):
        """First call with no prior state should not commit (no previous hypothesis to compare)."""
        state = StableTextCommitState()
        update = update_stable_text_commit(state, "hello world", hold_back_units=1, stable_iterations=1)
        assert update.delta_text == ""
        assert update.committed_text == ""

    def test_stable_prefix_committed_on_second_pass(self):
        """Second pass with a growing hypothesis should commit the stable prefix."""
        state = StableTextCommitState()
        update_stable_text_commit(state, "hello world foo bar", hold_back_units=1, stable_iterations=1)
        update = update_stable_text_commit(state, "hello world foo bar baz", hold_back_units=1, stable_iterations=1)
        # LCP of ["hello ", "world ", "foo ", "bar"] vs ["hello ", "world ", "foo ", "bar ", "baz"] = 3
        # (last unit "bar" != "bar " due to trailing space)
        # candidate_len = max(0, 3-1) = 2
        assert update.committed_text == "hello world"
        assert "hello" in update.delta_text

    def test_hold_back_delays_commit(self):
        """With hold_back=3, more units are held back."""
        state = StableTextCommitState()
        # Build a longer hypothesis so LCP is large enough to test hold-back
        h1 = "one two three four five six"
        h2 = "one two three four five six seven"
        update_stable_text_commit(state, h1, hold_back_units=3, stable_iterations=1)
        update = update_stable_text_commit(state, h2, hold_back_units=3, stable_iterations=1)
        units_h1 = split_text_units(h1)
        units_h2 = split_text_units(h2)
        lcp = longest_common_text_prefix_length(units_h1, units_h2)
        expected_committed = lcp - 3  # hold_back_units
        assert update.committed_unit_count == expected_committed

    def test_revision_resets_stable_candidate(self):
        """If the hypothesis changes its committed prefix, the stable candidate resets."""
        state = StableTextCommitState()
        update_stable_text_commit(state, "alpha beta", hold_back_units=0, stable_iterations=1)
        update = update_stable_text_commit(state, "alpha beta", hold_back_units=0, stable_iterations=1)
        assert update.committed_text == "alpha beta"
        # Now revise: change the prefix
        update = update_stable_text_commit(state, "gamma delta", hold_back_units=0, stable_iterations=1)
        assert update.committed_text == "alpha beta"  # committed doesn't shrink
        assert update.delta_text == ""  # no new committed text

    def test_final_flushes_everything(self):
        """final=True commits all remaining units."""
        state = StableTextCommitState()
        update_stable_text_commit(state, "hello world foo", hold_back_units=6, stable_iterations=2)
        update = update_stable_text_commit(state, "hello world foo", hold_back_units=6, stable_iterations=2)
        # With hold_back=6 and only 3 units, nothing is committed yet
        assert update.committed_text == ""
        # Final flush
        update = update_stable_text_commit(state, "hello world foo", hold_back_units=6, stable_iterations=2, final=True)
        assert update.committed_text == "hello world foo"


# ---------------------------------------------------------------------------
# asr_commit: update_stable_prefix_commit (token-ID variant)
# ---------------------------------------------------------------------------

class TestUpdateStablePrefixCommit:
    def test_basic_commit(self):
        state = StablePrefixCommitState()
        update_stable_prefix_commit(state, [1, 2, 3, 4], hold_back_tokens=1, stable_iterations=1)
        update = update_stable_prefix_commit(state, [1, 2, 3, 4, 5], hold_back_tokens=1, stable_iterations=1)
        # LCP of [1,2,3,4] and [1,2,3,4,5] = 4, hold_back=1 → commit 3
        assert update.committed_tokens == [1, 2, 3]
        assert update.delta_tokens == [1, 2, 3]

    def test_no_commit_on_revision(self):
        state = StablePrefixCommitState()
        update_stable_prefix_commit(state, [1, 2, 3], hold_back_tokens=0, stable_iterations=1)
        update = update_stable_prefix_commit(state, [1, 2, 3], hold_back_tokens=0, stable_iterations=1)
        assert update.committed_tokens == [1, 2, 3]
        # Revise prefix
        update = update_stable_prefix_commit(state, [9, 8, 7], hold_back_tokens=0, stable_iterations=1)
        assert update.committed_tokens == [1, 2, 3]  # doesn't shrink
        assert update.delta_tokens == []

    def test_final_flush(self):
        state = StablePrefixCommitState()
        update_stable_prefix_commit(state, [1, 2, 3], hold_back_tokens=10, stable_iterations=5)
        update = update_stable_prefix_commit(state, [1, 2, 3], hold_back_tokens=10, stable_iterations=5, final=True)
        assert update.committed_tokens == [1, 2, 3]
        assert update.delta_tokens == [1, 2, 3]


# ---------------------------------------------------------------------------
# asr_commit: StableCommitTransform
# ---------------------------------------------------------------------------

class TestStableCommitTransform:
    def test_transform_commits_stable_prefix(self):
        """The transform reads hypothesis from inner.get_hypothesis() and commits stable text."""
        transform = StableCommitTransform(hold_back_units=1, stable_iterations=1)

        class FakeInner:
            def __init__(self):
                self._buffer_text = ""

            def get_hypothesis(self):
                class FakeTranscript:
                    text = self._buffer_text
                return FakeTranscript()

            def get_buffer(self):
                class FakeTranscript:
                    text = self._buffer_text
                return FakeTranscript()

        inner = FakeInner()
        # First pass: no commit (no prior hypothesis)
        inner._buffer_text = "one two three four"
        result = transform(([], 1.0), inner)
        assert result == ([], 1.0)

        # Second pass: growing hypothesis. The LCP of the two unit lists
        # is 3 ("one ", "two ", "three " all match; "four" != "four "),
        # so with hold_back=1 we commit 2 units.
        inner._buffer_text = "one two three four five"
        result = transform(([], 2.0), inner)
        tokens, end_time = result
        assert len(tokens) == 1
        assert "one" in tokens[0].text
        assert end_time == 2.0

    def test_reset_clears_state(self):
        transform = StableCommitTransform(hold_back_units=0, stable_iterations=1)

        class FakeInner:
            def __init__(self):
                self._buffer_text = "alpha beta"

            def get_hypothesis(self):
                class FakeTranscript:
                    text = self._buffer_text
                return FakeTranscript()

            def get_buffer(self):
                class FakeTranscript:
                    text = self._buffer_text
                return FakeTranscript()

        inner = FakeInner()
        # Commit something
        transform(([], 1.0), inner)
        transform(([], 2.0), inner)
        assert transform._state.committed_units  # has committed text

        # Reset
        transform.reset()
        assert not transform._state.committed_units  # cleared

    def test_reset_called_at_boundary(self):
        """AsrWrapper calls reset() on stateful transforms at start_silence/finish."""
        transform = StableCommitTransform(hold_back_units=0, stable_iterations=1)

        class FakeInner:
            def __init__(self):
                self._buffer_text = "test data"

            def get_hypothesis(self):
                class FakeTranscript:
                    text = self._buffer_text
                return FakeTranscript()

            def get_buffer(self):
                class FakeTranscript:
                    text = self._buffer_text
                return FakeTranscript()

            def process_iter(self, is_last=False):
                return [], 1.0

            def start_silence(self):
                return [ASRToken(0, 1, "final text")], 1.0

            def finish(self):
                return [ASRToken(0, 1, "final text")], 1.0

            sep = " "

        inner = FakeInner()
        wrapper = AsrWrapper(inner, transforms=[transform])

        # process_iter: stable_commit runs, accumulates state
        wrapper.process_iter()
        wrapper.process_iter()
        assert transform._state.previous_hypothesis_units  # state exists

        # start_silence: should reset transform state
        wrapper.start_silence()
        assert not transform._state.previous_hypothesis_units  # state cleared

    def test_empty_hypothesis_no_crash(self):
        transform = StableCommitTransform()

        class FakeInner:
            def get_hypothesis(self):
                return None

        result = transform(([], 1.0), FakeInner())
        assert result == ([], 1.0)

    def test_get_hypothesis_full_text_not_buffer_tail(self):
        """The transform reads the FULL rolling hypothesis (get_hypothesis), not
        the unstable tail (get_buffer). With a committed prefix already present,
        get_buffer returns only the tail; the transform must still see the full
        text to compute the stable prefix across passes — this is the bug the
        get_hypothesis seam fixes (reading get_buffer emitted garbage deltas
        omitting the committed prefix)."""
        transform = StableCommitTransform(hold_back_units=0, stable_iterations=1)

        class FakeInner:
            def __init__(self):
                # The inner processor tracks a committed stable prefix and the
                # full rolling text. get_hypothesis returns the full text;
                # get_buffer returns only the tail (WLK contract).
                self._stable = ""
                self._full = ""

            def get_hypothesis(self):
                class T:
                    text = self._full
                return T()

            def get_buffer(self):
                class T:
                    text = self._full[len(self._stable):] if self._stable and self._full.startswith(self._stable) else self._full
                return T()

        inner = FakeInner()
        # Pass 1: full rolling text "alpha beta gamma"
        inner._full = "alpha beta gamma"
        transform(([], 1.0), inner)

        # Pass 2: same text — stable across passes, so with hold_back=0 and
        # stable_iterations=1 the prefix commits. get_buffer would return only
        # "gamma" (the tail after "alpha beta"); the transform must commit
        # from the full hypothesis, producing a delta containing "alpha".
        inner._stable = "alpha beta"  # simulate the backend's stable_text advancing
        inner._full = "alpha beta gamma"
        result = transform(([], 2.0), inner)
        tokens, _ = result
        assert len(tokens) == 1
        # The delta must contain the committed prefix units, not just the tail.
        assert "alpha" in tokens[0].text
        # Sanity: get_buffer returns only the tail ("gamma"), proving this test
        # would fail if the transform read get_buffer.
        assert inner.get_buffer().text.strip() == "gamma"


# ---------------------------------------------------------------------------
# asr_timestamps: WordTimestampTracker
# ---------------------------------------------------------------------------

class TestWordTimestampTracker:
    def test_basic_tracking(self):
        tracker = WordTimestampTracker(secs_per_token=0.08, delay_secs=0.0)
        tracker.record_word_start(10)
        tracker.record_word_start(25)
        tracker.record_word_end(30)

        t0, t1 = tracker.word_time_range(0)
        assert t0 == pytest.approx(0.8)  # 10 * 0.08
        assert t1 == pytest.approx(2.0)  # 25 * 0.08 (start of next word)

        t0, t1 = tracker.word_time_range(1)
        assert t0 == pytest.approx(2.0)  # 25 * 0.08
        assert t1 == pytest.approx(2.48)  # (30 + 1) * 0.08

    def test_delay_compensation(self):
        tracker = WordTimestampTracker(secs_per_token=0.08, delay_secs=0.48)
        tracker.record_word_start(10)
        tracker.record_word_start(20)
        t0, _ = tracker.word_time_range(0)
        # 10 * 0.08 - 0.48 = 0.32, but max(0, ...) = 0.32
        assert t0 == pytest.approx(0.32)

    def test_time_offset(self):
        tracker = WordTimestampTracker(secs_per_token=0.08, delay_secs=0.0)
        tracker.time_offset = 5.0
        tracker.record_word_start(10)
        tracker.record_word_start(20)
        t0, _ = tracker.word_time_range(0)
        assert t0 == pytest.approx(5.8)  # 10 * 0.08 + 5.0

    def test_reset(self):
        tracker = WordTimestampTracker(secs_per_token=0.08)
        tracker.record_word_start(10)
        tracker.reset()
        assert tracker.word_audio_starts == []
        assert tracker.word_audio_ends == []
        assert not tracker.has_current_word

    def test_has_current_word(self):
        tracker = WordTimestampTracker(secs_per_token=0.08)
        assert not tracker.has_current_word
        tracker.record_word_start(10)
        assert tracker.has_current_word
        tracker.record_word_end(15)
        assert not tracker.has_current_word

    def test_empty_tracker(self):
        tracker = WordTimestampTracker(secs_per_token=0.08)
        t0, t1 = tracker.word_time_range(0)
        assert t0 == 0.0
        assert t1 == 0.0


# ---------------------------------------------------------------------------
# asr_wrapper: AsrWrapper
# ---------------------------------------------------------------------------

class TestAsrWrapper:
    def test_token_normalize_only(self):
        """AsrWrapper with no transforms = just token normalization (same as _ASRTokenNormalizer)."""

        class FakeInner:
            def process_iter(self, is_last=False):
                return [_FakeToken(0, 1, "hello")], 1.0

            def start_silence(self):
                return [_FakeToken(0, 1, "world")], 1.0

            def finish(self):
                return [_FakeToken(0, 1, "end")], 1.0

            sep = " "

        class _FakeToken:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text
                self.speaker = -1
                self.detected_language = None
                self.probability = None

            def is_silence(self):
                return False

        wrapper = AsrWrapper(FakeInner())
        tokens, end = wrapper.process_iter()
        assert len(tokens) == 1
        assert isinstance(tokens[0], ASRToken)
        assert tokens[0].text == "hello"

        tokens, end = wrapper.start_silence()
        assert tokens[0].text == "world"

        tokens, end = wrapper.finish()
        assert tokens[0].text == "end"

    def test_transforms_applied_on_process_iter(self):
        calls = []

        def my_transform(result, inner):
            tokens, end_time = result
            calls.append("transform")
            return tokens, end_time

        class FakeInner:
            def process_iter(self, is_last=False):
                return [], 1.0

            def start_silence(self):
                return [], 1.0

            def finish(self):
                return [], 1.0

            sep = " "

        wrapper = AsrWrapper(FakeInner(), transforms=[my_transform])
        wrapper.process_iter()
        assert calls == ["transform"]

        # start_silence should NOT call the transform (boundary)
        calls.clear()
        wrapper.start_silence()
        assert calls == []

    def test_reset_on_boundary(self):
        reset_calls = []

        class MyTransform:
            def __call__(self, result, inner):
                return result

            def reset(self):
                reset_calls.append("reset")

        class FakeInner:
            def process_iter(self, is_last=False):
                return [], 1.0

            def start_silence(self):
                return [], 1.0

            def finish(self):
                return [], 1.0

            sep = " "

        wrapper = AsrWrapper(FakeInner(), transforms=[MyTransform()])
        wrapper.start_silence()
        assert reset_calls == ["reset"]

        reset_calls.clear()
        wrapper.finish()
        assert reset_calls == ["reset"]

    def test_get_buffer_forwarded(self):
        class FakeInner:
            def get_buffer(self):
                return "buffer_value"

            def process_iter(self, is_last=False):
                return [], 1.0

            def start_silence(self):
                return [], 1.0

            sep = " "

        wrapper = AsrWrapper(FakeInner())
        assert wrapper.get_buffer() == "buffer_value"

    def test_new_speaker_calls_start_silence(self):
        class FakeInner:
            def __init__(self):
                self.silence_called = False

            def start_silence(self):
                self.silence_called = True
                return [ASRToken(0, 1, "flush")], 1.0

            sep = " "

        inner = FakeInner()
        wrapper = AsrWrapper(inner)
        result = wrapper.new_speaker(change_speaker=None)
        assert inner.silence_called
        tokens, _ = result
        assert tokens[0].text == "flush"

    def test_backward_compat_ASRTokenNormalizer(self):
        """_ASRTokenNormalizer is still importable and works like AsrWrapper with no transforms."""

        class FakeInner:
            def process_iter(self, is_last=False):
                return [], 1.0

            def start_silence(self):
                return [], 1.0

            sep = " "

        normalizer = _ASRTokenNormalizer(FakeInner())
        assert isinstance(normalizer, AsrWrapper)
        result = normalizer.process_iter()
        assert result == ([], 1.0)


# ---------------------------------------------------------------------------
# asr_wrapper: _to_wlk_token
# ---------------------------------------------------------------------------

class TestToWlkToken:
    def test_already_timed_text(self):
        tok = ASRToken(0, 1, "hello")
        assert _to_wlk_token(tok) is tok

    def test_silence_token(self):
        class SilenceTok:
            start = 0
            end = 0
            text = ""

            def is_silence(self):
                return True

        tok = SilenceTok()
        result = _to_wlk_token(tok)
        assert result is tok  # silence tokens pass through

    def test_foreign_token(self):
        class ForeignTok:
            def __init__(self):
                self.start = 1.0
                self.end = 2.0
                self.text = "world"
                self.speaker = 3
                self.detected_language = "en"
                self.probability = 0.95

            def is_silence(self):
                return False

        result = _to_wlk_token(ForeignTok())
        assert isinstance(result, ASRToken)
        assert result.start == 1.0
        assert result.end == 2.0
        assert result.text == "world"
        assert result.speaker == 3
        assert result.detected_language == "en"
        assert result.probability == 0.95


# ---------------------------------------------------------------------------
# finalization dedup (no text duplication)
# ---------------------------------------------------------------------------

class TestFinalizeDedup:
    """Integration tests for finalization dedup: _finalize_utterance must emit only the
    uncommitted delta, not the full re-decoded text, to avoid duplicating the
    stable prefix that was already emitted during streaming."""

    def test_compute_finalize_delta_no_emitted(self):
        """No streaming commit → emit full text."""
        from whisperlivekit.asr_mlx_qwen3 import _compute_finalize_delta

        assert _compute_finalize_delta("alpha beta gamma", "") == "alpha beta gamma"

    def test_compute_finalize_delta_prefix_match(self):
        """final_text starts with emitted_stable → emit only the suffix."""
        from whisperlivekit.asr_mlx_qwen3 import _compute_finalize_delta

        assert _compute_finalize_delta("alpha beta gamma delta", "alpha beta") == "gamma delta"

    def test_compute_finalize_delta_exact_match(self):
        """final_text == emitted_stable → emit nothing (already fully committed)."""
        from whisperlivekit.asr_mlx_qwen3 import _compute_finalize_delta

        assert _compute_finalize_delta("alpha beta", "alpha beta") == ""

    def test_compute_finalize_delta_correction(self):
        """Re-decode corrected the prefix → emit full corrected text."""
        from whisperlivekit.asr_mlx_qwen3 import _compute_finalize_delta

        # The re-decode changed "alpha beta" to "alpha bet" — emit the full
        # corrected text; the stale prefix remains (known limitation).
        assert _compute_finalize_delta("alpha bet gamma", "alpha beta") == "alpha bet gamma"

    def test_transform_updates_emitted_stable(self):
        """StableCommitTransform tracks committed text on inner._emitted_stable."""
        transform = StableCommitTransform(hold_back_units=0, stable_iterations=1)

        class FakeInner:
            def __init__(self):
                self._text = ""
                self._emitted_stable = ""

            def get_hypothesis(self):
                class T:
                    text = self._text
                return T()

        inner = FakeInner()
        # Pass 1: no commit (no prior hypothesis)
        inner._text = "one two three"
        transform(([], 1.0), inner)
        assert inner._emitted_stable == ""

        # Pass 2: growing hypothesis — commits stable prefix
        inner._text = "one two three four"
        transform(([], 2.0), inner)
        assert inner._emitted_stable  # something was committed
        assert "one" in inner._emitted_stable

    def test_integration_no_duplication(self):
        """Integration: streaming process_iter commits a stable prefix via
        StableCommitTransform, then start_silence finalizes with only the
        delta — the combined output has NO duplication.

        This test exercises the real StableCommitTransform, AsrWrapper, and
        _compute_finalize_delta together. The FakeInner simulates
        MlxQwen3AsrOnlineProcessor's _finalize_utterance by calling the real
        _compute_finalize_delta with the _emitted_stable that the transform
        populated during streaming.

        Falsifier: on the pre-fix code (full-text finalize, _emitted_stable
        never populated), the finalized token would repeat the full text,
        producing "alpha beta ... alpha beta gamma delta"."""
        from whisperlivekit.asr_mlx_qwen3 import _compute_finalize_delta

        transform = StableCommitTransform(hold_back_units=0, stable_iterations=1)
        all_emitted = []  # collect every token text emitted

        class FakeInner:
            """Simulates MlxQwen3AsrOnlineProcessor: tracks _emitted_stable
            (populated by the transform during streaming) and finalizes via
            the real _compute_finalize_delta."""

            def __init__(self):
                self._text = ""
                self._stable_text = ""
                self._emitted_stable = ""
                self._audio_end_time = 0.0

            def get_hypothesis(self):
                class T:
                    text = self._text
                return T()

            def get_buffer(self):
                class T:
                    text = self._text[len(self._stable_text):] if self._stable_text else self._text
                return T()

            def process_iter(self, is_last=False):
                return [], self._audio_end_time

            def start_silence(self):
                # Simulate the re-decode producing the full clean text.
                final_text = "alpha beta gamma delta"
                # Use the REAL dedup logic (same as _finalize_utterance).
                text_to_emit = _compute_finalize_delta(final_text, self._emitted_stable)
                self._emitted_stable = ""
                if text_to_emit:
                    return [ASRToken(0, 0, text_to_emit)], 0.0
                return [], 0.0

            def finish(self):
                return self.start_silence()

            sep = " "

        inner = FakeInner()
        wrapper = AsrWrapper(inner, transforms=[transform])

        # Streaming pass 1: hypothesis appears
        inner._text = "alpha beta gamma"
        tokens, _ = wrapper.process_iter()
        all_emitted.extend(t.text for t in tokens)

        # Streaming pass 2: hypothesis grows — stable prefix commits
        inner._text = "alpha beta gamma delta"
        tokens, _ = wrapper.process_iter()
        all_emitted.extend(t.text for t in tokens)

        # The transform should have committed a stable prefix during streaming.
        assert inner._emitted_stable, "streaming should have committed a prefix"
        streaming_text = inner._emitted_stable

        # Finalize: start_silence triggers _finalize_utterance.
        tokens, _ = wrapper.start_silence()
        all_emitted.extend(t.text for t in tokens)

        # Combine all emitted text.
        combined = " ".join(all_emitted).strip()

        # The combined output must be the full utterance with NO duplication.
        # The streaming commits emitted the stable prefix; the finalization
        # emitted only the delta (the remaining suffix).
        assert combined == "alpha beta gamma delta", (
            f"duplication detected: combined={combined!r}"
        )

        # Verify the finalized token is only the delta, not the full text.
        finalize_text = tokens[0].text if tokens else ""
        assert streaming_text not in finalize_text, (
            f"finalized token repeats streaming prefix: {finalize_text!r}"
        )

    def test_integration_no_duplication_short_utterance(self):
        """Short utterance where no streaming commit happens (_emitted_stable
        is empty): finalization emits the full text (no duplication because
        nothing was committed during streaming)."""
        from whisperlivekit.asr_mlx_qwen3 import _compute_finalize_delta

        transform = StableCommitTransform(hold_back_units=6, stable_iterations=2)
        all_emitted = []

        class FakeInner:
            def __init__(self):
                self._text = ""
                self._emitted_stable = ""
                self._audio_end_time = 0.0

            def get_hypothesis(self):
                class T:
                    text = self._text
                return T()

            def process_iter(self, is_last=False):
                return [], self._audio_end_time

            def start_silence(self):
                final_text = "short"
                text_to_emit = _compute_finalize_delta(final_text, self._emitted_stable)
                self._emitted_stable = ""
                if text_to_emit:
                    return [ASRToken(0, 0, text_to_emit)], 0.0
                return [], 0.0

            def finish(self):
                return self.start_silence()

            sep = " "

        inner = FakeInner()
        wrapper = AsrWrapper(inner, transforms=[transform])

        # Only one streaming pass with a short hypothesis.
        inner._text = "short"
        tokens, _ = wrapper.process_iter()
        all_emitted.extend(t.text for t in tokens)

        # With hold_back=6 and stable_iterations=2, nothing is committed.
        assert inner._emitted_stable == ""
        assert all_emitted == []

        # Finalize: emit the full text (no streaming commit to dedup against).
        tokens, _ = wrapper.start_silence()
        all_emitted.extend(t.text for t in tokens)

        combined = " ".join(all_emitted).strip()
        assert combined == "short"
