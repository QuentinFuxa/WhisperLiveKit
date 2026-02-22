"""Tests for whisperlivekit.metrics — WER, timestamp accuracy, normalization."""

import pytest

from whisperlivekit.metrics import compute_wer, compute_timestamp_accuracy, normalize_text


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strip_punctuation(self):
        assert normalize_text("Hello, world!") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_text("  hello   world  ") == "hello world"

    def test_keep_hyphens(self):
        assert normalize_text("real-time") == "real-time"

    def test_keep_apostrophes(self):
        assert normalize_text("don't") == "don't"

    def test_unicode_normalized(self):
        # e + combining accent should be same as precomposed
        assert normalize_text("caf\u0065\u0301") == normalize_text("caf\u00e9")

    def test_empty(self):
        assert normalize_text("") == ""

    def test_only_punctuation(self):
        assert normalize_text("...!?") == ""


class TestComputeWER:
    def test_perfect_match(self):
        result = compute_wer("hello world", "hello world")
        assert result["wer"] == 0.0
        assert result["substitutions"] == 0
        assert result["insertions"] == 0
        assert result["deletions"] == 0

    def test_case_insensitive(self):
        result = compute_wer("Hello World", "hello world")
        assert result["wer"] == 0.0

    def test_punctuation_ignored(self):
        result = compute_wer("Hello, world!", "hello world")
        assert result["wer"] == 0.0

    def test_one_substitution(self):
        result = compute_wer("hello world", "hello earth")
        assert result["wer"] == pytest.approx(0.5)
        assert result["substitutions"] == 1

    def test_one_insertion(self):
        result = compute_wer("hello world", "hello big world")
        assert result["wer"] == pytest.approx(0.5)
        assert result["insertions"] == 1

    def test_one_deletion(self):
        result = compute_wer("hello big world", "hello world")
        assert result["wer"] == pytest.approx(1 / 3)
        assert result["deletions"] == 1

    def test_completely_different(self):
        result = compute_wer("the cat sat", "a dog ran")
        assert result["wer"] == pytest.approx(1.0)

    def test_empty_reference(self):
        result = compute_wer("", "hello")
        assert result["wer"] == 1.0  # 1 insertion / 0 ref → treated as float(m)
        assert result["ref_words"] == 0

    def test_empty_hypothesis(self):
        result = compute_wer("hello world", "")
        assert result["wer"] == pytest.approx(1.0)
        assert result["deletions"] == 2

    def test_both_empty(self):
        result = compute_wer("", "")
        assert result["wer"] == 0.0

    def test_ref_and_hyp_word_counts(self):
        result = compute_wer("one two three", "one two three four")
        assert result["ref_words"] == 3
        assert result["hyp_words"] == 4


class TestComputeTimestampAccuracy:
    def test_perfect_match(self):
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
        result = compute_timestamp_accuracy(words, words)
        assert result["mae_start"] == 0.0
        assert result["max_delta_start"] == 0.0
        assert result["n_matched"] == 2

    def test_constant_offset(self):
        ref = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
        pred = [
            {"word": "hello", "start": 0.1, "end": 0.6},
            {"word": "world", "start": 0.6, "end": 1.1},
        ]
        result = compute_timestamp_accuracy(pred, ref)
        assert result["mae_start"] == pytest.approx(0.1)
        assert result["max_delta_start"] == pytest.approx(0.1)
        assert result["n_matched"] == 2

    def test_mismatched_word_counts(self):
        ref = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "beautiful", "start": 0.5, "end": 1.0},
            {"word": "world", "start": 1.0, "end": 1.5},
        ]
        pred = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 1.1, "end": 1.6},
        ]
        result = compute_timestamp_accuracy(pred, ref)
        assert result["n_matched"] == 2
        assert result["n_ref"] == 3
        assert result["n_pred"] == 2

    def test_empty_predicted(self):
        ref = [{"word": "hello", "start": 0.0, "end": 0.5}]
        result = compute_timestamp_accuracy([], ref)
        assert result["mae_start"] is None
        assert result["n_matched"] == 0

    def test_empty_reference(self):
        pred = [{"word": "hello", "start": 0.0, "end": 0.5}]
        result = compute_timestamp_accuracy(pred, [])
        assert result["mae_start"] is None
        assert result["n_matched"] == 0

    def test_case_insensitive_matching(self):
        ref = [{"word": "Hello", "start": 0.0, "end": 0.5}]
        pred = [{"word": "hello", "start": 0.1, "end": 0.6}]
        result = compute_timestamp_accuracy(pred, ref)
        assert result["n_matched"] == 1
        assert result["mae_start"] == pytest.approx(0.1)
