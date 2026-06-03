from qwen3_streaming.ctc import (
    build_compact_ctc_vocab,
    build_ctc_token_targets,
    ctc_greedy_decode,
)
from qwen3_streaming.realtime_targets import WordAlignment


class FakeTokenizer:
    def __init__(self) -> None:
        self.values = {
            "hello": [10],
            " world": [0, 99, 11, 12],
            "again": [13],
        }

    def encode(self, text: str, add_special_tokens: bool = False):
        return self.values[text]


def test_ctc_target_builder_filters_wait_and_word_start_tokens():
    targets = build_ctc_token_targets(
        words=[
            WordAlignment("hello", 0.0, 0.2),
            WordAlignment("world", 0.2, 0.4),
        ],
        tokenizer=FakeTokenizer(),
        blank_token_id=0,
        ignored_token_ids={99},
    )

    assert targets == [10, 11, 12]


def test_ctc_greedy_decode_collapses_repeats_but_keeps_repeated_token_after_blank():
    decoded = ctc_greedy_decode(
        [0, 5, 5, 0, 5, 6, 6, 99, 0],
        blank_token_id=0,
        ignored_token_ids={99},
    )

    assert decoded.token_ids == [5, 5, 6]
    assert decoded.blank_count == 3
    assert decoded.raw_text_token_count == 5


def test_ctc_greedy_decode_respects_previous_stream_token():
    decoded = ctc_greedy_decode(
        [7, 7, 0, 7],
        blank_token_id=0,
        previous_token_id=7,
    )

    assert decoded.token_ids == [7]


def test_compact_ctc_vocab_orders_by_frequency_and_roundtrips():
    vocab = build_compact_ctc_vocab(
        [[10, 20, 10], [30, 20, 10]],
        blank_token_id=0,
    )

    assert vocab.token_ids == [0, 10, 20, 30]
    assert vocab.encode([30, 10]) == [3, 1]
    assert vocab.decode([1, 2, 3]) == [10, 20, 30]


def test_compact_ctc_vocab_can_be_capped():
    vocab = build_compact_ctc_vocab(
        [[10, 20, 10], [30, 20, 40]],
        blank_token_id=0,
        max_tokens=3,
    )

    assert vocab.token_ids == [0, 10, 20]
