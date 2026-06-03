from qwen3_streaming.realtime_targets import (
    WordAlignment,
    build_frame_targets,
    heuristic_word_alignments,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab = {"hello": [10], " hello": [13], " world": [11, 12]}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.vocab[text]


def test_frame_targets_schedule_words_with_delay_and_word_start():
    targets = build_frame_targets(
        words=[
            WordAlignment("hello", start_sec=0.0, end_sec=0.2),
            WordAlignment("world", start_sec=0.16, end_sec=0.4),
        ],
        tokenizer=FakeTokenizer(),
        duration_sec=0.5,
        wait_token_id=0,
        word_start_token_id=99,
        bos_token_id=1,
        frame_sec=0.08,
        delay_sec=0.16,
    )

    assert targets.labels[:7] == [0, 0, 99, 10, 99, 11, 12]
    assert targets.previous_input_ids[:7] == [1, 0, 0, 99, 10, 99, 11]
    assert [event.frame_index for event in targets.emissions] == [2, 3, 4, 5, 6]


def test_frame_targets_shift_collisions_to_next_free_frame():
    targets = build_frame_targets(
        words=[
            WordAlignment("hello", start_sec=0.0, end_sec=0.2),
            WordAlignment("hello", start_sec=0.0, end_sec=0.2),
        ],
        tokenizer=FakeTokenizer(),
        duration_sec=0.3,
        wait_token_id=0,
        word_start_token_id=99,
        bos_token_id=None,
        frame_sec=0.08,
        delay_sec=0.0,
    )

    assert targets.labels[:4] == [99, 10, 99, 13]
    assert targets.previous_input_ids[:4] == [0, 99, 10, 99]


def test_heuristic_word_alignments_cover_duration():
    words = heuristic_word_alignments("a longer word", duration_sec=2.0)

    assert [word.text for word in words] == ["a", "longer", "word"]
    assert words[0].start_sec == 0.0
    assert words[-1].end_sec == 2.0
    assert all(left.end_sec == right.start_sec for left, right in zip(words, words[1:]))
