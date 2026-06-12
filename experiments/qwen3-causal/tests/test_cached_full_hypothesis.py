from qwen3_streaming.cached_full_hypothesis import (
    CachedFullHypothesisConfig,
    CachedFullHypothesisStreamer,
    SegmentedCachedFullHypothesisStreamer,
    decode_clean_token_ids,
    ends_with_sentence_punctuation,
    expand_audio_prompt_placeholders,
    qwen_asr_prompt_text,
    trailing_text_words,
    trim_at_stop,
)


class FakeTokenizer:
    def __init__(self):
        self.vocab = {
            99: "[P]",
            98: "[W]",
            7: "<|audio_pad|>",
            1: "hello",
            2: "world",
            3: "today",
            4: "done.",
        }
        self.last_encoded_text = ""

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = []
        for token_id in token_ids:
            token = self.vocab[int(token_id)]
            if skip_special_tokens and token.startswith("["):
                continue
            pieces.append(token)
        return " ".join(pieces)

    def encode(self, text, add_special_tokens=False):
        self.last_encoded_text = text
        return [10, 7, 11]


class FakeModel:
    def init_cached_audio_decode_state(self):
        return object()


class FakeFrameHidden:
    def __init__(self, steps: int):
        self.shape = (1, int(steps), 2)

    def __getitem__(self, key):
        step_selector = key[1]
        if isinstance(step_selector, slice):
            start, stop, stride = step_selector.indices(self.shape[1])
            return FakeFrameHidden(len(range(start, stop, stride)))
        raise TypeError("FakeFrameHidden only supports step slices")


class FakeState:
    def __init__(self, steps: int):
        self.frame_hidden = FakeFrameHidden(steps)


def test_prompt_helpers_trim_and_expand_audio_placeholders():
    prompt = qwen_asr_prompt_text(context="ctx", language="English")

    assert "ctx" in prompt
    assert "language English<asr_text>" in prompt
    assert trim_at_stop([1, 2, 0, 3], 0) == [1, 2]
    assert expand_audio_prompt_placeholders(
        [10, 7, 11],
        audio_placeholder_token_id=7,
        audio_steps=3,
    ) == [10, 7, 7, 7, 11]


def test_decode_clean_token_ids_filters_realtime_specials():
    assert (
        decode_clean_token_ids(
            FakeTokenizer(),
            [99, 1, 98, 2],
            wait_token_id=99,
            word_start_token_id=98,
        )
        == "hello world"
    )


def test_trailing_text_words_keeps_only_recent_context():
    assert trailing_text_words("one two three four", 2) == "three four"
    assert trailing_text_words("one two", 8) == "one two"
    assert trailing_text_words("one two", 0) == ""


def test_streamer_commits_stable_word_prefix_with_min_commit_delay():
    streamer = CachedFullHypothesisStreamer(
        FakeModel(),
        FakeTokenizer(),
        CachedFullHypothesisConfig(
            wait_token_id=99,
            word_start_token_id=98,
            hold_back_words=0,
            stable_iterations=1,
            min_commit_audio_sec=3.0,
        ),
    )

    first = streamer.update_from_hypothesis([1, 2], audio_sec=1.0)
    second = streamer.update_from_hypothesis([1, 2, 3], audio_sec=2.0)
    third = streamer.update_from_hypothesis([1, 2, 3], audio_sec=3.0)

    assert first["committed"] == ""
    assert second["committed"] == ""
    assert third["committed"] == "hello world today"
    assert streamer.finalize(finalize_mode="latest").final_text == "hello world today"


def test_streamer_token_mode_finalizes_latest_tokens():
    streamer = CachedFullHypothesisStreamer(
        FakeModel(),
        FakeTokenizer(),
        CachedFullHypothesisConfig(
            wait_token_id=99,
            word_start_token_id=98,
            hold_back_tokens=0,
            stable_iterations=1,
            commit_mode="token",
        ),
    )

    streamer.update_from_hypothesis([1, 2], audio_sec=1.0)
    streamer.update_from_hypothesis([1, 2, 3], audio_sec=2.0)
    final = streamer.finalize(finalize_mode="latest")

    assert final.final_tokens == [1, 2, 3]
    assert final.final_text == "hello world today"


def test_segmented_streamer_rolls_completed_text_into_global_hypothesis():
    streamer = SegmentedCachedFullHypothesisStreamer(
        FakeModel(),
        FakeTokenizer(),
        CachedFullHypothesisConfig(
            wait_token_id=99,
            word_start_token_id=98,
            hold_back_words=0,
            stable_iterations=1,
        ),
    )

    first = streamer.update_from_hypothesis([1, 2], audio_sec=1.0, cached_steps=4)
    segment_final = streamer.roll_segment()
    second = streamer.update_from_hypothesis([3], audio_sec=2.0, cached_steps=1)
    final = streamer.finalize(finalize_mode="latest")

    assert first["hypothesis"] == "hello world"
    assert segment_final.final_text == "hello world"
    assert streamer.segments_finalized == 1
    assert second["segment_hypothesis"] == "today"
    assert second["hypothesis"] == "hello world today"
    assert second["display"] == "hello world today"
    assert final.final_text == "hello world today"


def test_segmented_streamer_auto_rolls_and_trims_cached_audio_tail():
    streamer = SegmentedCachedFullHypothesisStreamer(
        FakeModel(),
        FakeTokenizer(),
        CachedFullHypothesisConfig(
            wait_token_id=99,
            word_start_token_id=98,
            hold_back_words=0,
            stable_iterations=1,
        ),
        state=FakeState(steps=6),
        segment_max_cached_steps=4,
        segment_keep_tail_steps=2,
    )

    event = streamer.update_from_hypothesis([1, 2], audio_sec=1.0, cached_steps=6)

    assert event["segment_rollover"] is True
    assert event["segment_final_text"] == "hello world"
    assert event["completed_text_after_roll"] == "hello world"
    assert event["active_cached_steps_after_roll"] == 2
    assert streamer.state.frame_hidden.shape[1] == 2
    assert streamer.dropped_cached_steps_total == 4
    assert streamer.segments_finalized == 1


def test_segmented_streamer_builds_dynamic_prompt_from_completed_tail():
    tokenizer = FakeTokenizer()
    streamer = SegmentedCachedFullHypothesisStreamer(
        FakeModel(),
        tokenizer,
        CachedFullHypothesisConfig(
            wait_token_id=99,
            word_start_token_id=98,
            audio_placeholder_token_id=7,
        ),
        segment_prompt_context_words=2,
        segment_prompt_base_context="base instructions",
        segment_prompt_language="English",
    )
    streamer.completed_text = "hello world today"

    prefix = streamer.prompt_prefix_token_ids(audio_steps=3)

    assert prefix == [10, 7, 7, 7, 11]
    assert "base instructions" in tokenizer.last_encoded_text
    assert "Previous transcript context:" in tokenizer.last_encoded_text
    assert "world today" in tokenizer.last_encoded_text
    assert "language English<asr_text>" in tokenizer.last_encoded_text


def test_segment_rollover_can_reset_encoder_state():
    class AudioState:
        def __init__(self, emitted_steps=0, frames_seen=0):
            self.emitted_steps = emitted_steps
            self.frames_seen = frames_seen
            self.mel_buffer = None

    class DecodeState:
        def __init__(self):
            self.audio = AudioState(emitted_steps=4321, frames_seen=999)
            self.frame_hidden = None

    class Encoder:
        def init_state(self):
            return AudioState()

    class Model:
        audio_encoder = Encoder()

        def init_cached_audio_decode_state(self):
            return DecodeState()

    config = CachedFullHypothesisConfig(wait_token_id=99, word_start_token_id=98)
    streamer = SegmentedCachedFullHypothesisStreamer(
        Model(),
        FakeTokenizer(),
        config,
        segment_max_cached_steps=2,
        reset_encoder_on_rollover=True,
    )
    streamer.roll_segment()

    # Encoder positions restarted; stream-time bookkeeping preserved.
    assert streamer.state.audio.emitted_steps == 0
    assert streamer.state.audio.frames_seen == 999


def test_ends_with_sentence_punctuation():
    assert ends_with_sentence_punctuation("Hello world.")
    assert ends_with_sentence_punctuation("Really?!")
    assert ends_with_sentence_punctuation('He said "stop."')
    assert ends_with_sentence_punctuation("Done.”  ")
    assert not ends_with_sentence_punctuation("Hello world")
    assert not ends_with_sentence_punctuation("3,5")
    assert not ends_with_sentence_punctuation("")
    assert not ends_with_sentence_punctuation('   ""  ')


def make_punct_streamer(**kwargs):
    return SegmentedCachedFullHypothesisStreamer(
        FakeModel(),
        FakeTokenizer(),
        CachedFullHypothesisConfig(
            wait_token_id=99,
            word_start_token_id=98,
            hold_back_words=0,
            stable_iterations=1,
        ),
        **kwargs,
    )


def test_segmented_streamer_rolls_at_sentence_punctuation():
    streamer = make_punct_streamer(
        segment_punct_rollover=True,
        segment_punct_min_steps=3,
    )

    # Punctuated but below the minimum segment length: no roll.
    early = streamer.update_from_hypothesis([1, 4], audio_sec=1.0, cached_steps=2)
    assert early["segment_rollover"] is False

    # Long enough but no terminal punctuation: no roll.
    plain = streamer.update_from_hypothesis([1, 2], audio_sec=2.0, cached_steps=5)
    assert plain["segment_rollover"] is False

    # Long enough and sentence-final: roll with the punctuation reason.
    rolled = streamer.update_from_hypothesis([1, 2, 4], audio_sec=3.0, cached_steps=5)
    assert rolled["segment_rollover"] is True
    assert rolled["segment_rollover_reason"] == "punctuation"
    assert rolled["segment_final_text"] == "hello world done."
    assert streamer.segments_finalized == 1


def test_cap_rollover_reason_takes_precedence_over_punctuation():
    streamer = make_punct_streamer(
        segment_max_cached_steps=4,
        segment_punct_rollover=True,
        segment_punct_min_steps=1,
    )
    event = streamer.update_from_hypothesis([1, 2, 4], audio_sec=1.0, cached_steps=6)
    assert event["segment_rollover"] is True
    assert event["segment_rollover_reason"] == "cap"
