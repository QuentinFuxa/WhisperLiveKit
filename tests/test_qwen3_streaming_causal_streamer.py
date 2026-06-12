"""Segmented streamer with the causal model: rolling parity across
rollovers, roll-before-generate, punct rollover, per-segment template caching.

Ported from experiments/qwen3-causal/tests/test_decoder_rolling_kv.py @ 9d4b99a
(streamer-level section)."""

import pytest

torch = pytest.importorskip("torch")

from whisperlivekit.qwen3_streaming.streamer import (  # noqa: E402
    CachedFullHypothesisConfig,
    SegmentedCachedFullHypothesisStreamer,
)

from qwen3_streaming_fakes import N_MELS  # noqa: E402
from test_qwen3_streaming_causal_rolling import (  # noqa: E402
    EOS,
    PH,
    TEMPLATE,
    make_rolling_model,
)


class IntTokenizer:
    """Tokenizer over small int vocab; encode returns the test template."""

    def __init__(self):
        self.encode_calls = 0

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"t{int(t)}" for t in token_ids)

    def encode(self, text, add_special_tokens=False):
        self.encode_calls += 1
        return list(TEMPLATE)


def make_streamer(model, tokenizer, *, rolling: bool, **streamer_kwargs):
    config = CachedFullHypothesisConfig(
        wait_token_id=0,
        word_start_token_id=98,
        eos_token_id=EOS,
        max_new_tokens=8,
        hold_back_words=0,
        stable_iterations=1,
        suppress_token_ids=(2, 3),
        repetition_penalty=1.3,
        no_repeat_ngram_size=2,
        prompt_prefix_template=tuple(TEMPLATE),
        audio_placeholder_token_id=PH,
        decoder_rolling_kv=rolling,
        speculative_draft=rolling,
    )
    return SegmentedCachedFullHypothesisStreamer(
        model,
        tokenizer,
        config,
        segment_max_cached_steps=5,
        segment_finalize_mode="latest",
        reset_encoder_on_rollover=True,
        **streamer_kwargs,
    )


def test_streamer_rolling_parity_end_to_end_with_rollovers():
    torch.manual_seed(20)
    mel_chunks = [torch.randn(1, 24, N_MELS) for _ in range(6)]

    old = make_streamer(make_rolling_model(), IntTokenizer(), rolling=False)
    new = make_streamer(make_rolling_model(), IntTokenizer(), rolling=True)

    saw_rollover = False
    saw_draft = False
    for chunk in mel_chunks:
        event_old = old.append_mel_chunk(chunk.clone())
        event_new = new.append_mel_chunk(chunk.clone())
        assert event_new["hypothesis"] == event_old["hypothesis"]
        assert event_new["committed"] == event_old["committed"]
        assert "generate_ms" in event_new
        if event_new.get("segment_rollover"):
            saw_rollover = True
            assert getattr(new.state, "decoder", None) is None
        elif event_new.get("decoder_path", "").startswith("rolling"):
            assert new.state.decoder is not None
        if event_new.get("draft_tokens"):
            saw_draft = True
    assert saw_rollover, "test must cross at least one segment rollover"
    assert saw_draft, "speculative draft never engaged"
    assert (
        new.finalize(finalize_mode="latest").final_text
        == old.finalize(finalize_mode="latest").final_text
    )


def test_roll_before_generate_skips_over_limit_generation():
    streamer = make_streamer(
        make_rolling_model(),
        IntTokenizer(),
        rolling=True,
        segment_roll_before_generate=True,
    )
    torch.manual_seed(22)
    # 24 mel frames = 3 steps per chunk on the tiny tower; cap is 5 steps, so
    # chunk 2 would land at 6 steps: the roll must happen BEFORE its decode.
    events = [streamer.append_mel_chunk(torch.randn(1, 24, N_MELS)) for _ in range(4)]

    pre_rolls = [e for e in events if e.get("segment_rolled_before_generate")]
    assert pre_rolls, "pre-generate rollover never triggered"
    # No generation ever ran on an over-cap cache, and the boundary chunk's
    # audio opened the next segment.
    assert all(e["cached_steps"] <= 5 for e in events)
    assert all(e["segment_rollover"] is False for e in events)
    assert streamer.segments_finalized >= 1
    assert streamer.finalize(finalize_mode="latest").final_text


def test_segment_prompt_template_encoded_once_per_segment():
    tokenizer = IntTokenizer()
    streamer = make_streamer(
        make_rolling_model(),
        tokenizer,
        rolling=True,
        segment_prompt_context_words=4,
        segment_prompt_language="English",
    )
    torch.manual_seed(21)
    rollovers = 0
    for _ in range(4):
        event = streamer.append_mel_chunk(torch.randn(1, 16, N_MELS))
        rollovers += int(bool(event.get("segment_rollover")))
    # One encode for the first segment's template plus one per rollover.
    assert tokenizer.encode_calls == 1 + rollovers


def test_flush_pending_audio_decodes_buffered_partial_block():
    """40 mel frames at block_frames=32: forward emits 1 block (4 steps),
    8 frames stay pending; the flush encodes exactly one more conv block."""
    from whisperlivekit.qwen3_streaming.causal import (
        Qwen3ASRRealtimeQwenAudioCausalModel,
    )
    from qwen3_streaming_fakes import (
        D_MODEL,
        FakeCachingQwenTextModel,
        TinyQwenAudioTower,
        tiny_config,
    )

    torch.manual_seed(0)
    model = Qwen3ASRRealtimeQwenAudioCausalModel(
        tiny_config(
            qwen_audio_block_bidirectional=True, qwen_audio_block_frames=32
        ),
        qwen_model_id="fake",
        audio_tower=TinyQwenAudioTower(),
        text_model=FakeCachingQwenTextModel(),
        lm_head=torch.nn.Linear(D_MODEL, 16, bias=False),
        bos_token_id=1,
        wait_token_id=0,
    ).eval()
    streamer = make_streamer(model, IntTokenizer(), rolling=True)

    torch.manual_seed(21)
    event = streamer.append_mel_chunk(torch.randn(1, 40, N_MELS))
    assert event["cached_steps"] == 4  # one full 32-frame block

    flush_event = streamer.flush_pending_audio()
    assert flush_event is not None
    assert flush_event["is_flush"] is True
    assert flush_event["cached_steps"] == 5  # +1 step from the 8 pending frames

    assert streamer.flush_pending_audio() is None  # idempotent
