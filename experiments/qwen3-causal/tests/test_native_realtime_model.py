import pytest

torch = pytest.importorskip("torch")

from qwen3_streaming.native_realtime_model import (  # noqa: E402
    LoRALinear,
    Qwen3ASRRealtimeQwenDecoderModel,
    Qwen3ASRRealtimeNativeModel,
    QwenAudioSurgeryEncoder,
    QwenAudioSurgeryFrameAdapter,
    StreamingFrameAdapter,
    CausalAudioEncoder,
    _apply_repetition_controls_to_logits,
    _banned_ngram_tokens,
    add_lora_to_linear_modules,
    configure_compact_ctc_head,
    configure_rnnt_lite_head,
)
from qwen3_streaming.metrics import token_repetition_stats  # noqa: E402
from qwen3_streaming.realtime_config import RealtimeAudioConfig  # noqa: E402


def tiny_config() -> RealtimeAudioConfig:
    return RealtimeAudioConfig(
        d_model=32,
        audio_num_layers=2,
        audio_num_heads=4,
        audio_ffn_multiplier=2,
        conv_kernel_size=3,
        audio_window_sec=30.0,
        dropout=0.0,
    )


def test_causal_audio_encoder_chunked_matches_full_causal_reference():
    torch.manual_seed(0)
    config = tiny_config()
    encoder = CausalAudioEncoder(config).eval()
    mels = torch.randn(2, 23, config.n_mels)

    full = encoder.forward_full(mels)
    state = encoder.init_state()
    chunks = []
    cursor = 0
    for size in [5, 7, 11]:
        out, state = encoder.forward_chunk(mels[:, cursor : cursor + size, :], state)
        chunks.append(out)
        cursor += size
    chunked = torch.cat(chunks, dim=1)

    torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-5)
    assert state.frames_seen == mels.shape[1]


def test_frame_adapter_emits_one_decoder_step_per_80ms():
    torch.manual_seed(0)
    config = tiny_config()
    adapter = StreamingFrameAdapter(config).eval()
    hidden = torch.randn(1, 20, config.d_model)

    full = adapter.forward_full(hidden)
    state = adapter.init_state()
    out1, state = adapter.forward_chunk(hidden[:, :5, :], state)
    out2, state = adapter.forward_chunk(hidden[:, 5:10, :], state)
    out3, state = adapter.forward_chunk(hidden[:, 10:, :], state)
    chunked = torch.cat([out1, out2, out3], dim=1)

    assert full.shape[1] == 2
    assert state.pending is not None
    assert state.pending.shape[1] == 4
    torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-5)


def test_native_model_streaming_keeps_state_and_emits_only_completed_frames():
    torch.manual_seed(0)
    config = tiny_config()
    model = Qwen3ASRRealtimeNativeModel(
        config,
        vocab_size=32,
        bos_token_id=1,
        decoder_num_layers=1,
        decoder_num_heads=4,
        decoder_ffn_multiplier=2,
    ).eval()

    state = model.init_stream_state(batch_size=1, device="cpu")
    logits1, tokens1, state = model.stream_chunk(torch.randn(1, 20, config.n_mels), state)
    logits2, tokens2, state = model.stream_chunk(torch.randn(1, 8, config.n_mels), state)

    assert logits1.shape == (1, 2, 32)
    assert tokens1.shape == (1, 2)
    assert logits2.shape == (1, 1, 32)
    assert tokens2.shape == (1, 1)
    assert state.audio.frames_seen == 28
    assert state.decoder.steps_seen == 3
    assert state.adapter.pending is not None
    assert state.adapter.pending.shape[1] == 4


def test_native_model_save_and_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    config = tiny_config()
    model = Qwen3ASRRealtimeNativeModel(
        config,
        vocab_size=32,
        bos_token_id=1,
        decoder_num_layers=1,
        decoder_num_heads=4,
        decoder_ffn_multiplier=2,
    ).eval()

    model.save_pretrained(tmp_path)
    loaded = Qwen3ASRRealtimeNativeModel.from_pretrained(tmp_path).eval()
    mels = torch.randn(1, 16, config.n_mels)
    previous = torch.ones(1, 2, dtype=torch.long)

    torch.testing.assert_close(model(mels, previous), loaded(mels, previous))
    torch.testing.assert_close(model.forward_ctc_logits(mels), loaded.forward_ctc_logits(mels))


def test_native_model_save_and_load_roundtrip_with_compact_ctc_head(tmp_path):
    torch.manual_seed(0)
    config = tiny_config()
    model = Qwen3ASRRealtimeNativeModel(
        config,
        vocab_size=32,
        bos_token_id=1,
        decoder_num_layers=1,
        decoder_num_heads=4,
        decoder_ffn_multiplier=2,
    ).eval()
    configure_compact_ctc_head(model, [0, 5, 7], blank_index=0)

    model.save_pretrained(tmp_path)
    loaded = Qwen3ASRRealtimeNativeModel.from_pretrained(tmp_path).eval()
    mels = torch.randn(1, 16, config.n_mels)

    assert loaded.compact_ctc_token_ids == [0, 5, 7]
    torch.testing.assert_close(
        model.forward_compact_ctc_logits(mels),
        loaded.forward_compact_ctc_logits(mels),
    )


def test_native_model_save_and_load_roundtrip_with_rnnt_lite_head(tmp_path):
    torch.manual_seed(0)
    config = tiny_config()
    model = Qwen3ASRRealtimeNativeModel(
        config,
        vocab_size=32,
        bos_token_id=1,
        decoder_num_layers=1,
        decoder_num_heads=4,
        decoder_ffn_multiplier=2,
    ).eval()
    configure_rnnt_lite_head(model, [0, 5, 7], blank_index=0)

    model.save_pretrained(tmp_path)
    loaded = Qwen3ASRRealtimeNativeModel.from_pretrained(tmp_path).eval()
    mels = torch.randn(1, 16, config.n_mels)
    previous = torch.tensor([[0, 1]], dtype=torch.long)
    prefixes = torch.tensor([[0, 1, 2]], dtype=torch.long)

    assert loaded.rnnt_lite_token_ids == [0, 5, 7]
    torch.testing.assert_close(
        model.forward_rnnt_lite_logits(mels, previous),
        loaded.forward_rnnt_lite_logits(mels, previous),
    )
    torch.testing.assert_close(
        model.forward_rnnt_lite_joint_logits(mels, prefixes),
        loaded.forward_rnnt_lite_joint_logits(mels, prefixes),
    )


def test_lora_linear_wraps_target_modules_without_changing_initial_output():
    torch.manual_seed(0)

    class ToyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.other = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.q_proj(x) + self.other(x)

    model = ToyBlock().eval()
    x = torch.randn(2, 3, 8)
    before = model(x)

    replaced = add_lora_to_linear_modules(
        model,
        target_names=("q_proj",),
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )

    assert replaced == ["q_proj"]
    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.other, torch.nn.Linear)
    assert not model.q_proj.base.weight.requires_grad
    assert model.q_proj.lora_a.weight.requires_grad
    assert model.q_proj.lora_b.weight.requires_grad
    torch.testing.assert_close(model(x), before)


class FakeQwenAudioTower(torch.nn.Module):
    def __init__(self, n_mels: int, output_dim: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(n_mels, output_dim, bias=False)
        torch.nn.init.uniform_(self.proj.weight, a=-0.02, b=0.02)

    def _get_feat_extract_output_lengths(self, input_lengths):
        return input_lengths // 8

    def forward(self, input_features, feature_lens=None):
        frames = input_features.transpose(0, 1)
        usable = (frames.shape[0] // 8) * 8
        if usable == 0:
            return frames.new_zeros(0, self.proj.out_features)
        pooled = frames[:usable, :].view(usable // 8, 8, frames.shape[-1]).mean(dim=1)
        return self.proj(pooled)


class FakeQwenTextOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = None


class FakeQwenTextModel(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList()
        self.norm = torch.nn.Identity()

    def forward(self, inputs_embeds, **_kwargs):
        return FakeQwenTextOutput(inputs_embeds)


def test_cached_audio_decode_accumulates_finalized_audio_frames():
    torch.manual_seed(0)
    config = tiny_config()
    encoder = QwenAudioSurgeryEncoder(
        FakeQwenAudioTower(config.n_mels, config.d_model),
        config,
        left_context_frames=16,
        right_context_frames=8,
    ).eval()
    model = Qwen3ASRRealtimeQwenDecoderModel(
        config,
        qwen_model_id="fake",
        text_model=FakeQwenTextModel(vocab_size=32, hidden_size=config.d_model),
        lm_head=torch.nn.Linear(config.d_model, 32, bias=False),
        bos_token_id=1,
        wait_token_id=0,
        audio_encoder=encoder,
        adapter=QwenAudioSurgeryFrameAdapter(config.d_model, config.d_model),
    ).eval()
    mels = torch.randn(1, 32, config.n_mels)
    state = model.init_cached_audio_decode_state()

    cached1, delta1, state = model.append_audio_to_cache(mels[:, :16, :], state)
    cached2, delta2, state = model.append_audio_to_cache(mels[:, 16:, :], state)

    assert cached1.shape == (1, 1, config.d_model)
    assert delta1.shape == (1, 1, config.d_model)
    assert delta2.shape == (1, 2, config.d_model)
    assert cached2.shape == (1, 3, config.d_model)
    torch.testing.assert_close(cached2[:, : cached1.shape[1], :], cached1)
    assert state.decoder_steps_seen == 3
    assert state.audio.last_input_frames == 16
    assert state.audio.last_recomputed_frames <= 24
    assert state.audio.last_recomputed_context_frames <= 8


def test_generate_full_hypothesis_from_cached_audio_runs_greedy_decoder():
    torch.manual_seed(0)
    config = tiny_config()
    model = Qwen3ASRRealtimeQwenDecoderModel(
        config,
        qwen_model_id="fake",
        text_model=FakeQwenTextModel(vocab_size=8, hidden_size=config.d_model),
        lm_head=torch.nn.Linear(config.d_model, 8, bias=False),
        bos_token_id=1,
        wait_token_id=0,
    ).eval()
    for param in model.lm_head.parameters():
        torch.nn.init.zeros_(param)
    frame_hidden = torch.randn(1, 3, config.d_model)

    tokens = model.generate_full_hypothesis_from_cached_audio(
        frame_hidden,
        max_new_tokens=4,
        suppress_token_ids=[0],
    )

    assert tokens.shape == (1, 4)
    assert 0 not in tokens.reshape(-1).tolist()


def test_generate_full_hypothesis_replaces_audio_prompt_placeholders():
    torch.manual_seed(0)
    config = tiny_config()
    model = Qwen3ASRRealtimeQwenDecoderModel(
        config,
        qwen_model_id="fake",
        text_model=FakeQwenTextModel(vocab_size=8, hidden_size=config.d_model),
        lm_head=torch.nn.Linear(config.d_model, 8, bias=False),
        bos_token_id=1,
        wait_token_id=0,
    ).eval()
    frame_hidden = torch.randn(1, 3, config.d_model)

    tokens = model.generate_full_hypothesis_from_cached_audio(
        frame_hidden,
        prefix_token_ids=[2, 7, 7, 7, 3],
        audio_placeholder_token_id=7,
        max_new_tokens=2,
    )

    assert tokens.shape == (1, 2)


def test_generate_full_hypothesis_rejects_wrong_placeholder_count():
    torch.manual_seed(0)
    config = tiny_config()
    model = Qwen3ASRRealtimeQwenDecoderModel(
        config,
        qwen_model_id="fake",
        text_model=FakeQwenTextModel(vocab_size=8, hidden_size=config.d_model),
        lm_head=torch.nn.Linear(config.d_model, 8, bias=False),
        bos_token_id=1,
        wait_token_id=0,
    ).eval()
    frame_hidden = torch.randn(1, 3, config.d_model)

    with pytest.raises(ValueError, match="audio placeholder"):
        model.generate_full_hypothesis_from_cached_audio(
            frame_hidden,
            prefix_token_ids=[2, 7, 3],
            audio_placeholder_token_id=7,
            max_new_tokens=2,
        )


def test_qwen_audio_surgery_encoder_recomputes_only_bounded_window():
    torch.manual_seed(0)
    config = tiny_config()
    encoder = QwenAudioSurgeryEncoder(
        FakeQwenAudioTower(config.n_mels, config.d_model),
        config,
        left_context_frames=16,
        right_context_frames=8,
    ).eval()
    mels = torch.randn(1, 64, config.n_mels)

    state = encoder.init_state()
    outputs = []
    for start in range(0, mels.shape[1], 16):
        out, state = encoder.forward_chunk(mels[:, start : start + 16, :], state)
        outputs.append(out)

    chunked = torch.cat(outputs, dim=1)
    assert chunked.shape == (1, encoder.output_steps_for_mel_frames(56), config.d_model)
    assert state.frames_seen == 64
    assert state.emitted_steps == encoder.output_steps_for_mel_frames(56)
    assert state.last_input_frames == 16
    assert state.last_recomputed_frames <= 24
    assert state.last_recomputed_context_frames == 8
    assert state.window_start_frame == 40


def test_qwen_audio_surgery_waits_for_right_context_before_emitting():
    torch.manual_seed(0)
    config = tiny_config()
    encoder = QwenAudioSurgeryEncoder(
        FakeQwenAudioTower(config.n_mels, config.d_model),
        config,
        left_context_frames=16,
        right_context_frames=16,
    ).eval()
    state = encoder.init_state()

    out, state = encoder.forward_chunk(torch.randn(1, 8, config.n_mels), state)

    assert out.shape == (1, 0, config.d_model)
    assert state.frames_seen == 8
    assert state.emitted_steps == 0


def test_qwen_audio_surgery_adapter_is_identity_initialized_when_dims_match():
    torch.manual_seed(0)
    adapter = QwenAudioSurgeryFrameAdapter(input_dim=16, output_dim=16).eval()
    hidden = torch.randn(2, 5, 16)

    projected = adapter.forward_full(hidden)

    torch.testing.assert_close(projected, hidden)


def test_qwen_audio_surgery_adapter_residual_blocks_add_capacity():
    torch.manual_seed(0)
    adapter = QwenAudioSurgeryFrameAdapter(
        input_dim=16,
        output_dim=16,
        adapter_hidden_dim=32,
        adapter_layers=2,
        adapter_residual_scale=0.2,
    ).eval()
    hidden = torch.randn(2, 5, 16)

    projected = adapter.forward_full(hidden)

    assert projected.shape == hidden.shape
    assert len(adapter.blocks) == 2
    assert sum(param.numel() for param in adapter.blocks.parameters()) > 0
    assert not torch.allclose(projected, hidden)


def test_token_repetition_stats_ignores_wait_and_counts_ngrams():
    stats = token_repetition_stats(
        [0, 10, 11, 12, 10, 11, 12],
        ignored_token_ids={0},
    )

    assert stats["text_token_count"] == 6
    assert stats["unigram_repeated"] == 3
    assert stats["bigram_repeated"] == 2
    assert stats["trigram_repeated"] == 1
    assert stats["trigram_repetition_ratio"] == 0.25


def test_no_repeat_ngram_bans_only_recreated_continuations():
    history = [10, 20, 30, 10, 20, 40]

    assert _banned_ngram_tokens(history[:-1], 3) == {30}
    assert _banned_ngram_tokens(history, 3) == set()
    assert _banned_ngram_tokens(history, 1) == {10, 20, 30, 40}


def test_repetition_penalty_default_keeps_logits_unchanged():
    logits = torch.tensor([[2.0, -2.0, 1.0]])

    controlled = _apply_repetition_controls_to_logits(
        logits,
        token_history=[[0, 1]],
        consecutive_text_tokens=torch.tensor([0]),
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        max_consecutive_text_tokens=0,
        wait_token_id=2,
    )

    torch.testing.assert_close(controlled, logits)


def test_repetition_penalty_scales_seen_token_logits():
    logits = torch.tensor([[2.0, -2.0, 1.0]])

    controlled = _apply_repetition_controls_to_logits(
        logits,
        token_history=[[0, 1]],
        consecutive_text_tokens=torch.tensor([0]),
        repetition_penalty=2.0,
        no_repeat_ngram_size=0,
        max_consecutive_text_tokens=0,
        wait_token_id=2,
    )

    torch.testing.assert_close(controlled, torch.tensor([[1.0, -4.0, 1.0]]))


def test_max_consecutive_text_tokens_forces_wait_token():
    logits = torch.tensor([[0.0, 8.0, 4.0]])

    controlled = _apply_repetition_controls_to_logits(
        logits,
        token_history=[[1, 2]],
        consecutive_text_tokens=torch.tensor([2]),
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        max_consecutive_text_tokens=2,
        wait_token_id=0,
    )

    assert int(controlled.argmax(dim=-1).item()) == 0
    assert controlled[0, 1] < -1e20
    assert controlled[0, 2] < -1e20
