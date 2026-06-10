"""Bounded mutable-tail tests for QwenAudioCausalKVEncoder.

Uses a tiny tower with the real Qwen audio module geometry (conv2d stack,
per-layer self-attention, ln_post/proj head) so the per-layer KV freeze path
is exercised — the generic FakeQwenAudioTower only reaches the fallback path.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from qwen3_streaming.native_realtime_model import (  # noqa: E402
    Qwen3ASRRealtimeQwenAudioCausalModel,
    QwenAudioCausalKVEncoder,
)
from qwen3_streaming.realtime_config import RealtimeAudioConfig  # noqa: E402

D_MODEL = 32
N_MELS = 128


class _TinyAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)


class _TinyLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = _TinyAttention(d_model, num_heads)
        self.self_attn_layer_norm = torch.nn.LayerNorm(d_model)
        self.final_layer_norm = torch.nn.LayerNorm(d_model)
        self.fc1 = torch.nn.Linear(d_model, 2 * d_model)
        self.fc2 = torch.nn.Linear(2 * d_model, d_model)
        self.activation_fn = torch.nn.GELU()
        self.dropout = 0.0
        self.activation_dropout = 0.0


class _PositionalEmbedding(torch.nn.Module):
    def __init__(self, max_positions: int, d_model: int) -> None:
        super().__init__()
        half = d_model // 2
        inv = torch.exp(
            -math.log(10000.0) / float(max(1, half - 1)) * torch.arange(half)
        )
        positions = torch.arange(max_positions, dtype=torch.float32)
        scaled = positions[:, None] * inv[None, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([scaled.sin(), scaled.cos()], dim=1),
        )


class TinyQwenAudioTower(torch.nn.Module):
    """Real-geometry miniature of Qwen3-ASR's audio tower (8 mel frames -> 1 step)."""

    def __init__(self, n_mels: int = N_MELS, d_model: int = D_MODEL, num_layers: int = 2):
        super().__init__()
        self.conv2d1 = torch.nn.Conv2d(1, 4, 3, stride=2, padding=1)
        self.conv2d2 = torch.nn.Conv2d(4, 4, 3, stride=2, padding=1)
        self.conv2d3 = torch.nn.Conv2d(4, 4, 3, stride=2, padding=1)
        self.conv_out = torch.nn.Linear(4 * (n_mels // 8), d_model)
        self.positional_embedding = _PositionalEmbedding(4096, d_model)
        self.layers = torch.nn.ModuleList(
            [_TinyLayer(d_model, num_heads=2) for _ in range(num_layers)]
        )
        self.ln_post = torch.nn.LayerNorm(d_model)
        self.proj1 = torch.nn.Linear(d_model, d_model)
        self.act = torch.nn.GELU()
        self.proj2 = torch.nn.Linear(d_model, d_model)

    def _get_feat_extract_output_lengths(self, lengths):
        return lengths // 8


def tiny_config(**overrides) -> RealtimeAudioConfig:
    kwargs = dict(d_model=D_MODEL, audio_window_sec=30.0)
    kwargs.update(overrides)
    return RealtimeAudioConfig(**kwargs)


def make_encoder(mutable_tail_sec: float, **config_overrides) -> QwenAudioCausalKVEncoder:
    torch.manual_seed(0)
    tower = TinyQwenAudioTower().eval()
    config = tiny_config(**config_overrides)
    return QwenAudioCausalKVEncoder(
        tower, config, mutable_tail_sec=mutable_tail_sec
    ).eval()


def run_chunks(encoder, mels, chunk_sizes):
    state = encoder.init_state()
    outputs = []
    cursor = 0
    while cursor < mels.shape[1]:
        size = chunk_sizes[len(outputs) % len(chunk_sizes)]
        hidden, state = encoder.forward_chunk(mels[:, cursor : cursor + size, :], state)
        outputs.append(hidden)
        cursor += size
    return outputs, state


def test_strict_chunked_matches_full_causal_reference():
    encoder = make_encoder(mutable_tail_sec=0.0)
    torch.manual_seed(1)
    mels = torch.randn(1, 64, N_MELS)

    full = encoder.forward_full(mels)
    outputs, state = run_chunks(encoder, mels, chunk_sizes=[16, 8, 24, 16])
    chunked = torch.cat(outputs, dim=1)

    assert state.mutable_steps == 0
    torch.testing.assert_close(chunked, full, rtol=1e-4, atol=1e-4)


def test_unbounded_tail_final_output_matches_full_causal():
    # Tail larger than the whole stream: nothing ever freezes, so the last
    # chunk's re-decoded tail must equal the one-shot causal encoding.
    encoder = make_encoder(mutable_tail_sec=60.0)
    torch.manual_seed(2)
    mels = torch.randn(1, 64, N_MELS)

    full = encoder.forward_full(mels)
    outputs, state = run_chunks(encoder, mels, chunk_sizes=[16, 8, 24, 16])

    assert state.emitted_steps == 0
    assert state.mutable_steps == 8  # 64 mel frames -> 8 steps, all mutable
    torch.testing.assert_close(outputs[-1], full, rtol=1e-4, atol=1e-4)


def test_bounded_tail_freezes_old_steps_and_respects_budget():
    # 0.16s tail = 16 mel frames = 2 steps budget.
    encoder = make_encoder(mutable_tail_sec=0.16)
    assert encoder.mutable_tail_steps == 2
    torch.manual_seed(3)
    mels = torch.randn(1, 96, N_MELS)

    outputs, state = run_chunks(encoder, mels, chunk_sizes=[16])

    total_steps = 96 // 8
    assert state.emitted_steps + state.mutable_steps == total_steps
    # Block granularity: the mutable tail holds at least the budget and at
    # most budget + one block worth of steps.
    assert encoder.mutable_tail_steps <= state.mutable_steps <= encoder.mutable_tail_steps + 2
    assert state.emitted_steps > 0
    # Frozen KV caches bounded by the left context.
    for cache in state.layer_caches:
        assert cache.key is not None
        assert cache.key.shape[-2] <= encoder.left_context_steps


def test_bounded_tail_recompute_cost_is_bounded():
    encoder = make_encoder(mutable_tail_sec=0.16)
    torch.manual_seed(4)
    mels = torch.randn(1, 160, N_MELS)

    state = encoder.init_state()
    max_recompute = 0
    for start in range(0, 160, 16):
        _, state = encoder.forward_chunk(mels[:, start : start + 16, :], state)
        max_recompute = max(max_recompute, state.last_recomputed_frames)
    # Recompute is bounded by tail mels (<= (budget + block) * 8 frames) + new chunk.
    tail_bound_frames = (encoder.mutable_tail_steps + 2) * 8
    assert max_recompute <= tail_bound_frames + 16


def test_model_level_overwrite_keeps_frozen_prefix_stable():
    torch.manual_seed(5)
    tower = TinyQwenAudioTower().eval()
    config = tiny_config(qwen_audio_mutable_tail_sec=0.16)

    class FakeQwenTextModel(torch.nn.Module):
        def __init__(self, vocab_size: int, hidden_size: int) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
            self.layers = torch.nn.ModuleList()
            self.norm = torch.nn.Identity()

        def forward(self, inputs_embeds, **_kwargs):
            class Output:
                def __init__(self, last_hidden_state):
                    self.last_hidden_state = last_hidden_state

            return Output(inputs_embeds)

    model = Qwen3ASRRealtimeQwenAudioCausalModel(
        config,
        qwen_model_id="fake",
        audio_tower=tower,
        text_model=FakeQwenTextModel(vocab_size=8, hidden_size=D_MODEL),
        lm_head=torch.nn.Linear(D_MODEL, 8, bias=False),
        bos_token_id=1,
        wait_token_id=0,
    ).eval()
    assert model.audio_encoder.mutable_tail_steps == 2

    torch.manual_seed(6)
    mels = torch.randn(1, 96, N_MELS)
    state = model.init_cached_audio_decode_state()

    snapshots = []
    for start in range(0, 96, 16):
        cached, delta, state = model.append_audio_to_cache(
            mels[:, start : start + 16, :], state
        )
        snapshots.append(cached.clone())
        assert state.adapter.decoder_steps_seen == cached.shape[1]

    # Final cache covers every step exactly once.
    assert snapshots[-1].shape[1] == 96 // 8

    # Steps outside the mutable tail never change after being frozen.
    for earlier, later in zip(snapshots, snapshots[1:]):
        frozen_len = earlier.shape[1] - int(model.audio_encoder.mutable_tail_steps + 2)
        if frozen_len <= 0:
            continue
        torch.testing.assert_close(
            later[:, :frozen_len, :],
            earlier[:, :frozen_len, :],
            rtol=1e-4,
            atol=1e-4,
        )


def test_strict_mode_unchanged_at_model_level():
    torch.manual_seed(7)
    tower = TinyQwenAudioTower().eval()
    config = tiny_config()  # mutable tail defaults to 0

    class FakeQwenTextModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, D_MODEL)
            self.layers = torch.nn.ModuleList()
            self.norm = torch.nn.Identity()

        def forward(self, inputs_embeds, **_kwargs):
            class Output:
                def __init__(self, last_hidden_state):
                    self.last_hidden_state = last_hidden_state

            return Output(inputs_embeds)

    model = Qwen3ASRRealtimeQwenAudioCausalModel(
        config,
        qwen_model_id="fake",
        audio_tower=tower,
        text_model=FakeQwenTextModel(),
        lm_head=torch.nn.Linear(D_MODEL, 8, bias=False),
        bos_token_id=1,
        wait_token_id=0,
    ).eval()
    assert model.audio_encoder.mutable_tail_steps == 0

    torch.manual_seed(8)
    mels = torch.randn(1, 32, N_MELS)
    state = model.init_cached_audio_decode_state()
    cached1, delta1, state = model.append_audio_to_cache(mels[:, :16, :], state)
    cached2, delta2, state = model.append_audio_to_cache(mels[:, 16:, :], state)

    # Strict append-only: previously cached steps are bit-identical.
    torch.testing.assert_close(cached2[:, : cached1.shape[1], :], cached1)
    assert delta2.shape[1] == cached2.shape[1] - cached1.shape[1]
    assert state.audio.last_recomputed_context_frames == 0


def test_block_bidirectional_sees_future_within_block_only():
    torch.manual_seed(9)
    mels = torch.randn(1, 32, N_MELS)
    altered = mels.clone()
    altered[:, 16:, :] += 1.0  # change only the second half of the block

    for flag, expect_change in ((False, False), (True, True)):
        torch.manual_seed(0)
        tower = TinyQwenAudioTower().eval()
        config = tiny_config(qwen_audio_block_bidirectional=flag)
        encoder = QwenAudioCausalKVEncoder(tower, config).eval()

        out_a, _ = encoder.forward_chunk(mels, encoder.init_state())
        out_b, _ = encoder.forward_chunk(altered, encoder.init_state())
        # First half steps: 16 frames -> 2 steps
        changed = not torch.allclose(out_a[:, :2, :], out_b[:, :2, :], atol=1e-5)
        assert changed == expect_change, (
            f"flag={flag}: early-step dependence on later block content "
            f"should be {expect_change}"
        )


def test_block_bidirectional_stays_append_only_across_blocks():
    torch.manual_seed(0)
    tower = TinyQwenAudioTower().eval()
    config = tiny_config(qwen_audio_block_bidirectional=True)
    encoder = QwenAudioCausalKVEncoder(tower, config).eval()
    torch.manual_seed(10)
    mels = torch.randn(1, 64, N_MELS)

    state = encoder.init_state()
    first, state = encoder.forward_chunk(mels[:, :32, :], state)
    second, state = encoder.forward_chunk(mels[:, 32:, :], state)

    # Emitted steps from block 1 are frozen; block 2 only appends.
    state2 = encoder.init_state()
    first_again, _ = encoder.forward_chunk(mels[:, :32, :], state2)
    torch.testing.assert_close(first, first_again, rtol=1e-5, atol=1e-5)
    assert state.last_recomputed_context_frames == 0
    assert second.shape[1] == 4  # 32 frames -> 4 steps
