"""Shared synthetic models for qwen3-streaming decoder/encoder tests.

Ported from experiments/qwen3-causal/tests/{test_mutable_tail,test_decoder_kv_cache,
test_decoder_rolling_kv}.py @ 9d4b99a. Real Qwen audio-tower geometry (conv2d
stack, per-layer attention, ln_post/proj head) and history-dependent text-model
fakes (hidden[t] = mean of all previous inputs) so cache mis-wiring cannot pass.
"""

import math

import torch

from whisperlivekit.qwen3_streaming.model_config import RealtimeAudioConfig

D_MODEL = 32
N_MELS = 128
VOCAB = 16


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
    kwargs = dict(d_model=D_MODEL, audio_window_sec=30.0, qwen_audio_left_context_sec=2.0)
    kwargs.update(overrides)
    return RealtimeAudioConfig(**kwargs)


class FakeCache:
    def __init__(self):
        self.running_sum = None  # [B, D]
        self.count = 0

    def get_seq_length(self):
        return self.count


class _Out:
    def __init__(self, last_hidden_state, past_key_values=None):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values


class FakeCachingQwenTextModel(torch.nn.Module):
    """hidden[t] = mean(inputs[0..t]) — strictly history-dependent."""

    def __init__(self, vocab_size: int = VOCAB, hidden_size: int = D_MODEL):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList()
        self.norm = torch.nn.Identity()
        self.forwarded_positions = 0
        self.calls = 0

    def forward(self, inputs_embeds, past_key_values=None, use_cache=False, **_kwargs):
        self.calls += 1
        self.forwarded_positions += int(inputs_embeds.shape[1])
        if past_key_values is None:
            cache = FakeCache() if use_cache else None
            running = torch.zeros(
                inputs_embeds.shape[0], inputs_embeds.shape[2],
                dtype=inputs_embeds.dtype, device=inputs_embeds.device,
            )
            count = 0
        else:
            cache = past_key_values
            running = cache.running_sum.clone()
            count = cache.count
        hidden = []
        for t in range(inputs_embeds.shape[1]):
            running = running + inputs_embeds[:, t, :]
            count += 1
            hidden.append(running / count)
        out = torch.stack(hidden, dim=1)
        if cache is not None:
            cache.running_sum = running
            cache.count = count
        return _Out(out, past_key_values=cache if use_cache else None)
