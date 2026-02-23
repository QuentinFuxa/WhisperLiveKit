"""
Voxtral Realtime MLX model — encoder, decoder, adapter, and top-level model.

Architecture:
    audio → StreamingEncoder → EncoderToDecoderAdapter → TextDecoder → logits
    with DelayEmbedding providing time-conditioning to the decoder.

The model supports both batch inference (full audio) and incremental streaming
(one chunk at a time with cached encoder/decoder state).
"""

import math

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------


class SlidingKVCache:
    """Bounded key-value cache with rotating buffer for sliding-window attention.

    Uses in-place writes for single-token autoregressive steps and
    concatenation for multi-token prefills. Pre-allocates in blocks of
    ``alloc_step`` entries to reduce repeated allocation.
    """

    alloc_step = 256

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.keys = None
        self.values = None
        self._offset = 0
        self._write_idx = 0

    @property
    def offset(self) -> int:
        return self._offset

    # -- helpers --

    def _reorder(self, buf):
        """Return *buf* in temporal order (unwrap the circular buffer)."""
        if self._write_idx == buf.shape[2]:
            return buf
        if self._write_idx < self._offset:
            return mx.concatenate(
                [buf[..., self._write_idx:, :], buf[..., : self._write_idx, :]],
                axis=2,
            )
        return buf[..., : self._write_idx, :]

    def _drop_oldest(self, buf, n_drop, tail=None):
        parts = [buf[..., n_drop:, :]] if n_drop > 0 else [buf]
        if tail is not None:
            parts.append(tail)
        return mx.concatenate(parts, axis=2)

    # -- update strategies --

    def _append_concat(self, k, v):
        """Multi-token update via concatenation (used during prefill)."""
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = self._reorder(self.keys)
            self.values = self._reorder(self.values)
            self._write_idx = self.keys.shape[2]
            overflow = self._write_idx - self.capacity + 1
            self.keys = self._drop_oldest(self.keys, overflow, k)
            self.values = self._drop_oldest(self.values, overflow, v)
        self._offset += k.shape[2]
        self._write_idx = self.keys.shape[2]
        return self.keys, self.values

    def _write_inplace(self, k, v):
        """Single-token update via in-place write (autoregressive step)."""
        B, n_heads, S, dim_k = k.shape
        dim_v = v.shape[3]
        prev = self._offset

        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.capacity
        ):
            n_new = min(self.alloc_step, self.capacity - prev)
            fresh_k = mx.zeros((B, n_heads, n_new, dim_k), k.dtype)
            fresh_v = mx.zeros((B, n_heads, n_new, dim_v), v.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, fresh_k], axis=2)
                self.values = mx.concatenate([self.values, fresh_v], axis=2)
            else:
                self.keys, self.values = fresh_k, fresh_v
            self._write_idx = prev

        overflow = self.keys.shape[2] - self.capacity
        if overflow > 0:
            self.keys = self._drop_oldest(self.keys, overflow)
            self.values = self._drop_oldest(self.values, overflow)
            self._write_idx = self.capacity

        if self._write_idx == self.capacity:
            self._write_idx = 0

        self.keys[..., self._write_idx : self._write_idx + S, :] = k
        self.values[..., self._write_idx : self._write_idx + S, :] = v
        self._offset += S
        self._write_idx += S

        if self._offset < self.capacity:
            return (
                self.keys[..., : self._offset, :],
                self.values[..., : self._offset, :],
            )
        return self.keys, self.values

    # -- public API --

    def update_and_fetch(self, k, v):
        if k.shape[2] == 1:
            return self._write_inplace(k, v)
        return self._append_concat(k, v)


# ---------------------------------------------------------------------------
# Encoder components
# ---------------------------------------------------------------------------


class CausalConv(nn.Module):
    """1-D causal convolution (left-padded so no future leakage)."""

    def __init__(self, channels_in: int, channels_out: int, kernel: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.kernel = kernel
        self.left_pad = kernel - stride
        self.weight = mx.zeros((channels_out, kernel, channels_in))
        self.bias = mx.zeros((channels_out,))

    def __call__(self, x: mx.array) -> mx.array:
        if self.left_pad > 0:
            x = mx.pad(x, [(0, 0), (self.left_pad, 0), (0, 0)])
        return mx.conv1d(x, self.weight, stride=self.stride) + self.bias


class _EncoderSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, rope_theta: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=True)
        self.rope_theta = rope_theta

    def __call__(self, x, mask, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        pos = cache.offset if cache is not None else 0
        q = mx.fast.rope(q, self.head_dim, traditional=True, base=self.rope_theta, scale=1.0, offset=pos)
        k = mx.fast.rope(k, self.head_dim, traditional=True, base=self.rope_theta, scale=1.0, offset=pos)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class _EncoderFFN(nn.Module):
    """SwiGLU feed-forward for encoder layers."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=True)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class _EncoderBlock(nn.Module):
    def __init__(self, dim, n_heads, head_dim, hidden, rope_theta):
        super().__init__()
        self.pre_attn_norm = nn.RMSNorm(dim, eps=1e-5)
        self.self_attn = _EncoderSelfAttention(dim, n_heads, head_dim, rope_theta)
        self.pre_ffn_norm = nn.RMSNorm(dim, eps=1e-5)
        self.ffn = _EncoderFFN(dim, hidden)

    def __call__(self, x, mask, cache=None):
        x = x + self.self_attn(self.pre_attn_norm(x), mask, cache=cache)
        x = x + self.ffn(self.pre_ffn_norm(x))
        return x


class StreamingEncoder(nn.Module):
    """Causal Whisper-style encoder with two causal convolutions followed by
    a stack of transformer blocks.  Supports both full-sequence and
    incremental (streaming) forward passes."""

    def __init__(
        self,
        mel_channels: int = 128,
        dim: int = 1280,
        n_layers: int = 32,
        n_heads: int = 32,
        head_dim: int = 64,
        hidden_dim: int = 5120,
        rope_theta: float = 1e6,
        sliding_window: int = 750,
    ):
        super().__init__()
        self.conv1 = CausalConv(mel_channels, dim, kernel=3, stride=1)
        self.conv2 = CausalConv(dim, dim, kernel=3, stride=2)
        self.blocks = [
            _EncoderBlock(dim, n_heads, head_dim, hidden_dim, rope_theta)
            for _ in range(n_layers)
        ]
        self.final_norm = nn.RMSNorm(dim, eps=1e-5)
        self.sliding_window = sliding_window

    # -- full-sequence --

    def _apply_convs(self, mel: mx.array) -> mx.array:
        x = mel.T[None, :, :]  # [1, T, mel_channels]
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        return x

    def forward(self, mel: mx.array) -> mx.array:
        x = self._apply_convs(mel.astype(self.conv1.weight.dtype))
        for blk in self.blocks:
            x = blk(x, mask="causal")
        return self.final_norm(x)

    # -- incremental (streaming) --

    def forward_conv_incremental(self, x_in, tail1, tail2):
        """Process new mel frames through the two causal convs using cached tails.

        Args:
            x_in: [1, N, mel_channels]
            tail1: [1, pad1, mel_channels] or None (first call)
            tail2: [1, pad2, dim] or None (first call)

        Returns:
            (out, new_tail1, new_tail2)
        """
        # Conv1 (kernel=3, stride=1 → left_pad=2)
        if tail1 is not None:
            c1_in = mx.concatenate([tail1, x_in], axis=1)
        else:
            c1_in = mx.pad(x_in, [(0, 0), (self.conv1.left_pad, 0), (0, 0)])
        new_tail1 = x_in[:, -self.conv1.left_pad :, :]
        c1_out = nn.gelu(
            mx.conv1d(c1_in, self.conv1.weight, stride=self.conv1.stride) + self.conv1.bias
        )

        # Conv2 (kernel=3, stride=2 → left_pad=1)
        if tail2 is not None:
            c2_in = mx.concatenate([tail2, c1_out], axis=1)
        else:
            c2_in = mx.pad(c1_out, [(0, 0), (self.conv2.left_pad, 0), (0, 0)])
        new_tail2 = c1_out[:, -self.conv2.left_pad :, :]
        c2_out = nn.gelu(
            mx.conv1d(c2_in, self.conv2.weight, stride=self.conv2.stride) + self.conv2.bias
        )

        return c2_out, new_tail1, new_tail2

    def forward_transformer_incremental(self, x, cache_list):
        """Run transformer blocks with per-layer KV caches."""
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask="causal", cache=cache_list[i])
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Decoder components
# ---------------------------------------------------------------------------


class _DecoderAttention(nn.Module):
    """Grouped-query attention for the text decoder."""

    def __init__(self, dim, n_heads, n_kv_heads, head_dim, rope_theta):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.rope_theta = rope_theta

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        pos = cache.offset if cache is not None else 0
        q = mx.fast.rope(q, self.head_dim, traditional=True, base=self.rope_theta, scale=1.0, offset=pos)
        k = mx.fast.rope(k, self.head_dim, traditional=True, base=self.rope_theta, scale=1.0, offset=pos)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class _DecoderFFN(nn.Module):
    """SwiGLU feed-forward for decoder layers."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class AdaptiveScaling(nn.Module):
    """Small MLP that produces a multiplicative scale from the delay embedding,
    used to condition the FFN on the streaming delay."""

    def __init__(self, dim, bottleneck):
        super().__init__()
        self.proj_in = nn.Linear(dim, bottleneck, bias=False)
        self.proj_out = nn.Linear(bottleneck, dim, bias=False)

    def __call__(self, cond):
        return self.proj_out(nn.gelu(self.proj_in(cond)))


class _DecoderBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim, hidden, rope_theta, cond_dim):
        super().__init__()
        self.pre_attn_norm = nn.RMSNorm(dim, eps=1e-5)
        self.self_attn = _DecoderAttention(dim, n_heads, n_kv_heads, head_dim, rope_theta)
        self.adaptive_scale = AdaptiveScaling(dim, cond_dim)
        self.pre_ffn_norm = nn.RMSNorm(dim, eps=1e-5)
        self.ffn = _DecoderFFN(dim, hidden)

    def __call__(self, x, delay_cond, mask=None, cache=None):
        x = x + self.self_attn(self.pre_attn_norm(x), mask, cache)
        scaled = self.pre_ffn_norm(x) * (1.0 + self.adaptive_scale(delay_cond))
        x = x + self.ffn(scaled)
        return x


class TextDecoder(nn.Module):
    """Mistral-style causal language model with adaptive time-conditioning."""

    def __init__(
        self,
        dim: int = 3072,
        n_layers: int = 26,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        hidden_dim: int = 9216,
        vocab_size: int = 131072,
        rope_theta: float = 1e6,
        cond_dim: int = 32,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.blocks = [
            _DecoderBlock(dim, n_heads, n_kv_heads, head_dim, hidden_dim, rope_theta, cond_dim)
            for _ in range(n_layers)
        ]
        self.final_norm = nn.RMSNorm(dim, eps=1e-5)

    def embed(self, token_ids: mx.array) -> mx.array:
        return self.token_embedding(token_ids)

    def __call__(self, x, delay_cond, mask=None, cache=None):
        delay_cond = delay_cond.astype(x.dtype)
        for i, blk in enumerate(self.blocks):
            blk_cache = cache[i] if cache is not None else None
            x = blk(x, delay_cond, mask, blk_cache)
        x = self.final_norm(x)
        return self.token_embedding.as_linear(x)


# ---------------------------------------------------------------------------
# Adapter & embeddings
# ---------------------------------------------------------------------------


class EncoderToDecoderAdapter(nn.Module):
    """Two-layer projection from encoder space to decoder space."""

    def __init__(self, enc_dim: int, dec_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(enc_dim, dec_dim, bias=False)
        self.linear2 = nn.Linear(dec_dim, dec_dim, bias=False)

    def __call__(self, x):
        return self.linear2(nn.gelu(self.linear1(x)))


class DelayEmbedding(nn.Module):
    """Sinusoidal embedding that encodes the streaming delay as a conditioning
    vector for the decoder's adaptive scaling."""

    def __init__(self, dim: int = 3072, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = mx.exp(-math.log(theta) * mx.arange(half, dtype=mx.float32) / half)
        self._freqs = freqs

    def __call__(self, delay: mx.array) -> mx.array:
        t = delay.reshape(-1, 1).astype(mx.float32)
        angles = t * self._freqs
        return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class VoxtralMLXModel(nn.Module):
    """Top-level Voxtral Realtime model wiring encoder, adapter, and decoder."""

    def __init__(self, config: dict):
        super().__init__()

        enc_cfg = config["multimodal"]["whisper_model_args"]["encoder_args"]
        audio_cfg = enc_cfg["audio_encoding_args"]
        ds_factor = config["multimodal"]["whisper_model_args"]["downsample_args"]["downsample_factor"]

        self.encoder = StreamingEncoder(
            mel_channels=audio_cfg["num_mel_bins"],
            dim=enc_cfg["dim"],
            n_layers=enc_cfg["n_layers"],
            n_heads=enc_cfg["n_heads"],
            head_dim=enc_cfg["head_dim"],
            hidden_dim=enc_cfg["hidden_dim"],
            rope_theta=enc_cfg["rope_theta"],
            sliding_window=enc_cfg["sliding_window"],
        )

        adapter_input_dim = enc_cfg["dim"] * ds_factor
        decoder_dim = config["dim"]
        cond_bottleneck = config.get("ada_rms_norm_t_cond_dim", 32)

        self.adapter = EncoderToDecoderAdapter(adapter_input_dim, decoder_dim)

        self.decoder = TextDecoder(
            dim=decoder_dim,
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            n_kv_heads=config["n_kv_heads"],
            head_dim=config["head_dim"],
            hidden_dim=config["hidden_dim"],
            vocab_size=config["vocab_size"],
            rope_theta=config["rope_theta"],
            cond_dim=cond_bottleneck,
        )

        self.delay_embedding = DelayEmbedding(dim=decoder_dim)
        self.ds_factor = ds_factor

    # -- batch encode --

    def encode(self, mel: mx.array) -> mx.array:
        T = mel.shape[1]
        if T % 2 != 0:
            mel = mel[:, 1:]

        h = self.encoder.forward(mel)  # [1, T/2, enc_dim]
        h = h[0]

        n = h.shape[0]
        trim = n % self.ds_factor
        if trim:
            h = h[trim:]
            n = h.shape[0]

        h = h.reshape(n // self.ds_factor, -1)
        return self.adapter(h)

    # -- incremental encode --

    def encode_incremental(self, new_mel, conv_tail1, conv_tail2, enc_cache, ds_remainder):
        """Incrementally encode new mel frames.

        Returns:
            (audio_embeds | None, conv_tail1, conv_tail2, enc_cache, ds_remainder)
        """
        x = new_mel.T[None, :, :].astype(self.encoder.conv1.weight.dtype)

        x, conv_tail1, conv_tail2 = self.encoder.forward_conv_incremental(x, conv_tail1, conv_tail2)

        if enc_cache is None:
            enc_cache = [SlidingKVCache(100_000) for _ in range(len(self.encoder.blocks))]

        x = self.encoder.forward_transformer_incremental(x, enc_cache)
        x = x[0]  # [N, enc_dim]

        if ds_remainder is not None:
            x = mx.concatenate([ds_remainder, x])

        n_full = (x.shape[0] // self.ds_factor) * self.ds_factor
        if n_full == 0:
            return None, conv_tail1, conv_tail2, enc_cache, x

        leftover = x[n_full:] if x.shape[0] > n_full else None
        x = x[:n_full].reshape(n_full // self.ds_factor, -1)
        return self.adapter(x), conv_tail1, conv_tail2, enc_cache, leftover

    # -- decode --

    def decode(self, embeddings, delay_cond, mask=None, cache=None):
        return self.decoder(embeddings, delay_cond, mask, cache)
