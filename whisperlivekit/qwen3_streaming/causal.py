"""Causal-KV audio backend for the Qwen3 streaming backend.

Append-only execution of the pretrained Qwen3-ASR audio tower: bidirectional
attention within fixed blocks (the trained regime is 96/192 mel frames),
causal per-layer KV across blocks with a bounded left window, positions that
continue monotonically — each mel frame transits the tower exactly once. The
fine-tuned tower checkpoint (embedding distillation, see
``experiments/qwen3-causal/RUNS.md``) is loaded on top of the base model.

Promoted from ``experiments/qwen3-causal/qwen3_streaming/native_realtime_model.py``
@ 9d4b99a; the fixed-block buffering (``block_frames``) is new here: production
pacing delivers variable-size mel chunks, so the encoder buffers and consumes
exact multiples of the trained block size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .model import (
    CachedAudioDecodeState,
    Qwen3ASRRealtimeQwenAudioSurgeryModel,
    Qwen3ASRRealtimeQwenDecoderModel,
    QwenAudioSurgeryFrameAdapter,
    _module_device_dtype,
    _qwen_audio_output_lengths,
)
from .model_config import RealtimeAudioConfig


@dataclass
class AudioLayerCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None


@dataclass
class QwenAudioCausalKVState:
    mel_buffer: torch.Tensor | None = None
    layer_caches: list[AudioLayerCache] = field(default_factory=list)
    frames_seen: int = 0
    emitted_steps: int = 0
    last_input_frames: int = 0
    last_recomputed_frames: int = 0
    last_recomputed_context_frames: int = 0
    pending_frames: int = 0
    # Bounded mutable tail bookkeeping: conv blocks whose output steps are
    # still re-computable, as (mel_block, output_steps) pairs, plus the number
    # of currently mutable output steps. emitted_steps counts frozen steps.
    tail_blocks: list[tuple[torch.Tensor, int]] = field(default_factory=list)
    mutable_steps: int = 0


class QwenAudioCausalKVEncoder(nn.Module):
    """Append-only causal execution for Qwen3-ASR's pretrained audio tower.

    This is the first backend that satisfies the core realtime invariant: once a
    mel frame has been encoded and emitted, later chunks never feed it through
    the audio tower again. Qwen's conv/projection/layer weights are reused, but
    audio self-attention is executed with a causal KV cache instead of Qwen's
    offline bidirectional mask.

    The conv stack is still chunked. By default the chunk is one decoder step
    (80 ms), which makes latency useful for streaming and gives the following
    training problem to LoRA/adapters: adapt Qwen audio weights to this stricter
    causal execution.
    """

    def __init__(
        self,
        audio_tower: nn.Module,
        config: RealtimeAudioConfig,
        *,
        left_context_frames: int | None = None,
        chunk_frames: int | None = None,
        mutable_tail_sec: float | None = None,
        block_frames: int | None = None,
    ) -> None:
        super().__init__()
        self.audio_tower = audio_tower
        self.config = config
        self.left_context_frames = (
            config.qwen_audio_left_context_frames
            if left_context_frames is None
            else int(left_context_frames)
        )
        if self.left_context_frames <= 0:
            raise ValueError("left_context_frames must be > 0")
        self.chunk_frames = (
            config.frames_per_decoder_step if chunk_frames is None else int(chunk_frames)
        )
        if self.chunk_frames <= 0:
            raise ValueError("chunk_frames must be > 0")
        self.left_context_steps = max(
            1,
            self.output_steps_for_mel_frames(self.left_context_frames),
        )
        tail_sec = (
            config.qwen_audio_mutable_tail_sec
            if mutable_tail_sec is None
            else float(mutable_tail_sec)
        )
        if tail_sec < 0.0:
            raise ValueError("mutable_tail_sec must be >= 0")
        tail_frames = int(round(tail_sec * 1000.0 / config.mel_hop_ms))
        self.mutable_tail_steps = (
            self.output_steps_for_mel_frames(tail_frames) if tail_frames > 0 else 0
        )
        self.block_bidirectional = bool(
            getattr(config, "qwen_audio_block_bidirectional", False)
        )
        self.block_frames = (
            int(getattr(config, "qwen_audio_block_frames", 0))
            if block_frames is None
            else int(block_frames)
        )
        if self.block_frames < 0:
            raise ValueError("block_frames must be >= 0")
        if self.block_frames > 0:
            if self.block_frames % self.chunk_frames != 0:
                raise ValueError("block_frames must be a multiple of chunk_frames")
            if self.mutable_tail_steps > 0:
                raise ValueError(
                    "fixed attention blocks and a mutable tail are exclusive"
                )

    @property
    def right_context_frames(self) -> int:
        return 0

    def init_state(self) -> QwenAudioCausalKVState:
        layers = getattr(self.audio_tower, "layers", ())
        return QwenAudioCausalKVState(
            layer_caches=[AudioLayerCache() for _ in range(len(layers))]
        )

    def output_steps_for_mel_frames(self, mel_frames: int) -> int:
        if mel_frames <= 0:
            return 0
        fn = getattr(self.audio_tower, "_get_feat_extract_output_lengths", None)
        device, _ = _module_device_dtype(self.audio_tower)
        lengths = torch.tensor([mel_frames], device=device, dtype=torch.long)
        with torch.no_grad():
            output_lengths = (
                fn(lengths) if fn is not None else _qwen_audio_output_lengths(lengths)
            )
        return int(output_lengths.reshape(-1)[0].detach().cpu())

    def _call_audio_tower(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        try:
            outputs = self.audio_tower(
                input_features=input_features,
                feature_lens=feature_lens,
            )
        except TypeError:
            try:
                outputs = self.audio_tower(input_features, feature_lens=feature_lens)
            except TypeError:
                outputs = self.audio_tower(input_features, feature_lens)

        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            hidden = outputs[0]
        else:
            hidden = outputs
        if hidden.ndim == 3 and hidden.shape[0] == 1:
            hidden = hidden[0]
        if hidden.ndim != 2:
            raise ValueError(
                "Qwen audio tower must return [steps, hidden] or [1, steps, hidden], "
                f"got {tuple(hidden.shape)}"
            )
        return hidden

    def _has_qwen_audio_internals(self) -> bool:
        required = (
            "conv2d1",
            "conv2d2",
            "conv2d3",
            "conv_out",
            "positional_embedding",
            "layers",
            "ln_post",
            "proj1",
            "proj2",
        )
        return all(hasattr(self.audio_tower, name) for name in required)

    def _position_embedding(
        self,
        *,
        offset: int,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        table = self.audio_tower.positional_embedding.positional_embedding
        if offset + length <= table.shape[0]:
            return table[offset : offset + length, :].to(device=device, dtype=dtype)

        dim = int(table.shape[1])
        if dim % 2 != 0:
            raise ValueError("Qwen audio sinusoidal dim must be even")
        half = dim // 2
        inv = torch.exp(
            -math.log(10000.0)
            / float(max(1, half - 1))
            * torch.arange(half, device=device, dtype=torch.float32)
        )
        positions = torch.arange(
            offset,
            offset + length,
            device=device,
            dtype=torch.float32,
        )
        scaled = positions[:, None] * inv[None, :]
        return torch.cat([scaled.sin(), scaled.cos()], dim=1).to(dtype=dtype)

    def _conv_one_block(self, block: torch.Tensor, *, position_offset: int) -> torch.Tensor:
        tower = self.audio_tower
        device, dtype = _module_device_dtype(tower)
        block = block.to(device=device, dtype=dtype)
        x = block.transpose(1, 2).unsqueeze(1)
        x = F.gelu(tower.conv2d1(x))
        x = F.gelu(tower.conv2d2(x))
        x = F.gelu(tower.conv2d3(x))
        batch, channels, freq, steps = x.size()
        x = tower.conv_out(
            x.permute(0, 3, 1, 2).contiguous().view(batch, steps, channels * freq)
        )
        pos = self._position_embedding(
            offset=position_offset,
            length=steps,
            device=x.device,
            dtype=x.dtype,
        )
        return x + pos.unsqueeze(0)

    def _conv_blocks(self, mels: torch.Tensor, *, position_offset: int) -> torch.Tensor:
        if mels.shape[1] == 0:
            device, _ = _module_device_dtype(self.audio_tower)
            return mels.new_zeros(mels.shape[0], 0, self.config.d_model).to(device=device)
        outputs: list[torch.Tensor] = []
        step_offset = int(position_offset)
        for start in range(0, int(mels.shape[1]), self.chunk_frames):
            block = mels[:, start : start + self.chunk_frames, :]
            hidden = self._conv_one_block(block, position_offset=step_offset)
            outputs.append(hidden)
            step_offset += int(hidden.shape[1])
        return torch.cat(outputs, dim=1)

    def _fallback_encode_blocks(self, mels: torch.Tensor) -> torch.Tensor:
        device, dtype = _module_device_dtype(self.audio_tower)
        hidden_by_sample: list[torch.Tensor] = []
        max_steps = 0
        for sample in mels:
            sample_chunks: list[torch.Tensor] = []
            for start in range(0, int(sample.shape[0]), self.chunk_frames):
                block = sample[start : start + self.chunk_frames, :]
                feature_lens = torch.tensor(
                    [block.shape[0]],
                    device=device,
                    dtype=torch.long,
                )
                input_features = block.to(device=device, dtype=dtype).transpose(0, 1)
                sample_chunks.append(self._call_audio_tower(input_features, feature_lens))
            if sample_chunks:
                hidden = torch.cat(sample_chunks, dim=0)
            else:
                hidden = sample.new_zeros(0, self.config.d_model).to(device=device)
            hidden_by_sample.append(hidden)
            max_steps = max(max_steps, int(hidden.shape[0]))
        if not hidden_by_sample:
            return mels.new_zeros(0, 0, self.config.d_model)
        hidden_size = int(hidden_by_sample[0].shape[-1])
        output = hidden_by_sample[0].new_zeros(len(hidden_by_sample), max_steps, hidden_size)
        for idx, hidden in enumerate(hidden_by_sample):
            output[idx, : hidden.shape[0], :] = hidden
        return output

    def _attention_chunk(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        cache: AudioLayerCache,
        *,
        position_offset: int,
    ) -> tuple[torch.Tensor, AudioLayerCache]:
        batch, length, _ = hidden_states.shape
        if length == 0:
            return hidden_states, cache
        num_heads = int(attn.num_heads)
        head_dim = int(attn.head_dim)
        query_states = attn.q_proj(hidden_states).reshape(
            batch, length, num_heads, head_dim
        )
        key_states = attn.k_proj(hidden_states).reshape(
            batch, length, num_heads, head_dim
        )
        value_states = attn.v_proj(hidden_states).reshape(
            batch, length, num_heads, head_dim
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        past_k = cache.key
        past_v = cache.value
        past_len = 0 if past_k is None else int(past_k.shape[-2])
        if past_k is not None and past_v is not None:
            all_k = torch.cat([past_k.to(key_states.device), key_states], dim=-2)
            all_v = torch.cat([past_v.to(value_states.device), value_states], dim=-2)
        else:
            all_k = key_states
            all_v = value_states

        total_len = int(all_k.shape[-2])
        cache_start_pos = int(position_offset) - past_len
        q_positions = torch.arange(
            int(position_offset),
            int(position_offset) + length,
            device=hidden_states.device,
        )
        k_positions = torch.arange(
            cache_start_pos,
            cache_start_pos + total_len,
            device=hidden_states.device,
        )
        if self.block_bidirectional and length > 0:
            # Chunked-attention pattern: every query in the current block sees
            # every key of the block (bidirectional within block), plus the
            # frozen causal prefix. Still append-only; latency = block size.
            block_max = int(position_offset) + length - 1
            allowed = k_positions[None, :] <= block_max
        else:
            allowed = k_positions[None, :] <= q_positions[:, None]
        allowed = allowed & (
            k_positions[None, :]
            >= (q_positions[:, None] - self.left_context_steps + 1)
        )

        scores = torch.matmul(
            query_states.float(),
            all_k.float().transpose(-2, -1),
        ) * float(attn.scaling)
        scores = scores.masked_fill(
            ~allowed[None, None, :, :],
            torch.finfo(scores.dtype).min,
        )
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(
            weights,
            p=float(getattr(attn, "attention_dropout", 0.0)),
            training=self.training,
        )
        context = torch.matmul(weights.to(dtype=all_v.dtype), all_v)
        context = context.transpose(1, 2).contiguous().view(batch, length, -1)
        output = attn.out_proj(context.to(dtype=attn.out_proj.weight.dtype))

        keep = min(total_len, self.left_context_steps)
        next_cache = AudioLayerCache(
            key=all_k[:, :, -keep:, :].detach(),
            value=all_v[:, :, -keep:, :].detach(),
        )
        return output.to(dtype=hidden_states.dtype), next_cache

    def _layer_chunk(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        cache: AudioLayerCache,
        *,
        position_offset: int,
    ) -> tuple[torch.Tensor, AudioLayerCache]:
        residual = hidden_states
        normed = layer.self_attn_layer_norm(hidden_states)
        attn_out, next_cache = self._attention_chunk(
            layer.self_attn,
            normed,
            cache,
            position_offset=position_offset,
        )
        hidden_states = residual + F.dropout(
            attn_out,
            p=float(getattr(layer, "dropout", 0.0)),
            training=self.training,
        )
        residual = hidden_states
        hidden_states = layer.final_layer_norm(hidden_states)
        hidden_states = layer.fc1(hidden_states)
        hidden_states = layer.activation_fn(hidden_states)
        hidden_states = F.dropout(
            hidden_states,
            p=float(getattr(layer, "activation_dropout", 0.0)),
            training=self.training,
        )
        hidden_states = layer.fc2(hidden_states)
        hidden_states = residual + F.dropout(
            hidden_states,
            p=float(getattr(layer, "dropout", 0.0)),
            training=self.training,
        )
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states,
                min=-clamp_value,
                max=clamp_value,
            )
        return hidden_states, next_cache

    def _attention_tail(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        cache: AudioLayerCache,
        *,
        position_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attention over [frozen cache + tail] WITHOUT updating the cache.

        Returns the attention output plus the tail's per-layer key/value
        states so the caller can freeze a leading slice of them later.
        """
        batch, length, _ = hidden_states.shape
        num_heads = int(attn.num_heads)
        head_dim = int(attn.head_dim)
        query_states = attn.q_proj(hidden_states).reshape(
            batch, length, num_heads, head_dim
        ).transpose(1, 2)
        key_states = attn.k_proj(hidden_states).reshape(
            batch, length, num_heads, head_dim
        ).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).reshape(
            batch, length, num_heads, head_dim
        ).transpose(1, 2)

        past_k = cache.key
        past_v = cache.value
        past_len = 0 if past_k is None else int(past_k.shape[-2])
        if past_k is not None and past_v is not None:
            all_k = torch.cat([past_k.to(key_states.device), key_states], dim=-2)
            all_v = torch.cat([past_v.to(value_states.device), value_states], dim=-2)
        else:
            all_k = key_states
            all_v = value_states

        total_len = int(all_k.shape[-2])
        cache_start_pos = int(position_offset) - past_len
        q_positions = torch.arange(
            int(position_offset),
            int(position_offset) + length,
            device=hidden_states.device,
        )
        k_positions = torch.arange(
            cache_start_pos,
            cache_start_pos + total_len,
            device=hidden_states.device,
        )
        if self.block_bidirectional and length > 0:
            # Chunked-attention pattern: every query in the current block sees
            # every key of the block (bidirectional within block), plus the
            # frozen causal prefix. Still append-only; latency = block size.
            block_max = int(position_offset) + length - 1
            allowed = k_positions[None, :] <= block_max
        else:
            allowed = k_positions[None, :] <= q_positions[:, None]
        allowed = allowed & (
            k_positions[None, :]
            >= (q_positions[:, None] - self.left_context_steps + 1)
        )

        scores = torch.matmul(
            query_states.float(),
            all_k.float().transpose(-2, -1),
        ) * float(attn.scaling)
        scores = scores.masked_fill(
            ~allowed[None, None, :, :],
            torch.finfo(scores.dtype).min,
        )
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(
            weights,
            p=float(getattr(attn, "attention_dropout", 0.0)),
            training=self.training,
        )
        context = torch.matmul(weights.to(dtype=all_v.dtype), all_v)
        context = context.transpose(1, 2).contiguous().view(batch, length, -1)
        output = attn.out_proj(context.to(dtype=attn.out_proj.weight.dtype))
        return output.to(dtype=hidden_states.dtype), key_states, value_states

    def _layer_tail(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        cache: AudioLayerCache,
        *,
        position_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normed = layer.self_attn_layer_norm(hidden_states)
        attn_out, tail_k, tail_v = self._attention_tail(
            layer.self_attn,
            normed,
            cache,
            position_offset=position_offset,
        )
        hidden_states = residual + F.dropout(
            attn_out,
            p=float(getattr(layer, "dropout", 0.0)),
            training=self.training,
        )
        residual = hidden_states
        hidden_states = layer.final_layer_norm(hidden_states)
        hidden_states = layer.fc1(hidden_states)
        hidden_states = layer.activation_fn(hidden_states)
        hidden_states = F.dropout(
            hidden_states,
            p=float(getattr(layer, "activation_dropout", 0.0)),
            training=self.training,
        )
        hidden_states = layer.fc2(hidden_states)
        hidden_states = residual + F.dropout(
            hidden_states,
            p=float(getattr(layer, "dropout", 0.0)),
            training=self.training,
        )
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states,
                min=-clamp_value,
                max=clamp_value,
            )
        return hidden_states, tail_k, tail_v

    def _encode_mutable_tail(
        self,
        ready_mels: torch.Tensor,
        state: QwenAudioCausalKVState,
    ) -> torch.Tensor:
        """Recompute the mutable tail plus new blocks over the frozen KV prefix.

        Returns hidden states for ALL current tail steps (the re-computed
        previously-mutable ones followed by the new ones). Leading tail blocks
        are frozen into the per-layer KV caches whenever the remaining tail
        exceeds ``mutable_tail_steps``.
        """
        if not self._has_qwen_audio_internals():
            raise ValueError(
                "the bounded mutable tail requires direct access to Qwen audio "
                "tower internals (conv/layers); the generic fallback path "
                "cannot freeze per-layer KV"
            )
        layers = getattr(self.audio_tower, "layers")
        if not state.layer_caches:
            state.layer_caches = [AudioLayerCache() for _ in range(len(layers))]
        if len(state.layer_caches) != len(layers):
            raise ValueError("Qwen causal audio cache layer count mismatch")

        new_blocks: list[torch.Tensor] = []
        for start in range(0, int(ready_mels.shape[1]), self.chunk_frames):
            new_blocks.append(ready_mels[:, start : start + self.chunk_frames, :])

        # Conv each block at its deterministic position (frozen steps offset).
        position = int(state.emitted_steps)
        conv_outputs: list[torch.Tensor] = []
        block_entries: list[tuple[torch.Tensor, int]] = []
        for mel_block, _ in state.tail_blocks:
            hidden = self._conv_one_block(mel_block, position_offset=position)
            conv_outputs.append(hidden)
            block_entries.append((mel_block, int(hidden.shape[1])))
            position += int(hidden.shape[1])
        for mel_block in new_blocks:
            hidden = self._conv_one_block(mel_block, position_offset=position)
            conv_outputs.append(hidden)
            block_entries.append((mel_block.detach(), int(hidden.shape[1])))
            position += int(hidden.shape[1])
        if not conv_outputs:
            device, _ = _module_device_dtype(self.audio_tower)
            return ready_mels.new_zeros(
                ready_mels.shape[0], 0, self.config.d_model
            ).to(device=device)

        hidden_states = torch.cat(conv_outputs, dim=1)
        position_offset = int(state.emitted_steps)
        tail_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, cache in zip(layers, state.layer_caches, strict=True):
            hidden_states, tail_k, tail_v = self._layer_tail(
                layer,
                hidden_states,
                cache,
                position_offset=position_offset,
            )
            tail_kv.append((tail_k, tail_v))

        tower = self.audio_tower
        hidden_states = tower.ln_post(hidden_states)
        hidden_states = tower.proj1(hidden_states)
        hidden_states = tower.act(hidden_states)
        hidden_states = tower.proj2(hidden_states)

        # Freeze leading blocks until the remaining tail fits the budget.
        total_steps = sum(steps for _, steps in block_entries)
        freeze_steps = 0
        freeze_blocks = 0
        while (
            freeze_blocks < len(block_entries)
            and total_steps - freeze_steps - block_entries[freeze_blocks][1]
            >= self.mutable_tail_steps
        ):
            freeze_steps += block_entries[freeze_blocks][1]
            freeze_blocks += 1
        if freeze_steps > 0:
            for cache, (tail_k, tail_v) in zip(
                state.layer_caches, tail_kv, strict=True
            ):
                new_k = tail_k[:, :, :freeze_steps, :].detach()
                new_v = tail_v[:, :, :freeze_steps, :].detach()
                if cache.key is not None and cache.value is not None:
                    new_k = torch.cat([cache.key.to(new_k.device), new_k], dim=-2)
                    new_v = torch.cat([cache.value.to(new_v.device), new_v], dim=-2)
                keep = min(int(new_k.shape[-2]), self.left_context_steps)
                cache.key = new_k[:, :, -keep:, :]
                cache.value = new_v[:, :, -keep:, :]
            state.emitted_steps += freeze_steps
        state.tail_blocks = block_entries[freeze_blocks:]
        state.mutable_steps = total_steps - freeze_steps
        return hidden_states

    def _encode_ready_mels(
        self,
        mels: torch.Tensor,
        state: QwenAudioCausalKVState,
    ) -> torch.Tensor:
        if mels.shape[1] == 0:
            device, _ = _module_device_dtype(self.audio_tower)
            return mels.new_zeros(mels.shape[0], 0, self.config.d_model).to(device=device)

        if not self._has_qwen_audio_internals():
            hidden = self._fallback_encode_blocks(mels)
            state.emitted_steps += int(hidden.shape[1])
            return hidden

        layers = getattr(self.audio_tower, "layers")
        if not state.layer_caches:
            state.layer_caches = [AudioLayerCache() for _ in range(len(layers))]
        if len(state.layer_caches) != len(layers):
            raise ValueError("Qwen causal audio cache layer count mismatch")

        position_offset = int(state.emitted_steps)
        hidden_states = self._conv_blocks(mels, position_offset=position_offset)
        next_caches: list[AudioLayerCache] = []
        for layer, cache in zip(layers, state.layer_caches, strict=True):
            hidden_states, next_cache = self._layer_chunk(
                layer,
                hidden_states,
                cache,
                position_offset=position_offset,
            )
            next_caches.append(next_cache)
        state.layer_caches = next_caches

        tower = self.audio_tower
        hidden_states = tower.ln_post(hidden_states)
        hidden_states = tower.proj1(hidden_states)
        hidden_states = tower.act(hidden_states)
        hidden_states = tower.proj2(hidden_states)
        state.emitted_steps += int(hidden_states.shape[1])
        return hidden_states

    def forward_full(self, mels: torch.Tensor) -> torch.Tensor:
        state = self.init_state()
        return self._encode_ready_mels(mels, state).to(device=mels.device)

    def flush_pending(
        self,
        state: QwenAudioCausalKVState,
    ) -> tuple[torch.Tensor, QwenAudioCausalKVState]:
        if state.mel_buffer is None or state.mel_buffer.shape[1] == 0:
            device, _ = _module_device_dtype(self.audio_tower)
            hidden = torch.zeros(1, 0, self.config.d_model, device=device)
            return hidden, state
        ready = state.mel_buffer
        # Keep the conv stack in-regime: encode whole 80 ms conv blocks only;
        # the sub-block remainder (< 80 ms) carries no decodable content.
        ready_frames = (int(ready.shape[1]) // self.chunk_frames) * self.chunk_frames
        if ready_frames == 0:
            state.mel_buffer = None
            state.pending_frames = 0
            device, _ = _module_device_dtype(self.audio_tower)
            hidden = torch.zeros(1, 0, self.config.d_model, device=device)
            return hidden, state
        ready = ready[:, :ready_frames, :]
        state.mel_buffer = None
        state.pending_frames = 0
        state.last_recomputed_frames = int(ready.shape[1])
        state.last_recomputed_context_frames = 0
        hidden = self._encode_ready_mels(ready, state)
        return hidden, state

    def forward_chunk(
        self,
        mels: torch.Tensor,
        state: QwenAudioCausalKVState | None = None,
    ) -> tuple[torch.Tensor, QwenAudioCausalKVState]:
        if state is None:
            state = self.init_state()
        if mels.ndim != 3:
            raise ValueError("mels must have shape [batch, frames, n_mels]")
        if mels.shape[-1] != self.config.n_mels:
            raise ValueError(
                f"expected {self.config.n_mels} mel bins, got {mels.shape[-1]}"
            )
        if state.mel_buffer is not None and state.mel_buffer.shape[0] != mels.shape[0]:
            raise ValueError("batch size changed inside qwen causal audio stream")

        state.last_input_frames = int(mels.shape[1])
        state.frames_seen += int(mels.shape[1])
        if mels.shape[1] == 0:
            empty = mels.new_zeros(mels.shape[0], 0, self.config.d_model)
            state.last_recomputed_frames = 0
            state.last_recomputed_context_frames = 0
            return empty, state

        if state.mel_buffer is None:
            buffer = mels
        else:
            buffer = torch.cat([state.mel_buffer.to(mels.device), mels], dim=1)

        consume_frames = (
            self.block_frames if self.block_frames > 0 else self.chunk_frames
        )
        ready_frames = (int(buffer.shape[1]) // consume_frames) * consume_frames
        ready = buffer[:, :ready_frames, :]
        state.mel_buffer = buffer[:, ready_frames:, :].detach()
        state.pending_frames = int(state.mel_buffer.shape[1])

        if self.mutable_tail_steps > 0:
            tail_mel_frames = sum(
                int(block.shape[1]) for block, _ in state.tail_blocks
            )
            if ready.shape[1] == 0 and tail_mel_frames == 0:
                state.last_recomputed_frames = 0
                state.last_recomputed_context_frames = 0
                empty = mels.new_zeros(mels.shape[0], 0, self.config.d_model)
                return empty, state
            # Recompute cost = previously-emitted tail mels + new mels.
            state.last_recomputed_frames = tail_mel_frames + int(ready.shape[1])
            state.last_recomputed_context_frames = tail_mel_frames
            hidden = self._encode_mutable_tail(ready, state)
            return hidden.to(device=mels.device), state

        state.last_recomputed_frames = int(ready.shape[1])
        state.last_recomputed_context_frames = 0

        if ready.shape[1] == 0:
            empty = mels.new_zeros(mels.shape[0], 0, self.config.d_model)
            return empty, state

        if self.block_frames > 0:
            outputs = [
                self._encode_ready_mels(
                    ready[:, start : start + self.block_frames, :], state
                )
                for start in range(0, int(ready.shape[1]), self.block_frames)
            ]
            hidden = torch.cat(outputs, dim=1)
        else:
            hidden = self._encode_ready_mels(ready, state)
        return hidden.to(device=mels.device), state


class Qwen3ASRRealtimeQwenAudioCausalModel(Qwen3ASRRealtimeQwenAudioSurgeryModel):
    """Qwen3-ASR realtime model with append-only causal audio KV execution."""

    def __init__(
        self,
        config: RealtimeAudioConfig,
        *,
        qwen_model_id: str,
        audio_tower: nn.Module,
        text_model: nn.Module,
        lm_head: nn.Module,
        bos_token_id: int,
        wait_token_id: int | None = None,
        audio_output_dim: int | None = None,
    ) -> None:
        audio_encoder = QwenAudioCausalKVEncoder(audio_tower, config)
        adapter = QwenAudioSurgeryFrameAdapter(
            input_dim=int(audio_output_dim or config.d_model),
            output_dim=config.d_model,
            adapter_hidden_dim=config.qwen_audio_adapter_hidden_dim,
            adapter_layers=config.qwen_audio_adapter_layers,
            adapter_dropout=config.qwen_audio_adapter_dropout,
            adapter_residual_scale=config.qwen_audio_adapter_residual_scale,
        )
        Qwen3ASRRealtimeQwenDecoderModel.__init__(
            self,
            config,
            qwen_model_id=qwen_model_id,
            text_model=text_model,
            lm_head=lm_head,
            bos_token_id=bos_token_id,
            wait_token_id=wait_token_id,
            audio_encoder=audio_encoder,
            adapter=adapter,
            audio_backend="qwen_audio_causal_kv",
        )

    @torch.no_grad()
    def append_audio_to_cache(
        self,
        mels: torch.Tensor,
        state: CachedAudioDecodeState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, CachedAudioDecodeState]:
        """Append audio; with a bounded mutable tail, overwrite the tail.

        The strict append-only path (mutable_tail_steps == 0) is inherited.
        With a mutable tail, the encoder re-emits hidden states for the
        previously-mutable steps plus the new ones, so the corresponding
        slice of ``state.frame_hidden`` is replaced rather than appended.
        """
        if int(getattr(self.audio_encoder, "mutable_tail_steps", 0)) <= 0:
            return super().append_audio_to_cache(mels, state)

        if state is None:
            state = self.init_cached_audio_decode_state()
        previous_mutable = int(getattr(state.audio, "mutable_steps", 0))
        audio_hidden, state.audio = self.audio_encoder.forward_chunk(
            mels, state.audio
        )
        tail_hidden = self.adapter._project(audio_hidden).detach()

        if state.frame_hidden is None:
            frozen_prefix = tail_hidden.new_zeros(
                tail_hidden.shape[0], 0, tail_hidden.shape[2]
            )
        else:
            keep = int(state.frame_hidden.shape[1]) - previous_mutable
            if keep < 0:
                raise ValueError(
                    "cached frame_hidden shorter than the previous mutable tail"
                )
            frozen_prefix = state.frame_hidden[:, :keep, :]
        state.frame_hidden = torch.cat(
            [frozen_prefix.to(tail_hidden.device), tail_hidden], dim=1
        )
        state.adapter.audio_frames_seen += int(mels.shape[1])
        state.adapter.decoder_steps_seen = int(state.frame_hidden.shape[1])

        delta = tail_hidden[:, previous_mutable:, :]
        return state.frame_hidden, delta, state

    @torch.no_grad()
    def flush_audio_to_cache(
        self,
        state: CachedAudioDecodeState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, CachedAudioDecodeState]:
        """Encode the partial block still buffered in the encoder.

        With fixed attention blocks the encoder holds up to block_frames-1 mel
        frames pending; at end of utterance they must be encoded once or up to
        ~1.9 s of trailing speech is silently dropped. The windowed model never
        buffers, so this method only exists on the causal model; the streamer
        treats its absence as a no-op.
        """
        if state is None:
            state = self.init_cached_audio_decode_state()
        audio_hidden, state.audio = self.audio_encoder.flush_pending(state.audio)
        frame_delta, state.adapter = self.adapter.forward_chunk(
            audio_hidden,
            state.adapter,
        )
        if state.frame_hidden is None:
            state.frame_hidden = frame_delta.detach()
        elif frame_delta.shape[1] > 0:
            state.frame_hidden = torch.cat(
                [state.frame_hidden.to(frame_delta.device), frame_delta.detach()],
                dim=1,
            )
        cached = state.frame_hidden
        if cached is None:
            cached = frame_delta.new_zeros(
                frame_delta.shape[0], 0, self.config.d_model
            )
            state.frame_hidden = cached
        return cached, frame_delta, state


def resolve_tower_checkpoint(reference: str) -> Path:
    """Resolve a tower-checkpoint reference to a local weights file.

    Accepts a local file (.pt or .safetensors), a local directory, or a
    Hugging Face repo id (downloaded via ``snapshot_download``).
    """
    path = Path(reference).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        from whisperlivekit.model_paths import resolve_model_path

        path = resolve_model_path(reference)
        if path.is_file():
            return path
    for pattern in ("*.safetensors", "*.pt"):
        matches = sorted(path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"no .safetensors or .pt tower checkpoint found at {reference!r}"
    )


def load_tower_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict:
    """Load fine-tuned audio-tower weights into ``model.audio_encoder``.

    Supports the training payload format (``{"tower_state_dict": ...}`` in a
    .pt file) and a flat tower state dict (.safetensors, the published HF
    format). ``strict=True`` so a 0.6B/1.7B mismatch fails loudly;
    ``load_state_dict`` casts to the model dtype.
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path))
        metadata: dict = {}
    else:
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if isinstance(payload, dict) and "tower_state_dict" in payload:
            state_dict = payload["tower_state_dict"]
            metadata = {
                key: payload.get(key)
                for key in ("step", "gate_wer", "model_id")
                if key in payload
            }
        else:
            state_dict = payload
            metadata = {}
    model.audio_encoder.audio_tower.load_state_dict(state_dict, strict=True)
    return metadata
