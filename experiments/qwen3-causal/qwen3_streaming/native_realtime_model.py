from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .realtime_config import RealtimeAudioConfig


@dataclass
class AudioLayerCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None


@dataclass
class QwenAudioSurgeryState:
    mel_buffer: torch.Tensor | None = None
    window_start_frame: int = 0
    frames_seen: int = 0
    emitted_steps: int = 0
    last_input_frames: int = 0
    last_recomputed_frames: int = 0
    last_recomputed_context_frames: int = 0


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


@dataclass
class FrameAdapterState:
    pending: torch.Tensor | None = None
    audio_frames_seen: int = 0
    decoder_steps_seen: int = 0


@dataclass
class CachedAudioDecodeState:
    audio: QwenAudioSurgeryState | QwenAudioCausalKVState
    adapter: FrameAdapterState
    frame_hidden: torch.Tensor | None = None

    @property
    def decoder_steps_seen(self) -> int:
        return int(self.adapter.decoder_steps_seen)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _batch_token_ids(
    token_ids: torch.Tensor | Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(token_ids, torch.Tensor):
        out = token_ids.to(device=device, dtype=torch.long)
        if out.ndim == 1:
            out = out.unsqueeze(0).expand(batch_size, -1).contiguous()
    else:
        out = torch.tensor(
            list(token_ids),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1).contiguous()
    if out.ndim != 2 or out.shape[0] != batch_size:
        raise ValueError(
            "token ids must have shape [batch, steps] or [steps], "
            f"got {tuple(out.shape)} for batch={batch_size}"
        )
    return out


def _cached_audio_prefix_embeds(
    model: nn.Module,
    frame_hidden: torch.Tensor,
    *,
    prefix_token_ids: torch.Tensor | Sequence[int],
    audio_placeholder_token_id: int,
) -> torch.Tensor:
    batch_size = int(frame_hidden.shape[0])
    prefix = _batch_token_ids(
        prefix_token_ids,
        batch_size=batch_size,
        device=frame_hidden.device,
    )
    audio_mask = prefix == int(audio_placeholder_token_id)
    audio_steps = int(frame_hidden.shape[1])
    counts = audio_mask.sum(dim=1)
    if int(counts.min().item()) != audio_steps or int(counts.max().item()) != audio_steps:
        raise ValueError(
            "prefix_token_ids must contain exactly one audio placeholder per "
            f"cached audio step; got {counts.tolist()} placeholders for "
            f"{audio_steps} cached steps"
        )
    embeds = model.embed_tokens(prefix)
    audio_values = frame_hidden.to(device=embeds.device, dtype=embeds.dtype)
    return embeds.masked_scatter(audio_mask.unsqueeze(-1), audio_values.reshape(-1))


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    for param in module.parameters(recurse=True):
        return param.device, param.dtype
    for buffer in module.buffers(recurse=True):
        return buffer.device, buffer.dtype
    return torch.device("cpu"), torch.float32


def _qwen_audio_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Match Qwen3-ASR's audio conv length helper without importing internals."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2
        + 1
        + (input_lengths // 100) * 13
    )
    return output_lengths.clamp_min(0)


def _banned_ngram_tokens(history: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 0 or len(history) < ngram_size - 1:
        return set()
    if ngram_size == 1:
        return set(history)
    prefix = tuple(history[-(ngram_size - 1) :])
    banned: set[int] = set()
    for start in range(0, len(history) - ngram_size + 1):
        ngram = tuple(history[start : start + ngram_size])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return banned


def _apply_repetition_controls_to_logits(
    logits: torch.Tensor,
    *,
    token_history: list[list[int]],
    consecutive_text_tokens: torch.Tensor | None,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    max_consecutive_text_tokens: int,
    wait_token_id: int | None,
) -> torch.Tensor:
    if (
        repetition_penalty == 1.0
        and no_repeat_ngram_size <= 0
        and max_consecutive_text_tokens <= 0
    ):
        return logits

    controlled = logits.clone()
    vocab_size = controlled.shape[-1]
    for batch_idx, history in enumerate(token_history):
        if repetition_penalty != 1.0 and repetition_penalty > 0.0:
            for token_id in set(history):
                if 0 <= token_id < vocab_size:
                    score = controlled[batch_idx, token_id]
                    controlled[batch_idx, token_id] = torch.where(
                        score < 0,
                        score * repetition_penalty,
                        score / repetition_penalty,
                    )

        for token_id in _banned_ngram_tokens(history, no_repeat_ngram_size):
            if 0 <= token_id < vocab_size:
                controlled[batch_idx, token_id] = torch.finfo(controlled.dtype).min

        if (
            max_consecutive_text_tokens > 0
            and wait_token_id is not None
            and 0 <= int(wait_token_id) < vocab_size
            and consecutive_text_tokens is not None
            and int(consecutive_text_tokens[batch_idx].item())
            >= max_consecutive_text_tokens
        ):
            wait_score = controlled[batch_idx, int(wait_token_id)].clone()
            controlled[batch_idx, :] = torch.finfo(controlled.dtype).min
            controlled[batch_idx, int(wait_token_id)] = wait_score
    return controlled


class QwenAudioSurgeryEncoder(nn.Module):
    """Streaming wrapper around Qwen3-ASR's pretrained audio tower.

    This first surgery pass keeps Qwen's convolution, projection and encoder
    weights intact, but replaces full-history recomputation with bounded-window
    recomputation. It is intentionally conservative: outputs are only emitted
    once the configured right context is available.
    """

    def __init__(
        self,
        audio_tower: nn.Module,
        config: RealtimeAudioConfig,
        *,
        left_context_frames: int | None = None,
        right_context_frames: int | None = None,
    ) -> None:
        super().__init__()
        self.audio_tower = audio_tower
        self.config = config
        self.left_context_frames = (
            config.qwen_audio_left_context_frames
            if left_context_frames is None
            else int(left_context_frames)
        )
        self.right_context_frames = (
            config.qwen_audio_right_context_frames
            if right_context_frames is None
            else int(right_context_frames)
        )
        if self.left_context_frames <= 0:
            raise ValueError("left_context_frames must be > 0")
        if self.right_context_frames < 0:
            raise ValueError("right_context_frames must be >= 0")

    @property
    def max_recompute_frames(self) -> int:
        return self.left_context_frames + self.right_context_frames

    def init_state(self) -> QwenAudioSurgeryState:
        return QwenAudioSurgeryState()

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

    def _encode_window(self, mels: torch.Tensor) -> torch.Tensor:
        if mels.ndim != 3:
            raise ValueError("mels must have shape [batch, frames, n_mels]")
        if mels.shape[-1] != self.config.n_mels:
            raise ValueError(
                f"expected {self.config.n_mels} mel bins, got {mels.shape[-1]}"
            )

        device, dtype = _module_device_dtype(self.audio_tower)
        hidden_by_sample: list[torch.Tensor] = []
        max_steps = 0
        for sample in mels:
            feature_lens = torch.tensor(
                [sample.shape[0]],
                device=device,
                dtype=torch.long,
            )
            input_features = sample.to(device=device, dtype=dtype).transpose(0, 1)
            hidden = self._call_audio_tower(input_features, feature_lens)
            hidden_by_sample.append(hidden)
            max_steps = max(max_steps, hidden.shape[0])

        if not hidden_by_sample:
            return mels.new_zeros(0, 0, self.config.d_model)
        hidden_size = hidden_by_sample[0].shape[-1]
        output = hidden_by_sample[0].new_zeros(len(hidden_by_sample), max_steps, hidden_size)
        for idx, hidden in enumerate(hidden_by_sample):
            output[idx, : hidden.shape[0], :] = hidden
        return output.to(device=mels.device)

    def forward_full(self, mels: torch.Tensor) -> torch.Tensor:
        return self._encode_window(mels)

    def forward_chunk(
        self,
        mels: torch.Tensor,
        state: QwenAudioSurgeryState | None = None,
    ) -> tuple[torch.Tensor, QwenAudioSurgeryState]:
        if state is None:
            state = self.init_state()
        if mels.ndim != 3:
            raise ValueError("mels must have shape [batch, frames, n_mels]")
        if mels.shape[1] == 0:
            empty = mels.new_zeros(mels.shape[0], 0, self.config.d_model)
            return empty, state

        new_mels = mels.detach()
        if state.mel_buffer is None:
            state.mel_buffer = new_mels
        else:
            if state.mel_buffer.shape[0] != new_mels.shape[0]:
                raise ValueError("batch size changed inside qwen audio stream")
            state.mel_buffer = torch.cat([state.mel_buffer, new_mels], dim=1)
        state.frames_seen += int(new_mels.shape[1])
        state.last_input_frames = int(new_mels.shape[1])

        max_frames = self.max_recompute_frames
        if state.mel_buffer.shape[1] > max_frames:
            drop = state.mel_buffer.shape[1] - max_frames
            state.mel_buffer = state.mel_buffer[:, drop:, :]
            state.window_start_frame += int(drop)
        state.last_recomputed_frames = int(state.mel_buffer.shape[1])
        state.last_recomputed_context_frames = max(
            0,
            state.last_recomputed_frames - state.last_input_frames,
        )

        hidden = self._encode_window(state.mel_buffer)
        window_start_steps = self.output_steps_for_mel_frames(state.window_start_frame)
        final_mel_frame = max(0, state.frames_seen - self.right_context_frames)
        final_global_steps = self.output_steps_for_mel_frames(final_mel_frame)

        local_start = max(state.emitted_steps, window_start_steps) - window_start_steps
        local_end = min(final_global_steps - window_start_steps, hidden.shape[1])
        local_start = max(0, min(local_start, hidden.shape[1]))
        local_end = max(local_start, min(local_end, hidden.shape[1]))
        if local_end <= local_start:
            empty = hidden.new_zeros(hidden.shape[0], 0, hidden.shape[-1])
            return empty, state

        output = hidden[:, local_start:local_end, :]
        state.emitted_steps = window_start_steps + local_end
        return output, state


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

        ready_frames = (int(buffer.shape[1]) // self.chunk_frames) * self.chunk_frames
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

        hidden = self._encode_ready_mels(ready, state)
        return hidden.to(device=mels.device), state


class QwenAudioSurgeryAdapterBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.mlp(self.norm(x))
        return x + self.dropout(update) * self.residual_scale


class QwenAudioSurgeryFrameAdapter(nn.Module):
    """Project Qwen audio hidden states into the Qwen text hidden space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        adapter_hidden_dim: int = 0,
        adapter_layers: int = 0,
        adapter_dropout: float = 0.0,
        adapter_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.adapter_layers = int(adapter_layers)
        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        if self.input_dim == self.output_dim:
            nn.init.eye_(self.proj.weight)
        else:
            nn.init.xavier_uniform_(self.proj.weight)
        if self.adapter_layers < 0:
            raise ValueError("adapter_layers must be >= 0")
        if self.adapter_layers and self.adapter_hidden_dim <= 0:
            raise ValueError("adapter_hidden_dim must be > 0 when adapter_layers is enabled")
        self.blocks = nn.ModuleList(
            [
                QwenAudioSurgeryAdapterBlock(
                    dim=self.output_dim,
                    hidden_dim=self.adapter_hidden_dim,
                    dropout=adapter_dropout,
                    residual_scale=adapter_residual_scale,
                )
                for _ in range(self.adapter_layers)
            ]
        )

    def init_state(self) -> FrameAdapterState:
        return FrameAdapterState()

    def _project(self, audio_hidden: torch.Tensor) -> torch.Tensor:
        projected = self.proj(audio_hidden.to(dtype=self.proj.weight.dtype))
        for block in self.blocks:
            projected = block(projected)
        return projected.to(device=audio_hidden.device)

    def forward_chunk(
        self,
        audio_hidden: torch.Tensor,
        state: FrameAdapterState | None = None,
    ) -> tuple[torch.Tensor, FrameAdapterState]:
        if state is None:
            state = self.init_state()
        state.audio_frames_seen += int(audio_hidden.shape[1])
        state.decoder_steps_seen += int(audio_hidden.shape[1])
        return self._project(audio_hidden), state

    def forward_full(self, audio_hidden: torch.Tensor) -> torch.Tensor:
        return self._project(audio_hidden)


def _register_qwen3_asr_transformers() -> None:
    from qwen_asr.core.transformers_backend import (
        Qwen3ASRConfig,
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRProcessor,
    )
    from transformers import AutoConfig, AutoModel, AutoProcessor

    try:
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    except ValueError:
        pass
    try:
        AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    except ValueError:
        pass
    try:
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
    except ValueError:
        pass


def qwen3_asr_text_hidden_size(model_id: str) -> int:
    _register_qwen3_asr_transformers()
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    return int(config.thinker_config.text_config.hidden_size)


class Qwen3ASRRealtimeQwenDecoderModel(nn.Module):
    """Realtime ASR scaffold using Qwen3-ASR's pretrained text decoder."""

    def __init__(
        self,
        config: RealtimeAudioConfig,
        *,
        qwen_model_id: str,
        text_model: nn.Module,
        lm_head: nn.Module,
        bos_token_id: int,
        wait_token_id: int | None = None,
        audio_encoder: nn.Module,
        adapter: nn.Module,
        audio_backend: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_model_id = qwen_model_id
        self.bos_token_id = bos_token_id
        self.wait_token_id = wait_token_id
        self.audio_backend = audio_backend
        self.audio_encoder = audio_encoder
        self.adapter = adapter
        self.text_model = text_model
        self.lm_head = lm_head
        self.embed_tokens = self.text_model.embed_tokens
        self.vocab_size = int(self.lm_head.weight.shape[0])


    def init_cached_audio_decode_state(self) -> CachedAudioDecodeState:
        return CachedAudioDecodeState(
            audio=self.audio_encoder.init_state(),
            adapter=self.adapter.init_state(),
        )

    @torch.no_grad()
    def append_audio_to_cache(
        self,
        mels: torch.Tensor,
        state: CachedAudioDecodeState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, CachedAudioDecodeState]:
        if state is None:
            state = self.init_cached_audio_decode_state()
        audio_hidden, state.audio = self.audio_encoder.forward_chunk(mels, state.audio)
        frame_delta, state.adapter = self.adapter.forward_chunk(
            audio_hidden,
            state.adapter,
        )

        if state.frame_hidden is None:
            state.frame_hidden = frame_delta.detach()
        elif frame_delta.shape[1] > 0:
            if state.frame_hidden.shape[0] != frame_delta.shape[0]:
                raise ValueError("batch size changed inside cached audio decode state")
            state.frame_hidden = torch.cat(
                [state.frame_hidden.to(frame_delta.device), frame_delta.detach()],
                dim=1,
            )

        cached = state.frame_hidden
        if cached is None:
            cached = mels.new_zeros(mels.shape[0], 0, self.config.d_model)
            state.frame_hidden = cached
        return cached, frame_delta, state

    @torch.no_grad()
    def generate_full_hypothesis_from_cached_audio(
        self,
        frame_hidden: torch.Tensor,
        *,
        prefix_token_ids: torch.Tensor | Sequence[int] | None = None,
        audio_placeholder_token_id: int | None = None,
        prompt_token_ids: torch.Tensor | Sequence[int] | None = None,
        max_new_tokens: int = 128,
        eos_token_id: int | None = None,
        stop_token_ids: Sequence[int] | None = None,
        suppress_token_ids: Sequence[int] | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        max_consecutive_text_tokens: int = 0,
    ) -> torch.Tensor:
        """Greedy full-hypothesis decode over cached finalized audio embeddings.

        This intentionally reruns the text decoder on the full cached audio prefix
        for each streaming update. The saving comes from not recomputing the old
        audio tower outputs; decoder caching across revisions is a later step.
        """
        if frame_hidden.ndim != 3:
            raise ValueError("frame_hidden must have shape [batch, steps, hidden]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        batch_size = int(frame_hidden.shape[0])
        device = frame_hidden.device
        if max_new_tokens == 0:
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)

        if prefix_token_ids is not None:
            if audio_placeholder_token_id is None:
                raise ValueError(
                    "audio_placeholder_token_id is required when prefix_token_ids is set"
                )
            fixed_prefix = _cached_audio_prefix_embeds(
                self,
                frame_hidden,
                prefix_token_ids=prefix_token_ids,
                audio_placeholder_token_id=int(audio_placeholder_token_id),
            )
        else:
            fixed_prefix = None

        if prefix_token_ids is not None and prompt_token_ids is None:
            prompt = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        elif prompt_token_ids is None:
            prompt = torch.full(
                (batch_size, 1),
                int(self.bos_token_id),
                dtype=torch.long,
                device=device,
            )
        elif isinstance(prompt_token_ids, torch.Tensor):
            prompt = prompt_token_ids.to(device=device, dtype=torch.long)
            if prompt.ndim == 1:
                prompt = prompt.unsqueeze(0).expand(batch_size, -1).contiguous()
        else:
            prompt = torch.tensor(
                list(prompt_token_ids),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0).expand(batch_size, -1).contiguous()
        if prompt.ndim != 2 or prompt.shape[0] != batch_size:
            raise ValueError(
                "prompt_token_ids must have shape [batch, prompt_steps] or "
                f"[prompt_steps], got {tuple(prompt.shape)} for batch={batch_size}"
            )

        generated = prompt
        prompt_steps = int(prompt.shape[1])
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        stop_ids = set(int(token_id) for token_id in (stop_token_ids or ()))
        if eos_token_id is not None:
            stop_ids.add(int(eos_token_id))
        suppress_ids = [
            int(token_id)
            for token_id in (suppress_token_ids or ())
            if 0 <= int(token_id) < self.vocab_size
        ]
        control_wait_token_id = (
            int(self.wait_token_id)
            if self.wait_token_id is not None
            and int(self.wait_token_id) not in set(suppress_ids)
            else None
        )

        for _ in range(max_new_tokens):
            parts: list[torch.Tensor] = []
            if fixed_prefix is not None:
                parts.append(fixed_prefix)
            elif frame_hidden.shape[1] > 0:
                parts.append(frame_hidden.to(dtype=self.embed_tokens.weight.dtype))
            if generated.shape[1] > 0:
                parts.append(self.embed_tokens(generated))
            if not parts:
                break

            decoder_inputs = torch.cat(parts, dim=1)
            outputs = self.text_model(inputs_embeds=decoder_inputs, use_cache=False)
            logits = self.lm_head(outputs.last_hidden_state[:, -1, :])
            if suppress_ids:
                logits = logits.clone()
                logits[:, suppress_ids] = -torch.inf
            token_history = [
                [int(token_id) for token_id in row]
                for row in generated[:, prompt_steps:].detach().cpu().tolist()
            ]
            consecutive_text_tokens = (
                torch.tensor(
                    [len(row) for row in token_history],
                    dtype=torch.long,
                    device=device,
                )
                if max_consecutive_text_tokens > 0
                else None
            )
            logits = _apply_repetition_controls_to_logits(
                logits,
                token_history=token_history,
                consecutive_text_tokens=consecutive_text_tokens,
                repetition_penalty=float(repetition_penalty),
                no_repeat_ngram_size=int(no_repeat_ngram_size),
                max_consecutive_text_tokens=int(max_consecutive_text_tokens),
                wait_token_id=control_wait_token_id,
            )
            next_token = logits.argmax(dim=-1)
            if stop_ids:
                if bool(finished.any().item()):
                    stop_fill_id = (
                        int(eos_token_id) if eos_token_id is not None else min(stop_ids)
                    )
                    next_token = torch.where(
                        finished,
                        torch.full_like(next_token, stop_fill_id),
                        next_token,
                    )
                finished_next = torch.tensor(
                    [int(token_id) in stop_ids for token_id in next_token.tolist()],
                    dtype=torch.bool,
                    device=device,
                )
                finished = finished | finished_next
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            if bool(finished.all().item()):
                break

        return generated[:, prompt_steps:]

class Qwen3ASRRealtimeQwenAudioSurgeryModel(Qwen3ASRRealtimeQwenDecoderModel):
    """Realtime scaffold that reuses Qwen3-ASR's pretrained audio tower."""

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
        audio_encoder = QwenAudioSurgeryEncoder(audio_tower, config)
        adapter = QwenAudioSurgeryFrameAdapter(
            input_dim=int(audio_output_dim or config.d_model),
            output_dim=config.d_model,
            adapter_hidden_dim=config.qwen_audio_adapter_hidden_dim,
            adapter_layers=config.qwen_audio_adapter_layers,
            adapter_dropout=config.qwen_audio_adapter_dropout,
            adapter_residual_scale=config.qwen_audio_adapter_residual_scale,
        )
        super().__init__(
            config,
            qwen_model_id=qwen_model_id,
            text_model=text_model,
            lm_head=lm_head,
            bos_token_id=bos_token_id,
            wait_token_id=wait_token_id,
            audio_encoder=audio_encoder,
            adapter=adapter,
            audio_backend="qwen_audio_surgery",
        )

    @classmethod
    def from_qwen_pretrained(
        cls,
        qwen_model_id: str,
        *,
        config: RealtimeAudioConfig | None = None,
        bos_token_id: int,
        wait_token_id: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict[str, Any] | None = "cpu",
    ) -> "Qwen3ASRRealtimeQwenAudioSurgeryModel":
        _register_qwen3_asr_transformers()
        from transformers import AutoConfig, AutoModel

        qwen_config = AutoConfig.from_pretrained(qwen_model_id)
        text_config = qwen_config.thinker_config.text_config
        audio_config = qwen_config.thinker_config.audio_config
        if config is None:
            config = RealtimeAudioConfig(
                d_model=int(text_config.hidden_size),
                audio_window_sec=15.0,
            )
        if config.d_model != int(text_config.hidden_size):
            raise ValueError(
                f"Realtime d_model={config.d_model} must match Qwen hidden_size="
                f"{int(text_config.hidden_size)}"
            )
        if config.n_mels != int(getattr(audio_config, "num_mel_bins", config.n_mels)):
            raise ValueError(
                f"Realtime n_mels={config.n_mels} must match Qwen num_mel_bins="
                f"{int(audio_config.num_mel_bins)}"
            )

        qwen_model = AutoModel.from_pretrained(
            qwen_model_id,
            dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        thinker = qwen_model.thinker
        return cls(
            config,
            qwen_model_id=qwen_model_id,
            audio_tower=thinker.audio_tower,
            text_model=thinker.model,
            lm_head=thinker.lm_head,
            bos_token_id=bos_token_id,
            wait_token_id=wait_token_id,
            audio_output_dim=int(getattr(audio_config, "output_dim", config.d_model)),
        )


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


