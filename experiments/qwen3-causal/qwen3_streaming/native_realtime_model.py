from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
class AudioEncoderState:
    conv_tail: torch.Tensor | None = None
    layer_caches: list[AudioLayerCache] = field(default_factory=list)
    frames_seen: int = 0


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


@dataclass
class FrameAdapterState:
    pending: torch.Tensor | None = None
    audio_frames_seen: int = 0
    decoder_steps_seen: int = 0


@dataclass
class DecoderState:
    layer_caches: list[AudioLayerCache] = field(default_factory=list)
    steps_seen: int = 0


@dataclass
class QwenDecoderState:
    past_key_values: Any = None
    steps_seen: int = 0


@dataclass
class RealtimeModelState:
    audio: AudioEncoderState | QwenAudioSurgeryState | QwenAudioCausalKVState
    adapter: FrameAdapterState
    decoder: DecoderState | QwenDecoderState
    last_token_ids: torch.Tensor
    token_history: list[list[int]] = field(default_factory=list)
    consecutive_text_tokens: torch.Tensor | None = None


@dataclass
class CachedAudioDecodeState:
    audio: AudioEncoderState | QwenAudioSurgeryState | QwenAudioCausalKVState
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


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        lora_input = self.dropout(x).to(dtype=self.lora_a.weight.dtype)
        update = self.lora_b(self.lora_a(lora_input)) * self.scaling
        return output + update.to(dtype=output.dtype)


def _set_child_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def _init_ctc_head(
    ctc_head: nn.Linear,
    *,
    lm_head: nn.Module | None = None,
    blank_token_id: int | None = None,
    blank_logit_bias: float = 2.0,
) -> None:
    if lm_head is not None and ctc_head.weight.shape == lm_head.weight.shape:
        with torch.no_grad():
            ctc_head.weight.copy_(lm_head.weight)
    if ctc_head.bias is not None:
        nn.init.zeros_(ctc_head.bias)
        if blank_token_id is not None and 0 <= int(blank_token_id) < ctc_head.bias.numel():
            with torch.no_grad():
                ctc_head.bias[int(blank_token_id)] = float(blank_logit_bias)


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


def configure_compact_ctc_head(
    model: nn.Module,
    token_ids: list[int] | tuple[int, ...],
    *,
    blank_index: int = 0,
    blank_logit_bias: float = 0.0,
    init_from_ctc_head: bool = True,
) -> None:
    compact_token_ids = [int(token_id) for token_id in token_ids]
    if not compact_token_ids:
        raise ValueError("compact CTC vocab must not be empty")
    if not 0 <= int(blank_index) < len(compact_token_ids):
        raise ValueError("compact CTC blank index is out of range")

    config = getattr(model, "config")
    head = nn.Linear(config.d_model, len(compact_token_ids), bias=True)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)

    source_head = getattr(model, "ctc_head", None)
    if init_from_ctc_head and isinstance(source_head, nn.Linear):
        with torch.no_grad():
            for compact_idx, token_id in enumerate(compact_token_ids):
                if 0 <= token_id < source_head.weight.shape[0]:
                    head.weight[compact_idx].copy_(source_head.weight[token_id])
                    if source_head.bias is not None:
                        head.bias[compact_idx].copy_(source_head.bias[token_id])
    with torch.no_grad():
        head.bias[int(blank_index)] = float(blank_logit_bias)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.compact_ctc_head = head.to(device=device, dtype=dtype)
    model.compact_ctc_token_ids = compact_token_ids
    model.compact_ctc_blank_index = int(blank_index)


def _add_compact_ctc_meta(model: nn.Module, meta: dict[str, Any]) -> None:
    token_ids = getattr(model, "compact_ctc_token_ids", None)
    head = getattr(model, "compact_ctc_head", None)
    if token_ids is not None and isinstance(head, nn.Linear):
        meta["compact_ctc_token_ids"] = [int(token_id) for token_id in token_ids]
        meta["compact_ctc_blank_index"] = int(
            getattr(model, "compact_ctc_blank_index", 0)
        )


def _restore_compact_ctc_from_meta(model: nn.Module, meta: dict[str, Any]) -> None:
    token_ids = meta.get("compact_ctc_token_ids")
    if token_ids is None:
        return
    configure_compact_ctc_head(
        model,
        [int(token_id) for token_id in token_ids],
        blank_index=int(meta.get("compact_ctc_blank_index", 0)),
        init_from_ctc_head=False,
    )


def configure_rnnt_lite_head(
    model: nn.Module,
    token_ids: list[int] | tuple[int, ...],
    *,
    blank_index: int = 0,
    pred_dim: int = 0,
    joint_dim: int = 0,
    blank_logit_bias: float = 0.0,
    init_from_ctc_head: bool = True,
) -> None:
    rnnt_token_ids = [int(token_id) for token_id in token_ids]
    if not rnnt_token_ids:
        raise ValueError("RNNT-lite vocab must not be empty")
    if not 0 <= int(blank_index) < len(rnnt_token_ids):
        raise ValueError("RNNT-lite blank index is out of range")

    config = getattr(model, "config")
    d_model = int(config.d_model)
    pred_dim = int(pred_dim) if int(pred_dim) > 0 else d_model
    joint_dim = int(joint_dim) if int(joint_dim) > 0 else d_model

    predictor = nn.Embedding(len(rnnt_token_ids), pred_dim)
    audio_proj = nn.Linear(d_model, joint_dim, bias=False)
    pred_proj = nn.Linear(pred_dim, joint_dim, bias=False)
    norm = nn.LayerNorm(joint_dim)
    head = nn.Linear(joint_dim, len(rnnt_token_ids), bias=True)

    nn.init.normal_(predictor.weight, mean=0.0, std=0.02)
    nn.init.xavier_uniform_(audio_proj.weight)
    nn.init.xavier_uniform_(pred_proj.weight)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)

    source_embed = getattr(model, "embed_tokens", None)
    if (
        isinstance(source_embed, nn.Embedding)
        and source_embed.weight.shape[1] == pred_dim
    ):
        with torch.no_grad():
            for compact_idx, token_id in enumerate(rnnt_token_ids):
                if 0 <= token_id < source_embed.weight.shape[0]:
                    predictor.weight[compact_idx].copy_(source_embed.weight[token_id])

    source_head = getattr(model, "ctc_head", None)
    if (
        init_from_ctc_head
        and isinstance(source_head, nn.Linear)
        and source_head.weight.shape[1] == joint_dim
    ):
        with torch.no_grad():
            for compact_idx, token_id in enumerate(rnnt_token_ids):
                if 0 <= token_id < source_head.weight.shape[0]:
                    head.weight[compact_idx].copy_(source_head.weight[token_id])
                    if source_head.bias is not None:
                        head.bias[compact_idx].copy_(source_head.bias[token_id])
    with torch.no_grad():
        head.bias[int(blank_index)] = float(blank_logit_bias)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.rnnt_lite_predictor = predictor.to(device=device, dtype=dtype)
    model.rnnt_lite_audio_proj = audio_proj.to(device=device, dtype=dtype)
    model.rnnt_lite_pred_proj = pred_proj.to(device=device, dtype=dtype)
    model.rnnt_lite_norm = norm.to(device=device, dtype=dtype)
    model.rnnt_lite_head = head.to(device=device, dtype=dtype)
    model.rnnt_lite_token_ids = rnnt_token_ids
    model.rnnt_lite_blank_index = int(blank_index)
    model.rnnt_lite_pred_dim = pred_dim
    model.rnnt_lite_joint_dim = joint_dim


def _add_rnnt_lite_meta(model: nn.Module, meta: dict[str, Any]) -> None:
    token_ids = getattr(model, "rnnt_lite_token_ids", None)
    head = getattr(model, "rnnt_lite_head", None)
    if token_ids is not None and isinstance(head, nn.Linear):
        meta["rnnt_lite_token_ids"] = [int(token_id) for token_id in token_ids]
        meta["rnnt_lite_blank_index"] = int(
            getattr(model, "rnnt_lite_blank_index", 0)
        )
        meta["rnnt_lite_pred_dim"] = int(getattr(model, "rnnt_lite_pred_dim", 0))
        meta["rnnt_lite_joint_dim"] = int(getattr(model, "rnnt_lite_joint_dim", 0))


def _restore_rnnt_lite_from_meta(model: nn.Module, meta: dict[str, Any]) -> None:
    token_ids = meta.get("rnnt_lite_token_ids")
    if token_ids is None:
        return
    configure_rnnt_lite_head(
        model,
        [int(token_id) for token_id in token_ids],
        blank_index=int(meta.get("rnnt_lite_blank_index", 0)),
        pred_dim=int(meta.get("rnnt_lite_pred_dim", 0)),
        joint_dim=int(meta.get("rnnt_lite_joint_dim", 0)),
        init_from_ctc_head=False,
    )


def _rnnt_lite_logits_from_frames(
    model: nn.Module,
    frame_hidden: torch.Tensor,
    previous_compact_ids: torch.Tensor,
) -> torch.Tensor:
    predictor = getattr(model, "rnnt_lite_predictor", None)
    audio_proj = getattr(model, "rnnt_lite_audio_proj", None)
    pred_proj = getattr(model, "rnnt_lite_pred_proj", None)
    norm = getattr(model, "rnnt_lite_norm", None)
    head = getattr(model, "rnnt_lite_head", None)
    if not all(
        isinstance(module, nn.Module)
        for module in (predictor, audio_proj, pred_proj, norm, head)
    ):
        raise ValueError("RNNT-lite head is not configured")
    steps = min(frame_hidden.shape[1], previous_compact_ids.shape[1])
    if steps == 0:
        return frame_hidden.new_zeros(frame_hidden.shape[0], 0, head.out_features)
    frame_hidden = frame_hidden[:, :steps, :]
    previous_compact_ids = previous_compact_ids[:, :steps]
    pred = predictor(previous_compact_ids)
    joint = audio_proj(frame_hidden.to(dtype=audio_proj.weight.dtype))
    joint = joint + pred_proj(pred.to(dtype=pred_proj.weight.dtype))
    joint = torch.tanh(norm(joint))
    return head(joint.to(dtype=head.weight.dtype))


def _rnnt_lite_joint_logits_from_frames(
    model: nn.Module,
    frame_hidden: torch.Tensor,
    previous_compact_ids: torch.Tensor,
) -> torch.Tensor:
    predictor = getattr(model, "rnnt_lite_predictor", None)
    audio_proj = getattr(model, "rnnt_lite_audio_proj", None)
    pred_proj = getattr(model, "rnnt_lite_pred_proj", None)
    norm = getattr(model, "rnnt_lite_norm", None)
    head = getattr(model, "rnnt_lite_head", None)
    if not all(
        isinstance(module, nn.Module)
        for module in (predictor, audio_proj, pred_proj, norm, head)
    ):
        raise ValueError("RNNT-lite head is not configured")
    if frame_hidden.ndim != 3:
        raise ValueError("frame_hidden must have shape [batch, T, d_model]")
    if previous_compact_ids.ndim != 2:
        raise ValueError("previous_compact_ids must have shape [batch, U + 1]")
    if frame_hidden.shape[0] != previous_compact_ids.shape[0]:
        raise ValueError("frame and prediction state batch sizes differ")
    if frame_hidden.shape[1] == 0 or previous_compact_ids.shape[1] == 0:
        return frame_hidden.new_zeros(
            frame_hidden.shape[0],
            frame_hidden.shape[1],
            previous_compact_ids.shape[1],
            head.out_features,
        )
    audio = audio_proj(frame_hidden.to(dtype=audio_proj.weight.dtype))
    pred = predictor(previous_compact_ids)
    pred = pred_proj(pred.to(dtype=pred_proj.weight.dtype))
    joint = audio[:, :, None, :] + pred[:, None, :, :]
    joint = torch.tanh(norm(joint))
    return head(joint.to(dtype=head.weight.dtype))


def add_lora_to_linear_modules(
    root: nn.Module,
    *,
    target_names: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    replaced: list[str] = []
    for name, module in list(root.named_modules()):
        if not name or isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name not in target_names:
            continue
        _set_child_module(
            root,
            name,
            LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
        replaced.append(name)
    return replaced


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / float(head_dim)
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        position_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = q.shape[-2]
        positions = torch.arange(
            position_offset,
            position_offset + length,
            device=q.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("t,d->td", positions, self.inv_freq)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        cos = emb.cos().to(dtype=q.dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=q.dtype)[None, None, :, :]
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class CausalConvStem(nn.Module):
    def __init__(self, config: RealtimeAudioConfig) -> None:
        super().__init__()
        self.kernel_size = config.conv_kernel_size
        self.proj = nn.Conv1d(
            config.n_mels,
            config.d_model,
            kernel_size=self.kernel_size,
            bias=False,
        )

    def forward_chunk(
        self,
        mels: torch.Tensor,
        tail: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mels.ndim != 3:
            raise ValueError("mels must have shape [batch, frames, n_mels]")
        batch, _, n_mels = mels.shape
        left = self.kernel_size - 1
        if tail is None:
            tail = mels.new_zeros(batch, left, n_mels)
        elif tail.shape != (batch, left, n_mels):
            raise ValueError(
                f"conv tail must have shape {(batch, left, n_mels)}, got {tuple(tail.shape)}"
            )

        combined = torch.cat([tail, mels], dim=1)
        output = self.proj(combined.transpose(1, 2)).transpose(1, 2)
        next_tail = combined[:, -left:, :].detach() if left else combined[:, :0, :]
        return output, next_tail


class CachedSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        rope_theta: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim, theta=rope_theta)

    def forward_chunk(
        self,
        x: torch.Tensor,
        cache: AudioLayerCache,
        *,
        position_offset: int,
        window_frames: int | None,
    ) -> tuple[torch.Tensor, AudioLayerCache]:
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.rope(q, k, position_offset=position_offset)

        past_k = cache.key
        past_v = cache.value
        past_len = 0 if past_k is None else past_k.shape[-2]
        if past_k is not None and past_v is not None:
            all_k = torch.cat([past_k, k], dim=-2)
            all_v = torch.cat([past_v, v], dim=-2)
        else:
            all_k = k
            all_v = v

        total_len = all_k.shape[-2]
        cache_start_pos = position_offset - past_len
        q_positions = torch.arange(
            position_offset,
            position_offset + length,
            device=x.device,
        )
        k_positions = torch.arange(
            cache_start_pos,
            cache_start_pos + total_len,
            device=x.device,
        )
        allowed = k_positions[None, :] <= q_positions[:, None]
        if window_frames is not None:
            allowed &= k_positions[None, :] >= (
                q_positions[:, None] - window_frames + 1
            )

        scores = torch.matmul(q, all_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed[None, None, :, :], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        context = torch.matmul(weights, all_v)
        context = context.transpose(1, 2).contiguous().view(batch, length, self.d_model)

        keep = window_frames if window_frames is not None else total_len
        next_cache = AudioLayerCache(
            key=all_k[:, :, -keep:, :].detach(),
            value=all_v[:, :, -keep:, :].detach(),
        )
        return self.out(context), next_cache


class CausalTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        ffn_multiplier: int,
        rope_theta: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CachedSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope_theta=rope_theta,
            dropout=dropout,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, hidden_dim=d_model * ffn_multiplier)
        self.dropout = nn.Dropout(dropout)

    def forward_chunk(
        self,
        x: torch.Tensor,
        cache: AudioLayerCache,
        *,
        position_offset: int,
        window_frames: int | None,
    ) -> tuple[torch.Tensor, AudioLayerCache]:
        attn_out, next_cache = self.attn.forward_chunk(
            self.attn_norm(x),
            cache,
            position_offset=position_offset,
            window_frames=window_frames,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, next_cache


class CausalAudioEncoder(nn.Module):
    def __init__(self, config: RealtimeAudioConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = CausalConvStem(config)
        self.layers = nn.ModuleList(
            [
                CausalTransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.audio_num_heads,
                    ffn_multiplier=config.audio_ffn_multiplier,
                    rope_theta=config.rope_theta,
                    dropout=config.dropout,
                )
                for _ in range(config.audio_num_layers)
            ]
        )
        self.norm = RMSNorm(config.d_model)

    def init_state(self) -> AudioEncoderState:
        return AudioEncoderState(
            layer_caches=[AudioLayerCache() for _ in range(len(self.layers))]
        )

    def forward_chunk(
        self,
        mels: torch.Tensor,
        state: AudioEncoderState | None = None,
    ) -> tuple[torch.Tensor, AudioEncoderState]:
        if state is None:
            state = self.init_state()
        if not state.layer_caches:
            state.layer_caches = [AudioLayerCache() for _ in range(len(self.layers))]
        if len(state.layer_caches) != len(self.layers):
            raise ValueError("audio cache layer count does not match encoder layers")

        x, state.conv_tail = self.stem.forward_chunk(mels, state.conv_tail)
        position_offset = state.frames_seen
        next_caches: list[AudioLayerCache] = []
        for layer, cache in zip(self.layers, state.layer_caches, strict=True):
            x, next_cache = layer.forward_chunk(
                x,
                cache,
                position_offset=position_offset,
                window_frames=self.config.audio_window_frames,
            )
            next_caches.append(next_cache)
        state.layer_caches = next_caches
        state.frames_seen += mels.shape[1]
        return self.norm(x), state

    def forward_full(self, mels: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_chunk(mels, self.init_state())
        return output


class StreamingFrameAdapter(nn.Module):
    def __init__(self, config: RealtimeAudioConfig) -> None:
        super().__init__()
        self.frames_per_step = config.frames_per_decoder_step
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def init_state(self) -> FrameAdapterState:
        return FrameAdapterState()

    def forward_chunk(
        self,
        audio_hidden: torch.Tensor,
        state: FrameAdapterState | None = None,
    ) -> tuple[torch.Tensor, FrameAdapterState]:
        if state is None:
            state = self.init_state()
        new_frames = audio_hidden.shape[1]
        if state.pending is not None:
            audio_hidden = torch.cat([state.pending, audio_hidden], dim=1)

        usable = (audio_hidden.shape[1] // self.frames_per_step) * self.frames_per_step
        complete = audio_hidden[:, :usable, :]
        state.pending = audio_hidden[:, usable:, :].detach()
        state.audio_frames_seen += new_frames

        if usable == 0:
            empty = audio_hidden.new_zeros(audio_hidden.shape[0], 0, audio_hidden.shape[-1])
            return empty, state
        steps = complete.view(
            complete.shape[0],
            usable // self.frames_per_step,
            self.frames_per_step,
            complete.shape[-1],
        ).mean(dim=2)
        state.decoder_steps_seen += steps.shape[1]
        return self.proj(steps), state

    def forward_full(self, audio_hidden: torch.Tensor) -> torch.Tensor:
        steps, _ = self.forward_chunk(audio_hidden, self.init_state())
        return steps


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


def _ensure_decoding_state(
    state: RealtimeModelState,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    if len(state.token_history) != batch_size:
        state.token_history = [[] for _ in range(batch_size)]
    if (
        state.consecutive_text_tokens is None
        or state.consecutive_text_tokens.shape != (batch_size,)
    ):
        state.consecutive_text_tokens = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=device,
        )
    else:
        state.consecutive_text_tokens = state.consecutive_text_tokens.to(device=device)


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


def _update_decoding_history(
    state: RealtimeModelState,
    next_token: torch.Tensor,
    *,
    wait_token_id: int | None,
    max_history_tokens: int = 512,
) -> None:
    if state.consecutive_text_tokens is None:
        state.consecutive_text_tokens = torch.zeros_like(next_token)
    for batch_idx, token in enumerate(next_token.detach().cpu().tolist()):
        token_id = int(token)
        if wait_token_id is not None and token_id == int(wait_token_id):
            state.consecutive_text_tokens[batch_idx] = 0
            continue
        state.token_history[batch_idx].append(token_id)
        if len(state.token_history[batch_idx]) > max_history_tokens:
            state.token_history[batch_idx] = state.token_history[batch_idx][
                -max_history_tokens:
            ]
        state.consecutive_text_tokens[batch_idx] += 1


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
        allowed = k_positions[None, :] <= q_positions[:, None]
        allowed &= k_positions[None, :] >= (
            q_positions[:, None] - self.left_context_steps + 1
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


class CachedDecoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_multiplier: int,
        rope_theta: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CausalTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_multiplier=ffn_multiplier,
                    rope_theta=rope_theta,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)

    def init_state(self) -> DecoderState:
        return DecoderState(layer_caches=[AudioLayerCache() for _ in range(len(self.layers))])

    def forward_chunk(
        self,
        inputs_embeds: torch.Tensor,
        state: DecoderState | None = None,
    ) -> tuple[torch.Tensor, DecoderState]:
        if state is None:
            state = self.init_state()
        if not state.layer_caches:
            state.layer_caches = [AudioLayerCache() for _ in range(len(self.layers))]
        if len(state.layer_caches) != len(self.layers):
            raise ValueError("decoder cache layer count does not match decoder layers")

        x = inputs_embeds
        position_offset = state.steps_seen
        next_caches: list[AudioLayerCache] = []
        for layer, cache in zip(self.layers, state.layer_caches, strict=True):
            x, next_cache = layer.forward_chunk(
                x,
                cache,
                position_offset=position_offset,
                window_frames=None,
            )
            next_caches.append(next_cache)
        state.layer_caches = next_caches
        state.steps_seen += inputs_embeds.shape[1]
        return self.norm(x), state


class Qwen3ASRRealtimeNativeModel(nn.Module):
    """Native realtime ASR scaffold with append-only audio and decoder caches.

    This class is intentionally decoder-agnostic for the first experiments. The
    smoke path uses a small cached decoder; a later H100 iteration can replace
    ``decoder``, ``embed_tokens`` and ``lm_head`` with Qwen3-ASR weights.
    """

    def __init__(
        self,
        config: RealtimeAudioConfig,
        *,
        vocab_size: int,
        bos_token_id: int,
        decoder_num_layers: int = 4,
        decoder_num_heads: int | None = None,
        decoder_ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.decoder_num_layers = decoder_num_layers
        self.decoder_num_heads = decoder_num_heads or config.audio_num_heads
        self.decoder_ffn_multiplier = decoder_ffn_multiplier
        self.audio_encoder = CausalAudioEncoder(config)
        self.adapter = StreamingFrameAdapter(config)
        self.embed_tokens = nn.Embedding(vocab_size, config.d_model)
        self.decoder = CachedDecoder(
            d_model=config.d_model,
            num_layers=decoder_num_layers,
            num_heads=self.decoder_num_heads,
            ffn_multiplier=decoder_ffn_multiplier,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.ctc_head = nn.Linear(config.d_model, vocab_size, bias=True)
        self.repetition_penalty = 1.0
        self.no_repeat_ngram_size = 0
        self.max_consecutive_text_tokens = 0

    def init_stream_state(
        self,
        *,
        batch_size: int,
        device: torch.device | str,
        last_token_ids: torch.Tensor | None = None,
    ) -> RealtimeModelState:
        if last_token_ids is None:
            last_token_ids = torch.full(
                (batch_size,),
                self.bos_token_id,
                dtype=torch.long,
                device=device,
            )
        return RealtimeModelState(
            audio=self.audio_encoder.init_state(),
            adapter=self.adapter.init_state(),
            decoder=self.decoder.init_state(),
            last_token_ids=last_token_ids,
        )

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
                    stop_fill_id = int(eos_token_id) if eos_token_id is not None else min(stop_ids)
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

    def forward(
        self,
        mels: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(self.forward_hidden(mels, previous_token_ids))

    def forward_audio_frames(self, mels: torch.Tensor) -> torch.Tensor:
        audio_hidden = self.audio_encoder.forward_full(mels)
        return self.adapter.forward_full(audio_hidden)

    def forward_ctc_logits(self, mels: torch.Tensor) -> torch.Tensor:
        frame_hidden = self.forward_audio_frames(mels)
        return self.ctc_head(frame_hidden.to(dtype=self.ctc_head.weight.dtype))

    def forward_compact_ctc_logits(self, mels: torch.Tensor) -> torch.Tensor:
        head = getattr(self, "compact_ctc_head", None)
        if not isinstance(head, nn.Linear):
            raise ValueError("compact CTC head is not configured")
        frame_hidden = self.forward_audio_frames(mels)
        return head(frame_hidden.to(dtype=head.weight.dtype))

    def forward_rnnt_lite_logits(
        self,
        mels: torch.Tensor,
        previous_compact_ids: torch.Tensor,
    ) -> torch.Tensor:
        frame_hidden = self.forward_audio_frames(mels)
        return self.forward_rnnt_lite_logits_from_frames(
            frame_hidden,
            previous_compact_ids,
        )

    def forward_rnnt_lite_logits_from_frames(
        self,
        frame_hidden: torch.Tensor,
        previous_compact_ids: torch.Tensor,
    ) -> torch.Tensor:
        return _rnnt_lite_logits_from_frames(self, frame_hidden, previous_compact_ids)

    def forward_rnnt_lite_joint_logits(
        self,
        mels: torch.Tensor,
        previous_compact_ids: torch.Tensor,
    ) -> torch.Tensor:
        frame_hidden = self.forward_audio_frames(mels)
        return self.forward_rnnt_lite_joint_logits_from_frames(
            frame_hidden,
            previous_compact_ids,
        )

    def forward_rnnt_lite_joint_logits_from_frames(
        self,
        frame_hidden: torch.Tensor,
        previous_compact_ids: torch.Tensor,
    ) -> torch.Tensor:
        return _rnnt_lite_joint_logits_from_frames(
            self,
            frame_hidden,
            previous_compact_ids,
        )

    def forward_hidden(
        self,
        mels: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        frame_hidden = self.forward_audio_frames(mels)
        if previous_token_ids.shape != frame_hidden.shape[:2]:
            raise ValueError(
                "previous_token_ids must have shape [batch, decoder_steps]; "
                f"got {tuple(previous_token_ids.shape)}, expected {tuple(frame_hidden.shape[:2])}"
            )
        decoder_inputs = frame_hidden + self.embed_tokens(previous_token_ids)
        hidden, _ = self.decoder.forward_chunk(decoder_inputs, self.decoder.init_state())
        return hidden

    @torch.no_grad()
    def stream_chunk(
        self,
        mels: torch.Tensor,
        state: RealtimeModelState,
    ) -> tuple[torch.Tensor, torch.Tensor, RealtimeModelState]:
        audio_hidden, state.audio = self.audio_encoder.forward_chunk(mels, state.audio)
        frame_hidden, state.adapter = self.adapter.forward_chunk(audio_hidden, state.adapter)
        if frame_hidden.shape[1] == 0:
            empty_logits = mels.new_zeros(mels.shape[0], 0, self.vocab_size)
            empty_tokens = torch.empty(mels.shape[0], 0, dtype=torch.long, device=mels.device)
            return empty_logits, empty_tokens, state

        _ensure_decoding_state(state, batch_size=mels.shape[0], device=mels.device)
        logits_out: list[torch.Tensor] = []
        token_out: list[torch.Tensor] = []
        for idx in range(frame_hidden.shape[1]):
            token_embed = self.embed_tokens(state.last_token_ids).unsqueeze(1)
            decoder_input = frame_hidden[:, idx : idx + 1, :] + token_embed
            hidden, state.decoder = self.decoder.forward_chunk(decoder_input, state.decoder)
            logits = self.lm_head(hidden)
            logits_step = logits[:, -1, :]
            controlled_logits = _apply_repetition_controls_to_logits(
                logits_step,
                token_history=state.token_history,
                consecutive_text_tokens=state.consecutive_text_tokens,
                repetition_penalty=float(self.repetition_penalty),
                no_repeat_ngram_size=int(self.no_repeat_ngram_size),
                max_consecutive_text_tokens=int(self.max_consecutive_text_tokens),
                wait_token_id=getattr(self, "wait_token_id", None),
            )
            if controlled_logits is not logits_step:
                logits = logits.clone()
                logits[:, -1, :] = controlled_logits
            next_token = controlled_logits.argmax(dim=-1)
            state.last_token_ids = next_token
            _update_decoding_history(
                state,
                next_token,
                wait_token_id=getattr(self, "wait_token_id", None),
            )
            logits_out.append(logits)
            token_out.append(next_token[:, None])

        return torch.cat(logits_out, dim=1), torch.cat(token_out, dim=1), state

    def save_pretrained(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "realtime_config.json").write_text(
            json.dumps(asdict(self.config), indent=2),
            encoding="utf-8",
        )
        meta = {
            "vocab_size": self.vocab_size,
            "bos_token_id": self.bos_token_id,
            "decoder_num_layers": self.decoder_num_layers,
            "decoder_num_heads": self.decoder_num_heads,
            "decoder_ffn_multiplier": self.decoder_ffn_multiplier,
            "class": self.__class__.__name__,
        }
        _add_compact_ctc_meta(self, meta)
        _add_rnnt_lite_meta(self, meta)
        (output / "realtime_model_meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        torch.save(self.state_dict(), output / "model.pt")

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "Qwen3ASRRealtimeNativeModel":
        model_path = Path(model_dir)
        config = RealtimeAudioConfig(
            **json.loads((model_path / "realtime_config.json").read_text(encoding="utf-8"))
        )
        meta = json.loads(
            (model_path / "realtime_model_meta.json").read_text(encoding="utf-8")
        )
        run_config_path = model_path / "run_config.json"
        run_args = {}
        if run_config_path.exists():
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            run_args = run_config.get("args", {})

        model = cls(
            config,
            vocab_size=int(meta["vocab_size"]),
            bos_token_id=int(meta["bos_token_id"]),
            decoder_num_layers=int(
                meta.get("decoder_num_layers", run_args.get("decoder_layers", 4))
            ),
            decoder_num_heads=int(
                meta.get(
                    "decoder_num_heads",
                    run_args.get("decoder_heads", config.audio_num_heads),
                )
            ),
            decoder_ffn_multiplier=int(
                meta.get("decoder_ffn_multiplier", run_args.get("decoder_ffn_multiplier", 2))
            ),
        )
        _restore_compact_ctc_from_meta(model, meta)
        _restore_rnnt_lite_from_meta(model, meta)
        state_dict = torch.load(
            model_path / "model.pt",
            map_location=map_location,
            weights_only=True,
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        allowed_missing = {"ctc_head.weight", "ctc_head.bias"}
        if unexpected or any(key not in allowed_missing for key in missing):
            raise RuntimeError(
                f"Unexpected checkpoint keys. missing={missing}, unexpected={unexpected}"
            )
        return model


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
        audio_encoder: nn.Module | None = None,
        adapter: nn.Module | None = None,
        audio_backend: str = "scratch",
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_model_id = qwen_model_id
        self.bos_token_id = bos_token_id
        self.wait_token_id = wait_token_id
        self.audio_backend = audio_backend
        self.emit_threshold = 0.5
        self.repetition_penalty = 1.0
        self.no_repeat_ngram_size = 0
        self.max_consecutive_text_tokens = 0
        self.audio_encoder = audio_encoder if audio_encoder is not None else CausalAudioEncoder(config)
        self.adapter = adapter if adapter is not None else StreamingFrameAdapter(config)
        self.text_model = text_model
        self.lm_head = lm_head
        self.emit_head = nn.Linear(config.d_model, 1)
        self.embed_tokens = self.text_model.embed_tokens
        self.vocab_size = int(self.lm_head.weight.shape[0])
        self.ctc_head = nn.Linear(config.d_model, self.vocab_size, bias=True)
        _init_ctc_head(
            self.ctc_head,
            lm_head=self.lm_head,
            blank_token_id=self.wait_token_id,
        )
        self.qwen_lora_config: dict[str, Any] | None = None
        self.qwen_lora_modules: list[str] = []
        self.qwen_audio_lora_config: dict[str, Any] | None = None
        self.qwen_audio_lora_modules: list[str] = []

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
    ) -> "Qwen3ASRRealtimeQwenDecoderModel":
        _register_qwen3_asr_transformers()
        from transformers import AutoConfig, AutoModel

        qwen_config = AutoConfig.from_pretrained(qwen_model_id)
        text_config = qwen_config.thinker_config.text_config
        if config is None:
            config = RealtimeAudioConfig(
                d_model=int(text_config.hidden_size),
                audio_num_layers=3,
                audio_num_heads=8,
                audio_ffn_multiplier=2,
                conv_kernel_size=5,
                audio_window_sec=15.0,
            )
        if config.d_model != int(text_config.hidden_size):
            raise ValueError(
                f"Realtime d_model={config.d_model} must match Qwen hidden_size="
                f"{int(text_config.hidden_size)}"
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
            text_model=thinker.model,
            lm_head=thinker.lm_head,
            bos_token_id=bos_token_id,
            wait_token_id=wait_token_id,
        )

    def freeze_qwen_layers(self, *, train_last_n_layers: int = 0) -> None:
        if train_last_n_layers < 0:
            raise ValueError("train_last_n_layers must be >= 0")
        if train_last_n_layers > len(self.text_model.layers):
            raise ValueError(
                f"train_last_n_layers={train_last_n_layers} exceeds "
                f"Qwen layer count {len(self.text_model.layers)}"
            )
        for param in self.text_model.layers.parameters():
            param.requires_grad = False
        for param in self.text_model.norm.parameters():
            param.requires_grad = False
        if train_last_n_layers:
            for layer in self.text_model.layers[-train_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.text_model.norm.parameters():
                param.requires_grad = True

    def freeze_qwen_all(self) -> None:
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def add_qwen_lora(
        self,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        target_names: tuple[str, ...] = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    ) -> list[str]:
        modules = add_lora_to_linear_modules(
            self.text_model,
            target_names=target_names,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        if not modules:
            raise ValueError(
                "No Qwen linear modules matched LoRA targets: "
                f"{', '.join(target_names)}"
            )
        self.qwen_lora_modules = modules
        self.qwen_lora_config = {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "target_names": list(target_names),
            "modules": modules,
        }
        return modules

    def init_stream_state(
        self,
        *,
        batch_size: int,
        device: torch.device | str,
        last_token_ids: torch.Tensor | None = None,
    ) -> RealtimeModelState:
        if last_token_ids is None:
            last_token_ids = torch.full(
                (batch_size,),
                self.bos_token_id,
                dtype=torch.long,
                device=device,
            )
        return RealtimeModelState(
            audio=self.audio_encoder.init_state(),
            adapter=self.adapter.init_state(),
            decoder=QwenDecoderState(),
            last_token_ids=last_token_ids,
        )

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

    def forward(
        self,
        mels: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(self.forward_hidden(mels, previous_token_ids))

    def forward_audio_frames(self, mels: torch.Tensor) -> torch.Tensor:
        audio_hidden = self.audio_encoder.forward_full(mels)
        return self.adapter.forward_full(audio_hidden)

    def forward_qwen_ar_logits_from_cached_audio(
        self,
        frame_hidden: torch.Tensor,
        *,
        prefix_token_ids: torch.Tensor | Sequence[int],
        audio_placeholder_token_id: int,
        target_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced Qwen next-token logits over cached audio embeddings.

        ``prefix_token_ids`` must already contain one audio placeholder token per
        finalized audio frame. Logits are aligned with ``target_token_ids``:
        the last prefix position predicts target token 0, then each previous
        target token predicts the next target token.
        """
        if frame_hidden.ndim != 3:
            raise ValueError("frame_hidden must have shape [batch, steps, hidden]")
        if target_token_ids.ndim != 2:
            raise ValueError("target_token_ids must have shape [batch, target_steps]")
        batch_size = int(frame_hidden.shape[0])
        if int(target_token_ids.shape[0]) != batch_size:
            raise ValueError(
                "target_token_ids batch size must match frame_hidden; got "
                f"{int(target_token_ids.shape[0])} and {batch_size}"
            )
        target_steps = int(target_token_ids.shape[1])
        if target_steps <= 0:
            return frame_hidden.new_zeros(batch_size, 0, self.vocab_size)

        fixed_prefix = _cached_audio_prefix_embeds(
            self,
            frame_hidden,
            prefix_token_ids=prefix_token_ids,
            audio_placeholder_token_id=int(audio_placeholder_token_id),
        )
        prefix_steps = int(fixed_prefix.shape[1])
        if prefix_steps <= 0:
            raise ValueError("Qwen AR CE requires a non-empty prompt/audio prefix")

        parts = [fixed_prefix]
        if target_steps > 1:
            previous_targets = target_token_ids[:, :-1].to(
                device=frame_hidden.device,
                dtype=torch.long,
            )
            parts.append(self.embed_tokens(previous_targets))
        decoder_inputs = torch.cat(parts, dim=1)
        outputs = self.text_model(inputs_embeds=decoder_inputs, use_cache=False)
        target_hidden = outputs.last_hidden_state[
            :, prefix_steps - 1 : prefix_steps - 1 + target_steps, :
        ]
        return self.lm_head(target_hidden.to(dtype=self.lm_head.weight.dtype))

    def forward_ctc_logits(self, mels: torch.Tensor) -> torch.Tensor:
        frame_hidden = self.forward_audio_frames(mels)
        return self.ctc_head(frame_hidden.to(dtype=self.ctc_head.weight.dtype))

    def forward_compact_ctc_logits(self, mels: torch.Tensor) -> torch.Tensor:
        head = getattr(self, "compact_ctc_head", None)
        if not isinstance(head, nn.Linear):
            raise ValueError("compact CTC head is not configured")
        frame_hidden = self.forward_audio_frames(mels)
        return head(frame_hidden.to(dtype=head.weight.dtype))

    def forward_rnnt_lite_logits(
        self,
        mels: torch.Tensor,
        previous_compact_ids: torch.Tensor,
    ) -> torch.Tensor:
        frame_hidden = self.forward_audio_frames(mels)
        return self.forward_rnnt_lite_logits_from_frames(
            frame_hidden,
            previous_compact_ids,
        )

    def forward_rnnt_lite_logits_from_frames(
        self,
        frame_hidden: torch.Tensor,
        previous_compact_ids: torch.Tensor,
    ) -> torch.Tensor:
        return _rnnt_lite_logits_from_frames(self, frame_hidden, previous_compact_ids)

    def forward_rnnt_lite_joint_logits(
        self,
        mels: torch.Tensor,
        previous_compact_ids: torch.Tensor,
    ) -> torch.Tensor:
        frame_hidden = self.forward_audio_frames(mels)
        return self.forward_rnnt_lite_joint_logits_from_frames(
            frame_hidden,
            previous_compact_ids,
        )

    def forward_rnnt_lite_joint_logits_from_frames(
        self,
        frame_hidden: torch.Tensor,
        previous_compact_ids: torch.Tensor,
    ) -> torch.Tensor:
        return _rnnt_lite_joint_logits_from_frames(
            self,
            frame_hidden,
            previous_compact_ids,
        )

    def forward_hidden(
        self,
        mels: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        frame_hidden = self.forward_audio_frames(mels)
        if previous_token_ids.shape != frame_hidden.shape[:2]:
            raise ValueError(
                "previous_token_ids must have shape [batch, decoder_steps]; "
                f"got {tuple(previous_token_ids.shape)}, expected {tuple(frame_hidden.shape[:2])}"
            )
        token_embeds = self.embed_tokens(previous_token_ids)
        decoder_inputs = frame_hidden.to(dtype=token_embeds.dtype) + token_embeds
        outputs = self.text_model(inputs_embeds=decoder_inputs, use_cache=False)
        return outputs.last_hidden_state

    @torch.no_grad()
    def stream_chunk(
        self,
        mels: torch.Tensor,
        state: RealtimeModelState,
    ) -> tuple[torch.Tensor, torch.Tensor, RealtimeModelState]:
        if not isinstance(state.decoder, QwenDecoderState):
            raise TypeError("Qwen decoder backend requires QwenDecoderState")
        audio_hidden, state.audio = self.audio_encoder.forward_chunk(mels, state.audio)
        frame_hidden, state.adapter = self.adapter.forward_chunk(audio_hidden, state.adapter)
        if frame_hidden.shape[1] == 0:
            empty_logits = mels.new_zeros(mels.shape[0], 0, self.vocab_size)
            empty_tokens = torch.empty(mels.shape[0], 0, dtype=torch.long, device=mels.device)
            return empty_logits, empty_tokens, state

        _ensure_decoding_state(state, batch_size=mels.shape[0], device=mels.device)
        logits_out: list[torch.Tensor] = []
        token_out: list[torch.Tensor] = []
        for idx in range(frame_hidden.shape[1]):
            token_embed = self.embed_tokens(state.last_token_ids).unsqueeze(1)
            decoder_input = frame_hidden[:, idx : idx + 1, :].to(
                dtype=token_embed.dtype
            ) + token_embed
            outputs = self.text_model(
                inputs_embeds=decoder_input,
                past_key_values=state.decoder.past_key_values,
                use_cache=True,
            )
            state.decoder.past_key_values = outputs.past_key_values
            state.decoder.steps_seen += 1
            logits = self.lm_head(outputs.last_hidden_state)
            logits_step = logits[:, -1, :]
            controlled_logits = _apply_repetition_controls_to_logits(
                logits_step,
                token_history=state.token_history,
                consecutive_text_tokens=state.consecutive_text_tokens,
                repetition_penalty=float(self.repetition_penalty),
                no_repeat_ngram_size=int(self.no_repeat_ngram_size),
                max_consecutive_text_tokens=int(self.max_consecutive_text_tokens),
                wait_token_id=self.wait_token_id,
            )
            if controlled_logits is not logits_step:
                logits = logits.clone()
                logits[:, -1, :] = controlled_logits
            text_token = controlled_logits.argmax(dim=-1)
            if self.wait_token_id is None:
                next_token = text_token
            else:
                emit_logits = self.emit_head(
                    outputs.last_hidden_state[:, -1, :].to(
                        dtype=self.emit_head.weight.dtype
                    )
                )
                emit = emit_logits.squeeze(-1).sigmoid() >= self.emit_threshold
                wait_token = torch.full_like(text_token, int(self.wait_token_id))
                next_token = torch.where(emit, text_token, wait_token)
            state.last_token_ids = next_token
            _update_decoding_history(
                state,
                next_token,
                wait_token_id=self.wait_token_id,
            )
            logits_out.append(logits)
            token_out.append(next_token[:, None])

        return torch.cat(logits_out, dim=1), torch.cat(token_out, dim=1), state

    def save_pretrained(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "realtime_config.json").write_text(
            json.dumps(asdict(self.config), indent=2),
            encoding="utf-8",
        )
        meta = {
            "vocab_size": self.vocab_size,
            "bos_token_id": self.bos_token_id,
            "wait_token_id": self.wait_token_id,
            "qwen_model_id": self.qwen_model_id,
            "qwen_lora_config": self.qwen_lora_config,
            "qwen_audio_lora_config": self.qwen_audio_lora_config,
            "audio_backend": self.audio_backend,
            "class": self.__class__.__name__,
        }
        _add_compact_ctc_meta(self, meta)
        _add_rnnt_lite_meta(self, meta)
        (output / "realtime_model_meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        torch.save(self.state_dict(), output / "model.pt")

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "Qwen3ASRRealtimeQwenDecoderModel":
        model_path = Path(model_dir)
        config = RealtimeAudioConfig(
            **json.loads((model_path / "realtime_config.json").read_text(encoding="utf-8"))
        )
        meta = json.loads(
            (model_path / "realtime_model_meta.json").read_text(encoding="utf-8")
        )
        model = cls.from_qwen_pretrained(
            str(meta["qwen_model_id"]),
            config=config,
            bos_token_id=int(meta["bos_token_id"]),
            wait_token_id=(
                None if meta.get("wait_token_id") is None else int(meta["wait_token_id"])
            ),
            dtype=torch.bfloat16,
            device_map="cpu",
        )
        lora_config = meta.get("qwen_lora_config")
        if lora_config:
            model.freeze_qwen_all()
            model.add_qwen_lora(
                rank=int(lora_config["rank"]),
                alpha=float(lora_config["alpha"]),
                dropout=float(lora_config.get("dropout", 0.0)),
                target_names=tuple(str(name) for name in lora_config["target_names"]),
            )
        state_dict = torch.load(
            model_path / "model.pt",
            map_location=map_location,
            weights_only=True,
        )
        _restore_compact_ctc_from_meta(model, meta)
        _restore_rnnt_lite_from_meta(model, meta)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            "emit_head.weight",
            "emit_head.bias",
            "ctc_head.weight",
            "ctc_head.bias",
        }
        if unexpected or any(key not in allowed_missing for key in missing):
            raise RuntimeError(
                f"Unexpected checkpoint keys. missing={missing}, unexpected={unexpected}"
            )
        return model


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

    def freeze_qwen_audio_all(self) -> None:
        for param in self.audio_encoder.audio_tower.parameters():
            param.requires_grad = False

    def add_qwen_audio_lora(
        self,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        target_names: tuple[str, ...] = (
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "fc1",
            "fc2",
        ),
    ) -> list[str]:
        modules = add_lora_to_linear_modules(
            self.audio_encoder.audio_tower,
            target_names=target_names,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        if not modules:
            raise ValueError(
                "No Qwen audio tower linear modules matched LoRA targets: "
                f"{', '.join(target_names)}"
            )
        self.qwen_audio_lora_modules = modules
        self.qwen_audio_lora_config = {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "target_names": list(target_names),
            "modules": modules,
        }
        return modules

    def freeze_qwen_audio_layers(self, *, train_last_n_layers: int = 0) -> None:
        if train_last_n_layers < 0:
            raise ValueError("train_last_n_layers must be >= 0")
        tower = self.audio_encoder.audio_tower
        for param in tower.parameters():
            param.requires_grad = False
        layers = getattr(tower, "layers", None)
        if train_last_n_layers == 0:
            return
        if layers is None:
            raise ValueError("Qwen audio tower has no layers attribute")
        if train_last_n_layers > len(layers):
            raise ValueError(
                f"train_last_n_layers={train_last_n_layers} exceeds "
                f"Qwen audio layer count {len(layers)}"
            )
        for layer in layers[-train_last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        for name in ("ln_post", "proj1", "proj2"):
            module = getattr(tower, name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

    def forward_hidden(
        self,
        mels: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        frame_hidden = self.forward_audio_frames(mels)
        steps = min(previous_token_ids.shape[1], frame_hidden.shape[1])
        if steps == 0:
            return frame_hidden.new_zeros(frame_hidden.shape[0], 0, self.config.d_model)
        frame_hidden = frame_hidden[:, :steps, :]
        previous_token_ids = previous_token_ids[:, :steps]
        token_embeds = self.embed_tokens(previous_token_ids)
        decoder_inputs = frame_hidden.to(dtype=token_embeds.dtype) + token_embeds
        outputs = self.text_model(inputs_embeds=decoder_inputs, use_cache=False)
        return outputs.last_hidden_state

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "Qwen3ASRRealtimeQwenAudioSurgeryModel":
        model_path = Path(model_dir)
        config = RealtimeAudioConfig(
            **json.loads((model_path / "realtime_config.json").read_text(encoding="utf-8"))
        )
        meta = json.loads(
            (model_path / "realtime_model_meta.json").read_text(encoding="utf-8")
        )
        model = cls.from_qwen_pretrained(
            str(meta["qwen_model_id"]),
            config=config,
            bos_token_id=int(meta["bos_token_id"]),
            wait_token_id=(
                None if meta.get("wait_token_id") is None else int(meta["wait_token_id"])
            ),
            dtype=torch.bfloat16,
            device_map="cpu",
        )
        lora_config = meta.get("qwen_lora_config")
        if lora_config:
            model.freeze_qwen_all()
            model.add_qwen_lora(
                rank=int(lora_config["rank"]),
                alpha=float(lora_config["alpha"]),
                dropout=float(lora_config.get("dropout", 0.0)),
                target_names=tuple(str(name) for name in lora_config["target_names"]),
            )
        audio_lora_config = meta.get("qwen_audio_lora_config")
        if audio_lora_config:
            model.freeze_qwen_audio_all()
            model.add_qwen_audio_lora(
                rank=int(audio_lora_config["rank"]),
                alpha=float(audio_lora_config["alpha"]),
                dropout=float(audio_lora_config.get("dropout", 0.0)),
                target_names=tuple(
                    str(name) for name in audio_lora_config["target_names"]
                ),
            )
        state_dict = torch.load(
            model_path / "model.pt",
            map_location=map_location,
            weights_only=True,
        )
        _restore_compact_ctc_from_meta(model, meta)
        _restore_rnnt_lite_from_meta(model, meta)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            "emit_head.weight",
            "emit_head.bias",
            "ctc_head.weight",
            "ctc_head.bias",
        }
        if unexpected or any(key not in allowed_missing for key in missing):
            raise RuntimeError(
                f"Unexpected checkpoint keys. missing={missing}, unexpected={unexpected}"
            )
        return model


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


def load_realtime_model(
    model_dir: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> nn.Module:
    model_path = Path(model_dir)
    meta = json.loads(
        (model_path / "realtime_model_meta.json").read_text(encoding="utf-8")
    )
    class_name = meta.get("class", "Qwen3ASRRealtimeNativeModel")
    if class_name == "Qwen3ASRRealtimeQwenAudioCausalModel":
        return Qwen3ASRRealtimeQwenAudioCausalModel.from_pretrained(
            model_path,
            map_location=map_location,
        )
    if class_name == "Qwen3ASRRealtimeQwenAudioSurgeryModel":
        return Qwen3ASRRealtimeQwenAudioSurgeryModel.from_pretrained(
            model_path,
            map_location=map_location,
        )
    if class_name == "Qwen3ASRRealtimeQwenDecoderModel":
        return Qwen3ASRRealtimeQwenDecoderModel.from_pretrained(
            model_path,
            map_location=map_location,
        )
    if class_name == "Qwen3ASRRealtimeNativeModel":
        return Qwen3ASRRealtimeNativeModel.from_pretrained(
            model_path,
            map_location=map_location,
        )
    raise ValueError(f"Unknown realtime model class: {class_name}")
