"""Differentiable block-bidirectional tower forward for distillation training.

The inference path (``QwenAudioCausalKVEncoder`` with
``qwen_audio_block_bidirectional=True``, no mutable tail) processes audio in
blocks: bidirectional attention within a block, causal KV to previous blocks,
bounded left window. Because nothing is ever recomputed, the streaming values
equal a single full-sequence forward under a block-diagonal-causal mask.

``block_bidirectional_forward`` computes that full-sequence forward in
parallel, differentiably, over the same tower modules — so training optimizes
exactly the execution served at inference. Parity is enforced by
``tests/test_tower_distill.py``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .native_realtime_model import _module_device_dtype


def conv_blocks_batched(
    tower: nn.Module,
    mels: torch.Tensor,
    *,
    chunk_frames: int = 8,
) -> torch.Tensor:
    """Per-block conv stem + positional embedding, vectorized over blocks.

    Matches ``QwenAudioCausalKVEncoder._conv_blocks``: each ``chunk_frames``
    mel block goes through the conv2d stack independently and gets the
    sinusoidal position of its output step offset.

    mels: [batch, frames, n_mels] with frames a multiple of chunk_frames.
    Returns [batch, steps, d_model].
    """
    batch, frames, n_mels = mels.shape
    if frames % chunk_frames != 0:
        raise ValueError(f"frames={frames} not a multiple of chunk_frames={chunk_frames}")
    n_blocks = frames // chunk_frames
    device, dtype = _module_device_dtype(tower)
    x = mels.to(device=device, dtype=dtype)
    # [batch * n_blocks, chunk_frames, n_mels]
    x = x.reshape(batch * n_blocks, chunk_frames, n_mels)
    x = x.transpose(1, 2).unsqueeze(1)  # [B*N, 1, n_mels, chunk_frames]
    x = F.gelu(tower.conv2d1(x))
    x = F.gelu(tower.conv2d2(x))
    x = F.gelu(tower.conv2d3(x))
    bn, channels, freq, steps = x.size()
    x = tower.conv_out(
        x.permute(0, 3, 1, 2).contiguous().view(bn, steps, channels * freq)
    )  # [B*N, steps_per_block, d_model]
    steps_per_block = int(x.shape[1])
    x = x.reshape(batch, n_blocks * steps_per_block, -1)

    table = tower.positional_embedding.positional_embedding
    total_steps = n_blocks * steps_per_block
    if total_steps > table.shape[0]:
        raise ValueError("sequence longer than the positional embedding table")
    pos = table[:total_steps, :].to(device=x.device, dtype=x.dtype)
    return x + pos.unsqueeze(0)


def block_causal_mask(
    total_steps: int,
    *,
    steps_per_block: int,
    left_context_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """[q, k] boolean mask: bidirectional within block, causal across blocks,
    bounded left window — identical to the inference attention pattern."""
    positions = torch.arange(total_steps, device=device)
    block_end = (positions // steps_per_block + 1) * steps_per_block - 1
    allowed = positions[None, :] <= block_end[:, None]
    allowed &= positions[None, :] >= positions[:, None] - left_context_steps + 1
    return allowed


def block_bidirectional_forward(
    tower: nn.Module,
    mels: torch.Tensor,
    *,
    chunk_frames: int = 8,
    block_frames: int = 100,
    left_context_steps: int,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full-sequence differentiable forward under the streaming mask.

    block_frames: bidirectional block size in mel frames (e.g. 100 = 1s).
    lengths: optional per-sample valid mel frame counts (padding masked out
    of the attention keys).
    Returns [batch, steps, output_dim] after ln_post/proj1/act/proj2.
    """
    if block_frames % chunk_frames != 0:
        raise ValueError("block_frames must be a multiple of chunk_frames")
    hidden = conv_blocks_batched(tower, mels, chunk_frames=chunk_frames)
    batch, total_steps, _ = hidden.shape
    # conv emits a fixed number of steps per chunk_frames block
    steps_per_chunk = total_steps * chunk_frames // mels.shape[1]
    steps_per_block = block_frames // chunk_frames * steps_per_chunk

    allowed = block_causal_mask(
        total_steps,
        steps_per_block=steps_per_block,
        left_context_steps=left_context_steps,
        device=hidden.device,
    )  # [q, k]
    attn_mask = allowed[None, None, :, :]
    if lengths is not None:
        valid_steps = (lengths // chunk_frames * steps_per_chunk).to(hidden.device)
        key_valid = (
            torch.arange(total_steps, device=hidden.device)[None, :]
            < valid_steps[:, None]
        )
        attn_mask = attn_mask & key_valid[:, None, None, :]

    for layer in tower.layers:
        residual = hidden
        normed = layer.self_attn_layer_norm(hidden)
        attn = layer.self_attn
        num_heads = int(attn.num_heads)
        head_dim = int(attn.head_dim)
        q = attn.q_proj(normed).reshape(batch, total_steps, num_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(normed).reshape(batch, total_steps, num_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(normed).reshape(batch, total_steps, num_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * float(attn.scaling)
        scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights.to(dtype=v.dtype), v)
        context = context.transpose(1, 2).contiguous().view(batch, total_steps, -1)
        attn_out = attn.out_proj(context.to(dtype=attn.out_proj.weight.dtype))
        hidden = residual + attn_out
        residual = hidden
        hidden = layer.final_layer_norm(hidden)
        hidden = layer.fc2(layer.activation_fn(layer.fc1(hidden)))
        hidden = residual + hidden

    hidden = tower.ln_post(hidden)
    hidden = tower.proj1(hidden)
    hidden = tower.act(hidden)
    return tower.proj2(hidden)


@torch.no_grad()
def teacher_forward(
    tower: nn.Module,
    mels: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Offline bidirectional teacher embeddings, per-sample (native tower)."""
    device, dtype = _module_device_dtype(tower)
    outputs = []
    max_steps = 0
    for sample, length in zip(mels, lengths):
        valid = sample[: int(length), :].to(device=device, dtype=dtype)
        feature_lens = torch.tensor([valid.shape[0]], device=device, dtype=torch.long)
        try:
            out = tower(input_features=valid.transpose(0, 1), feature_lens=feature_lens)
        except TypeError:
            out = tower(valid.transpose(0, 1), feature_lens)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        if hidden.ndim == 3 and hidden.shape[0] == 1:
            hidden = hidden[0]
        outputs.append(hidden)
        max_steps = max(max_steps, int(hidden.shape[0]))
    result = outputs[0].new_zeros(len(outputs), max_steps, outputs[0].shape[-1])
    for idx, hidden in enumerate(outputs):
        result[idx, : hidden.shape[0], :] = hidden
    return result


def distill_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    lengths_steps: torch.Tensor,
    cosine_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """MSE + cosine distance over valid steps only."""
    steps = min(student.shape[1], teacher.shape[1])
    student = student[:, :steps, :].float()
    teacher = teacher[:, :steps, :].float()
    mask = (
        torch.arange(steps, device=student.device)[None, :]
        < lengths_steps[:, None].to(student.device)
    ).unsqueeze(-1)
    denom = mask.sum().clamp_min(1) * student.shape[-1]
    mse = (((student - teacher) * mask) ** 2).sum() / denom
    cos = F.cosine_similarity(student, teacher, dim=-1)
    cos_mask = mask.squeeze(-1)
    cos_loss = ((1.0 - cos) * cos_mask).sum() / cos_mask.sum().clamp_min(1)
    loss = mse + cosine_weight * cos_loss
    return loss, {
        "mse": float(mse.detach()),
        "cosine_distance": float(cos_loss.detach()),
    }
