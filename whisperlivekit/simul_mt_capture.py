"""Simultaneous-MT capture layer: MLX Q/K attention capture + AlignAtt commit policy.

mlx-lm's ``hunyuan_v1_dense.Attention`` fuses Q/K into
``scaled_dot_product_attention`` and discards the attention weights; this
wrapper replicates the forward with a manual ``softmax(QK^T)`` so the
alignment-head attention is capturable. The forward is bit-identical to the
original (only attention storage is added).

The 8 production head indices + TS scores come from a calibration run on
tencent/Hy-MT2-1.8B (zh→en), hardcoded here so the simultaneous variant is
self-contained.

Load-bearing details (learned during development):
  1. ``create_attention_mask`` returns the string ``"causal"``, not an
     array — the manual forward must build the additive causal mask itself.
  2. ``__call__`` dispatch is on the TYPE, not the instance — must use a
     wrapper ``nn.Module`` whose class defines ``__call__``, not an
     instance ``__call__`` assignment.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn

# 8 calibrated production head indices (layer, head) for tencent/Hy-MT2-1.8B
# zh→en. Top head (L9, H5, TS=0.79) is the primary alignment signal; the
# others are cross-check / stability heads. Hardcoded from the calibration
# run so the variant is self-contained.
ALIGNMENT_HEADS: List[Tuple[int, int]] = [
    (9, 5), (13, 1), (9, 6), (12, 11), (14, 2), (14, 0), (4, 12), (1, 10),
]
TOP_HEAD: Tuple[int, int] = (9, 5)

# TS (alignment-stream) scores for each calibrated head. Used for TS-weighted
# multi-head voting. Keys are (layer, head) tuples matching ALIGNMENT_HEADS.
HEAD_TS_SCORES: Dict[Tuple[int, int], float] = {
    (9, 5): 0.7942,
    (13, 1): 0.7270,
    (9, 6): 0.5638,
    (12, 11): 0.5507,
    (14, 2): 0.5070,
    (14, 0): 0.4380,
    (4, 12): 0.3587,
    (1, 10): 0.3231,
}


class CapturedAttention(nn.Module):
    """Wraps ``hunyuan_v1_dense.Attention``; replicates the forward with a
    manual ``softmax(QK^T)`` so attention weights are capturable for the
    alignment heads.

    Only stores attention for selected layers (the layers containing
    calibrated heads); other layers compute attention manually but discard
    it (small overhead). Shares the original projections/norms/rope — no
    weight duplication.
    """

    def __init__(self, orig, layer_idx, capture, selected_layers):
        super().__init__()
        self.orig = orig
        self.layer_idx = layer_idx
        self.capture = capture
        self.selected_layers = selected_layers

    def __call__(self, x, mask=None, cache=None):
        a = self.orig
        B, L, D = x.shape
        q, k, v = a.q_proj(x), a.k_proj(x), a.v_proj(x)
        q = q.reshape(B, L, a.n_heads, a.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, a.n_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, a.n_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
        if cache is not None:
            q = a.rope(q, offset=cache.offset)
            k = a.rope(k, offset=cache.offset)
        else:
            q = a.rope(q)
            k = a.rope(k)
        if a.use_qk_norm:
            q = a.query_layernorm(q)
            k = a.key_layernorm(k)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        # GQA: repeat KV heads to match Q heads
        if a.n_kv_heads != a.n_heads:
            rep = a.n_heads // a.n_kv_heads
            k = mx.repeat(k, rep, axis=1)
            v = mx.repeat(v, rep, axis=1)
        scores = (q @ k.transpose(0, 1, 3, 2)) * a.scale
        if mask is not None:
            if isinstance(mask, str):  # "causal" sentinel from create_attention_mask
                Lq, Lk = scores.shape[-2], scores.shape[-1]
                idx = mx.arange(Lq)[:, None]
                scores = scores + mx.where(
                    idx >= mx.arange(Lk)[None, :], 0.0, -float("inf")
                )
            else:
                scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.layer_idx in self.selected_layers:
            self.capture.setdefault(self.layer_idx, []).append(
                attn.astype(mx.float32)
            )
        return a.o_proj(out)


def install_capture(model, heads=ALIGNMENT_HEADS):
    """Patch a loaded mlx-lm hunyuan_v1_dense model to capture attention for
    the selected head layers.

    Idempotent: if a layer's ``self_attn`` is already a ``CapturedAttention``
    (e.g. the shared cached model was patched by another simul instance), it
    is not re-wrapped.

    Returns a ``dict[layer_idx -> list of attn tensors]`` that fills on each
    forward pass; clear it between runs.
    """
    selected_layers = {h[0] for h in heads}
    capture: Dict[int, list] = {}
    for i, block in enumerate(model.model.layers):
        if isinstance(block.self_attn, CapturedAttention):
            # Already patched (shared model); point at the existing capture dict.
            capture = block.self_attn.capture
            continue
        block.self_attn = CapturedAttention(
            block.self_attn, i, capture, selected_layers
        )
    return capture


def source_span(tok, prompt_str, source_text):
    """Find the ``[start, end)`` token positions of ``source_text`` within
    ``prompt_str``. The source sits as a contiguous block (BPE-merged) in the
    user content; locate it by encoding source alone and matching the id
    subsequence from the end."""
    try:
        src_ids = tok.encode(source_text, add_special_tokens=False)
        pids = tok.encode(prompt_str, add_special_tokens=False)
    except TypeError:
        src_ids = tok.encode(source_text)
        pids = tok.encode(prompt_str)
    n = len(src_ids)
    for i in range(len(pids) - n, -1, -1):
        if pids[i : i + n] == src_ids:
            return i, i + n
    return len(pids) - n - 4, len(pids) - 4


def committed_src_end_from_text(tok, src_ids, committed_text):
    """Map the committed source-text prefix to a source-token boundary.

    Decodes increasing source-token prefixes and keeps the longest whose
    decoded text is a prefix of ``committed_text`` (rounds DOWN to the last
    complete BPE token — a partial BPE token at the boundary is held,
    conservative). Returns a source-token count (0-indexed boundary within the
    source span)."""
    if not committed_text:
        return 0
    cend = 0
    for k in range(1, len(src_ids) + 1):
        decoded = tok.decode(src_ids[:k])
        if len(decoded) <= len(committed_text) and committed_text.startswith(decoded):
            cend = k
        else:
            break
    return cend


def apply_commit_policy(
    capture: Dict[int, list],
    top_head: Tuple[int, int],
    n_tokens: int,
    src_start: int,
    src_end: int,
    committed_src_end: int,
) -> int:
    """Apply the AlignAtt commit policy with the top alignment head.

    For each generated target token (decode step), check if the top head's
    attention argmax over the source span lands on a source token index <
    ``committed_src_end`` (i.e. ASR has committed that source token). The
    committable prefix is contiguous: the first HOLD stops it.

    Returns the number of committed target tokens (a contiguous prefix
    length). If no attention was captured for the top head's layer, all
    tokens are committed (degenerates to no-hold).
    """
    import numpy as np

    layer, head = top_head
    if layer not in capture:
        return n_tokens
    # Decode-step attentions only: shape (H, 1, Lk). Prefill steps have
    # shape (H, Lq, Lk) with Lq > 1 — filter them out.
    steps = [s for s in capture[layer] if np.array(s).shape[2] == 1]
    steps = steps[:n_tokens]
    committed_len = 0
    for i, s in enumerate(steps):
        a = np.array(s)[0]  # (H, 1, Lk)
        src_attn = a[head, 0, src_start:src_end]  # (n_src,)
        amax = int(np.argmax(src_attn))
        if amax < committed_src_end:
            committed_len = i + 1
        else:
            break
    return committed_len
