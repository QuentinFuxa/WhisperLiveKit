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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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

# Language codes that count as "Chinese" for registry key normalization.
_ZH_CODES = {"zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant", "cmn", "yue"}


@dataclass
class CalibrationEntry:
    """A calibrated (model, source, target) tuple's alignment heads.

    ``heads`` is the list of (layer, head) indices to capture.
    ``ts_scores`` maps each head to its alignment-stream score.
    ``top_head`` is the primary commit-signal head.
    ``disabled_quants`` lists quant suffixes (e.g. ``"4bit"``) that share
    the model id but whose attention patterns diverge too far from the
    calibrated weights to use these heads; a repo matching one of them
    silently deactivates (returns ``None`` from ``lookup_calibration``).
    """
    heads: List[Tuple[int, int]]
    ts_scores: Dict[Tuple[int, int], float]
    top_head: Tuple[int, int]
    disabled_quants: set = field(default_factory=set)


# Per-(model_id, source_lang, target_lang) calibration registry. The key is
# a NORMALIZED model id (org prefix and quant suffix stripped) so the same
# calibration entry is reusable across implementations (MLX, vLLM/CUDA) and
# quantizations. Only tuples that have passed the AlignAtt4LLM promotion gate
# (TS > 0.1 for >=8 heads, stability, eligible_for_promotion) are seeded here.
# The simultaneous variant looks up its (repo, src, tgt) at init; a missing
# tuple triggers silent deactivation (translate-on-close, no provisional).
#
# The 8bit entry is calibrated on tencent/Hy-MT2-1.8B (bf16 base model);
# the heads transfer to the 8bit MLX quantization. The 4bit quantization
# was probed (48.9% argmax match vs 8bit — attention patterns differ too
# much) and the formal promotion gate could not be run (AlignAtt4LLM
# requires PyTorch/transformers, which can't load MLX-format repos), so
# 4bit is in ``disabled_quants`` — it silently deactivates (translation still
# works via the base class).
#
# TODO: refactor heads to load from external JSON files (the AlignAtt4LLM
# ``translation_heads_<model>_<direction>.json`` pattern) instead of being
# hardcoded in Python, so calibration entries are shareable across
# implementations (MLX, vLLM, etc.) without code changes.
CALIBRATION_REGISTRY: Dict[Tuple[str, str, str], CalibrationEntry] = {
    ("hy-mt2-1.8b", "zh", "en"): CalibrationEntry(
        heads=ALIGNMENT_HEADS,
        ts_scores=HEAD_TS_SCORES,
        top_head=TOP_HEAD,
        disabled_quants={"4bit"},
    ),
}


def _normalize_lang(lang: str) -> str:
    """Normalize a language code for registry lookup. Chinese variants
    collapse to ``zh`` (matching ``resolve_prompt``'s normalization)."""
    lang = (lang or "").strip().lower()
    if lang in _ZH_CODES:
        return "zh"
    return lang


# Recognized MLX quant suffixes, longest first so '-4bit' doesn't shadow
# a hypothetical '-4bit-grouped' (none currently exist, but this is robust).
_QUANT_SUFFIXES = ("-8bit", "-4bit", "-bf16", "-f16")


def _normalize_model_id(repo: str) -> str:
    """Normalize a model repo to a canonical model id: strip the org prefix
    (everything up to and including the last ``/``) and the quant suffix,
    then lowercase. Calibration entries are keyed by model id only, so the
    same architecture across quants (8bit/4bit/bf16) and implementations
    (MLX ``mlx-community/…``, vLLM ``tencent/…``) shares a single entry.
    """
    name = (repo or "").rsplit("/", 1)[-1].lower()
    for suffix in _QUANT_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _extract_quant(repo: str) -> str:
    """Extract the quant suffix from a model repo (e.g. ``'4bit'`` from
    ``'mlx-community/Hy-MT2-1.8B-4bit'``). Returns ``''`` if no known suffix."""
    name = (repo or "").rsplit("/", 1)[-1].lower()
    for suffix in _QUANT_SUFFIXES:
        if name.endswith(suffix):
            return suffix[1:]  # strip leading '-'
    return ""


def lookup_calibration(
    model_repo: str, source_lang: str, target_lang: str
) -> Optional[CalibrationEntry]:
    """Look up the calibration entry for a (model_repo, source, target) tuple.

    The repo is normalized to a model id (org prefix + quant suffix stripped)
    before lookup, so all quants of the same architecture share one entry.
    If the model id matches but the repo's quant is in the entry's
    ``disabled_quants``, returns ``None`` (silently deactivates).

    Returns the ``CalibrationEntry`` if the tuple is calibrated and not
    quant-disabled, or ``None`` if it is not (the caller should silently
    deactivate in that case).
    """
    key = (
        _normalize_model_id(model_repo),
        _normalize_lang(source_lang),
        _normalize_lang(target_lang),
    )
    entry = CALIBRATION_REGISTRY.get(key)
    if entry is None:
        return None
    if _extract_quant(model_repo) in entry.disabled_quants:
        return None
    return entry


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
