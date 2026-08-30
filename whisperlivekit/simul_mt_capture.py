"""Scoped Hunyuan attention capture for AlignAtt on mlx-lm.

Output uses native fused attention. Only selected heads are materialized for
alignment. Callers hold the shared model lock throughout the capture context.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class AttentionRows:
    query_start: int
    weights: object


class CapturedAttention:
    def __init__(self, original, layer, heads, capture, target_start):
        self.original = original
        self.layer = layer
        self.heads = heads
        self.capture = capture
        self.target_start = target_start

    def __call__(self, x, mask=None, cache=None):
        import mlx.core as mx
        from mlx_lm.models.base import scaled_dot_product_attention

        a = self.original
        batch, length, _ = x.shape
        offset = int(cache.offset) if cache is not None else 0
        q = a.q_proj(x).reshape(batch, length, a.n_heads, a.head_dim).transpose(0, 2, 1, 3)
        k = a.k_proj(x).reshape(batch, length, a.n_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
        v = a.v_proj(x).reshape(batch, length, a.n_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
        q, k = a.rope(q, offset=offset), a.rope(k, offset=offset)
        if a.use_qk_norm:
            q, k = a.query_layernorm(q), a.key_layernorm(k)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        output = scaled_dot_product_attention(q, k, v, cache=cache, scale=a.scale, mask=mask)
        first = max(0, self.target_start - offset)
        if first < length:
            if batch != 1 or not isinstance(k, mx.array):
                raise ValueError("Alignment capture requires one sequence and an unquantized KV cache")
            keys = k[0, [h // (a.n_heads // a.n_kv_heads) for h in self.heads]]
            queries = q[0, list(self.heads), first:, :]
            scores = (queries @ keys.transpose(0, 2, 1)) * a.scale
            if isinstance(mask, str):
                if mask != "causal":
                    raise ValueError(f"Unsupported attention mask: {mask}")
                visible = mx.arange(offset + first, offset + length)[:, None] >= mx.arange(k.shape[-2])
                scores = mx.where(visible, scores, -float("inf"))
            elif mask is not None:
                selected_mask = mask[..., first:, :]
                if selected_mask.dtype == mx.bool_:
                    scores = mx.where(selected_mask, scores, -float("inf"))
                else:
                    scores = scores + selected_mask
            weights = mx.softmax(scores, axis=-1, precise=True).astype(mx.float32)
            for index, head in enumerate(self.heads):
                self.capture.setdefault((self.layer, head), []).append(
                    AttentionRows(offset + first, weights[index])
                )
        return a.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


@contextmanager
def capture_attention(model, heads, *, target_start=0):
    """Install for one generation and always restore the original layers."""
    import mlx.nn as nn
    from mlx_lm.models.hunyuan_v1_dense import Attention

    class Wrapper(CapturedAttention, nn.Module):
        def __init__(self, *args):
            nn.Module.__init__(self)
            CapturedAttention.__init__(self, *args)

    layers, selected = model.model.layers, {}
    for layer, head in heads:
        if not 0 <= layer < len(layers):
            raise ValueError(f"Calibration layer out of range: {layer}")
        original = layers[layer].self_attn
        if not isinstance(original, Attention):
            raise ValueError("MLX alignment capture currently supports hunyuan_v1_dense only")
        if not 0 <= head < original.n_heads:
            raise ValueError(f"Calibration head out of range: {(layer, head)}")
        selected.setdefault(layer, set()).add(head)
    if not selected:
        raise ValueError("Alignment capture requires calibrated heads")
    capture, originals = {}, {}
    try:
        for layer, chosen in selected.items():
            originals[layer] = layers[layer].self_attn
            layers[layer].self_attn = Wrapper(originals[layer], layer, tuple(sorted(chosen)), capture, target_start)
        yield capture
    finally:
        for layer, original in originals.items():
            layers[layer].self_attn = original


def snapshot_capture(capture):
    """Detach a draft's attention from the model and from later sessions."""
    import numpy as np

    snapshot = {}
    for head, entries in capture.items():
        rows = []
        for entry in entries:
            weights = np.array(entry.weights, dtype=np.float32, copy=True)
            weights.flags.writeable = False
            rows.append(AttentionRows(entry.query_start, weights))
        snapshot[head] = tuple(rows)
    return MappingProxyType(snapshot)


def source_span(tokenizer, prompt, source):
    """Locate source tokens exactly; an approximate span can commit wrong text."""
    source_ids = tokenizer.encode(source, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if source_ids:
        for start in range(len(prompt_ids) - len(source_ids), -1, -1):
            if prompt_ids[start:start + len(source_ids)] == source_ids:
                return start, start + len(source_ids)
    raise ValueError("Cannot locate the source token span in the translation prompt")


def committed_src_end_from_text(tokenizer, source_ids, committed_text):
    end = 0
    for count in range(1, len(source_ids) + 1):
        decoded = tokenizer.decode(source_ids[:count])
        if decoded.endswith("\ufffd"):
            continue
        if not decoded or not committed_text.startswith(decoded):
            break
        end = count
    return end


def apply_commit_policy(capture, heads, n_tokens, prompt_length, src_start, src_end,
                        committed_src_end, mode="paper", mass_threshold=0.5):
    """Return a contiguous target prefix backed by captured target-query rows."""
    import numpy as np

    if mode not in {"paper", "argmax", "mass"}:
        raise ValueError(f"Unknown alignment policy: {mode}")
    if not 0 < mass_threshold <= 1 or not 0 <= committed_src_end <= src_end - src_start:
        raise ValueError("Invalid alignment frontier or mass threshold")
    if n_tokens == 0:
        return 0
    if not heads or any(not capture.get(head) for head in heads):
        raise RuntimeError("Missing alignment attention; translation draft was not committed")
    per_head = []
    for head in heads:
        rows = {}
        for entry in capture[head]:
            for index, row in enumerate(np.asarray(entry.weights)):
                rows[entry.query_start + index] = row[src_start:src_end]
        per_head.append(rows)
    available = 0
    while available < n_tokens and all(prompt_length + available in rows for rows in per_head):
        available += 1
    if not available:
        raise RuntimeError("No target-query attention captured for translation draft")
    values = np.array([[rows[prompt_length + i] for rows in per_head] for i in range(available)])
    if values.shape[-1] != src_end - src_start or not values.shape[-1] or not np.isfinite(values).all():
        raise RuntimeError("Invalid source attention in translation draft")
    if committed_src_end == 0:
        return 0
    if mode == "paper":
        accepts = _paper_stabilized_argmax(values) < committed_src_end + 1
    elif mode == "argmax":
        accepts = values[:, 0].argmax(axis=-1) < committed_src_end
    else:
        total = values[:, 0].sum(axis=-1)
        accepts = values[:, 0, :committed_src_end].sum(axis=-1) / np.maximum(total, 1e-12) >= mass_threshold
    accepts &= np.all(values.sum(axis=-1) > 0, axis=-1)
    held = np.flatnonzero(~accepts)
    return int(held[0]) if held.size else available


def _paper_stabilized_argmax(span_rows):
    """Stabilized source argmax per draft token (paper §4.4 branch B).

    ``span_rows``: (T, H, S) decode-step attention over the source span for
    the calibrated heads. Per head, z-score each source position with
    prefix-online (Welford) moments accumulated across the draft's decode
    steps; average the z-scored rows over heads; apply a width-7 median
    filter along the source axis. Returns (T, S) filtered scores; the
    caller takes the argmax.
    """
    import numpy as np
    T, H, S = span_rows.shape
    mean = np.zeros((H, S), dtype=np.float64)
    m2 = np.zeros((H, S), dtype=np.float64)
    z = np.zeros((T, S), dtype=np.float64)
    for t in range(T):
        x = span_rows[t].astype(np.float64)          # (H, S)
        if t:
            std = np.sqrt(m2 / t)
            zs = np.where(std > 1e-12, (x - mean) / np.where(std > 1e-12, std, 1.0), 0.0)
        else:
            zs = np.zeros((H, S), dtype=np.float64)
        z[t] = zs.mean(axis=0)                       # average over heads
        # Welford update AFTER scoring: prefix-online uses steps < t
        delta = x - mean
        mean += delta / (t + 1)
        m2 += delta * (x - mean)
    # width-7 median filter along the source axis (edge-clamped windows)
    half = 3
    zf = np.empty_like(z)
    for s in range(S):
        lo, hi = max(0, s - half), min(S, s + half + 1)
        zf[:, s] = np.median(z[:, lo:hi], axis=1)
    return zf.argmax(axis=1)
