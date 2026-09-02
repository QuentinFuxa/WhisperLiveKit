#!/usr/bin/env python3
"""Re-run the paper's TS calibration ON THE MLX PATH (the decisive A/B).

The shipped head set was calibrated on the PyTorch/transformers path:
teacher-forced prefill, per aligned target token the attention row's
FULL-sequence argmax must land on a gold-aligned source token (TS = hit
rate). Production reads mlx decode steps. This script re-runs the EXACT
calibration semantics on the mlx capture path (same tokenizer, same
prompt template, same gold alignments, mlx prefill attention via
CapturedAttention) and compares the head ranking with the PyTorch result.

If the rankings agree, the mlx capture is faithful and the production
failure is the frontier/policy usage, not the heads. If they disagree,
the phase (prefill vs decode) or the runtime differs and heads must be
re-selected on this path.

Usage:
  .venv/bin/python scripts/calibrate_ts_mlx.py [--pairs 100] [--out /tmp/ts_mlx.json]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

ALIGN_PATH = "/Users/clkao/git/asr/Alignatt4LLM/data/alignatt_heads/word_alignments_zh-en.json"
PROMPT_TEMPLATE = "把下面的文本翻译成{target_lang}，不要额外解释。\n\n{text}"
LANG_NAMES = {"en": "English", "zh": "Chinese (Simplified)"}


def project_char_span_to_token_indices(offsets, start_char, end_char):
    out = []
    for idx, (ts, te) in enumerate(offsets):
        if te <= start_char:
            continue
        if ts >= end_char:
            break
        if ts < end_char and te > start_char:
            out.append(idx)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--out", default="/tmp/ts_mlx.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import whisperlivekit.simul_mt_capture as smc
    tok = AutoTokenizer.from_pretrained("mlx-community/Hy-MT2-1.8B-8bit")

    from mlx_lm import load
    import os
    mdl = os.environ.get("TS_MLX_MODEL", "hy-mt2-1.8b-8bit")
    if os.path.isdir(mdl):
        model, _ = load(mdl)
    else:
        from whisperlivekit.translation_mlx_llm_mt import MlxLlmTranslation
        base = MlxLlmTranslation(model_id=mdl,
                                 source_language="zh", target_language="en", warmup=False)
        model, _ = base._ensure_model(base._config)
    capture = smc.install_capture(model, heads=[(l, 0) for l in range(32)])
    n_layers = len(model.model.layers)

    rows = json.load(open(ALIGN_PATH))[: args.pairs]
    n_heads = model.model.layers[0].self_attn.orig.n_heads
    score_sum = np.zeros((n_layers, n_heads))
    span_sum = np.zeros((n_layers, n_heads))   # argmax lands anywhere in the source span
    used_pairs = 0
    used_tokens = 0

    import mlx.core as mx
    for row in rows:
        src, tgt = row["source_text"], row["target_text"]
        prompt_text = PROMPT_TEMPLATE.format(target_lang=LANG_NAMES["en"], text=src)
        full_text = prompt_text + tgt
        prompt_off = [tuple(map(int, o)) for o in
                      tok(prompt_text, return_offsets_mapping=True)["offset_mapping"]]
        full_off = [tuple(map(int, o)) for o in
                    tok(full_text, return_offsets_mapping=True)["offset_mapping"]]
        src_cs = prompt_text.rfind(src)
        if src_cs < 0:
            continue
        src_ce = src_cs + len(src)
        tgt_cs, tgt_ce = len(prompt_text), len(full_text)
        src_tok_pos = project_char_span_to_token_indices(prompt_off, src_cs, src_ce)
        tgt_tok_pos = project_char_span_to_token_indices(full_off, tgt_cs, tgt_ce)
        if not src_tok_pos or not tgt_tok_pos:
            continue

        # gold: target local idx -> set of GLOBAL source token positions
        gold: dict[int, set] = {}
        for al in row["alignments"]:
            # two annotation schema variants: flat source_start/source_end and
            # nested source_span/target_span pairs
            if "source_start" in al:
                s0, s1 = al["source_start"], al["source_end"]
                t0, t1 = al["target_start"], al["target_end"]
            else:
                s0, s1 = al["source_span"]
                t0, t1 = al["target_span"]
            # words are plain strings here — derive char spans by
            # sequential find (words appear in order in the text)
            def word_char_spans(words, text):
                spans, pos = [], 0
                for w in words:
                    i = text.find(w, pos)
                    if i < 0:
                        return None
                    spans.append((i, i + len(w)))
                    pos = i + len(w)
                return spans
            sspans = word_char_spans(row["source_words"], src)
            tspans = word_char_spans(row["target_words"], tgt)
            # both word-span lists are relative to their own text; shift into
            # the coordinate space of the offsets being projected against
            _scs = prompt_text.rfind(src)
            if _scs < 0:
                continue
            if sspans is not None:
                sspans = [(a + _scs, b + _scs) for a, b in sspans]
            if not sspans or not tspans or s1 > len(sspans) or t1 > len(tspans):
                continue
            sl = project_char_span_to_token_indices(prompt_off, sspans[s0][0], sspans[s1 - 1][1])
            # target word spans are relative to tgt; full_off is relative to
            # full_text — shift by the prompt length
            _poff = len(prompt_text)
            tl = project_char_span_to_token_indices(
                full_off, tspans[t0][0] + _poff, tspans[t1 - 1][1] + _poff)
            if not sl or not tl:
                continue
            gset = set(sl)
            for t in tl:
                gold.setdefault(t, set()).update(gset)
        if not gold:
            continue

        ids = tok.encode(full_text, add_special_tokens=False)
        capture.clear()
        # single teacher-forced prefill forward (no generation)
        arr = mx.array(ids)[None]
        model(arr)
        # prefill attention: the LAST entry per layer has Lq == len(ids)
        valid_tgt = sorted(gold)  # GLOBAL target token positions (indices into full ids)
        if not valid_tgt:
            continue
        used_pairs += 1
        for layer in range(n_layers):
            entries = capture.get(layer) or []
            pre = [e for e in entries if e.ndim == 4 and e.shape[2] == len(ids)]
            if not pre:
                continue
            attn = np.array(pre[-1])[0]  # (H, L, L) first (only) prefill
            s_lo, s_hi = min(src_tok_pos), max(src_tok_pos) + 1
            for gpos in valid_tgt:
                if gpos >= attn.shape[1]:
                    continue
                row_attn = attn[:, gpos, :]           # (H, L)
                amax = row_attn.argmax(-1)            # (H,)
                for h in range(attn.shape[0]):
                    score_sum[layer, h] += int(int(amax[h]) in gold[gpos])
                    span_sum[layer, h] += int(s_lo <= int(amax[h]) < s_hi)
        used_tokens += len(valid_tgt)

    # per-head TS
    denom = max(used_tokens, 1)
    results = {}
    for l in range(n_layers):
        for h in range(n_heads):
            results[f"{l},{h}"] = round(float(score_sum[l, h] / denom), 4)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)

    cal = ["9,5", "13,1", "9,6", "12,11", "14,2", "14,0", "4,12", "1,10"]
    span_results = {}
    for l in range(n_layers):
        for h in range(n_heads):
            span_results[f"{l},{h}"] = round(float(span_sum[l, h] / denom), 4)
    with open(args.out.replace('.json', '_span.json'), 'w') as f:
        json.dump(span_results, f, indent=1)
    ranked_span = sorted(span_results.items(), key=lambda kv: -kv[1])
    print(f"\ntop 15 by SPAN hit (argmax anywhere in source span):")
    for k, v in ranked_span[:15]:
        mark = "*" if k in cal else " "
        print(f"  {k:>7}: span={v:.3f} ts={results[k]:.3f} {mark}")
    ranked = sorted(results.items(), key=lambda kv: -kv[1])
    print(f"pairs used: {used_pairs}, aligned-target-token evals/layer ~ {denom:.0f}")
    print(f"\ntop 25 heads (mlx prefill, paper TS semantics):")
    for k, v in ranked[:25]:
        mark = "*" if k in cal else " "
        print(f"  {k:>7}: TS={v:.3f} {mark}")
    print("\ncalibrated heads on this path:")
    for k in cal:
        print(f"  {k:>7}: TS={results[k]:.3f}")
    ranks = {k: i for i, (k, _) in enumerate(ranked)}
    print("\ncalibrated heads' ranks:", {k: ranks[k] for k in cal})
    print(f"[saved {args.out}]")


if __name__ == "__main__":
    main()
