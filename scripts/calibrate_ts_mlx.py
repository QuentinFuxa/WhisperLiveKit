#!/usr/bin/env python3
"""Score MLX alignment heads from AlignAtt4LLM word-alignment JSON.

Uses the runtime chat prompt and records the exact model revision and
quantization. Three seeded half-sample checks determine stability. A failed
gate is retained in the report and exits unsuccessfully; it is never promoted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np


def word_spans(words, text):
    spans, position = [], 0
    for word in words:
        start = text.find(word, position)
        if start < 0:
            raise ValueError(f"Cannot align word {word!r} to its text")
        position = start + len(word)
        spans.append((start, position))
    return spans


def token_indices(offsets, start, end):
    return [i for i, (left, right) in enumerate(offsets) if left < end and right > start]


def gold_positions(row, prompt, tokenizer):
    source, target = row["source_text"], row["target_text"]
    full = prompt + target
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    source_start = prompt.rfind(source)
    if source_start < 0:
        raise ValueError("Source text is absent from the runtime prompt")
    source_words = word_spans(row["source_words"], source)
    target_words = word_spans(row["target_words"], target)
    gold = {}
    for alignment in row["alignments"]:
        if "source_span" in alignment:
            s0, s1 = alignment["source_span"]
            t0, t1 = alignment["target_span"]
        else:
            s0, s1 = alignment["source_start"], alignment["source_end"]
            t0, t1 = alignment["target_start"], alignment["target_end"]
        if not 0 <= s0 < s1 <= len(source_words) or not 0 <= t0 < t1 <= len(target_words):
            raise ValueError("Word alignment span is out of range")
        source_ids = token_indices(offsets, source_start + source_words[s0][0], source_start + source_words[s1 - 1][1])
        target_ids = token_indices(offsets, len(prompt) + target_words[t0][0], len(prompt) + target_words[t1 - 1][1])
        for target_id in target_ids:
            gold.setdefault(target_id, set()).update(source_ids)
    return encoded["input_ids"], {key: values for key, values in gold.items() if values}


def ranked_heads(scores, count):
    return sorted((dict(layer=l, head=h, ts=round(float(scores[l, h]), 6), count=count)
                   for l in range(scores.shape[0]) for h in range(scores.shape[1])),
                  key=lambda head: (-head["ts"], head["layer"], head["head"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="hy-mt2-1.8b-8bit")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--direction", default="en-zh")
    parser.add_argument("--alignments", type=Path, required=True)
    parser.add_argument("--source-url", required=True, help="Provenance URL for the alignment file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=0, help="0 uses every pair")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    import mlx.core as mx
    from huggingface_hub import hf_hub_download

    from whisperlivekit.simul_mt_capture import capture_attention, snapshot_capture
    from whisperlivekit.translation_mlx_llm_mt import MlxLlmTranslation

    source_language, target_language = args.direction.split("-", 1)
    engine = MlxLlmTranslation(args.model_id, target_language, source_language, warmup=False)
    engine._config = replace(engine._config, revision=args.revision)
    model, tokenizer = engine._ensure_model(engine._config)
    config_path = hf_hub_download(engine._config.repo, "config.json", revision=args.revision, local_files_only=True)
    config = json.loads(Path(config_path).read_text())
    layers = len(model.model.layers)
    n_heads = model.model.layers[0].self_attn.n_heads
    heads = [(layer, head) for layer in range(layers) for head in range(n_heads)]
    source_bytes = args.alignments.read_bytes()
    rows = json.loads(source_bytes)
    if args.pairs:
        rows = rows[:args.pairs]
    pair_scores, used_tokens, failures = [], 0, []
    for index, row in enumerate(rows):
        try:
            if row.get("direction") != args.direction:
                raise ValueError("Alignment direction differs from the requested direction")
            content = engine._prompt["template"].format(target_lang=engine._prompt["target_name"], text=row["source_text"])
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                                  add_generation_prompt=True, tokenize=False)
            ids, gold = gold_positions(row, prompt, tokenizer._tokenizer)
            if not gold:
                raise ValueError("No aligned target tokens")
        except ValueError as exc:
            failures.append({"index": index, "pair_id": row.get("pair_id"),
                             "kind": "invalid_annotation", "error": str(exc)})
            continue
        # Model/capture errors abort the run; malformed annotations are retained
        # separately so their exclusion can be inspected alongside the scores.
        try:
            with engine._MODEL_LOCK, capture_attention(model, heads, target_start=min(gold)) as capture:
                mx.eval(model(mx.array([ids])))
                snapshot = snapshot_capture(capture)
            scores = np.zeros((layers, n_heads))
            for head, entries in snapshot.items():
                entry = entries[0]
                argmax = entry.weights.argmax(axis=-1)
                scores[head] = sum(int(argmax[position - entry.query_start]) in source_ids
                                   for position, source_ids in gold.items()) / len(gold)
            pair_scores.append(scores)
            used_tokens += len(gold)
        except Exception as exc:
            raise RuntimeError(f"Calibration inference failed on pair {index}") from exc
        if (index + 1) % 25 == 0 or index == 0:
            print(f"{index + 1}/{len(rows)} pairs; {len(pair_scores)} scored; {len(failures)} failed", flush=True)
    if not pair_scores:
        raise RuntimeError("No usable calibration pairs")
    matrix = np.array(pair_scores)
    mean = matrix.mean(axis=0)  # same per-pair averaging as AlignAtt4LLM
    ranked = ranked_heads(mean, len(matrix))
    selected = ranked[:8]
    full_heads = {(head["layer"], head["head"]) for head in selected}
    rng, checks = random.Random(args.seed), []
    for split in range(3):
        subset = rng.sample(range(len(matrix)), max(1, len(matrix) // 2))
        split_mean = matrix[subset].mean(axis=0)
        split_ranked = ranked_heads(split_mean, len(subset))[:8]
        overlap = full_heads & {(h["layer"], h["head"]) for h in split_ranked}
        delta = max(abs(float(split_mean[head] - mean[head])) for head in full_heads)
        checks.append(dict(split_index=split, sample_size=len(subset), max_abs_ts_delta_vs_full=delta,
                           top_heads=split_ranked, stable_vs_full=len(overlap) == 8 and delta <= 0.03))
    eligible = len(matrix) >= 100 and all(head["ts"] > 0.1 for head in selected) and all(c["stable_vs_full"] for c in checks)
    report = dict(
        model=engine._config.repo, direction=args.direction, datasets=[args.source_url],
        num_layers=layers, num_heads=n_heads, attempted_pairs=len(rows), used_pairs=len(matrix), used_target_tokens=used_tokens,
        paper_threshold=0.1, score_name="paper_translation_score_argmax",
        token_alignment_heads=[head for head in ranked if head["ts"] > 0.1],
        all_heads_ranked=ranked, ts_matrix=mean.tolist(), stability_checks=checks,
        failures=failures, promotion_gate={"eligible_for_promotion": eligible, "minimum_pairs": 100},
        runtime=dict(backend="mlx-lm", model_revision=args.revision,
                     quantization=config.get("quantization", config.get("quantization_config", {})),
                     source_sha256=hashlib.sha256(source_bytes).hexdigest(), source_url=args.source_url,
                     prompt=engine._prompt, seed=args.seed,
                     versions={name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "transformers")}),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "eligible": eligible, "top_heads": selected}), flush=True)
    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
