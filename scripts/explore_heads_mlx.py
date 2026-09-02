#!/usr/bin/env python3
"""Explore ALL hunyuan_v1_dense heads on the production mlx decode path.

The shipped calibration scored heads on the PyTorch prefill path (TS metric).
Production reads DECODE-step attention on the mlx path. This script runs the
real pipeline (zh→en, simul) with capture widened to every layer, records the
per-decode-step attention rows seen at each ``apply_commit_policy`` call,
then scores every (layer, head):

  span_share    mean fraction of the full attention row on the source span
                (paper §4.2: ~9% accessible + ~8% inaccessible + ~81%
                prompt/sink is NORMAL — do not rank by this alone)
  frontier_acc  among steps with a real inaccessible tail (cend < 0.95*n_src),
                fraction whose within-span argmax lands in the committed span
                — the paper's gate (argmax < frontier, τ=0). 1.0 = always
                commits (no discrimination); near 0 = always holds.
  hold_rate     fraction of informative steps where the argmax lands on the
                inaccessible tail (the head must SOMETIMES hold to be useful)
  track         Pearson corr between the argmax position and the committed
                boundary across steps (does the head's alignment track the
                frontier as ASR commits more source?)
  committed_frac  mean within-span committed mass fraction (what the current
                "mass" mode thresholds at 0.5)

Ranking is by a composite of frontier_acc (high but < 1) and hold_rate (> 0),
with track as a tiebreaker. The acceptance test is empirical: re-run the
coverage litmus (scripts/check_simul_heads.py) with LC_SIMUL_HEAD set to the
winner.

Usage:
  .venv/bin/python scripts/explore_heads_mlx.py --audio <zh wav> [--out /tmp/head_explore.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict

import numpy as np

RECORDS: list = []  # per policy call: dict(layer, span(H,S) rows, cend, n_src)


def install_hooks() -> None:
    import whisperlivekit.simul_mt_capture as smc
    import whisperlivekit.translation_mlx_llm_mt_simul as tms

    orig_install = smc.install_capture
    orig_policy = smc.apply_commit_policy

    def all_layer_install(model, heads=None):
        n_layers = len(model.model.layers)
        print(f"[explore] capturing ALL {n_layers} layers", file=sys.stderr)
        return orig_install(model, heads=[(l, 0) for l in range(n_layers)])

    def recording_policy(capture, top_head, n_tokens, src_start, src_end, cend,
                         mode="argmax", mass_threshold=0.5):
        per_layer = {}
        for layer, steps in capture.items():
            dec = []
            for s in steps:
                a = np.array(s)
                if a.ndim == 4 and a.shape[2] == 1:
                    dec.append(a)
            dec = dec[:n_tokens]
            if not dec:
                continue
            # decode steps have GROWING key length (Lk = prompt_len + i), so
            # extract the source-span slice per step instead of stacking rows
            spans, totals = [], []
            for a in dec:
                row = a[0, :, 0, :]                        # (H, Lk)
                spans.append(row[:, src_start:src_end])    # (H, S)
                totals.append(row.sum(-1))                 # (H,)
            per_layer[int(layer)] = dict(
                span=np.stack(spans).astype(np.float32),   # (T, H, S)
                total=np.stack(totals).astype(np.float32), # (T, H)
            )
        RECORDS.append(dict(layers=per_layer, cend=int(cend),
                            src_start=int(src_start), src_end=int(src_end)))
        return orig_policy(capture, top_head, n_tokens, src_start, src_end, cend,
                           mode=mode, mass_threshold=mass_threshold)

    # patch the names translation_mlx_llm_mt_simul imported by reference
    tms.install_capture = all_layer_install
    tms.apply_commit_policy = recording_policy
    smc.apply_commit_policy = orig_policy  # the recorder delegates to the real one


def score() -> dict:
    per_head: dict = {}
    for rec in RECORDS:
        cend, s0, s1 = rec["cend"], rec["src_start"], rec["src_end"]
        n_src = s1 - s0
        if n_src <= 0:
            continue
        # informative: there is a real inaccessible tail, so the gate decides
        informative = cend < int(0.95 * n_src)
        for layer, d in rec["layers"].items():
            span = d["span"]                       # (T, H, S)
            total = d["total"]                     # (T, H)
            span_mass = span.sum(-1)               # (T, H)
            committed = span[:, :, :cend].sum(-1) if cend else np.zeros_like(span_mass)
            amax = span.argmax(-1)                 # (T, H)
            for h in range(span_mass.shape[1]):
                key = (int(layer), int(h))
                e = per_head.setdefault(key, dict(span=[], acc_n=0, acc_hits=0,
                                                  hold_hits=0, track_x=[], track_y=[]))
                e["span"].extend((span_mass[:, h] / np.maximum(total[:, h], 1e-9)).tolist())
                if informative:
                    e["acc_n"] += span_mass.shape[0]
                    e["acc_hits"] += int((amax[:, h] < cend).sum())
                e["track_x"].extend([float(cend) / n_src] * span_mass.shape[0])
                e["track_y"].extend((amax[:, h] / n_src).tolist())

    out = {}
    for (layer, head), e in per_head.items():
        n_acc = e["acc_n"]
        acc = e["acc_hits"] / n_acc if n_acc else float("nan")
        hold = 1.0 - acc if n_acc else 0.0
        x, y = np.array(e["track_x"]), np.array(e["track_y"])
        if len(x) > 2 and x.std() > 0 and y.std() > 0:
            track = float(np.corrcoef(x, y)[0, 1])
        else:
            track = float("nan")
        out[f"{layer},{head}"] = dict(
            layer=layer, head=head,
            span_share=round(float(np.mean(e["span"])), 4),
            frontier_acc=round(acc, 3),
            hold_rate=round(hold, 3),
            track=round(track, 3),
            steps=len(e["span"]),
            informative_steps=n_acc,
        )
    return out


async def run(audio: str, event_log: str | None, language: str = "zh",
              target_language: str = "en") -> None:
    sys.path.insert(0, "scripts")
    install_hooks()
    from lc_terminal import run_file, _make_engine_kwargs
    import lc_terminal as lct

    ns = argparse.Namespace(
        audio=audio, language=language, target_language=target_language,
        backend="mlx-qwen3-asr",
        mlx_qwen3_asr_model="mlx-community/Qwen3-ASR-0.6B-8bit", mlx_llm_mt_model="hy-mt2-1.8b-8bit",
        source="file", simultaneous=True, simul_commit=None, second_pass=False,
        diarize=False, hotwords=None, vad_threshold=None, vad_min_silence_ms=None,
        event_log=event_log, overlay=False, tui=False,
    )
    class _Sink:
        def __call__(self, state): pass
    await run_file(ns, _Sink())



def dump_calls(path: str = "/tmp/head_calls.json") -> None:
    """Per-call frontier state: cend vs n_src — diagnostic for commit cadence."""
    calls = [dict(cend=r["cend"], n_src=r["src_end"] - r["src_start"]) for r in RECORDS]
    with open(path, "w") as f:
        json.dump(calls, f, indent=1)
    print(f"[calls] {len(calls)} calls -> {path}", file=sys.stderr)
    for i, c in enumerate(calls):
        print(f"  call {i:>3}: cend={c['cend']:>3} n_src={c['n_src']:>3} "
              f"{'INFORMATIVE' if c['cend'] < 0.95*c['n_src'] else 'fully committed'}",
              file=sys.stderr)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="/Users/clkao/git/asr/_work/zh_long.wav")
    ap.add_argument("--out", default="/tmp/head_explore.json")
    ap.add_argument("--event-log", default=None)
    ap.add_argument("--language", default="zh")
    ap.add_argument("--target-language", default="en")
    args = ap.parse_args()

    asyncio.run(run(args.audio, args.event_log, args.language, args.target_language))

    scores = score()
    dump_calls()
    with open(args.out, "w") as f:
        json.dump(scores, f, indent=1)
    rows = sorted(scores.values(), key=lambda r: -(
        (r["frontier_acc"] if r["frontier_acc"] == r["frontier_acc"] else 0)
        * (r["hold_rate"] if 0 < r["hold_rate"] else 1e-6)
    ))
    calibrated = {(9, 5), (13, 1), (9, 6), (12, 11), (14, 2), (14, 0), (4, 12), (1, 10)}
    print(f"\n{'head':>8} {'span':>6} {'f_acc':>6} {'hold':>6} {'track':>6} {'steps':>6}  cal")
    for r in rows[:25]:
        mark = "*" if (r["layer"], r["head"]) in calibrated else " "
        print(f"{r['layer']:>4},{r['head']:<3} {r['span_share']:>6.3f} {r['frontier_acc']:>6.2f} "
              f"{r['hold_rate']:>6.2f} {r['track']:>6.2f} {r['informative_steps']:>6}  {mark}")
    print("\n* = shipped calibration head.  f_acc: frontier gate accuracy on informative steps")
    print("(1.0 = always commits, 0 = always holds; useful heads sit high with hold_rate > 0)")
    print(f"[saved {args.out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
