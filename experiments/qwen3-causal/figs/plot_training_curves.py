#!/usr/bin/env python3
"""Training curves for the causal tower (gate WER + distillation loss).

Reads runs/jl_d1_smoke, runs/jl_d1_full_en and runs/jl_ws2_mix_pos_p2b
history.json files; WS2 part-1 points (no local history) come from the
RUNS.md table. Output: figs/training_curves.svg (also used by the HF card
at qfuxa/qwen3-asr-0.6b-streaming).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent / "runs"
OUT = Path(__file__).resolve().parent / "training_curves.svg"

smoke = json.load(open(BASE / "jl_d1_smoke/history.json"))
d1 = json.load(open(BASE / "jl_d1_full_en/history.json"))
ws2 = json.load(open(BASE / "jl_ws2_mix_pos_p2b/history.json"))

# Cumulative axis: smoke [0,3k] -> D1 [3k,63k] -> WS2 [63k,123k]
# (WS2 p2b local steps 0..40k = WS2 cumulative 20k..60k; part-1 points from RUNS.md).
sx = [e["step"] for e in smoke]
sy = [e["gate_wer"] for e in smoke]
dx = [3000 + e["step"] for e in d1]
dy = [e["gate_wer"] for e in d1]
ws2_pts = [(0, 0.2492, 0.2651, 0.8828), (20000, 0.2514, 0.2408, 0.3681)]
for e in ws2:
    g = e["gate_wers"]
    ws2_pts.append((20000 + e["step"], g["960"], g["1920"], g["960@off4000"]))
ws2_pts = sorted(set(ws2_pts))
wx = [63000 + p[0] for p in ws2_pts]
w960 = [p[1] for p in ws2_pts]
w1920 = [p[2] for p in ws2_pts]
woff = [p[3] for p in ws2_pts]

loss_x = [3000 + e["step"] for e in d1 if "loss" in e] + [
    83000 + e["step"] for e in ws2 if "loss" in e
]
loss_y = [e["loss"] for e in d1 if "loss" in e] + [
    e["loss"] for e in ws2 if "loss" in e
]

GRAY = "#9a9a9a"
ACCENT, ACCENT2, ACCENT3 = "#0a66a3", "#c2542e", "#6b9e3f"
fig, ax = plt.subplots(figsize=(8.6, 3.8), constrained_layout=True)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e6e6e6", lw=0.8)
ax.set_axisbelow(True)

for x0 in (3000, 63000):
    ax.axvline(x0, color="#d0d0d0", lw=0.9, ls=(0, (4, 3)))
ax.text(1500, 0.97, "phase 1\n100 h", ha="center", va="top", fontsize=8, color=GRAY)
ax.text(
    33000, 0.97, "phase 2 — LibriSpeech 960 h, 0.96 s blocks",
    ha="center", va="top", fontsize=8, color=GRAY,
)
ax.text(
    93000, 0.97,
    "phase 3 — mixed 0.96/1.92 s blocks\n+ position-offset augmentation",
    ha="center", va="top", fontsize=8, color=GRAY,
)

ax.plot(sx + dx, sy + dy, "-o", ms=3, lw=1.6, color=ACCENT,
        label="streaming gate WER @0.96 s blocks")
ax.plot(wx, w960, "-o", ms=3, lw=1.6, color=ACCENT)
ax.plot(wx, w1920, "-s", ms=3, lw=1.6, color=ACCENT2, label="@1.92 s blocks")
ax.plot(wx, woff, "-^", ms=3, lw=1.4, color=ACCENT3,
        label="@0.96 s, position offset 4000 (≈85 min)")
ax.axhline(0.755, color=GRAY, lw=1.0, ls=":", xmax=0.08)
ax.text(200, 0.768, "untrained block-causal: 0.755", fontsize=7.5, color=GRAY,
        va="bottom")
ax.set_xlabel("cumulative training step")
ax.set_ylabel("held-out streaming WER (frozen decoder)")
ax.set_ylim(0.15, 1.0)
ax.set_xlim(-1500, 124500)
ax.legend(frameon=False, fontsize=8, loc="center right")
ax.set_title(
    "Held-out streaming WER during distillation (123k steps, ~6 H100-hours, audio only)",
    fontsize=10, loc="left",
)

fig.savefig(OUT, format="svg")
print("saved", OUT)
