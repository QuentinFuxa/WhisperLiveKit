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

# phase bands
ax.axvspan(-1500, 3000, color="#f3f3f3", zorder=0)
ax.axvspan(3000, 63000, color="#eaf1f7", zorder=0)
ax.axvspan(63000, 124500, color="#eef5e8", zorder=0)
for x0 in (3000, 63000):
    ax.axvline(x0, color="#bdbdbd", lw=0.8)

def phase_title(x, lines):
    ax.text(x, 0.985, lines[0], ha="center", va="top", fontsize=8.5,
            color="#444444", fontweight="bold")
    ax.text(x, 0.925, "\n".join(lines[1:]), ha="center", va="top",
            fontsize=7.5, color=GRAY)

phase_title(1200, ["Phase 1", "warm-up", "100 h"])
phase_title(33000, ["Phase 2: scale", "LibriSpeech 960 h, no labels", "fixed 0.96 s blocks"])
phase_title(93500, ["Phase 3: serving regime", "mixed 0.96 / 1.92 s blocks", "+ random position offsets (up to 2 h)"])

ax.plot(sx + dx, sy + dy, "-o", ms=3, lw=1.6, color=ACCENT,
        label="WER, 0.96 s blocks")
ax.plot(wx, w960, "-o", ms=3, lw=1.6, color=ACCENT)
ax.plot(wx, w1920, "-s", ms=3, lw=1.6, color=ACCENT2,
        label="WER, 1.92 s blocks (serving config)")
ax.plot(wx, woff, "-^", ms=3, lw=1.4, color=ACCENT3,
        label="WER as if 85 min into a stream (offset 4000)")

# direct annotations
ax.annotate("before training: 0.755", (0, 0.755), (8500, 0.83),
            fontsize=7.5, color=GRAY,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.7))
ax.annotate("long streams broken before phase 3\n(positions beyond the 120 s table)",
            (63000, 0.883), (52000, 0.70), fontsize=7.5, color=ACCENT3,
            ha="center",
            arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=0.8))
ax.annotate("fixed by offset\naugmentation: 0.27",
            (123000, 0.274), (109000, 0.45), fontsize=7.5, color=ACCENT3,
            ha="center",
            arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=0.8))
ax.annotate("0.20", (123000, 0.203), (124800, 0.203), fontsize=8,
            color=ACCENT2, va="center")

ax.set_xlabel("cumulative training step")
ax.set_ylabel("WER, held-out streaming eval (decoder frozen)")
ax.set_ylim(0.15, 1.0)
ax.set_xlim(-1500, 131000)
ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.30, 0.42))
ax.set_title(
    "Distilling the offline tower into causal execution: 123k steps, about 6 H100 hours, audio only",
    fontsize=10, loc="left",
)

fig.savefig(OUT, format="svg")
print("saved", OUT)
