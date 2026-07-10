#!/usr/bin/env python3
"""
Generate polished benchmark figures for WhisperLiveKit H100 results.

Reads data from results.json, outputs PNGs to this directory.
Run: python3 benchmarks/h100/generate_figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(DIR, "results.json")))

# ── Style constants ──
COLORS = {
    "whisper":  "#d63031",
    "qwen_b":   "#6c5ce7",
    "qwen_s":   "#00b894",
    "voxtral":  "#fdcb6e",
    "fw_m5":    "#74b9ff",
    "mlx_m5":   "#55efc4",
    "vox_m5":   "#ffeaa7",
}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, name):
    path = os.path.join(DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


# ──────────────────────────────────────────────────────────
# Figure 1: WER vs RTF scatter — H100 (LibriSpeech clean)
# ──────────────────────────────────────────────────────────
def fig_scatter_clean():
    ls = DATA["librispeech_clean"]["systems"]
    m5 = DATA["m5_reference"]["systems"]

    fig, ax = plt.subplots(figsize=(9, 7.5))

    ax.axhspan(0, 10, color="#f0fff0", alpha=0.5, zorder=0)

    # M5 (ghost dots)
    for k, v in m5.items():
        ax.scatter(v["rtf"], v["wer"], s=50, c="silver", marker="o",
                   alpha=0.22, zorder=2, linewidths=0.4, edgecolors="gray")

    # H100 systems — (name, data, color, marker, size, label_x_off, label_y_off)
    pts = [
        ("Whisper large-v3",            ls["whisper_large_v3_batch"],     COLORS["whisper"], "h", 240, -8, -16),
        ("Qwen3-ASR 0.6B (batch)",     ls["qwen3_0.6b_batch"],           COLORS["qwen_b"],  "h", 170,  8,   6),
        ("Qwen3-ASR 1.7B (batch)",     ls["qwen3_1.7b_batch"],           COLORS["qwen_b"],  "h", 240,  8, -16),
        ("Voxtral 4B (vLLM)",          ls["voxtral_4b_vllm_realtime"],   COLORS["voxtral"], "D", 260,  8,   6),
        ("Qwen3 0.6B SimulStream+KV",  ls["qwen3_0.6b_simulstream_kv"], COLORS["qwen_s"],  "s", 220,  8,   6),
        ("Qwen3 1.7B SimulStream+KV",  ls["qwen3_1.7b_simulstream_kv"], COLORS["qwen_s"],  "s", 280,  8,  -16),
    ]

    for name, d, color, marker, sz, lx, ly in pts:
        ax.scatter(d["rtf"], d["wer"], s=sz, c=color, marker=marker,
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.annotate(name, (d["rtf"], d["wer"]), fontsize=8.5, fontweight="bold",
                    xytext=(lx, ly), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5))

    ax.set_xlabel("RTF  (lower = faster)")
    ax.set_ylabel("WER %  (lower = better)")
    ax.set_title("Speed vs Accuracy  —  LibriSpeech test-clean  (H100 80 GB)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-0.005, 0.20)
    ax.set_ylim(-0.3, 10)
    ax.grid(True, alpha=0.12)

    legend = [
        mpatches.Patch(color=COLORS["whisper"], label="Whisper large-v3"),
        mpatches.Patch(color=COLORS["qwen_b"],  label="Qwen3-ASR (batch)"),
        mpatches.Patch(color=COLORS["qwen_s"],  label="Qwen3 SimulStream+KV"),
        mpatches.Patch(color=COLORS["voxtral"], label="Voxtral 4B (vLLM)"),
        plt.Line2D([0],[0], marker="h", color="w", mfc="gray", ms=8, label="Batch"),
        plt.Line2D([0],[0], marker="s", color="w", mfc="gray", ms=8, label="Streaming"),
    ]
    ax.legend(handles=legend, fontsize=8.5, loc="upper right", framealpha=0.85, ncol=2)
    _save(fig, "wer_vs_rtf_clean.png")


# ──────────────────────────────────────────────────────────
# Figure 2: ACL6060 conference talks — the realistic test
# ──────────────────────────────────────────────────────────
def fig_scatter_acl6060():
    acl = DATA["acl6060"]["systems"]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axhspan(0, 15, color="#f0fff0", alpha=0.4, zorder=0)

    pts = [
        ("Voxtral 4B\n(vLLM Realtime)",    acl["voxtral_4b_vllm_realtime"],  COLORS["voxtral"], "D", 380),
        ("Qwen3 1.7B\nSimulStream+KV",     acl["qwen3_1.7b_simulstream_kv"], COLORS["qwen_s"],  "s", 380),
        ("Qwen3 0.6B\nSimulStream+KV",     acl["qwen3_0.6b_simulstream_kv"], COLORS["qwen_s"],  "s", 260),
        ("Whisper large-v3\n(batch)",       acl["whisper_large_v3_batch"],    COLORS["whisper"], "h", 320),
    ]
    label_off = [(10, -12), (10, 6), (10, 6), (10, 6)]

    for (name, d, color, marker, sz), (lx, ly) in zip(pts, label_off):
        wer = d["avg_wer"]
        rtf = d["avg_rtf"]
        ax.scatter(rtf, wer, s=sz, c=color, marker=marker,
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.annotate(name, (rtf, wer), fontsize=9.5, fontweight="bold",
                    xytext=(lx, ly), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6))

    # Cascade annotation
    ax.annotate("Full STT+MT cascade\nRTF 0.15 (real-time)",
                xy=(0.151, 1), xytext=(0.25, 4),
                fontsize=9, fontstyle="italic", color="#1565c0",
                arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="#e3f2fd", ec="#90caf9", alpha=0.9))

    ax.set_xlabel("RTF  (lower = faster)")
    ax.set_ylabel("WER %  (lower = better)")
    ax.set_title("ACL6060 Conference Talks  —  5 talks, 58 min  (H100 80 GB)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-0.005, 0.30)
    ax.set_ylim(-1, 26)
    ax.grid(True, alpha=0.12)
    _save(fig, "wer_vs_rtf_acl6060.png")


# ──────────────────────────────────────────────────────────
# Figure 3: Bar chart — WER + RTF side-by-side
# ──────────────────────────────────────────────────────────
def fig_bars():
    names = [
        "Whisper\nlarge-v3", "Voxtral 4B\n(vLLM)", "Qwen3 0.6B\n(batch)",
        "Qwen3 1.7B\n(batch)", "Qwen3 0.6B\nSimulStream", "Qwen3 1.7B\nSimulStream",
    ]
    wer_c = [2.02, 2.71, 2.30, 2.46, 6.44, 8.09]
    wer_o = [7.79, 9.26, 6.12, 5.34, 9.27, 9.56]
    rtf_c = [0.071, 0.137, 0.065, 0.069, 0.109, 0.117]
    fwl   = [472, 137, 432, 457, 91, 94]  # ms
    cols  = [COLORS["whisper"], COLORS["voxtral"], COLORS["qwen_b"],
             COLORS["qwen_b"], COLORS["qwen_s"], COLORS["qwen_s"]]
    cols_l = ["#ff7675", "#ffeaa7", "#a29bfe", "#a29bfe", "#55efc4", "#55efc4"]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # WER
    ax = axes[0]
    w = 0.36
    ax.bar(x - w/2, wer_c, w, color=cols, alpha=0.9, edgecolor="white", label="test-clean")
    ax.bar(x + w/2, wer_o, w, color=cols_l, alpha=0.65, edgecolor="white", label="test-other")
    ax.set_ylabel("WER %")
    ax.set_title("Word Error Rate", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5, rotation=25, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.15)
    for i, v in enumerate(wer_c):
        ax.text(i - w/2, v + 0.2, f"{v:.1f}", ha="center", fontsize=7, fontweight="bold")

    # RTF
    ax = axes[1]
    ax.bar(x, rtf_c, 0.55, color=cols, alpha=0.9, edgecolor="white")
    ax.set_ylabel("RTF  (lower = faster)")
    ax.set_title("Real-Time Factor (test-clean)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.15)
    for i, v in enumerate(rtf_c):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    # First-word latency
    ax = axes[2]
    ax.bar(x, fwl, 0.55, color=cols, alpha=0.9, edgecolor="white")
    ax.set_ylabel("ms")
    ax.set_title("First Word Latency", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.15)
    for i, v in enumerate(fwl):
        ax.text(i, v + 8, f"{v}", ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("LibriSpeech Benchmark  —  H100 80 GB", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "bars_wer_rtf_latency.png")


# ──────────────────────────────────────────────────────────
# Figure 4: Clean vs Other robustness
# ──────────────────────────────────────────────────────────
def fig_robustness():
    models = [
        ("Whisper large-v3",          2.02, 7.79, COLORS["whisper"], "h", 280),
        ("Qwen3 0.6B (batch)",       2.30, 6.12, COLORS["qwen_b"],  "h", 180),
        ("Qwen3 1.7B (batch)",       2.46, 5.34, COLORS["qwen_b"],  "h", 280),
        ("Voxtral 4B (vLLM)",        2.71, 9.26, COLORS["voxtral"], "D", 280),
        ("Qwen3 0.6B\nSimulStream",  6.44, 9.27, COLORS["qwen_s"],  "s", 240),
        ("Qwen3 1.7B\nSimulStream",  8.09, 9.56, COLORS["qwen_s"],  "s", 300),
    ]
    # Manual label offsets — carefully placed to avoid overlap
    offsets = [(-55, 10), (8, 10), (8, -18), (-55, -18), (-10, 12), (10, -18)]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.plot([0, 13], [0, 13], "--", color="#ccc", lw=1, zorder=1)
    ax.fill_between([0, 13], [0, 13], [13, 13], color="#fff5f5", alpha=0.5, zorder=0)
    ax.text(4, 11, "degrades more\non noisy audio", fontsize=9, color="#bbb", fontstyle="italic")

    for (name, wc, wo, color, marker, sz), (lx, ly) in zip(models, offsets):
        ax.scatter(wc, wo, s=sz, c=color, marker=marker,
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.annotate(name, (wc, wo), fontsize=8.5, fontweight="bold",
                    xytext=(lx, ly), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6))
        deg = wo - wc
        ax.annotate(f"+{deg:.1f}%", (wc, wo), fontsize=7, color="#999",
                    xytext=(-6, -13), textcoords="offset points")

    ax.set_xlabel("WER % on test-clean")
    ax.set_ylabel("WER % on test-other")
    ax.set_title("Clean vs Noisy Robustness  (H100 80 GB)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-0.3, 12)
    ax.set_ylim(-0.3, 12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.12)
    _save(fig, "robustness_clean_vs_other.png")


# ──────────────────────────────────────────────────────────
# Figure 5: ACL6060 per-talk breakdown (Qwen3 vs Voxtral)
# ──────────────────────────────────────────────────────────
def fig_per_talk():
    q = DATA["acl6060"]["systems"]["qwen3_1.7b_simulstream_kv"]["per_talk"]
    v = DATA["acl6060"]["systems"]["voxtral_4b_vllm_realtime"]["per_talk"]
    talks = DATA["acl6060"]["talks"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(talks))
    w = 0.35

    bars_v = ax.bar(x - w/2, [v[t] for t in talks], w, color=COLORS["voxtral"],
                    edgecolor="white", label="Voxtral 4B (vLLM)")
    bars_q = ax.bar(x + w/2, [q[t] for t in talks], w, color=COLORS["qwen_s"],
                    edgecolor="white", label="Qwen3 1.7B SimulStream+KV")

    for bar in bars_v:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", fontsize=8)
    for bar in bars_q:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", fontsize=8)

    ax.set_xlabel("ACL6060 Talk ID")
    ax.set_ylabel("WER %")
    ax.set_title("Per-Talk WER  —  ACL6060 Conference Talks  (H100 80 GB)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Talk {t}" for t in talks])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.15)
    ax.set_ylim(0, 18)
    _save(fig, "acl6060_per_talk.png")


if __name__ == "__main__":
    print("Generating H100 benchmark figures...")
    fig_scatter_clean()
    fig_scatter_acl6060()
    fig_bars()
    fig_robustness()
    fig_per_talk()
    print("Done!")
