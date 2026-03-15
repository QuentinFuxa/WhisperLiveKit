#!/usr/bin/env python3
"""
Generate combined M5 vs H100 benchmark figure for WhisperLiveKit.

Produces a WER vs RTF scatter plot comparing Apple M5 (MLX) and
NVIDIA H100 results on LibriSpeech test-clean.

Note: M5 uses per-utterance evaluation (500 samples), while H100
uses chapter-grouped evaluation (91 chapters). Per-utterance WER
is typically lower because short utterances avoid long-range errors.

Run: python3 benchmarks/m5/generate_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DIR = os.path.dirname(os.path.abspath(__file__))
H100_DATA = json.load(open(os.path.join(DIR, "..", "h100", "results.json")))
M5_DATA = json.load(open(os.path.join(DIR, "results.json")))

# -- Style --
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "whisper":  "#d63031",
    "qwen_b":   "#6c5ce7",
    "qwen_s":   "#00b894",
    "voxtral":  "#fdcb6e",
    "m5_qwen":  "#0984e3",
}


def _save(fig, name):
    path = os.path.join(DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {name}")


def fig_m5_vs_h100():
    """WER vs RTF scatter: M5 (MLX) and H100 (CUDA) on LibriSpeech test-clean."""
    h100 = H100_DATA["librispeech_clean"]["systems"]
    m5 = M5_DATA["models"]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Light green band for "good WER" zone
    ax.axhspan(0, 5, color="#f0fff0", alpha=0.5, zorder=0)

    # --- H100 points ---
    h100_pts = [
        ("Whisper large-v3\n(H100, batch)",          h100["whisper_large_v3_batch"],     COLORS["whisper"], "h", 220),
        ("Qwen3 0.6B batch\n(H100)",                 h100["qwen3_0.6b_batch"],           COLORS["qwen_b"],  "h", 170),
        ("Qwen3 1.7B batch\n(H100)",                 h100["qwen3_1.7b_batch"],           COLORS["qwen_b"],  "h", 220),
        ("Voxtral 4B vLLM\n(H100)",                  h100["voxtral_4b_vllm_realtime"],   COLORS["voxtral"], "D", 240),
        ("Qwen3 0.6B SimulStream+KV\n(H100)",        h100["qwen3_0.6b_simulstream_kv"],  COLORS["qwen_s"],  "s", 200),
        ("Qwen3 1.7B SimulStream+KV\n(H100)",        h100["qwen3_1.7b_simulstream_kv"],  COLORS["qwen_s"],  "s", 260),
    ]
    h100_offsets = [(-55, 10), (-55, -22), (8, -18), (8, 10), (8, 10), (8, -18)]

    for (name, d, color, marker, sz), (lx, ly) in zip(h100_pts, h100_offsets):
        ax.scatter(d["rtf"], d["wer"], s=sz, c=color, marker=marker,
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.annotate(name, (d["rtf"], d["wer"]), fontsize=7.5, fontweight="bold",
                    xytext=(lx, ly), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5))

    # --- M5 points ---
    m5_pts = [
        ("Qwen3 0.6B SimulStream\n(M5, MLX)", m5["qwen3-asr-0.6b-simul"], COLORS["m5_qwen"], "^", 260),
        ("Qwen3 1.7B SimulStream\n(M5, MLX)", m5["qwen3-asr-1.7b-simul"], COLORS["m5_qwen"], "^", 300),
    ]
    m5_offsets = [(8, 8), (8, -18)]

    for (name, d, color, marker, sz), (lx, ly) in zip(m5_pts, m5_offsets):
        ax.scatter(d["rtf"], d["wer"], s=sz, c=color, marker=marker,
                   edgecolors="white", linewidths=1.5, zorder=6)
        ax.annotate(name, (d["rtf"], d["wer"]), fontsize=7.5, fontweight="bold",
                    xytext=(lx, ly), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5))

    # --- Connecting lines between same models on different hardware ---
    # 0.6B: H100 SimulStream+KV -> M5 SimulStream
    ax.plot([h100["qwen3_0.6b_simulstream_kv"]["rtf"], m5["qwen3-asr-0.6b-simul"]["rtf"]],
            [h100["qwen3_0.6b_simulstream_kv"]["wer"], m5["qwen3-asr-0.6b-simul"]["wer"]],
            "--", color="#0984e3", alpha=0.3, lw=1.5, zorder=3)
    # 1.7B: H100 SimulStream+KV -> M5 SimulStream
    ax.plot([h100["qwen3_1.7b_simulstream_kv"]["rtf"], m5["qwen3-asr-1.7b-simul"]["rtf"]],
            [h100["qwen3_1.7b_simulstream_kv"]["wer"], m5["qwen3-asr-1.7b-simul"]["wer"]],
            "--", color="#0984e3", alpha=0.3, lw=1.5, zorder=3)

    # --- RTF = 1 line (real-time boundary) ---
    ax.axvline(x=1.0, color="#e17055", linestyle=":", alpha=0.5, lw=1.5, zorder=1)
    ax.text(1.02, 0.5, "real-time\nboundary", fontsize=8, color="#e17055",
            fontstyle="italic", alpha=0.7, va="bottom")

    # --- Methodology note ---
    ax.text(0.98, 0.02,
            "H100: chapter-grouped WER (91 chapters)  |  M5: per-utterance WER (500 samples)\n"
            "Per-utterance WER is typically lower -- results are not directly comparable.",
            transform=ax.transAxes, fontsize=7.5, color="#666",
            ha="right", va="bottom", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff9e6", ec="#ddd", alpha=0.9))

    ax.set_xlabel("RTF  (lower = faster)")
    ax.set_ylabel("WER %  (lower = better)")
    ax.set_title("H100 vs M5 (MLX)  --  Qwen3-ASR on LibriSpeech test-clean",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-0.01, 1.1)
    ax.set_ylim(-0.5, 10)
    ax.grid(True, alpha=0.12)

    legend = [
        mpatches.Patch(color=COLORS["whisper"], label="Whisper large-v3 (H100)"),
        mpatches.Patch(color=COLORS["qwen_b"],  label="Qwen3-ASR batch (H100)"),
        mpatches.Patch(color=COLORS["qwen_s"],  label="Qwen3 SimulStream+KV (H100)"),
        mpatches.Patch(color=COLORS["voxtral"], label="Voxtral 4B vLLM (H100)"),
        mpatches.Patch(color=COLORS["m5_qwen"], label="Qwen3 SimulStream (M5, MLX)"),
        plt.Line2D([0], [0], marker="h", color="w", mfc="gray", ms=8, label="Batch mode"),
        plt.Line2D([0], [0], marker="s", color="w", mfc="gray", ms=8, label="Streaming (H100)"),
        plt.Line2D([0], [0], marker="^", color="w", mfc="gray", ms=8, label="Streaming (M5)"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="upper right", framealpha=0.85, ncol=2)
    _save(fig, "m5_vs_h100_wer_rtf.png")


if __name__ == "__main__":
    print("Generating M5 vs H100 benchmark figure...")
    fig_m5_vs_h100()
    print("Done!")
