#!/usr/bin/env python3
"""Generate the architecture.png diagram for WhisperLiveKit README."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Colours ──
C_BG       = "#1a1a2e"
C_PANEL    = "#16213e"
C_PANEL2   = "#0f3460"
C_ACCENT   = "#e94560"
C_GREEN    = "#4ecca3"
C_ORANGE   = "#f5a623"
C_BLUE     = "#4a9eff"
C_PURPLE   = "#b06af2"
C_PINK     = "#ff6b9d"
C_YELLOW   = "#f0e68c"
C_TEXT     = "#e8e8e8"
C_TEXTDIM  = "#a0a0b0"
C_BOX_BG   = "#1e2d4a"
C_BOX_BG2  = "#2a1a3a"
C_BOX_BG3  = "#1a3a2a"
C_BORDER   = "#3a4a6a"

fig, ax = plt.subplots(1, 1, figsize=(20, 12), facecolor=C_BG)
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)


def box(x, y, w, h, label, color=C_BORDER, bg=C_BOX_BG, fontsize=8, bold=False,
        text_color=C_TEXT, radius=0.15):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=bg, edgecolor=color, linewidth=1.2,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight=weight, family="monospace")
    return rect


def arrow(x1, y1, x2, y2, color=C_TEXTDIM, style="->", lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def section_box(x, y, w, h, title, bg=C_PANEL, border=C_BORDER, title_color=C_ACCENT):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor=bg, edgecolor=border, linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(x + 0.15, y + h - 0.25, title, ha="left", va="top",
            fontsize=9, color=title_color, fontweight="bold", family="monospace")


# ═══════════════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════════════
ax.text(10, 11.7, "WhisperLiveKit Architecture", ha="center", va="center",
        fontsize=16, color=C_TEXT, fontweight="bold", family="monospace")
ax.text(10, 11.35, "CLI commands:  serve | listen | run | transcribe | bench | diagnose | models | pull | rm | check",
        ha="center", va="center", fontsize=7, color=C_TEXTDIM, family="monospace")

# ═══════════════════════════════════════════════════════════════════
# Left: Client / Server
# ═══════════════════════════════════════════════════════════════════
section_box(0.1, 7.0, 3.5, 4.0, "FastAPI Server", border=C_GREEN)

box(0.3, 10.0, 1.5, 0.5, "Web UI\nHTML + JS", color=C_GREEN, fontsize=7)
box(2.0, 10.0, 1.4, 0.5, "Frontend\n(optional)", color=C_GREEN, fontsize=7)

box(0.3, 9.1, 3.1, 0.6, "WebSocket /asr  •  /v1/listen", color=C_GREEN, fontsize=7, bold=True)
box(0.3, 8.3, 3.1, 0.5, "REST /v1/audio/transcriptions", color=C_GREEN, fontsize=7)
box(0.3, 7.4, 3.1, 0.5, "Health  •  /v1/models", color=C_GREEN, fontsize=7)

# Clients
ax.text(0.2, 6.5, "Clients:", fontsize=7, color=C_TEXTDIM, family="monospace")
for i, client in enumerate(["Browser", "OpenAI SDK", "Deepgram SDK", "TestHarness"]):
    box(0.3 + i * 0.9, 5.8, 0.8, 0.5, client, fontsize=5.5, bg="#1a2a1a", color="#3a6a3a")

# ═══════════════════════════════════════════════════════════════════
# Centre: Audio Processor (per-session pipeline)
# ═══════════════════════════════════════════════════════════════════
section_box(4.0, 5.5, 5.5, 5.5, "Audio Processor (per session)", border=C_BLUE)

box(4.3, 10.0, 2.0, 0.6, "FFmpeg\nDecoding", color=C_BLUE, bg="#1a2a4a", bold=True)
arrow(3.6, 9.4, 4.3, 10.2, color=C_GREEN)

box(6.6, 10.0, 2.6, 0.6, "Silero VAD\nspeech / silence", color=C_BLUE, bg="#1a2a4a")
arrow(6.3, 10.3, 6.6, 10.3, color=C_BLUE)

box(4.3, 8.8, 4.9, 0.8, "SessionASRProxy\nthread-safe per-session language override", color=C_BLUE, fontsize=7)
arrow(6.0, 10.0, 6.0, 9.6, color=C_BLUE)

box(4.3, 7.6, 2.3, 0.8, "DiffTracker\n(opt-in ?mode=diff)", color="#5a5a7a", fontsize=7)
box(6.9, 7.6, 2.3, 0.8, "Result Formatter\n→ FrontData.to_dict()", color=C_BLUE, fontsize=7)

# Streaming policies
ax.text(4.3, 7.1, "Streaming policies:", fontsize=7, color=C_ORANGE, fontweight="bold", family="monospace")
box(4.3, 6.2, 2.3, 0.7, "LocalAgreement\nHypothesisBuffer", color=C_ORANGE, bg="#2a2a1a", fontsize=7)
box(6.9, 6.2, 2.3, 0.7, "SimulStreaming\nAlignAtt (Whisper)", color=C_ORANGE, bg="#2a2a1a", fontsize=7)

# ═══════════════════════════════════════════════════════════════════
# Right: TranscriptionEngine (singleton)
# ═══════════════════════════════════════════════════════════════════
section_box(10.0, 0.3, 9.8, 10.7, "TranscriptionEngine (singleton — shared across sessions)",
            border=C_ACCENT, bg="#1e1520")

ax.text(10.2, 10.5, "6 ASR Backends", fontsize=9, color=C_ACCENT, fontweight="bold", family="monospace")

# ── Whisper backends ──
section_box(10.2, 7.3, 4.5, 3.0, "Whisper Family (chunk-based)", border=C_PURPLE, bg=C_BOX_BG2)

box(10.4, 9.2, 1.3, 0.6, "Faster\nWhisper", color=C_PURPLE, bg="#2a1a3a", fontsize=7, bold=True)
box(11.9, 9.2, 1.3, 0.6, "MLX\nWhisper", color=C_PURPLE, bg="#2a1a3a", fontsize=7, bold=True)
box(13.4, 9.2, 1.1, 0.6, "OpenAI\nWhisper", color=C_PURPLE, bg="#2a1a3a", fontsize=7)

ax.text(10.4, 8.7, "PCM → Encoder → Decoder → Tokens", fontsize=6.5, color=C_TEXTDIM, family="monospace")
ax.text(10.4, 8.3, "Uses LocalAgreement or SimulStreaming (AlignAtt)", fontsize=6, color=C_PURPLE, family="monospace")
ax.text(10.4, 7.9, "Language detection • Buffer trimming", fontsize=6, color=C_TEXTDIM, family="monospace")
ax.text(10.4, 7.5, "CPU / CUDA / MLX", fontsize=6, color=C_TEXTDIM, family="monospace")

# ── Voxtral backends ──
section_box(10.2, 3.8, 4.5, 3.2, "Voxtral (native streaming)", border=C_PINK, bg="#2a1520")

box(10.4, 5.9, 1.8, 0.6, "Voxtral MLX\n(Apple Silicon)", color=C_PINK, bg="#2a1520", fontsize=7, bold=True)
box(12.5, 5.9, 2.0, 0.6, "Voxtral HF\n(CUDA/MPS/CPU)", color=C_PINK, bg="#2a1520", fontsize=7, bold=True)

ax.text(10.4, 5.4, "Incremental encoder → Autoregressive decoder", fontsize=6.5, color=C_TEXTDIM, family="monospace")
ax.text(10.4, 5.0, "Sliding KV cache • Token-by-token output", fontsize=6, color=C_PINK, family="monospace")
ax.text(10.4, 4.6, "No chunking needed — truly streams audio", fontsize=6, color=C_TEXTDIM, family="monospace")
ax.text(10.4, 4.2, "4B params • 15 languages • 6-bit quant (MLX)", fontsize=6, color=C_TEXTDIM, family="monospace")

# ── Qwen3 backend ──
section_box(15.0, 3.8, 4.6, 3.2, "Qwen3 ASR (batch + aligner)", border=C_GREEN, bg=C_BOX_BG3)

box(15.2, 5.9, 1.5, 0.6, "Qwen3 ASR\n1.7B / 0.6B", color=C_GREEN, bg="#1a3a2a", fontsize=7, bold=True)
box(16.9, 5.9, 1.5, 0.6, "Qwen3\nSimul", color=C_GREEN, bg="#1a3a2a", fontsize=7, bold=True)
box(18.6, 5.9, 1.0, 0.6, "Forced\nAligner", color=C_GREEN, bg="#1a3a2a", fontsize=6.5)

ax.text(15.2, 5.4, "Batch + SimulStreaming (AlignAtt)", fontsize=6.5, color=C_TEXTDIM, family="monospace")
ax.text(15.2, 5.0, "ForcedAligner provides word timestamps", fontsize=6, color=C_GREEN, family="monospace")
ax.text(15.2, 4.6, "LocalAgreement or border-distance policy", fontsize=6, color=C_TEXTDIM, family="monospace")
ax.text(15.2, 4.2, "29 languages • CUDA/MPS/CPU", fontsize=6, color=C_TEXTDIM, family="monospace")

# ── OpenAI API ──
box(15.2, 7.7, 4.2, 0.6, "OpenAI API (cloud)", color="#5a6a7a", fontsize=7)
ax.text(15.2, 7.4, "Remote transcription • API key required", fontsize=6, color=C_TEXTDIM, family="monospace")

# ── Shared components ──
section_box(10.2, 0.5, 9.4, 3.0, "Shared Components", border="#5a6a7a", bg="#151520")

box(10.4, 2.2, 2.5, 0.8, "Mel Spectrogram\ncached DFT + filterbank",
    color="#5a6a7a", fontsize=7)
box(13.2, 2.2, 2.5, 0.8, "Diarization\nSortformer / pyannote",
    color="#5a6a7a", fontsize=7)
box(16.0, 2.2, 3.4, 0.8, "Translation\nNLLB • CTranslate2",
    color="#5a6a7a", fontsize=7)

box(10.4, 0.8, 4.0, 0.8, "WhisperLiveKitConfig\n(single source of truth)",
    color=C_ACCENT, fontsize=7, bold=True)
box(14.8, 0.8, 2.3, 0.8, "TestHarness\npipeline testing",
    color="#5a6a7a", fontsize=7)
box(17.3, 0.8, 2.3, 0.8, "Benchmark\n8 langs • 13 samples",
    color=C_ORANGE, fontsize=7, bold=True)

# ═══════════════════════════════════════════════════════════════════
# Arrows: main data flow
# ═══════════════════════════════════════════════════════════════════

# Audio processor → TranscriptionEngine
arrow(9.5, 8.5, 10.2, 8.5, color=C_ACCENT, lw=2)
ax.text(9.6, 8.8, "PCM audio", fontsize=6, color=C_ACCENT, family="monospace")

# TranscriptionEngine → Audio processor (results)
arrow(10.2, 7.0, 9.5, 7.0, color=C_GREEN, lw=2)
ax.text(9.6, 7.3, "ASRTokens", fontsize=6, color=C_GREEN, family="monospace")

# Streaming policy connections
arrow(5.5, 6.2, 5.5, 5.5, color=C_ORANGE, style="->")
arrow(8.1, 6.2, 8.1, 5.5, color=C_ORANGE, style="->")
ax.text(4.3, 5.6, "Whisper + Qwen3", fontsize=5.5, color=C_ORANGE, family="monospace")
ax.text(6.9, 5.6, "Whisper + Qwen3-simul", fontsize=5.5, color=C_ORANGE, family="monospace")

# Voxtral note (no policy needed)
ax.text(10.2, 3.5, "Voxtral: own streaming processor (no external policy)", fontsize=6,
        color=C_PINK, family="monospace", style="italic")


# ═══════════════════════════════════════════════════════════════════
# Legend
# ═══════════════════════════════════════════════════════════════════
legend_y = 5.0
ax.text(0.3, legend_y, "Streaming modes:", fontsize=7, color=C_TEXT, fontweight="bold", family="monospace")
for i, (label, color) in enumerate([
    ("Native streaming (Voxtral)", C_PINK),
    ("Chunk-based (Whisper)", C_PURPLE),
    ("Batch + aligner (Qwen3)", C_GREEN),
]):
    ax.plot([0.3], [legend_y - 0.4 - i * 0.35], "s", color=color, markersize=6)
    ax.text(0.6, legend_y - 0.4 - i * 0.35, label, fontsize=6.5, color=color,
            va="center", family="monospace")


plt.savefig("architecture.png", dpi=200, facecolor=C_BG, bbox_inches="tight", pad_inches=0.1)
print("Saved architecture.png")
