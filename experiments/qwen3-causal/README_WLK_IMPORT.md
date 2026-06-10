# Qwen3 Causal ASR Experiment Import

This directory is a local import of the experimental `qwen3-asr-streaming-h100`
workspace used for Qwen3-ASR streaming/causal research on JarvisLab H100.

It is intentionally isolated from the WhisperLiveKit runtime. The files here are
for inspection, replaying experiments, and future manual integration planning.

Included:
- `qwen3_streaming/`: experimental streaming model/runtime helpers.
- `scripts/`: training, evaluation, manifest, teacher annotation, and benchmark scripts.
- `tests/`: unit tests from the experimental workspace.
- `configs/`: H100/JarvisLab config examples.
- `runs/`: lightweight JSON/JSONL metrics and summaries from the latest runs.
- `data/`: lightweight manifests only.

Excluded:
- model checkpoints such as `model.pt`;
- audio files;
- downloaded datasets;
- virtual environments and cache directories;
- archived tarballs.

Final state (2026-06-10, see RUNS.md for the full session log):
- The causal question is settled by the bounded mutable-tail sweep: WER stays
  flat at ~0.90 from zero recompute to full per-chunk recompute under a causal
  mask, while bidirectional windowed recompute holds 0.20. The audio tower
  requires bidirectional intra-window attention; a strict-causal Qwen3-ASR
  exists only through training (joint causal-mask distillation), not through
  inference engineering.
- The validated bounded-window runtime was promoted as the WhisperLiveKit
  `qwen3-streaming` backend. At its defaults (left12/seg200/chunk2s) it scores
  WER 0.084 vs MCIF human references (whisper normalization) at RTF 0.29 on
  H100, first display at 2s.
- Offline one-pass decoding degrades badly on long-form audio (0.6B: 0.207,
  1.7B: 0.120 vs the same references) — the segmented streamer beats both.
- Historical caveats: the context-distill Stage A/B table was
  language-corrupted (withdrawn), and preserve-regularized audio-only
  adaptation either reproduced the identity or collapsed (line closed).
