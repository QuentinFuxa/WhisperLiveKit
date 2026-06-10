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

Latest useful result (updated 2026-06-10 after repatriation audit, see RUNS.md):
- The context-distill Stage A/B WER table was language-corrupted (no explicit
  English prompt; accented items auto-switched to French). Its conclusions are
  withdrawn.
- Preserve-regularized audio-only adaptation (left8/left12) either matches the
  identity or collapses; the audio-side-only training line is closed.
- Best validated v1: untrained Qwen3-ASR-0.6B + segmented cached
  full-hypothesis streamer at `left12/seg200/chunk10s`: WER 0.110 vs MCIF
  human references with Whisper normalization (0.158 under the old
  teacher-ref/legacy-norm scoring), RTF 0.10 on 21 full WLK audios.
- Strict append-only audio (`qwen_audio_causal_kv`) remains broken (WER 0.91
  vs 0.20 for windowed recompute at the same zero right context). The open
  question is the bounded mutable-tail sweep proposed in the 2026-06-03
  backend diagnostic.
