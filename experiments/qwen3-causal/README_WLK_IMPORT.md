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

Latest useful result:
- Stage A context distillation improved WLK first-20 slightly at `left=8s/6s`.
- Stage B `{6s,4s}` regressed and is not promoted.
- The current direction should shift from chunk/local recompute to a true
  append-only audio KV cache if we want Voxtral-like behavior.
