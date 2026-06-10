# Qwen3-ASR Streaming Experiments

Research workspace for streaming Qwen3-ASR. The goal was a Voxtral-style
causal model where appending audio never recomputes the past. The full
experiment history (including every failed line and the 2026-06-10 audits)
lives in [RUNS.md](RUNS.md); this README only describes what survived.

## Current state (2026-06-10)

**Validated v1 — bounded-recompute streaming, no training.** The unmodified
`Qwen/Qwen3-ASR-0.6B` audio tower is re-run over a bounded local window
(`left_context=12s`, `right_context=640ms`); finalized audio embeddings are
cached append-only; the text decoder regenerates a full hypothesis per chunk
inside a bounded segment; words are committed with a stable-prefix rule.

Measured on 21 full WLK/MCIF talks (H100, chunk 10s, seg200, vs MCIF human
references with Whisper text normalization):

```text
WER 0.110   RTF 0.10   cache_bound_violations 0   committed_revisions 0
```

**Closed line — audio-side-only adaptation.** Training only the audio path
against a frozen Qwen decoder failed four ways (adapter CE, audio LoRA,
context distillation, preserve-regularized CE): strong regularization
reproduces the identity, weak regularization collapses WLK quality.

**Open question — how much mutable history does the audio tower need?**
Strict append-only audio (`qwen_audio_causal_kv` backend, zero recompute)
collapses to WER 0.91 while a recomputed window at the same zero right
context holds 0.20. The decisive unrun experiment is the bounded
mutable-tail sweep: freeze per-layer KV older than T seconds, recompute only
the last T, sweep T in {0, 0.5, 1, 2, 4, 8, 12}s.

## Layout

- `qwen3_streaming/cached_full_hypothesis.py` — `CachedFullHypothesisStreamer`
  and `SegmentedCachedFullHypothesisStreamer`, the validated runtime.
- `qwen3_streaming/stable_commit.py` — stable-prefix commit logic.
- `qwen3_streaming/native_realtime_model.py` — inference model wrappers:
  `Qwen3ASRRealtimeQwenAudioSurgeryModel` (bounded-window recompute) and
  `Qwen3ASRRealtimeQwenAudioCausalModel` (strict append-only KV, kept for the
  mutable-tail sweep), plus the cached-audio decode entry points.
- `qwen3_streaming/metrics.py` — WER and repetition/streaming metrics.
- `scripts/infer_cached_full_hypothesis.py` / `eval_cached_full_hypothesis.py`
  — single-file and batch streaming evals. `--language` is required.
- `scripts/build_mcif_reference_manifest.py` / `rescore_jsonl.py` — MCIF human
  references and offline re-scoring of per-item prediction JSONLs.
- Data tooling: `make_audio_manifest.py`, `annotate_teacher_transcripts.py`,
  `filter_teacher_manifest.py`, `prepare_qwen_aligned_jsonl.py`,
  `align_manifest_with_qwen.py`, `slice_audio_manifest.py`.
- `runs/` — lightweight metrics from all historical runs; `data/` — manifests.

The dead experiment code (scratch causal encoder, emit gate, CTC/RNNT
objectives, LoRA training, the 4k-line training script) was removed in the
2026-06-10 cleanup; consult git history if needed.

## Running the validated eval

```bash
python scripts/eval_cached_full_hypothesis.py \
  --model-id Qwen/Qwen3-ASR-0.6B \
  --manifest-jsonl data/wlk_teacher_1p7b_v0/manifest.teacher.jsonl \
  --output-jsonl runs/my_eval.jsonl \
  --device cuda \
  --chunk-ms 10000 \
  --qwen-audio-left-context-sec 12 \
  --qwen-audio-right-context-ms 640 \
  --segment-max-cached-steps 200 \
  --language English
```

Always pass an explicit `--language`. Auto detection flips accented English
audio to French and silently corrupts WER (RUNS.md 2026-06-10 audit).

Re-score any per-item JSONL against human references:

```bash
python scripts/build_mcif_reference_manifest.py
python scripts/rescore_jsonl.py runs/my_eval.jsonl
```

## Tests

```bash
PYTHONPATH=. python3 -m pytest -q tests
```

Model tests use fake towers/decoders and run on CPU.

## Next steps

1. Promote the validated runtime as the `qwen3-streaming` WhisperLiveKit
   backend (in progress, see `whisperlivekit/qwen3_streaming/`).
2. GPU session: offline 0.6B/1.7B upper bounds vs human references, the
   bounded mutable-tail sweep, and a realistic-latency eval at chunk 1-2s.
3. Decide the strict-causal path from the sweep result: engineering
   (append-mostly cache with small mutable tail) vs joint distillation
   training vs staying on the bounded-window v1.
