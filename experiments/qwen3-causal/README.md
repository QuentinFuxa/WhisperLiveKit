# Qwen3-ASR Native Realtime H100 Experiments

This workspace is the GPU-first lab for a native realtime Qwen3-ASR variant.
The current target is not prefix-SFT over the offline model. The target is an
append-only audio path: causal audio encoder cache, 80 ms frame adapter, and
decoder KV cache.

Working order:

1. Keep vLLM CUDA Qwen3-ASR offline/realtime baselines as references.
2. Build and train `Qwen3-ASR-Realtime` on H100 in PyTorch.
3. Validate stream-synchronous quality and latency on GPU.
4. Patch vLLM CUDA serving for the native architecture.
5. Port to vllm-metal only after the GPU model proves cache, quality, and latency.

WhisperLiveKit is intentionally not modified here. Its contribution rules reject PRs that are fully or mostly AI-generated, so this repo is an external experiment/runbook.

## Native Realtime Prototype

The new implementation lives in:

- `qwen3_streaming/realtime_config.py`: realtime timing and audio config.
- `qwen3_streaming/realtime_targets.py`: `[P]` / `[W]` frame-synchronous target builder from word timestamps.
- `qwen3_streaming/native_realtime_model.py`: PyTorch causal audio encoder, Qwen audio surgery backend, adapters, cached decoder scaffold, and stream state.
- `scripts/train_realtime_smoke.py`: tiny H100 smoke train that verifies the stack can optimize and save a checkpoint.
- `scripts/train_realtime_tiny_asr.py`: first end-to-end tiny ASR run on real FLEURS audio using Qwen tokenizer and provisional heuristic word timings.
- `scripts/prepare_qwen_aligned_jsonl.py`: builds realtime manifests using `Qwen/Qwen3-ForcedAligner-0.6B` word timestamps.

Core invariant:

```text
encode(0:20s) then append(20:21s) processes only the new mel frames;
it does not re-run the audio encoder over 0:20s.
```

For `--decoder-backend qwen_audio_surgery`, v1 is slightly weaker but more
practical: it reuses Qwen3-ASR's pretrained `thinker.audio_tower` and recomputes
only a bounded local window, default `left_context=15s` plus
`right_context=640ms`. That keeps the offline audio weights as the starting
point while avoiding full-history re-embedding.

Run local tests:

```bash
python3 -m pytest -q
```

The model tests require PyTorch and are skipped on machines without it. On H100:

```bash
source .venv/bin/activate
python scripts/train_realtime_smoke.py \
  --output-dir runs/realtime_smoke \
  --steps 100 \
  --device cuda
```

This is only a smoke run. A useful model still needs aligned FR/EN data and Qwen decoder initialization.

First real-audio tiny ASR v0:

```bash
source .venv/bin/activate
python scripts/train_realtime_tiny_asr.py \
  --output-dir runs/realtime_fleurs_tiny_v0 \
  --max-train-per-source 48 \
  --max-eval-per-source 8 \
  --steps 200 \
  --batch-size 2 \
  --device cuda
```

This trains on actual FLEURS audio/transcripts, but the word timestamps are still
heuristic. Treat the checkpoint as an architecture artifact, not as a quality
model.

Qwen forced-aligned data:

```bash
source .venv/bin/activate
python scripts/prepare_qwen_aligned_jsonl.py \
  --out-dir data/qwen_aligned_fleurs_tiny \
  --sources fleurs_en fleurs_fr \
  --max-train-per-source 128 \
  --max-eval-per-source 16 \
  --device-map cuda:0 \
  --dtype bfloat16
```

Then train from those manifests:

```bash
python scripts/train_realtime_tiny_asr.py \
  --output-dir runs/realtime_fleurs_qwen_aligned_v0 \
  --train-manifest-jsonl data/qwen_aligned_fleurs_tiny/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_tiny/eval_manifest.jsonl \
  --steps 500 \
  --batch-size 2 \
  --device cuda \
  --wait-loss-weight 0.05
```

Qwen decoder-initialized diagnostic run:

```bash
python scripts/train_realtime_tiny_asr.py \
  --decoder-backend qwen \
  --qwen-decoder-model Qwen/Qwen3-ASR-0.6B \
  --output-dir runs/realtime_fleurs_qwen_decoder_v0 \
  --train-manifest-jsonl data/qwen_aligned_fleurs_tiny/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_tiny/eval_manifest.jsonl \
  --steps 500 \
  --batch-size 2 \
  --device cuda \
  --audio-layers 3 \
  --audio-heads 8 \
  --wait-loss-weight 0.05 \
  --freeze-qwen-layers
```

This backend reuses Qwen3-ASR's pretrained text decoder, token embeddings, and
LM head while replacing the offline audio tower with the causal audio encoder.
It is still experimental: the current tiny runs prove that the causal scaffold
loads, trains, saves, reloads, and streams, but the ASR quality is not usable
yet. `--no-word-start-token` and `--wait-loss-weight 0.0` are diagnostic knobs
for separating wait-token collapse from text repetition collapse.

Qwen audio-tower surgery backend:

```bash
python scripts/train_realtime_tiny_asr.py \
  --decoder-backend qwen_audio_surgery \
  --qwen-decoder-model Qwen/Qwen3-ASR-0.6B \
  --output-dir runs/realtime_qwen_audio_surgery_smoke \
  --train-manifest-jsonl data/qwen_aligned_fleurs_tiny/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_tiny/eval_manifest.jsonl \
  --steps 100 \
  --batch-size 1 \
  --device cuda \
  --audio-heads 8 \
  --qwen-audio-left-context-sec 15 \
  --qwen-audio-right-context-ms 640 \
  --freeze-qwen-all \
  --freeze-qwen-audio \
  --emit-gate-loss-weight 0.1 \
  --emit-gate-wait-weight 0.2 \
  --lr 1e-4 \
  --no-word-start-token
```

This backend loads Qwen3-ASR's pretrained audio tower, text decoder, tokenizer
embeddings, and LM head. The stream state keeps a pending mel buffer and emits
only audio steps whose fixed right context is available. Stage 0 should prove
chunked/offline wiring and bounded recompute before any longer training run.
Useful knobs:

- `--freeze-qwen-audio`: freeze the pretrained audio tower and train only the
  identity-initialized audio projector plus emit gate.
- `--train-qwen-audio-last-n-layers N`: freeze most of the audio tower and
  unfreeze only the last `N` Qwen audio layers plus output projection modules.
- `--qwen-audio-strict-causal`: set right context to zero for a harsher
  diagnostic, not the recommended v1 default.

Stage 0.5 validation for bounded recompute and chunk latency:

```bash
python scripts/validate_qwen_audio_surgery.py \
  --audio-dir data/wlk_audio_stage05 \
  --max-files 5 \
  --max-audio-sec 30 \
  --chunk-ms 320 \
  --left-context-sec 15 \
  --right-context-ms 640 \
  --device cuda \
  --output-jsonl runs/stage05/qwen_audio_surgery_validate.jsonl \
  --summary-json runs/stage05/qwen_audio_surgery_validate.summary.json
```

Latest Qwen audio-surgery training status:

```text
Stage A 1k, frozen Qwen audio:       eval_loss 5.2562
Stage B 1.5k, last 4 audio unfrozen: eval_loss 5.2421
Stage A 4k, frozen Qwen audio:       eval_loss 5.0468
Stage A+ 4k, 2x2048 adapter blocks:  eval_loss 4.9985
Stage A+ +2k resume, LR 2e-5:        eval_loss 4.8830
Stage A+ +4k resume, LR 1e-5:        eval_loss 4.8568
```

The current conclusion is to keep improving the frozen-tower adapter/gate path.
Stage A+ adds optional residual SwiGLU adapter blocks after the Qwen audio
projector and beats the previous frozen-tower Stage A baseline without unfreezing
Qwen audio. Continued training improves FLEURS eval loss, but WLK qualitative
output gets longer and more repetitive, so this is no longer a pure training-time
problem. The next step should improve data/targets or decoding regularization,
not just add more steps.

Stage A+ command shape:

```bash
python scripts/train_realtime_tiny_asr.py \
  --decoder-backend qwen_audio_surgery \
  --qwen-decoder-model Qwen/Qwen3-ASR-0.6B \
  --qwen-dtype bfloat16 \
  --output-dir runs/realtime_qwen_audio_surgery_stageAplus_adapter2x2048 \
  --train-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/eval_manifest.jsonl \
  --steps 4000 \
  --batch-size 2 \
  --device cuda \
  --qwen-audio-adapter-hidden-dim 2048 \
  --qwen-audio-adapter-layers 2 \
  --qwen-audio-adapter-residual-scale 0.1 \
  --freeze-qwen-all \
  --freeze-qwen-audio \
  --emit-gate-loss-weight 1.0 \
  --emit-gate-wait-weight 1.5 \
  --lr 5e-5 \
  --no-word-start-token \
  --max-audio-sec 16
```

Continue an existing checkpoint:

```bash
python scripts/train_realtime_tiny_asr.py \
  --resume-from-checkpoint runs/realtime_qwen_audio_surgery_stageAplus_adapter2x2048 \
  --output-dir runs/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume \
  --train-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/eval_manifest.jsonl \
  --steps 2000 \
  --batch-size 2 \
  --device cuda \
  --freeze-qwen-all \
  --freeze-qwen-audio \
  --emit-gate-loss-weight 1.0 \
  --emit-gate-wait-weight 1.5 \
  --lr 2e-5 \
  --no-word-start-token \
  --max-audio-sec 16
```

Checkpoint resume reloads model weights and config but not optimizer state.
Reapply freeze flags explicitly, because `requires_grad` is runtime state.

Current Qwen-decoder training uses a separate emit/wait gate. Text CE is applied
only on text frames, while the binary gate decides whether the stream emits a
token or waits. This makes the wait/text ratio controllable without forcing
`[P]` into the language-model vocabulary loss:

```bash
python scripts/train_realtime_tiny_asr.py \
  --decoder-backend qwen \
  --qwen-decoder-model Qwen/Qwen3-ASR-0.6B \
  --output-dir runs/realtime_qwen_emit_gate_w15_v0 \
  --train-manifest-jsonl data/qwen_aligned_fleurs_tiny/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_tiny/eval_manifest.jsonl \
  --steps 2000 \
  --batch-size 2 \
  --device cuda \
  --audio-layers 3 \
  --audio-heads 8 \
  --emit-gate-loss-weight 1.0 \
  --emit-gate-wait-weight 1.5 \
  --freeze-qwen-layers \
  --train-qwen-last-n-layers 4 \
  --lr 3e-5 \
  --no-word-start-token
```

Streaming inference can sweep the gate threshold without retraining:

```bash
python scripts/infer_realtime_checkpoint.py \
  --checkpoint runs/realtime_qwen_emit_gate_w15_v0 \
  --audio data/wlk_audio/myfXyntFYL_20s.wav \
  --device cuda \
  --chunk-ms 320 \
  --emit-threshold 0.30 \
  --json
```

LoRA Qwen-decoder training freezes the pretrained Qwen text model and LM head,
then wraps decoder linear projections with local LoRA adapters. The causal audio
encoder, frame adapter, emit gate, and LoRA weights stay trainable:

```bash
python scripts/train_realtime_tiny_asr.py \
  --decoder-backend qwen \
  --qwen-decoder-model Qwen/Qwen3-ASR-0.6B \
  --output-dir runs/realtime_qwen_lora_r16_emit_gate_v0 \
  --train-manifest-jsonl data/qwen_aligned_fleurs_tiny/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_tiny/eval_manifest.jsonl \
  --steps 2000 \
  --batch-size 2 \
  --device cuda \
  --audio-layers 3 \
  --audio-heads 8 \
  --emit-gate-loss-weight 1.0 \
  --emit-gate-wait-weight 1.5 \
  --qwen-lora-rank 16 \
  --qwen-lora-alpha 32 \
  --qwen-lora-dropout 0.05 \
  --lr 8e-5 \
  --no-word-start-token
```

This is cheaper than unfreezing Qwen decoder blocks, but it still needs a much
larger aligned/distilled dataset before the text is usable.

Larger FLEURS manifest with Qwen3 forced alignments:

```bash
python scripts/prepare_qwen_aligned_jsonl.py \
  --out-dir data/qwen_aligned_fleurs_16s_v0 \
  --sources fleurs_en fleurs_fr \
  --max-train-per-source 1200 \
  --max-eval-per-source 120 \
  --max-audio-sec 16 \
  --drop-long-audio \
  --device-map cuda:0 \
  --dtype bfloat16 \
  --allow-heuristic-fallback \
  --skip-existing
```

`--drop-long-audio` is important: truncating audio while keeping the full
transcript creates bad word targets. The script also reads dataset audio with
`datasets.Audio(decode=False)` and `soundfile`, so it does not require
`torchcodec`.

Larger LoRA diagnostic run:

```bash
python scripts/train_realtime_tiny_asr.py \
  --decoder-backend qwen \
  --qwen-decoder-model Qwen/Qwen3-ASR-0.6B \
  --output-dir runs/realtime_qwen_lora_r16_fleurs_16s_space_v0 \
  --train-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/eval_manifest.jsonl \
  --steps 4000 \
  --batch-size 2 \
  --device cuda \
  --audio-layers 3 \
  --audio-heads 8 \
  --emit-gate-loss-weight 1.0 \
  --emit-gate-wait-weight 1.5 \
  --qwen-lora-rank 16 \
  --qwen-lora-alpha 32 \
  --qwen-lora-dropout 0.05 \
  --lr 5e-5 \
  --no-word-start-token \
  --max-audio-sec 16 \
  --log-every 200
```

Current status from the 2400-example runs: the pipeline trains and streams, but
the model is not usable ASR yet. The 16k-step rank-32 run drove train loss down
to `0.50` and calibrated the wait/text gate, but eval loss worsened to `11.48`
and WLK inference still hallucinated frequent English/French pseudo-text. This
is now clearly a data/generalization problem rather than an emit-threshold
problem.

Fixing Qwen tokenizer word spacing removed the joined-word artifact, but WLK
inference still hallucinates frequent English/French tokens. The next quality
step is scale plus teacher distillation, not threshold tuning.

## Setup on Jarvislab H100

```bash
cd /Users/quentin/Documents/repos/qwen3-asr-streaming-h100
bash scripts/bootstrap_jarvislab.sh
source .venv/bin/activate
```

The bootstrap installs `qwen-asr`, `datasets`, `vllm[audio]`, and optionally FlashAttention.

## Baseline vLLM REST

Terminal 1:

```bash
source .venv/bin/activate
MODEL=Qwen/Qwen3-ASR-0.6B PORT=8000 bash scripts/serve_vllm_offline.sh
```

Terminal 2:

```bash
source .venv/bin/activate
python scripts/benchmark_transcriptions.py \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-ASR-0.6B \
  --audio-dir /Users/quentin/Downloads/mcif-long-trans/audio \
  --output-jsonl runs/baseline_wlk_audio_0.6b.jsonl
```

Repeat with `MODEL=Qwen/Qwen3-ASR-1.7B` for the teacher baseline.

## Baseline vLLM Realtime

Qwen3-ASR realtime is served through vLLM's `/v1/realtime` WebSocket path:

```bash
source .venv/bin/activate
MODEL=Qwen/Qwen3-ASR-0.6B PORT=8000 bash scripts/serve_vllm_realtime.sh
```

Then:

```bash
python scripts/benchmark_realtime_ws.py \
  --host localhost \
  --port 8000 \
  --model Qwen/Qwen3-ASR-0.6B \
  --audio-dir /Users/quentin/Downloads/mcif-long-trans/audio \
  --output-jsonl runs/realtime_wlk_audio_0.6b.jsonl \
  --chunk-ms 250
```

Treat this as a baseline, not a final quality target. vLLM's Qwen3-ASR realtime implementation may segment statelessly, so REST can remain the quality reference.

## Legacy Prefix-SFT Data

The prefix-SFT path is retained only as a baseline/legacy utility. It does not
solve the cache problem because it still uses Qwen3-ASR's offline audio layout.

```bash
python scripts/prepare_public_jsonl.py \
  --out-dir data/public_prefix_smoke \
  --sources fleurs_en fleurs_fr \
  --max-train-per-source 20 \
  --max-eval-per-source 5 \
  --prefix-mode
```

Larger H100 run:

```bash
python scripts/prepare_public_jsonl.py \
  --out-dir data/public_prefix \
  --sources fleurs_en fleurs_fr librispeech_clean_100 \
  --max-train-per-source 5000 \
  --max-eval-per-source 500 \
  --prefix-mode \
  --min-prefix-sec 2.0 \
  --prefix-stride-sec 1.0 \
  --right-context-sec 1.0
```

Outputs:

- `train.jsonl` / `eval.jsonl`: minimal official SFT format with `audio` and `text`.
- `train_manifest.jsonl` / `eval_manifest.jsonl`: same rows plus metadata for benchmarking.

The prefix labels are heuristic when word timestamps are absent: for a prefix ending at `t`, the target is the transcript fraction expected to be stable before `t - right_context`. Later, a forced-aligner pass can replace this with exact word-level labels without changing the SFT interface.

## Legacy Prefix-SFT Student

```bash
source .venv/bin/activate
MODEL_PATH=Qwen/Qwen3-ASR-0.6B \
TRAIN_FILE=data/public_prefix/train.jsonl \
EVAL_FILE=data/public_prefix/eval.jsonl \
OUTPUT_DIR=runs/qwen3-asr-0.6b-prefix-sft \
BATCH_SIZE=16 \
GRAD_ACC=4 \
LR=2e-5 \
EPOCHS=1 \
bash scripts/train_qwen3_sft.sh
```

The script clones `QwenLM/Qwen3-ASR` under `external/` if needed and calls its official `finetuning/qwen3_asr_sft.py`.

## Validate Legacy Student with vLLM

Serve the checkpoint:

```bash
MODEL=runs/qwen3-asr-0.6b-prefix-sft/checkpoint-200 \
PORT=8000 \
bash scripts/serve_vllm_offline.sh
```

Benchmark on held-out prefix examples:

```bash
python scripts/benchmark_transcriptions.py \
  --base-url http://localhost:8000/v1 \
  --model runs/qwen3-asr-0.6b-prefix-sft/checkpoint-200 \
  --manifest-jsonl data/public_prefix/eval_manifest.jsonl \
  --output-jsonl runs/student_prefix_eval.jsonl
```

Native realtime promotion gate before any Metal work:

- causal chunked encoder equals full causal reference;
- appending 1 s after 20 s does not recompute the first 20 s;
- WER/CER remains close to Qwen3-ASR offline on FR/EN;
- first stable word latency improves by roughly 30 percent;
- vLLM CUDA can serve long sessions without unbounded memory growth.

## Stage A+ Anti-Repetition Diagnostics

Realtime inference exposes experimental decoding controls, disabled by default:

```bash
python scripts/infer_realtime_checkpoint.py \
  --checkpoint runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume4k_lr1e5_v0 \
  --audio data/wlk_audio/myfXyntFYL_20s.wav \
  --emit-threshold 0.75 \
  --repetition-penalty 1.2 \
  --no-repeat-ngram-size 3 \
  --max-consecutive-text-tokens 12 \
  --json
```

Training also has optional anti-over-emission regularizers:

- `--emit-rate-loss-weight`: MSE between mean emit probability and the true text-frame rate.
- `--text-label-smoothing`: label smoothing on text-token CE only; wait frames remain unsmoothed.

Current H100 result: decoding constraints remove WLK trigram repetition on the
known 20 s sample, but they do not fix semantic ASR quality. The regularized
resume from `resume2k_lr2e5_v0` is not a keeper: it reduces emission rate but
misses the eval-loss gate.

## Stage A++ Data+Teacher Smoke

The Stage A++ smoke adds a mixed aligned dataset and teacher quality filter:

- sources: `fleurs_en`, `fleurs_fr`, `librispeech_clean_100`;
- audio cap: 16 s, long audio dropped;
- alignment: `Qwen/Qwen3-ForcedAligner-0.6B` only, failed rows skipped;
- teacher: `Qwen/Qwen3-ASR-1.7B` through vLLM REST;
- filter: keep public labels only when teacher annotation succeeds and
  `teacher_wer <= 0.35`.

New helpers:

- `scripts/make_audio_manifest.py`
- `scripts/annotate_teacher_transcripts.py`
- `scripts/filter_teacher_manifest.py`
- `scripts/eval_realtime_checkpoint.py`

Short H100 result:

```text
filtered train/eval: 3491 / 339 rows
teacher errors: 0 percent
heuristic alignments in final manifests: 0

checkpoint: runs/jl_417527/realtime_qwen_audio_surgery_stageAplus_mix_teacher_filter_smoke_v0
eval_loss: 4.9540
pred_text_ratio: 0.3173
label_text_ratio: 0.2565
WLK teacher-WER: 0.9961
WLK relative improvement vs resume2k: about 0.16 percent
```

Verdict: keep the data/teacher tooling, but do not promote the checkpoint. It
barely passes the eval-loss gate, over-emits on the mixed eval set, and does not
materially improve WLK teacher-WER. The next experiment should target the emit
gate and frame-label distribution before any longer run.

## WLK Chunked Diagnostics

The original WLK teacher-eval used full 277-425 s WAVs, while the current
Stage A models are trained on 16 s examples. The chunked diagnostic corrects
that mismatch:

- `scripts/slice_audio_manifest.py` slices manifest audio into short WAVs.
- `scripts/align_manifest_with_qwen.py` adds Qwen3 forced alignments to an
  existing manifest using `teacher_text` or another text field.
- `scripts/sweep_realtime_decoding.py` sweeps emit thresholds and decoding
  controls on one loaded checkpoint.

Result on 21 first-20s WLK chunks:

```text
resume2k, threshold 0.60:
  WER=0.9655
  text_ratio=0.0792

Stage A++ data+teacher smoke, best threshold 0.75:
  WER=0.9742
  text_ratio=0.0338
```

Result on 445 WLK 16 s teacher chunks:

```text
aligned chunks: 445 / 445, all qwen3_forced_aligner
held-out eval: 107 chunks from 5 parent videos

resume2k, threshold 0.60:
  WER=0.9787
  text_ratio=0.0667

WLK domain adapter wait_weight=0.5:
  eval_loss=5.1854
  pred_text_ratio=0.5605
  label_text_ratio=0.1932
  threshold 0.75 WER=1.2123

WLK domain adapter wait_weight=1.0:
  eval_loss=5.5479
  pred_text_ratio=0.4275
  label_text_ratio=0.1932
```

Verdict: the chunked WLK data is useful, but the current Stage A objective is
not. Short domain fine-tuning turns under-emission into over-emission and
repetition. The next useful model change should replace the current greedy
per-frame text CE plus BCE gate with a better blank/token alignment objective.

## References

- Qwen3-ASR fine-tuning JSONL and `torchrun`: <https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning>
- vLLM Qwen3-ASR REST serving recipe: <https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html>
- vLLM Realtime WebSocket protocol: <https://docs.vllm.ai/en/stable/examples/speech_to_text/realtime/>
- vLLM Qwen3-ASR realtime caveat to watch: <https://github.com/vllm-project/vllm/issues/35767>
- Voxtral-style realtime ASR reference: <https://arxiv.org/pdf/2602.11298>
