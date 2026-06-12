# Jarvislab Runs

## 2026-05-28 - Instance 417011, H100 IN2

Status: paused after run.

Remote workspace:

```text
/home/ubuntu/qwen3-asr-streaming-h100/qwen3-asr-streaming-h100
```

Local artifacts:

```text
runs/jl_417011/
```

### Environment

- GPU: NVIDIA H100 80GB HBM3
- Driver: 580.126.20
- Python: 3.10.12
- vLLM: `0.21.1rc1.dev359+gd6b48f928`
- Torch: `2.11.0+cu129`
- Qwen-ASR package: `qwen-asr==0.0.6`

Notes:

- `python3.10-venv` and `python3.10-dev` were missing on the VM and were installed with `apt`.
- `flash-attn` source install failed because the system CUDA detected by the build was 13.0 while vLLM/Torch used CUDA 12.9. The project bootstrap now skips standalone `flash-attn` by default.
- vLLM eager mode was required. Without `--enforce-eager`, Torch Inductor/Triton failed during server startup.

### REST Baselines

Dataset: 21 WAV files from `/Users/quentin/Downloads/mcif-long-trans/audio`, uploaded to the VM.

`Qwen/Qwen3-ASR-0.6B`, vLLM REST `/v1/audio/transcriptions`:

```json
{
  "count": 21,
  "ok": 21,
  "latency_p50": 1.5519600119998813,
  "latency_p95": 1.7457201429999714,
  "latency_mean": 1.557243340714281,
  "wer_mean": null
}
```

`Qwen/Qwen3-ASR-1.7B`, vLLM REST `/v1/audio/transcriptions`:

```json
{
  "count": 21,
  "ok": 21,
  "latency_p50": 1.576499947000002,
  "latency_p95": 1.8398378930000945,
  "latency_mean": 1.5858860170475986,
  "wer_mean": null
}
```

Observation: on this short WLK audio set, 0.6B and 1.7B REST latency are nearly identical on H100 in eager mode. Quality still needs reference transcripts or teacher comparison.

### Realtime Baseline

Server:

```bash
MODEL=Qwen/Qwen3-ASR-0.6B ENFORCE_EAGER=1 bash scripts/serve_vllm_realtime.sh
```

Architecture override in logs: `Qwen3ASRRealtimeGeneration`.

Result:

```json
{
  "count": 21,
  "ok": 20,
  "latency_p50": 7.434306622999884,
  "latency_p95": 12.23092574999987,
  "first_delta_p50": 1.0297667810000348,
  "first_delta_p95": 1.1811898259998088,
  "wer_mean": null,
  "revision_words_mean": 364.3
}
```

Failed file:

```text
ccpXHNfaoy.wav
```

Error:

```text
Qwen3ASRProcessor processing_error on a near-empty/silent audio chunk.
```

Notes:

- First delta latency is good, around 1s.
- Final Realtime completion is much slower than REST because the benchmark waits for the whole WebSocket finalization path.
- `revision_words_mean` is not meaningful yet without references and better event aggregation; the script currently treats full generated text as revision when no reference exists.

### Next Fixes

- Harden `benchmark_realtime_ws.py` to skip all-zero chunks and aggregate Realtime events according to vLLM's exact event schema.
- Add reference transcripts or compare 0.6B against 1.7B teacher outputs to compute WER/CER-like deltas.
- Split training and serving into separate venvs if Qwen's official SFT script needs Torch 2.12 while vLLM nightly pins Torch 2.11.
- Keep `--enforce-eager` as the default for baseline serving on this Jarvislab image.

## 2026-05-28 - Instance 417029, Native Realtime Prototype

Status: v0/v1 prototype completed; instance paused after artifact download.

Remote workspace:

```text
/home/ubuntu/qwen3-asr-streaming-h100/qwen3-asr-streaming-h100
```

Implemented and validated:

- Causal PyTorch audio encoder with append-only KV cache.
- 80 ms frame adapter.
- Cached decoder scaffold.
- Qwen tokenizer with `[P]` wait and `[W]` word-start tokens.
- `from_pretrained()` reload path.
- Greedy streaming inference CLI for WAV files.

GPU tests:

```text
13 passed in 0.95s
```

Smoke synthetic checkpoint:

```text
runs/realtime_smoke_native_v0
first_loss: 5.9592509269714355
last_loss: 0.026107419282197952
```

First real-audio checkpoint, unweighted:

```text
runs/realtime_fleurs_tiny_v0
train_examples: 96
eval_examples: 16
first_train_loss: 11.765230178833008
last_train_loss: 2.4520552158355713
eval_loss: 2.6229661256074905
```

This model collapsed to `[P]` at streaming inference, so it is retained only as
a diagnostic artifact.

First real-audio checkpoint, wait-weighted:

```text
runs/realtime_fleurs_tiny_v1_weighted
train_examples: 256
eval_examples: 32
wait_loss_weight: 0.05
valid_labels: 33458
wait_ratio: 0.5948054277004005
first_train_loss: 11.95308780670166
last_train_loss: 3.788281202316284
eval_loss: 4.651238292455673
```

Streaming reload check:

```text
checkpoint: runs/realtime_fleurs_tiny_v0
vocab_size: 151707
d_model: 192
audio_layers: 3
decoder_layers: 3
80 mel frames -> 10 decoder steps
```

Known limitation:

- The v1 model emits text tokens but quality is intentionally poor. It repeats
  frequent Qwen tokens such as `the` and `de` because the decoder is still
  trained from scratch on a tiny set with heuristic word timings. The next real
  step is Qwen decoder initialization plus real WhisperX/MFA word alignments.

## 2026-05-28 - Instance 417053, Qwen3 ForcedAligner Manifests

Status: Qwen3 forced-alignment path implemented and validated; instance paused
after artifact download.

Source used:

```text
Qwen/Qwen3-ForcedAligner-0.6B
```

The official Qwen package exposes `Qwen3ForcedAligner.align(audio=..., text=...,
language=...)`. It returned word-level timestamps in seconds on FLEURS examples.

Example alignment:

```json
[
  {"text": "bowen", "start_sec": 1.12, "end_sec": 1.44},
  {"text": "island", "start_sec": 1.44, "end_sec": 1.84},
  {"text": "is", "start_sec": 1.84, "end_sec": 2.08},
  {"text": "popular", "start_sec": 2.08, "end_sec": 2.56}
]
```

Artifacts:

```text
data/qwen_aligned_fleurs_tiny/train_manifest.jsonl
data/qwen_aligned_fleurs_tiny/eval_manifest.jsonl
runs/realtime_fleurs_qwen_aligned_v0
```

Alignment run:

```text
sources: fleurs_en, fleurs_fr
train_records: 256
eval_records: 32
speed after warmup: roughly 20-26 utterances/sec on H100
```

Training run:

```text
runs/realtime_fleurs_qwen_aligned_v0
train_steps: 500
train_examples: 256
eval_examples: 32
valid_labels: 33458
wait_labels: 19631
text_or_word_labels: 13827
wait_ratio: 0.586735608823002
wait_loss_weight: 0.05
first_train_loss: 11.883075714111328
last_train_loss: 3.7703630924224854
eval_loss: 4.720618575811386
```

Conclusion:

- Qwen3-ForcedAligner works well enough to replace heuristic timestamps.
- On this tiny run, quality did not materially improve because the decoder is
  still trained from scratch. The next bottleneck is Qwen3-ASR 0.6B decoder /
  token embedding / LM head initialization, not timestamp quality.

## 2026-05-28 - Instance 417191, Longer Rank-32 LoRA FLEURS Run

Status: completed; lightweight artifacts downloaded locally. Full `model.pt`
remains on the H100 because the checkpoint is not quality-usable.

Remote workspace:

```text
/home/ubuntu/qwen3-asr-streaming-h100/qwen3-asr-streaming-h100
```

Local lightweight artifacts:

```text
runs/jl_417191/realtime_qwen_lora_r32_fleurs_16s_space_16k_v0/
runs/jl_417191/realtime_qwen_lora_r32_fleurs_16s_space_16k_v0.log
```

Command:

```bash
python scripts/train_realtime_tiny_asr.py \
  --decoder-backend qwen \
  --qwen-decoder-model Qwen/Qwen3-ASR-0.6B \
  --output-dir runs/realtime_qwen_lora_r32_fleurs_16s_space_16k_v0 \
  --train-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/eval_manifest.jsonl \
  --steps 16000 \
  --batch-size 2 \
  --device cuda \
  --audio-layers 4 \
  --audio-heads 8 \
  --emit-gate-loss-weight 1.0 \
  --emit-gate-wait-weight 1.5 \
  --emit-threshold 0.50 \
  --qwen-lora-rank 32 \
  --qwen-lora-alpha 64 \
  --qwen-lora-dropout 0.05 \
  --lr 5e-5 \
  --no-word-start-token \
  --max-audio-sec 16 \
  --log-every 500
```

Metrics:

```json
{
  "train_steps": 16000,
  "train_examples": 2400,
  "eval_examples": 240,
  "trainable_params": 63842305,
  "first_train_loss": 18.73767852783203,
  "last_train_loss": 0.49518877267837524,
  "eval_loss": 11.47832874059677,
  "pred_wait_ratio": 0.7601311615639591,
  "label_wait_ratio": 0.7639932993548847
}
```

WLK 20 s threshold sweep on `data/wlk_audio/myfXyntFYL_20s.wav`:

```text
0.40: careeray som many any a gljing alleyailylogy linear...
0.50: careeray som many any a gljing alleyaily linear...
0.65: careeray som many any a gljing alley linear...
0.80: careeray som many any a gljing alley linear...
```

Train-sample free streaming sanity check:

```text
reference:  whistler 1.5 hour drive from vancouver is expensive but well-known because of the 2010 winter olympics
hypothesis: whistler 1 but drive vancouver culture culture is is expensive but wellknownamount because of the 2010 ten winter olympics
```

Eval-sample and WLK outputs remain unusable.

Conclusion:

- More training on the 2400-example FLEURS manifest makes the model memorize
  train examples, but it does not generalize.
- The separate emit gate is calibrated after the longer run: predicted wait/text
  ratios now match label ratios closely. The blocker is content quality, not
  threshold selection.
- Next run should scale data before capacity: all available FLEURS plus
  LibriSpeech/MLS/Common Voice with Qwen3-ForcedAligner, teacher transcripts
  from Qwen3-ASR-1.7B, and periodic eval/checkpointing to avoid selecting an
  overfit late step.

## 2026-05-28 - Instance 417168, Larger FLEURS LoRA Runs

Status: completed; instance paused after artifact download.

Remote workspace:

```text
/home/ubuntu/qwen3-asr-streaming-h100/qwen3-asr-streaming-h100
```

Local lightweight artifacts:

```text
runs/jl_417168/
data/qwen_aligned_fleurs_16s_v0/{train_manifest.jsonl,eval_manifest.jsonl}
```

Full `model.pt` checkpoints remain on the paused H100 instance because each is
about 1.3 GB.

### Code fixes

- `prepare_qwen_aligned_jsonl.py` now supports `--drop-long-audio`, so long
  utterances are skipped instead of truncating audio while keeping the full
  transcript.
- Dataset audio is loaded with `datasets.Audio(decode=False)` plus `soundfile`,
  avoiding the optional `torchcodec` dependency on the Jarvislab image.
- `realtime_targets.py` now encodes non-initial words with a leading space. This
  is required for Qwen's tokenizer; encoding each aligned word independently
  produced joined-word targets and joined-word hypotheses.

Validation:

```text
local:  9 passed, 1 skipped
H100:   14 passed
```

### Manifest

Command:

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

Result:

```json
{
  "train_records": 2400,
  "eval_records": 240,
  "alignment_source": "qwen3_forced_aligner",
  "languages": {"train": {"en": 1200, "fr": 1200}, "eval": {"en": 120, "fr": 120}},
  "train_avg_duration_sec": 10.282,
  "train_max_duration_sec": 15.96,
  "eval_avg_duration_sec": 9.405,
  "eval_max_duration_sec": 15.48
}
```

### LoRA run with old word-token targets

Checkpoint:

```text
runs/realtime_qwen_lora_r16_fleurs_16s_v0
```

Metrics:

```json
{
  "train_steps": 4000,
  "train_examples": 2400,
  "eval_examples": 240,
  "first_train_loss": 19.37409210205078,
  "last_train_loss": 5.621718406677246,
  "eval_loss": 7.665455977121989,
  "pred_wait_ratio": 0.6734504758170866,
  "label_wait_ratio": 0.7237872908721531
}
```

WLK 20 s inference was not usable and exposed the target bug:

```text
peopleherunonthatinisisofbethatthatolywaylyarchly...
```

### LoRA run with leading-space token targets

Checkpoint:

```text
runs/realtime_qwen_lora_r16_fleurs_16s_space_v0
```

Metrics:

```json
{
  "train_steps": 4000,
  "train_examples": 2400,
  "eval_examples": 240,
  "first_train_loss": 18.869348526000977,
  "last_train_loss": 6.341559410095215,
  "eval_loss": 8.56280548175176,
  "pred_wait_ratio": 0.6428342303168549,
  "label_wait_ratio": 0.7639932993548847
}
```

WLK 20 s inference at threshold 0.50:

```text
paredver were run than in is the may the that had a the easy muchchesterature...
```

Threshold sweep:

```text
0.50: long hallucinated pseudo-English
0.60: shorter but still wrong
0.70: shorter but still wrong
0.80: "pared that hadchester the the used for is wrong work bombing the"
0.90: "for is"
```

Conclusion:

- The native causal pipeline is now end-to-end trainable with Qwen decoder
  initialization, LoRA, Qwen3 forced alignments, and correct Qwen word-spacing
  targets.
- The model is not quality-usable yet. The current audio encoder and adapter are
  still learning the acoustic-to-Qwen interface from scratch on only 2400 short
  examples. Threshold tuning cannot fix the content errors.
- Next useful step is scale and distillation: tens of thousands of aligned
  examples, teacher transcripts/logits from Qwen3-ASR-1.7B, then a longer
  staged run that first trains the causal audio path and only then expands LoRA
  or unfreezes decoder blocks.

## 2026-05-28 - Instance 417067, Qwen Decoder Initialization

Status: Qwen decoder backend implemented and tested; instance paused after run
inspection and artifact download.

Remote workspace:

```text
/home/ubuntu/qwen3-asr-streaming-h100/qwen3-asr-streaming-h100
```

Environment note:

- The project `.venv` is a symlink to the Jarvislab base venv:
  `/home/ubuntu/wlk-jarvis-min.Ckv6m0/.venv`.
- CUDA check after repair: `torch 2.11.0+cu130`, CUDA available.
- Remote tests after repair: `13 passed in 1.25s`.

Implemented:

- `Qwen3ASRRealtimeQwenDecoderModel`, a realtime causal model that keeps the
  local causal audio encoder and 80 ms adapter, then drives Qwen3-ASR's
  pretrained text decoder through `inputs_embeds`.
- Qwen3-ASR 0.6B text hidden-size discovery from HF config.
- `load_realtime_model()` dispatch for native vs Qwen-decoder checkpoints.
- Training flags:
  - `--decoder-backend qwen`
  - `--qwen-decoder-model`
  - `--qwen-dtype`
  - `--freeze-qwen-layers`
  - `--freeze-qwen-all`
  - `--no-word-start-token`
- Inference reload path for both native and Qwen-decoder checkpoints.

Primary Qwen decoder run:

```text
runs/realtime_fleurs_qwen_decoder_v0
train_steps: 500
train_examples: 256
eval_examples: 32
valid_labels: 33458
wait_labels: 19631
text_or_word_labels: 13827
wait_ratio: 0.586735608823002
wait_loss_weight: 0.05
decoder_backend: qwen
total_params: 629218304
trainable_params: 188750848
first_train_loss: 18.125
last_train_loss: 4.75
eval_loss: 5.779296875
```

Eval diagnostic:

```text
pred_wait: 391
pred_word: 3160
pred_text: 0
label_wait: 2090
label_word: 562
label_text: 899
valid: 3551
```

Conclusion: with `[W]` enabled, the model learns the special word-start marker
shortcut and emits no text.

No-word-start run:

```text
runs/realtime_fleurs_qwen_decoder_no_w_v0
train_steps: 1000
wait_loss_weight: 0.05
include_word_start: false
first_train_loss: 17.625
last_train_loss: 6.1875
eval_loss: 7.9765625
```

Eval diagnostic:

```text
pred_wait: 3551
pred_word: 0
pred_text: 0
label_wait: 2645
label_word: 0
label_text: 906
valid: 3551
```

Conclusion: without `[W]`, the same objective collapses to pure wait tokens.

Text-only diagnostic:

```text
runs/realtime_fleurs_qwen_decoder_text_only_v0
train_steps: 1000
wait_loss_weight: 0.0
include_word_start: false
first_train_loss: 17.25
last_train_loss: 6.65625
eval_loss: 8.78515625
```

Eval diagnostic:

```text
pred_wait: 0
pred_word: 0
pred_text: 3551
label_wait: 2645
label_word: 0
label_text: 906
valid: 3551
```

WLK 20s streaming reload check:

```text
checkpoint: runs/realtime_fleurs_qwen_decoder_text_only_v0
audio: data/wlk_audio/myfXyntFYL_20s.wav
tokens: 249
audio_frames_seen: 1998
decoder_steps_seen: 249
chunk_ms: 320
hypothesis: repeated "the/un/s" token pattern
```

Conclusion:

- The causal model exists, loads, saves, reloads, and streams with append-only
  audio frames and decoder cache.
- Initializing the Qwen text decoder is necessary but not sufficient. With the
  Qwen transformer frozen, the new causal audio encoder/adapter cannot yet map
  audio frames into the pretrained text manifold.
- The immediate next training step is to unfreeze at least the last Qwen decoder
  blocks and use a better objective/schedule that avoids both wait collapse and
  text-every-frame repetition. The architecture work should continue before any
  vLLM or vllm-metal port.

## 2026-05-28 - Instance 417126, Emit/Wait Gate

Status: Qwen decoder emit-gate training completed; artifacts downloaded
locally; instance paused after run.

Remote workspace:

```text
/home/ubuntu/qwen3-asr-streaming-h100/qwen3-asr-streaming-h100
```

Implemented:

- Separate binary emit/wait gate on top of Qwen decoder hidden states.
- Text CE only on non-wait frames when the gate is active.
- `--train-qwen-last-n-layers` to unfreeze the last Qwen decoder blocks while
  keeping the rest frozen.
- Eval prediction stats for wait/text ratios.
- `infer_realtime_checkpoint.py --emit-threshold` for post-training threshold
  sweeps.

Remote tests after sync:

```text
13 passed
```

Smoke diagnostics:

```text
runs/realtime_qwen_balanced_smoke
loss_mode: balanced
train_qwen_last_n_layers: 4
pred_wait: 0
pred_text: 3551
```

```text
runs/realtime_qwen_emit_gate_smoke
steps: 80
emit_gate_wait_weight: 1.0
pred_wait: 995
pred_text: 2556
label_wait: 2645
label_text: 906
```

```text
runs/realtime_qwen_emit_gate_w3_smoke
steps: 120
emit_gate_wait_weight: 3.0
pred_wait: 3551
pred_text: 0
```

```text
runs/realtime_qwen_emit_gate_w15_smoke
steps: 160
emit_gate_wait_weight: 1.5
pred_wait: 2575
pred_text: 976
label_wait: 2645
label_text: 906
```

Primary run:

```text
runs/realtime_qwen_emit_gate_w15_v0
train_steps: 2000
train_examples: 256
eval_examples: 32
decoder_backend: qwen
total_params: 629219329
trainable_params: 251676673
emit_gate_loss_weight: 1.0
emit_gate_wait_weight: 1.5
train_qwen_last_n_layers: 4
first_train_loss: 19.1931095123291
last_train_loss: 6.365322589874268
eval_loss: 10.308069229125977
loss_decreased: true
```

Default-threshold eval stats:

```text
threshold: 0.50
valid: 3551
pred_wait: 3202
pred_text: 349
label_wait: 2645
label_text: 906
pred_wait_ratio: 0.9017
```

Threshold sweep on eval:

```text
threshold 0.25 -> pred_wait_ratio 0.6745, pred_text_ratio 0.3255
threshold 0.30 -> pred_wait_ratio 0.7308, pred_text_ratio 0.2692
threshold 0.35 -> pred_wait_ratio 0.7840, pred_text_ratio 0.2160
label_wait_ratio: 0.7449
```

WLK 20s streaming reload check at threshold 0.30:

```text
checkpoint: runs/realtime_qwen_emit_gate_w15_v0
audio: data/wlk_audio/myfXyntFYL_20s.wav
tokens: 249
audio_frames_seen: 1998
decoder_steps_seen: 249
chunk_ms: 320
emit_threshold: 0.30
hypothesis: repeated "the/ed/country" token pattern
```

Conclusion:

- The gate fixes the worst objective problem: wait/text emission is now
  tunable instead of collapsing to pure wait or pure text.
- The causal streaming invariant still holds: appending audio advances the
  cached audio encoder and decoder by new frames/steps only.
- ASR quality is not usable yet. The output remains high-frequency Qwen token
  repetition, which means the new audio encoder/adapter has not learned a good
  bridge into the Qwen text decoder manifold from this tiny FLEURS run.
- Next useful step: train on a much larger aligned manifest, add teacher
  distillation from Qwen3-ASR-1.7B logits/transcripts, and consider gradually
  unfreezing more Qwen blocks only after the gate remains stable.

## 2026-05-28 - Instance 417149, Qwen LoRA Adapters

Status: LoRA training path implemented; rank 16 checkpoint and lightweight
artifacts downloaded locally; instance paused after run.

Remote workspace:

```text
/home/ubuntu/qwen3-asr-streaming-h100/qwen3-asr-streaming-h100
```

Implemented:

- Local `LoRALinear` wrapper for `nn.Linear`, without adding a PEFT dependency.
- Qwen decoder LoRA injection on `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, and `down_proj`.
- LoRA checkpoint metadata so `load_realtime_model()` rebuilds wrappers before
  loading the saved state dict.
- Training flags:
  - `--qwen-lora-rank`
  - `--qwen-lora-alpha`
  - `--qwen-lora-dropout`
  - `--qwen-lora-targets`

Tests:

```text
local: 9 passed, 1 skipped
remote H100: 14 passed
```

Rank 8 smoke:

```text
runs/realtime_qwen_lora_r8_emit_gate_smoke
steps: 300
qwen_lora_rank: 8
qwen_lora_alpha: 16
qwen_lora_dropout: 0.05
total_params: 634265601
trainable_params: 38215681
first_train_loss: 19.067916870117188
last_train_loss: 8.060051918029785
eval_loss: 8.822314441204071
pred_wait: 1654
pred_text: 1897
label_wait: 2645
label_text: 906
pred_wait_ratio: 0.4658
```

Threshold sweep on rank 8 showed the gate needs a higher threshold after short
training:

```text
threshold 0.65 -> pred_wait_ratio 0.6961, pred_text_ratio 0.3039
threshold 0.70 -> pred_wait_ratio 0.8130, pred_text_ratio 0.1870
label_wait_ratio: 0.7449
```

Rank 16 run:

```text
runs/realtime_qwen_lora_r16_emit_gate_v0
steps: 2000
qwen_lora_rank: 16
qwen_lora_alpha: 32
qwen_lora_dropout: 0.05
total_params: 639311873
trainable_params: 43261953
first_train_loss: 18.60001564025879
last_train_loss: 2.491319179534912
eval_loss: 12.86189091205597
pred_wait: 2679
pred_text: 872
label_wait: 2645
label_text: 906
pred_wait_ratio: 0.7544
```

WLK 20s streaming reload check at threshold 0.50:

```text
checkpoint: runs/realtime_qwen_lora_r16_emit_gate_v0
audio: data/wlk_audio/myfXyntFYL_20s.wav
tokens: 249
audio_frames_seen: 1998
decoder_steps_seen: 249
chunk_ms: 320
emit_threshold: 0.50
hypothesis: mixed repeated pseudo-text, still not usable ASR
```

Conclusion:

- LoRA is the right adaptation mechanism for the pretrained Qwen decoder side:
  it cuts trainable parameters from about 251.7M in the last-4-layer unfreeze run
  to about 43.3M with rank 16 while keeping the gate well calibrated.
- LoRA on the causal audio embedder itself is not meaningful yet because that
  embedder is new, not a pretrained module being adapted. The audio encoder and
  frame adapter should continue training directly.
- Quality is still bad. The model overfits the tiny 256-example FLEURS manifest
  and emits pseudo-text rather than transcription. The next limiting factor is
  a much larger aligned/distilled dataset, not more decoder capacity.

## 2026-05-28 - Instance 417223, Qwen Audio Surgery Backend

Status: `qwen_audio_surgery` backend implemented and smoke-tested on H100.

Implemented:

- `QwenAudioSurgeryEncoder`, a streaming wrapper around Qwen3-ASR's pretrained
  `thinker.audio_tower`.
- Bounded-window audio recompute with default `left_context=15s` and
  `right_context=640ms`, plus a `strict_causal` diagnostic option.
- `QwenAudioSurgeryFrameAdapter`, identity-initialized when Qwen audio output
  dim matches the text hidden size (`1024 -> 1024`).
- `Qwen3ASRRealtimeQwenAudioSurgeryModel`, sharing the existing Qwen decoder,
  emit-gate, LoRA metadata, checkpoint save/load, and streaming contract.
- Training flags:
  - `--decoder-backend qwen_audio_surgery`
  - `--qwen-audio-right-context-ms`
  - `--qwen-audio-left-context-sec`
  - `--qwen-audio-strict-causal`
  - `--freeze-qwen-audio`
  - `--train-qwen-audio-last-n-layers`

Tests:

```text
local: 17 passed
remote H100: 17 passed
```

Real Qwen shape smoke:

```text
model: Qwen/Qwen3-ASR-0.6B
audio_hidden_shape: [1, 26, 1024] for 200 mel frames
hidden_shape: [1, 26, 1024]
stream chunks 80/80/40 mel frames -> token steps 2, 10, 5
```

Bounded recompute smoke:

```text
left_context: 1500 mel frames
right_context: 64 mel frames
max_recompute_frames: 1564
after encode 20s: recomputed 1564 frames, not 2000
after append +1s: recomputed 1564 frames, not 2100
emitted_steps: 242 -> 254
```

Training smoke:

```text
runs/jl_417223/realtime_qwen_audio_surgery_smoke
train examples: 8
eval examples: 4
steps: 3
freeze_qwen_all: true
freeze_qwen_audio: true
trainable_params: 1049601
first_train_loss: 13.65472412109375
last_train_loss: 11.640889167785645
eval_loss: 10.684858322143555
loss_decreased: true
```

Checkpoint reload check:

```text
checkpoint: runs/jl_417223/realtime_qwen_audio_surgery_smoke
audio_frames_seen: 1114
decoder_steps_seen: 131
chunk_ms: 320
hypothesis: repeated pseudo-text, expected after a 3-step smoke
```

Conclusion:

- This is the right v2 direction: it no longer starts the audio embedder from
  scratch. It reuses the pretrained Qwen audio tower and only adds a small,
  identity-initialized projector/gate path for the first training stage.
- The current implementation is bounded-window surgery, not yet a per-layer KV
  cache inside Qwen audio attention. That is deliberate for Stage 0: it proves
  the reuse/wiring and eliminates full-history re-embedding before deeper model
  surgery.
- Next step: compare chunked bounded-window output against an offline masked
  reference, then run a longer Stage A with frozen Qwen decoder/audio tower
  before trying LoRA on the last Qwen audio blocks.

## 2026-05-28 - Instance 417230, Stage 0.5 + Stage A

Status: Stage 0.5 validation and a short Stage A training run completed.

Stage 0.5 validation:

```text
script: scripts/validate_qwen_audio_surgery.py
audio: 5 WLK wavs copied to data/wlk_audio_stage05
max_audio_sec: 30
chunk_ms: 320
left_context_sec: 15
right_context_ms: 640
max_allowed_recompute_frames: 1564
max_observed_recompute_frames: 1564
all_recompute_bounded: true
mean_stream_chunk_ms: 6.733
max_stream_chunk_ms: 59.711
mean_peak_mem_gb: 1.846
artifacts: runs/jl_417230/stage05/
```

Stage A command:

```bash
python scripts/train_realtime_tiny_asr.py \
  --decoder-backend qwen_audio_surgery \
  --qwen-decoder-model Qwen/Qwen3-ASR-0.6B \
  --qwen-dtype bfloat16 \
  --output-dir runs/jl_417230/realtime_qwen_audio_surgery_stageA_1k_v0 \
  --train-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/eval_manifest.jsonl \
  --steps 1000 \
  --batch-size 2 \
  --device cuda \
  --audio-heads 8 \
  --qwen-audio-left-context-sec 15 \
  --qwen-audio-right-context-ms 640 \
  --freeze-qwen-all \
  --freeze-qwen-audio \
  --emit-gate-loss-weight 1.0 \
  --emit-gate-wait-weight 1.5 \
  --lr 1e-4 \
  --no-word-start-token \
  --max-audio-sec 16
```

Stage A metrics:

```text
checkpoint: runs/jl_417230/realtime_qwen_audio_surgery_stageA_1k_v0
train_examples: 2400
eval_examples: 240
trainable_params: 1049601
total_params: 783475713
first_train_loss: 18.235950469970703
last_train_loss: 4.23997688293457
eval_loss: 5.25618989666303
label_wait_ratio: 0.763998
pred_wait_ratio: 0.731867
loss_decreased: true
```

Qualitative eval:

```text
FLEURS hypotheses are still repeated pseudo-text, but much less degenerate than
the scratch-audio runs. The gate learned a plausible wait/text ratio.
```

WLK 20s threshold sweep:

```text
threshold 0.50 -> "and then we have the second part of the story ..."
threshold 0.65 -> "and the other two are the two most important things ..."
threshold 0.75 -> "and the other two are isthe"
tokens: 241
audio_frames_seen: 1998
decoder_steps_seen: 241
```

Conclusion:

- Stage 0.5 passes: the backend is stable on multiple WLK wavs and never
  recomputes beyond the configured local audio window.
- Stage A learns the gate and lowers loss with only the projector/gate trainable.
  Quality is still not usable ASR. This means frozen audio tower + frozen
  decoder is too restrictive for text accuracy, but the pretrained audio tower
  direction is healthier than the scratch encoder.
- Next useful run: Stage B with Qwen audio tower mostly frozen but last 2-4
  audio layers unfrozen, or LoRA on Qwen audio attention/MLP. Keep Qwen text
  decoder frozen initially to isolate audio adaptation.

## 2026-05-28 - Instance 417231, Stage B Probe + Longer Stage A

Status: Stage B was tried and was trainable, but it did not clearly beat frozen
Stage A. A longer Stage A run improved the eval loss more cleanly.

Stage B smoke:

```text
checkpoint: runs/jl_417231/realtime_qwen_audio_surgery_stageB_smoke
trainable_params: 41355393
audio layers unfrozen: last 4 Qwen audio blocks
first_train_loss: 18.53
last_train_loss: 10.96
eval_loss: 10.84
result: backward/reload path works, no OOM on H100
```

Stage B full run:

```bash
python scripts/train_realtime_tiny_asr.py \
  --decoder-backend qwen_audio_surgery \
  --qwen-decoder-model Qwen/Qwen3-ASR-0.6B \
  --qwen-dtype bfloat16 \
  --output-dir runs/jl_417231/realtime_qwen_audio_surgery_stageB_audio4_1500_v0 \
  --train-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/eval_manifest.jsonl \
  --steps 1500 \
  --batch-size 2 \
  --grad-acc 1 \
  --device cuda \
  --audio-heads 8 \
  --qwen-audio-left-context-sec 15 \
  --qwen-audio-right-context-ms 640 \
  --freeze-qwen-all \
  --train-qwen-audio-last-n-layers 4 \
  --emit-gate-loss-weight 1.0 \
  --emit-gate-wait-weight 1.5 \
  --emit-threshold 0.5 \
  --lr 2e-5 \
  --no-word-start-token \
  --max-audio-sec 16
```

Stage B metrics:

```text
checkpoint: runs/jl_417231/realtime_qwen_audio_surgery_stageB_audio4_1500_v0
train_examples: 2400
eval_examples: 240
trainable_params: 41355393
first_train_loss: 18.235950469970703
last_train_loss: 4.8441386222839355
eval_loss: 5.242083154122034
pred_wait_ratio: 0.687458
pred_text_ratio: 0.312542
loss_decreased: true
```

Stage B WLK 20s threshold sweep:

```text
threshold 0.50 -> "le 1er janvier 1990 est le jour de l'annivers de la création le"
threshold 0.65 -> "les deux parties ont été enfin enfin enfin"
threshold 0.75 -> "les"
tokens: 241
audio_frames_seen: 1998
decoder_steps_seen: 241
```

Longer Stage A run:

```bash
python scripts/train_realtime_tiny_asr.py \
  --decoder-backend qwen_audio_surgery \
  --qwen-decoder-model Qwen/Qwen3-ASR-0.6B \
  --qwen-dtype bfloat16 \
  --output-dir runs/jl_417231/realtime_qwen_audio_surgery_stageA_4000_v1 \
  --train-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/eval_manifest.jsonl \
  --steps 4000 \
  --batch-size 2 \
  --grad-acc 1 \
  --device cuda \
  --audio-heads 8 \
  --qwen-audio-left-context-sec 15 \
  --qwen-audio-right-context-ms 640 \
  --freeze-qwen-all \
  --freeze-qwen-audio \
  --emit-gate-loss-weight 1.0 \
  --emit-gate-wait-weight 1.5 \
  --emit-threshold 0.5 \
  --lr 1e-4 \
  --no-word-start-token \
  --max-audio-sec 16
```

Longer Stage A metrics:

```text
checkpoint: runs/jl_417231/realtime_qwen_audio_surgery_stageA_4000_v1
train_examples: 2400
eval_examples: 240
trainable_params: 1049601
first_train_loss: 18.235950469970703
last_train_loss: 4.132987976074219
eval_loss: 5.046820811430613
pred_wait_ratio: 0.710518
pred_text_ratio: 0.289482
loss_decreased: true
```

Longer Stage A WLK 20s threshold sweep:

```text
threshold 0.50 -> "the 1990 census showed that the city had 1,000 people ..."
threshold 0.65 -> "the 100 most popular songs were released in 200 2 2"
threshold 0.75 -> "the new new the"
tokens: 241
audio_frames_seen: 1998
decoder_steps_seen: 241
```

Comparison:

```text
Stage A 1k eval_loss:       5.25618989666303
Stage B audio4 eval_loss:   5.242083154122034
Stage A 4k eval_loss:       5.046820811430613
```

Conclusion:

- Stage B is mechanically valid, but opening 41M Qwen audio parameters did not
  provide a clear win in this data/step regime.
- Continuing Stage A was the better move: the frozen Qwen audio tower plus small
  projector/gate still had undertrained capacity.
- ASR quality remains unusable and repetitive. The useful signal is that the
  causalized Qwen audio path trains stably, keeps bounded recompute, and now has
  a stronger frozen-tower baseline.
- Recommended next step: improve Stage A capacity before unfreezing Qwen audio
  again, e.g. a residual MLP/projector adapter or checkpoint-resumed Stage A,
  then retry Stage B only after the adapter baseline plateaus.

## 2026-05-28 - Instance 417274, Stage A+ Residual Adapter

Status: Stage A was reinforced with trainable residual adapter blocks while
keeping Qwen audio and Qwen text frozen.

Code changes:

- `RealtimeAudioConfig` now has optional Qwen audio adapter settings:
  `qwen_audio_adapter_hidden_dim`, `qwen_audio_adapter_layers`,
  `qwen_audio_adapter_dropout`, and `qwen_audio_adapter_residual_scale`.
- `QwenAudioSurgeryFrameAdapter` still defaults to the old identity-initialized
  projection. When enabled, it adds RMSNorm + SwiGLU residual blocks after the
  projection.
- `scripts/train_realtime_tiny_asr.py` exposes the new flags. Old commands are
  unchanged because the default adapter layer count is zero.

Validation:

```text
local tests: 9 passed, 1 skipped
H100 tests: 9 passed
```

Stage A+ smoke:

```text
checkpoint: runs/jl_417274/realtime_qwen_audio_surgery_stageAplus_smoke_adapter2x2048_v0
adapter: 2 residual SwiGLU blocks, hidden_dim 2048, residual_scale 0.1
trainable_params: 13634561
first_train_loss: 11.991044998168945
last_train_loss: 9.906524658203125
eval_loss: 9.801168402036032
result: backward/reload path works, no OOM on H100
```

Stage A+ full run:

```bash
python scripts/train_realtime_tiny_asr.py \
  --decoder-backend qwen_audio_surgery \
  --qwen-decoder-model Qwen/Qwen3-ASR-0.6B \
  --qwen-dtype bfloat16 \
  --output-dir runs/jl_417274/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_4000_v0 \
  --train-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/eval_manifest.jsonl \
  --steps 4000 \
  --batch-size 2 \
  --grad-acc 1 \
  --device cuda \
  --audio-heads 8 \
  --qwen-audio-left-context-sec 15 \
  --qwen-audio-right-context-ms 640 \
  --qwen-audio-adapter-hidden-dim 2048 \
  --qwen-audio-adapter-layers 2 \
  --qwen-audio-adapter-residual-scale 0.1 \
  --freeze-qwen-all \
  --freeze-qwen-audio \
  --emit-gate-loss-weight 1.0 \
  --emit-gate-wait-weight 1.5 \
  --emit-threshold 0.5 \
  --lr 5e-5 \
  --no-word-start-token \
  --max-audio-sec 16
```

Stage A+ metrics:

```text
checkpoint: runs/jl_417274/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_4000_v0
train_examples: 2400
eval_examples: 240
trainable_params: 13634561
first_train_loss: 11.991044998168945
last_train_loss: 4.548867225646973
eval_loss: 4.998533650239309
pred_wait_ratio: 0.743344
pred_text_ratio: 0.256656
loss_decreased: true
```

WLK 20s threshold sweep:

```text
threshold 0.50 -> "the new rules which were introduced in 2014 by the european councilthe new rules which were"
threshold 0.65 -> "the new law the new law"
threshold 0.75 -> "we're here to help you youwe're here to help you"
tokens: 241
audio_frames_seen: 1998
decoder_steps_seen: 241
```

Comparison:

```text
Stage A 1k eval_loss:         5.25618989666303
Stage B audio4 eval_loss:     5.242083154122034
Stage A 4k eval_loss:         5.046820811430613
Stage A+ 2x2048 4k eval_loss: 4.998533650239309
```

Conclusion:

- Reinforcing A did help: A+ is the best eval loss so far without unfreezing
  Qwen audio.
- The improvement is real but small, and qualitative ASR is still not usable.
  Repetition and language drift remain.
- Next useful experiment is not another shallow Stage B immediately. Better
  candidates are a larger adapter variant, an adapter LR sweep, or a checkpoint
  resume path so Stage A+ can continue training without restarting from scratch.

## 2026-05-28 - Instance 417339, Stage A+ Resume

Status: checkpoint resume was implemented and used to continue the best A+
adapter run.

Code changes:

- `scripts/train_realtime_tiny_asr.py` now supports
  `--resume-from-checkpoint CHECKPOINT_DIR`.
- Resume loads `realtime_config.json`, model weights, and the checkpoint
  tokenizer if available.
- Optimizer state is not restored. The intended use is to resume weights with
  an explicit lower LR.
- Freeze flags must be passed again, because `requires_grad` is runtime state
  and is not stored in `model.pt`.

Validation:

```text
local tests: 9 passed, 1 skipped
py_compile: scripts/train_realtime_tiny_asr.py OK
H100 py_compile: scripts/train_realtime_tiny_asr.py OK
```

Resume +2k command:

```bash
python scripts/train_realtime_tiny_asr.py \
  --resume-from-checkpoint runs/jl_417274/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_4000_v0 \
  --decoder-backend qwen_audio_surgery \
  --qwen-decoder-model Qwen/Qwen3-ASR-0.6B \
  --qwen-dtype bfloat16 \
  --output-dir runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume2k_lr2e5_v0 \
  --train-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/eval_manifest.jsonl \
  --steps 2000 \
  --batch-size 2 \
  --grad-acc 1 \
  --device cuda \
  --freeze-qwen-all \
  --freeze-qwen-audio \
  --emit-gate-loss-weight 1.0 \
  --emit-gate-wait-weight 1.5 \
  --emit-threshold 0.5 \
  --lr 2e-5 \
  --no-word-start-token \
  --max-audio-sec 16
```

Resume +2k metrics:

```text
checkpoint: runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume2k_lr2e5_v0
trainable_params: 13634561
first_train_loss: 4.355401039123535
last_train_loss: 6.171995162963867
eval_loss: 4.882986927032471
pred_wait_ratio: 0.729551
pred_text_ratio: 0.270449
```

Resume +4k command:

```bash
python scripts/train_realtime_tiny_asr.py \
  --resume-from-checkpoint runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume2k_lr2e5_v0 \
  --decoder-backend qwen_audio_surgery \
  --qwen-decoder-model Qwen/Qwen3-ASR-0.6B \
  --qwen-dtype bfloat16 \
  --output-dir runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume4k_lr1e5_v0 \
  --train-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/eval_manifest.jsonl \
  --steps 2000 \
  --batch-size 2 \
  --grad-acc 1 \
  --device cuda \
  --freeze-qwen-all \
  --freeze-qwen-audio \
  --emit-gate-loss-weight 1.0 \
  --emit-gate-wait-weight 1.5 \
  --emit-threshold 0.5 \
  --lr 1e-5 \
  --no-word-start-token \
  --max-audio-sec 16
```

Resume +4k metrics:

```text
checkpoint: runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume4k_lr1e5_v0
trainable_params: 13634561
first_train_loss: 4.2444634437561035
last_train_loss: 5.983463287353516
eval_loss: 4.856786131858826
pred_wait_ratio: 0.697794
pred_text_ratio: 0.302206
```

WLK 20s threshold sweep, +4k:

```text
threshold 0.50 -> "the 2010 census showed that the city had a population of 1,000 people ..."
threshold 0.65 -> "the 2012 olympics will be the first time that the olympics will be held ..."
threshold 0.75 -> "the 2012 olympics will be held in são barcelona ..."
```

Comparison:

```text
Stage A 1k eval_loss:           5.25618989666303
Stage B audio4 eval_loss:       5.242083154122034
Stage A 4k eval_loss:           5.046820811430613
Stage A+ 2x2048 4k eval_loss:   4.998533650239309
Stage A+ resume +2k eval_loss:  4.882986927032471
Stage A+ resume +4k eval_loss:  4.856786131858826
```

Conclusion:

- Continuing A+ still improves FLEURS eval loss, but the marginal gain is
  shrinking.
- The WLK output gets longer and more repetitive as training continues. The
  model is over-optimizing the current aligned FLEURS objective rather than
  learning robust ASR.
- Do not spend more H100 time on blind continuation. The next useful step is to
  improve targets and regularization: better/harder eval, more diverse data,
  repetition penalty during realtime inference, or an auxiliary loss that
  constrains over-emission.

## 2026-05-28 - Stage A+ Anti-Repetition / Anti-Over-Emission

Code changes:

- Added token repetition metrics in `qwen3_streaming.metrics`.
- Extended `infer_realtime_checkpoint.py` JSON with token counts, wait/text
  ratios, hypothesis length, and unigram/bigram/trigram repetition stats.
- Added streaming decoding controls, all disabled by default:
  `--repetition-penalty`, `--no-repeat-ngram-size`, and
  `--max-consecutive-text-tokens`.
- Added training regularizers, disabled by default:
  `--emit-rate-loss-weight` and `--text-label-smoothing`.

Validation:

```text
local: python3 -m py_compile qwen3_streaming/metrics.py qwen3_streaming/native_realtime_model.py scripts/infer_realtime_checkpoint.py scripts/train_realtime_tiny_asr.py
local: python3 -m pytest -q -> 9 passed, 2 skipped
H100:  .venv/bin/python -m pytest -q -> 25 passed
```

Sweep target:

```text
checkpoint: runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume4k_lr1e5_v0
audio: data/wlk_audio/myfXyntFYL_20s.wav
artifacts: runs/jl_417352/stageAplus_antirepetition_sweep_resume4k/
```

Key sweep results:

```text
baseline th=0.65:
  text_ratio=0.195021
  trigram_repetition_ratio=0.200000
  hyp="the 2012 olympics will be the first time that the olympics will be held in a city that is not in the united states nor in the united in the world where the olympics will be held"

th=0.65, repetition_penalty=1.1, no_repeat_ngram_size=3, max_text=12:
  text_ratio=0.053942
  trigram_repetition_ratio=0.000000
  hyp="the 2014 earthquake occurred in the sikkar"

baseline th=0.75:
  text_ratio=0.136929
  trigram_repetition_ratio=0.258065
  hyp="the 2012 olympics will be held in são barcelona and will be the first time that the 2012 olympics will be"

th=0.75, repetition_penalty=1.2, no_repeat_ngram_size=3, max_text=12:
  text_ratio=0.087137
  trigram_repetition_ratio=0.000000
  hyp="the 2014 world cup of football was the fifteenth edition and third time that a team"
```

The decoding controls meet the narrow repetition goal on WLK 20 s: trigram
repetition drops by more than 30 percent, usually to zero. This is only a
decoding guardrail; semantic ASR quality remains poor.

Regularized resume command, corrected:

```bash
python scripts/train_realtime_tiny_asr.py \
  --output-dir runs/jl_417352/realtime_qwen_audio_surgery_stageAplus_antioveremit_resume1500_noword_v0 \
  --resume-from-checkpoint runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume2k_lr2e5_v0 \
  --decoder-backend qwen_audio_surgery \
  --train-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_fleurs_16s_v0/eval_manifest.jsonl \
  --steps 1500 \
  --lr 1e-5 \
  --batch-size 2 \
  --grad-acc 1 \
  --emit-gate-loss-weight 1.0 \
  --emit-gate-wait-weight 1.5 \
  --emit-rate-loss-weight 0.2 \
  --text-label-smoothing 0.05 \
  --emit-threshold 0.5 \
  --freeze-qwen-audio \
  --freeze-qwen-all \
  --no-word-start-token \
  --max-audio-sec 16.0 \
  --target-delay-sec 0.8
```

Regularized resume metrics:

```text
checkpoint: runs/jl_417352/realtime_qwen_audio_surgery_stageAplus_antioveremit_resume1500_noword_v0
eval_loss: 5.730818569660187
pred_text_ratio: 0.262109
label_text_ratio: 0.235984
repetition_trigram_repetition_ratio: 0.069075
loss_decreased: false
WLK th=0.75 baseline hyp="the new government government is a new"
WLK th=0.75 + controls hyp="the new government administration will be"
```

Conclusion:

- Keep the decoding controls and metrics.
- Do not promote the regularized resume checkpoint. It improves emission rate
  versus `resume2k` but fails the `eval_loss <= 5.0` gate and under-emits on
  WLK.
- The next useful training change is not another blind Stage A continuation.
  It needs better targets/data or a more explicit quality signal.

## 2026-05-28 - Stage A++ Data+Teacher Short Smoke

Goal:

- Stop repeating Stage A on the same FLEURS-only 2400-example set.
- Test whether a more diverse aligned dataset plus Qwen3-ASR-1.7B teacher
  filtering improves generalization.
- Keep the budget short: one 1000-step smoke from `resume2k_lr2e5_v0`, then
  decide by gates instead of extending the run.

Code changes:

- `prepare_qwen_aligned_jsonl.py` gained `--alignment-failure-mode` with
  `error`, `skip`, and explicit legacy `heuristic`. The Stage A++ dataset uses
  `skip`, so failed Qwen forced-alignment rows are dropped rather than filled
  with heuristic timestamps.
- `prepare_qwen_aligned_jsonl.py` now streams LibriSpeech clean-100 directly
  from targeted parquet files to avoid downloading the full `clean` config.
- Added `scripts/annotate_teacher_transcripts.py` for vLLM REST teacher
  annotation with resumable JSONL output and optional worker concurrency.
- Added `scripts/filter_teacher_manifest.py` to keep rows with no teacher
  error and `teacher_wer <= 0.35`.
- Added `scripts/make_audio_manifest.py` for WLK eval-only manifests.
- Added `scripts/eval_realtime_checkpoint.py` for checkpoint eval on manifest
  files with WER vs `teacher_text`, wait/text ratios, latency, and repetition
  metrics.
- Added tests for teacher annotation/filtering and audio manifest creation.

Validation:

```text
local: python3 -m py_compile scripts/prepare_qwen_aligned_jsonl.py scripts/make_audio_manifest.py scripts/annotate_teacher_transcripts.py scripts/filter_teacher_manifest.py scripts/eval_realtime_checkpoint.py
local: python3 -m pytest -q tests/test_teacher_pipeline.py -> 4 passed
H100:  .venv/bin/python -m pytest -q -> 29 passed
```

H100 instance:

```text
original: 417352
resumed:  417527
workspace: /home/ubuntu/qwen3-asr-streaming-h100/qwen3-asr-streaming-h100
teacher: Qwen/Qwen3-ASR-1.7B through vLLM /v1/audio/transcriptions
```

Aligned mixed data:

```bash
python scripts/prepare_qwen_aligned_jsonl.py \
  --out-dir data/qwen_aligned_mix_16s_v0 \
  --sources fleurs_en fleurs_fr librispeech_clean_100 \
  --max-train-per-source 1200 \
  --max-eval-per-source 120 \
  --max-audio-sec 16 \
  --drop-long-audio \
  --device-map cuda:0 \
  --dtype bfloat16 \
  --alignment-failure-mode skip \
  --skip-existing
```

Unfiltered aligned counts:

```text
train: 3600 rows, 1200 per source
eval:  360 rows, 120 per source
alignment_sources: qwen3_forced_aligner only
missing word_alignments: 0
```

Teacher annotation:

```text
eval smoke 10: ok=10, errors=0, teacher_wer_mean=0.1502
train full:   ok=3600, errors=0, teacher_wer_mean=0.1611
eval full:    ok=360, errors=0, teacher_wer_mean=0.1761
WLK eval:     ok=21, errors=0, teacher_latency_mean=12.93s
```

Teacher-filtered data:

```text
data/qwen_aligned_mix_16s_teacher_filter_v0/train_manifest.jsonl
  input=3600 kept=3491 rejected=109
  sources: librispeech_clean_100=1159, fleurs_en=1159, fleurs_fr=1173
  alignment_sources: qwen3_forced_aligner=3491

data/qwen_aligned_mix_16s_teacher_filter_v0/eval_manifest.jsonl
  input=360 kept=339 rejected=21
  sources: fleurs_fr=115, fleurs_en=117, librispeech_clean_100=107
  alignment_sources: qwen3_forced_aligner=339
```

Data verdict:

- Passes the minimum train/eval counts: 3491 train, 339 eval.
- Teacher request error rate is 0 percent.
- Final manifests contain no heuristic alignments.

Baseline WLK teacher-eval with controlled decoding:

```text
checkpoint: runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume2k_lr2e5_v0
output: runs/jl_417527/wlk_teacher_eval_resume2k_controlled.summary.json
emit_threshold=0.75, repetition_penalty=1.2, no_repeat_ngram_size=3, max_consecutive_text_tokens=12

wer_mean: 0.9977377669826761
text_token_ratio: 0.0015456299288107674
wait_token_ratio: 0.9984543700711892
trigram_repetition_ratio: 0.0
```

The controlled baseline almost never emits text on the 21 WLK files, so this
metric mostly measures under-emission rather than repetition.

Training smoke:

```bash
python scripts/train_realtime_tiny_asr.py \
  --output-dir runs/jl_417527/realtime_qwen_audio_surgery_stageAplus_mix_teacher_filter_smoke_v0 \
  --resume-from-checkpoint runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume2k_lr2e5_v0 \
  --decoder-backend qwen_audio_surgery \
  --train-manifest-jsonl data/qwen_aligned_mix_16s_teacher_filter_v0/train_manifest.jsonl \
  --eval-manifest-jsonl data/qwen_aligned_mix_16s_teacher_filter_v0/eval_manifest.jsonl \
  --steps 1000 \
  --lr 5e-6 \
  --batch-size 2 \
  --grad-acc 1 \
  --emit-gate-loss-weight 1.0 \
  --emit-gate-wait-weight 1.5 \
  --emit-threshold 0.5 \
  --freeze-qwen-audio \
  --freeze-qwen-all \
  --no-word-start-token \
  --max-audio-sec 16.0 \
  --target-delay-sec 0.8 \
  --num-workers 0
```

Training metrics:

```text
checkpoint: runs/jl_417527/realtime_qwen_audio_surgery_stageAplus_mix_teacher_filter_smoke_v0
train_examples: 3491
eval_examples: 339
trainable_params: 13634561
first_train_loss: 4.6482625007629395
last_train_loss: 3.5463759899139404
loss_decreased: true
eval_loss: 4.954030572666841
pred_text_ratio: 0.31728045325779036
label_text_ratio: 0.25648503027273234
label_text_ratio + 0.05: 0.3064850302727323
eval_trigram_repetition_ratio: 0.08649692067285596
```

WLK teacher-eval after smoke:

```text
checkpoint: runs/jl_417527/realtime_qwen_audio_surgery_stageAplus_mix_teacher_filter_smoke_v0
output: runs/jl_417527/wlk_teacher_eval_mix_teacher_filter_smoke_controlled.summary.json
emit_threshold=0.75, repetition_penalty=1.2, no_repeat_ngram_size=3, max_consecutive_text_tokens=12

wer_mean: 0.9961079053215807
relative WER improvement vs resume2k: about 0.16 percent
text_token_ratio: 0.0019404988887259272
wait_token_ratio: 0.9980595011112741
trigram_repetition_ratio: 0.0
```

Conclusion:

- The data+teacher pipeline itself worked and should be kept.
- Do not promote the smoke checkpoint. It passes `eval_loss <= 5.0`, but misses
  the `pred_text_ratio <= label_text_ratio + 0.05` gate by about 0.011 and is
  nowhere near the requested 10 percent WLK teacher-WER improvement.
- The main remaining failure is under-emission on WLK with controlled decoding:
  both baseline and smoke emit almost all wait tokens.
- Next step should diagnose the emit gate and frame-label distribution, not run
  a longer continuation of this exact setup.

## 2026-05-29 - Instance 417597, WLK Chunked Diagnostics

Goal:

- Re-check the WLK diagnosis after noticing that the previous WLK teacher-eval
  used full 277-425 s WAVs while Stage A models were trained with
  `max_audio_sec=16`.
- Build WLK chunk manifests that match the training horizon.
- Test whether domain teacher chunks can improve WLK behavior with a short
  adapter run.

Code changes:

- Added `scripts/sweep_realtime_decoding.py` to sweep emit thresholds and
  repetition controls while loading the checkpoint once.
- Added `scripts/slice_audio_manifest.py` to slice existing manifest audio into
  short WAV chunks.
- Added `scripts/align_manifest_with_qwen.py` to add Qwen3 forced alignments to
  an existing manifest using a chosen text field, e.g. `teacher_text`.
- Added focused tests for sweep config generation and audio chunk span logic.

Validation:

```text
local: python3 -m py_compile scripts/align_manifest_with_qwen.py scripts/slice_audio_manifest.py scripts/sweep_realtime_decoding.py
local: python3 -m pytest -q tests/test_slice_audio_manifest.py tests/test_sweep_realtime_decoding.py tests/test_teacher_pipeline.py -> 9 passed
H100:  .venv/bin/python -m pytest -q tests/test_slice_audio_manifest.py tests/test_sweep_realtime_decoding.py tests/test_teacher_pipeline.py -> 9 passed
```

First-20s WLK teacher eval:

```text
data/wlk_first20_teacher_1p7b_v0/manifest.jsonl
  input_records: 21
  output_records: 21
  chunk_sec: 20

teacher annotation:
  ok: 21
  errors: 0
  teacher_latency_mean: 1.02s
```

`resume2k` baseline on first-20s:

```text
checkpoint: runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume2k_lr2e5_v0
threshold 0.60: WER=0.9655, text_ratio=0.0792, trigram_rep=0.0
threshold 0.75: WER=0.9806, text_ratio=0.0251, trigram_rep=0.0
```

Stage A++ data+teacher smoke on first-20s:

```text
checkpoint: runs/jl_417527/realtime_qwen_audio_surgery_stageAplus_mix_teacher_filter_smoke_v0
threshold 0.20: WER=1.9715, text_ratio=0.4734
threshold 0.30: WER=1.5402, text_ratio=0.3596
threshold 0.40: WER=1.2647, text_ratio=0.2962
threshold 0.50: WER=1.0833, text_ratio=0.2298
threshold 0.60: WER=0.9899, text_ratio=0.1496
threshold 0.75: WER=0.9742, text_ratio=0.0338
trigram_rep: 0.0 for all swept settings
```

Conclusion from first-20s:

- The previous full-length WLK eval was out-of-distribution, but chunking does
  not rescue the Stage A++ smoke. Its best first-20s WER is still worse than
  `resume2k`.
- Lower thresholds mostly create over-emission, not better transcripts.

Full WLK 16 s chunk dataset:

```text
data/wlk_chunks16_teacher_1p7b_v0/manifest.jsonl
  input_records: 21
  output_records: 445
  chunk_sec: 16

teacher annotation:
  ok: 445
  errors: 0
  teacher_latency_mean: 0.758s

forced alignment:
  input: 445
  kept: 445
  rejected: 0
  alignment_sources: qwen3_forced_aligner=445

parent split:
  train: 338 chunks from 16 parent videos
  eval: 107 chunks from 5 held-out parent videos
```

`resume2k` baseline on held-out WLK 16 s chunks:

```text
threshold: 0.60
WER: 0.978724435613115
text_token_ratio: 0.0666897199031764
wait_token_ratio: 0.9333102800968236
trigram_repetition_ratio: 0.0
```

Domain-adapter attempt 1:

```text
checkpoint: runs/jl_417597/realtime_qwen_audio_surgery_stageAplus_wlk16_teacher_domain_wait05_v0
source: resume2k
train: 338 WLK teacher chunks
eval: 107 held-out WLK teacher chunks
steps: 800
lr: 1e-5
emit_gate_wait_weight: 0.5
freeze_qwen_audio: true
freeze_qwen_all: true

eval_loss: 5.185411815290098
pred_text_ratio: 0.5605478932650837
label_text_ratio: 0.19323190672543722
eval_trigram_repetition_ratio: 0.25306787951600446

threshold 0.75 WER: 1.2122575851311437
threshold 0.75 text_ratio: 0.24215778293731166
threshold 0.75 trigram_rep: 0.0
```

Domain-adapter attempt 2:

```text
checkpoint: runs/jl_417597/realtime_qwen_audio_surgery_stageAplus_wlk16_teacher_domain_wait10_v0
source: resume2k
train: 338 WLK teacher chunks
eval: 107 held-out WLK teacher chunks
steps: 500
lr: 5e-6
emit_gate_wait_weight: 1.0
freeze_qwen_audio: true
freeze_qwen_all: true

eval_loss: 5.547862644548769
pred_text_ratio: 0.42746101710981566
label_text_ratio: 0.19323190672543722
eval_trigram_repetition_ratio: 0.23192839338318605
```

Conclusion:

- Do not promote either WLK domain checkpoint.
- WLK pseudo-label fine-tuning with the current Stage A objective makes the emit
  gate over-emit and reintroduces repetitions. It does not improve held-out WLK
  WER.
- The best checkpoint among these tests remains `resume2k` with threshold 0.60,
  but it is still mostly wait tokens and not usable ASR.
- Next useful change is architectural/objective-level: train the emit decision
  as a calibrated CTC/RNNT-style blank-vs-token objective, or add a monotonic
  alignment/blank head, instead of using the current per-frame greedy text CE
  plus BCE gate.

## 2026-05-29 - CTC-lite v1 Objective Smoke

Implementation:

- Added `--alignment-loss ctc` to `scripts/train_realtime_tiny_asr.py`.
- Added a separate `ctc_head` on top of `audio_encoder.forward_full(mels)` plus
  `adapter.forward_full(audio_hidden)`.
- Kept the Qwen text decoder and LM head loaded/frozen for checkpoint
  compatibility; CTC loss does not use `previous_input_ids`, `[W]`, or the emit
  gate.
- Added `--decode-mode ctc` to `scripts/eval_realtime_checkpoint.py`, with CTC
  greedy collapse, optional blank-logit adjustment, and optional post-collapse
  `no_repeat_ngram_size`.
- Old checkpoints load with a newly initialized CTC head. For Qwen-backed
  models the CTC weight starts from `lm_head.weight`; `[P]` gets a small blank
  prior.

H100:

```text
instance: 418187
source checkpoint:
  runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume2k_lr2e5_v0
train/eval:
  data/qwen_aligned_mix_16s_teacher_filter_v0/train_manifest.jsonl
  data/qwen_aligned_mix_16s_teacher_filter_v0/eval_manifest.jsonl
run:
  runs/jl_418187/realtime_qwen_audio_surgery_ctc_lite_mix1500_blankbias2_v0
steps: 1500
lr: 1e-5
batch_size: 2
freeze_qwen_audio: true
freeze_qwen_all: true
alignment_loss: ctc
```

Training metrics:

```text
first_train_loss: 50.4884
last_train_loss: 8.0742
eval_loss: 8.6752
train_ctc_label_text_ratio: 0.2592
eval_pred_text_ratio greedy: 0.0
eval_wait_ratio greedy: 1.0
```

Interpretation: the CTC objective learns a much lower loss, but greedy decoding
is dominated by blank. Suppressing blank at decode time recovers text, but the
tokens are mostly repetitive and semantically weak.

WLK first-20s eval:

```text
baseline resume2k threshold 0.60:
  WER: 0.9655
  text_ratio: 0.0792
  trigram_rep: 0.0

CTC greedy, blank_adjust 0:
  WER: 1.0
  text_ratio: 0.0

CTC, blank_adjust -5:
  WER: 0.9715
  text_ratio: 0.2630
  trigram_rep: 0.9489

CTC, blank_adjust -5, no_repeat_ngram_size 3:
  WER: 0.9726
  text_ratio: 0.2630
  trigram_rep: 0.0
```

WLK held-out chunks16 eval:

```text
baseline resume2k threshold 0.60:
  WER: 0.9787
  text_ratio: 0.0667
  trigram_rep: 0.0

CTC, blank_adjust -5:
  WER: 0.9487
  text_ratio: 0.3057
  trigram_rep: 0.9454

CTC, blank_adjust -5, no_repeat_ngram_size 3:
  WER: 0.9536
  text_ratio: 0.3057
  trigram_rep: 0.0
```

Mixed public eval:

```text
CTC, blank_adjust -5:
  WER: 0.9757
  text_ratio: 0.1599
  trigram_rep: 0.8825
```

Conclusion:

- Do not promote the CTC-lite checkpoint.
- Positive signal: CTC can reduce objective loss and improves WLK chunks16 WER
  versus the weak `resume2k` baseline when blank is suppressed.
- Negative signal: first-20s does not improve, mixed public WER is still bad,
  and the text recovered from CTC is dominated by loops unless filtered.
- Next useful step is not a longer run of the same setup. The CTC head needs a
  better calibrated small-vocab/token-projection objective or RNNT-lite/joint
  network so blank/token competition is learned without brute-force blank
  suppression.

## 2026-05-29 - Compact CTC / Monotonic Projection Smoke

Implementation:

- Added `--alignment-loss compact_ctc`, a compact CTC projection over the token
  IDs actually observed in the aligned public manifests.
- Added `configure_compact_ctc_head(...)` and checkpoint metadata so the compact
  token-id mapping survives save/load.
- Added `--decode-mode compact_ctc` to eval, with the same CTC greedy collapse,
  blank-logit adjustment, and no-repeat post-filter.
- Kept the audio surgery architecture and frozen Qwen decoder/audio settings.

H100:

```text
instance: 418253
source checkpoint:
  runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume2k_lr2e5_v0
run:
  runs/jl_418253/realtime_qwen_audio_surgery_compact_ctc_mix1500_vocab32768_blank1p5_v1
steps: 1500
lr: 1e-5
compact_ctc_max_tokens: 32768
compact_vocab_size: 11844
compact_blank_bias: 1.5
train_examples: 3491
eval_examples: 339
```

Training metrics:

```text
first_train_loss: 24.1303
last_train_loss: 8.1486
eval_loss: 8.4835
train_ctc_label_text_ratio: 0.2592
eval_pred_text_ratio greedy: 0.0
```

Blank calibration:

```text
WLK first-20s, no repeat filter off:
  blank_adjust -1/-2/-3: WER 1.0, text_ratio 0.0
  blank_adjust -4: WER 0.9990, text_ratio 0.0004
  blank_adjust -5: WER 1.0276, text_ratio 0.2043, trigram_rep 0.7025
  blank_adjust -6/-7/-8: WER 1.1588, text_ratio 1.0
```

Controlled decoding:

```text
WLK first-20s baseline resume2k threshold 0.60:
  WER: 0.9655

Compact CTC, blank_adjust -5, no_repeat_ngram_size 3:
  WER: 0.9616
  text_ratio: 0.2043
  trigram_rep: 0.0

Compact CTC, blank_adjust -5.5, no_repeat_ngram_size 3:
  WER: 0.9626
  text_ratio: 1.0
  trigram_rep: 0.0

WLK chunks16 baseline resume2k threshold 0.60:
  WER: 0.9787

Compact CTC, blank_adjust -5, no_repeat_ngram_size 3:
  WER: 0.9307
  text_ratio: 0.2137
  trigram_rep: 0.0

Mixed public eval, blank_adjust -5, no_repeat_ngram_size 3:
  WER: 0.9564
  text_ratio: 0.1331
  trigram_rep: 0.0
```

Conclusion:

- This is a better probe than full-vocab CTC: chunks16 improves from the
  previous CTC `0.9536` to `0.9307`, and first-20s barely beats `resume2k`.
- It is not enough to promote. Greedy is still all blank, useful decoding needs
  manual blank suppression, and unigram/bigram repetition remains high even when
  trigram loops are blocked.
- The next real step should be RNNT-lite or a joint monotonic head with an
  explicit prediction/state branch, not more length on this CTC-only head.

## 2026-05-29 - RNNT-lite / Aligned Joint Head Smoke

Implementation:

- Added `--alignment-loss rnnt_lite`, keeping the Qwen audio-surgery streaming
  stack and adding a compact monotonic joint head.
- Added `configure_rnnt_lite_head(...)` with checkpoint metadata for
  `rnnt_lite_token_ids`, blank index, predictor dim, and joint dim.
- The v1 loss is aligned frame CE over compact labels with a predictor state
  equal to the previous nonblank compact token. This is not a full RNNT lattice;
  it is a cheap joint/prediction-state probe.
- Added `--decode-mode rnnt_lite` to the eval script. Streaming decode keeps
  only `last_ctc_token_id`-style compact prediction state plus the existing
  cached audio/adapter state.

Validation:

```text
local:  python3 -m py_compile ... OK
local:  python3 -m pytest -q -> 23 passed, 2 skipped
H100:   python3 -m pytest -q -> 43 passed
```

H100:

```text
instance: 418299
source checkpoint:
  runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume2k_lr2e5_v0
run:
  runs/jl_418299/realtime_qwen_audio_surgery_rnnt_lite_mix1500_blank0_wait10_v0
steps: 1500
lr: 1e-5
batch_size: 2
grad_acc: 1
rnnt_lite_max_tokens: 32768
compact_vocab_size: 11844
rnnt_lite_blank_bias: 0.0
wait_loss_weight: 1.0
train_examples: 3491
eval_examples: 339
trainable_params: 195736517
```

Smoke sweep before full run:

```text
blank_bias 1.5, wait_weight 1.0:
  eval_loss: 4.5
  greedy: all blank

blank_bias 0.0, wait_weight 0.5:
  eval_loss: 5.9344
  greedy: over-emits, text_ratio 0.874

blank_bias 0.0, wait_weight 0.3:
  eval_loss: 6.2250
  greedy: over-emits, text_ratio 0.920

blank_bias 0.0, wait_weight 1.0:
  eval_loss: 5.5563
  greedy: over-emits, text_ratio 0.815
```

Full run training metrics:

```text
first_train_loss: 9.375
last_train_loss: 6.3125
eval_loss: 6.2344
greedy eval: all blank
```

Blank calibration on WLK first-20s:

```text
no repeat filter off:
  blank_adjust 0/-1: WER 1.0, text_ratio 0.0
  blank_adjust -1.8: WER 0.9991, text_ratio 0.0113, trigram_rep 0.8095
  blank_adjust -2.0: WER 1.3839, text_ratio 0.5754, trigram_rep 0.8913
  blank_adjust <= -2.2: WER 1.6816, text_ratio 1.0, trigram_rep 0.9036

with no_repeat_ngram_size 3:
  blank_adjust -1.8: WER 0.9991, text_ratio 0.0113, trigram_rep 0.0
  blank_adjust -2.0: WER 0.9981, text_ratio 0.5754, trigram_rep 0.0
  blank_adjust -2.2/-2.5: WER 0.9990, text_ratio 1.0, trigram_rep 0.0
```

Best controlled decode checked on held-out sets:

```text
RNNT-lite, blank_adjust -2.0, no_repeat_ngram_size 3:
  WLK first-20s WER: 0.9981
  WLK chunks16 WER: 0.9951
  mixed public eval WER: 0.9986
  trigram_rep: 0.0
```

Conclusion:

- Do not promote the RNNT-lite v1 checkpoint.
- The joint head is wired correctly and trains, but it still has a sharp
  blank/token phase transition. Around the transition it either stays silent or
  emits attractor fragments such as `errupt` / `or`.
- This is worse than compact CTC on every useful WLK metric:
  first-20s `0.9981` vs `0.9616`, chunks16 `0.9951` vs `0.9307`.
- Next step should be a real monotonic criterion, not another calibrated
  aligned-CE run: either implement a small-vocab RNNT forward-backward loss or
  a CTC/attention monotonic alignment head where blank/token competition is
  trained by marginalization instead of hard frame labels.

## 2026-05-29 - RNNT Forward-Backward v1 Smoke

Implementation:

- Added `qwen3_streaming/rnnt.py` with exact log-space RNNT
  forward-backward loss over a compact vocab.
- Added `rnnt_prefix_targets(...)` to build the `U + 1` prediction states:
  blank/BOS state followed by previous target tokens.
- Added `--alignment-loss rnnt_fb`.
- Reused the compact RNNT joint head from `rnnt_lite`, but changed training
  from hard frame CE to marginalization over all blank/token paths.
- Added `forward_rnnt_lite_joint_logits(...)` on the realtime models, producing
  `[B, T, U + 1, Vcompact]` logits.
- Added length normalization for RNNT-FB with
  `--rnnt-fb-normalize-by-length/--no-rnnt-fb-normalize-by-length`; default is
  on. Without this, loss magnitude is dominated by sequence length.
- Added `--decode-mode rnnt_fb` as an eval alias for the same streaming compact
  joint head.

Validation:

```text
local: python3 -m py_compile qwen3_streaming/rnnt.py ... OK
H100:  python3 -m pytest -q -> 46 passed
```

H100:

```text
instance: 418328
source checkpoint:
  runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume2k_lr2e5_v0
```

RNNT-FB smoke without length normalization:

```text
run:
  runs/jl_418328/realtime_qwen_audio_surgery_rnnt_fb_mix100_vocab4096_v0
steps: 100
compact_vocab_size: 4096
train_examples after vocab filter: 257 / 3491
eval_examples after vocab filter: 46 / 339
first_train_loss: 359.2044
last_train_loss: 854.9139
eval_loss: 485.8917
loss_decreased: false
```

RNNT-FB normalized, capped vocab:

```text
run:
  runs/jl_418328/realtime_qwen_audio_surgery_rnnt_fb_mix60_vocab4096_norm_v0
steps: 60
compact_vocab_size: 4096
train_examples after vocab filter: 257 / 3491
eval_examples after vocab filter: 46 / 339
first_train_loss: 7.9823
last_train_loss: 6.3487
eval_loss: 6.3697
loss_decreased: true
eval_pred_text_ratio: 0.3048
eval_label_text_ratio: 0.3123
```

RNNT-FB normalized, full manifest vocab:

```text
run:
  runs/jl_418328/realtime_qwen_audio_surgery_rnnt_fb_mix20_vocabfull_norm_v0
steps: 20
compact_vocab_size: 11844
train_examples: 3491
eval_examples: 339
first_train_loss: 9.0233
last_train_loss: 8.5847
eval_loss: 8.5747
loss_decreased: true
```

WLK first-20s streaming eval:

```text
4096-vocab normalized checkpoint:
  blank_adjust 0.0, no_repeat_ngram_size 3:
    WER: 1.0
    text_ratio: 0.8471
    trigram_rep: 0.0
  blank_adjust 0.5, no_repeat_ngram_size 3:
    WER: 1.0
    text_ratio: 0.0022

full-vocab normalized checkpoint:
  blank_adjust 0.0/0.5/1.0, no_repeat_ngram_size 3:
    WER: 1.0
    text_ratio: 1.0
  blank_adjust 2.0, no_repeat_ngram_size 3:
    WER: 1.0
    text_ratio: 0.0041
  blank_adjust >= 3.0:
    WER: 1.0
    text_ratio: 0.0
```

Conclusion:

- The true RNNT forward-backward objective is now implemented and trainable.
- Length normalization is required. Without it, the smoke is noisy and does not
  reduce loss; with it, loss decreases cleanly.
- `Vcompact=4096` is too small for the mixed data: it drops 3234/3491 train
  examples and produces bad WLK fragments.
- `Vcompact=11844` keeps the full manifest but is expensive: several steps take
  10-18 seconds because logits scale as `T * (U + 1) * Vcompact`.
- The current checkpoints are not quality checkpoints. Free decoding still
  flips between over-emission and silence under blank-bias sweeps.

Next useful step:

- Keep `rnnt_fb` as the correct objective path.
- Optimize memory/speed before a longer run:
  compute RNNT log-probs in chunks over `U` or use a fused/torchaudio-style
  RNNT loss if available.
- Add a proper RNNT greedy/beam decoder with multiple symbols per frame and
  calibrated blank prior during training, instead of relying on the old
  one-symbol-per-frame eval loop.
- Then run a real `11844` vocab checkpoint for hundreds/thousands of steps.

## RNNT Greedy Decoder + Blank Prior Smoke

Instance: Jarvislab `418364`

Code change:

- Added true RNNT greedy decoding with up to `max_symbols_per_frame` nonblank
  emissions per frame.
- Added `rnnt_forced_advance_count` to detect nonblank loops that only stop
  because the per-frame cap is hit.
- `decode-mode=rnnt_fb` now uses this true RNNT greedy path; `rnnt_lite`
  remains the old one-symbol-per-frame path.

Validation:

```text
H100 targeted tests: 9 passed
H100 full tests:     48 passed
```

Checkpoint smoke:

```text
run:
  runs/jl_418364/realtime_qwen_audio_surgery_rnnt_fb_mix20_vocabfull_norm_blank2_v0
source checkpoint:
  runs/jl_417339/realtime_qwen_audio_surgery_stageAplus_adapter2x2048_resume2k_lr2e5_v0
steps: 20
compact_vocab_size: 11844
rnnt_lite_blank_bias: 2.0
first_train_loss: 7.3269
last_train_loss: 7.0886
eval_loss: 6.9925
eval_pred_text_ratio: 0.0011
eval_label_text_ratio: 0.2623
loss_decreased: true
```

WLK first-20s true RNNT greedy, coarse sweep:

```text
max_symbols_per_frame=1:
  adj  0.0: WER 1.0000, text_ratio 0.0036, forced 18
  adj -0.5: WER 1.0000, text_ratio 0.8062, forced 4080
  adj -1.0: WER 1.0026, text_ratio 1.0000, forced 5061

max_symbols_per_frame=2:
  adj  0.0: WER 1.0000, text_ratio 0.0039, forced 2
  adj -0.5: WER 1.0186, text_ratio 0.8524, forced 3732
  adj -1.0: WER 1.0685, text_ratio 1.0000, forced 5061

max_symbols_per_frame=4:
  adj  0.0: WER 1.0000, text_ratio 0.0039, forced 0
  adj -0.5: WER 1.0617, text_ratio 0.9158, forced 3659
  adj -1.0: WER 1.2630, text_ratio 1.0000, forced 5061

max_symbols_per_frame=8:
  adj  0.0: WER 1.0000, text_ratio 0.0039, forced 0
  adj -0.5: WER 1.1958, text_ratio 0.9444, forced 3377
  adj -1.0: WER 1.6210, text_ratio 1.0000, forced 5061
```

Fine blank sweep:

```text
max_symbols_per_frame=1:
  adj -0.1: WER 1.0000, text_ratio 0.0231, forced 117
  adj -0.2: WER 1.0000, text_ratio 0.1130, forced 572
  adj -0.3: WER 1.0000, text_ratio 0.2498, forced 1264
  adj -0.4: WER 1.0000, text_ratio 0.5819, forced 2945

max_symbols_per_frame=2:
  adj -0.1: WER 1.0000, text_ratio 0.0156, forced 16
  adj -0.2: WER 1.0000, text_ratio 0.0792, forced 152
  adj -0.3: WER 1.0000, text_ratio 0.4001, forced 1174
  adj -0.4: WER 0.9990, text_ratio 0.6167, forced 2186
```

Qualitative samples at the "best" calibrated points are still unusable. They
emit fragments such as `pton`, `AINEDerteULAR...`, `URITYchezichage...` rather
than transcript text.

Conclusion:

- The proper RNNT greedy decoder confirms the earlier diagnosis instead of
  fixing it.
- A positive blank prior stops over-emission but collapses to silence.
- Suppressing blank even slightly reintroduces forced-advance loops and
  nonsensical Qwen subword fragments.
- `max_symbols_per_frame > 1` is necessary for real RNNT decoding, but with
  this checkpoint it mostly increases the amount of bad text.

Next useful step:

- Do not promote this checkpoint.
- Do not just run longer from the same objective unchanged.
- Add an explicit duration / emission prior inside training, or move to a
  monotonic marginal loss with stronger blank calibration.
- Keep the true RNNT greedy decoder and forced-advance metric as diagnostics
  for future runs.

## RNNT-FB Duration Prior Smoke

Instance: Jarvislab `418383`

Code change:

- Added `rnnt_token_frames` targets derived from aligned word emissions while
  preserving the exact CTC target token order.
- Added optional `label_frame_targets` to `rnnt_forward_backward_loss`.
- Added a soft squared-distance label-emission prior controlled by:
  `--rnnt-duration-prior-weight`, `--rnnt-duration-prior-sigma-frames`, and
  `--rnnt-duration-prior-max-penalty`.
- Default behavior is unchanged when `rnnt_duration_prior_weight=0`.

Validation:

```text
H100 targeted tests: 13 passed
H100 full tests:     52 passed
```

Runs:

```text
A:
  blank_bias: 1.0
  duration_weight: 0.3
  sigma_frames: 6
  first_train_loss: 8.2505
  last_train_loss: 6.0586
  eval_loss: 6.1431
  eval_pred_text_ratio: 0.0
  eval_label_text_ratio: 0.2623

B:
  blank_bias: 1.0
  duration_weight: 0.7
  sigma_frames: 6
  first_train_loss: 8.2796
  last_train_loss: 6.0901
  eval_loss: 6.1758
  eval_pred_text_ratio: 0.0
  eval_label_text_ratio: 0.2623

C:
  blank_bias: 1.5
  duration_weight: 0.7
  sigma_frames: 8
  first_train_loss: 7.8353
  last_train_loss: 5.6576
  eval_loss: 5.7596
  eval_pred_text_ratio: 0.0
  eval_label_text_ratio: 0.2623
```

WLK first-20s true RNNT greedy:

```text
All A/B/C sweeps:
  max_symbols_per_frame: 1 or 2
  blank_adjust: -0.2, 0.0, 0.2
  WER: 1.0
  text_ratio: 0.0
  wait_ratio: 1.0
  forced_advance_count: 0
```

Operational note:

- The first A attempt filled the 100GB disk while saving `model.pt`.
- Old remote `model.pt` files were deleted except the required source
  checkpoint. Final runs delete each smoke `model.pt` after WLK eval; only logs,
  configs, tokenizers, predictions, and metrics are retained.

Conclusion:

- The duration prior is wired correctly: loss decreases and forced-advance loops
  disappear.
- This specific prior + blank-bias combination over-corrects into a blank
  attractor. The model no longer over-emits; it emits nothing.
- None of A/B/C meets the promotion criteria because `text_ratio=0.0` on mixed
  eval and WLK.

Next useful step:

- Do not continue these runs.
- Try the opposite calibration: no positive blank bias, weaker duration prior,
  and/or a target emission-rate term inside RNNT rather than only a path timing
  prior.
- Candidate next smoke: `blank_bias=0.0`, `duration_weight=0.05/0.1`, plus a
  mild blank logit penalty during training or a nonblank prior tied to
  `target_length / input_length`.

## RNNT-FB Nonblank Rate Calibration Smoke

Instance: Jarvislab `418456`

Code change:

- Added optional RNNT-FB nonblank-rate regularization controlled by
  `--rnnt-nonblank-rate-loss-weight`.
- The regularizer computes the mean lattice `1 - P(blank)` and targets
  `target_tokens / (frames + target_tokens)`.
- Added metrics for `rnnt_nonblank_rate_loss`, `rnnt_pred_nonblank_rate`, and
  `rnnt_target_nonblank_rate` in train/eval output.
- Default behavior is unchanged when the weight is `0`.

Validation:

```text
Local py_compile:        passed
Local tests:             skipped locally, no torch
H100 targeted tests:     16 passed
H100 tests directory:    55 passed
```

Runs:

```text
A:
  blank_bias: 0.0
  duration_weight: 0.0
  nonblank_rate_weight: 0.5
  first_train_loss: 9.3832
  last_train_loss: 7.1481
  eval_loss: 7.1784
  train_pred_nonblank_rate: 0.999503
  train_target_nonblank_rate: 0.200820
  eval_pred_nonblank_rate: 0.998898
  eval_target_nonblank_rate: 0.208347
  eval_pred_text_ratio: 0.0000278

B:
  blank_bias: 0.0
  duration_weight: 0.05
  nonblank_rate_weight: 0.5
  first_train_loss: 9.4074
  last_train_loss: 7.1672
  eval_loss: 7.1945
  train_pred_nonblank_rate: 0.999504
  train_target_nonblank_rate: 0.200820
  eval_pred_nonblank_rate: 0.998898
  eval_target_nonblank_rate: 0.208347
  eval_pred_text_ratio: 0.0000278

C:
  blank_bias: 0.0
  duration_weight: 0.1
  nonblank_rate_weight: 1.0
  first_train_loss: 9.7796
  last_train_loss: 7.5200
  eval_loss: 7.5183
  train_pred_nonblank_rate: 0.999504
  train_target_nonblank_rate: 0.200820
  eval_pred_nonblank_rate: 0.998899
  eval_target_nonblank_rate: 0.208347
  eval_pred_text_ratio: 0.0000278
```

WLK first-20s true RNNT greedy:

```text
All A/B/C sweeps:
  max_symbols_per_frame: 1 or 2
  blank_adjust: 0.0, 0.5, 1.0
  WER: 1.0
  text_ratio: 0.0
  forced_advance_count: 0
  trigram_repetition_ratio: 0.0
```

Conclusion:

- The regularizer is wired and test-covered, but this formulation does not fix
  greedy emission.
- The key diagnostic is that `1 - P(blank)` is already near `1.0`; the mass is
  spread across thousands of nonblank classes while blank remains the top single
  class for decoding.
- Therefore, a total nonblank-probability target is the wrong control variable
  for this model. It lowers/raises aggregate probability mass, but it does not
  create a winning text-token margin.
- None of A/B/C is promotable.

Next useful step:

- Stop RNNT calibration based on aggregate `1 - P(blank)`.
- Move to a margin-aware or monotonic objective:
  - either a blank-vs-best-nonblank margin/rank loss around aligned token frames;
  - or the planned monotonic marginal loss banded by token/frame windows.
- Keep `rnnt_nonblank_rate_loss` only as a diagnostic/optional ablation, not as
  the main next direction.

## RNNT-FB Target Margin Smoke

Instance: Jarvislab `418489`

Code change:

- Added optional aligned-window RNNT target margin losses:
  - `--rnnt-target-blank-margin-loss-weight`
  - `--rnnt-target-other-margin-loss-weight`
  - `--rnnt-target-margin-window-frames`
  - `--rnnt-target-blank-margin`
  - `--rnnt-target-other-margin`
- The loss uses `word_alignments -> rnnt_token_frames`, then compares each
  aligned target token against blank and the strongest competing nonblank in a
  small frame window.
- Default behavior is unchanged when both margin weights are `0`.

Validation:

```text
Local py_compile:        passed
Local tests:             skipped locally, no torch
H100 targeted tests:     19 passed
H100 tests directory:    58 passed
```

Runs:

```text
M1:
  blank_margin_weight: 0.5
  other_margin_weight: 0.0
  window_frames: 2
  first_train_loss: 9.7246
  last_train_loss: 8.4015
  eval_loss: 8.4681
  eval_pred_text_ratio: 1.0000
  eval_trigram_repetition_ratio: 0.5556
  best_wlk_wer: 0.9534
  best_wlk_text_ratio: 0.9025
  best_wlk_forced_advance_count: 4148

M2:
  blank_margin_weight: 0.5
  other_margin_weight: 0.1
  window_frames: 2
  first_train_loss: 9.9484
  last_train_loss: 8.6478
  eval_loss: 8.7215
  eval_pred_text_ratio: 0.7628
  eval_trigram_repetition_ratio: 0.2192
  best_wlk_wer: 0.9528
  best_wlk_text_ratio: 0.8685
  best_wlk_forced_advance_count: 3880

M3:
  blank_margin_weight: 0.2
  other_margin_weight: 0.2
  window_frames: 2
  first_train_loss: 9.7514
  last_train_loss: 8.0733
  eval_loss: 8.1122
  eval_pred_text_ratio: 0.1094
  eval_trigram_repetition_ratio: 0.1324
  best_wlk_wer: 0.9664
  best_wlk_text_ratio: 0.1699
  best_wlk_forced_advance_count: 445

M4:
  blank_margin_weight: 0.2
  other_margin_weight: 0.5
  window_frames: 2
  first_train_loss: 10.4227
  last_train_loss: 8.8121
  eval_loss: 8.8390
  eval_pred_text_ratio: 0.3184
  eval_trigram_repetition_ratio: 0.0689
  best_wlk_wer: 0.9894
  best_wlk_text_ratio: 0.2545
  best_wlk_forced_advance_count: 1288
```

Conclusion:

- The blank-vs-target margin works as a diagnostic and escapes the all-blank
  attractor.
- The competing-nonblank margin reduces verbosity/repetition on mixed eval, but
  WLK quality remains poor. Outputs are still dominated by fragments such as
  `the`, `2 M`, `an`, and malformed subwords.
- M1/M2 get superficially lower WER by over-emitting many wrong tokens. M3 has
  the cleanest emission control, but still no useful transcript quality. M4
  improves mixed-eval repetition over M3, but regresses WLK WER.
- None of M1-M4 is promotable. Smoke `model.pt` files are deleted; JSON metrics
  and prediction artifacts are retained.

Next useful step:

- Do not spend more H100 time on this exact RNNT calibration family.
- The next experiment should train token identity more directly:
  - aligned-window CE or sampled softmax over target plus hard negatives;
  - or the banded monotonic marginal loss where each target token is allowed a
    local time window, rather than only adding margin terms to RNNT logits.
- Keep the RNNT-FB path as infrastructure, but treat the current joint head as
  undertrained for lexical identity.

## Aligned-Window Token Identity Smoke

Instance: Jarvislab `418873`

Code change:

- Added `--alignment-loss aligned_window_ce` on the compact CTC projection.
- Added `--alignment-loss aligned_window_sampled_ce` with hard-negative sampled
  denominators.
- Added CLI controls:
  - `--aligned-window-frames`
  - `--aligned-window-blank-loss-weight`
  - `--aligned-window-sampled-hard-negatives`
- WLK eval can use `--decode-mode aligned_window_ce` and
  `aligned_window_sampled_ce`; both decode through the compact CTC head.

Validation:

```text
Local py_compile:        passed
Local tests:             skipped locally, no torch
H100 targeted tests:     23 passed
H100 tests directory:    62 passed
Sampled CE smoke:        2 steps, save/eval path passed
```

Runs:

```text
A:
  alignment_loss: aligned_window_ce
  window_frames: 3
  blank_loss_weight: 0.05
  steps: 300
  first_train_loss: 9.8489
  last_train_loss: 9.0159
  eval_loss: 8.7150
  eval_pred_text_ratio: 1.0000
  eval_label_text_ratio: 0.2623
  eval_target_blank_margin: 0.6777
  eval_target_other_margin: -3.7138
  WLK best WER: 0.9826
  WLK output pattern: "the"

B:
  alignment_loss: aligned_window_ce
  window_frames: 5
  blank_loss_weight: 0.02
  steps: 300
  first_train_loss: 9.5672
  last_train_loss: 8.7583
  eval_loss: 8.4562
  eval_pred_text_ratio: 1.0000
  eval_label_text_ratio: 0.2623
  eval_target_blank_margin: 0.9523
  eval_target_other_margin: -3.6532
  WLK best WER: 0.9826
  WLK output pattern: "the"
```

WLK first-20s sweeps:

```text
A/B:
  blank_adjust: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0
  text_token_ratio: 1.0 for every setting
  trigram_repetition_ratio: 0.0 after CTC collapse
  hypothesis: one collapsed token, consistently "the"
```

Conclusion:

- The implementation is wired and fast, but the objective collapses to the most
  frequent lexical token instead of learning audio-conditioned identity.
- Positive blank adjustment cannot recover a useful operating point because the
  frame argmax is dominated by one nonblank token before CTC collapse.
- Run C was not launched because A/B did not produce plausible words.
- These checkpoints are not promotable; model weights are deleted after metrics
  capture.

Next useful step:

- Stop pure aligned-window CE on the compact head.
- Add an explicit anti-frequency-collapse term before any longer training:
  - class-balanced token CE or inverse-frequency weights;
  - optional per-example negative sampling that always includes frequent tokens
    such as `the`;
  - monitor top-1 token histogram and entropy per eval batch.
- If that still collapses, move to a stronger projection/prediction head rather
  than extending the same head.

## Aligned-Window Anti-Frequency Smoke

Instance: Jarvislab `418884`

Code change:

- Added class-balanced target weighting for `aligned_window_*` losses:
  `--aligned-window-token-weighting {none,inverse_sqrt,inverse}`.
- Added weight bounds:
  `--aligned-window-min-token-weight`,
  `--aligned-window-max-token-weight`.
- Added frequent hard negatives for sampled CE:
  `--aligned-window-frequent-negative-count`.
- Added eval diagnostics for compact top-1 histograms:
  `top1_total`, `top1_unique`, `top1_entropy`, `top1_top`.

Validation:

```text
Local py_compile:        passed
Local tests:             skipped locally, no torch
H100 targeted tests:     25 passed
H100 tests directory:    64 passed
```

Runs:

```text
C1:
  alignment_loss: aligned_window_ce
  token_weighting: inverse_sqrt
  window_frames: 3
  blank_loss_weight: 0.2
  steps: 300
  first_train_loss: 12.2182
  last_train_loss: 17.1059
  eval_loss: 11.2190
  eval_pred_text_ratio: 0.9165
  eval_top1_unique: 3
  eval_top1_top: compact 11 / token 6 at 91.0%, blank 8.4%
  WLK best WER: 0.9980
  WLK output pattern: punctuation/apostrophes

C2:
  alignment_loss: aligned_window_sampled_ce
  token_weighting: inverse_sqrt
  sampled_hard_negatives: 128
  frequent_negative_count: 64
  window_frames: 3
  blank_loss_weight: 0.2
  steps: 300
  first_train_loss: 6.8221
  last_train_loss: 9.3874
  eval_loss: 6.4750
  eval_pred_text_ratio: 0.0036
  eval_top1_unique: 46
  eval_top1_top: blank at 99.6%
  WLK best WER: 1.0 at blank decode, >3.2 when blank is suppressed
```

WLK first-20s:

```text
C1:
  blank_adjust 0.0: WER 0.9980, text_ratio 0.8860
  blank_adjust 1.0/3.0: WER 1.0, text_ratio 0.0

C2:
  blank_adjust 0.0: WER 1.0, text_ratio 0.0014
  blank_adjust -1.0/-2.0/-3.0: WER 3.2359, text_ratio 1.0
```

Conclusion:

- The anti-frequency plumbing works and exposes the failure mode clearly.
- Inverse-frequency weighting breaks the previous `the` collapse, but does not
  recover audio-conditioned lexical identity.
- Full CE shifts to punctuation/nonblank collapse; sampled CE with frequent
  negatives shifts to blank collapse. Suppressing blank at decode time produces
  fluent-looking but unrelated token soup.
- No checkpoint is promotable; run C beyond 300 steps is not justified.

Next useful step:

- Stop using a shallow compact CTC head as the only lexical classifier.
- Add a stronger token-identity head before more training:
  - two-layer MLP or lightweight prediction network over audio frames;
  - keep class-balanced diagnostics;
  - optionally initialize from Qwen embeddings/LM head but train enough capacity
    to separate lexical classes.
- A useful next smoke is a compact-frame transformer/prediction head trained with
  balanced aligned-window CE, not more weighting on the current linear head.

## Cached Full-Hypothesis Streaming Pivot

Date: 2026-05-30

Reason:

- The RNNT/CTC/aligned-window smokes control blank/text rate, but they do not
  recover useful lexical identity.
- The more pragmatic next path is to keep the pretrained Qwen decoder path and
  solve the streaming cost directly:
  - cache finalized audio embeddings;
  - rerun a full text hypothesis on each update;
  - commit only stable hypothesis prefixes.

Code change:

- Added `CachedAudioDecodeState`.
- Added `append_audio_to_cache(mels, state)`:
  - uses the existing Qwen audio surgery chunk encoder;
  - appends only newly finalized frame embeddings;
  - preserves the bounded recompute window via `last_recomputed_frames`.
- Added `generate_full_hypothesis_from_cached_audio(...)`:
  - greedy full-hypothesis decode over cached finalized audio embeddings;
  - reruns the text decoder per update;
  - intentionally does not try to cache/reuse decoder KV yet.
- Added `qwen3_streaming/stable_commit.py`:
  - token LCP;
  - hold-back tokens;
  - stable-over-N-iterations commit;
  - final flush commit.
- Added `scripts/infer_cached_full_hypothesis.py`:
  - chunks a WAV;
  - caches audio embeddings incrementally;
  - decodes a full hypothesis each chunk;
  - outputs committed/stable-prefix events.

Validation:

```text
Local py_compile:
  qwen3_streaming/native_realtime_model.py
  qwen3_streaming/stable_commit.py
  scripts/infer_cached_full_hypothesis.py
  passed

Local pytest:
  28 passed, 2 skipped
```

H100 status:

```text
jl resume 418884 -y
-> Insufficient balance
```

No H100 smoke was run because the Jarvislab instance could not be resumed.

Follow-up after balance refill:

```text
jl resume 418884 -y
-> instance changed to 418889, H100 running at 217.18.55.82

H100 py_compile:
  qwen3_streaming/native_realtime_model.py
  qwen3_streaming/stable_commit.py
  scripts/infer_cached_full_hypothesis.py
  passed

H100 targeted tests:
  25 passed

H100 full tests:
  73 passed
```

Smoke results:

```text
Checkpoint StageA resume2k + repo_mel + naive audio-prefix decode:
  final_text: "None"
  cached_steps: 249
  audio_frames_seen: 2062
  last_recomputed_frames: 1564

Checkpoint StageA resume2k + Qwen prompt + Qwen processor features:
  final_text: "Hey! Hey! My name is. My name is..."
  trigram_repetition_ratio: 0.5556
  cached_steps: 250
  audio_frames_seen: 2064
  last_recomputed_frames: 1564

Base Qwen/Qwen3-ASR-0.6B + Qwen prompt + Qwen processor features:
  stable_committed_text:
    "Hello everyone. My name is Ilyal, and I will give you a short"
  last full hypothesis at flush:
    "Hello everyone. My name is Ilyich Bilad, and I will give an short overview
    of the paper \"Prompting Parm from Translation: Assessing Strategies and
    Performance.\" This is joint work with my colleagues from Google Translate.
    Parm is a 540 billion parameters large language model presented last year
    in 2022."
  cached_steps: 250
  audio_frames_seen: 2064
  last_recomputed_frames: 1564

Official qwen-asr package baseline, same 20 s WAV:
  "Hello everyone. My name is Adil Bilad, and I will give an short overview of
  the paper \"Prompting Parm from Translation: Assessing Strategies and
  Performance.\" This is joint work with my colleagues from Google Translate.
  Parm is a five hundred forty billion parameters language model presented last
  year in two thousand twenty-two. Its."
```

Interpretation:

- The audio cache works: `last_recomputed_frames=1564` is exactly the bounded
  `15 s left context + 640 ms right context`, while cached audio steps grow
  monotonically.
- The naive prefix decode was wrong because it bypassed Qwen-ASR's real prompt
  layout and used the repo's mel frontend.
- Using the real Qwen prompt and official processor features makes the cached
  full-hypothesis path qualitatively close to the official 0.6B baseline.
- The old StageA adapter checkpoint is not compatible with this full-hypothesis
  Qwen decoder path; it reintroduces repetition.
- The stable-prefix committer is too strict for final output: it correctly
  avoids revisions, but it can lock an early wording (`"you a short"`) while the
  later full hypothesis improves to `"an short overview..."`. The eval script
  must report both stable committed text and latest full hypothesis.

Current interpretation:

- This is a real runtime/wiring win for the base Qwen path: cached streaming
  audio plus full-hypothesis decode can stay close to official Qwen-ASR output
  on the WLK 20 s fixture.
- It directly attacks the original streaming cost: old finalized audio frames
  are cached and not re-embedded from the full 0:t window.
- The remaining recompute is bounded to the Qwen audio surgery window and the
  right-context tail. Text decoding is still full-hypothesis rerun per update.

Next gate:

- Update the inference summary to report both stable committed text and latest
  full hypothesis.
- Evaluate more WLK first-20s fixtures with base `Qwen/Qwen3-ASR-0.6B`,
  `--feature-mode qwen_processor`, and the Qwen prompt.
- Tune commit policy at word level instead of token level; keep a final mode
  that can revise the last unstable phrase.
- Only after that, decide whether we need a small training step. The old StageA
  checkpoints should not be reused for this full-hypothesis path.

## Word-Level Stable Commit

Date: 2026-05-30

Reason:

- Token-level LCP committed a too-specific early phrase:
  `"Hello everyone. My name is Ilyal, and I will give you a short"`.
- The later full hypothesis was much better, so the streaming layer needs a
  stable committed prefix plus a separately revisable display tail.

Code change:

- Added `StableTextCommitState` and `update_stable_text_commit(...)`.
- Added word-like text units, hold-back words, stable-over-N iterations, and a
  final mode that can revise the last unstable phrase.
- `scripts/infer_cached_full_hypothesis.py` now supports:
  - `--commit-mode {word,token}`;
  - `--hold-back-words`;
  - `--finalize-mode {latest,stable}`;
  - event fields `display`, `unstable`, `candidate`;
  - summary fields `final_display_text` and `stable_committed_text`.

Validation:

```text
Local py_compile:
  qwen3_streaming/stable_commit.py
  scripts/infer_cached_full_hypothesis.py
  passed

Local tests:
  32 passed, 2 skipped

H100 instance:
  418889 -> resumed as 418900

H100 targeted stable commit tests:
  9 passed

H100 full tests:
  77 passed
```

H100 smoke:

```text
command:
  scripts/infer_cached_full_hypothesis.py
  --model-id Qwen/Qwen3-ASR-0.6B
  --audio data/wlk_audio/myfXyntFYL_20s.wav
  --feature-mode qwen_processor
  --prompt-mode qwen_asr
  --commit-mode word
  --finalize-mode latest
  --hold-back-words 6
  --stable-iterations 2

final_text:
  "Hello everyone. My name is Ilyich Bilad, and I will give an short overview
  of the paper \"Prompting Parm from Translation: Assessing Strategies and
  Performance.\" This is joint work with my colleagues from Google Translate.
  Parm is a 540 billion parameters large language model presented last year
  in 2022."

stable_committed_text before final latest revision:
  "Hello everyone. My name is Ilyal, and I will give"

cache:
  cached_steps: 250
  audio_frames_seen: 2064
  last_recomputed_frames: 1564

repetition:
  bigram_repetition_ratio: 0.0
  trigram_repetition_ratio: 0.0
```

Artifacts:

- `runs/jl_418900/cached_fullhyp_wordcommit_base_qwen_myf_20s_summary.json`
- `runs/jl_418900/cached_fullhyp_wordcommit_base_qwen_myf_20s_events.jsonl`

Conclusion:

- The v1 runtime shape is now coherent:
  incremental Qwen audio embeddings, cached finalized audio prefix,
  full-hypothesis Qwen decode, stable committed prefix, and revisable display
  tail.
- The final output can use the latest full hypothesis while the streaming UI can
  expose only the committed prefix as immutable.
- Remaining work is evaluation breadth and commit policy tuning, not RNNT/CTC.

## Cached Full-Hypothesis Batch Eval

Date: 2026-05-30

Code change:

- Added `scripts/eval_cached_full_hypothesis.py`.
- Loads the cached full-hypothesis model once, then evaluates a manifest or
  audio directory.
- Outputs JSONL rows plus a summary with:
  - WER vs `teacher_text` / `raw_text` / `text`;
  - final/latest/stable text WER;
  - cache bound violations;
  - n-gram repetition;
  - latency and token counts.

Validation:

```text
Local py_compile:
  scripts/eval_cached_full_hypothesis.py
  passed

Local tests:
  32 passed, 2 skipped

H100 instance:
  418900 -> resumed as 418904

H100 py_compile:
  scripts/eval_cached_full_hypothesis.py
  scripts/infer_cached_full_hypothesis.py
  qwen3_streaming/stable_commit.py
  passed

H100 stable commit tests:
  9 passed

H100 full tests:
  77 passed
```

Eval setup:

```text
model: Qwen/Qwen3-ASR-0.6B
manifest: data/wlk_first20_teacher_1p7b_v0/manifest.teacher.jsonl
items: 21 first-20s WLK chunks
reference: Qwen3-ASR-1.7B teacher_text
chunk_ms: 1000
feature_mode: qwen_processor
prompt_mode: qwen_asr
commit_mode: word
finalize_mode: latest
hold_back_words: 6
stable_iterations: 2
```

Full 21-item result:

```text
ok: 21 / 21
total_latency_sec: 314.19
latency_mean_sec: 14.96
wer_final_mean: 0.1682
wer_latest_mean: 0.1682
wer_stable_mean: 0.7130
cache_bound_violations: 0
final_tokens_mean: 60.52
bigram_repetition_ratio: 0.0200
trigram_repetition_ratio: 0.0033
```

Worst rows are normal ASR disagreements against the 1.7B teacher, not cache
failures:

```text
QTlIuodOsA: WER 0.3333
myfXyntFYL: WER 0.3214
miPjvjWOvI: WER 0.2979
rOwZgUjcwB: WER 0.2857
krJSAnVcGR: WER 0.2500
```

Artifacts:

- `runs/jl_418904/cached_fullhyp_wordcommit_base_qwen_wlk_first20_full.jsonl`
- `runs/jl_418904/cached_fullhyp_wordcommit_base_qwen_wlk_first20_full.summary.json`
- `runs/jl_418904/cached_fullhyp_wordcommit_base_qwen_wlk_first20_limit5.jsonl`
- `runs/jl_418904/cached_fullhyp_wordcommit_base_qwen_wlk_first20_limit5.summary.json`

Interpretation:

- The cache invariant is now verified across the whole WLK first-20s diagnostic:
  every row stays at or below `1564` recomputed frames.
- Final/latest quality is close enough to the 1.7B teacher to justify this
  v1 path.
- Stable committed text is intentionally conservative and therefore has high
  WER if scored as a final transcript. It should be treated as immutable UI
  prefix, while `display/final_text` carries the revisable tail.

Next useful step:

- Tune `hold_back_words` and `stable_iterations` using the batch script:
  - lower hold-back improves committed coverage but risks visible revisions;
  - `finalize-mode latest` should remain the default for final transcript.
- Add streaming-specific metrics from event histories: time-to-first-display,
  time-to-first-committed-word, committed coverage over time, and revision count.

## Cached Full-Hypothesis Streaming Metrics + Commit Tuning

Date: 2026-05-30

Status: H100 instance `418909` paused after artifact download.

Code changes:

- Added per-event streaming metrics to `scripts/eval_cached_full_hypothesis.py`.
- Added `scripts/tune_stable_commit_from_events.py` to replay saved hypotheses
  and tune commit policy without rerunning ASR.
- Added optional normalized text-unit matching for stable commit comparison.
  Default behavior is unchanged.

Eval setup:

```text
model: Qwen/Qwen3-ASR-0.6B
manifest: data/wlk_first20_teacher_1p7b_v0/manifest.teacher.jsonl
items: 21 first-20s WLK chunks
chunk_ms: 1000
feature_mode: qwen_processor
commit_mode: word
finalize_mode: latest
hold_back_words: 6
stable_iterations: 2
```

Streaming metrics:

```text
ok: 21 / 21
wer_final_mean: 0.1682
wer_latest_mean: 0.1682
wer_stable_mean: 0.7130
cache_bound_violations: 0
first_display_sec_mean: 1.0952
first_commit_sec_mean: 12.4187
stable_coverage_ratio_mean: 0.3355
display_revision_events_mean: 17.4762
display_revision_words_mean: 104.0952
committed_revision_events_total: 0
bigram_repetition_ratio: 0.0200
trigram_repetition_ratio: 0.0033
```

Replay tuning result:

```text
Best low-risk current policy:
  hold_back_words: 6
  stable_iterations: 2
  first_commit_sec_mean: 12.42
  stable_coverage_ratio_mean: 0.3355
  stable_final_prefix_mismatch_count: 2 / 21

More aggressive policy:
  hold_back_words: 1
  stable_iterations: 1
  first_commit_sec_mean: 4.48
  stable_coverage_ratio_mean: 0.4808
  stable_final_prefix_mismatch_count: 13 / 21

Strict zero-mismatch policy:
  hold_back_words: 1
  stable_iterations: 3
  first_commit_sec_mean: 15.66
  stable_coverage_ratio_mean: 0.1359
  stable_final_prefix_mismatch_count: 0 / 21
```

Interpretation:

- First display is already good: roughly `1s`.
- Final/latest transcript quality remains good against the 1.7B teacher.
- Immutable commit is the unresolved product tradeoff:
  - aggressive commit is fast but freezes too many wrong prefixes;
  - strict commit is safe but nearly useless before finalization;
  - current `hold_back_words=6`, `stable_iterations=2` is conservative but
    still has 2 final-prefix mismatches on this 21-item WLK set.
- Normalized punctuation/spacing matching did not improve the aggregate
  tradeoff on this batch, so it should remain off by default for now.

Artifacts:

- `runs/jl_418909/cached_fullhyp_wordcommit_base_qwen_wlk_first20_streammetrics.jsonl`
- `runs/jl_418909/cached_fullhyp_wordcommit_base_qwen_wlk_first20_streammetrics.summary.json`
- `runs/jl_418909/events_first20/*.jsonl`
- `runs/jl_418909/stable_commit_tuning_grid.jsonl`
- `runs/jl_418909/stable_commit_tuning_grid_normalized.jsonl`

## Commit Delay Tuning

Date: 2026-05-30

Code change:

- Added `allow_commit` to stable token/text commit updates.
- Added `--min-commit-audio-sec` to:
  - `scripts/infer_cached_full_hypothesis.py`;
  - `scripts/eval_cached_full_hypothesis.py`;
  - `scripts/tune_stable_commit_from_events.py`.

Rationale:

- The false immutable prefixes are not only punctuation/spacing artifacts.
- Main failure cases are lexical corrections of early names or formulaic intro
  text, for example:
  - `Ilyal` -> `Ilyich Bilad`;
  - `Yu Xinjiang` -> `Yu Xin Zhang`;
  - `Hi, and I'm` -> `Hi, I'm`.
- A minimum commit delay keeps the fast revisable display path unchanged while
  preventing early false immutable commits.

Offline replay result on the same WLK first-20s event histories:

```text
Best zero-final-prefix-mismatch policy found:
  hold_back_words: 2
  stable_iterations: 2
  min_commit_audio_sec: 12.0
  first_commit_sec_mean: 15.8853
  stable_coverage_ratio_mean: 0.3527
  stable_word_count_mean: 15.95
  stable_final_prefix_mismatch_count: 0 / 21

Previous zero-mismatch policy without min delay:
  hold_back_words: 1
  stable_iterations: 3
  first_commit_sec_mean: 15.6629
  stable_coverage_ratio_mean: 0.1359
  stable_final_prefix_mismatch_count: 0 / 21

Previous default:
  hold_back_words: 6
  stable_iterations: 2
  min_commit_audio_sec: 0.0
  first_commit_sec_mean: 12.4187
  stable_coverage_ratio_mean: 0.3355
  stable_final_prefix_mismatch_count: 2 / 21
```

Interpretation:

- `min_commit_audio_sec=12` is a better safety gate than simply increasing
  `stable_iterations`.
- It does not solve early immutable streaming latency; first immutable commit is
  still late on 20s clips.
- The practical v1 UI should therefore distinguish:
  - fast `display` text from the latest full hypothesis;
  - conservative `committed` text for irreversible UI/actions;
  - final transcript from `finalize-mode latest`.

Artifact:

- `runs/jl_418909/stable_commit_tuning_grid_minsec.jsonl`

## Objective Audit: Incremental Audio Encoder v1

Date: 2026-05-30

Status:

- H100 instance `418909` resumed as `418918` for verification, then paused.

Code changes:

- Added explicit recompute diagnostics to `QwenAudioSurgeryState`:
  - `last_input_frames`;
  - `last_recomputed_context_frames`.
- Added those fields to cached full-hypothesis inference/eval event JSON.
- Added summary fields in batch eval:
  - `max_last_recomputed_frames`;
  - `max_recomputed_context_frames`;
  - `cache_bound_frames`.
- Added the same context-recompute metric to
  `scripts/validate_qwen_audio_surgery.py`.
- Strengthened tests to verify that cached finalized audio embeddings remain
  append-only and that Qwen audio surgery recomputation remains bounded.

Verification:

```text
Local:
  python3 -m py_compile ...
  python3 -m pytest -q
  44 passed, 2 skipped

H100 418918 targeted:
  .venv/bin/python -m py_compile ...
  .venv/bin/python -m pytest -q tests/test_native_realtime_model.py \
    tests/test_stable_commit.py tests/test_tune_stable_commit_from_events.py \
    tests/test_streaming_metrics.py
  41 passed

H100 418918 full:
  .venv/bin/python -m pytest -q
  89 passed
```

Audit against the pivot objective:

```text
Qwen autoregressive decoder:
  implemented for cached audio full-hypothesis decode.

Cached finalized audio embeddings:
  implemented; tests verify previously cached frame embeddings stay unchanged
  when new audio is appended.

Right-context finalization:
  implemented via Qwen audio surgery `right_context_frames`, default 640 ms.

Avoid full-window recompute:
  implemented as bounded-window recomputation, not full-audio recomputation.
  WLK first-20s H100 result had max_last_recomputed_frames=1564 for
  2064 seen frames, with cache_bound_violations=0.

Encode only new chunk + tail:
  partially implemented. The current tail is a bounded local window
  `left_context + right_context`, current default about 2.64s. This is not yet true
  per-layer audio KV append-only.

Stable prefix commit:
  implemented with token/text LCP, hold-back, stable-iteration gating,
  optional normalized matching, and optional minimum audio time before commit.
```

Remaining gap before calling the objective fully done:

- The audio tower still calls the pretrained Qwen audio encoder on a bounded
  mel window every update. It does not yet replace Qwen audio self-attention
  with real streaming attention/KV cache.
- Therefore the v1 is a pragmatic cached full-hypothesis prototype with bounded
  recompute, not the final “append only Qwen audio KV” implementation.

## Audio Context Override Smoke

Date: 2026-05-30

Status:

- H100 instance `418918` resumed as `418920`, then paused after verification.

Code change:

- Added context override flags to cached full-hypothesis inference/eval:
  - `--qwen-audio-left-context-sec`;
  - `--qwen-audio-right-context-ms`.
- Defaults were unchanged at this point: 15s left context and 640ms right
  context. They were changed later after the WLK context sweep.
- Batch summaries now report the effective context frame counts.

Validation:

```text
Local:
  python3 -m py_compile scripts/eval_cached_full_hypothesis.py \
    scripts/infer_cached_full_hypothesis.py qwen3_streaming/realtime_config.py
  python3 -m pytest -q tests/test_realtime_config.py tests/test_stable_commit.py \
    tests/test_tune_stable_commit_from_events.py
  21 passed

H100 418920 targeted:
  .venv/bin/python -m py_compile ...
  .venv/bin/python -m pytest -q tests/test_realtime_config.py \
    tests/test_native_realtime_model.py tests/test_stable_commit.py \
    tests/test_tune_stable_commit_from_events.py
  41 passed

H100 418920 full:
  .venv/bin/python -m pytest -q
  92 passed
```

Real smoke, WLK first item only:

```text
model: Qwen/Qwen3-ASR-0.6B
left_context_sec: 2.0
right_context_ms: 640
qwen_audio_left_context_frames: 200
qwen_audio_right_context_frames: 64
cache_bound_frames: 264
max_last_recomputed_frames: 264
max_recomputed_context_frames: 200
cache_bound_violations: 0
wer_final_mean: 0.0217
first_display_sec_mean: 1.0
first_commit_sec_mean: 16.0
stable_coverage_ratio_mean: 0.6222
```

Interpretation:

- The override works and proves the v1 can reduce recompute from the default
  1564-frame bound to a 264-frame bound for this setting.
- This is still bounded-window recompute, but now the tail size is measurable
  and tunable. The next useful experiment is a 21-item WLK sweep over left
  contexts such as 1s, 2s, 4s, 8s, 15s while holding right context at 640ms.

Artifacts:

- `runs/jl_418920/context_override_smoke_left2s_limit1.jsonl`
- `runs/jl_418920/context_override_smoke_left2s_limit1.summary.json`
- `runs/jl_418920/context_override_smoke_events/*.jsonl`

## Audio Left-Context Sweep on WLK First-20s

Date: 2026-05-30

Status:

- H100 instance `418920` resumed as `418924`.
- Full sweep completed, artifacts downloaded, instance paused.

Code change:

- Added `scripts/summarize_cached_context_sweep.py` to aggregate context sweep
  `.summary.json` files into JSON and Markdown tables.
- Added tests in `tests/test_summarize_cached_context_sweep.py`.

Validation:

```text
Local:
  python3 -m pytest -q
  49 passed, 2 skipped

H100 targeted:
  .venv/bin/python -m py_compile scripts/eval_cached_full_hypothesis.py \
    scripts/infer_cached_full_hypothesis.py \
    scripts/summarize_cached_context_sweep.py qwen3_streaming/realtime_config.py
  .venv/bin/python -m pytest -q tests/test_summarize_cached_context_sweep.py \
    tests/test_realtime_config.py tests/test_native_realtime_model.py
  25 passed
```

Eval setup:

```text
model: Qwen/Qwen3-ASR-0.6B
manifest: data/wlk_first20_teacher_1p7b_v0/manifest.teacher.jsonl
items: 21 first-20s WLK chunks
reference: Qwen3-ASR-1.7B teacher_text
right_context_ms: 640
left_context_sec sweep: 1, 2, 4, 8, 15
chunk_ms: 1000
feature_mode: qwen_processor
commit_mode: word
finalize_mode: latest
hold_back_words: 2
stable_iterations: 2
min_commit_audio_sec: 12
```

Result:

```text
| left_s | right_ms | recompute | ctx_recompute | WER    | WER_delta | first_display | first_commit | stable_cov |
| ------ | -------- | --------- | ------------- | ------ | --------- | ------------- | ------------ | ---------- |
| 1.00   | 640      | 164       | 100           | 0.1662 | -0.0020   | 1.0952        | 15.5760      | 0.3341     |
| 2.00   | 640      | 264       | 200           | 0.1534 | -0.0148   | 1.0952        | 14.1538      | 0.2959     |
| 4.00   | 640      | 464       | 400           | 0.1562 | -0.0121   | 1.0952        | 14.5714      | 0.2776     |
| 8.00   | 640      | 864       | 800           | 0.1538 | -0.0145   | 1.0952        | 15.6629      | 0.3515     |
| 15.00  | 640      | 1564      | 1500          | 0.1682 | +0.0000   | 1.0952        | 15.8853      | 0.3527     |
```

Interpretation:

- `left_context=2s` is the best point in this sweep:
  - best WER against the 1.7B teacher: `0.1534`;
  - recompute bound `264` frames instead of `1564` at 15s;
  - same first display latency as 15s.
- `left_context=1s` is also viable on WER but slightly worse than 2s.
- Larger contexts do not improve quality on this WLK first-20s diagnostic.
- This gives a pragmatic v1 default candidate, now applied in code:
  - `left_context_sec=2.0`;
  - `right_context_ms=640`;
  - still bounded-window recompute, not true audio KV append-only.

Artifacts:

- `runs/jl_418924/context_sweep_wlk_first20/*.jsonl`
- `runs/jl_418924/context_sweep_wlk_first20/*.summary.json`
- `runs/jl_418924/context_sweep_wlk_first20/summary.json`
- `runs/jl_418924/context_sweep_wlk_first20/summary.md`

## Default Audio Context Update

Date: 2026-05-30

Status:

- H100 instance `418924` resumed as `418934`.
- Default-context smoke artifacts downloaded, instance paused.

Code change:

- Changed the experimental Qwen audio surgery default left context from `15s`
  to `2s` in `RealtimeAudioConfig`.
- Updated CLI defaults in:
  - `scripts/validate_qwen_audio_surgery.py`;
  - `scripts/train_realtime_tiny_asr.py`.
- Explicit CLI overrides still allow larger contexts, including the old 15s
  setting.

Rationale:

- The WLK first-20s sweep showed `left_context=2s`, `right_context=640ms` as
  the best measured point:
  - `WER=0.1534` vs `0.1682` at 15s;
  - recompute bound `264` frames vs `1564` at 15s.
- This makes the default v1 behavior closer to the target shape:
  new chunk plus small bounded tail, cached finalized audio embeddings, and
  normal Qwen autoregressive full-hypothesis decode.

Validation:

```text
Local:
  python3 -m pytest -q
  50 passed, 2 skipped

H100 targeted:
  .venv/bin/python -m py_compile qwen3_streaming/realtime_config.py \
    scripts/validate_qwen_audio_surgery.py scripts/train_realtime_tiny_asr.py
  .venv/bin/python -m pytest -q tests/test_realtime_config.py \
    tests/test_native_realtime_model.py
  24 passed
```

Default no-override smoke, WLK first item only:

```text
model: Qwen/Qwen3-ASR-0.6B
explicit context override: none
qwen_audio_left_context_frames: 200
qwen_audio_right_context_frames: 64
cache_bound_frames: 264
max_last_recomputed_frames: 264
max_recomputed_context_frames: 200
cache_bound_violations: 0
wer_final_mean: 0.0217
first_display_sec_mean: 1.0
```

Artifacts:

- `runs/jl_418934/default_left2s_smoke_limit1.jsonl`
- `runs/jl_418934/default_left2s_smoke_limit1.summary.json`
- `runs/jl_418934/default_left2s_smoke_events/*.jsonl`

## Cached Full-Hypothesis Runtime Extraction

Date: 2026-05-30

Status:

- H100 instance `418934` resumed as `418937`.
- Refactored runtime smoke artifacts downloaded, instance paused.

Code change:

- Added `qwen3_streaming/cached_full_hypothesis.py`.
- Extracted the stateful v1 runtime from the CLI scripts:
  - audio embedding cache state;
  - full-hypothesis Qwen decode;
  - token/word stable-prefix commit;
  - finalization behavior;
  - per-chunk event payloads.
- Updated:
  - `scripts/infer_cached_full_hypothesis.py`;
  - `scripts/eval_cached_full_hypothesis.py`.
- Added `tests/test_cached_full_hypothesis.py`.

Validation:

```text
Local:
  python3 -m pytest -q
  54 passed, 2 skipped

H100 targeted:
  .venv/bin/python -m py_compile qwen3_streaming/cached_full_hypothesis.py \
    scripts/infer_cached_full_hypothesis.py scripts/eval_cached_full_hypothesis.py
  .venv/bin/python -m pytest -q tests/test_cached_full_hypothesis.py \
    tests/test_native_realtime_model.py tests/test_realtime_config.py
  28 passed
```

Refactored runtime smoke, WLK first item only:

```text
model: Qwen/Qwen3-ASR-0.6B
qwen_audio_left_context_frames: 200
qwen_audio_right_context_frames: 64
cache_bound_frames: 264
max_last_recomputed_frames: 264
max_recomputed_context_frames: 200
cache_bound_violations: 0
wer_final_mean: 0.0217
first_display_sec_mean: 1.0
first_commit_sec_mean: 16.0
stable_coverage_ratio_mean: 0.6222
```

Interpretation:

- The v1 path is now a reusable stateful runtime, not just duplicated CLI
  logic.
- This is the unit to plug into a server or future vLLM/vllm-metal stateful
  path: append mel/audio chunks, update cached finalized audio embeddings,
  decode a complete Qwen hypothesis, emit `display` and conservative
  `committed` text.

Artifacts:

- `runs/jl_418937/refactored_streamer_smoke_limit1.jsonl`
- `runs/jl_418937/refactored_streamer_smoke_limit1.summary.json`
- `runs/jl_418937/refactored_streamer_smoke_events/*.jsonl`

## Full-Length WLK Smoke: Context/Latency Tradeoff

Date: 2026-05-30

Status:

- H100 instance `418937` resumed as `418938`.
- Long-audio artifacts downloaded.
- H100 instance `418938` paused.

Code change:

- Fixed long-audio feature extraction in:
  - `scripts/infer_cached_full_hypothesis.py`;
  - `scripts/eval_cached_full_hypothesis.py`.
- `WhisperFeatureExtractor` defaults to a 30s chunk length for Qwen ASR
  features. The cached full-hypothesis path now passes `truncation=False`
  explicitly; otherwise full WLK files silently evaluate only the first ~30s.
- Added per-item and summary real-time factor metrics to
  `scripts/eval_cached_full_hypothesis.py`:
  - `audio_duration_sec`;
  - `realtime_factor`;
  - `audio_duration_total_sec`;
  - `realtime_factor_mean`;
  - `realtime_factor_total`.

Validation:

```text
Local:
  python3 -m py_compile scripts/eval_cached_full_hypothesis.py \
    scripts/infer_cached_full_hypothesis.py
  python3 -m pytest -q
  54 passed, 2 skipped

H100 targeted:
  .venv/bin/python -m py_compile scripts/eval_cached_full_hypothesis.py \
    scripts/infer_cached_full_hypothesis.py
  .venv/bin/python -m pytest -q tests/test_cached_full_hypothesis.py \
    tests/test_native_realtime_model.py tests/test_realtime_config.py
  28 passed
```

Initial bug-finding smoke:

```text
input: WLK full EqmWoxNDIr.wav, 301.0s
feature extraction before fix: audio_frames_seen=3064 (~30.6s)
result: WER=0.9073, final_tokens=80
```

After `truncation=False`, the same file sees the full audio:

```text
audio_frames_seen=30164
```

Context sweep on one full WLK file:

```text
model: Qwen/Qwen3-ASR-0.6B
manifest: data/wlk_teacher_1p7b_v0/manifest.teacher.jsonl
item: EqmWoxNDIr, 301.0s
chunk_ms: 10000
right_context: 640ms
max_new_tokens: 2048
reference: Qwen/Qwen3-ASR-1.7B teacher transcript
```

| left context | recompute bound | RTF | final WER | final tokens |
| --- | ---: | ---: | ---: | ---: |
| 2s | 264 frames | 0.320 | 0.850 | 155 |
| 4s | 464 frames | 0.601 | 0.665 | 293 |
| 8s | 864 frames | 1.425 | 0.324 | 595 |
| 15s | 1564 frames | 2.073 | 0.126 | 724 |

Interpretation:

- The bounded-window audio cache is working mechanically:
  `cache_bound_violations=0` for all runs.
- The 2s default from the 20s sweep is too aggressive for full-length talks:
  it is fast, but it loses too much long-range audio context and produces a
  short, skipped hypothesis.
- The old 15s left context restores usable final quality on this 301s file,
  but it is slower than real time on H100 with the current full-hypothesis
  decode loop.
- The main bottleneck is now clear:
  - audio recompute is bounded by `left_context + right_context`;
  - decoder work is still full-hypothesis repeated for every chunk;
  - conservative stable-prefix commit performs poorly on long audio because
    full hypotheses revise heavily across chunks.
- This means the current v1 is a useful correctness probe, not the final
  realtime design. To get real realtime quality, the next step is either:
  - deeper Qwen audio surgery with a larger effective context but cheaper
    incremental KV reuse; or
  - segment-level serving with committed audio/text windows so the decoder does
    not repeatedly regenerate the whole transcript.

Artifacts:

- `runs/jl_418938/cached_fullhyp_wlk_full_limit1_left2s.summary.json`
- `runs/jl_418938/cached_fullhyp_wlk_full_limit1_left2s_notrunc_chunk10s.summary.json`
- `runs/jl_418938/cached_fullhyp_wlk_full_limit1_left4s_notrunc_chunk10s.summary.json`
- `runs/jl_418938/cached_fullhyp_wlk_full_limit1_left8s_notrunc_chunk10s.summary.json`
- `runs/jl_418938/cached_fullhyp_wlk_full_limit1_left15s_notrunc_chunk10s.summary.json`

## Segmented Cached Full-Hypothesis Streamer

Date: 2026-05-30

Status:

- H100 instance `418938` resumed as `418957`.
- Full WLK 21-audio segmented eval completed.
- Artifacts downloaded.
- H100 instance `418957` paused.

Code change:

- Added `SegmentedCachedFullHypothesisStreamer` in
  `qwen3_streaming/cached_full_hypothesis.py`.
- The base streamer remains the default.
- The segmented streamer:
  - keeps the same Qwen autoregressive full-hypothesis decode inside the active
    window;
  - finalizes a completed segment;
  - appends the segment text/tokens to global completed output;
  - trims `state.frame_hidden` to `segment_keep_tail_steps`;
  - resets stable-prefix state for the next active segment;
  - preserves global `display`, `committed`, and final transcript text.
- Added CLI flags to `scripts/infer_cached_full_hypothesis.py` and
  `scripts/eval_cached_full_hypothesis.py`:
  - `--segment-max-cached-steps`;
  - `--segment-keep-tail-steps`;
  - `--segment-finalize-mode {latest,stable}`.
- Added summary metrics:
  - `segments_finalized_mean`;
  - `dropped_cached_steps_total`.
- Added unit tests for manual rollover and automatic cache trimming.

Validation:

```text
Local:
  python3 -m py_compile qwen3_streaming/cached_full_hypothesis.py \
    scripts/infer_cached_full_hypothesis.py \
    scripts/eval_cached_full_hypothesis.py
  python3 -m pytest -q
  56 passed, 2 skipped

H100 targeted:
  .venv/bin/python -m py_compile qwen3_streaming/cached_full_hypothesis.py \
    scripts/infer_cached_full_hypothesis.py \
    scripts/eval_cached_full_hypothesis.py
  .venv/bin/python -m pytest -q tests/test_cached_full_hypothesis.py \
    tests/test_native_realtime_model.py tests/test_realtime_config.py
  30 passed
```

Real-model segmented smoke:

```text
model: Qwen/Qwen3-ASR-0.6B
manifest: data/wlk_first20_teacher_1p7b_v0/manifest.teacher.jsonl
limit: 1
chunk_ms: 2000
left_context: 15s
right_context: 640ms
segment_max_cached_steps: 20
segment_keep_tail_steps: 0

ok: 1
RTF: 0.204
WER final: 0.2609
segments_finalized: 9
dropped_cached_steps_total: 242
latest_tokens_mean: 3.0
final_tokens_mean: 65.0
```

This proves the segment accumulator works with the real Qwen model: the active
window ends with only a few latest tokens, while the final transcript still
contains the completed segment tokens.

Full-length single-item comparison:

```text
item: EqmWoxNDIr.wav, 301.0s
reference: Qwen/Qwen3-ASR-1.7B teacher transcript
chunk_ms: 10000
left_context: 15s
right_context: 640ms
```

| mode | segment_max_cached_steps | RTF | final WER | final tokens |
| --- | ---: | ---: | ---: | ---: |
| global full-hypothesis | 0 | 2.073 | 0.126 | 724 |
| segmented | 200 | 0.127 | 0.139 | 717 |

Interpretation:

- The segmented path preserves nearly the same final quality on this file.
- It is about 16x faster than the global full-hypothesis loop on the same
  H100 run (`RTF 2.073 -> 0.127`).
- This directly addresses the previous bottleneck: the audio tower can keep a
  large 15s left context, while the Qwen text decoder no longer regenerates the
  entire talk every chunk.

Full WLK 21-audio eval:

```text
model: Qwen/Qwen3-ASR-0.6B
manifest: data/wlk_teacher_1p7b_v0/manifest.teacher.jsonl
items: 21 full WLK audios
total audio duration: 7105.5s
chunk_ms: 10000
left_context: 15s
right_context: 640ms
segment_max_cached_steps: 200
segment_keep_tail_steps: 0
segment_finalize_mode: latest
```

Results:

```text
ok: 21/21
total latency: 1040.7s
RTF total: 0.1465
RTF mean: 0.1462
WER final mean: 0.1534
WER stable mean: 0.1665
cache_bound_violations: 0
max_last_recomputed_frames: 1564
segments_finalized_mean: 16.7
dropped_cached_steps_total: 87457
first_display_sec_mean: 10.0
first_commit_sec_mean: 30.0
committed_revision_events_total: 0
```

Best/worst WER examples:

```text
best:
  wJAPXMIoIG  WER=0.0909  RTF=0.1430
  rxrToXvRyM  WER=0.1112  RTF=0.1582
  ICWfTnUMio  WER=0.1234  RTF=0.1597

worst:
  xhjBHGVOyQ  WER=0.1927  RTF=0.1411
  ccpXHNfaoy  WER=0.1946  RTF=0.1228
  UOlPKyCVgg  WER=0.2646  RTF=0.1613
```

Current conclusion:

- This is the first v1 path that looks structurally viable for long streaming:
  Qwen audio tower with large context, bounded audio recompute, normal Qwen
  autoregressive decode, and stable/segmented commit.
- The remaining rough edge is semantic: segment boundaries can still drop or
  alter local context, and `segment_keep_tail_steps=0` avoids duplication at the
  cost of hard boundaries.
- The next engineering step is to tune segment size/tail/prompt context:
  - `segment_max_cached_steps` sweep, likely 120/160/200/240;
  - small tail sweep, likely 0/8/16/32 steps;
  - optional previous-segment text as prompt context, if it improves boundary
    continuity without causing duplication.

Artifacts:

- `runs/jl_418957/segmented_streamer_first20_limit1.jsonl`
- `runs/jl_418957/segmented_streamer_first20_limit1.summary.json`
- `runs/jl_418957/segmented_streamer_first20_limit1_events/*.jsonl`
- `runs/jl_418957/segmented_streamer_full_limit1_left15s_seg200.jsonl`
- `runs/jl_418957/segmented_streamer_full_limit1_left15s_seg200.summary.json`
- `runs/jl_418957/segmented_streamer_full_wlk21_left15s_seg200.jsonl`
- `runs/jl_418957/segmented_streamer_full_wlk21_left15s_seg200.summary.json`

## Segmented Boundary Tuning: Prompt Context, Tail, Segment Length

Date: 2026-05-30

Status:

- H100 instance `418957` resumed as `418958`.
- Boundary tuning runs completed and downloaded.
- H100 instance `418958` paused.

Code change:

- Added optional dynamic segment prompt context to
  `SegmentedCachedFullHypothesisStreamer`.
- New CLI flags in `scripts/infer_cached_full_hypothesis.py` and
  `scripts/eval_cached_full_hypothesis.py`:
  - `--segment-prompt-context-words`;
  - `--segment-prompt-context-prefix`.
- When enabled, each active segment prompt is rebuilt with the last N words of
  completed transcript in the Qwen system context.
- Added unit coverage for:
  - trailing text context extraction;
  - dynamic Qwen prompt construction with audio placeholder expansion.

Validation:

```text
Local:
  python3 -m py_compile qwen3_streaming/cached_full_hypothesis.py \
    scripts/infer_cached_full_hypothesis.py \
    scripts/eval_cached_full_hypothesis.py
  python3 -m pytest -q
  58 passed, 2 skipped

H100 targeted:
  .venv/bin/python -m py_compile qwen3_streaming/cached_full_hypothesis.py \
    scripts/infer_cached_full_hypothesis.py \
    scripts/eval_cached_full_hypothesis.py
  .venv/bin/python -m pytest -q tests/test_cached_full_hypothesis.py \
    tests/test_native_realtime_model.py tests/test_realtime_config.py
  32 passed
```

Limit-5 boundary sweep:

```text
model: Qwen/Qwen3-ASR-0.6B
manifest: data/wlk_teacher_1p7b_v0/manifest.teacher.jsonl
limit: 5 full WLK audios
chunk_ms: 10000
left_context: 15s
right_context: 640ms
segment_finalize_mode: latest
```

| segment steps | tail steps | prompt ctx words | RTF | WER |
| ---: | ---: | ---: | ---: | ---: |
| 120 | 0 | 0 | 0.105 | 0.163 |
| 200 | 0 | 0 | 0.148 | 0.144 |
| 200 | 0 | 40 | 0.159 | 0.208 |
| 200 | 16 | 0 | 0.161 | 0.186 |
| 200 | 16 | 40 | 0.167 | 0.222 |
| 160 | 0 | 40 | 0.158 | 0.208 |
| 360 | 0 | 0 | 0.196 | 0.137 |

Interpretation:

- Prompting the next segment with previous transcript text hurts this Qwen ASR
  path. It increases repetitions and worsens WER on the limit-5 sample.
- Carrying an audio embedding tail also hurts with the current naive stitching;
  it likely duplicates boundary content.
- Short 10s-ish segments are fastest, but quality drops.
- 30s-ish segments (`segment_max_cached_steps=360`) are the best quality point
  in this sweep, at higher but still sub-real-time cost.

Full WLK 21-audio eval for the best quality candidate:

```text
segment_max_cached_steps: 360
segment_keep_tail_steps: 0
segment_prompt_context_words: 0
```

Results:

```text
ok: 21/21
total audio duration: 7105.5s
total latency: 1365.4s
RTF total: 0.1922
WER final mean: 0.1473
WER stable mean: 0.1736
cache_bound_violations: 0
segments_finalized_mean: 10.95
committed_revision_events_total: 0
```

Comparison with previous full WLK `seg200`:

| config | RTF total | WER final mean | segments finalized mean |
| --- | ---: | ---: | ---: |
| seg200, tail0, ctx0 | 0.1465 | 0.1534 | 16.71 |
| seg360, tail0, ctx0 | 0.1922 | 0.1473 | 10.95 |

Current recommendation:

- Keep prompt context disabled for now.
- Keep audio tail disabled until there is overlap-aware de-duplication.
- Use `seg200` as the low-latency default.
- Use `seg360` when quality matters more than latency; it is still comfortably
  faster than real time on H100.

Next useful work:

- Add overlap-aware boundary stitching before retrying audio tails.
- Try chunk-level finalization/commit with smaller `chunk_ms` once the server
  interface exists, because current 10s chunks make first display/commit coarse.
- Then wrap the segmented streamer in a minimal stateful serving API.

Artifacts:

- `runs/jl_418958/segmented_context_sweep_seg120_tail0_ctx0_limit5.summary.json`
- `runs/jl_418958/segmented_context_sweep_seg200_tail0_ctx0_limit5.summary.json`
- `runs/jl_418958/segmented_context_sweep_seg200_tail0_ctx40_limit5.summary.json`
- `runs/jl_418958/segmented_context_sweep_seg200_tail16_ctx0_limit5.summary.json`
- `runs/jl_418958/segmented_context_sweep_seg200_tail16_ctx40_limit5.summary.json`
- `runs/jl_418958/segmented_context_sweep_seg160_tail0_ctx40_limit5.summary.json`
- `runs/jl_418958/segmented_context_sweep_seg360_tail0_ctx0_limit5.summary.json`
- `runs/jl_418958/segmented_full_wlk21_left15s_seg360_tail0_ctx0.summary.json`

## 2026-05-31 - Qwen AR CE audio-adapter Stage A

Goal: test the GPU-first plan where the normal Qwen autoregressive decoder stays
frozen and the streaming-local-context audio path is trained with teacher-forced
transcript CE. This is meant to adapt the audio side to `left_context=8s`,
`right_context=640ms` before attempting shorter contexts or audio LoRA.

Implementation landed:

- `--alignment-loss qwen_ar_ce` in `scripts/train_realtime_tiny_asr.py`.
- Qwen ASR prompt + audio placeholders are used for teacher forcing.
- CE labels are active only on transcript assistant tokens; prompt/audio prefix
  positions are not labels.
- Qwen text decoder and LM head are frozen for this objective.
- Per-sample prompt language is read from manifests.
- Qwen audio LoRA support added separately via `--qwen-audio-lora-rank`.
  Default audio LoRA targets match the audio tower names:
  `q_proj,k_proj,v_proj,out_proj,fc1,fc2`.

Validation:

```text
local: python3 -m pytest -q -> 58 passed, 2 skipped
H100: python -m pytest -q -> 105 passed
H100 targeted qwen_ar/audio_lora tests -> passed
```

Smoke results:

```text
qwen_ar_ce smoke, 20 train / 4 eval, 3 steps:
first_train_loss: 4.6875
last_train_loss: 4.59375
eval_loss: 4.3359

audio LoRA smoke, rank=2:
loaded checkpoint with qwen_audio_lora_config: true
default matched audio LoRA modules after target fix: 108
```

Stage A run:

```text
run: runs/qwen_ar_ce_stageA_left8_adapter2x2048_v0
base: Qwen/Qwen3-ASR-0.6B
objective: qwen_ar_ce
left_context: 8s
right_context: 640ms
adapter: 2 layers, hidden_dim=2048, residual_scale=0.1
freeze: Qwen audio tower + Qwen text decoder/LM head
steps: 1500
batch_size: 2
lr: 1e-4
trainable_params: 13.6M

first_train_loss: 5.34375
last_train_loss: 3.3125
eval_loss: 3.3429
train qwen_ar_token_accuracy mean: 0.365
eval qwen_ar_token_accuracy mean: 0.370
```

WLK first20 eval, `left=8s`, `seg200`, `right=640ms`, 21 chunks of 20s:

| model | RTF total | WER final mean | WER stable mean | trigram repetition | cache violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline Qwen 0.6B | 0.5476 | 0.1884 | 0.2879 | 0.0016 | 0 |
| Stage A qwen_ar_ce | 0.3056 | 1.1145 | 0.9602 | 0.5048 | 0 |

Interpretation:

- The teacher-forced AR CE wiring works and optimizes the public eval loss.
- But this Stage A checkpoint is not promotable: WLK quality collapses into
  generic/repetitive text even though RTF improves.
- Do not continue to Stage B (`left=4s/2s`) from this checkpoint.
- Likely issue: training only a post-audio adapter with full teacher-forced CE
  changes the embedding distribution in a way the frozen Qwen decoder can exploit
  on labels but not robustly decode in streaming full-hypothesis mode.

Next recommendation:

- Retry with a much smaller adapter update before LoRA:
  `lr=1e-5`, fewer steps, and regularize against the baseline audio embeddings
  with an L2/KL-style distillation term.
- Add a WLK-first20 eval gate during training before running full 1500-step
  jobs.
- If that still fails, move adaptation deeper but constrained: low-rank audio
  LoRA with very small LR and a baseline-output preservation loss.

Artifacts kept locally:

- `runs/qwen_ar_ce_stageA_left8_adapter2x2048_v0/train_metrics.json`
- `runs/qwen_ar_ce_stageA_left8_adapter2x2048_v0/eval_predictions.jsonl`
- `runs/qwen_ar_ce_stageA_left8_adapter2x2048_v0/wlk_eval/left8_seg200_first20.jsonl`
- `runs/qwen_ar_ce_stage0_baseline/wlk_left8_seg200_first20.jsonl`

The failed heavy `model.pt` artifacts were removed from the H100 after metrics
were downloaded.

## 2026-06-03 - Qwen Audio Causal KV Backend Smoke

Goal:

- Add a first native Qwen audio backend that is append-only at the audio tower
  level: new mel frames are encoded, previous finalized audio states are reused,
  and no right context is used.
- This is a mechanical causal/KV validation smoke, not a promoted ASR-quality
  checkpoint.

Implementation:

- Backend: `qwen_audio_causal_kv`
- Model class: `Qwen3ASRRealtimeQwenAudioCausalModel`
- Encoder: `QwenAudioCausalKVEncoder`
- Training objective: `qwen_ar_ce`
- Streaming audio chunks: 8 mel frames, about 80 ms
- Trainable params in smoke: Qwen audio LoRA rank 4 over Q/K/V/O and MLP
- Frozen: Qwen audio base weights, text decoder, LM head

Validation:

```text
local:
PYTHONPATH=experiments/qwen3-causal python3 -m pytest -q experiments/qwen3-causal/tests
116 passed in 2.25s

H100:
decoder_backend=qwen_audio_causal_kv
train_steps=1
train_loss=3.40625
eval_loss=4.48681640625
eval qwen_ar_token_accuracy=0.2883
```

Interpretation:

- The causal audio KV path loads the real `Qwen/Qwen3-ASR-0.6B` audio tower.
- The backend can run a CUDA training/eval smoke without shape errors or NaNs.
- The append-only invariant is covered by unit tests with fake towers:
  finalized cached audio grows across chunks and `last_recomputed_context_frames`
  stays at `0`.
- This does not prove usable transcription quality. The smoke is only one step,
  and the eval token accuracy is low.

Current answer to "do we have a functional causal Qwen3-ASR?":

- Functional as an experimental causal audio backend: yes.
- Functional as a usable ASR model with acceptable WER: no, not yet.

Artifacts kept locally:

- `runs/qwen_audio_causal_kv_smoke_v0/train_metrics.json`
- `runs/qwen_audio_causal_kv_smoke_v0/run_config.json`
- `runs/qwen_audio_causal_kv_smoke_v0/realtime_config.json`
- `runs/qwen_audio_causal_kv_smoke_v0/realtime_model_meta.json`
- `runs/qwen_audio_causal_kv_smoke_v0/eval_predictions.jsonl`

The H100 instance `420813` was paused after downloading the lightweight
artifacts. The 2.2 GB smoke `model.pt` was not downloaded or promoted.

## 2026-06-03 - Causal KV Decode Controls + Adapter Distill Smokes

Goal:

- Evaluate whether the strict append-only `qwen_audio_causal_kv` backend can be
  pushed toward usable ASR with conservative audio adaptation.
- Add no-repeat/repetition controls to cached full-hypothesis decoding so WLK
  evaluation is not dominated by endless greedy loops.

Code change:

- `CachedFullHypothesisConfig` now exposes:
  - `repetition_penalty`
  - `no_repeat_ngram_size`
  - `max_consecutive_text_tokens`
- `infer_cached_full_hypothesis.py` and `eval_cached_full_hypothesis.py` expose
  matching CLI flags.
- Full-hypothesis Qwen greedy decoding applies the same repetition control
  helper already used by the frame-synchronous decoder.

Validation:

```text
local:
python3 -m py_compile qwen3_streaming/native_realtime_model.py \
  qwen3_streaming/cached_full_hypothesis.py \
  scripts/eval_cached_full_hypothesis.py \
  scripts/infer_cached_full_hypothesis.py

PYTHONPATH=experiments/qwen3-causal python3 -m pytest -q \
  experiments/qwen3-causal/tests/test_native_realtime_model.py \
  experiments/qwen3-causal/tests/test_cached_full_hypothesis.py

31 passed

H100:
py_compile passed
tests/test_native_realtime_model.py passed
```

WLK first20 teacher eval, limit 5, strict causal audio KV:

| model / decode | WER final | RTF | final tokens | trigram repetition | cache violations | max recomputed context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct causal greedy | 1.3907 | 2.4674 | 71.0 | 0.8986 | 0 | 0 |
| direct causal + no-repeat3, rep 1.15 | 0.9151 | 0.8642 | 15.0 | 0.0000 | 0 | 0 |
| adapter distill 50 + no-repeat3, rep 1.15 | 0.8585 | 1.0212 | 17.8 | 0.0000 | 0 | 0 |
| adapter mix300 + no-repeat3, rep 1.15 | 0.8829 | 1.0119 | 15.6 | 0.0000 | 0 | 0 |

Training smokes:

```text
adapter distill 50:
data: qwen_aligned_fleurs_tiny
trainable: zero-init post-audio adapter only
steps: 50
lr: 5e-6
loss_decreased: true
first_train_loss: 0.9311
last_train_loss: 0.9207
eval_loss: 4.2100
train_qwen_ar_token_accuracy: 0.3236

adapter mix300:
data: qwen_aligned_mix_16s_teacher_filter_v0
trainable: zero-init post-audio adapter only
steps: 300
lr: 5e-6
loss_decreased: false
first_train_loss: 0.9902
last_train_loss: 1.0939
eval_loss: 4.1702
train_qwen_ar_token_accuracy: 0.3025
```

Qualitative result:

- Direct causal recognizes short openings, then often loops:
  `Hello everyone... I'm from the universe...`
- No-repeat controls remove the runaway loops, but the hypotheses are still too
  short and hallucinated.
- The 50-step adapter smoke improves the 5-item WLK signal slightly, but not to
  a usable ASR level.
- The 300-step mixed-data adapter run does not improve over the 50-step smoke.

Current status:

- We have a mechanically functional strict causal audio path:
  `max_recomputed_context_frames=0` and `cache_bound_violations=0`.
- We do not yet have a functional causal ASR model. The best WLK first20
  teacher WER observed here is still `0.8585` on only five examples.

Decision:

- Do not promote any checkpoint from these runs.
- Do not continue adapter-only distillation as-is.
- Next useful work should target the actual bottleneck: the frozen Qwen decoder
  does not robustly interpret the causalized audio embeddings beyond the opening
  phrase. A stronger but constrained adaptation is needed, likely with:
  - decoder-side calibration or a tiny trainable bridge;
  - stronger teacher-forced generation checks during training;
  - WLK eval gates every short interval;
  - no LoRA over the full audio tower until the adapter path beats the direct
    causal baseline by a clear margin.

Artifacts kept locally:

- `runs/qwen_audio_causal_kv_direct_decode_v0/`
- `runs/qwen_audio_causal_kv_adapter_distill50_v0/`
- `runs/qwen_audio_causal_kv_adapter_mix300_v0/`

The H100 instance `420815` was paused. Non-promoted `model.pt` files were
deleted from `/tmp`.

## 2026-06-03 - Causal KV WLK Chunk Fine-Tune Smoke

Goal:

- Test whether training directly on WLK-style 16s chunks can make the strict
  append-only `qwen_audio_causal_kv` path usable without changing code again.
- Keep WLK as a diagnostic/eval set and avoid promoting a checkpoint unless
  streaming quality improves clearly.

Data:

```text
train chunks: 338
eval chunks: 107
audio chunks written remotely: 445
alignment source: Qwen3 ForcedAligner manifests
```

Training:

```text
output: runs/qwen_audio_causal_kv_wlk_last2_300_v0/
backend: qwen_audio_causal_kv
loss: qwen_ar_context_distill
steps: 300
lr: 1e-6
trainable: last 2 Qwen audio layers + zero-init adapter
decoder / LM head: frozen
first_train_loss: 3.5698
last_train_loss: 2.7273
eval_loss: 3.9705
eval token accuracy: 0.3156
```

WLK chunks eval, 20 held-out chunks, no-repeat3 + repetition penalty 1.15:

```json
{
  "wer_final_mean": 0.8962655764363544,
  "realtime_factor_mean": 0.9593619788569075,
  "cache_bound_violations": 0,
  "max_recomputed_context_frames": 0,
  "first_display_sec_mean": 0.32,
  "first_commit_sec_mean": 5.722352941176471,
  "trigram_repetition_ratio": 0.0
}
```

Qualitative result:

- The causal cache invariant still holds: no past audio context is recomputed.
- The anti-repeat decoding keeps trigram repetition at zero.
- The model is still not a functional ASR model. Outputs remain short and
  hallucinated, for example: `The the me is all, so we love.`
- Teacher-forced training improves loss, but greedy streaming decode does not
  acquire reliable lexical identity.

Decision:

- Do not promote the checkpoint.
- Do not call the current branch a working causal Qwen3-ASR model.
- Treat the branch as an experimental causal runtime plus negative training
  evidence. The next substantial step should revisit the modeling assumption,
  not run longer versions of this same objective.

Artifacts kept locally:

- `runs/qwen_audio_causal_kv_wlk_last2_300_v0/train_metrics.json`
- `runs/qwen_audio_causal_kv_wlk_last2_300_v0/eval_predictions.jsonl`
- `runs/qwen_audio_causal_kv_wlk_last2_300_v0/chunks16_eval_limit20_norepeat3.jsonl`
- `runs/qwen_audio_causal_kv_wlk_last2_300_v0/chunks16_eval_limit20_norepeat3.summary.json`

The H100 instance `420839` was paused. Non-promoted `model.pt` files were
deleted from `/tmp`.

## 2026-06-03 - Strict Causal Audio LoRA Diagnostics

Goal:

- Test whether LoRA on the Qwen audio tower can make the strict append-only
  `qwen_audio_causal_kv` path learn lexical identity while keeping the Qwen text
  decoder frozen.
- Use only existing training/eval code in this branch.

Common setup:

```text
backend: qwen_audio_causal_kv
loss: qwen_ar_ce
trainable: Qwen audio tower LoRA rank 8 + zero-init audio adapter
decoder / LM head: frozen
streaming train chunk frames: 32
eval chunk: 320 ms
decode controls: repetition_penalty=1.15, no_repeat_ngram_size=3
```

Public mix smoke:

```text
output: runs/qwen_audio_causal_kv_lora8_mix120_ce_cf32_v0/
data: qwen_aligned_mix_16s_teacher_filter_v0
steps: 120
lr: 1e-5
trainable_params: 15,955,968
first_train_loss: 4.5625
last_train_loss: 2.7378
eval_loss: 3.7982
eval token accuracy: 0.3291
```

WLK chunks eval, 20 held-out chunks:

```json
{
  "wer_final_mean": 0.9099421475504256,
  "realtime_factor_mean": 1.0005624510348599,
  "cache_bound_violations": 0,
  "max_recomputed_context_frames": 0,
  "trigram_repetition_ratio": 0.0
}
```

WLK-domain diagnostic smoke:

```text
output: runs/qwen_audio_causal_kv_lora8_wlk200_ce_cf32_v0/
data: wlk_chunks16_teacher_aligned_split_v0
steps: 200
lr: 1e-5
trainable_params: 15,955,968
first_train_loss: 4.7813
last_train_loss: 3.4339
eval_loss: 3.6288
eval token accuracy: 0.3370
```

WLK chunks eval, 20 held-out chunks:

```json
{
  "wer_final_mean": 0.9072655230996075,
  "realtime_factor_mean": 0.9936728690488577,
  "cache_bound_violations": 0,
  "max_recomputed_context_frames": 0,
  "trigram_repetition_ratio": 0.0
}
```

Qualitative result:

- Both runs learn under teacher forcing: loss decreases and token accuracy is
  around 0.33.
- Neither run improves streaming ASR quality. WER stays around 0.91, worse than
  the previous strict causal WLK smoke at 0.8963 and far from usable.
- Outputs are still short/hallucinated. Examples include:
  - `What is my special place?`
  - `So what is first? First, you know.`
- The cache invariant remains good: no past context recomputation and no cache
  bound violations.

Decision:

- Do not promote either checkpoint.
- Audio LoRA + frozen Qwen decoder is not enough for strict causal Qwen3-ASR.
- The current evidence points to a deeper mismatch: the offline Qwen decoder is
  not robustly decoding the strict causalized audio representation even when the
  audio side is adapted. The next meaningful step likely requires either
  decoder-side adaptation or a model/objective designed for stream-synchronous
  emission, which cannot be tested further with the current `qwen_ar_*` code path
  because it freezes the Qwen text decoder.

Artifacts kept locally:

- `runs/qwen_audio_causal_kv_lora8_mix120_ce_cf32_v0/`
- `runs/qwen_audio_causal_kv_lora8_wlk200_ce_cf32_v0/`

The H100 instance `420856` was paused. Non-promoted `model.pt` files were
deleted from `/tmp`.

## 2026-06-03 - Backend Diagnostic: Recompute Window vs Append-Only

Goal:

- Separate two possible failure modes:
  - Qwen audio needs future/right context and cannot work causally.
  - The strict append-only `qwen_audio_causal_kv` backend loses the behavior
    Qwen gets from recomputing its original local/offline audio tower.

Setup:

```text
model: Qwen/Qwen3-ASR-0.6B
data: 20 held-out WLK 16s chunks
decode: repetition_penalty=1.15, no_repeat_ngram_size=3
chunk: 320 ms
left_context: 15s
```

Results:

| backend | right context | WER | RTF | max recomputed context frames | final tokens mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen_audio_surgery` | 640 ms | 0.1547 | 1.5256 | 1532 | 39.55 |
| `qwen_audio_surgery` | 0 ms | 0.2017 | 1.6940 | 1468 | 38.95 |
| `qwen_audio_causal_kv` | 0 ms | 0.9065 | 1.0147 | 0 | 15.15 |

Interpretation:

- Qwen does not collapse merely because output `right_context=0`.
- Important nuance: `qwen_audio_surgery/right=0` is not strict causal audio
  attention. It recomputes the whole local window through Qwen's original
  offline/bidirectional audio tower, then emits all frames up to the current
  audio time.
- That still gives usable quality: `WER=0.2017`.
- The strict append-only backend is the rupture point: same model, same eval,
  same output right context, but `WER=0.9065`.
- Therefore the next priority is not more LoRA training. It is understanding
  how much mutable history Qwen needs, then either:
  - implement a bounded mutable tail that recomputes only a small recent suffix;
  - or train a genuinely causal audio tower/decoder pair so old embeddings no
    longer need to be rewritten.

Technical gap:

- Existing tests prove append-only behavior using `FakeQwenAudioTower`.
- They do not prove quality parity with real Qwen, and strict parity with
  `qwen_audio_surgery` is not expected because surgery recomputes an offline
  bidirectional tower.
- The real question is now empirical: how small can the mutable/recomputed tail
  become before WER collapses?

Decision:

- Stop training strict append-only LoRA as-is.
- Next useful diagnostic: sweep bounded mutable tails, e.g. recompute only the
  last `0.5s/1s/2s/4s` while keeping older embeddings fixed, and compare WER to
  full 15s-window recompute.
- If a small tail works, that is the practical realtime path.
- If only a large tail works, strict no-recompute requires a real causal
  training objective/architecture, not a small LoRA on the existing frozen
  decoder path.

Artifacts kept locally:

- `runs/qwen_audio_backend_compare_wlk20_v0/surgery_r640.summary.json`
- `runs/qwen_audio_backend_compare_wlk20_v0/surgery_r0.summary.json`
- `runs/qwen_audio_backend_compare_wlk20_v0/causal_kv_r0.summary.json`
- Matching JSONL prediction files for all three runs.

The H100 instance `420865` was paused.

## 2026-06-01/02 - Stage0 Baseline Left-Context Sweep on Full WLK

Date of runs: 2026-06-01/02. Repatriated and documented: 2026-06-10 from
instances `wlk-gpu-check-in2` and `qwen-ar-distill-left12`.

Setup:

```text
model: Qwen/Qwen3-ASR-0.6B, untrained, cached full-hypothesis segmented streamer
manifest: data/wlk_teacher_1p7b_v0/manifest.teacher.jsonl (21 full WLK audios)
chunk_ms: 10000
right_context: 640ms
segment_max_cached_steps: 200
language: English (explicit)
```

Results:

| left context | WER final mean | RTF mean | cache violations |
| --- | ---: | ---: | ---: |
| 2s | 0.8502 | 0.0643 | 0 |
| 4s | 0.6798 | 0.0795 | 0 |
| 8s | 0.3243 | 0.1018 | 0 |
| 12s | 0.1575 | 0.1013 | 0 |
| 15s (jl_418957) | 0.1534 | 0.1462 | 0 |

Interpretation:

- Long-form quality saturates around `left_context=12s`: WER 0.1575 vs 0.1534
  at 15s, at roughly 30 percent lower RTF.
- `left12 / seg200 / chunk10s` is the new recommended v1 operating point.
- Below 8s the bounded window loses too much context on long talks; the curve
  is steep between 4s and 12s.

Artifacts:

- `runs/qwen_ar_ce_stage0_baseline/wlk_full_left{2,4,8,12}_seg200_chunk10s.{jsonl,summary.json}`

## 2026-06-01/02 - Preserve-Regularized Audio Adaptation (left8/left12)

Date of runs: 2026-06-01/02. Repatriated and documented: 2026-06-10.

Goal: follow-up to the 2026-05-31 recommendation. Train only the audio-side
projector/adapter with `qwen_ar_ce` plus an L2 preservation loss against the
frozen baseline audio embeddings (`--qwen-ar-audio-preserve-loss-weight`,
reference `projection`), to adapt the audio path to shorter left contexts
without collapsing WLK streaming quality.

Runs and WLK first20 evals (21 chunks, chunk 1s, seg200, English):

| run | lr | preserve weight | steps | WER final | WER stable |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline left8 (stage0) | - | - | - | 0.1884 | 0.2879 |
| left8 zero-adapter streaming | 5e-6 | 1000 | 200 | 0.1894 | 0.2926 |
| left8 preserve-proj streaming | 1e-5 | 100 | 200 | 1.3498 | 1.1189 |
| left8 preserve-proj | 1e-5 | 100 | 500 | 1.2493 | 0.8951 |

Full WLK left12 (21 full audios, chunk 10s, seg200):

| run | WER final | RTF |
| --- | ---: | ---: |
| baseline left12 (stage0) | 0.1575 | 0.1013 |
| left12 zero-adapter lr5e-6 preserve1000, 500 steps | 0.1589 | 0.0950 |

Interpretation:

- The preservation loss controls the failure mode but not the outcome:
  - strong preservation (weight 1000, lr 5e-6) keeps WLK quality intact and
    delivers no improvement; the trained model is functionally the identity;
  - weaker preservation (weight 100, lr 1e-5) collapses WLK streaming quality
    exactly like the unregularized Stage A run.
- Audio-side-only adaptation with a frozen Qwen decoder has now failed across
  four mechanisms: plain adapter CE, audio LoRA, context distillation, and
  preserve-regularized CE. This line is closed; see the 2026-06-03 backend
  diagnostic for the structural explanation.

Artifacts (lightweight only; `model.pt` checkpoints stay on the paused
instances, none are promoted):

- `runs/qwen_ar_ce_stageA_left8_streaming_zero_adapter_lr5e6_preserve1000_200_v0/`
- `runs/qwen_ar_ce_stageA_left8_streaming_preserve_proj_lr1e5_200_v0/`
- `runs/qwen_ar_ce_stageA_left8_preserve_proj_lr1e5_500_v0/`
- `runs/qwen_ar_ce_stageA_left12_streaming_zero_adapter_lr5e6_preserve1000_500_v0/wlk_eval/`
- `runs/qwen_ar_ce_smoke_v0/`, `runs/qwen_ar_ce_audio_lora_smoke_v0/`

## 2026-06-10 - Audit: Context-Distill Stage A/B Eval Was Language-Corrupted

The previously imported context-distillation comparison
(`runs/qwen_ar_context_distill_stageB_summary_table.json`, runs
`qwen_ar_context_distill_stageA_left8_6_lora_v0` and
`qwen_ar_context_distill_stageB_left6_4_lora_v0`) reported:

```text
baseline left8/6/4: WER 0.4456 / 0.4546 / 0.4323
stageA  left8/6/4:  WER 0.4246 / 0.4295 / 0.4289
stageB  left6/4:    WER 0.5131 / 0.5261
```

Audit result: these WERs are invalid as absolute numbers. The eval ran without
an explicit English language directive in the Qwen prompt. On WLK items with
accented non-native English speakers, the model auto-switched to French
(`"Hi, je suis Myra et aujourd'hui je vais parler de notre article..."` for
English audio). Per-item comparison against the clean stage0 baseline shows the
same decode config and near-identical hypotheses on unaffected items
(EqmWoxNDIr: 0.1739 in both), while affected items differ by up to 0.8 WER
(rxrToXvRyM: 0.906 corrupted vs 0.113 clean). The clean English-prompt baseline
at left8/chunk1s is WER 0.1884, far below every number in the corrupted table.

Consequences:

- The earlier note "Stage A context distillation improved WLK first-20
  slightly at left=8s/6s" is withdrawn. No promotable conclusion can be drawn
  from these runs in either direction.
- Stage B left6/4 regression is also unproven, although consistent with the
  preserve-run evidence that shorter contexts plus audio-only training fail.
- Any future eval must pass an explicit `--language` (or per-item manifest
  language). Auto language detection is unstable on accented WLK audio and
  silently destroys WER comparability.

## 2026-06-10 - Repatriation Sweep and Instance State

All remaining qwen3 instances were resumed, inspected, drained of lightweight
artifacts, and paused again:

- `qwen-context-distill-smoke` (L4, 422507 -> 424402): staging box; RUNS.md and
  runs/ strictly older than the local import; nothing new repatriated. Holds
  heavy tarballs (`wlk_audio_full.tgz`, dataset archives) that are reproducible.
- `qwen-ar-distill-left12` (H100, 419641 -> 424403): left12 preserve run
  `wlk_eval` and stage0 `wlk_full_left12` baseline repatriated.
- `wlk-gpu-check-in2` (H100, 419199 -> 424404): left8 preserve runs, smoke
  runs, and the full stage0 left-context sweep repatriated. Still holds the
  project venv, FLEURS/mix aligned datasets with audio (~2GB), WLK audio/chunk
  data, and non-promoted `model.pt` checkpoints.
- `alignatt-enzh-clean` (A100): unrelated project, untouched.

Local `experiments/qwen3-causal` in WhisperLiveKit is now the single source of
truth for code, manifests, run metrics, and this run log.

## 2026-06-10 - Re-Audit Against MCIF Human References (Phase 0)

Motivation: all historical evals scored against Qwen3-ASR-1.7B teacher
transcripts with minimal normalization (lowercase + whitespace; punctuation
counted as errors). Human references exist in the MCIF long-form set
(`ref/en.txt`, 919 segment lines mapped 1:1 to `audio-segments.yaml`, 21 wavs,
contiguous groups, monotonic offsets — verified).

New tooling:

- `scripts/build_mcif_reference_manifest.py` ->
  `data/mcif_refs/manifest.human.jsonl` (per-wav concatenated human reference
  plus per-segment offsets, so time-windowed references for chunked evals can
  be derived).
- `scripts/rescore_jsonl.py`: rescores existing per-item JSONLs under
  2 reference sets (teacher = row's own `reference`, human = MCIF) x
  2 normalizers (legacy, Whisper EnglishTextNormalizer via
  `whisper-normalizer`). No GPU needed.

Full WLK 21 audios, chunk 10s, seg200 (stage0 baseline curve):

| left context | WER orig (teacher, legacy) | teacher + whisper-norm | human + legacy | human + whisper-norm |
| --- | ---: | ---: | ---: | ---: |
| 2s | 0.8502 | 0.8281 | 0.8514 | 0.8287 |
| 4s | 0.6798 | 0.6419 | 0.6853 | 0.6460 |
| 8s | 0.3243 | 0.2636 | 0.3440 | 0.2768 |
| 12s | 0.1575 | 0.0937 | 0.1867 | 0.1101 |

First-20s chunks (windowed human refs, midpoint segment selection) and
preserve runs:

| run | teacher+whisper | human+whisper |
| --- | ---: | ---: |
| baseline left2 first20 | 0.1054 | 0.1662 |
| baseline left4 first20 | 0.1093 | 0.1699 |
| baseline left8 first20 | 0.1232 | 0.1693 |
| preserve zero-adapter lr5e-6 left8 | 0.1258 | 0.1737 |
| preserve proj lr1e-5 200 | 1.3113 | 1.3142 |
| preserve proj lr1e-5 500 | 1.1903 | 1.1819 |
| left12 preserve, full WLK | 0.0950 | 0.1129 |
| left12 baseline, full WLK | 0.0937 | 0.1101 |

Verdicts:

- The left-context curve shape and the left12 saturation HOLD under human
  references and proper normalization. left12/seg200 remains the operating
  point.
- The true quality is better than historically reported: WER 0.110 vs human
  references (0.094 vs teacher) instead of the headline 0.158 — legacy
  normalization (punctuation/case) inflated WER by roughly 30-40 percent
  relative at the good end of the curve.
- Teacher references were a reasonable proxy: human-ref WER tracks teacher-ref
  WER within ~0.01-0.02 on full-length audio.
- Preserve-run verdicts survive re-scoring: lr5e-6/weight1000 is the identity
  (0.1737 vs baseline 0.1693), lr1e-5/weight100 collapses (1.18-1.31), and the
  trained left12 run adds nothing over baseline (0.1129 vs 0.1101). The
  audio-side-only adaptation line stays closed.
- Caveat: windowed first20 human references are midpoint-based, so chunk-edge
  words inflate absolute WER slightly; use them for relative comparisons.

Eval hardening:

- `eval_cached_full_hypothesis.py` and `infer_cached_full_hypothesis.py` now
  REQUIRE `--language` (no silent default) and the eval script accepts
  `--reference-field` (e.g. `human_text`).

Open items deferred to the GPU session: left15 per-item JSONLs
(`jl_418957/418958`) only exist on `wlk-gpu-check-in2`, so the left12-vs-left15
confirmation against human refs and the offline full-file upper bound
(0.6B/1.7B) are still pending.

Note: the previously committed
`runs/qwen_ar_ce_stage0_baseline/wlk_full_left12_seg200_chunk10s.jsonl` was a
6-item partial run from instance 424404 that overwrote the complete 21-item
file from 424403 during repatriation; the full file is restored in this
commit. All numbers above use the 21-item file.

## 2026-06-10 - Dead Code Cleanup (Phase 1)

Removed the abandoned experiment lines from the workspace (history stays in
git and in this log):

- modules: `qwen3_streaming/ctc.py`, `qwen3_streaming/rnnt.py`;
  `realtime_targets.py` trimmed to `WordAlignment` +
  `heuristic_word_alignments` (still used by manifest prep).
- scripts: `train_realtime_tiny_asr.py` (4106 lines), `train_realtime_smoke.py`,
  `eval_realtime_checkpoint.py`, `infer_realtime_checkpoint.py`,
  `sweep_realtime_decoding.py`, `train_qwen3_sft.sh`, `prepare_public_jsonl.py`.
- tests of those lines: `test_ctc.py`, `test_realtime_targets.py`,
  `test_training_losses.py`, `test_sweep_realtime_decoding.py`; the scratch
  encoder / LoRA / save-load tests inside `test_native_realtime_model.py`.
- `native_realtime_model.py` trimmed 3055 -> ~1270 lines: deleted the scratch
  causal encoder stack, `CachedDecoder`, `Qwen3ASRRealtimeNativeModel`,
  LoRA wrappers, CTC/RNNT heads, training forwards, `stream_chunk`,
  checkpoint save/load, and `load_realtime_model`. Kept: surgery + causal-KV
  encoders/models, the cached-audio decode entry points
  (`init_cached_audio_decode_state` / `append_audio_to_cache` /
  `generate_full_hypothesis_from_cached_audio`), repetition-control helpers,
  and transformers registration. The base model `__init__` now requires an
  explicit `audio_encoder`/`adapter`/`audio_backend`.
- eval/infer scripts lost their `--checkpoint` path (no promoted checkpoints
  exist; training is gone).

Deliberate deviations from the cleanup list: kept
`tune_stable_commit_from_events.py` and its test (it tunes the validated
stable-commit path from event JSONLs), and kept the trimmed
`realtime_targets.py` for data tooling.

Validation: `PYTHONPATH=. python3 -m pytest -q tests` -> 64 passed; pyflakes
clean over `qwen3_streaming/`, `scripts/`, `tests/`.

README.md rewritten around the validated v1 path, the closed adaptation line,
and the open mutable-tail question.

## 2026-06-10 - Promotion: qwen3-streaming Backend in WhisperLiveKit (Phase 2)

The validated runtime was promoted into `whisperlivekit/qwen3_streaming/` as a
first-class backend (`--backend qwen3-streaming`):

- `streamer.py` / `stable_commit.py`: ports of the validated cached
  full-hypothesis streamers and stable-commit logic.
- `model.py`: surgery-only inference subset of `native_realtime_model.py`
  (bounded-recompute encoder; the causal-KV encoder stays in experiments for
  the mutable-tail sweep).
- `features.py`: new `StreamingMelExtractor` — incremental Whisper-mel with
  sample-exact frame parity vs one-shot extraction (window-local clamp is the
  one documented approximation).
- `asr.py` / `online.py`: shared model holder (HF Transformers, CUDA/MPS/CPU,
  explicit-language required) and per-session online processor (self-pacing
  decode cadence, interpolated word timestamps, append-only emission across
  segment rollovers).
- Wiring: config fields, CLI group `--qwen3-streaming-*`, core routing,
  `qwen3-streaming` pip extra (uv conflict with voxtral-hf).

Validation:

```text
pytest tests/ -> 116 passed, 1 skipped
  (new: 21 streamer/stable-commit, 4 mel parity, 12 processor contract)
E2E on Apple Silicon (MPS, fp16, Qwen/Qwen3-ASR-0.6B, real audio through the
full WLK pipeline): 1 passed in 17.6s, WER < 0.35 on librispeech_short,
monotonic line timestamps.
```

Defaults encode the re-audited operating point: left12 / right 640ms / seg200
/ chunk 2s with self-pacing.

## 2026-06-10 - H100 Session: Mutable-Tail Sweep, Offline Baselines, Latency (Phase 3)

Instance `wlk-gpu-check-in2` (424404 -> 424446), runs `r_bdd820b3` and
`r_bec09e55`. Code synced from the cleaned local workspace; 22 targeted tests
passed on the H100 before the experiments.

### Bounded mutable-tail sweep — the causal question is settled

Protocol identical to the 2026-06-03 backend diagnostic: 20 held-out WLK 16s
chunks, `qwen_audio_causal_kv`, chunk 320ms, left context 15s, decode controls
(repetition penalty 1.15, no-repeat-3), explicit English. The new
`--qwen-audio-mutable-tail-sec` keeps encoder steps younger than T
re-computable each chunk over the frozen per-layer KV prefix.

| mutable tail | WER final | RTF | max recomputed context frames |
| ---: | ---: | ---: | ---: |
| 0s (strict) | 0.9065 | 0.898 | 0 |
| 0.5s | 0.8960 | 0.913 | 56 |
| 1s | 0.9023 | 0.936 | 104 |
| 2s | 0.9020 | 0.931 | 208 |
| 4s | 0.9032 | 0.984 | 416 |
| 8s | 0.9050 | 0.924 | 832 |
| 12s | 0.9050 | 0.938 | 1248 |
| 16s (full recompute) | 0.9050 | 0.944 | 1568 |

The curve is FLAT. tail=0 reproduces the 06-03 strict baseline exactly
(0.9065), and even recomputing the entire 16s window on every chunk under the
causal mask (tail=16) stays at 0.905 — while the surgery backend (bidirectional
attention within the window, same zero output right-context) holds 0.20.

**Verdict: the failure of the strict-causal path is the causal attention mask
itself, not the absence of recompute.** Qwen3-ASR's audio tower requires
bidirectional intra-window attention. No inference-time engineering recovers
it; a genuinely causal Qwen3-ASR requires training (joint distillation of a
causal-masked tower + decoder from the offline teacher). The mutable-tail
engineering shortcut is dead; the bounded-window surgery path is the
practical realtime design.

### Offline full-file baselines vs MCIF human references

Official `qwen_asr.Qwen3ASRModel.transcribe`, one pass over each full WLK
file (277-425s), explicit English, scored with Whisper normalization:

| model | WER (human, whisper-norm) | WER (human, legacy) | mean latency |
| --- | ---: | ---: | ---: |
| Qwen3-ASR-0.6B offline one-pass | 0.2075 | 0.2660 | 56.7s |
| Qwen3-ASR-1.7B offline one-pass | 0.1202 | 0.1736 | 83.6s |

Naive one-pass long-form decoding degrades severely (drift/skips over 5-7
minute files). The segmented streamer at 0.6B (0.0843 below) beats both — the
segmentation is not a streaming compromise, it is also the better long-form
decoding strategy.

### Latency / chunk-size eval, surgery left12/seg200, full WLK 21 audios

| config | WER (teacher, legacy) | WER (human, whisper) | RTF | first display | first commit |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk 10s (stage0 ref) | 0.1575 | 0.1101 | 0.10 | 10.0s | 30.0s |
| chunk 2s | 0.1367 | 0.0843 | 0.29 | 2.0s | 18.7s |
| chunk 1s (limit 5) | 0.1202 | 0.0705 | 0.58 | 1.0s | 14.0s |

Shorter chunks improve BOTH latency and quality (more frequent hypothesis
updates make the stable-prefix commit and segment boundaries finer).
Committed-revision events: 3 total across 7105s of audio at chunk 2s;
cache_bound_violations 0 everywhere. The promoted backend default (chunk 2s
with self-pacing) is validated: WER 0.084 vs human references at RTF 0.29 on
H100.

### left15 vs left12 against human references (Phase 0 leftover)

Per-item JSONLs from jl_418957/418958 repatriated and rescored:

| config | WER (human, whisper-norm) |
| --- | ---: |
| left15 / seg200 / chunk10s | 0.1083 |
| left15 / seg360 / chunk10s | 0.1090 |
| left12 / seg200 / chunk10s | 0.1101 |

left12 == left15 within noise; the left12 default stands.

### Updated overall picture (0.6B unless noted, human refs, whisper norm)

```text
offline 0.6B one-pass:            0.2075
offline 1.7B one-pass:            0.1202
streaming left12/seg200/chunk10s: 0.1101   RTF 0.10
streaming left12/seg200/chunk2s:  0.0843   RTF 0.29   display 2.0s
streaming left12/seg200/chunk1s:  0.0705*  RTF 0.58   display 1.0s  (*5 files)
strict causal (any mutable tail): ~0.90    -- closed: needs training
```

Artifacts: `runs/jl_20260610/` (sweep JSONLs+summaries, latency evals, offline
baselines, logs). Instance 424446 paused after the session.

## 2026-06-10 - D0: Untrained Block-Bidirectional Causal-KV Eval

The strict-causal goal is active again. D0 measures the standard
streaming-encoder attention pattern — bidirectional within the processed
block, causal KV to the frozen prefix, append-only, latency = block size —
without any training (`--qwen-audio-block-bidirectional`). Same protocol as
the mutable-tail sweep (20 held-out WLK 16s chunks, left 15s, decode
controls, explicit English). Instance 424446 -> 424637, run `r_acdf90c2`.

| config (untrained, append-only) | WER final | RTF |
| --- | ---: | ---: |
| strict causal, 320ms (reference) | 0.9065 | 0.898 |
| strict causal, block 1s (control) | 0.9083 | 0.289 |
| strict causal, block 2s (control) | 0.9032 | 0.154 |
| bidir block 320ms | 0.7868 | 1.179 |
| bidir block 1s | 0.7239 | 0.412 |
| bidir block 2s | 0.6742 | 0.215 |
| bidir block 1s + mutable tail 1s | 0.6498 | 0.425 |
| bidir block 2s + mutable tail 2s | 0.6623 | 0.222 |
| surgery bidirectional window 15s (reference) | 0.20 | — |

Findings:

- The strict-mask controls stay at ~0.90 for every block size: the gains
  below are purely from intra-block bidirectionality, confirming the
  mutable-tail verdict from the other direction.
- Untrained block-bidirectionality recovers ~28 percent relative
  (0.91 -> 0.65 at block 1s + tail 1s), monotone in block size.
- It is NOT sufficient untrained: 0.65 is far from the 0.20 bidirectional
  window. Training (D1) is required.

Decision: D1 (causal/block-bidirectional tower distilled toward offline
tower embeddings, audio-only data, teacher computed on the fly, frozen
decoder as the eval gate) starts from the block-bidir + mutable-tail
execution (0.65) rather than strict causal (0.91) — about a third of the
gap is closed for free, and the residual training task is to teach the
tower to encode block-boundary context, not to relearn attention from
scratch.

Artifacts: `runs/jl_d0_blockbidir/`. Instance 424637 paused.

## 2026-06-10 - D1 Smoke: Tower Distillation GATE PASSED

Run `r_715553ad` on instance 424647 (paused after). Student = audio tower
under the block-bidirectional streaming mask (96-frame blocks = 0.96s, left
15s), trained with the parity-proven parallel forward
(`qwen3_streaming/tower_distill.py`); teacher = frozen original tower,
offline, computed on the fly; loss = masked MSE + 0.5 x cosine distance on
output embeddings; data = LibriSpeech train-clean-100 streamed, audio only.

```text
steps: 3000   batch: 8   lr: 1e-5 (200 warmup)   ~20 min H100 fp32
audio seen: ~67 hours-equivalent
gate: frozen-decoder streaming WER, 10 held-out WLK chunks, chunk 960ms
```

| step | gate WER (frozen decoder) | cos distance |
| ---: | ---: | ---: |
| 0 (untrained) | 0.7554 | — |
| 500 | 0.6822 | 0.122 |
| 1000 | 0.5346 | — |
| 1500 | 0.4686 | — |
| 2000 | 0.4596 | — |
| 2500 | **0.4225** | — |
| 3000 | 0.4255 | 0.054 |

Verdict: GATE PASSED (target < 0.5, near the < 0.4 stretch goal). Matching
offline embeddings transfers directly to frozen-decoder streaming WER:
-44 percent relative in 20 GPU-minutes on 100h of audio, with the decoder,
LM head and adapter untouched. The curve is still descending at 2500 and
flattens only at this tiny data scale (LS-clean-100 loops).

This validates the D1 mechanism. The remaining gap to the bidirectional
window (0.42 -> 0.20 untrained-window reference, then toward the 0.084
production point) is now a scale problem: more hours, more diversity
(EN+FR), an LR schedule, and afterwards D2 (CE + decoder LoRA co-adaptation).

Artifacts: `runs/jl_d1_smoke/{history.json,final_metrics.json}` local;
`runs/jl_d1_smoke/tower_best.pt` (746MB, step 2500, gate 0.4225) kept on the
paused instance.

## 2026-06-10 - D1 Full EN: Gate 0.2492, Distillation Scales

Run `r_1dd1c3c2` on instance 424662 (paused after). Resumed from the smoke
checkpoint (gate 0.4225); LibriSpeech 960h (clean-100 + clean-360 + other-500,
shuffled interleave, audio only); 60k steps, batch 8, lr 2e-5 warmup 500 with
cosine decay to 2e-6; ~2.8h H100 fp32 (~6.4 steps/s, ~1.3 epochs).

| step | gate WER (frozen decoder, 10 WLK chunks, 960ms) |
| ---: | ---: |
| 0 (= smoke best) | 0.4225 |
| 5000 | 0.3257 |
| 10000 | 0.3457 |
| 15000 | 0.2949 |
| 20000 | 0.3045 |
| 25000 | 0.2831 |
| 30000 | 0.2849 |
| 35000 | 0.2595 |
| 40000 | 0.2958 |
| 45000 | 0.2613 |
| 50000 | 0.2593 |
| 55000 | 0.2647 |
| 60000 | **0.2492** |

Verdict:

- Target <= 0.30 cleared by 15k; the LR decay kept paying to the very end —
  best gate is the final step. Embedding distillation has NOT plateaued at
  this scale.
- The strict-causal trajectory so far, all on the same 10-chunk gate:
  strict 0.91 -> block-bidir untrained 0.755 -> smoke 0.4225 -> full EN
  0.2492. The untrained bidirectional-window reference (0.20) is nearly
  matched by a fully append-only, 1s-latency tower with the decoder
  untouched.
- Remaining gap candidates, in order: D2 (teacher-forced CE + decoder LoRA
  co-adaptation, the decoder has never seen these embeddings), more data
  diversity (FR via MLS, spontaneous speech), longer training (the curve was
  still descending).

Artifacts: `runs/jl_d1_full_en/{history.json,final_metrics.json}` local;
`runs/jl_d1_full_en/tower_best.pt` (746MB, step 60000, gate 0.2492) on the
paused instance.

## 2026-06-10 - D2a: Decoder LoRA CE Overfits — Decoder Is Not the Bottleneck

Run `r_9a56ed44` on instance 424784 (stopped at ~step 1500 of 4000, machine
paused). LoRA r16/a32 on all decoder projections (10.1M trainable, 196
modules), teacher-forced CE on the existing teacher manifests (3,829 rows,
~15h: qwen_aligned_mix_16s_teacher_filter_v0 + WLK chunks train split),
frozen D1 tower (`tower_best.pt`, gate 0.2492).

```text
step 0    gate 0.2492   (exact LoRA no-op resume sanity)
step 25   train acc 0.9445
step 500  gate 0.2669   train acc ~0.97
step 1000 gate 0.2556   train acc ~0.99
step 1500 train acc 1.0000, loss 0.005  -> stopped (pure memorization)
```

Findings:

- Teacher-forced token accuracy was already 0.92-0.96 BEFORE any training:
  the decoder largely understands the distilled causal embeddings. CE has
  almost nothing to teach it except style memorization of 15h of text.
- The gate degraded while train accuracy saturated — immediate overfit on
  the small CE set, exactly the failure mode anticipated in the plan risks.

Verdict: decoder co-adaptation is NOT the current bottleneck. The remaining
gap (0.2492 vs 0.20 window-untrained vs ~0.13 window at this chunk size) is
on the tower/embedding side, where D1 was still descending at 60k steps.

Decision: do not pursue CE-on-small-data. Next lever is D1-continue at
larger scale and diversity — resume the tower from `tower_best.pt` with a
fresh LR cycle and add Multilingual LibriSpeech French (serves the FR goal
simultaneously). Pseudo-label CE at LS scale stays as a later option if the
tower path saturates above target.

Artifacts: `runs/jl_d2a_lora/history.json` local; no checkpoint promoted.

## 2026-06-10 - WS1: D1 Checkpoint Eval — Position Extrapolation Collapse Found

Instance 424892, run `r_8191bfd0`. Checkpoint `runs/jl_d1_full_en/tower_best.pt`
loaded via `--tower-state-dict`, bf16, decode controls rp1.15/ngram3, English.

| eval | WER (teacher, legacy) | RTF |
| --- | ---: | ---: |
| gate reproduction (10 chunks, 960ms) | 0.2512 | 0.55 |
| 20 chunks, 960ms | 0.2428 | 0.53 |
| 20 chunks, 1920ms | 0.2445 | 0.31 |
| 20 chunks, 960ms, position offset 1300 | **0.9367** | 0.30 |
| 20 chunks, 960ms, position offset 4000 | **0.8629** | 0.33 |
| full files limit 5, 960ms, seg200 | **0.9451** | 0.17 |
| full files limit 5, 1920ms, seg200 | **0.9525** | 0.09 |

Findings:

- bf16 drift vs the fp32 training gate is negligible (0.2512 vs 0.2492).
- The 96-frame-trained tower already serves at 1920ms blocks with no loss
  (0.2445 vs 0.2428) — good news for the 2s operating point.
- **Position extrapolation collapse confirmed**: the causal encoder assigns
  global positions (`emitted_steps`), trained only on 0-200. At offsets
  1300/4000 — and therefore on any audio beyond ~16s — WER collapses to ~0.9.
  The long-file evals fail for exactly this reason, not segmentation.
- The D0 untrained block-bidir results and all D1 gates were silently
  protected by 16s chunks; this is the first probe outside the trained range.

Decision: position-offset augmentation is now mandatory in the D1-continue
run. Implemented: `position_offset` in the parallel training forward
(extrapolation parity-tested against the encoder, including streaming-at-
offset-4000 == training-at-offset-4000), per-step log-uniform offset sampling
(`--position-offset-max 6000 --position-offset-prob 0.5`, teacher stays at 0
— forces position-invariant embeddings), and a third gate at offset 4000.

Corpus-2 note: voxpopuli (script dataset) is dead under datasets 4.8.5;
peoples_speech and mls_eng both hung on HF metadata resolution from the
instance. WS2 runs on reshuffled LS-960 alone; diversity retry later.

Artifacts: `runs/jl_ws1/`. Checkpoint backed up to
`~/Downloads/qwen3_checkpoints/` locally (no HF auth on the machine).

## 2026-06-11 - WS2 Interrupted Mid-Run (Instance Paused Externally)

WS2 ("D1-continue": mixed blocks 96/192 + position-offset augmentation
log-uniform [1,6000] @ p=0.5 + triple gate) ran as two parts on instance
424892:

- Part 1 (`r_e1a643fd`, lr 1e-5 cosine, target 60k steps): died at ~step
  20.5k on a transient HF CDN 408 while streaming LibriSpeech (no retry in
  the dataloader — since fixed: the epoch stream now restarts on transient
  errors). Checkpoints saved at step 20000.
- Part 2 (`r_e446787b`, resumed from part-1 `tower_last.pt`, lr 7.5e-6 ->
  1e-6, target 40k more steps): healthy at ~step 600+ (6.85 steps/s, offset
  loss spikes visibly shrinking), then the instance was paused externally
  around 00:30-01:15 local.

Gate trajectory so far (10 chunks, fp32, teacher refs, legacy norm):

| step (cumulative) | gate@960 | gate@1920 | gate@960@off4000 |
| ---: | ---: | ---: | ---: |
| 0 (D1 resume) | 0.2492 | 0.2651 | 0.8828 |
| 5000 | 0.2595 | 0.2652 | 0.4066 |
| 10000 | 0.2656 | 0.2615 | 0.3779 |
| 15000 | 0.2536 | 0.2490 | 0.4093 |
| 20000 | 0.2514 | **0.2408** | **0.3681** |

Reading at the interruption point: base quality intact (abort rule never
triggered), 1920ms gate IMPROVING past the 960 one (the block mix works),
and position invariance largely learned (0.88 -> 0.37 at offset 4000) with
~39k steps of budget still to spend.

To resume (instance data intact on the paused machine):

```bash
jl resume <wlk-gpu-check-in2 id>   # new machine id after resume
# part-2 checkpoints: runs/jl_ws2_mix_pos_p2/tower_last.pt (if any gate point
# was reached in part 2) else part-1 runs/jl_ws2_mix_pos/tower_last.pt (step 20k)
# relaunch scripts/train_tower_distill.py with --resume-tower <that checkpoint>,
# --steps <remaining>, lr continuing the cosine (~7.5e-6 from 20k).
```

## 2026-06-11 - Session B: WS2 Final, KV-Cache Parity, Chain-Drift Fix, Long-Form Causal Works

Instance 424892 -> 424993 (paused after). Runs `r_e1a643fd`/`r_fa4f0e3f`
(WS2 parts), `r_719b4463` (evals), `r_92cafe07` (reset evals).

### WS2 complete (60k cumulative steps in 3 parts: HF-CDN crash at 20.5k,
### balance-pause at ~21k, clean 40k finish)

Mixed blocks 96/192 (50/50), position offsets log-uniform [1,6000] @ p=0.5,
triple gate, LS-960 reshuffled, lr 1e-5 -> 1e-6 (cosine across parts):

| cumulative step | gate@960 | gate@1920 | gate@960@off4000 |
| ---: | ---: | ---: | ---: |
| 0 (D1) | 0.2492 | 0.2651 | 0.8828 |
| 20000 | 0.2514 | 0.2408 | 0.3681 |
| 40000 | 0.2201 | 0.2194 | 0.2977 |
| 60000 (final = best) | **0.2117** | **0.2031** | 0.2742 |

20-chunk confirmation: @960 0.2325, @1920 0.2113. gate@1920 reaches parity
with the untrained windowed-bidirectional right=0 reference (0.2017).
Checkpoint `runs/jl_ws2_mix_pos_p2b/tower_last.pt`, backed up locally.

### Decoder KV-cache V1: GPU parity exact, RTF gain real

5 chunks, cache off vs on: WER identical (0.1807), **5/5 final texts
byte-identical**, RTF 0.658 -> 0.516 (-22 percent on short chunks; the gain
grows with prefix size). Default ON everywhere now.

### Long-form chain-drift: diagnosed and countered

With position invariance learned (off4000 gate 0.27), the 21-file eval STILL
collapsed (~0.95) — transcripts start clean and degrade mid-file
("...My name is Jingwei from the University of Science and Technology of
China..." -> "Do do do do. So, so, so. Oh, oh, oh!"). Root cause: training
sequences chain at most 2-3 blocks (16s); serving chains 300+ blocks through
the per-layer KV, and representation drift compounds. Positions were only the
first of two long-form failure modes.

Immediate fix shipped: `reset_encoder_on_rollover` — encoder positions + KV
restart at each ~15s segment roll (decoder text flow untouched). Every block
stays within the trained regime; cross-segment acoustic context is the cost.

| 21 full MCIF files | WER (teacher, legacy) | WER (human, whisper) | RTF |
| --- | ---: | ---: | ---: |
| causal, no reset, 960 | 0.9492 | — | 0.176 |
| causal + reset, 960 | 0.2619 | 0.1957 | 0.531 |
| causal + reset, 1920 | **0.2485** | **0.1870** | 0.289 |
| surgery windowed (prod, chunk 2s) | 0.1367 | 0.0843 | 0.293 |
| offline one-pass 0.6B | — | 0.2075 | — |
| offline one-pass 1.7B | — | 0.1202 | — |

**The append-only causal model now works on long-form audio** — 0.1870 vs
human references, BETTER than offline one-pass 0.6B (0.2075) and ~2.2x from
the windowed production point (0.0843), at equal RTF (0.29).

### Next

1. WS2c: long-sequence training (concatenate streamed utterances to 16-96s,
   12-19 chained blocks) — the principled fix for chain drift, to retire the
   reset and recover cross-segment context.
2. WS4: causal mode in the qwen3-streaming backend
   (--qwen3-streaming-audio-backend causal + tower checkpoint).
3. French (MLS) on top of the same recipe.

Artifacts: `runs/jl_sessionB/`, `runs/jl_ws2_mix_pos_p2b/{history,final_metrics}.json`,
rescore in `runs/rescore_mcif_v0/sessionB_reset.json`. Checkpoints on the
paused machine + `~/Downloads/qwen3_checkpoints/`.

## 2026-06-11 - WS2c: Long-Sequence Distillation — Negative Result

Run `r_c9bf18da` on instance 425323 (paused after). Attempt to retire the
rollover reset by training the chain directly: streamed utterances
concatenated to 16-96s samples (12-19 chained blocks, log-uniform), frame-
budget batching, mixed blocks 96/192, offsets, 30k steps lr 5e-6 from the
WS2 tower. New long-form gate: 3 full WLK files, segmented, NO reset.

| step | gate@960 | gate@1920 | gate@long (no reset) |
| ---: | ---: | ---: | ---: |
| 0 (WS2 tower) | 0.2117 | 0.2031 | ~0.8710 |
| 10000 | — | — | 0.8710 |
| 20000 | 0.3148 | 0.2978 | 0.8455 |
| 30000 | 0.3063 | 0.3082 | 0.8338 |

Findings:

- The long gate barely moves (0.871 -> 0.834, -4 percent relative in 30k
  steps) while short-chunk quality DEGRADES (0.21 -> 0.31) and never
  recovers. The distillation loss on long sequences stays 2-3x above the
  short-sequence level throughout: matching a fully-bidirectional teacher
  through a KV chain over 96s appears structurally hard, not
  budget-limited.
- No checkpoint improves on the WS2 tower; nothing promoted.

Decision: the **rollover-reset design stands as the production answer** for
long-form causal serving (0.2485 teacher-legacy / 0.1870 human-whisper on
the 21 files, RTF 0.29 — better than offline one-pass 0.6B). Voxtral-style
windowed-session behavior, achieved. Retiring the reset would need a
different attack (length curriculum at higher LR accepting a quality dip,
or sequence-level/text-level objectives) with uncertain payoff — parked.

Next: WS4 (causal mode in the qwen3-streaming backend with
reset-on-rollover default), then French (MLS) on the same recipe.

## 2026-06-12 - Min-Compute-Per-Chunk: Rolling Decoder KV + Speculative Draft + Punctuation Rollover

Objective pivot: WER 0.18-0.20 long-form is accepted; the target is now the
least computation per new audio chunk. Per-chunk audit (5-agent workflow,
file:line): the causal encoder is already at its floor (~12-15 GFLOPs,
append-only, 1-3 ms); the decoder is 85-90 percent of wall-clock through
three redundancies: (1) the [prompt + audio] prefix re-prefilled from
scratch every chunk (44 -> 212 positions across a segment, 4.8x redundant,
zero KV reuse across chunks); (2) the full hypothesis re-decoded from token
0 every chunk (7 -> 49 sequential steps, 4.6x); (3) per-step overhead in
`_control_logits_and_pick` — O(T) `generated.cpu().tolist()` per step, a
Python repetition-penalty loop doing per-token GPU scalar indexing, 3-5
device syncs (3-10 ms/step instead of the ~1.5-2 ms bandwidth floor).
Bonus finds: `tokenizer.encode` of the prompt template ran EVERY chunk when
`segment_prompt_context_words > 0`; rollover was checked AFTER generation
(the segment's largest generation ran on a cache dropped right after); the
eval CLI's post-construction left-context override silently failed to update
`left_context_steps` on the causal encoder (config path was live; override
now re-derives the step window so it can never be a silent no-op).

Implementation (all training-free, flags default OFF):

- `_GreedyControlSession` (native_realtime_model.py): incremental host-side
  token history (no per-step O(T) transfer), vectorized repetition penalty
  over a cached unique-token GPU index, ngram ban from the host list, one
  device sync per step. Kept `_apply_repetition_controls_to_logits` as the
  reference spec; bit-equivalence tests (fp32 + bf16) over 8 ordering/dtype
  risk points. Both legacy decode loops now run through the session.
- Rolling decoder KV (`generate_full_hypothesis_rolling` + new
  `DecoderRollingState` on `CachedAudioDecodeState`): the decoder KV over
  [prompt head + cached audio] persists across chunks; per chunk ONE
  parallel forward of [new audio delta + template tail + previous
  hypothesis as a speculative draft], greedy verification position-by-
  position with exact control replay (same `controlled_logits` code as the
  sequential stepper), `DynamicCache.crop` drops rejected-draft KV and
  restores the [head + audio] prefix after every generation. Capability
  probe with one-shot `disabled` fallback to the legacy path (non-croppable
  cache / batch != 1 / prompt_token_ids).
- Streamer: `decoder_rolling_kv` / `speculative_draft` config flags, segment
  prompt template tokenized once per segment, rolling state invalidated at
  rollover, per-event stats (decoder_path, draft_tokens/accepted,
  decode_steps, prefill_positions, generate_ms).
- Punctuation rollover (IWSLT agent_simulstream-style, audio cut at chunk
  boundary instead of forced-aligner timestamps): roll when the segment
  hypothesis ends with sentence-final punctuation and the segment has
  >= `--segment-punct-min-steps` (default 100) cached steps; the step cap
  stays the no-punctuation fallback. `--segment-roll-before-generate`
  rolls before decoding when the incoming chunk would exceed the cap.
- Eval CLI: `--decoder-rolling-kv`, `--speculative-draft`,
  `--decoder-parity-check` (dual-streamer harness), punctuation flags;
  summary gains generate_ms_mean, draft_acceptance_rate, decode/prefill
  totals, parity fields. gate.py passes the new flags through (defaults
  unchanged for trainers).

Tests: 124 local / 52 on the H100 (new tests/test_decoder_rolling_kv.py:
control bit-equivalence vs legacy spec, rolling-vs-full parity across
chunks on croppable cumulative-mean fakes, 10 draft cases incl.
corrected-pick-is-eos and budget edges, fallback/disabled paths, end-to-end
streamer parity across rollovers, template encoded once per segment,
punctuation/pre-roll triggers).

Parity semantics, measured (5 MCIF files, 1920 ms, seg200 + reset): the
speculative draft is exactly faithful to the rolling path (spec == roll on
every chunk, including divergent ones). Rolling vs monolithic re-prefill is
NOT byte-stable in bf16 — K/V computed in different matmul shapes flip a
near-tie argmax on rare chunks ("high-sounding" vs "high-speed"),
self-correcting next chunk since text is regenerated. Quality impact: zero
— per-file WER delta vs the session-B baseline run: +0.0032/-0.0082/
+0.0019/+0.0033/-0.0013, mean -0.0002. Gate redefined accordingly (flip
rate + WER delta, not byte equality) and PASSED.

Speed (same 5 files): generate_ms mean 161-234 vs 431-530 baseline —
**2.1-2.6x on the dominant term**, draft acceptance 0.67-0.88 (mean 0.76),
sequential decode steps now ~constant in segment age.

### 21-file results (chunk 1920 ms, seg200 + reset, runs/jl_mincompute/)

| run | WER teacher-legacy | WER human-whisper | RTF | segments/file | rollovers punct/cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline (session B, full re-prefill) | 0.2485 | 0.1870 | 0.289 | ~19 | 0/all |
| A: rolling KV + speculative draft | 0.2488 | 0.1873 | **0.115** | 19.1 | 0/402 |
| B: A + punct rollover (min 100 steps ~8s) | 0.2696 | 0.1902 | 0.0997 | 31.8 | 655/13 |
| C: B + roll-before-generate | 0.2672 | 0.1885 | **0.0950** | 31.9 | 657/0 |
| D: A + punct rollover (min 150 steps ~12s) | **0.2468** | **0.1807** | 0.1065 | 23.4 | 436/55 |
| E: D + roll-before-generate | 0.2465 | 0.1810 | 0.1094 | 23.7 | 434/0 |

Readings:

- A is the drop-in production point: WER identical to the baseline in both
  metrics (+0.0002 / +0.0003), **RTF 0.289 -> 0.115 (2.5x)**, draft
  acceptance 0.74, generate_ms_mean 205 (chunk-level speedup grows with
  segment depth: 2.1-2.6x measured). Defaults flipped ON
  (CachedFullHypothesisConfig, gate.py, eval CLI auto-resolves on unless
  --decoder-cache off).
- The teacher-legacy "degradation" of B/C (+0.02) is mostly a normalization
  artifact: under human refs + Whisper normalizer C is +0.0012 vs A —
  within noise — while cutting RTF another 17 percent and rolling on
  sentence boundaries exclusively (657 punct / 0 cap). Punct flags stay
  opt-in because they change output text (policy choice, not a pure
  optimization).
- Punct min-steps matters: at min 100 (~8s segments) WER degrades (+0.003
  human-whisper, +0.02 teacher-legacy); at min 150 (~12-16s segments,
  run D) sentence-boundary cuts BEAT the fixed cap on BOTH axes —
  **0.1807 human-whisper (best long-form causal number to date, vs 0.1870
  baseline) at RTF 0.1065 (2.7x)**. Cutting at linguistic boundaries
  avoids the mid-sentence word losses of the cap roll, as long as segments
  stay long enough to keep encoder context.
- Total speedup baseline -> C: **3.0x end-to-end** (RTF 0.289 -> 0.095) at
  WER 0.1885 human-whisper; recommended quality point is D/E (punct min
  150, preroll optional): **0.1807-0.1810 human-whisper at RTF ~0.11
  (2.7x)** — better WER than the baseline AND 2.7x its speed. E adds the
  clean invariant that no generation ever runs on an over-cap cache
  (434 punct / 0 cap rollovers); D vs E WER/RTF deltas are run noise.
  Per 33-min talk: ~3.5 min of H100 compute.

Shipping config (WS4 defaults): rolling KV + speculative draft ON
(already default), `--segment-punct-rollover --segment-punct-min-steps 150
--segment-roll-before-generate` recommended for long-form serving.
Artifacts: runs/jl_mincompute/ (+ ~/jl_mincompute_artifacts.tgz on the
machine). Machine paused after this session.

## 2026-06-12 - WS4: Causal Backend Promoted to whisperlivekit (Production)

The causal stack is now a first-class mode of the qwen3-streaming backend
(commit a495c2e): `--qwen3-streaming-audio-backend causal
--qwen3-streaming-tower-checkpoint <path-or-hf-repo>`. New
`whisperlivekit/qwen3_streaming/causal.py` (encoder + causal model +
checkpoint loaders), rolling decoder KV + speculative draft ported into
prod model.py (defaults OFF in windowed mode, ON in causal), punct
rollover + roll-before-generate + reset-on-rollover in the prod streamer,
causal defaults derived automatically in asr.py (run-D operating point).

Key production-only design: **fixed attention blocks inside the encoder**
(block_frames=192). Production pacing delivers variable-size mel chunks;
the encoder buffers and consumes exact multiples of the trained block size
(pacing-invariance pinned by unit tests), and `flush_pending` encodes the
partial tail block once at end of utterance (mandatory: no right-context
zeros in causal mode).

GPU parity gate (3 MCIF files, H100 bf16, machine 425907): PASSED —
prod fed 192-frame chunks == experiments harness (gate 1, near-tie bf16
flips only), and variable chunking does not degrade quality vs exact
blocks (gate 2, one-sided; on one file paced was 0.026 BETTER — punct
boundaries shift with decode points, transcripts differ in form at equal
quality; this is expected and documented in the test).

169 WLK tests + 124 experiments tests pass. Benchmarks (21 MCIF via prod
stack + LibriSpeech test-clean/other) follow below.

### WS4 benchmarks (production stack, H100 bf16, runs/ws4_bench/, machine 425907)

| eval | metric | value |
| --- | --- | ---: |
| 21 MCIF long-form (prod causal, 1920ms) | WER teacher-legacy / human-whisper | 0.2465 / **0.1810** |
| 21 MCIF long-form (prod causal) | RTF | 0.160 |
| LibriSpeech test-clean (2620 utt, streaming) | corpus WER whisper-norm | **0.0364** |
| LibriSpeech test-other (2939 utt, streaming) | corpus WER whisper-norm | **0.0716** |

The production backend reproduces the experiments run-E numbers exactly
(0.2465/0.1810 — same checkpoint, same blocks, ported code). LS numbers are
short-form streaming through the full prod stack (per-utterance streamer,
1.92 s blocks, punct rollover mostly idle on short clips). Context: Voxtral
Mini Realtime 3B @480ms publishes 2.1/5.5; for a 0.6B append-only streaming
model, 3.6/7.2 is the headline short-form result. Offline LS numbers for the
base model are published by Qwen (not re-run here). Artifacts:
runs/ws4_bench/ + ~/Downloads/qwen3_checkpoints/ws4_bench_artifacts.tgz
(local). Total phase-2 GPU spend ~ $10.

### FLOPs correction: measured (FlopCounterMode) vs the analytic audit

The 2026-06-12 per-chunk ledger's GFLOPs were analytic estimates; now
measured with torch.utils.flop_counter on the real modules (real tower
weights, steady-state caches) — scripts/measure_flops.py:

| quantity | audit (analytic) | measured |
| --- | ---: | ---: |
| causal encoder, 1.92s block @steady | 12-15 | **17.1** |
| windowed encoder, 2.0s update @steady | ~37 | **114.5** |
| decoder prefill 128/212 pos | ~154/254 | **112.7/186.7** |
| causal rolling pass (~62 pos) | ~46-96 | **54.6** |
| 1 sequential step @past250 | ~1.2 | **0.88** |
| windowed total per audio-sec | ~110 avg | **125.9 avg / 172.2 peak** |
| causal total per audio-sec | ~50 | **41.5 (constant)** |

Main audit error: the windowed encoder was 3x underestimated — the audit
ignored the conv stem (3x conv2d at 480 channels over the full 1264-frame
window; the conv alone is ~2/3 of the window cost) and assumed the audio
tower at d=1024 when it is d=896 (the text decoder is 1024). The headline
ratio survives: causal is 3.0x cheaper on average, 4.1x at segment end,
and constant in stream age. HF card updated with the measured numbers.

### Two corrections to earlier notes

1. MCIF/WLK file durations: the 21 talks are 277-426 s (mean 5.6 min, 2.0 h
   total), NOT ~30 min as claimed in some later notes. Consequently the
   per-talk compute at RTF 0.107-0.16 is ~36-54 s of H100 (the "3.5 min per
   33-min talk" correction of 2026-06-12 was itself wrong; the original
   "~32-35 s per talk" seconds were right, the premise about duration was
   not).
2. HF card: the offline 0.6B row (20.8) is now explicitly labeled as the
   NAIVE one-pass usage (single transcribe() call on a 5-7 min file:
   long-generation drift/skips, deletion-dominated; 1.7B scores 12.0 under
   the same protocol). Properly segmented offline decoding is the strong
   baseline, and the windowed backend (8.4) is that strategy made streaming.
   This was already the RUNS.md conclusion ("segmentation is not a streaming
   compromise, it is the better long-form decoding strategy"); the card
   table now says it instead of implying offline < streaming.
