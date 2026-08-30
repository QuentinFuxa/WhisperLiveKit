# M5 streaming backend screening — 7 September 2026

This screening compares the optional adapters in [#425](https://github.com/QuentinFuxa/WhisperLiveKit/pull/425)
and [#426](https://github.com/QuentinFuxa/WhisperLiveKit/pull/426) with Whisper small
MLX/LocalAgreement and the existing Qwen3 vLLM Metal backend. It uses the full WLK
audio pipeline, including voice activity detection, pauses and EOF handling.
The published historical figures are a separate experiment and remain unchanged.

## Corpus and procedure

The [fixed FLEURS manifest](../../corpora/fleurs-90.json) contains 30 test clips per
language: English, French and Mandarin Chinese. Its SHA-256 is
`95586eb74f53e665f35983f53da31ea510867b54c7fe5f481466e2d6af2619e5`.
The dataset revision, sentence/recording IDs, references, extraction boundaries
and original/PCM16 audio hashes are in that manifest. Audio is cached locally;
it is not redistributed in this directory.

Each backend/language combination runs in a fresh process. One full clip warms
the engine, then all 30 clips run three times at real-time feed speed. The warmup
record includes model loading and is excluded from quality and steady-state
latency summaries. Downloads happen before timing. Model files are hashed after
the run. Startup is not a cold filesystem-cache measurement.

Reports use schema 3.1. Each audio packet is delivered at an absolute deadline
corresponding to its last sample; EOF follows the final write without another
sleep. Earlier schema-3.0 trials accumulated pacing drift and waited after the
last packet, artificially reducing measured EOF delay. Those trials were stopped
and are excluded from this comparison.

Each backend also receives one continuous ten-minute Chinese stream, built from
the same manifest with recorded component boundaries. It runs in its own process,
without another ten-minute warmup. Its truncated last component has no exact
reference, so this run is used for resources and stream boundaries, without an
aggregate WER/CER score.

GPU jobs run serially: Whisper, Nemotron, native Qwen, then Qwen Metal; within each
backend: English, French, Chinese, continuous Chinese. The order is fixed, not
randomized. Three passes reveal repeat variation but are not 90 independent
utterances or a confidence interval for unseen speech. Ordinary desktop activity
is not controlled. Only this M5 host is measured; no NVIDIA result is implied.

## Model and runtime configuration

All backends use their default decoder settings except the explicit model paths
and Whisper's LocalAgreement policy. Per-clip effective options and runtime
versions are retained in the source reports.

| Backend | Model repository | Model revision | Path option |
|---|---|---|---|
| Whisper small MLX | `mlx-community/whisper-small-mlx` | `45f3915923c7a79a5a5b5a7d909d39aeb0e5630e` | `model_dir` |
| Qwen native MLX | `Qwen/Qwen3-ASR-0.6B` | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` | `mlx_qwen3_asr_model` |
| Qwen Metal | `Qwen/Qwen3-ASR-0.6B` | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` | `model_dir` |
| Nemotron native MLX | `mlx-community/nemotron-3.5-asr-streaming-0.6b` | `e550040c0478027ed679b2b6b0d055502c103663` | `nemotron_mlx_asr_model` |

Nemotron requires the converted MLX checkpoint. The original proposed NVIDIA
checkpoint is a different format and failed at startup. The adapter now uses
mlx-audio 0.5.1's mel/encoder caches and a small stateful RNNT loop. Direct text
parity with upstream decoding passed on one English, French and Chinese clip,
including ragged input chunks and EOF; that is an adapter check, not a quality
ranking.

Repeated sessions exposed an MLX stream-ownership error in the initial adapter:
a mutex serialized decoding but let it change threads. Model loading and all
Nemotron decoding now run on one dedicated thread. A real shared-model check
failed before that change and passes after it: two concurrent English/Chinese
sessions retain exactly the same text as sequential sessions. Native Qwen also
passes that check with its existing shared lock. These checks validate isolation,
not multi-user throughput; their script and results are retained here.

Native Qwen uses mlx-qwen3-asr 0.3.5, accuracy mode and its incremental finalizer.
The upstream finalizer resolves a default tokenizer when it receives a model
object. This run uses that same default model/tokenizer revision; these results
do not validate arbitrary custom checkpoints or tokenizer revisions.

Metal uses stable vllm-metal 0.2.0 and the host's separately installed vLLM
0.21.0+cpu, with missing audio/benchmark dependencies supplied by an isolated overlay.
Its Transformers version is 5.8.1; Nemotron uses 5.15.0 and native Qwen does not
require Transformers. These are adapter/runtime comparisons, not a controlled
comparison of inference libraries with identical dependencies. The existing user
environment is unchanged. Package versions in the reports describe what ran;
[additional audio/runtime package versions](environment-packages.json) are retained
alongside them.

The existing Metal backend builds an automatic-language prompt, even when the
session configuration contains `lan`. Native Qwen and Nemotron receive the
specified language. This difference must accompany any comparison. The Metal
model is pinned through `model_dir`; `vllm_model` is not its model-path option.

## Reading the measurements

Quality pools edit counts over reference words (English/French) or characters
(Chinese), without averaging clip percentages. Normalization applies NFC,
lowercase and punctuation handling, then removes whitespace for Chinese CER.
It does not equate spoken and written number forms, such as “twenty-nine” and
“29”, or simplify traditional Chinese. These differences can affect a one-point
quality threshold. Original references and hypotheses remain available to inspect.

First visible text includes provisional text; first committed text is separate.
EOF finalization starts after the last audio feed returns. Source-end lag also
includes slow feeding/backpressure and must be read alongside EOF time. ASR RTF
counts inference calls, including dispatch/lock waits as well as model work;
it is not GPU kernel time or end-to-end latency. Per-pass and pooled p95 use
linear interpolation over clip measurements.

RSS is sampled process memory at 50 ms intervals. MLX reports allocator peak
since each sample's reset and active allocations at completion. The two memory
quantities are separate and must not be added on unified memory. Warmup/startup
memory is retained separately from warmed-pass peaks. The harness and pipeline
are part of the measured process. Line timestamp flags check ordering and
nonnegative durations; they are not an assessment of word-alignment accuracy.

Failed/skipped clips remain in the JSON, with any partial hypothesis and error.
They cannot contribute a successful quality score or establish a numerical pass.
The final-output and explicit ASR error fixes from #440 are present in all runs;
earlier smoke/baseline files are not included as selection evidence.

## Selection rule and reproduction

Compare each candidate with each existing backend on the same audio identities,
hashes, references, repeats and normalization. The numerical gate requires at
least 20% lower EOF p95 or measured memory in all three passes, using the same
metric throughout, and no more than one absolute point worse WER/CER in any
pass. A failed, incomplete or ambiguous comparison does not justify merging.
The continuous stream and hypotheses still require review; a numerical pass
alone does not establish the absence of streaming defects.

To reproduce a run, prepare the pinned corpus as described in
[the benchmark guide](../../README.md), install the selected backend extra in its
own environment, and put its pinned local snapshot path in `config.json` using
the option above. Whisper also needs `"backend_policy": "localagreement"`.

```bash
wlk bench --backend BACKEND --model MODEL --languages en \
  --manifest benchmarks/corpora/fleurs-90.json --config config.json \
  --speed 1 --warmup --repeats 3 --json report.json
wlk bench --backend BACKEND --model MODEL --languages zh \
  --manifest benchmarks/corpora/fleurs-90.json --config config.json \
  --speed 1 --continuous --json continuous.json
```

Repeat the first command for French and Chinese in fresh processes. Backend
names are `mlx-whisper`, `mlx-qwen3-asr`, `qwen3-vllm-metal` and
`nemotron-mlx-asr`; model labels are `small` for Whisper and `0.6B` otherwise.
Candidate source revisions must be checked out explicitly; the adapter PRs are
not assumed to be on main. The reports record the actual source commits.

`analyze.py` reads schema-3.1 reports (plain JSON or losslessly compressed `.json.gz`),
recomputes per-pass and pooled summaries, verifies paired identities, and leaves
missing runs pending. It does not run inference or decide to merge a PR.
