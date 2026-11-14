# ASR backends

Choose a backend for the hardware and languages you need. Installation extras are declared in [pyproject.toml](../pyproject.toml).

## Voxtral Backend

WhisperLiveKit supports [Voxtral Mini](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602),
a 4B-parameter streaming speech model from Mistral AI. Its model card lists 13 supported languages.

```bash
# Apple Silicon (native MLX, recommended)
pip install "whisperlivekit[voxtral-mlx]"
wlk --backend voxtral-mlx

# Linux/GPU (HuggingFace transformers)
pip install "whisperlivekit[voxtral-hf]"
wlk --backend voxtral
```

Voxtral uses its own streaming policy and does not use LocalAgreement or SimulStreaming.
See [the benchmark notes](../benchmarks/README.md) for evaluation guidance.

## FunASR / SenseVoiceSmall

Install the optional backend and run
[SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) through
WLK's existing LocalAgreement and VAC/VAD pipeline:

```bash
pip install "whisperlivekit[funasr]"
wlk --backend funasr --language auto
```

Use a verified local model snapshot without executing remote model code:

```bash
wlk --backend funasr --model_dir /path/to/SenseVoiceSmall --language yue
```

The initial integration supports SenseVoiceSmall transcription in Mandarin
(`zh`), Cantonese (`yue`), English (`en`), Japanese (`ja`), Korean (`ko`), and
automatic detection. FunASR uses LocalAgreement only; selecting it with the
default policy switches that policy automatically. It does not support
`--direct-english-translation`, and WLK remains responsible for voice activity
control rather than enabling FunASR's internal VAD. This compatibility contract
does not cover arbitrary FunASR models. SenseVoiceSmall is distributed under
its [model license](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE).

## Qwen3-ASR streaming (HF Transformers)

`qwen3-streaming` runs Qwen3-ASR through plain HF Transformers with a
bounded-recompute audio cache: the pretrained audio tower only re-encodes a
local window (default 12 s) per update, cached audio embeddings are
append-only, and text is committed with a stable-prefix rule. Works on CUDA,
Apple Silicon (MPS) and CPU, no vLLM required.

```bash
pip install "whisperlivekit[qwen3-streaming]"
wlk --backend qwen3-streaming --language en
```

Notes:
- An explicit `--language` is required (automatic detection switches language
  mid-stream on accented audio).
- Word timestamps are interpolated estimates. `--backend qwen3-vllm`
  uses a ForcedAligner instead; validate timing on your audio when word boundaries matter.
- Decode pacing self-adjusts to the hardware; on GPUs slower than real time
  the update cadence grows instead of lagging. Plan one realtime session per
  GPU.
- Defaults encode the validated operating point (12 s left context, ~15 s
  segments); see `--help` for the `--qwen3-streaming-*` knobs.

**Causal mode (minimum compute per chunk).** The windowed default re-encodes
up to 12 s of audio on every update. The causal mode runs an append-only
causal-KV encoder instead: each ~2 s audio block is encoded exactly once,
memory is bounded (15 s window + sentence-boundary segment resets), and
per-chunk compute is constant in stream length:

```bash
wlk --backend qwen3-streaming --language en \
    --qwen3-streaming-audio-backend causal \
    --qwen3-streaming-tower-checkpoint qfuxa/qwen3-asr-0.6b-streaming
```

The fine-tuned tower ([qfuxa/qwen3-asr-0.6b-streaming](https://huggingface.co/qfuxa/qwen3-asr-0.6b-streaming))
downloads automatically. The Qwen runtime, tests, experiments, benchmarks and
figures now live in [Qwen3-ASR-causal](https://github.com/QuentinFuxa/Qwen3-ASR-causal),
which is consumed here through `third_party/qwen3-asr-causal`. WhisperLiveKit
keeps only the backend wiring and CLI flags.

The causal tower is English-only. Evaluate its quality and compute cost against
the windowed mode on your target audio. Detailed experiments are maintained in
the Qwen3-ASR-causal repository.

**Experimental vLLM Metal causal mode.** On Apple Silicon, `qwen3-vllm-metal`
can also load the same causal tower and keep a rolling MLX decoder KV over the
`[prompt + audio]` prefix. New audio blocks are encoded once, and the decoder
replays only the new audio embeddings, prompt tail, and previous-hypothesis
draft:

```bash
wlk --backend qwen3-vllm-metal --language en \
    --qwen3-vllm-metal-audio-backend causal \
    --qwen3-vllm-metal-tower-checkpoint qfuxa/qwen3-asr-0.6b-streaming
```

This path has passed local short-form smoke tests and is useful for comparing
the Metal decoder path against the HF/MPS causal backend. It is still
experimental; validate it on your target workload before deployment.

**Experimental vLLM CUDA causal mode.** On NVIDIA GPUs, `qwen3-vllm` can load
the same causal tower, use vLLM for the text decoder, and keep vLLM
ForcedAligner timestamps:

```bash
WLK_QWEN3_VLLM_LIVE_MULTIPROCESSING=1 \
wlk --backend qwen3-vllm --language en \
    --qwen3-vllm-audio-backend causal \
    --qwen3-vllm-causal-decoder-backend vllm-live \
    --qwen3-vllm-tower-checkpoint qfuxa/qwen3-asr-0.6b-streaming
```

The `vllm-text` backend is the conservative fallback when you do not want the
live request-local append path; it still uses vLLM prefix caching but starts one
text-decoder request per chunk. The `append-kv` and `rolling` names remain as
compatibility aliases for the HF decoder path. Treat the causal decoder routes as experimental; the standard route remains available
for comparison on the same audio.

## Canary Backend

WhisperLiveKit supports [NVIDIA Canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2)
via [NeMo](https://github.com/NVIDIA/NeMo), a 1B-parameter model covering 25 European
languages with native word-level timestamps. Automatic language detection uses NeMo's
AmberNet language-ID model when `--language auto` is set; the detected language is locked
in once enough audio has accumulated. Canary streams through the LocalAgreement policy.

```bash
pip install "whisperlivekit[canary]"
wlk --backend canary --language auto
```

Notes:
- Runs on CUDA and CPU. A GPU is strongly recommended: CPU works (see
  [`scripts/smoke_canary.py`](../scripts/smoke_canary.py)) but is slow for a 1B-parameter model. On Apple
  Silicon, NeMo's current restore path uses CPU; this backend does not enable
  MPS device placement. NeMo is a heavy dependency (torch, Lightning, and
  friends).
- Supports WhisperLiveKit's Python range, 3.11 through 3.13. The `canary` extra
  intentionally selects `nemo-toolkit[asr]>=3.0,<4`, the NeMo line validated
  with Canary timestamps and Python 3.13.
- Explicit `--language <code>` skips language detection entirely. Tune detection
  with `--canary-lid-min-sec` (minimum audio before detecting) and
  `--canary-lid-min-conf` (confidence threshold to lock in the detected language).
- Canary-1b-v2 is distributed under CC-BY-4.0. The separate
  [AmberNet language-ID model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/langid_ambernet)
  is downloaded from NVIDIA NGC; review its model-card terms for your deployment.


## Nemotron through MLX

```bash
pip install 'whisperlivekit[nemotron-mlx-asr]'
wlk --backend nemotron-mlx-asr --language fr
```

This adapter requires `mlx-audio>=0.5.1,<0.6` and an MLX conversion such as
[`mlx-community/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b).
The NVIDIA repository's Transformers checkpoint has a different configuration
and weight layout; passing that repository directly to the MLX loader fails.

WLK feeds mlx-audio's incremental mel frontend and cached Conformer encoder.
A small RNNT loop retains decoder state between pushes. Pauses, speaker changes
and EOF flush the final mel/encoder frames; there is no separate VAD or full
utterance re-decode. Sessions have separate caches and language prompts. Model
loading and decoding stay on one dedicated thread because cached MLX streams
cannot migrate between caller threads. Session terminology context is
unsupported and rejected explicitly.

`--nemotron-mlx-asr-att-context 56 6` is the default left/right encoder context.
Emitted token timestamps follow encoder frames and are clipped to received
audio bounds; they are not forced word alignments.

This is a low-latency option, not a general replacement for the other backends.
On the fixed [benchmark corpus](../benchmarks/README.md)
([review](../benchmarks/evaluations/m5-20261107/REVIEW.md)) pooled end-of-file
p95 is 0.12 s English, 0.11 s French and 0.05 s Chinese, 71% to 87% below
Whisper small and 74% to 94% below Qwen Metal. A ten-minute Chinese stream
completes with valid timestamps and a 0.03 s EOF delay and does not enter a
repetition loop, though its output carries one artifact: twenty Thai `า`
characters inside a sentence at character 1543.

Recognition quality is strongest in French, where 14.97% WER beats both
baselines (Whisper small 17.66%, Qwen Metal 24.68%). Chinese CER (22.62%) beats
Whisper small (25.77%) but trails Qwen Metal (11.85%). English is the weak pass:
19.27% WER beats Qwen Metal (21.36%) but is 7.6 points behind Whisper small
(11.66%), so prefer Whisper where English accuracy matters more than
end-of-utterance latency.

## Environment constraints

For uv source installs, initialize `third_party/qwen3-asr-causal` with
`git submodule update --init --recursive`. Published pip extras resolve the
Qwen package as a dependency instead of using that checkout.

Several extras require incompatible Torch, Transformers, or vLLM versions.
The full conflict list is in `[tool.uv].conflicts` in
[pyproject.toml](../pyproject.toml); do not install all extras into one environment.
Diart supports Python 3.11–3.12; use Sortformer on Python 3.13.
The vLLM Metal path additionally needs the vLLM Metal runtime; the source checkout
pins a Python 3.12 macOS ARM64 wheel in `[tool.uv.sources]`.

## Qwen3-ASR through native MLX (candidate)

The `mlx-qwen3-asr` extra pins the upstream incremental stream
(`mlx-qwen3-asr>=0.4.4,<0.5`):

```bash
pip install 'whisperlivekit[mlx-qwen3-asr]'
wlk --backend mlx-qwen3-asr --language fr
```

The adapter commits the model's stable prefix and drains `finish_streaming()`
on pauses, speaker changes and EOF. In 0.4.x that call re-decodes the bounded
window together with the pending tail and replaces the window text with the
result, so a segment's text is produced once: the adapter commits only the part
clients do not already hold. WLK owns voice activity detection. Model loading
and warmup run once at engine startup; sessions have separate decoder state and
share a serialized model. Session language (including `auto`) and terminology
context are passed to the native stream, which uses the session language for
every window decode.

`stable_text` stays empty for the first two chunks of a window, so a short
utterance would otherwise commit nothing before its pause. While the library
reports no stable prefix, the adapter commits the units two consecutive window
decodes agree on (LocalAgreement-1) and holds the last five units back; CJK
text is compared per character so the prefix can advance inside a Chinese
sentence. A reported prefix that is not a prefix of the current text, or that
rewrites text already committed, is repaired into an append of what the model
now reports (upstream issue #31) instead of aborting the session. A pause whose
leftover audio cannot be decoded — the log-mel frontend needs more than 200
samples — ends the segment without a final decode instead of raising.

`--mlx-qwen3-asr-finalization-mode accuracy|latency` is accepted for API
compatibility with earlier versions; 0.4.x decodes the pending tail in both
modes, so the flag no longer selects between two finalize paths. The chunk
default is two seconds.

Timestamps describe decode chunk bounds, not aligned words.

### Reproducing the fixed-corpus measurement

The comparison in the native-MLX review used the fixed manifest: 30 English, 30
French and 30 Mandarin clips at a pinned FLEURS revision, three paced passes per
backend. Metric definitions and the manifest's limits are in
[benchmarks/README.md](../benchmarks/README.md).

```bash
pip install 'whisperlivekit[test,mlx-qwen3-asr,mlx-whisper,qwen3-vllm-metal]'
python scripts/prepare_fleurs_benchmark.py

wlk bench --backend mlx-qwen3-asr --model Qwen/Qwen3-ASR-0.6B \
  --languages en,fr,zh --manifest benchmarks/corpora/fleurs-90.json \
  --warmup --repeats 3 --speed 1 --json native-mlx-qwen3.json

wlk bench --backend qwen3-vllm-metal --model Qwen/Qwen3-ASR-0.6B \
  --languages en,fr,zh --manifest benchmarks/corpora/fleurs-90.json \
  --warmup --repeats 3 --speed 1 --json qwen-metal.json

wlk bench --backend mlx-whisper --model small \
  --languages en,fr,zh --manifest benchmarks/corpora/fleurs-90.json \
  --warmup --repeats 3 --speed 1 --json mlx-whisper-small.json
```

`--speed 1` is the paced pass; `--repeats 3` keeps every repetition in the
report. Compare the two Qwen runs and the Whisper baseline on word error rate
for `en`/`fr`, character error rate for `zh`, and end-of-file p95, sample by
sample, after confirming the same 30 recording IDs are present in all three
reports. A pooled p95 alone does not establish the comparison; check each
repetition, and keep model downloads outside the measured runs.

