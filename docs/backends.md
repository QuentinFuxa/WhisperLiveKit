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
