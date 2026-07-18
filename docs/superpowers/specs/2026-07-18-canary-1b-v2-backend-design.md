# Design: NVIDIA Canary-1b-v2 backend for WhisperLiveKit

**Date:** 2026-07-18
**Status:** Approved (Phase 1)
**Author:** brainstorming session

## Summary

Add support for NVIDIA's `nvidia/canary-1b-v2` speech model as a new ASR backend
in WhisperLiveKit, driven by the existing **LocalAgreement** streaming policy.
Phase 1 delivers a working `canary` backend with multilingual auto language
detection. Phase 2 (a separate, later spec) will add a lower-latency
`canary-streaming` backend built on NeMo's native AlignAtt decoder.

## Background

### The model

Canary-1b-v2 (paper: arXiv 2509.14128; card: huggingface.co/nvidia/canary-1b-v2)
is a **978M-parameter attention encoder-decoder (AED)** model:

- **Encoder:** FastConformer, 32 layers.
- **Decoder:** Transformer, 8 layers, autoregressive.
- **Tokenizer:** unified SentencePiece, 16,384 tokens.
- **Languages:** 25 European languages (Bulgarian, Croatian, Czech, Danish,
  Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian,
  Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Russian, Slovak,
  Slovenian, Spanish, Swedish, Ukrainian).
- **Tasks:** ASR and AST (translation), selected purely via `source_lang` /
  `target_lang` kwargs (same lang = ASR, different = translation).
- **Timestamps:** native word-level and segment-level via
  `model.transcribe(..., timestamps=True)`.
- **Streaming:** the base model is **offline/full-utterance**. `source_lang` is
  **mandatory** — there is no native auto-detect mode.

### NeMo API (verified)

```python
from nemo.collections.asr.models import ASRModel  # resolves to EncDecMultiTaskModel
asr_model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")

# audio may be a list of file paths OR a list of 16kHz mono float32 numpy arrays
output = asr_model.transcribe(
    [audio_np],            # 16 kHz mono float32
    source_lang="en",
    target_lang="en",
    timestamps=True,
)
hyp = output[0]
hyp.text                                  # full text
hyp.timestamp["word"]                     # [{'word': str, 'start': float, 'end': float}, ...]
hyp.timestamp["segment"]                  # [{'segment': str, 'start': float, 'end': float}, ...]
```

Package: `nemo_toolkit[asr]`. Timestamp support needs a recent NeMo
(main / >= 2.5). Runs on CUDA (target here), CPU (slow), and MPS (partial).

### Why LocalAgreement fits

The framework's `OnlineASRProcessor` (`local_agreement/online_asr.py`) already
turns an **offline** `transcribe()` into a streaming experience: it accumulates a
growing audio window, re-transcribes it each tick, and commits words via the
`HypothesisBuffer` LocalAgreement policy (confirm a word when consecutive
inferences agree). This is exactly how the Whisper backends are driven. Canary
returns native per-word timestamps, so it satisfies the
`transcribe()` / `ts_words()` / `segments_end_ts()` contract with no timestamp
reconstruction. This is the lowest-risk path and reuses the entire
VAD → ASR → output pipeline unchanged.

The alternative — NeMo's native AlignAtt streaming decoder wired onto the
SimulStreaming policy — is much lower latency but tightly coupled to NeMo
streaming internals and far more complex. It is deferred to Phase 2.

## Goals (Phase 1)

- New `--backend canary` selectable backend, running on CUDA.
- Real-time streaming transcription via the existing LocalAgreement policy.
- Native word-level timestamps feeding the existing UI / diarization alignment.
- Per-session explicit language selection (25 languages) via the existing
  `SessionASRProxy` seam.
- Multilingual **auto** language detection (`lan=auto`) via a lightweight
  NeMo LID front-step (AmberNet), detect-once-then-lock per session.
- Diarization and the existing NLLB/AlignAtt translation engines continue to
  compose unchanged.

## Non-goals (Phase 1)

- Canary's built-in direct translation (AST) as a translation backend — possible
  later add-on; word timestamps are unreliable for AST (segment-level only).
- NeMo native AlignAtt streaming (`canary-streaming`) — Phase 2, separate spec.
- Apple Silicon / CPU optimization — device-agnostic code, but CUDA is the
  supported/target path.
- Re-detection of language mid-session after the initial lock.

## Architecture

### New file: `whisperlivekit/canary_backend.py`

Lazy-imports NeMo (so non-Canary users never load it). Contains two classes.

#### `CanaryASR` — shared model holder + LocalAgreement contract

Loaded once by the `TranscriptionEngine` singleton.

Required attributes (mirroring `ASRBase` consumers in `online_asr.py`):
`sep=" "`, `original_language`, `backend_choice="canary"`,
`confidence_validation`, `tokenizer`, `buffer_trimming`, `buffer_trimming_sec`.

Methods:

- `load_model(...)` — `ASRModel.from_pretrained("nvidia/canary-1b-v2")` (or a
  local path / alternate Canary model from `--canary-model`), moved to CUDA.
- `transcribe(audio, init_prompt="", source_lang=None)` — resolves the effective
  source language (`source_lang` arg > `self.original_language` > default),
  calls `model.transcribe([audio], source_lang=L, target_lang=L,
  timestamps=True)`, returns the hypothesis object. `init_prompt` is accepted for
  interface compatibility (Canary has no direct prompt slot — passed through only
  if a supported mechanism exists, otherwise ignored).
- `ts_words(res)` — map `res.timestamp["word"]` `{word,start,end}` → `ASRToken`.
- `segments_end_ts(res)` — end times from `res.timestamp["segment"]`.
- `use_vad()` — no-op / warning (VAD handled upstream by Silero).

#### `CanaryLID` — shared language-ID model

Loaded once (only when any session may need `auto`).

- `load_model(...)` — `EncDecSpeakerLabelModel.from_pretrained("langid_ambernet")`,
  moved to CUDA. ~29M params / ~110 MB; VoxLingua107 (107 langs) — a superset of
  Canary's 25.
- `detect(audio_np) -> (lang_code, confidence)` — run `forward()` on the 16 kHz
  array, softmax, argmax → VoxLingua107 code, then map to Canary's accepted
  `source_lang` set via a small code map (validate edge cases e.g. `el`, `uk`,
  `mt`). Returns the mapped code and the max probability.

### `core.py` changes

- In `TranscriptionEngine._do_init()`, add an
  `elif config.backend == "canary":` branch **before** the `backend_policy`
  check (placed with the `voxtral` / `qwen3` branches). Instantiate `CanaryASR`;
  instantiate `CanaryLID` when `config.lan == "auto"`. Stash the LID instance on
  the engine so `online_factory()` can reach it.
- In `online_factory()`, add `if backend == "canary": ...` returning an
  `OnlineASRProcessor` wrapping the (possibly proxied) ASR — routed **before**
  the `backend_policy == "simulstreaming"` branch so it works regardless of the
  default policy.

### Auto language detection

Constraint: `CanaryASR` and `CanaryLID` are shared singletons, but the detected
language is per-session. The existing `SessionASRProxy` is the per-session seam
(wraps `transcribe()` under a lock, overrides the language).

**Mechanism (`lan == "auto"` sessions only):**

1. Introduce `CanarySessionASR`, a subclass of `SessionASRProxy` (or a thin
   wrapper) that holds a reference to the shared `CanaryLID` and a per-session
   `detected_lang` (initially unset).
2. Until `detected_lang` is set, `transcribe()` uses a configurable fallback
   `--canary-default-lang` (default `en`) as `source_lang`, so the first partial
   window still produces output.
3. On the first window with committed audio duration `>= --canary-lid-min-sec`
   (default `2.0`), call `CanaryLID.detect(window)` once. If confidence
   `>= --canary-lid-min-conf` (default `0.5`), cache and lock `detected_lang`
   for the rest of the session. Otherwise keep the fallback and retry on the next
   window, up to a bounded number of attempts, then stay on the fallback.
4. Explicit-language sessions (e.g. `?lang=de`) bypass all of this via the normal
   proxy path — no LID model touched.

This keeps LID concerns out of both the shared `CanaryASR` and the generic
`SessionASRProxy`. Detect-once-and-lock; no mid-session re-detection in Phase 1.

### Data flow (unchanged pipeline)

```
FFmpeg decode → Silero VAD → OnlineASRProcessor (growing window)
  → [CanarySessionASR / SessionASRProxy] → CanaryASR.transcribe() on CUDA
  → ts_words() → HypothesisBuffer LocalAgreement commit → stream to client
```

Diarization (sortformer/diart) and translation (NLLB / AlignAtt) engines are
untouched and compose exactly as today.

## Configuration & dependencies

### CLI / config additions (`parse_args.py`, `config.py`)

- `--backend canary` — add `"canary"` to the `--backend` choices.
- `--canary-model` (default `nvidia/canary-1b-v2`) — HF repo id or local path.
- `--canary-default-lang` (default `en`) — fallback while auto-detecting.
- `--canary-lid-model` (default `langid_ambernet`).
- `--canary-lid-min-sec` (default `2.0`) — min committed audio before LID runs.
- `--canary-lid-min-conf` (default `0.5`) — min LID confidence to lock.
- Reuse existing `buffer_trimming` / `buffer_trimming_sec` (already default
  `segment` / `15.0`, which suits Canary's long-window tolerance).

Add the corresponding fields to `WhisperLiveKitConfig` and thread them through
`transcription_common_params` (or a `canary_params` dict, matching the pattern
used for `qwen3_streaming_params` / `voxtral`).

### Dependencies

- `nemo_toolkit[asr]` (heavy: PyTorch Lightning + full ASR stack). Timestamp
  support requires recent NeMo (main / >= 2.5).
- Add as an **optional extra** in `pyproject.toml` (e.g. `.[canary]`),
  **lazy-imported** inside `canary_backend.py` so non-Canary installs never load
  NeMo — consistent with the other heavy backends (voxtral, qwen3).

## Testing

Per CLAUDE.md: real audio via `TestHarness`, no mock-based unit tests. All
Canary tests gated to run only when NeMo + model weights are available (skip
otherwise), like the other heavy-backend tests.

1. **`CanaryASR`**: load model, `transcribe()` a known WAV, assert `ts_words()`
   returns `ASRToken`s with monotonic, sane timestamps and WER below a threshold.
2. **`CanaryLID`**: feed clips in 2–3 languages; assert correct detected code and
   correct VoxLingua107 → Canary mapping.
3. **End-to-end (`TestHarness`)**: backend=`canary`, feed real audio at
   `speed=1.0`, `drain()`, assert committed text + WER threshold. A second run
   with `lan="auto"` asserts the session locks to the correct language after the
   first window.
4. **Routing**: `--backend canary` with the default `backend_policy` routes to
   `OnlineASRProcessor` (not SimulStreaming), regardless of policy default.

## Phased delivery

- **Phase 1 (this spec):** `canary` backend on LocalAgreement + AmberNet
  auto-detect.
- **Phase 2 (future, separate spec):** `canary-streaming` backend wrapping
  NeMo's `speech_to_text_aed_streaming` AlignAtt decoder onto the SimulStreaming
  policy for lower latency.

## Open questions / risks

- **NeMo version pin:** timestamp API needs a recent NeMo; pin a known-good
  version in the optional extra and document it.
- **`init_prompt`:** Canary has no direct prompt-conditioning slot like Whisper;
  the LocalAgreement `prompt()` context may be unused. Confirm whether Canary
  exposes any decoder-context mechanism; if not, document that `init_prompt` is
  ignored (LocalAgreement still functions without it).
- **Repeated full-window decode cost:** LocalAgreement re-decodes the growing
  window each tick. On CUDA with a 1B model this is acceptable, but tune
  `min_chunk_size` / `buffer_trimming_sec` to bound latency and compute; Phase 2
  streaming removes this cost.
- **LID code mapping:** validate every VoxLingua107 → Canary code (Greek `el`,
  Ukrainian `uk`, Maltese `mt`) against Canary's exact accepted `source_lang`
  token set.
