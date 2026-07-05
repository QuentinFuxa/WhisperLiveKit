# AlignAtt translation backend (Alignatt4LLM sidecar)

`--translation-backend alignatt` replaces the in-process NLLB translator
with [Alignatt4LLM](https://github.com/QuentinFuxa/Alignatt4LLM): a
decoder-only LLM (Gemma-4-E4B by default) drafts the translation, and the
AlignAtt policy commits only the target prefix whose attention lands on
source words the ASR has actually committed. Output is append-only: a
committed translation is never retracted.

## Why it pairs well with WhisperLiveKit

- The unstable ASR hypothesis tail is sent to the translator as
  "inaccessible" source: the model drafts ahead over it, but cannot commit
  against it. When the ASR commits those words, the sidecar releases the
  held target tokens from its cached draft without a new MT call, so
  translation latency tracks ASR commit latency instead of adding to it.
- With the qwen3 causal backend (append-only commits, English-only tower),
  en to de/it/zh/cs/fr translation restores multilingual output while the
  whole chain stays append-only end to end.
- Punctuation, silence and speaker-change boundaries trigger a full-quality
  re-translation of the finished line (the streamed partial is reused as a
  prefill, so the upgrade is cheap).

## Running the sidecar (CUDA box, one-time setup)

The MT engine is vLLM-on-CUDA only (about 40 GB VRAM with the defaults).
In a clone of Alignatt4LLM:

```bash
tools/bootstrap/setup_inference_qwen_asr_vllm.sh   # pinned vLLM stack, Python 3.13

# Gemma-4-E4B is gated on Hugging Face: accept the license, then
huggingface-cli download google/gemma-4-E4B-it

.venv-inference/bin/alignatt-mt-server --preset gemma_low_latency --port 8765
```

Supported directions ship as calibrated alignment-head files:
en to de, it, zh, cs, fr for Gemma; en to de for the ungated Qwen3-1.7B
fallback (`--mt-backend qwen_vllm_alignatt`). The server rejects other
directions at session init with the supported list.

## Running WhisperLiveKit against it

```bash
wlk --backend qwen3-streaming --language en \
    --qwen3-streaming-audio-backend causal \
    --target-language de \
    --translation-backend alignatt \
    --alignatt-url ws://gpu-host:8765
```

Per-session targets keep working: `ws://host:8000/asr?target_language=zh`.
If the sidecar is down or the direction unsupported, the session keeps
transcribing, translation stays empty, and the client reconnects with
backoff (append-only output is preserved across reconnects).

## Latency points

| `--alignatt-latency` | behavior | typical word-to-translation p50 (causal ASR) |
|---|---|---|
| `quality` | committed words only, one unit of target holdback | commit lag + 1-2 s |
| `balanced` (default) | drafts over the unstable tail, releases on commit | about commit lag + 0.3 s |
| `low` | balanced, plus zero target-side holdback | lowest; pair with `--qwen3-streaming-hold-back-words 2` (about 2.5 s end to end) |

`--translate-on-complete` composes with this backend as a finals-only mode:
tokens are held until punctuation, so the sidecar only runs full-quality
utterance translations (the cheapest multi-session configuration).

`--alignatt-context "talk title, glossary terms"` injects domain context
into the MT prompt for every session of the server.

## Protocol

The client speaks protocol v1 of `alignatt-mt-server`, specified in the
Alignatt4LLM repository under `docs/mt_server_protocol.md`. The WLK side
lives in [translation_alignatt.py](../whisperlivekit/translation_alignatt.py)
and is exercised against an in-process fake sidecar in
[tests/test_translation_alignatt.py](../tests/test_translation_alignatt.py)
(no GPU needed).
