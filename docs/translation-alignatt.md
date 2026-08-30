# Simultaneous translation with AlignAtt4LLM

[AlignAtt4LLM](https://github.com/QuentinFuxa/Alignatt4LLM) is a companion project by Quentin Fuxa and Dominik Macháček. Their [paper, accepted to IWSLT 2026](https://arxiv.org/abs/2606.03967), describes how to apply AlignAtt to decoder-only LLMs using selected translation alignment heads and runtime query/key capture.

WhisperLiveKit is a client of its `alignatt-mt-server`: WLK handles audio, ASR, and display; the separate process loads the translation model. This lets an ASR client on one machine use a CUDA translation server on another, without combining their Python environments.

## What is shared with the paper

The translation policy and runtime come from AlignAtt4LLM. The paper evaluates a specific Qwen3-ASR/forced-alignment and Gemma cascade for English to German, Italian, and Chinese. Changing the WLK ASR backend, hardware, or buffering creates a different configuration; the paper's latency and quality results do not automatically transfer to it.

For citation metadata, see [AlignAtt4LLM's citation file](https://github.com/QuentinFuxa/Alignatt4LLM/blob/main/CITATION.cff) and [the paper](https://arxiv.org/abs/2606.03967).

An optional [MLX implementation](simul_mt_calibration.md) runs translation locally
on Apple Silicon. Its calibrations and measurements are specific to that runtime;
the remote server remains a separate backend with its own supported models.

## Start the translation server

Use a separate CUDA environment. Follow the [AlignAtt4LLM installation instructions](https://github.com/QuentinFuxa/Alignatt4LLM#install), which pin its inference dependencies and model revisions. Models must be present in the Hugging Face cache; the Gemma model requires access to its gated weights.

```bash
git clone https://github.com/QuentinFuxa/Alignatt4LLM.git
cd Alignatt4LLM
tools/bootstrap/setup_inference_qwen_asr_vllm.sh
# Complete the model-cache setup documented in that repository, then:
.venv-inference/bin/alignatt-mt-server --preset gemma_low_latency --port 8765
```

The MT server checks source/target directions against the calibrated alignment-head files it has installed. Consult its returned `unsupported_direction` error for the supported list. An ASR model's language coverage does not imply the same translation coverage.

## Connect WhisperLiveKit

For English audio and German translation:

```bash
wlk --model base --language en --target-language de \
    --translation-backend alignatt \
    --alignatt-url ws://localhost:8765
```

Replace `localhost` with the reachable translation-server host when using separate machines. Install WhisperLiveKit and its ASR extra in its own environment. The `translation` extra is for NLLB and is not required by this remote backend.

For the causal Qwen3 ASR route:

```bash
pip install "whisperlivekit[qwen3-streaming]"
wlk --backend qwen3-streaming --language en \
    --qwen3-streaming-audio-backend causal \
    --target-language de --translation-backend alignatt \
    --alignatt-url ws://gpu-host:8765
```

Use an explicit source language matching the audio. A native WebSocket session can override both the source and target with `/asr?language=fr&target_language=zh`. Each MT connection receives its session's direction; supported directions still depend on the server's calibration.

## Streaming behavior

Committed source words are sent with timestamps. In `balanced` and `low` modes, WLK also sends its current unstable hypothesis tail without timestamps. AlignAtt4LLM can draft over that tail while restricting emission to source words already committed. If source commitment advances without changing the draft's prompt, the server can release held translation tokens without another decode.

| `--alignatt-latency` | Client behavior |
|---|---|
| `quality` | Sends committed words only; requests one unit of target holdback |
| `balanced` (default) | Also sends the unstable ASR tail; keeps the server's holdback settings |
| `low` | Sends the tail and requests zero target holdback and minimum emission size |

These are policy settings, not measured latency guarantees. `--alignatt-context "talk title, glossary terms"` supplies server-wide translation context. `--translate-on-complete` holds source tokens until WLK finalizes a segment, which reduces partial translation updates.

Partial translations are exposed as `buffer_translation`. Within an open utterance, accepted partials extend the existing prefix. The protocol also supports a final translation that replaces the partial at the line level; this is not a blanket guarantee that all displayed text is immutable.

## Current limits

The protocol adapter is covered by a local WebSocket scenario test, including a partial before punctuation, a final at a boundary, and the remaining words at end-of-stream. This checks transport and WLK state propagation, not model quality. The IWSLT evaluation remains in the research repository.

If the server is unavailable or rejects a direction, WLK keeps transcribing and exposes `translation_error` in full/diff results and in the browser. Reconnection uses backoff and sends the accepted target prefix and recent history. A successful update clears the error.

Silence and speaker boundaries request the final quality pass before committing a translation segment. End-of-stream drains pending finals and an unpunctuated last utterance. Cleanup closes the WebSocket, including on cancellation. If the remote server cannot finish, the last partial remains a buffer and `translation_error` explicitly reports incomplete translation. These lifecycle guarantees are checked against the local protocol server; quality and CUDA performance still require the real MT model.

The [protocol specification](https://github.com/QuentinFuxa/Alignatt4LLM/blob/main/docs/mt_server_protocol.md) is authoritative. The WLK implementation is [translation_alignatt.py](../whisperlivekit/translation_alignatt.py), with [protocol tests](../tests/test_translation_alignatt.py).
