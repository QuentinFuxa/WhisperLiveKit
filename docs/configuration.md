# Server configuration

`wlk serve --help` is the complete CLI reference. Python integrations use
[`WhisperLiveKitConfig`](../whisperlivekit/config.py); the examples below use the
same settings as the CLI.

## Common settings

| Option | Purpose | Default |
|---|---|---|
| `--backend` | ASR implementation; see [backends](backends.md) | `auto` |
| `--model` | Model size or name for the selected backend | `base` |
| `--language` | Source language; coverage depends on the backend | `auto` |
| `--backend-policy` | Whisper streaming policy: `simulstreaming` or `localagreement` | `simulstreaming` |
| `--target-language` | Enable translation to this language | disabled |
| `--translation-backend` | `nllb`, the [AlignAtt4LLM server](translation-alignatt.md), or [local MLX translation](translation-mlx.md) | `nllb` |
| `--diarization` | Attribute words to speakers | disabled |
| `--pause-segmentation-seconds` | Split after a VAD pause longer than this threshold; `0` disables it | `5.0` |
| `--asr-coalesce-min-s` | Accumulate new audio before inference; trades update cadence for fewer calls | `0` |
| `--max-buffered-audio` | Maximum queued audio, in seconds, per ASR/diarization stage | `30` |
| `--backpressure-timeout` | Maximum wait for room in a processing queue, in seconds | `30` |
| `--pcm-input` | Accept mono 16 kHz signed 16-bit little-endian PCM and bypass FFmpeg | disabled |
| `--api-token` | Require authentication; falls back to `WLK_API_TOKEN` | unset |
| `--cors-origins` | Comma-separated browser origins allowed to call the API | none |
| `--host` / `--port` | Listening address | `localhost` / `8000` |

Backend-specific model paths, language restrictions and optional dependencies
are described in [backends.md](backends.md) and
[default_and_custom_models.md](default_and_custom_models.md).

## Decoding and session context

SimulStreaming selects greedy decoding when `--beams 1`, and beam search with
more beams. `--decoder beam` or `--decoder greedy` overrides that choice;
`greedy` requires one beam. `--init-prompt`, `--static-init-prompt` and
`--max-context-tokens` configure Whisper context. Backend support differs.

The native WebSocket accepts per-session `language`, `target_language` and
`context` parameters. REST uses `language` and `prompt`. Unsupported context is
rejected before processing audio. See [API.md](API.md) for protocol details.

The old `--punctuation-split` and `--disable-punctuation-split` flags are retained
for compatibility but have no effect and emit a warning. Diarization speaker
turns and `--pause-segmentation-seconds` control the corresponding boundaries.

## Long sessions and file requests

`--retention-seconds` sets transcript history retention. Without an override,
`mode=full` retains the entire transcript and `mode=diff` retains 300 seconds on
the server. Diff clients must maintain their own accumulated transcript. Zero
or a negative explicit value means unlimited retention. A finite retention in
full mode intentionally removes older lines from subsequent responses.

REST accepts up to 512 MB of encoded audio and up to 512 MB after PCM conversion.
Conversion has a 120-second timeout by default. The subsequent pipeline budget,
including feeding and final drainage, is `max(120 seconds, 2.5 × audio duration)`.
`--rest-timeout N` sets each phase's budget to N seconds. Timeout returns HTTP 408;
cancellation and errors close the processor's tasks and conversion process.

Pending audio is bounded independently in the ASR and diarization queues.
When a queue is full, audio ingestion waits for room instead of dropping
samples. Processing queues also hold at most 256 items, including silence and
speaker boundaries; the translation queue uses this item limit. A queue that
cannot accept work within `--backpressure-timeout` ends the session with an
explicit error (HTTP 503 for REST; an error message followed by close code 1011
for WebSocket). Increase the timeout for slow models or long inference calls.
Both settings must be finite and positive.

`SESSION_METRICS` logs include `peak_queued_audio_s` (the largest audio queue)
and `backpressure_wait_s` (cumulative producer wait across processing queues).
They exclude the batch currently being processed and are not word latency.

These bounds do not cap all process memory: model buffers, an incoming network
message, REST uploads, translation-backend history and full transcript history
have their own lifetimes. Measure the workload and limit concurrent sessions
at deployment. See [deployment.md](deployment.md).

## Embedding the server

Importing the server does not parse your program's command-line arguments.
Construct an ASGI application explicitly:

```python
from whisperlivekit import WhisperLiveKitConfig
from whisperlivekit.basic_server import create_app

app = create_app(WhisperLiveKitConfig(
    backend="faster-whisper",
    model_size="small",
    lan="fr",
    pcm_input=True,
))
```

The model loads at application startup. The transcription engine is shared
within the process; use separate processes for different model configurations.
`whisperlivekit.basic_server:app` remains available with default settings.
