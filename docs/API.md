# WhisperLiveKit API Reference

This document describes all APIs: the WebSocket streaming API, the OpenAI-compatible REST API, and the CLI.

---

## REST API (OpenAI-compatible)

### POST /v1/audio/transcriptions

Compatibility-oriented subset of the OpenAI Audio Transcriptions API. Only the
parameters listed below are accepted, and some are currently ignored as noted.

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F response_format=json
```

**Parameters (multipart form):**

| Parameter                 | Type     | Default | Description |
|--------------------------|----------|---------|-------------|
| `file`                   | file     | required | Audio file (any format ffmpeg can decode) |
| `model`                  | string   | `""`     | Accepted but ignored (uses server's backend) |
| `language`               | string   | `null`   | ISO 639-1 language code or null for auto-detection |
| `prompt`                 | string   | `""`     | Per-request terminology or names used as decoder context (Whisper-family and SimulStreaming backends) |
| `response_format`        | string   | `"json"` | `json`, `verbose_json`, `diarized_json`, `text`, `srt`, `vtt` |
| `timestamp_granularities`| array    | `null`   | Accepted for compatibility |

**Response formats:**

`json` (default):
```json
{
  "text": "Hello world, how are you?",
  "usage": {"type": "duration", "seconds": 7}
}
```

`verbose_json`:
```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 7.16,
  "text": "Hello world",
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.5},
    {"word": "world", "start": 0.5, "end": 1.0}
  ],
  "segments": [
    {"id": 0, "start": 0.0, "end": 1.0, "text": "Hello world"}
  ],
  "usage": {"type": "duration", "seconds": 7}
}
```

`diarized_json`:

This format requires the server to be started with `--diarization`. Requests
made without diarization enabled return HTTP 400 before the audio is processed.

```json
{
  "task": "transcribe",
  "duration": 7.16,
  "text": "A: Hello world\nB: Hi there",
  "segments": [
    {
      "type": "transcript.text.segment",
      "id": "seg_001",
      "start": 0.0,
      "end": 1.0,
      "text": "Hello world",
      "speaker": "A"
    },
    {
      "type": "transcript.text.segment",
      "id": "seg_002",
      "start": 1.1,
      "end": 2.0,
      "text": "Hi there",
      "speaker": "B"
    }
  ],
  "usage": {"type": "duration", "seconds": 7}
}
```

`text`: Plain text response.

`srt` / `vtt`: Subtitle format.

When no speech is detected, the endpoint still returns the schema selected by
`response_format`. JSON formats contain empty text and segment or word arrays
where applicable. Text and SRT responses are empty, while VTT contains its
`WEBVTT` header.

### GET /v1/models

List the currently loaded model.

```bash
curl http://localhost:8000/v1/models
```

### GET /health

Server health check.

```bash
curl http://localhost:8000/health
```

---

## Deepgram-Compatible WebSocket API

### WS /v1/listen

Compatibility-oriented subset of Deepgram's Live Transcription WebSocket.
The adapter is verified with `deepgram-sdk==7.7.0`. Install that version and use
only the parameters and messages documented below.

```bash
pip install deepgram-sdk==7.7.0
```

```python
from deepgram import AsyncDeepgramClient
from deepgram.environment import DeepgramClientEnvironment

environment = DeepgramClientEnvironment(
    base="http://localhost:8000",
    production="ws://localhost:8000",
    agent="ws://localhost:8000",
    agent_rest="http://localhost:8000",
)
# Use the WLK API token here when the server was started with --api-token.
deepgram = AsyncDeepgramClient(api_key="unused", environment=environment)

async with deepgram.listen.v1.connect(
    model="nova-3",
    language="en",
    encoding="linear16",
    sample_rate=16000,
    channels=1,
    interim_results=True,
    endpointing=300,
    utterance_end_ms=1000,
    vad_events=True,
) as connection:
    await connection.send_media(pcm_audio)
    message = await connection.recv()
    await connection.send_close_stream()
```

**Supported Query Parameters:**

- `language`: per-session source language
- `context`: WLK extension for per-session terminology, names, or phrase-list
  text used as decoder context
- `model`: accepted for SDK compatibility; the server's configured WLK backend
  remains authoritative
- `encoding=linear16`: select raw signed 16-bit PCM input; requires
  `sample_rate=16000`
- `sample_rate=16000`: WLK's fixed input sample rate for raw PCM
- `channels=1`: mono input; this is the only supported channel count
- `interim_results=true|false`: emit revisable `is_final=false` hypotheses;
  defaults to `false`
- `endpointing=false|N`: disable pause endpoints or set a positive silence
  threshold in milliseconds; defaults to 10 ms
- `utterance_end_ms=N`: enable the independent finalized-word gap detector;
  requires `interim_results=true` and accepts integers from 1000 to 5000
- `vad_events=true|false`: emit `SpeechStarted` on VAD speech transitions;
  defaults to `false`

Numeric endpointing and `vad_events=true` require the server's VAC to be enabled.
Endpointing uses submitted audio timestamps, not wall-clock time, and is
independent from `--pause-segmentation-seconds`. A pause endpoint emits
`speech_final=true` once. `UtteranceEnd` uses its own timer after the last word of
a final result and is deduplicated for that word timestamp.

**Client Messages:**
- Binary audio frames
- `{"type": "KeepAlive"}`: keep the connection alive without advancing audio time
- `{"type": "CloseStream"}`: drain pending audio, receive final Results and
  summary Metadata, then close

Deepgram's non-terminal `Finalize` control is rejected explicitly. WLK backends
currently expose only a terminal flush, so accepting more audio afterward would
silently lose it. Send `CloseStream` when the audio is complete. An empty binary
frame is also rejected and is not treated as EOF.

**Server Messages:**
- `Metadata`: request metadata and the final stream summary
- `Results`: revisable or immutable transcript ranges with independent
  `is_final` and `speech_final` flags
- `UtteranceEnd`: finalized-word gap reached when configured
- `SpeechStarted`: VAD speech begins when `vad_events=true`

**Limitations vs Deepgram:**
- Self-hosted authentication uses the optional WLK API token rather than a
  Deepgram project key
- Native ASR token timestamps are preserved when available; multi-word source
  tokens fall back to interpolation within their token span
- Confidence scores are 0.0 (not available)
- `Finalize` and channel-specific finalization are not supported

---

## CLI

### `wlk` / `wlk serve`

Start the transcription server.

```bash
wlk                                    # Start with defaults
wlk --backend voxtral --model base     # Specific backend
wlk serve --port 9000 --lan fr         # Explicit serve command
```

### `wlk listen`

Live microphone transcription. Requires `sounddevice` (`pip install sounddevice`).

```bash
wlk listen                             # Transcribe from microphone
wlk listen --backend voxtral           # Use specific backend
wlk listen --language fr               # Force French
wlk listen --diarization               # With speaker identification
wlk listen -o transcript.txt           # Save to file on exit
```

Committed lines print as they are finalized. The current buffer (partial transcription) is shown in gray and updates in-place. Press Ctrl+C to stop; remaining audio is flushed before exit.

### `wlk run`

Auto-pull model if not downloaded, then start the server.

```bash
wlk run voxtral                        # Pull voxtral + start server
wlk run large-v3                       # Pull large-v3 + start server
wlk run faster-whisper:base            # Specific backend + model
wlk run qwen3-vllm-metal:0.6b          # Qwen3-ASR on Apple Silicon
wlk run voxtral --lan fr --port 9000   # Extra server options passed through
```

### `wlk transcribe`

Transcribe audio files offline (no server needed).

```bash
wlk transcribe audio.wav                          # Plain text output
wlk transcribe --format srt audio.wav             # SRT subtitles
wlk transcribe --format json audio.wav             # JSON output
wlk transcribe --backend voxtral audio.wav         # Specific backend
wlk transcribe --model large-v3 --language fr *.wav # Multiple files
wlk transcribe --output result.srt --format srt audio.wav
```

### `wlk bench`

Benchmark speed (RTF) and accuracy (WER) on standard test audio.

```bash
wlk bench                              # Benchmark with defaults
wlk bench --backend faster-whisper     # Specific backend
wlk bench --model large-v3             # Larger model
wlk bench --json results.json          # Export results
```

Downloads test audio from LibriSpeech on first run. Reports WER (Word Error Rate) and RTF (Real-Time Factor: processing time / audio duration).

### `wlk diagnose`

Run pipeline diagnostics on an audio file. Feeds audio through the full pipeline while probing internal backend state at regular intervals. Produces a timeline, flags anomalies, and prints health checks.

```bash
wlk diagnose audio.wav                        # Diagnose with default backend
wlk diagnose audio.wav --backend voxtral      # Diagnose specific backend
wlk diagnose --speed 0 --probe-interval 1     # Instant feed, probe every 1s
wlk diagnose                                   # Use built-in test sample
```

Useful for debugging issues like: no output appearing, slow transcription, stuck pipelines, or generate thread errors.

### `wlk models`

List available backends, installation status, and downloaded models.

```bash
wlk models
```

### `wlk pull`

Download models for offline use.

```bash
wlk pull base                      # Download for best available backend
wlk pull faster-whisper:large-v3   # Specific backend + model
wlk pull voxtral                   # Voxtral HF model
wlk pull qwen3-vllm-metal:0.6b    # Qwen3-ASR vLLM Metal 0.6B
```

### `wlk rm`

Delete downloaded models to free disk space.

```bash
wlk rm base                        # Delete base model
wlk rm voxtral                     # Delete Voxtral model
wlk rm faster-whisper:large-v3     # Delete specific backend model
```

### `wlk check`

Verify system dependencies (Python, ffmpeg, torch, etc.).

### `wlk version`

Print the installed version.

### Python Client (OpenAI SDK)

WhisperLiveKit's REST API is compatible with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-base",  # ignored, uses server's backend
        file=f,
        response_format="verbose_json",
    )
print(result.text)
```

### Programmatic Python API

For direct in-process usage without a server:

```python
import asyncio
from whisperlivekit import TranscriptionEngine, AudioProcessor

async def transcribe(audio_path):
    engine = TranscriptionEngine(model_size="base", lan="en")
    # ... use AudioProcessor for full pipeline control
```

Or use the TestHarness for simpler usage:

```python
import asyncio
from whisperlivekit import TestHarness

async def main():
    async with TestHarness(model_size="base", lan="en") as h:
        await h.feed("audio.wav", speed=0)
        result = await h.finish()
        print(result.text)

asyncio.run(main())
```

---

## WebSocket Streaming API

This section describes the WebSocket API for clients that want to stream audio and receive real-time transcription results from a WhisperLiveKit server.

---

## Connection

### Endpoint

```
ws://<host>:<port>/asr
```

### Query Parameters

| Parameter  | Type   | Default  | Description |
|------------|--------|----------|-------------|
| `language` | string | _(none)_ | Per-session language override. ISO 639-1 code (e.g. `fr`, `en`) or `"auto"` for automatic detection. When omitted, uses the server-wide language setting. Multiple sessions with different languages work concurrently. |
| `context` | string | _(none)_ | Terminology, names, or phrase-list text used as decoder context for this session. Maximum 1000 characters. |
| `mode`     | string | `"full"` | Output mode. `"full"` sends complete state on every update. `"diff"` sends incremental diffs after an initial snapshot. |

Example:
```
ws://localhost:8000/asr?language=fr&context=WhisperLiveKit%2C+Qwen3-ASR&mode=diff
```

### Connection Flow

1. Client opens a WebSocket connection to `/asr`.
2. Server accepts the connection and immediately sends a **config message**.
3. Client streams binary audio frames to the server.
4. Server sends transcription updates as JSON messages.
5. Client sends empty bytes (`b""`) to signal end of audio.
6. Server finishes processing remaining audio and sends a **ready_to_stop** message.

---

## Server to Client Messages

### Config Message

Sent once, immediately after the connection is accepted.

```json
{
  "type": "config",
  "useAudioWorklet": true,
  "mode": "full",
  "context": {
    "supported": true,
    "maxCharacters": 1000,
    "backend": "faster-whisper"
  }
}
```

| Field             | Type   | Description |
|-------------------|--------|-------------|
| `type`            | string | Always `"config"`. |
| `useAudioWorklet` | bool   | `true` when the server expects PCM s16le 16kHz mono input (started with `--pcm-input`). `false` when the server expects encoded audio (decoded server-side via FFmpeg). |
| `mode`            | string | `"full"` or `"diff"`, echoing the requested mode. |
| `context`         | object | Decoder-context capability for the configured backend. |

### Transcription Update (full mode)

Sent repeatedly as audio is processed. This message has **no `type` field**.

```json
{
  "status": "active_transcription",
  "lines": [
    {
      "speaker": 1,
      "text": "Hello world, how are you?",
      "start": "0:00:00",
      "end": "0:00:03"
    },
    {
      "speaker": 2,
      "text": "I am fine, thanks.",
      "start": "0:00:04",
      "end": "0:00:06",
      "translation": "Je vais bien, merci.",
      "detected_language": "en"
    }
  ],
  "buffer_transcription": "And you",
  "buffer_diarization": "",
  "buffer_translation": "",
  "remaining_time_transcription": 1.2,
  "remaining_time_diarization": 0.5
}
```

| Field                          | Type   | Description |
|--------------------------------|--------|-------------|
| `status`                       | string | `"active_transcription"` during normal operation. `"no_audio_detected"` when no speech has been detected yet. |
| `lines`                        | array  | Committed transcription segments. Each update sends the **full list** of all committed lines (not incremental). |
| `buffer_transcription`         | string | Ephemeral transcription text not yet committed to a line. Displayed in real time but overwritten on every update. |
| `buffer_diarization`           | string | Ephemeral text waiting for speaker attribution. |
| `buffer_translation`           | string | Ephemeral translation text for the current buffer. |
| `remaining_time_transcription` | float  | Seconds of audio waiting to be transcribed (processing lag). |
| `remaining_time_diarization`   | float  | Seconds of audio waiting for speaker diarization. |
| `error`                        | string | Only present when an error occurred (e.g. FFmpeg failure). |

#### Line Object

Each element in `lines` has the following shape:

| Field               | Type   | Presence    | Description |
|---------------------|--------|-------------|-------------|
| `speaker`           | int    | Always      | Speaker ID. Normally `1`, `2`, `3`, etc. The special value `-2` indicates a silence segment. When diarization is disabled, defaults to `1`. |
| `text`              | string | Always      | The transcribed text for this segment. `null` for silence segments. |
| `start`             | string | Always      | Start timestamp formatted as `H:MM:SS` (e.g. `"0:00:03"`). |
| `end`               | string | Always      | End timestamp formatted as `H:MM:SS`. |
| `translation`       | string | Conditional | Present only when translation is enabled and available for this line. |
| `detected_language` | string | Conditional | Present only when language detection produced a result for this line (e.g. `"en"`). |

### Snapshot (diff mode)

When `mode=diff`, the first transcription message is always a snapshot containing the full state. It has the same fields as a full-mode transcription update, plus metadata fields.

```json
{
  "type": "snapshot",
  "seq": 1,
  "status": "active_transcription",
  "lines": [ ... ],
  "buffer_transcription": "",
  "buffer_diarization": "",
  "buffer_translation": "",
  "remaining_time_transcription": 0.0,
  "remaining_time_diarization": 0.0
}
```

| Field  | Type   | Description |
|--------|--------|-------------|
| `type` | string | `"snapshot"`. |
| `seq`  | int    | Monotonically increasing sequence number, starting at 1. |
| _(remaining fields)_ | | Same as a full-mode transcription update. |

### Diff (diff mode)

All messages after the initial snapshot are diffs.

```json
{
  "type": "diff",
  "seq": 4,
  "status": "active_transcription",
  "n_lines": 5,
  "lines_pruned": 1,
  "new_lines": [
    {
      "speaker": 1,
      "text": "This is a new line.",
      "start": "0:00:12",
      "end": "0:00:14"
    }
  ],
  "buffer_transcription": "partial text",
  "buffer_diarization": "",
  "buffer_translation": "",
  "remaining_time_transcription": 0.3,
  "remaining_time_diarization": 0.1
}
```

| Field                          | Type   | Presence    | Description |
|--------------------------------|--------|-------------|-------------|
| `type`                         | string | Always      | `"diff"`. |
| `seq`                          | int    | Always      | Sequence number. |
| `status`                       | string | Always      | Same as full mode. |
| `n_lines`                      | int    | Always      | Total number of lines the client should have after applying this diff. Use this to verify sync. |
| `lines_pruned`                 | int    | Conditional | Number of lines to remove from the **front** of the client's line list. Only present when > 0. |
| `new_lines`                    | array  | Conditional | Lines to append to the **end** of the client's line list. Only present when there are new lines. |
| `buffer_transcription`         | string | Always      | Replaces the previous buffer value. |
| `buffer_diarization`           | string | Always      | Replaces the previous buffer value. |
| `buffer_translation`           | string | Always      | Replaces the previous buffer value. |
| `remaining_time_transcription` | float  | Always      | Replaces the previous value. |
| `remaining_time_diarization`   | float  | Always      | Replaces the previous value. |
| `error`                        | string | Conditional | Only present on error. |

### Ready to Stop

Sent after all audio has been processed (i.e., after the client sent the end-of-audio signal and the server finished processing the remaining audio).

```json
{
  "type": "ready_to_stop"
}
```

---

## Client to Server Messages

### Audio Frames

Send binary WebSocket frames containing audio data.

**When `useAudioWorklet` is `true` (server started with `--pcm-input`):**
- PCM signed 16-bit little-endian, 16 kHz, mono (`s16le`).
- Any chunk size works. A typical chunk is 0.5 seconds (16,000 bytes).

**When `useAudioWorklet` is `false`:**
- Raw encoded audio bytes (any format FFmpeg can decode: WAV, MP3, FLAC, OGG, etc.).
- The server pipes these bytes through FFmpeg for decoding.

### End-of-Audio Signal

Send an empty binary frame (`b""`) to tell the server that no more audio will follow. The server will finish processing any remaining audio and then send a `ready_to_stop` message.

---

## Diff Protocol: Client Reconstruction

Clients using `mode=diff` must maintain a local list of lines and apply diffs incrementally.

### Algorithm

```python
def reconstruct_state(msg, lines):
    """Apply a snapshot or diff message to a local lines list.

    Args:
        msg: The parsed JSON message from the server.
        lines: The client's mutable list of line objects.

    Returns:
        A full-state dict with all fields.
    """
    if msg["type"] == "snapshot":
        lines.clear()
        lines.extend(msg.get("lines", []))
        return msg

    # Apply diff
    n_pruned = msg.get("lines_pruned", 0)
    if n_pruned > 0:
        del lines[:n_pruned]

    new_lines = msg.get("new_lines", [])
    lines.extend(new_lines)

    # Volatile fields are replaced wholesale
    return {
        "status": msg.get("status", ""),
        "lines": lines[:],
        "buffer_transcription": msg.get("buffer_transcription", ""),
        "buffer_diarization": msg.get("buffer_diarization", ""),
        "buffer_translation": msg.get("buffer_translation", ""),
        "remaining_time_transcription": msg.get("remaining_time_transcription", 0),
        "remaining_time_diarization": msg.get("remaining_time_diarization", 0),
    }
```

### Verification

After applying a diff, check that `len(lines) == msg["n_lines"]`. A mismatch indicates the client fell out of sync and should reconnect.

---

## Silence Representation

Silence segments are represented as lines with `speaker` set to `-2`. Their
`text` is `null` with diarization and an empty string without diarization, so
clients should identify them by `speaker`:

```json
{
  "speaker": -2,
  "text": null,
  "start": "0:00:10",
  "end": "0:00:12"
}
```

Silence segments are generated when a VAD pause is longer than
`--pause-segmentation-seconds`, which defaults to 5 seconds. The qualifying pause
creates one stable line boundary without changing word text or timestamps, both
with and without diarization. Use `--pause-segmentation-seconds 0` to disable these
boundaries. The boundary is committed after the pause ends, when speech resumes
or end of input is received; an in-progress pause is not added to `lines`. The
native `/asr` WebSocket uses the server-wide setting. The
Deepgram-compatible `/v1/listen` endpoint instead keeps its per-session
`endpointing` timer separate from these native output lines.

---

## Per-Session Language

The `language` query parameter creates an isolated language context for the session using `SessionASRProxy`. The proxy temporarily overrides the shared ASR backend's language during transcription calls, protected by a lock. This means:

- Each WebSocket session can transcribe in a different language.
- Sessions are thread-safe and do not interfere with each other.
- Pass `"auto"` to use automatic language detection for the session regardless of the server-wide setting.

## Per-Session Decoder Context

The native and Deepgram-compatible WebSockets accept `context`; the
OpenAI-compatible REST endpoint uses its standard `prompt` form field. The value
is isolated per session and limited to 1000 characters.

- LocalAgreement prepends the context to its rolling committed-transcript prompt
  on every inference.
- SimulStreaming adds it to a per-session copy of the static prompt, so it does
  not scroll out as audio advances.
- Whisper, Faster-Whisper, MLX Whisper, OpenAI API, and SimulStreaming support
  this conditioning. FunASR, Canary, Qwen3, and Voxtral currently do not expose
  a compatible prompt slot; requesting context with them returns HTTP 400 or a
  WebSocket error with close code 4400 before audio processing starts.

Context influences decoding but does not guarantee that a phrase will appear.
N-best hypotheses, external scorer hooks, and public confidence streams remain
separate capabilities and are not introduced by this option.
