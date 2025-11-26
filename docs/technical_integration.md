# Technical Integration Guide

This document introduce how to reuse the core components when you do **not** want to ship the bundled frontend, FastAPI server, or even the provided CLI.

---

## 1. Runtime Components

| Layer | File(s) | Purpose |
|-------|---------|---------|
| Transport | `whisperlivekit/basic_server.py`, any ASGI/WebSocket server | Accepts audio over WebSocket (MediaRecorder WebM or raw PCM chunks) and streams JSON updates back |
| Audio processing | `whisperlivekit/audio_processor.py` | Buffers audio, orchestrates transcription, diarization, translation, handles FFmpeg/PCM input |
| Engines | `whisperlivekit/core.py`, `whisperlivekit/simul_whisper/*`, `whisperlivekit/local_agreement/*` | Load models once (SimulStreaming or LocalAgreement), expose `TranscriptionEngine` and helpers |
| Frontends | `whisperlivekit/web/*`, `chrome-extension/*` | Optional UI layers feeding the WebSocket endpoint |

**Key idea:** The server boundary is just `AudioProcessor.process_audio()` for incoming bytes and the async generator returned by `AudioProcessor.create_tasks()` for outgoing updates (`FrontData`). Everything else is optional.

---

## 2. Running Without the Bundled Frontend

1. Start the server/engine however you like:
   ```bash
   wlk --model small --language en --host 0.0.0.0 --port 9000
   # or launch your own app that instantiates TranscriptionEngine(...)
   ```
2. Build your own client (browser, mobile, desktop) that:
   - Opens `ws(s)://<host>:<port>/asr`
   - Sends either MediaRecorder/Opus WebM blobs **or** raw PCM (`--pcm-input` on the server tells the client to use the AudioWorklet).
   - Consumes the JSON payload defined in `docs/API.md`.

---

## 3. Running Without FastAPI

`whisperlivekit/basic_server.py` is just an example. Any async framework works, as long as you:

1. Create a global `TranscriptionEngine` (expensive to initialize; reuse it).
2. Instantiate `AudioProcessor(transcription_engine=engine)` for each connection.
3. Call `create_tasks()` to get the async generator, `process_audio()` with incoming bytes, and ensure `cleanup()` runs when the client disconnects.


If you prefer to send compressed audio, instantiate `AudioProcessor(pcm_input=False)` and pipe encoded chunks through `FFmpegManager` transparently. Just ensure `ffmpeg` is available.