# Python integration

Create a `TranscriptionEngine` once at startup, then an `AudioProcessor` per audio stream. `process_audio()` accepts incoming bytes; `create_tasks()` returns an async generator of `FrontData` objects. Serialize each update with `to_dict()`.

| Component | Responsibility |
|---|---|
| `TranscriptionEngine` | Shared ASR, VAD, diarization, and translation models |
| `AudioProcessor` | Per-session queues, buffers, processing tasks, and transcript state |
| `FrontData` | Output data; its `to_dict()` method defines the native JSON format |

## Minimal WebSocket server

Save this as `app.py` and run `uvicorn app:app`. This example accepts PCM s16le, mono, 16 kHz audio. An empty binary frame signals end of input; the connection stays open while the pipeline drains.

```python
import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from whisperlivekit import AudioProcessor, TranscriptionEngine


@asynccontextmanager
async def lifespan(app):
    app.state.engine = TranscriptionEngine(
        model_size="base", lan="en", pcm_input=True
    )
    yield


app = FastAPI(lifespan=lifespan)


async def send_results(websocket, results):
    async for response in results:
        await websocket.send_json(response.to_dict())
        if response.status == "error":
            await websocket.close(code=1011, reason="transcription failed")
            return
    await websocket.send_json({"type": "ready_to_stop"})


@app.websocket("/asr")
async def transcribe(websocket: WebSocket):
    await websocket.accept()
    processor = AudioProcessor(transcription_engine=app.state.engine)
    sender = None
    try:
        await websocket.send_json({"type": "config", "useAudioWorklet": True})
        results = await processor.create_tasks()
        sender = asyncio.create_task(send_results(websocket, results))
        while True:
            chunk = await websocket.receive_bytes()
            await processor.process_audio(chunk)
            if not chunk:
                await sender
                break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            if sender is not None:
                sender.cancel()
                with suppress(asyncio.CancelledError, WebSocketDisconnect):
                    await sender
        finally:
            await processor.cleanup()
```

For compressed audio, configure the engine with `pcm_input=False`, install FFmpeg, and send a continuous encoded audio stream. Do not change the format halfway through a session.

The bundled [server](../whisperlivekit/basic_server.py) adds authentication, per-session options, REST transcription, and Deepgram compatibility. This example only demonstrates the native pipeline lifecycle. For the message schema and session parameters, see [API.md](API.md).

## Other async frameworks

The same lifecycle works without FastAPI: create the processor, start consuming its result generator, feed bytes, send an empty chunk at EOF, drain results, and always call `cleanup()` in a `finally` block. Avoid changing a shared backend's language directly; pass session overrides to `AudioProcessor`.

`TranscriptionEngine` is a process-wide singleton. Construct it during startup before accepting sessions. `reset()` is intended for tests and backend comparisons, not for switching models while users are connected.
