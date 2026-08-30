import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import APIRouter, FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from whisperlivekit import AudioProcessor, TranscriptionEngine, get_inline_ui_html, parse_args
from whisperlivekit.api_auth import websocket_token
from whisperlivekit.config import WhisperLiveKitConfig, parse_cors_origins
from whisperlivekit.processing_queue import PipelineClosed, PipelineOverloaded
from whisperlivekit.timed_objects import FrontData
from whisperlivekit.timed_objects import format_subtitle_timestamp as _srt_timestamp

logger = logging.getLogger(__name__)

config = WhisperLiveKitConfig()
transcription_engine = None

_API_TOKEN = getattr(config, "api_token", None) or os.environ.get("WLK_API_TOKEN") or None


def _token_ok(candidate: Optional[str], connection=None) -> bool:
    """No token configured = open server; otherwise constant-time compare."""
    settings = getattr(getattr(getattr(connection, "app", None), "state", None), "config", None)
    expected = _API_TOKEN if settings is None else settings.api_token or os.environ.get("WLK_API_TOKEN") or None
    return expected is None or bool(candidate) and hmac.compare_digest(candidate, expected)


def _session_settings(connection):
    state = getattr(getattr(connection, "app", None), "state", None)
    return (
        getattr(state, "config", config),
        getattr(state, "transcription_engine", None) or transcription_engine,
    )


def _bearer_token(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.transcription_engine = TranscriptionEngine(config=app.state.config)
    yield


router = APIRouter()

@router.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())


@router.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    config, transcription_engine = _session_settings(request)
    backend = getattr(transcription_engine.config, "backend", "whisper") if transcription_engine else None
    return JSONResponse({
        "status": "ok",
        "backend": backend,
        "ready": transcription_engine is not None,
    })


async def handle_websocket_results(websocket, results_generator, diff_tracker=None):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            if diff_tracker is not None:
                await websocket.send_json(diff_tracker.to_message(response))
            else:
                await websocket.send_json(response.to_dict())
            if response.status == "error":
                await websocket.close(code=1011, reason="transcription failed")
                return
        # when the results_generator finishes it means all audio has been processed
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results (client likely closed connection).")
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@router.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    config, transcription_engine = _session_settings(websocket)

    # Authentication (when --api-token / WLK_API_TOKEN is set): accept the
    # token either as a query parameter or an Authorization: Bearer header.
    ws_token = websocket_token(websocket)
    if not _token_ok(ws_token, websocket):
        await websocket.close(code=4401, reason="invalid or missing API token")
        logger.warning("WebSocket rejected: invalid or missing API token")
        return

    # Read per-session options from query parameters
    session_language = websocket.query_params.get("language", None)
    mode = websocket.query_params.get("mode", "full")
    session_target_language = websocket.query_params.get("target_language", None)
    session_context = websocket.query_params.get("context", None)

    try:
        audio_processor = AudioProcessor(
            transcription_engine=transcription_engine,
            language=session_language,
            mode=mode,
            target_language=session_target_language,
            context=session_context,
        )
    except ValueError as e:
        # Bad per-session parameters (e.g. a language the backend does not
        # support): tell the client instead of failing before the handshake.
        await websocket.accept()
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        finally:
            await websocket.close(code=4400, reason="invalid session parameters")
        logger.warning("WebSocket rejected: %s", e)
        return
    websocket_task = None
    try:
        await websocket.accept()
        logger.info(
            "WebSocket connection opened.%s%s%s",
            f" language={session_language}" if session_language else "",
            f" target_language={session_target_language}" if session_target_language else "",
            f" context_chars={len(session_context)}" if session_context else "",
        )
        diff_tracker = None
        if mode == "diff":
            from whisperlivekit.diff_protocol import DiffTracker
            diff_tracker = DiffTracker()
            logger.info("Client requested diff mode")

        from whisperlivekit.session_asr_proxy import session_context_capability

        await websocket.send_json({
            "type": "config",
            "useAudioWorklet": bool(config.pcm_input),
            "mode": mode,
            "context": session_context_capability(
                transcription_engine.args,
                transcription_engine.asr,
            ),
        })
        results_generator = await audio_processor.create_tasks()
        websocket_task = asyncio.create_task(
            handle_websocket_results(websocket, results_generator, diff_tracker)
        )
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except KeyError as e:
        if 'bytes' in str(e):
            logger.warning("Client has closed the connection.")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "error": getattr(audio_processor, "overload_error", None) or "Unable to process audio.",
            })
            await websocket.close(code=1011, reason="transcription failed")
        except Exception:
            logger.debug("WebSocket was already disconnected", exc_info=True)
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if websocket_task is not None:
            if not websocket_task.done():
                websocket_task.cancel()
            await asyncio.gather(websocket_task, return_exceptions=True)

        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")


# ---------------------------------------------------------------------------
# Deepgram-compatible WebSocket API  (/v1/listen)
# ---------------------------------------------------------------------------

@router.websocket("/v1/listen")
async def deepgram_websocket_endpoint(websocket: WebSocket):
    """Deepgram-compatible live transcription WebSocket."""
    config, transcription_engine = _session_settings(websocket)
    if not _token_ok(websocket_token(websocket), websocket):
        await websocket.close(code=4401, reason="invalid or missing API token")
        logger.warning("Deepgram WebSocket rejected: invalid or missing API token")
        return
    from whisperlivekit.deepgram_compat import handle_deepgram_websocket
    await handle_deepgram_websocket(websocket, transcription_engine, config)


# ---------------------------------------------------------------------------
# OpenAI-compatible REST API  (/v1/audio/transcriptions)
# ---------------------------------------------------------------------------

_DIARIZED_JSON_REQUIRES_DIARIZATION = (
    "response_format=diarized_json requires diarization to be enabled on the "
    "server. Start the server with --diarization."
)
_OPENAI_RESPONSE_FORMATS = frozenset({
    "json",
    "verbose_json",
    "diarized_json",
    "text",
    "srt",
    "vtt",
})

_MAX_AUDIO_BYTES = 512 * 1024 * 1024


async def _convert_to_pcm(audio_bytes: bytes) -> bytes:
    """Convert audio with bounded output and a reaped process on every exit."""
    from fastapi import HTTPException

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", "pipe:0",
        "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-loglevel", "error", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def feed():
        try:
            for offset in range(0, len(audio_bytes), 65536):
                proc.stdin.write(audio_bytes[offset:offset + 65536])
                await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass  # ffmpeg's exit code and stderr explain an invalid input
        finally:
            proc.stdin.close()

    async def read_errors():
        tail = b""
        while chunk := await proc.stderr.read(65536):
            tail = (tail + chunk)[-16384:]
        return tail.decode(errors="replace").strip()

    feeder = asyncio.create_task(feed())
    errors = asyncio.create_task(read_errors())
    try:
        pcm = bytearray()
        while chunk := await proc.stdout.read(65536):
            if len(pcm) + len(chunk) > _MAX_AUDIO_BYTES:
                raise HTTPException(status_code=413, detail="Decoded audio exceeds the 512 MB limit")
            pcm.extend(chunk)
        await feeder
        detail = await errors
        await proc.wait()
        if proc.returncode != 0:
            raise HTTPException(status_code=400, detail=f"Audio conversion failed: {detail}")
        return bytes(pcm)
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
        for task in (feeder, errors):
            if not task.done():
                task.cancel()
        await asyncio.gather(feeder, errors, return_exceptions=True)


def _parse_time_str(time_str: str) -> float:
    """Parse 'H:MM:SS.cc' to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def _speaker_label_from_index(index: int) -> str:
    """Return A, B, ..., Z, AA, AB, ... for a zero-based speaker index."""
    label = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def _speaker_label(speaker, speaker_labels: dict) -> str:
    """Map internal speaker IDs to OpenAI-style speaker labels."""
    # Diarized WLK lines use 1-based numeric speaker IDs, while OpenAI's
    # diarized response examples use stable alphabetic labels.
    if isinstance(speaker, int) and speaker > 0:
        return _speaker_label_from_index(speaker - 1)
    if speaker not in speaker_labels:
        # Non-numeric speaker IDs can come from other diarization backends, so
        # keep a deterministic first-seen mapping for those labels.
        speaker_labels[speaker] = _speaker_label_from_index(len(speaker_labels))
    return speaker_labels[speaker]


def _duration_usage(duration: float) -> dict:
    return {
        "type": "duration",
        "seconds": round(duration),
    }


def _format_openai_response(front_data, response_format: str, language: Optional[str], duration: float, diarization_enabled: bool = True) -> dict:
    """Convert FrontData to OpenAI-compatible response."""
    d = front_data.to_dict()
    lines = d.get("lines", [])

    # Combine all speech text (exclude silence segments)
    text_parts = [l["text"] for l in lines if l.get("text") and l.get("speaker", 0) != -2]
    full_text = " ".join(text_parts).strip()

    if response_format == "text":
        return full_text

    speaker_labels = {}

    if response_format == "diarized_json":
        if not diarization_enabled:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail=_DIARIZED_JSON_REQUIRES_DIARIZATION,
            )
        segments = []
        text_parts = []
        for line in lines:
            # WLK represents silence as speaker -2; diarized_json only emits
            # spoken transcript segments with actual text.
            if line.get("speaker") == -2 or not line.get("text"):
                continue
            speaker = _speaker_label(line.get("speaker", 1), speaker_labels)
            start = _parse_time_str(line.get("start", "0:00:00"))
            end = _parse_time_str(line.get("end", "0:00:00"))
            text = line["text"]
            segments.append({
                "type": "transcript.text.segment",
                "id": f"seg_{len(segments) + 1:03d}",
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text,
                "speaker": speaker,
            })
            # Match the diarized_json top-level transcript style by preserving
            # speaker turns in the combined text instead of flattening them.
            text_parts.append(f"{speaker}: {text}")

        return {
            "task": "transcribe",
            "duration": round(duration, 2),
            "text": "\n".join(text_parts).strip(),
            "segments": segments,
            "usage": _duration_usage(duration),
        }

    # Build segments and words for verbose_json
    segments = []
    words = []

    # Prefer real Segment objects (carrying ASRToken timestamps) when available;
    # fall back to the serialized dict form (e.g. test mocks without .lines).
    real_segments = getattr(front_data, "lines", None)
    if real_segments is not None:
        seg_iter = [
            (s.start, s.end, s.text, getattr(s, "tokens", None))
            for s in real_segments
            if s.speaker != -2 and s.text
        ]
    else:
        seg_iter = [
            (
                _parse_time_str(line.get("start", "0:00:00")),
                _parse_time_str(line.get("end", "0:00:00")),
                line["text"],
                None,
            )
            for line in lines
            if line.get("speaker") != -2 and line.get("text")
        ]

    for start, end, text, real_tokens in seg_iter:
        segments.append({
            "id": len(segments),
            "start": round(start, 2),
            "end": round(end, 2),
            "text": text,
        })
        if real_tokens:
            for tok in real_tokens:
                if tok.text and tok.text.strip():
                    words.append({
                        "word": tok.text.strip(),
                        "start": round(tok.start, 2),
                        "end": round(tok.end, 2),
                    })
        else:
            # Fallback: interpolate word timestamps from segment boundaries
            seg_words = text.split()
            if seg_words:
                word_duration = (end - start) / max(len(seg_words), 1)
                for j, word in enumerate(seg_words):
                    words.append({
                        "word": word,
                        "start": round(start + j * word_duration, 2),
                        "end": round(start + (j + 1) * word_duration, 2),
                    })

    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": language or "unknown",
            "duration": round(duration, 2),
            "text": full_text,
            "words": words,
            "segments": segments,
            "usage": _duration_usage(duration),
        }

    if response_format in ("srt", "vtt"):
        lines_out = []
        if response_format == "vtt":
            lines_out.append("WEBVTT\n")
        for i, seg in enumerate(segments):
            start_ts = _srt_timestamp(seg["start"], response_format)
            end_ts = _srt_timestamp(seg["end"], response_format)
            if response_format == "srt":
                lines_out.append(f"{i + 1}")
            lines_out.append(f"{start_ts} --> {end_ts}")
            lines_out.append(seg["text"])
            lines_out.append("")
        return "\n".join(lines_out)

    # Default: json
    return {
        "text": full_text,
        "usage": _duration_usage(duration),
    }


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: Optional[str] = Form(default=None),
    prompt: str = Form(default=""),
    response_format: str = Form(default="json"),
    timestamp_granularities: Optional[List[str]] = Form(default=None),
):
    """OpenAI-compatible audio transcription endpoint.

    Implements the compatibility-oriented subset documented in docs/API.md.
    The `model` parameter is accepted but ignored (uses the server's configured
    backend). `prompt` supplies decoder context for backends that expose text
    conditioning and returns HTTP 400 for backends that do not.
    """
    config, transcription_engine = _session_settings(request)
    from fastapi import HTTPException

    if not _token_ok(_bearer_token(request), request):
        raise HTTPException(status_code=401, detail="invalid or missing API token")

    if response_format not in _OPENAI_RESPONSE_FORMATS:
        allowed = ", ".join(sorted(_OPENAI_RESPONSE_FORMATS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format={response_format!r}. Expected one of: {allowed}.",
        )

    diarization_enabled = bool(getattr(config, "diarization", False))
    if response_format == "diarized_json" and not diarization_enabled:
        raise HTTPException(
            status_code=400,
            detail=_DIARIZED_JSON_REQUIRES_DIARIZATION,
        )

    # Validate decoder context before reading or converting a potentially large
    # upload. AudioProcessor validates again at the session factory boundary.
    from whisperlivekit.session_asr_proxy import validate_session_context

    try:
        prompt = validate_session_context(
            getattr(transcription_engine, "args", config),
            getattr(transcription_engine, "asr", None),
            prompt,
        ) or ""
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    max_upload_mb = 512
    audio_bytes = await file.read(_MAX_AUDIO_BYTES + 1)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(audio_bytes) > _MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file exceeds the {max_upload_mb} MB upload limit",
        )

    # Convert to PCM for pipeline processing
    configured = float(getattr(config, "rest_timeout", 0) or 0)
    try:
        async with asyncio.timeout(configured if configured > 0 else 120):
            pcm_data = await _convert_to_pcm(audio_bytes)
    except TimeoutError:
        raise HTTPException(status_code=408, detail="Audio conversion timed out")
    del audio_bytes
    duration = len(pcm_data) / (16000 * 2)  # 16kHz, 16-bit

    # Process through the full pipeline
    try:
        processor = AudioProcessor(
            transcription_engine=transcription_engine,
            language=language,
            context=prompt,
            pcm_input=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    final_result = None
    collect_task = None

    async def collect():
        nonlocal final_result
        async for result in results_gen:
            if result.status == "error":
                raise HTTPException(
                    status_code=503 if getattr(processor, "overload_error", None) else 500,
                    detail=result.error or "Transcription failed",
                )
            final_result = result

    # The feed/drain budget scales with the audio length
    # (issue #374: a fixed 120 s silently truncated long files) and can be
    # overridden with --rest-timeout.
    configured = float(getattr(config, "rest_timeout", 0) or 0)
    timeout_sec = configured if configured > 0 else max(120.0, duration * 2.5)
    timed_out = False
    try:
        async with asyncio.timeout(timeout_sec):
            results_gen = await processor.create_tasks()
            collect_task = asyncio.create_task(collect())
            chunk_size = 16000 * 2
            for i in range(0, len(pcm_data), chunk_size):
                await processor.process_audio(pcm_data[i:i + chunk_size])
                # Uncontended queue writes may not suspend; let consumers and
                # cancellation run while feeding a large in-memory file.
                await asyncio.sleep(0)
                if collect_task.done():
                    await collect_task  # propagate errors while feeding
            await processor.process_audio(b"")
            await collect_task
    except (PipelineClosed, PipelineOverloaded) as exc:
        message = getattr(processor, "overload_error", None)
        raise HTTPException(status_code=503 if message else 500, detail=message or str(exc)) from exc
    except asyncio.TimeoutError:
        timed_out = True
        logger.warning(
            "Transcription timed out after %.0fs (audio duration %.0fs)",
            timeout_sec,
            duration,
        )
    finally:
        if collect_task is not None:
            if not collect_task.done():
                collect_task.cancel()
            await asyncio.gather(collect_task, return_exceptions=True)
        await processor.cleanup()

    if timed_out:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=408,
            detail=(
                f"Transcription did not finish within {timeout_sec:.0f}s "
                f"for {duration:.0f}s of audio. Retry with a longer "
                "--rest-timeout or faster hardware/backend."
            ),
        )

    if final_result is None:
        final_result = FrontData()

    result = _format_openai_response(
        final_result,
        response_format,
        language,
        duration,
        diarization_enabled=diarization_enabled,
    )

    if isinstance(result, str):
        return PlainTextResponse(result)
    return JSONResponse(result)


@router.get("/v1/models")
async def list_models(request: Request):
    """OpenAI-compatible model listing endpoint."""
    config, transcription_engine = _session_settings(request)
    backend = getattr(transcription_engine.config, "backend", "whisper") if transcription_engine else "whisper"
    model_size = getattr(transcription_engine.config, "model_size", "base") if transcription_engine else "base"
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": f"{backend}/{model_size}" if backend != "whisper" else f"whisper-{model_size}",
            "object": "model",
            "owned_by": "whisperlivekit",
        }],
    })


def create_app(server_config: WhisperLiveKitConfig | None = None) -> FastAPI:
    """Build an ASGI application without reading the host process's arguments."""
    application = FastAPI(lifespan=lifespan)
    application.state.config = server_config if server_config is not None else WhisperLiveKitConfig()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=parse_cors_origins(application.state.config.cors_origins),
        allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app(config)


def main():
    """Entry point for the CLI command."""
    import uvicorn

    from whisperlivekit.cli import print_banner

    settings = parse_args()
    logging.basicConfig(level=settings.log_level)
    config = settings
    ssl = bool(config.ssl_certfile and config.ssl_keyfile)
    print_banner(config, config.host, config.port, ssl=ssl)

    uvicorn_kwargs = {
        "app": create_app(config),
        "host": config.host,
        "port": config.port,
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }

    ssl_kwargs = {}
    if config.ssl_certfile or config.ssl_keyfile:
        if not (config.ssl_certfile and config.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": config.ssl_certfile,
            "ssl_keyfile": config.ssl_keyfile,
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}
    if config.forwarded_allow_ips:
        uvicorn_kwargs = {**uvicorn_kwargs, "forwarded_allow_ips": config.forwarded_allow_ips}

    uvicorn.run(**uvicorn_kwargs)

if __name__ == "__main__":
    main()
