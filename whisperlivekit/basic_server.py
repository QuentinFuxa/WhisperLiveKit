import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from whisperlivekit import (
    AudioProcessor,
    TranscriptionEngine,
    get_inline_ui_html,
    parse_args,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

args = parse_args()
transcription_engine = None


# NEW: Manage the Names of Speaker (SpeakerNameManager)
class SpeakerNameManager:
    """Manages mapping between numeric speaker IDs and custom names."""

    def __init__(self):
        self._names: Dict[int, str] = {}

    def set_name(self, speaker_id: int, name: str) -> None:
        """Assign a custom name to a speaker ID."""
        self._names[speaker_id] = name

    def get_name(self, speaker_id: int) -> str:
        """Get the display name for a speaker (custom name or default number)."""
        return self._names.get(speaker_id, str(speaker_id))

    def remove_name(self, speaker_id: int) -> None:
        """Remove custom name, revert to numeric display."""
        self._names.pop(speaker_id, None)

    def get_all_mappings(self) -> Dict[int, str]:
        """Return all current speaker name mappings."""
        return self._names.copy()

    def clear(self) -> None:
        """Clear all custom speaker names."""
        self._names.clear()


# Global speaker name manager instance
speaker_names = SpeakerNameManager()


# Pydantic models for API requests
class SpeakerNameUpdate(BaseModel):
    speaker_id: int
    name: str


class SpeakerNameDelete(BaseModel):
    speaker_id: int


# ===================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
    transcription_engine = TranscriptionEngine(
        **vars(args),
    )
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())


# NEW: Add Speaker Name API Endpoint
@app.get("/api/speakers")
async def get_speaker_names():
    """Get all speaker name mappings."""
    return {"speakers": speaker_names.get_all_mappings()}


@app.post("/api/speakers")
async def set_speaker_name(data: SpeakerNameUpdate):
    """Set or update a speaker's custom name."""
    speaker_names.set_name(data.speaker_id, data.name)
    return {"success": True, "speaker_id": data.speaker_id, "name": data.name}


@app.delete("/api/speakers/{speaker_id}")
async def delete_speaker_name(speaker_id: int):
    """Remove a speaker's custom name (revert to numeric)."""
    speaker_names.remove_name(speaker_id)
    return {"success": True, "speaker_id": speaker_id}


@app.delete("/api/speakers")
async def clear_all_speaker_names():
    """Clear all custom speaker names."""
    speaker_names.clear()
    return {"success": True}


# =========================================================


async def handle_websocket_results(websocket, results_generator):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            # Inject speaker names into the response
            response_dict = response.to_dict()
            for line in response_dict.get("lines", []):
                speaker_id = line.get("speaker")
                if speaker_id and speaker_id > 0:
                    line["speaker_name"] = speaker_names.get_name(speaker_id)
            await websocket.send_json(response_dict)
        logger.info("Results generator finished.  Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected while handling results (client likely closed connection)."
        )
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_engine
    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
    )
    await websocket.accept()
    logger.info("WebSocket connection opened.")

    try:
        await websocket.send_json(
            {"type": "config", "useAudioWorklet": bool(args.pcm_input)}
        )
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")

    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(
        handle_websocket_results(websocket, results_generator)
    )

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except KeyError as e:
        if "bytes" in str(e):
            logger.warning(f"Client has closed the connection.")
        else:
            logger.error(
                f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True
            )
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(
            f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True
        )
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")

        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")


def main():
    """Entry point for the CLI command."""
    import uvicorn

    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host": args.host,
        "port": args.port,
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }

    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError(
                "Both --ssl-certfile and --ssl-keyfile must be specified together."
            )
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile,
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}
    if args.forwarded_allow_ips:
        uvicorn_kwargs = {
            **uvicorn_kwargs,
            "forwarded_allow_ips": args.forwarded_allow_ips,
        }

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
