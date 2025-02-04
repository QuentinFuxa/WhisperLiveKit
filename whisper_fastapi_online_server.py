import io
import argparse
import asyncio
import numpy as np
import ffmpeg
from time import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from src.whisper_streaming.whisper_online import backend_factory, online_factory, add_shared_args
from pathlib import Path

import logging
import logging.config

def setup_logging():

    log_file= Path("logs/logfile.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.unlink(missing_ok=True)
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s %(levelname)s [%(name)s]: %(message)s',
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'formatter': 'standard',
                'filename': str(log_file),
            },
        },
        'root': {
            'handlers': ['console'],
            'level': 'INFO',
        },
        'loggers': {
            'uvicorn': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': False,
            },
            'uvicorn.error': {
                'level': 'INFO',  
            },
            'uvicorn.access': {
                'level': 'INFO',
            },
            'src.whisper_streaming': { 
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': False,
            },
            'src.diarization': {  
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': False,
            },
        },
    }

    logging.config.dictConfig(logging_config)

    logger = logging.getLogger(__name__)
    logger.info("I will log to {log_file} and to the console")
    

logger= setup_logging()







app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


parser = argparse.ArgumentParser(description="Whisper FastAPI Online Server")
parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="The host address to bind the server to.",
)
parser.add_argument(
    "--port", type=int, default=8000, help="The port number to bind the server to."
)
parser.add_argument(
    "--warmup-file",
    type=str,
    dest="warmup_file",
    help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .",
)

parser.add_argument(
    "--diarization",
    type=bool,
    default=False,
    help="Whether to enable speaker diarization.",
)


add_shared_args(parser)
args = parser.parse_args()

asr, tokenizer = backend_factory(args)

if args.diarization:
    from src.diarization.diarization_online import DiartDiarization


# Load demo HTML for the root endpoint
with open("src/web/live_transcription.html", "r", encoding="utf-8") as f:
    html = f.read()


@app.get("/")
async def get():
    return HTMLResponse(html)


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLES_PER_SEC = SAMPLE_RATE * int(args.min_chunk_size)
BYTES_PER_SAMPLE = 2  # s16le = 2 bytes per sample
BYTES_PER_SEC = SAMPLES_PER_SEC * BYTES_PER_SAMPLE


async def start_ffmpeg_decoder():
    """
    Start an FFmpeg process in async streaming mode that reads WebM from stdin
    and outputs raw s16le PCM on stdout. Returns the process object.
    """
    process = (
        ffmpeg.input("pipe:0", format="webm")
        .output(
            "pipe:1",
            format="s16le",
            acodec="pcm_s16le",
            ac=CHANNELS,
            ar=str(SAMPLE_RATE),
        )
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    return process



@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection opened.")

    ffmpeg_process = await start_ffmpeg_decoder()
    pcm_buffer = bytearray()
    print("Loading online.")
    online = online_factory(args, asr, tokenizer)
    print("Online loaded.")

    if args.diarization:
        diarization = DiartDiarization(SAMPLE_RATE)

    # Continuously read decoded PCM from ffmpeg stdout in a background task
    async def ffmpeg_stdout_reader():
        nonlocal pcm_buffer
        loop = asyncio.get_event_loop()
        full_transcription = ""
        beg = time()
        
        chunk_history = []  # Will store dicts: {beg, end, text, speaker}
        
        while True:
            try:
                elapsed_time = int(time() - beg)
                beg = time()
                chunk = await loop.run_in_executor(
                    None, ffmpeg_process.stdout.read, 32000 * elapsed_time
                )
                if (
                    not chunk
                ):  # The first chunk will be almost empty, FFmpeg is still starting up
                    chunk = await loop.run_in_executor(
                        None, ffmpeg_process.stdout.read, 4096
                    )
                    if not chunk:  # FFmpeg might have closed
                        print("FFmpeg stdout closed.")
                        break

                pcm_buffer.extend(chunk)

                if len(pcm_buffer) >= BYTES_PER_SEC:
                    # Convert int16 -> float32
                    pcm_array = (
                        np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    pcm_buffer = bytearray()
                    online.insert_audio_chunk(pcm_array)
                    beg_trans, end_trans, trans = online.process_iter()
                    
                    if trans:
                        chunk_history.append({
                        "beg": beg_trans,
                        "end": end_trans,
                        "text": trans,
                        "speaker": "0"
                        })
                    
                    full_transcription += trans
                    if args.vac:
                        buffer = online.online.concatenate_tsw(
                            online.online.transcript_buffer.buffer
                        )[
                            2
                        ]  # We need to access the underlying online object to get the buffer
                    else:
                        buffer = online.concatenate_tsw(online.transcript_buffer.buffer)[2]
                    if (
                        buffer in full_transcription
                    ):  # With VAC, the buffer is not updated until the next chunk is processed
                        buffer = ""
                                        
                    lines = [
                        {
                            "speaker": "0",
                            "text": "",
                        }
                    ]
                    
                    if args.diarization:
                        await diarization.diarize(pcm_array)
                        diarization.assign_speakers_to_chunks(chunk_history)

                    for ch in chunk_history:
                        if args.diarization and ch["speaker"] and ch["speaker"][-1] != lines[-1]["speaker"]:
                            lines.append(
                                {
                                    "speaker": ch["speaker"][-1],
                                    "text": ch['text'],
                                }
                            )
                        else:
                            lines[-1]["text"] += ch['text']

                    response = {"lines": lines, "buffer": buffer}
                    await websocket.send_json(response)
                    
            except Exception as e:
                print(f"Exception in ffmpeg_stdout_reader: {e}")
                break

        print("Exiting ffmpeg_stdout_reader...")

    stdout_reader_task = asyncio.create_task(ffmpeg_stdout_reader())

    try:
        while True:
            # Receive incoming WebM audio chunks from the client
            message = await websocket.receive_bytes()
            # Pass them to ffmpeg via stdin
            ffmpeg_process.stdin.write(message)
            ffmpeg_process.stdin.flush()

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
    except Exception as e:
        print(f"Error in websocket loop: {e}")
    finally:
        # Clean up ffmpeg and the reader task
        try:
            ffmpeg_process.stdin.close()
        except:
            pass
        stdout_reader_task.cancel()

        try:
            ffmpeg_process.stdout.close()
        except:
            pass

        ffmpeg_process.wait()
        del online
        
        if args.diarization:
            # Stop Diart
            diarization.close()




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "whisper_fastapi_online_server:app", host=args.host, port=args.port, reload=True,
        log_level="info"
    )
