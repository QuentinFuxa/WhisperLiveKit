import asyncio
import websockets
import numpy as np
import librosa
from pathlib import Path
import argparse
import ffmpeg
import json

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

async def send_audio(websocket, audio_file, chunk_size_s=1.0, duration_min=10.0):
    audio_file = Path(audio_file)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file {audio_file} not found")

    # Load audio file
    audio, sr_in = librosa.load(audio_file)
    
    # Calculate chunk size
    chunk_size = int(sr_in * chunk_size_s)
    duration_frames = min(int(duration_min*60 *sr_in), len(audio))

    logger.info(f"Sending audio for {duration_min} minutes")

    # Send audio in chunks
    for i in range(0, duration_frames, chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) > 0:
            # Convert to 16-bit PCM
            chunk_int16 = (chunk * 32768).astype(np.int16)
            sr_out=48000
            # Create WebM stream using ffmpeg
            stream = (
                ffmpeg
                .input('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=sr_in)
                .output('pipe:1', 
                    format='webm', 
                    acodec='libopus', 
                    ac=1, 
                    ar=sr_out,
                    audio_bitrate='128k',  # Set explicit bitrate
                    loglevel='error'       # Only show errors
                )
                .run_async(pipe_stdin=True, pipe_stdout=True)
            )
            
            # Write chunk to ffmpeg
            stream.stdin.write(chunk_int16.tobytes())
            stream.stdin.close()
            
            # Read WebM output
            webm_data = stream.stdout.read()
            
            # Send WebM data
            await websocket.send(webm_data)
            await asyncio.sleep(chunk_size_s)

    logger.debug("Sending stop signal")
    await websocket.send(b"")

async def receive_updates(websocket):
    while True:
        try:
            response = await websocket.recv()

            # read json from response
            json_response = json.loads(response)

            print("-"*50)

            if "lines" in json_response and len(json_response['lines']) > 0:

                for line in json_response['lines']:
                    print("{beg} - {end} Speaker {speaker}:\n {text}".format(**line))
                

            for key, value in json_response.items():
                if key=="lines":
                    continue
                print(f"{key}: {value}")
            print("-"*50)

            
        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Connection closed normally")
            break
        except Exception as e:
            logger.error(f"Error receiving updates: {e}")
            break

async def test_server(audio_file, host="localhost", port=8000, duration_min=10.0):
    async with websockets.connect(f"ws://{host}:{port}/asr") as websocket:
        # Start receiving updates
        logger.info(f"Connected to {host}:{port}")
        asyncio.create_task(receive_updates(websocket))
        
        # Send audio data
        await send_audio(websocket, audio_file, duration_min=duration_min)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Whisper FastAPI Server")
    parser.add_argument(
        "--audio-file",
        type=str,
        required=True,
        help="Path to wav audio file to transcribe"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port number"
    )
    parser.add_argument(
        "--duration-min",
        type=float,
        default=10.0,
        help="Duration of audio to send in minutes"
    )
    
    args = parser.parse_args()
    
    asyncio.run(test_server(args.audio_file, args.host, args.port, args.duration_min))