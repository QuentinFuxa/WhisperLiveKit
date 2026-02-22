"""Headless test client for WhisperLiveKit.

Feeds audio files to the transcription pipeline via WebSocket
and collects results — no browser or microphone needed.

Usage:
    # Against a running server (server must be started with --pcm-input):
    python -m whisperlivekit.test_client audio.wav

    # Custom server URL and speed:
    python -m whisperlivekit.test_client audio.wav --url ws://localhost:9090/asr --speed 0

    # Output raw JSON responses:
    python -m whisperlivekit.test_client audio.wav --json

    # Programmatic usage:
    from whisperlivekit.test_client import transcribe_audio
    result = asyncio.run(transcribe_audio("audio.wav"))
    print(result.text)
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # s16le


@dataclass
class TranscriptionResult:
    """Collected transcription results from a session."""

    responses: List[dict] = field(default_factory=list)
    audio_duration: float = 0.0

    @property
    def text(self) -> str:
        """Full transcription text from the last response (committed lines + buffer)."""
        if not self.responses:
            return ""
        for resp in reversed(self.responses):
            lines = resp.get("lines", [])
            buffer = resp.get("buffer_transcription", "")
            if lines or buffer:
                parts = [line["text"] for line in lines if line.get("text")]
                if buffer:
                    parts.append(buffer)
                return " ".join(parts)
        return ""

    @property
    def committed_text(self) -> str:
        """Only the committed (finalized) transcription lines, no buffer."""
        if not self.responses:
            return ""
        for resp in reversed(self.responses):
            lines = resp.get("lines", [])
            if lines:
                return " ".join(line["text"] for line in lines if line.get("text"))
        return ""

    @property
    def lines(self) -> List[dict]:
        """Committed lines from the last response."""
        for resp in reversed(self.responses):
            if resp.get("lines"):
                return resp["lines"]
        return []

    @property
    def n_updates(self) -> int:
        """Number of non-empty updates received."""
        return sum(
            1 for r in self.responses
            if r.get("lines") or r.get("buffer_transcription")
        )


def reconstruct_state(msg: dict, lines: List[dict]) -> dict:
    """Reconstruct full state from a diff or snapshot message.

    Mutates ``lines`` in-place (prune front, append new) and returns
    a full-state dict compatible with TranscriptionResult.
    """
    if msg.get("type") == "snapshot":
        lines.clear()
        lines.extend(msg.get("lines", []))
        return msg

    # Apply diff
    n_pruned = msg.get("lines_pruned", 0)
    if n_pruned > 0:
        del lines[:n_pruned]
    new_lines = msg.get("new_lines", [])
    lines.extend(new_lines)

    return {
        "status": msg.get("status", ""),
        "lines": lines[:],  # snapshot copy
        "buffer_transcription": msg.get("buffer_transcription", ""),
        "buffer_diarization": msg.get("buffer_diarization", ""),
        "buffer_translation": msg.get("buffer_translation", ""),
        "remaining_time_transcription": msg.get("remaining_time_transcription", 0),
        "remaining_time_diarization": msg.get("remaining_time_diarization", 0),
    }


def load_audio_pcm(audio_path: str, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Load an audio file and convert to PCM s16le mono via ffmpeg.

    Supports any format ffmpeg can decode (wav, mp3, flac, ogg, m4a, ...).
    """
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        "-loglevel", "error",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr.decode().strip()}")
    if not proc.stdout:
        raise RuntimeError(f"ffmpeg produced no output for {audio_path}")
    return proc.stdout


async def transcribe_audio(
    audio_path: str,
    url: str = "ws://localhost:8000/asr",
    chunk_duration: float = 0.5,
    speed: float = 1.0,
    timeout: float = 60.0,
    on_response: Optional[callable] = None,
    mode: str = "full",
) -> TranscriptionResult:
    """Feed an audio file to a running WhisperLiveKit server and collect results.

    Args:
        audio_path: Path to an audio file (any format ffmpeg supports).
        url: WebSocket URL of the /asr endpoint.
        chunk_duration: Duration of each audio chunk sent (seconds).
        speed: Playback speed multiplier (1.0 = real-time, 0 = as fast as possible).
        timeout: Max seconds to wait for the server after audio finishes.
        on_response: Optional callback invoked with each response dict as it arrives.
        mode: Output mode — "full" (default) or "diff" for incremental updates.

    Returns:
        TranscriptionResult with collected responses and convenience accessors.
    """
    import websockets

    result = TranscriptionResult()

    # Convert audio to PCM for both modes (we need duration either way)
    pcm_data = load_audio_pcm(audio_path)
    result.audio_duration = len(pcm_data) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    logger.info("Loaded %s: %.1fs of audio", audio_path, result.audio_duration)

    chunk_bytes = int(chunk_duration * SAMPLE_RATE * BYTES_PER_SAMPLE)

    # Append mode query parameter if using diff mode
    connect_url = url
    if mode == "diff":
        sep = "&" if "?" in url else "?"
        connect_url = f"{url}{sep}mode=diff"

    async with websockets.connect(connect_url) as ws:
        # Server sends config on connect
        config_raw = await ws.recv()
        config_msg = json.loads(config_raw)
        is_pcm = config_msg.get("useAudioWorklet", False)
        logger.info("Server config: %s", config_msg)

        if not is_pcm:
            logger.warning(
                "Server is not in PCM mode. Start the server with --pcm-input "
                "for the test client. Attempting raw file streaming instead."
            )

        done_event = asyncio.Event()
        diff_lines: List[dict] = []  # running state for diff mode reconstruction

        async def send_audio():
            if is_pcm:
                offset = 0
                n_chunks = 0
                while offset < len(pcm_data):
                    end = min(offset + chunk_bytes, len(pcm_data))
                    await ws.send(pcm_data[offset:end])
                    offset = end
                    n_chunks += 1
                    if speed > 0:
                        await asyncio.sleep(chunk_duration / speed)
                logger.info("Sent %d PCM chunks (%.1fs)", n_chunks, result.audio_duration)
            else:
                # Non-PCM: send raw file bytes for server-side ffmpeg decoding
                file_bytes = Path(audio_path).read_bytes()
                raw_chunk_size = 32000
                offset = 0
                while offset < len(file_bytes):
                    end = min(offset + raw_chunk_size, len(file_bytes))
                    await ws.send(file_bytes[offset:end])
                    offset = end
                    if speed > 0:
                        await asyncio.sleep(0.5 / speed)
                logger.info("Sent %d bytes of raw audio", len(file_bytes))

            # Signal end of audio
            await ws.send(b"")
            logger.info("End-of-audio signal sent")

        async def receive_results():
            try:
                async for raw_msg in ws:
                    data = json.loads(raw_msg)
                    if data.get("type") == "ready_to_stop":
                        logger.info("Server signaled ready_to_stop")
                        done_event.set()
                        return
                    # In diff mode, reconstruct full state for uniform API
                    if mode == "diff" and data.get("type") in ("snapshot", "diff"):
                        data = reconstruct_state(data, diff_lines)
                    result.responses.append(data)
                    if on_response:
                        on_response(data)
            except Exception as e:
                logger.debug("Receiver ended: %s", e)
            done_event.set()

        send_task = asyncio.create_task(send_audio())
        recv_task = asyncio.create_task(receive_results())

        # Total wait = time to send + time for server to process + timeout margin
        send_time = result.audio_duration / speed if speed > 0 else 1.0
        total_timeout = send_time + timeout

        try:
            await asyncio.wait_for(
                asyncio.gather(send_task, recv_task),
                timeout=total_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Timed out after %.0fs", total_timeout)
            send_task.cancel()
            recv_task.cancel()
            try:
                await asyncio.gather(send_task, recv_task, return_exceptions=True)
            except Exception:
                pass

    logger.info(
        "Session complete: %d responses, %d updates",
        len(result.responses), result.n_updates,
    )
    return result


def _print_result(result: TranscriptionResult, output_json: bool = False) -> None:
    """Print transcription results to stdout."""
    if output_json:
        for resp in result.responses:
            print(json.dumps(resp))
        return

    if result.lines:
        for line in result.lines:
            speaker = line.get("speaker", "")
            text = line.get("text", "")
            start = line.get("start", "")
            end = line.get("end", "")
            prefix = f"[{start} -> {end}]"
            if speaker and speaker != 1:
                prefix += f" Speaker {speaker}"
            print(f"{prefix} {text}")

    buffer = ""
    if result.responses:
        buffer = result.responses[-1].get("buffer_transcription", "")
    if buffer:
        print(f"[buffer] {buffer}")

    if not result.lines and not buffer:
        print("(no transcription received)")

    print(
        f"\n--- {len(result.responses)} responses | "
        f"{result.n_updates} updates | "
        f"{result.audio_duration:.1f}s audio ---"
    )


def main():
    parser = argparse.ArgumentParser(
        prog="whisperlivekit-test-client",
        description=(
            "Headless test client for WhisperLiveKit. "
            "Feeds audio files via WebSocket and prints the transcription."
        ),
    )
    parser.add_argument("audio", help="Path to audio file (wav, mp3, flac, ...)")
    parser.add_argument(
        "--url", default="ws://localhost:8000/asr",
        help="WebSocket endpoint URL (default: ws://localhost:8000/asr)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (1.0 = real-time, 0 = fastest, default: 1.0)",
    )
    parser.add_argument(
        "--chunk-duration", type=float, default=0.5,
        help="Chunk duration in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Max seconds to wait for server after audio ends (default: 60)",
    )
    parser.add_argument(
        "--language", "-l", default=None,
        help="Override transcription language for this session (e.g. en, fr, auto)",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON responses")
    parser.add_argument(
        "--diff", action="store_true",
        help="Use diff protocol (only receive incremental changes from server)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Print transcription updates as they arrive",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    live_callback = None
    if args.live:
        def live_callback(data):
            lines = data.get("lines", [])
            buf = data.get("buffer_transcription", "")
            parts = [l["text"] for l in lines if l.get("text")]
            if buf:
                parts.append(f"[{buf}]")
            if parts:
                print("\r" + " ".join(parts), end="", flush=True)

    # Build URL with query parameters for language and mode
    url = args.url
    params = []
    if args.language:
        params.append(f"language={args.language}")
    if args.diff:
        params.append("mode=diff")
    if params:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{'&'.join(params)}"

    result = asyncio.run(transcribe_audio(
        audio_path=str(audio_path),
        url=url,
        chunk_duration=args.chunk_duration,
        speed=args.speed,
        timeout=args.timeout,
        on_response=live_callback,
        mode="diff" if args.diff else "full",
    ))

    if args.live:
        print()  # newline after live output

    _print_result(result, output_json=args.json)


if __name__ == "__main__":
    main()
