import asyncio
import logging
from enum import Enum
from typing import Optional, Callable
import contextlib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ERROR_INSTALL_INSTRUCTIONS = f"""
{'='*50}
FFmpeg is not installed or not found in your system's PATH.
Alternative Solution: You can still use WhisperLiveKit without FFmpeg by adding the --pcm-input parameter. Note that when using this option, audio will not be compressed between the frontend and backend, which may result in higher bandwidth usage.

If you want to install FFmpeg:

# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

# macOS (using Homebrew):
brew install ffmpeg

# Windows:
# 1. Download the latest static build from https://ffmpeg.org/download.html
# 2. Extract the archive (e.g., to C:\\FFmpeg).
# 3. Add the 'bin' directory (e.g., C:\\FFmpeg\\bin) to your system's PATH environment variable.

After installation, please restart the application.
{'='*50}
"""

class FFmpegState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    RESTARTING = "restarting"
    FAILED = "failed"

class FFmpegManager:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels

        self.process: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None

        self.on_error_callback: Optional[Callable[[str], None]] = None

        self.state = FFmpegState.STOPPED
        self._state_lock = asyncio.Lock()

    async def start(self) -> bool:
        async with self._state_lock:
            if self.state != FFmpegState.STOPPED:
                logger.warning(f"FFmpeg already running in state: {self.state}")
                return False
            self.state = FFmpegState.STARTING

        try:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-i", "pipe:0",
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ac", str(self.channels),
                "-ar", str(self.sample_rate),
                "pipe:1"
            ]

            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            self._stderr_task = asyncio.create_task(self._drain_stderr())

            async with self._state_lock:
                self.state = FFmpegState.RUNNING

            logger.info("FFmpeg started.")
            return True

        except FileNotFoundError:
            logger.error(ERROR_INSTALL_INSTRUCTIONS)
            async with self._state_lock:
                self.state = FFmpegState.FAILED
            if self.on_error_callback:
                await self.on_error_callback("ffmpeg_not_found")
            return False

        except Exception as e:
            logger.error(f"Error starting FFmpeg: {e}")
            async with self._state_lock:
                self.state = FFmpegState.FAILED
            if self.on_error_callback:
                await self.on_error_callback("start_failed")
            return False

    async def stop(self):
        async with self._state_lock:
            if self.state == FFmpegState.STOPPED:
                return
            self.state = FFmpegState.STOPPED

        if self.process:
            if self.process.stdin and not self.process.stdin.is_closing():
                self.process.stdin.close()
                await self.process.stdin.wait_closed()
            await self.process.wait()
            self.process = None

        if self._stderr_task:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task

        logger.info("FFmpeg stopped.")

    async def write_data(self, data: bytes) -> bool:
        async with self._state_lock:
            if self.state != FFmpegState.RUNNING:
                logger.warning(f"Cannot write, FFmpeg state: {self.state}")
                return False

        try:
            self.process.stdin.write(data)
            await self.process.stdin.drain()
            return True
        except Exception as e:
            logger.error(f"Error writing to FFmpeg: {e}")
            if self.on_error_callback:
                await self.on_error_callback("write_error")
            return False

    async def read_data(self, size: int) -> Optional[bytes]:
        async with self._state_lock:
            if self.state != FFmpegState.RUNNING:
                logger.warning(f"Cannot read, FFmpeg state: {self.state}")
                return None

        try:
            data = await asyncio.wait_for(
                self.process.stdout.read(size),
                timeout=20.0
            )
            return data
        except asyncio.TimeoutError:
            logger.warning("FFmpeg read timeout.")
            return None
        except Exception as e:
            logger.error(f"Error reading from FFmpeg: {e}")
            if self.on_error_callback:
                await self.on_error_callback("read_error")
            return None

    async def get_state(self) -> FFmpegState:
        async with self._state_lock:
            return self.state

    async def restart(self) -> bool:
        async with self._state_lock:
            if self.state == FFmpegState.RESTARTING:
                logger.warning("Restart already in progress.")
                return False
            self.state = FFmpegState.RESTARTING

        logger.info("Restarting FFmpeg...")

        try:
            await self.stop()
            await asyncio.sleep(1)  # short delay before restarting
            return await self.start()
        except Exception as e:
            logger.error(f"Error during FFmpeg restart: {e}")
            async with self._state_lock:
                self.state = FFmpegState.FAILED
            if self.on_error_callback:
                await self.on_error_callback("restart_failed")
            return False

    async def _drain_stderr(self):
        try:
            while True:
                if not self.process or not self.process.stderr:
                    break
                line = await self.process.stderr.readline()
                if not line:
                    break
                logger.debug(f"FFmpeg stderr: {line.decode(errors='ignore').strip()}")
        except asyncio.CancelledError:
            logger.info("FFmpeg stderr drain task cancelled.")
        except Exception as e:
            logger.error(f"Error draining FFmpeg stderr: {e}")
