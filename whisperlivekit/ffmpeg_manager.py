import asyncio
import logging
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)

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
    STOPPING = "stopping"
    RESTARTING = "restarting"
    FAILED = "failed"

class FFmpegManager:
    _STOP_TIMEOUT = 2.0
    _TERMINATE_TIMEOUT = 1.0

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels

        self.process: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._stop_task: Optional[asyncio.Task] = None
        self._read_lock = asyncio.Lock()
        self._restart_lock = asyncio.Lock()

        self.on_error_callback: Optional[Callable[[str], None]] = None

        self.state = FFmpegState.STOPPED
        self._state_lock = asyncio.Lock()

    async def start(self) -> bool:
        try:
            async with self._state_lock:
                if self.state != FFmpegState.STOPPED:
                    logger.warning("FFmpeg already running in state: %s", self.state)
                    return False
                self.state = FFmpegState.STARTING
                self._stop_task = None
                self.process = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le",
                    "-ac", str(self.channels), "-ar", str(self.sample_rate), "pipe:1",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self._stderr_task = asyncio.create_task(self._drain_stderr())
                self.state = FFmpegState.RUNNING
            logger.info("FFmpeg started.")
            return True

        except asyncio.CancelledError:
            await self.stop()
            raise
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
        """Reap the process, discarding unread output on an aborted session.

        Normal EOF uses close_stdin() and lets the pipeline consume stdout
        before calling stop(). A disconnected client may leave both pipes full.
        """
        async with self._state_lock:
            if self._stop_task is None:
                self.state = FFmpegState.STOPPING
                self._stop_task = asyncio.create_task(self._stop_process())
            task = self._stop_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Request cancellation must not abandon the child process.
            await asyncio.shield(task)
            raise

    async def _stop_process(self):
        process = self.process
        drain_task = None
        try:
            if process is None:
                return
            if process.stdin:
                process.stdin.close()
            drain_task = asyncio.create_task(self._discard_stdout(process))
            if self._stderr_task is None:
                self._stderr_task = asyncio.create_task(self._drain_stderr())
            try:
                await asyncio.wait_for(process.wait(), self._STOP_TIMEOUT)
            except asyncio.TimeoutError:
                if process.returncode is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                try:
                    await asyncio.wait_for(process.wait(), self._TERMINATE_TIMEOUT)
                except asyncio.TimeoutError:
                    if process.returncode is None:
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                    await process.wait()
        finally:
            drains = [task for task in (drain_task, self._stderr_task) if task]
            if drains:
                await asyncio.gather(*drains, return_exceptions=True)
            self._stderr_task = None
            self.process = None
            async with self._state_lock:
                self.state = FFmpegState.STOPPED
            logger.info("FFmpeg stopped.")

    async def _discard_stdout(self, process):
        async with self._read_lock:
            if process.stdout:
                while await process.stdout.read(65536):
                    pass

    async def close_stdin(self):
        """Close FFmpeg stdin while keeping stdout readable for draining."""
        if self.process and self.process.stdin and not self.process.stdin.is_closing():
            self.process.stdin.close()
            await self.process.stdin.wait_closed()

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
            async with self._read_lock:
                process = self.process
                if process is None:
                    return b""
                data = await asyncio.wait_for(process.stdout.read(size), timeout=20.0)
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
        if self._restart_lock.locked():
            logger.warning("Restart already in progress.")
            return False
        async with self._restart_lock:
            logger.info("Restarting FFmpeg...")
            try:
                await self.stop()
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
                line = await self.process.stderr.read(65536)
                if not line:
                    break
                logger.debug(f"FFmpeg stderr: {line.decode(errors='ignore').strip()}")
        except asyncio.CancelledError:
            logger.info("FFmpeg stderr drain task cancelled.")
        except Exception as e:
            logger.error(f"Error draining FFmpeg stderr: {e}")
