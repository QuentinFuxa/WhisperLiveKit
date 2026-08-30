"""Bound PCM buffering and drain encoded input before signaling audio EOF."""

import asyncio
import logging
from time import monotonic

import numpy as np

from whisperlivekit.ffmpeg_manager import FFmpegManager, FFmpegState
from whisperlivekit.processing_queue import PipelineClosed, PipelineOverloaded

logger = logging.getLogger(__name__)


class AudioInput:
    def __init__(self, *, pcm_input, sample_rate, channels, chunk_bytes,
                 max_chunk_bytes, on_pcm, on_eof):
        self.buffer = bytearray()
        self.chunk_bytes = min(chunk_bytes, max_chunk_bytes)
        self.max_chunk_bytes = max_chunk_bytes
        self.on_pcm, self.on_eof = on_pcm, on_eof
        self.error = None
        self.finished = False
        self.decoder = None if pcm_input else FFmpegManager(sample_rate, channels)
        if self.decoder:
            self.decoder.on_error_callback = self._decoder_error

    async def _decoder_error(self, error):
        self.error = error
        logger.error("FFmpeg error: %s", error)

    async def write(self, message):
        if self.finished:
            return
        if self.decoder:
            if not await self.decoder.write_data(message):
                await self._decoder_error("write_error")
            return
        view = memoryview(message)
        for offset in range(0, len(view), self.max_chunk_bytes):
            self.buffer.extend(view[offset:offset + self.max_chunk_bytes])
            await self.drain()

    async def drain(self, *, final=False):
        while len(self.buffer) >= (2 if final else self.chunk_bytes):
            size = min(len(self.buffer), self.max_chunk_bytes) // 2 * 2
            if not size:
                break
            pcm = np.frombuffer(self.buffer[:size], dtype="<i2").astype(np.float32) / 32768.0
            del self.buffer[:size]
            await self.on_pcm(pcm)

    async def finish(self):
        if self.finished:
            return
        self.finished = True
        if self.decoder:
            # The reader owns decoded EOF. Closing stdin must not discard stdout.
            await self.decoder.close_stdin()
        else:
            await self.drain(final=True)
            await self.on_eof()

    async def read(self):
        """Read decoder output; cancellation leaves cleanup to the orchestrator."""
        previous = monotonic()
        try:
            while True:
                state = await self.decoder.get_state()
                if state in (FFmpegState.FAILED, FFmpegState.STOPPED):
                    break
                if state != FFmpegState.RUNNING:
                    await asyncio.sleep(0.1)
                    continue
                now = monotonic()
                size = min(max(int(32000 * (now - previous)), 4096), self.max_chunk_bytes)
                previous = now
                chunk = await self.decoder.read_data(size)
                if chunk is None:
                    if self.error:
                        break
                    await asyncio.sleep(0.05)
                    continue
                if not chunk:
                    break
                self.buffer.extend(chunk)
                await self.drain()
            await self.drain(final=True)
            await self.decoder.stop()
            await self.on_eof()
        except (PipelineClosed, PipelineOverloaded):
            # Release an encoded producer blocked in stdin.drain().
            await self.decoder.stop()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._decoder_error(str(exc))
            await self.decoder.stop()
