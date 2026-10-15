"""One dictation: feed audio into WhisperLiveKit's pipeline and get the final text."""

import asyncio
from typing import Callable

from whisperlivekit.timed_objects import FrontData

SILENCE_SPEAKER = -2


class TranscriptionFailed(Exception):
    """The pipeline reported an error or did not finish in time."""


def final_text(data: FrontData) -> str:
    """Join the spoken lines and any unconfirmed tail into one string."""
    parts = [line.text for line in data.lines if line.text and line.speaker != SILENCE_SPEAKER]
    parts.append(data.buffer_transcription)
    return " ".join(part.strip() for part in parts if part and part.strip())


class DictationSession:
    """Wraps one AudioProcessor for the length of a recording."""

    def __init__(self, processor_factory: Callable[[], object]):
        self._processor = processor_factory()
        self._collector: asyncio.Task | None = None
        self._last = FrontData()
        self._error: str | None = None

    async def start(self):
        results = await self._processor.create_tasks()
        self._collector = asyncio.create_task(self._collect(results))

    async def feed(self, pcm: bytes):
        await self._processor.process_audio(pcm)

    async def finish(self, timeout: float) -> str:
        """Mark the end of audio and wait for the final text."""
        await self._processor.process_audio(b"")
        try:
            await asyncio.wait_for(asyncio.shield(self._collector), timeout=timeout)
        except asyncio.TimeoutError:
            raise TranscriptionFailed(f"Transcription did not finish within {timeout:.0f}s") from None
        if self._error:
            raise TranscriptionFailed(self._error)
        return final_text(self._last)

    async def close(self):
        if self._collector and not self._collector.done():
            self._collector.cancel()
            await asyncio.gather(self._collector, return_exceptions=True)
        await self._processor.cleanup()

    async def _collect(self, results):
        async for result in results:
            if result.status == "error":
                self._error = result.error or "Transcription failed"
                return
            self._last = result
