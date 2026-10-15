"""Tests for DictationSession against a fake AudioProcessor."""

import asyncio

import pytest

from whisperlivekit.dictate.session import DictationSession, TranscriptionFailed, final_text
from whisperlivekit.timed_objects import FrontData, Segment


def _front(*texts, speaker=1, status="active_transcription", error=""):
    lines = [Segment(start=0, end=1, text=text, speaker=speaker) for text in texts]
    return FrontData(status=status, error=error, lines=lines)


class FakeProcessor:
    """Yields each result after audio arrives; the empty chunk ends the stream."""

    def __init__(self, results, finish_delay=0.0):
        self.results = results
        self.finish_delay = finish_delay
        self.received = []
        self.cleaned_up = False
        self._ended = asyncio.Event()

    async def create_tasks(self):
        async def generate():
            await self._ended.wait()
            await asyncio.sleep(self.finish_delay)
            for result in self.results:
                yield result
        return generate()

    async def process_audio(self, message):
        self.received.append(message)
        if not message:
            self._ended.set()

    async def cleanup(self):
        self.cleaned_up = True


def _session(processor):
    return DictationSession(processor_factory=lambda: processor)


def test_final_text_joins_lines_and_skips_silence():
    data = FrontData(lines=[
        Segment(start=0, end=1, text=" Hello there.", speaker=1),
        Segment(start=1, end=2, text=None, speaker=-2),
        Segment(start=2, end=3, text="How are you? ", speaker=1),
    ], buffer_transcription=" Fine")

    assert final_text(data) == "Hello there. How are you? Fine"


@pytest.mark.asyncio
async def test_finish_returns_last_result_text():
    processor = FakeProcessor([_front("partial"), _front("Hello", "world.")])
    session = _session(processor)
    await session.start()
    await session.feed(b"\x00\x00")

    text = await session.finish(timeout=5)

    assert text == "Hello world."
    assert processor.received == [b"\x00\x00", b""]


@pytest.mark.asyncio
async def test_error_result_raises():
    session = _session(FakeProcessor([_front(status="error", error="CUDA failed with error out of memory")]))
    await session.start()

    with pytest.raises(TranscriptionFailed, match="out of memory"):
        await session.finish(timeout=5)


@pytest.mark.asyncio
async def test_finish_timeout_raises():
    session = _session(FakeProcessor([_front("late")], finish_delay=1.0))
    await session.start()

    with pytest.raises(TranscriptionFailed, match="did not finish"):
        await session.finish(timeout=0.05)


@pytest.mark.asyncio
async def test_close_cleans_up():
    processor = FakeProcessor([_front("x")])
    session = _session(processor)
    await session.start()

    await session.close()

    assert processor.cleaned_up
