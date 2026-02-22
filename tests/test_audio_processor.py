"""Tests for AudioProcessor pipeline with mocked ASR backends.

These tests verify the async audio processing pipeline works correctly
without requiring any real ASR models to be loaded.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from whisperlivekit.timed_objects import ASRToken, Transcript


# ---------------------------------------------------------------------------
# Mock ASR components
# ---------------------------------------------------------------------------

class MockASR:
    """Mock ASR model holder."""
    sep = " "
    SAMPLING_RATE = 16000

    def __init__(self):
        self.transcribe_kargs = {}
        self.original_language = "en"
        self.backend_choice = "mock"

    def transcribe(self, audio):
        return None


class MockOnlineProcessor:
    """Mock online processor that returns canned tokens."""
    SAMPLING_RATE = 16000

    def __init__(self, asr=None):
        self.asr = asr or MockASR()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.end = 0.0
        self._call_count = 0
        self._finished = False

    def insert_audio_chunk(self, audio, audio_stream_end_time):
        self.audio_buffer = np.append(self.audio_buffer, audio)
        self.end = audio_stream_end_time

    def process_iter(self, is_last=False):
        self._call_count += 1
        # Emit a token on every call when we have audio
        if len(self.audio_buffer) > 0:
            t = self._call_count * 0.5
            return [ASRToken(start=t, end=t + 0.5, text=f"word{self._call_count}")], self.end
        return [], self.end

    def get_buffer(self):
        return Transcript(start=None, end=None, text="")

    def start_silence(self):
        return [], self.end

    def end_silence(self, silence_duration, offset):
        pass

    def new_speaker(self, change_speaker):
        pass

    def finish(self):
        self._finished = True
        return [], self.end

    def warmup(self, audio, init_prompt=""):
        pass


def _make_pcm_bytes(duration_s=0.1, sample_rate=16000):
    """Generate silent PCM s16le bytes."""
    n_samples = int(duration_s * sample_rate)
    audio = np.zeros(n_samples, dtype=np.float32)
    return (audio * 32768).clip(-32768, 32767).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_engine():
    """Create a mock TranscriptionEngine-like object."""
    engine = SimpleNamespace(
        asr=MockASR(),
        diarization_model=None,
        translation_model=None,
        args=SimpleNamespace(
            diarization=False,
            transcription=True,
            target_language="",
            vac=False,
            vac_chunk_size=0.04,
            min_chunk_size=0.1,
            pcm_input=True,
            punctuation_split=False,
            backend="mock",
            backend_policy="localagreement",
            vad=True,
            model_size="base",
            lan="en",
        ),
    )
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPCMConversion:
    """Test PCM byte conversion without needing the full pipeline."""

    def test_s16le_roundtrip(self):
        """Convert float32 → s16le → float32 and verify approximate roundtrip."""
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        s16 = (original * 32768).clip(-32768, 32767).astype(np.int16)
        pcm_bytes = s16.tobytes()
        # Direct numpy conversion (same logic as AudioProcessor.convert_pcm_to_float)
        recovered = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        np.testing.assert_allclose(recovered, original, atol=1 / 32768)


@pytest.mark.asyncio
class TestPipelineBasics:
    async def test_feed_audio_and_get_responses(self, mock_engine):
        """Feed audio through the pipeline and verify we get responses."""
        from whisperlivekit.audio_processor import AudioProcessor

        with patch("whisperlivekit.audio_processor.online_factory", return_value=MockOnlineProcessor()):
            processor = AudioProcessor(transcription_engine=mock_engine)
            results_gen = await processor.create_tasks()

            responses = []

            async def collect():
                async for resp in results_gen:
                    responses.append(resp)

            task = asyncio.create_task(collect())

            # Feed 2 seconds of audio in 100ms chunks
            for _ in range(20):
                await processor.process_audio(_make_pcm_bytes(0.1))

            # Signal EOF
            await processor.process_audio(None)

            await asyncio.wait_for(task, timeout=10.0)
            await processor.cleanup()

            # We should have gotten at least one response
            assert len(responses) > 0

    async def test_eof_terminates_pipeline(self, mock_engine):
        """Sending None (EOF) should cleanly terminate the pipeline."""
        from whisperlivekit.audio_processor import AudioProcessor

        with patch("whisperlivekit.audio_processor.online_factory", return_value=MockOnlineProcessor()):
            processor = AudioProcessor(transcription_engine=mock_engine)
            results_gen = await processor.create_tasks()

            responses = []

            async def collect():
                async for resp in results_gen:
                    responses.append(resp)

            task = asyncio.create_task(collect())

            # Send a small amount of audio then EOF
            await processor.process_audio(_make_pcm_bytes(0.5))
            await processor.process_audio(None)

            await asyncio.wait_for(task, timeout=10.0)
            await processor.cleanup()

            # Pipeline should have terminated without error
            assert task.done()

    async def test_empty_audio_no_crash(self, mock_engine):
        """Sending EOF immediately (no audio) should not crash."""
        from whisperlivekit.audio_processor import AudioProcessor

        with patch("whisperlivekit.audio_processor.online_factory", return_value=MockOnlineProcessor()):
            processor = AudioProcessor(transcription_engine=mock_engine)
            results_gen = await processor.create_tasks()

            responses = []

            async def collect():
                async for resp in results_gen:
                    responses.append(resp)

            task = asyncio.create_task(collect())
            await processor.process_audio(None)

            await asyncio.wait_for(task, timeout=10.0)
            await processor.cleanup()
            assert task.done()
