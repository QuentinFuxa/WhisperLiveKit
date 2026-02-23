"""Shared pytest fixtures for WhisperLiveKit tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from whisperlivekit.timed_objects import ASRToken, Silence, Transcript


AUDIO_TESTS_DIR = Path(__file__).parent.parent / "audio_tests"


@pytest.fixture
def sample_tokens():
    """A short sequence of ASRToken objects."""
    return [
        ASRToken(start=0.0, end=0.5, text="Hello"),
        ASRToken(start=0.5, end=1.0, text=" world"),
        ASRToken(start=1.0, end=1.5, text=" test."),
    ]


@pytest.fixture
def sample_silence():
    """A completed silence event."""
    s = Silence(start=1.5, end=3.0, is_starting=False, has_ended=True)
    s.compute_duration()
    return s


@pytest.fixture
def mock_args():
    """Minimal args namespace for AudioProcessor tests."""
    return SimpleNamespace(
        diarization=False,
        transcription=True,
        target_language="",
        vac=False,
        vac_chunk_size=0.04,
        min_chunk_size=0.1,
        pcm_input=True,
        punctuation_split=False,
        backend="faster-whisper",
        backend_policy="localagreement",
        vad=True,
    )


@pytest.fixture
def ground_truth_en():
    """Ground truth transcript for the 7s English audio (if available)."""
    path = AUDIO_TESTS_DIR / "00_00_07_english_1_speaker.transcript.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None
