"""Audio checks and conversion for dictation."""

import re

import numpy as np

SAMPLE_RATE = 16000

# Whole outputs Whisper tends to produce for silence or noise
HALLUCINATION_PHRASES = [
    "thanks for watching", "thank you for watching", "thanks for listening",
    "thank you for listening", "thank you", "thank you very much", "thanks",
    "please subscribe", "like and subscribe", "don't forget to subscribe",
    "see you next time", "see you in the next video", "subtitles by the amara.org community",
    "you", "uh", "um", "hmm", "hm", "huh", "ah", "oh", "bye", "bye bye",
    "[music]", "(music)", "[silence]", "[blank_audio]",
]


def to_pcm16(chunk: np.ndarray) -> bytes:
    """Convert float32 samples in [-1, 1] to 16-bit PCM bytes."""
    return (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def has_speech(audio: np.ndarray, threshold: float = 0.01, segment_ms: int = 50) -> bool:
    """True if any short segment is louder than the threshold.

    Checking segments rather than the whole recording catches short phrases
    surrounded by silence.
    """
    segment_samples = int(SAMPLE_RATE * segment_ms / 1000)
    for i in range(0, len(audio), segment_samples):
        segment = audio[i:i + segment_samples]
        if len(segment) < segment_samples // 2:
            continue
        if np.sqrt(np.mean(segment ** 2)) > threshold:
            return True
    return False


def is_hallucination(text: str) -> bool:
    """True if the whole text is filler Whisper produces from silence, possibly repeated."""
    t = _normalize(text)
    for phrase in sorted(HALLUCINATION_PHRASES, key=len, reverse=True):
        t = re.sub(rf"(?<![\w']){re.escape(phrase)}(?![\w'])", " ", t)
    return not any(char.isalnum() for char in t)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[.,!?;:…]", " ", text.lower()).split())
