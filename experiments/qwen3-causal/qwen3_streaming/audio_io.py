from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio_mono(path: str | Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 and resample when needed."""
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return np.asarray(audio, dtype=np.float32), sr


def write_pcm16_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono audio as PCM16 WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    sf.write(str(path), clipped, sample_rate, subtype="PCM_16")


def audio_duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    return float(len(audio)) / float(sample_rate) if sample_rate else 0.0


def prefix_audio(audio: np.ndarray, sample_rate: int, end_seconds: float) -> np.ndarray:
    n_samples = int(round(max(0.0, end_seconds) * sample_rate))
    return audio[: min(n_samples, len(audio))]


def pcm16_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()
