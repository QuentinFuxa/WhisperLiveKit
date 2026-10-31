"""MLX-native Sortformer diarization backend (pure MLX via mlx-audio).

A drop-in alternative to the NeMo-based SortformerDiarization that runs entirely
on Apple Silicon without NeMo, ONNX, or network access. Uses the mlx-community
MLX conversion of nvidia/diar_streaming_sortformer_4spk-v2.1.

The API mirrors the NeMo backend: a shared model holder (SortformerMLXDiarization)
+ a per-session online processor (SortformerMLXDiarizationOnline) with
``process(audio)`` → speaker segments and ``get_segments()``.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional

import numpy as np

from whisperlivekit.timed_objects import SpeakerSegment

logger = logging.getLogger(__name__)


class SortformerMLXDiarization:
    """Shared model holder for the MLX Sortformer diarization backend.

    Loads the mlx-community MLX conversion once; the online processor reuses
    the loaded model across sessions.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16",
        model_path: Optional[str] = None,
    ):
        from mlx_audio.vad import load as load_vad

        repo = model_path or model_name
        logger.info("Loading MLX Sortformer diarization model %s ...", repo)
        self.model = load_vad(repo)
        self.model_name = repo
        logger.info("MLX Sortformer loaded.")


class SortformerMLXDiarizationOnline:
    """Per-session streaming diarization via the MLX Sortformer.

    Feeds audio chunks to ``model.feed(chunk, state)`` and accumulates speaker
    segments. The caller polls ``get_segments()`` for the current diarization
    state (mirrors the NeMo backend's contract).
    """

    def __init__(
        self,
        shared_model: SortformerMLXDiarization,
        sample_rate: int = 16000,
        max_speakers: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.max_speakers = max_speakers
        self._model = shared_model.model
        self._state = self._model.init_streaming_state()
        self._segments: List[SpeakerSegment] = []
        self._lock = threading.Lock()
        self._buffer_audio = np.array([], dtype=np.float32)
        self._chunk_duration = 5.0  # seconds per diarize() call

    def insert_audio_chunk(self, pcm_array: np.ndarray):
        """Buffer incoming audio chunks (called from the diarization processor)."""
        self._buffer_audio = np.concatenate([self._buffer_audio, np.asarray(pcm_array, dtype=np.float32)])

    def insert_silence(self, silence_duration: Optional[float]):
        """Mark silence (no-op for MLX sortformer — it handles silence internally)."""
        pass

    async def diarize(self) -> List[SpeakerSegment]:
        """Process buffered audio and return new speaker segments.

        Feeds audio in chunk_duration-second chunks to the MLX model's streaming
        API, accumulates segments, and returns them. Called from the audio
        processor's diarization loop."""
        threshold = int(self._chunk_duration * self.sample_rate)
        new_segments: List[SpeakerSegment] = []

        while len(self._buffer_audio) >= threshold:
            chunk = self._buffer_audio[:threshold]
            self._buffer_audio = self._buffer_audio[threshold:]

            result, self._state = self._model.feed(
                chunk, self._state, sample_rate=self.sample_rate
            )

            if result and result.segments:
                for seg in result.segments:
                    spk = seg.speaker
                    if self.max_speakers is not None and spk is not None and spk >= self.max_speakers:
                        continue
                    new_segments.append(
                        SpeakerSegment(
                            start=round(seg.start, 2),
                            end=round(seg.end, 2),
                            speaker=spk,
                        )
                    )

        with self._lock:
            self._segments.extend(new_segments)

        return new_segments

    def get_segments(self) -> List[SpeakerSegment]:
        """Get a copy of all accumulated speaker segments."""
        with self._lock:
            return list(self._segments)

    def close(self):
        """Clear accumulated state."""
        with self._lock:
            self._segments.clear()
