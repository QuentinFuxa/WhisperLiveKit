"""Incremental Whisper-mel extraction for the Qwen3 streaming backend.

The validated offline evals featurized whole files in one call to the Qwen
``WhisperFeatureExtractor`` (hop 160 samples / 10 ms, ``n_fft`` 400, STFT
``center=True``). Streaming needs the same frames produced incrementally.

Frame ``t`` only depends on samples ``[t*160 - 200, t*160 + 200)``, so frames
are reproduced sample-exactly from a sliding window with enough margin. Two
documented approximations versus one-shot extraction remain:

- the dynamic-range clamp ``log_spec.max() - 8.0`` uses the local window's max
  instead of the file-global max. For speech this is identical in practice
  (any frame near the loudest one dominates both); it only differs on windows
  that are entirely near-silent, which transcribe to nothing anyway.
- a sub-frame tail shorter than the STFT pad (200 samples) is zero-padded at
  flush instead of reflect-padded.
"""

from __future__ import annotations

import numpy as np
import torch

_HOP = 160
_STFT_PAD = 200  # n_fft // 2, reflect padding reach of center=True
# Emitted frames must sit this many frames inside the featurized window so the
# window-edge reflect padding cannot influence them (2 frames = 320 > 200
# samples; 8 gives a comfortable margin).
_MARGIN_FRAMES = 8


class StreamingMelExtractor:
    """Produces the same mel frames as one-shot extraction, incrementally."""

    def __init__(self, feature_extractor, sample_rate: int = 16_000):
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start_frame = 0  # global frame index of self._buffer[0] / _HOP
        self._emitted_frames = 0
        self._total_samples = 0

    @property
    def emitted_frames(self) -> int:
        return self._emitted_frames

    def _featurize(self, samples: np.ndarray) -> torch.Tensor:
        """[frames, n_mels] mel for a raw sample window."""
        features = self.feature_extractor(
            samples,
            sampling_rate=self.sample_rate,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )["input_features"][0]
        return features.transpose(0, 1).contiguous()

    def _emit(self, upto_frame: int) -> torch.Tensor | None:
        """Featurize the buffered window and emit frames [emitted, upto_frame)."""
        if upto_frame <= self._emitted_frames:
            return None
        local_first = self._emitted_frames - self._buffer_start_frame
        local_last = upto_frame - self._buffer_start_frame
        mel = self._featurize(self._buffer)
        if mel.shape[0] < local_last:
            local_last = int(mel.shape[0])
            upto_frame = self._buffer_start_frame + local_last
            if upto_frame <= self._emitted_frames:
                return None
        frames = mel[local_first:local_last]
        self._emitted_frames = upto_frame

        # Drop samples no longer needed: keep _MARGIN_FRAMES frames of history
        # before the next frame to emit.
        keep_from_frame = max(self._buffer_start_frame, self._emitted_frames - _MARGIN_FRAMES)
        cut_samples = (keep_from_frame - self._buffer_start_frame) * _HOP
        if cut_samples > 0:
            self._buffer = self._buffer[cut_samples:]
            self._buffer_start_frame = keep_from_frame
        return frames.unsqueeze(0)  # [1, frames, n_mels]

    def append(self, audio: np.ndarray) -> torch.Tensor | None:
        """Buffer samples; return newly determined mel frames or None.

        Only frames whose full STFT window is covered by real samples are
        emitted; the tail is held back until more audio (or ``flush``).
        """
        if audio.size:
            self._buffer = np.concatenate([self._buffer, audio.astype(np.float32, copy=False)])
            self._total_samples += int(audio.size)
        if self._total_samples < _STFT_PAD + 1:
            return None
        ready = (self._total_samples - _STFT_PAD) // _HOP + 1
        ready = min(ready, self._total_samples // _HOP)
        return self._emit(ready)

    def flush(self) -> torch.Tensor | None:
        """Emit all remaining frames, padding the tail like end-of-file."""
        total_frames = self._total_samples // _HOP
        if total_frames <= self._emitted_frames:
            return None
        min_samples = 2 * _STFT_PAD + 1
        if self._buffer.size < min_samples:
            self._buffer = np.pad(self._buffer, (0, min_samples - self._buffer.size))
        return self._emit(total_frames)

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start_frame = 0
        self._emitted_frames = 0
        self._total_samples = 0
