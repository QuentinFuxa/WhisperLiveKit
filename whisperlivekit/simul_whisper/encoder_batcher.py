"""
EncoderBatcher: Batches encoder forward passes across multiple sessions.

This module provides GPU utilization optimization by collecting encoder requests
from multiple concurrent clients and processing them in a single batched GPU call.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import torch

from ..whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

logger = logging.getLogger(__name__)


class EncoderBatcher:
    """
    Batches encoder forward passes across multiple sessions for GPU efficiency.

    Instead of processing each client's audio through the encoder separately,
    this class collects requests over a short time window and processes them
    in a single batched GPU call.

    Architecture:
        Client 1 audio ──┐
        Client 2 audio ──┼──→ enqueue() → [5ms wait] → batched encoder() → results
        Client 3 audio ──┘

    Args:
        shared_model: The Whisper model containing the encoder
        device: Device to run inference on ('cuda', 'cpu', etc.)
        n_mels: Number of mel filterbank bins (default: 80)
        max_batch: Maximum requests per batch (default: 32)
        max_delay_ms: Maximum wait time to collect requests in ms (default: 5.0)
    """

    def __init__(
        self,
        shared_model,
        device: str,
        n_mels: int = 80,
        max_batch: int = 32,
        max_delay_ms: float = 5.0,
    ):
        self.shared_model = shared_model
        self.device = device
        self.n_mels = n_mels
        self.max_batch = max_batch
        self.max_delay_ms = max_delay_ms

        # Pending requests: List of (mel_tensor, future, content_mel_len)
        self._pending: List[Tuple[torch.Tensor, asyncio.Future, int]] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None

        # Single-worker executor to serialize GPU access
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="encoder_batch")

        logger.info(
            "EncoderBatcher initialized: max_batch=%d, max_delay_ms=%.1f, device=%s",
            max_batch, max_delay_ms, device
        )

    async def encode(self, audio_tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Enqueue audio for batched encoding.

        Args:
            audio_tensor: Raw audio waveform tensor, shape (N_samples,)

        Returns:
            Tuple of (encoder_features, content_mel_len):
                - encoder_features: Encoder output, shape (1, 1500, n_state)
                - content_mel_len: Length of actual audio content in mel frames
        """
        # Compute mel spectrogram (done per-request, relatively fast)
        mel_padded = log_mel_spectrogram(
            audio_tensor,
            n_mels=self.n_mels,
            padding=N_SAMPLES,
            device=self.device,
        ).unsqueeze(0)  # (1, n_mels, n_frames)

        mel = pad_or_trim(mel_padded, N_FRAMES)  # (1, n_mels, 3000)
        content_mel_len = int((mel_padded.shape[2] - mel.shape[2]) / 2)

        # Create future for this request
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        async with self._lock:
            self._pending.append((mel, fut, content_mel_len))

            # Start flush task if not already running
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._flush())

        # Wait for result
        return await fut

    async def _flush(self):
        """Process pending requests in batches after a short delay."""
        # Wait to collect more requests
        await asyncio.sleep(self.max_delay_ms / 1000)

        loop = asyncio.get_running_loop()

        while True:
            # Get batch under lock
            async with self._lock:
                if not self._pending:
                    return

                batch = self._pending[:self.max_batch]
                del self._pending[:len(batch)]

            batch_size = len(batch)
            logger.debug("Processing encoder batch of size %d", batch_size)

            try:
                # Stack mels: List[(1, n_mels, 3000)] -> (N, n_mels, 3000)
                mels = torch.cat([item[0] for item in batch], dim=0)

                # Run encoder in thread pool to not block event loop
                encoder_out = await loop.run_in_executor(
                    self._executor,
                    self._run_encoder,
                    mels,
                )

                # Distribute results back to futures
                for i, (_, fut, content_mel_len) in enumerate(batch):
                    if not fut.done():
                        # Extract this request's encoder output: (1, 1500, n_state)
                        fut.set_result((encoder_out[i:i+1], content_mel_len))

            except Exception as e:
                logger.error("Encoder batch failed: %s", e)
                # Fail all futures in the batch
                for _, fut, _ in batch:
                    if not fut.done():
                        fut.set_exception(e)

    def _run_encoder(self, mels: torch.Tensor) -> torch.Tensor:
        """
        Run encoder forward pass (called in thread pool).

        Args:
            mels: Batched mel spectrograms, shape (batch, n_mels, 3000)

        Returns:
            Encoder features, shape (batch, 1500, n_state)
        """
        with torch.no_grad():
            return self.shared_model.encoder(mels.to(self.device))

    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)
