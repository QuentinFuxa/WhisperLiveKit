"""Sortformer inference must not block the asyncio event loop."""

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_sortformer_max_speakers import sortformer_module  # noqa: F401

FORWARD_SECONDS = 0.5


class _SlowSortformer:
    """Stands in for a Sortformer forward pass on a slow or loaded host."""

    device = torch.device("cpu")

    def forward_streaming_step(self, *, processed_signal, streaming_state, total_preds, **_):
        time.sleep(FORWARD_SECONDS)
        frames = torch.tensor([[[0.9, 0.1, 0.0, 0.0]] * 4], dtype=torch.float32)
        return streaming_state, torch.cat([total_preds, frames], dim=1)


class _Mel:
    def get_features(self, audio, length):
        return torch.zeros((1, 128, 10)), torch.tensor([10])


def _online(sortformer_module):  # noqa: F811
    online = object.__new__(sortformer_module.SortformerDiarizationOnline)
    online.sample_rate = 16000
    online.chunk_duration_seconds = 1.0
    online.buffer_audio = np.zeros(16000, dtype=np.float32)
    online.diar_model = _SlowSortformer()
    online.audio2mel = _Mel()
    online.streaming_state = SimpleNamespace()
    online.total_preds = torch.zeros((1, 0, 4))
    online._previous_chunk_features = None
    online._chunk_index = 0
    online._len_prediction = None
    online.max_speakers = 4
    online.segment_lock = threading.Lock()
    online.global_time_offset = 0.0
    return online


@pytest.mark.asyncio
async def test_diarize_keeps_event_loop_responsive(sortformer_module):  # noqa: F811
    online = _online(sortformer_module)
    max_gap = 0.0
    done = asyncio.Event()

    async def heartbeat():
        nonlocal max_gap
        last = time.perf_counter()
        while not done.is_set():
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            max_gap = max(max_gap, now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.05)
    segments = await online.diarize()
    done.set()
    await beat

    assert [(int(s.speaker), s.start, s.end) for s in segments] == [(0, 0.0, 1.0)]
    assert online._chunk_index == 1
    assert len(online.buffer_audio) == 0
    assert max_gap < FORWARD_SECONDS / 2
