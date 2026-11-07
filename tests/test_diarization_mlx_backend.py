"""Contract tests for the MLX Sortformer diarization backend.

The backend's mlx-audio dependency is imported inside the model loader, so
these tests run hermetically: the mlx_audio modules are faked at the import
boundary and the streaming contract is exercised against a scripted model.
"""
import asyncio
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from whisperlivekit.timed_objects import SpeakerSegment


class FakeStreamingModel:
    """Scripted stand-in for the mlx-audio streaming sortformer model."""

    def __init__(self, segments_per_chunk=None):
        self.segments_per_chunk = segments_per_chunk or []
        self.received_chunks = []

    def init_streaming_state(self):
        return {"chunk_index": 0}

    def feed(self, chunk, state, sample_rate=16000):
        self.received_chunks.append(chunk)
        index = state["chunk_index"]
        state = {"chunk_index": index + 1}
        segments = self.segments_per_chunk[index] if index < len(self.segments_per_chunk) else []
        result = SimpleNamespace(segments=segments)
        return result, state


@pytest.fixture
def fake_mlx_audio(monkeypatch):
    """Install a fake mlx_audio package whose load() returns a scripted model."""

    def install(model):
        vad = ModuleType("mlx_audio.vad")
        vad.load = lambda repo: model
        mlx_audio = ModuleType("mlx_audio")
        mlx_audio.vad = vad
        monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio)
        monkeypatch.setitem(sys.modules, "mlx_audio.vad", vad)
        model.loaded_repo = None
        original_load = vad.load

        def load(repo):
            model.loaded_repo = repo
            return original_load(repo)

        vad.load = load

    return install


def _make_backend(fake_mlx_audio, model, **online_kwargs):
    from whisperlivekit.diarization.sortformer_mlx_backend import (
        SortformerMLXDiarization,
        SortformerMLXDiarizationOnline,
    )

    fake_mlx_audio(model)
    shared = SortformerMLXDiarization(model_name="test/repo")
    assert shared.model.loaded_repo == "test/repo"
    return SortformerMLXDiarizationOnline(shared_model=shared, **online_kwargs)


def test_module_import_is_lazy():
    """The module must import without mlx_audio in sys.modules (lazy loader)."""
    import whisperlivekit.diarization.sortformer_mlx_backend as backend

    assert hasattr(backend, "SortformerMLXDiarization")
    assert hasattr(backend, "SortformerMLXDiarizationOnline")


def test_loader_passes_repo_and_online_feeds_chunks(fake_mlx_audio):
    model = FakeStreamingModel(segments_per_chunk=[[SimpleNamespace(start=0.0, end=5.0, speaker=1)]])
    online = _make_backend(fake_mlx_audio, model)

    chunk = np.zeros(16000, dtype=np.float32)  # 1s of silence
    for _ in range(6):
        online.insert_audio_chunk(chunk)

    new_segments = asyncio.run(online.diarize())
    assert len(model.received_chunks) == 1  # one full 5s chunk fed
    assert new_segments == [SpeakerSegment(start=0.0, end=5.0, speaker=1)]
    assert online.get_segments() == new_segments


def test_partial_chunk_is_held_until_full(fake_mlx_audio):
    model = FakeStreamingModel()
    online = _make_backend(fake_mlx_audio, model)

    online.insert_audio_chunk(np.zeros(16000 * 3, dtype=np.float32))  # 3s < 5s chunk
    assert asyncio.run(online.diarize()) == []
    assert len(model.received_chunks) == 0

    online.insert_audio_chunk(np.zeros(16000 * 2, dtype=np.float32))
    asyncio.run(online.diarize())
    assert len(model.received_chunks) == 1


def test_max_speakers_filters_segments(fake_mlx_audio):
    model = FakeStreamingModel(
        segments_per_chunk=[
            [
                SimpleNamespace(start=0.0, end=2.0, speaker=1),
                SimpleNamespace(start=2.0, end=4.0, speaker=5),
            ]
        ]
    )
    online = _make_backend(fake_mlx_audio, model, max_speakers=4)

    online.insert_audio_chunk(np.zeros(16000 * 5, dtype=np.float32))
    new_segments = asyncio.run(online.diarize())

    assert [s.speaker for s in new_segments] == [1]
    assert all(s.speaker < 4 for s in online.get_segments())


def test_close_clears_segments(fake_mlx_audio):
    model = FakeStreamingModel(segments_per_chunk=[[SimpleNamespace(start=0.0, end=5.0, speaker=1)]])
    online = _make_backend(fake_mlx_audio, model)

    online.insert_audio_chunk(np.zeros(16000 * 5, dtype=np.float32))
    asyncio.run(online.diarize())
    assert online.get_segments()

    online.close()
    assert online.get_segments() == []


def test_config_accepts_mlx_sortformer_with_max_speakers():
    from whisperlivekit.config import WhisperLiveKitConfig

    config = WhisperLiveKitConfig(diarization_backend="mlx-sortformer", sortformer_max_speakers=2)
    assert config.diarization_backend == "mlx-sortformer"


def test_config_still_rejects_max_speakers_for_other_backends():
    from whisperlivekit.config import WhisperLiveKitConfig

    with pytest.raises(ValueError, match="sortformer_max_speakers requires"):
        WhisperLiveKitConfig(diarization_backend="diart", sortformer_max_speakers=2)


def test_online_diarization_factory_wires_mlx_backend(fake_mlx_audio):
    from types import SimpleNamespace as NS

    from whisperlivekit.core import online_diarization_factory
    from whisperlivekit.diarization import sortformer_mlx_backend as backend

    model = FakeStreamingModel()
    fake_mlx_audio(model)
    shared = backend.SortformerMLXDiarization(model_name="test/repo")
    args = NS(diarization_backend="mlx-sortformer", sortformer_max_speakers=3)

    online = online_diarization_factory(args, shared)

    assert isinstance(online, backend.SortformerMLXDiarizationOnline)
    assert online.max_speakers == 3
