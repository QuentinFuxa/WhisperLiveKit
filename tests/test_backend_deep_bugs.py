import asyncio
from types import SimpleNamespace

import numpy as np
import pytest


def _base_simul_kwargs(**overrides):
    kwargs = {
        "warmup_file": "",
        "min_chunk_size": 0.1,
        "model_size": "tiny",
        "model_cache_dir": None,
        "model_dir": None,
        "model_path": None,
        "encoder_model_path": None,
        "decoder_model_path": None,
        "lora_path": None,
        "lan": "en",
        "direct_english_translation": False,
        "disable_fast_encoder": False,
        "custom_alignment_heads": None,
        "frame_threshold": 25,
        "beams": 1,
        "decoder_type": None,
        "audio_max_len": 30.0,
        "audio_min_len": 0.0,
        "cif_ckpt_path": None,
        "never_fire": False,
        "init_prompt": None,
        "static_init_prompt": None,
        "max_context_tokens": None,
        "backend": "faster-whisper",
    }
    kwargs.update(overrides)
    return kwargs


def _make_ct2_dir(tmp_path):
    model_dir = tmp_path / "ct2"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"ct2")
    (model_dir / "vocabulary.json").write_text("{}", encoding="utf-8")
    return model_dir


def _make_pytorch_dir(tmp_path):
    model_dir = tmp_path / "pytorch"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"torch")
    return model_dir


def _make_mlx_dir(tmp_path):
    model_dir = tmp_path / "mlx"
    model_dir.mkdir()
    (model_dir / "weights.npz").write_bytes(b"mlx")
    return model_dir


def _patch_simul_loaders(monkeypatch):
    import whisperlivekit.simul_whisper.backend as backend

    load_calls = []

    class FakeWhisperModel:
        def __init__(self, model_ref, **kwargs):
            self.model_ref = model_ref
            self.kwargs = kwargs

    def fake_load_model(name, **kwargs):
        load_calls.append((name, kwargs))
        return SimpleNamespace(model_ref=name)

    monkeypatch.setattr(backend, "HAS_FASTER_WHISPER", True)
    monkeypatch.setattr(backend, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(backend, "load_model", fake_load_model)
    return backend, load_calls


def test_simulstreaming_uses_explicit_ct2_encoder_path(tmp_path, monkeypatch):
    backend, load_calls = _patch_simul_loaders(monkeypatch)
    ct2_dir = _make_ct2_dir(tmp_path)

    asr = backend.SimulStreamingASR(
        **_base_simul_kwargs(encoder_model_path=str(ct2_dir))
    )

    assert asr.encoder_backend == "faster-whisper"
    assert asr.fw_encoder.model_ref == str(ct2_dir)
    assert load_calls[0][0] == "tiny"
    assert load_calls[0][1]["decoder_only"] is True


def test_simulstreaming_uses_separate_encoder_and_decoder_paths(tmp_path, monkeypatch):
    backend, load_calls = _patch_simul_loaders(monkeypatch)
    ct2_dir = _make_ct2_dir(tmp_path)
    pytorch_dir = _make_pytorch_dir(tmp_path)

    asr = backend.SimulStreamingASR(
        **_base_simul_kwargs(
            encoder_model_path=str(ct2_dir),
            decoder_model_path=str(pytorch_dir),
        )
    )

    assert asr.fw_encoder.model_ref == str(ct2_dir)
    assert load_calls[0][0] == str(pytorch_dir)
    assert load_calls[0][1]["decoder_only"] is True


def test_simulstreaming_legacy_model_path_rejects_ct2_only_dir(tmp_path, monkeypatch):
    backend, _ = _patch_simul_loaders(monkeypatch)
    ct2_dir = _make_ct2_dir(tmp_path)

    with pytest.raises(FileNotFoundError, match="--encoder-model-path"):
        backend.SimulStreamingASR(
            **_base_simul_kwargs(model_path=str(ct2_dir))
        )


def test_simulstreaming_uses_explicit_mlx_encoder_path(tmp_path, monkeypatch):
    import whisperlivekit.simul_whisper.backend as backend

    mlx_dir = _make_mlx_dir(tmp_path)
    pytorch_dir = _make_pytorch_dir(tmp_path)
    load_calls = []
    mlx_calls = []

    def fake_load_model(name, **kwargs):
        load_calls.append((name, kwargs))
        return SimpleNamespace(model_ref=name)

    def fake_load_mlx_encoder(path_or_hf_repo):
        mlx_calls.append(path_or_hf_repo)
        return SimpleNamespace(model_ref=path_or_hf_repo)

    monkeypatch.setattr(backend, "HAS_MLX_WHISPER", True)
    monkeypatch.setattr(backend, "load_mlx_encoder", fake_load_mlx_encoder, raising=False)
    monkeypatch.setattr(backend, "load_model", fake_load_model)

    asr = backend.SimulStreamingASR(
        **_base_simul_kwargs(
            backend="mlx-whisper",
            encoder_model_path=str(mlx_dir),
            decoder_model_path=str(pytorch_dir),
        )
    )

    assert asr.encoder_backend == "mlx-whisper"
    assert mlx_calls == [str(mlx_dir)]
    assert load_calls[0][0] == str(pytorch_dir)


def test_ctranslate2_storage_view_converts_to_tensor():
    ctranslate2 = pytest.importorskip("ctranslate2")
    from whisperlivekit.simul_whisper.simul_whisper import _encoder_features_to_tensor

    data = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    view = ctranslate2.StorageView.from_array(data)

    tensor = _encoder_features_to_tensor(view, "cpu")

    assert tuple(tensor.shape) == data.shape
    np.testing.assert_allclose(tensor.numpy(), data)


def test_ctranslate2_storage_view_list_converts_to_tensor():
    ctranslate2 = pytest.importorskip("ctranslate2")
    from whisperlivekit.simul_whisper.simul_whisper import _encoder_features_to_tensor

    first = np.arange(6, dtype=np.float32).reshape(2, 3)
    second = first + 10
    views = [
        ctranslate2.StorageView.from_array(first),
        ctranslate2.StorageView.from_array(second),
    ]

    tensor = _encoder_features_to_tensor(views, "cpu")

    assert tuple(tensor.shape) == (2, 2, 3)
    np.testing.assert_allclose(tensor.numpy(), np.stack([first, second]))


class FakeFFmpegManager:
    def __init__(self, chunks=None):
        from whisperlivekit.ffmpeg_manager import FFmpegState

        self.chunks = list(chunks or [])
        self.closed = False
        self.stopped = False
        self.state = FFmpegState.RUNNING

    async def get_state(self):
        return self.state

    async def read_data(self, size):
        await asyncio.sleep(0)
        return self.chunks.pop(0)

    async def close_stdin(self):
        self.closed = True

    async def stop(self):
        self.stopped = True


@pytest.mark.asyncio
async def test_process_audio_non_pcm_closes_ffmpeg_stdin_without_sentinel():
    from whisperlivekit.audio_processor import AudioProcessor

    processor = object.__new__(AudioProcessor)
    processor.beg_loop = 1.0
    processor.is_stopping = False
    processor.is_pcm_input = False
    processor.ffmpeg_manager = FakeFFmpegManager()
    processor.transcription_queue = asyncio.Queue()
    processor.pcm_buffer = bytearray()

    await processor.process_audio(b"")

    assert processor.is_stopping is True
    assert processor.ffmpeg_manager.closed is True
    assert processor.transcription_queue.empty()


@pytest.mark.asyncio
async def test_ffmpeg_reader_drains_stdout_after_stop_before_sentinel():
    from whisperlivekit.audio_processor import SENTINEL, AudioProcessor

    processor = object.__new__(AudioProcessor)
    processor.is_stopping = True
    processor.ffmpeg_manager = FakeFFmpegManager([b"aaaa", None, b"bbbb", b""])
    processor.pcm_buffer = bytearray()
    processor.transcription_queue = asyncio.Queue()
    processor.diarization_queue = None
    processor.translation_queue = None
    processor.diarization = None
    processor.translation = None
    processor.bytes_per_sample = 2
    seen = []

    async def fake_handle_pcm_data():
        seen.append(bytes(processor.pcm_buffer))
        processor.pcm_buffer.clear()

    processor.handle_pcm_data = fake_handle_pcm_data

    await processor.ffmpeg_stdout_reader()

    assert seen == [b"aaaa", b"bbbb"]
    assert processor.ffmpeg_manager.stopped is True
    assert await processor.transcription_queue.get() is SENTINEL
    assert processor.transcription_queue.empty()
