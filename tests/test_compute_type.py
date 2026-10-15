"""--compute-type reaches every faster-whisper model WhisperLiveKit creates."""

import sys

import faster_whisper

from tests.test_backends import _base_simul_kwargs, _patch_simul_loaders


def test_cli_accepts_compute_type(monkeypatch):
    from whisperlivekit.parse_args import parse_args

    monkeypatch.setattr(sys, "argv", ["whisperlivekit-server", "--compute-type", "float16"])

    assert parse_args().compute_type == "float16"


def test_cli_compute_type_defaults_to_auto(monkeypatch):
    from whisperlivekit.parse_args import parse_args

    monkeypatch.setattr(sys, "argv", ["whisperlivekit-server"])

    assert parse_args().compute_type == "auto"


def test_simulstreaming_encoder_uses_compute_type(monkeypatch):
    backend, _ = _patch_simul_loaders(monkeypatch)

    asr = backend.SimulStreamingASR(**_base_simul_kwargs(compute_type="float16"))

    assert asr.fw_encoder.kwargs["compute_type"] == "float16"


def test_simulstreaming_encoder_compute_type_defaults_to_auto(monkeypatch):
    backend, _ = _patch_simul_loaders(monkeypatch)

    asr = backend.SimulStreamingASR(**_base_simul_kwargs())

    assert asr.fw_encoder.kwargs["compute_type"] == "auto"


def test_faster_whisper_asr_uses_compute_type(monkeypatch):
    from whisperlivekit.local_agreement.backends import FasterWhisperASR

    created = {}

    class FakeWhisperModel:
        def __init__(self, model_size_or_path, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)

    FasterWhisperASR(lan="en", model_size="tiny", compute_type="float16")

    assert created["compute_type"] == "float16"


def test_backend_factory_passes_compute_type_to_faster_whisper(monkeypatch):
    import whisperlivekit.local_agreement.whisper_online as whisper_online

    created = {}

    class FakeFasterWhisperASR:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.transcribe_kargs = {}

    monkeypatch.setattr(whisper_online, "FasterWhisperASR", FakeFasterWhisperASR)

    whisper_online.backend_factory(
        backend="faster-whisper",
        lan="en",
        model_size="tiny",
        model_cache_dir=None,
        model_dir=None,
        model_path=None,
        lora_path=None,
        direct_english_translation=False,
        buffer_trimming="segment",
        buffer_trimming_sec=15,
        confidence_validation=False,
        warmup_file="",  # "" skips warmup; None downloads a sample file
        compute_type="float16",
    )

    assert created["compute_type"] == "float16"
