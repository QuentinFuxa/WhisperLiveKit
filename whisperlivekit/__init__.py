"""Public API. Model and audio dependencies load when their API is requested."""

from importlib import import_module

from .config import WhisperLiveKitConfig
from .parse_args import parse_args

_LAZY_EXPORTS = {
    "TranscriptionEngine": "core",
    "AudioProcessor": "audio_processor",
    "transcribe_audio": "test_client",
    "TranscriptionResult": "test_client",
    "TestHarness": "test_harness",
    "TestState": "test_harness",
    "get_web_interface_html": "web.web_interface",
    "get_inline_ui_html": "web.web_interface",
}

__all__ = ["WhisperLiveKitConfig", "parse_args", *_LAZY_EXPORTS]


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_LAZY_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
