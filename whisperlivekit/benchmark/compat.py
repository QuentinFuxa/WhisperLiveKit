"""Backend detection and language compatibility matrix."""

import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# https://huggingface.co/Qwen/Qwen3-ASR-1.7B#released-models-description-and-download
_QWEN_LANGUAGES = {
    "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko", "ru",
    "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv", "da", "fi", "pl", "cs",
    "fil", "fa", "el", "hu", "mk", "ro",
}

# Language support per backend.
# None means all Whisper-supported languages.
# A set means only those languages are supported.
BACKEND_LANGUAGES: Dict[str, Optional[Set[str]]] = {
    "whisper": None,
    "faster-whisper": None,
    "mlx-whisper": None,
    "openai-api": None,
    # https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
    "voxtral-mlx": {"en", "es", "fr", "pt", "hi", "de", "nl", "it", "ru", "ar", "ko", "ja", "zh"},
    "voxtral": {"en", "es", "fr", "pt", "hi", "de", "nl", "it", "ru", "ar", "ko", "ja", "zh"},
    # The qwen3 combos declare their own language restriction (the causal
    # tower is English-only, the windowed tower is multilingual).
    "funasr": {"zh", "yue", "en", "ja", "ko"},
}


def backend_supports_language(backend: str, language: str) -> bool:
    """Check if a backend supports a given language code."""
    if backend == "canary":
        from whisperlivekit.canary_backend import CANARY_LANGS
        return language in CANARY_LANGS
    if backend in ("qwen3-streaming", "qwen3-vllm", "qwen3-vllm-metal"):
        # This CLI uses the default windowed/standard tower, not the
        # English-only experimental causal tower. Use its published inventory.
        return language in _QWEN_LANGUAGES
    if backend not in BACKEND_LANGUAGES:
        raise ValueError(f"Unknown benchmark backend: {backend}")
    langs = BACKEND_LANGUAGES[backend]
    if langs is None:
        from whisperlivekit.whisper.tokenizer import LANGUAGES
        return language in LANGUAGES
    return language in langs


def detect_available_backends() -> List[str]:
    """Probe which ASR backends are importable."""
    backends = []

    try:
        import whisper  # noqa: F401
        backends.append("whisper")
    except ImportError:
        pass

    try:
        import faster_whisper  # noqa: F401
        backends.append("faster-whisper")
    except ImportError:
        pass

    try:
        import mlx_whisper  # noqa: F401
        backends.append("mlx-whisper")
    except ImportError:
        pass

    try:
        import mlx.core  # noqa: F401

        from whisperlivekit.voxtral_mlx.loader import load_voxtral_model  # noqa: F401
        backends.append("voxtral-mlx")
    except ImportError:
        pass

    try:
        from transformers import VoxtralRealtimeForConditionalGeneration  # noqa: F401
        backends.append("voxtral")
    except ImportError:
        pass

    return backends


def resolve_backend(backend: str) -> str:
    """Resolve 'auto' to the best available backend."""
    if backend != "auto":
        return backend

    available = detect_available_backends()
    if not available:
        raise RuntimeError(
            "No ASR backend available. Install at least one: "
            "pip install openai-whisper, faster-whisper, or mlx-whisper"
        )

    # Priority order
    priority = [
        "faster-whisper", "mlx-whisper", "voxtral-mlx", "voxtral",
        "whisper",
    ]
    for p in priority:
        if p in available:
            return p
    return available[0]
