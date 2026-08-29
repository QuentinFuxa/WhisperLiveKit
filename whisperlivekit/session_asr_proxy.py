"""Per-session ASR proxy for language and decoder-context overrides.

Wraps a shared ASR backend so that each WebSocket session can use a
different transcription language and prompt context without modifying the
shared instance.
"""

import copy
import threading
from typing import Optional

MAX_SESSION_CONTEXT_CHARS = 1000
_PROMPT_BACKENDS = frozenset({
    "faster-whisper",
    "mlx-whisper",
    "openai-api",
    "whisper",
})
_NON_PROMPT_BACKENDS = frozenset({
    "canary",
    "funasr",
    "qwen3-streaming",
    "qwen3-vllm",
    "qwen3-vllm-metal",
    "voxtral",
    "voxtral-mlx",
})


def normalize_session_context(context: Optional[str]) -> Optional[str]:
    """Normalize a user-supplied prompt or reject an unsafe model payload."""
    if context is None:
        return None
    normalized = str(context).strip()
    if not normalized:
        return None
    if len(normalized) > MAX_SESSION_CONTEXT_CHARS:
        raise ValueError(
            f"context must be at most {MAX_SESSION_CONTEXT_CHARS} characters."
        )
    if "\x00" in normalized:
        raise ValueError("context must not contain NUL characters.")
    return normalized


def _backend_name(args, asr) -> str:
    return str(
        getattr(asr, "backend_choice", None)
        or getattr(args, "backend", None)
        or asr.__class__.__name__
    )


def supports_session_context(args, asr) -> bool:
    """Return whether this configured ASR can consume text conditioning."""
    backend = _backend_name(args, asr)
    if backend in _NON_PROMPT_BACKENDS:
        return False
    if getattr(args, "backend_policy", None) == "simulstreaming":
        return True
    return backend in _PROMPT_BACKENDS


def session_context_capability(args, asr) -> dict:
    """Return a protocol-friendly description of session context support."""
    return {
        "supported": supports_session_context(args, asr),
        "maxCharacters": MAX_SESSION_CONTEXT_CHARS,
        "backend": _backend_name(args, asr),
    }


def validate_session_context(args, asr, context: Optional[str]) -> Optional[str]:
    """Normalize context and fail explicitly when the backend cannot use it."""
    normalized = normalize_session_context(context)
    if normalized is not None and not supports_session_context(args, asr):
        backend = _backend_name(args, asr)
        raise ValueError(
            f"Session context is not supported by backend {backend!r}. "
            "Use Whisper, Faster-Whisper, MLX Whisper, OpenAI API, or "
            "SimulStreaming."
        )
    return normalized


def merge_session_context(context: Optional[str], prompt: Optional[str]) -> str:
    """Keep terminology context stable while preserving rolling ASR history."""
    parts = [part for part in (context, prompt) if part]
    return "\n".join(parts)


class SessionASRProxy:
    """Wrap a shared ASR backend with isolated language and prompt context.

    The proxy delegates all attribute access to the wrapped ASR except
    ``transcribe()``, which prepends session context and temporarily overrides
    ``original_language`` on the shared ASR (under a lock) so the correct
    language is used.

    SimulStreaming does not call ``transcribe()``. Its online decoder reads a
    config object when the session is created, so the proxy exposes a shallow
    per-session copy of that config with an isolated language and static prompt.

    Thread-safety: a per-ASR lock serializes ``transcribe()`` calls,
    which is acceptable because model inference is typically GPU-bound
    and cannot be parallelized anyway.
    """

    def __init__(
        self,
        asr,
        language: Optional[str] = None,
        *,
        context: Optional[str] = None,
        simulstreaming: bool = False,
    ):
        object.__setattr__(self, '_asr', asr)
        object.__setattr__(
            self,
            '_session_language',
            None if language in (None, "auto") else language,
        )
        object.__setattr__(self, '_override_language', language is not None)
        object.__setattr__(self, '_session_context', context)
        object.__setattr__(self, '_session_cfg', None)

        if simulstreaming:
            cfg = copy.copy(asr.cfg)
            if language is not None:
                cfg.language = language
            if context is not None:
                cfg.static_init_prompt = merge_session_context(
                    getattr(cfg, "static_init_prompt", None),
                    context,
                )
            object.__setattr__(self, '_session_cfg', cfg)

        # Attach a shared lock to the ASR instance (created once, reused by all proxies)
        if not hasattr(asr, '_session_lock'):
            asr._session_lock = threading.Lock()
        object.__setattr__(self, '_lock', asr._session_lock)

    def __getattr__(self, name):
        if name == "cfg" and self._session_cfg is not None:
            return self._session_cfg
        return getattr(self._asr, name)

    def transcribe(self, audio, init_prompt=""):
        """Call the backend with this session's language and stable context."""
        prompt = merge_session_context(self._session_context, init_prompt)
        with self._lock:
            saved = self._asr.original_language
            if self._override_language:
                self._asr.original_language = self._session_language
            try:
                return self._asr.transcribe(audio, init_prompt=prompt)
            finally:
                self._asr.original_language = saved
