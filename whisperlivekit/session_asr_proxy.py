"""Per-session ASR proxy for language override.

Wraps a shared ASR backend so that each WebSocket session can use a
different transcription language without modifying the shared instance.
"""

import threading


class SessionASRProxy:
    """Wraps a shared ASR backend with a per-session language override.

    The proxy delegates all attribute access to the wrapped ASR except
    ``transcribe()``, which temporarily overrides ``original_language``
    on the shared ASR (under a lock) so the correct language is used.

    Thread-safety: a per-ASR lock serializes ``transcribe()`` calls,
    which is acceptable because model inference is typically GPU-bound
    and cannot be parallelized anyway.
    """

    def __init__(self, asr, language: str):
        object.__setattr__(self, '_asr', asr)
        object.__setattr__(self, '_session_language', None if language == "auto" else language)
        # Attach a shared lock to the ASR instance (created once, reused by all proxies)
        if not hasattr(asr, '_session_lock'):
            asr._session_lock = threading.Lock()
        object.__setattr__(self, '_lock', asr._session_lock)

    def __getattr__(self, name):
        return getattr(self._asr, name)

    def transcribe(self, audio, init_prompt=""):
        """Call the backend's transcribe with the session's language."""
        with self._lock:
            saved = self._asr.original_language
            self._asr.original_language = self._session_language
            try:
                return self._asr.transcribe(audio, init_prompt=init_prompt)
            finally:
                self._asr.original_language = saved
