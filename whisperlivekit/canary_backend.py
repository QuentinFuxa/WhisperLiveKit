"""NVIDIA Canary-1b-v2 backend for WhisperLiveKit (LocalAgreement policy).

Contains:
  - CanaryASR: shared NeMo EncDecMultiTaskModel implementing the LocalAgreement
    transcribe()/ts_words()/segments_end_ts() contract.
  - CanaryLID: shared NeMo language-ID model (langid_ambernet).
  - CanarySessionASR: per-session proxy that resolves the source language,
    auto-detecting once via CanaryLID for ``auto`` sessions then locking it.

NeMo and torch are imported lazily inside methods so this module imports fine
on machines without ``nemo_toolkit`` installed (routing/unit tests, non-canary
deployments).
"""

import logging
from typing import List, Optional, Tuple

from whisperlivekit.session_asr_proxy import SessionASRProxy
from whisperlivekit.timed_objects import ASRToken

logger = logging.getLogger(__name__)


# Canary-1b-v2's 25 supported source-language codes.
CANARY_LANGS = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "ru", "sk", "sl", "es", "sv", "uk",
}

# VoxLingua107 (AmberNet) codes that differ from Canary's expected code.
# VoxLingua107 is mostly ISO 639-1 already, so the map is small; extend after
# validating against the actual langid_ambernet label set.
_VOXLINGUA_TO_CANARY = {
    # placeholder for known mismatches, e.g. "gr": "el"
}


def map_voxlingua_to_canary(code: str) -> Optional[str]:
    """Map an AmberNet/VoxLingua107 language code to Canary's source_lang set.

    Returns the mapped code if Canary supports it, else None.
    """
    if not code:
        return None
    code = _VOXLINGUA_TO_CANARY.get(code, code)
    return code if code in CANARY_LANGS else None


def canary_words_to_tokens(word_stamps) -> List[ASRToken]:
    """Convert Canary ``timestamp['word']`` entries to ASRToken objects."""
    if not word_stamps:
        return []
    tokens: List[ASRToken] = []
    for w in word_stamps:
        text = w.get("word")
        if not text:
            continue
        tokens.append(ASRToken(w["start"], w["end"], text))
    return tokens


def canary_segment_end_ts(segment_stamps) -> List[float]:
    """Extract segment end times from Canary ``timestamp['segment']`` entries."""
    if not segment_stamps:
        return []
    return [s["end"] for s in segment_stamps]


class CanarySessionASR(SessionASRProxy):
    """Per-session Canary proxy with auto language detection.

    For explicit-language sessions this behaves like SessionASRProxy, forcing
    ``source_lang`` to the chosen code. For ``auto`` sessions it uses
    ``default_lang`` until ``lid_min_sec`` of audio is buffered, then runs the
    shared LID once; on a confident result it locks that language for the rest
    of the session.
    """

    SAMPLING_RATE = 16000

    def __init__(self, asr, language, lid=None, default_lang="en",
                 lid_min_sec=2.0, lid_min_conf=0.5):
        super().__init__(asr, language)
        is_auto = (language is None) or (language == "auto")
        object.__setattr__(self, "_is_auto", is_auto)
        object.__setattr__(self, "_lid", lid)
        object.__setattr__(self, "_default_lang", default_lang)
        object.__setattr__(self, "_lid_min_sec", lid_min_sec)
        object.__setattr__(self, "_lid_min_conf", lid_min_conf)
        object.__setattr__(self, "_detected_lang", None)

    def _resolve_language(self, audio) -> str:
        # Explicit language: SessionASRProxy stored it as _session_language.
        if not self._is_auto:
            return self._session_language
        if self._detected_lang is not None:
            return self._detected_lang
        # audio is the LocalAgreement growing window, so len(audio) is the
        # accumulated buffer length; detection waits for it to reach lid_min_sec.
        if self._lid is not None and len(audio) >= self._lid_min_sec * self.SAMPLING_RATE:
            try:
                code, conf = self._lid.detect(audio)
            except Exception as e:  # noqa: BLE001
                logger.warning("Canary LID failed: %s", e)
                code, conf = None, 0.0
            if code and conf >= self._lid_min_conf and code in CANARY_LANGS:
                object.__setattr__(self, "_detected_lang", code)
                logger.info("Canary auto-detected language: %s (conf=%.2f)", code, conf)
                return code
        return self._default_lang

    def transcribe(self, audio, init_prompt=""):
        with self._lock:
            lang = self._resolve_language(audio)
            saved = self._asr.original_language
            self._asr.original_language = lang
            try:
                return self._asr.transcribe(audio, init_prompt=init_prompt)
            finally:
                self._asr.original_language = saved
