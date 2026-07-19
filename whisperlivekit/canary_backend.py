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


class CanaryASR:
    """Shared Canary model holder implementing the LocalAgreement contract."""

    sep = " "
    SAMPLING_RATE = 16000

    def __init__(self, lan="auto", canary_model="nvidia/canary-1b-v2",
                 buffer_trimming="segment", buffer_trimming_sec=15.0,
                 confidence_validation=False, canary_default_lang="en",
                 logfile=None, **_unused):
        import time

        self.original_language = None if lan == "auto" else lan
        self.canary_default_lang = canary_default_lang
        self.backend_choice = "canary"
        self.confidence_validation = confidence_validation
        self.tokenizer = None  # segment trimming needs no sentence tokenizer
        self.buffer_trimming = buffer_trimming
        self.buffer_trimming_sec = buffer_trimming_sec
        self.transcribe_kargs = {}
        self.lid_model = None  # attached by core.py when auto detection is enabled

        from nemo.collections.asr.models import ASRModel

        t = time.time()
        logger.info("Loading Canary model '%s' via NeMo...", canary_model)
        if canary_model.endswith(".nemo"):
            self.model = ASRModel.restore_from(canary_model)
        else:
            self.model = ASRModel.from_pretrained(model_name=canary_model)
        self.model.eval()
        logger.info("Canary model loaded in %.2fs", time.time() - t)

    def transcribe(self, audio, init_prompt="", source_lang=None):
        """Run Canary on a 16kHz mono float32 numpy window. Returns hyp[0]."""
        # init_prompt is accepted for LocalAgreement interface compatibility but
        # intentionally unused: Canary (offline attention enc-dec) has no
        # prompt-conditioning slot; LocalAgreement relies on the growing buffer.
        import numpy as np

        lang = source_lang or self.original_language or self.canary_default_lang
        audio = np.asarray(audio, dtype=np.float32)
        outputs = self.model.transcribe(
            [audio],
            source_lang=lang,
            target_lang=lang,
            timestamps=True,
            batch_size=1,
            verbose=False,
        )
        # An all-silence / empty window can yield no hypotheses. ts_words() and
        # segments_end_ts() both tolerate None (their helpers null-guard), so
        # returning None here degrades cleanly instead of raising IndexError.
        if not outputs:
            return None
        return outputs[0]

    def _word_stamps(self, res):
        ts = getattr(res, "timestamp", None) or {}
        return ts.get("word")

    def _segment_stamps(self, res):
        ts = getattr(res, "timestamp", None) or {}
        return ts.get("segment")

    def ts_words(self, res) -> List[ASRToken]:
        return canary_words_to_tokens(self._word_stamps(res))

    def segments_end_ts(self, res) -> List[float]:
        return canary_segment_end_ts(self._segment_stamps(res))

    def use_vad(self):
        logger.warning("VAD is handled upstream (Silero); CanaryASR.use_vad() is a no-op.")


class CanaryLID:
    """Shared spoken-language-ID model (NeMo langid_ambernet / AmberNet)."""

    SAMPLING_RATE = 16000

    def __init__(self, lid_model="langid_ambernet", logfile=None, **_unused):
        import time

        import nemo.collections.asr as nemo_asr

        t = time.time()
        logger.info("Loading Canary LID model '%s' via NeMo...", lid_model)
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=lid_model
        )
        self.model.eval()
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover
            self.device = "cpu"
        logger.info("Canary LID model loaded in %.2fs", time.time() - t)

    def detect(self, audio) -> Tuple[Optional[str], float]:
        """Return (canary_lang_code_or_None, confidence) for a 16kHz clip."""
        import numpy as np
        import torch

        arr = np.asarray(audio, dtype=np.float32)
        sig = torch.tensor(arr).unsqueeze(0).to(self.device)
        sig_len = torch.tensor([sig.shape[1]]).to(self.device)
        with torch.no_grad():
            logits, _ = self.model.forward(input_signal=sig, input_signal_length=sig_len)
            probs = logits.softmax(dim=-1)
            conf, idx = probs.max(dim=-1)
        raw_code = self.model.cfg.labels[int(idx.item())]
        confidence = float(conf.item())
        mapped = map_voxlingua_to_canary(raw_code)
        if mapped is None:
            logger.warning(
                "Canary LID predicted label %r (conf=%.2f) with no mapping into "
                "Canary's supported languages; extend _VOXLINGUA_TO_CANARY. "
                "Falling back to the default language.",
                raw_code, confidence,
            )
        return mapped, confidence
