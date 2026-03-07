import logging
import sys
from typing import List, Optional

import numpy as np

from whisperlivekit.local_agreement.backends import ASRBase
from whisperlivekit.timed_objects import ASRToken

logger = logging.getLogger(__name__)


def _patch_transformers_compat():
    """Patch transformers for qwen_asr compatibility.

    qwen_asr imports ``check_model_inputs`` from ``transformers.utils.generic``,
    but this decorator hasn't been released yet in any public transformers
    version.  We inject a no-op stub so the import succeeds.
    """
    try:
        import transformers.utils.generic as _g
        if not hasattr(_g, "check_model_inputs"):
            def check_model_inputs(*args, **kwargs):
                def decorator(fn):
                    return fn
                return decorator
            _g.check_model_inputs = check_model_inputs
    except ImportError:
        pass


_patch_transformers_compat()

# Whisper language codes → Qwen3 canonical language names
WHISPER_TO_QWEN3_LANGUAGE = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese",
    "ar": "Arabic", "de": "German", "fr": "French", "es": "Spanish",
    "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
    "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
    "ja": "Japanese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay",
    "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish",
    "pl": "Polish", "cs": "Czech", "fa": "Persian",
    "el": "Greek", "hu": "Hungarian", "mk": "Macedonian", "ro": "Romanian",
}

# Reverse mapping: Qwen3 canonical names → Whisper language codes
QWEN3_TO_WHISPER_LANGUAGE = {v: k for k, v in WHISPER_TO_QWEN3_LANGUAGE.items()}

# Short convenience names → HuggingFace model IDs
QWEN3_MODEL_MAPPING = {
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-0.6b": "Qwen/Qwen3-ASR-0.6B",
    # Whisper-style size aliases (map to closest Qwen3 model)
    "large": "Qwen/Qwen3-ASR-1.7B",
    "large-v3": "Qwen/Qwen3-ASR-1.7B",
    "medium": "Qwen/Qwen3-ASR-1.7B",
    "base": "Qwen/Qwen3-ASR-0.6B",
    "small": "Qwen/Qwen3-ASR-0.6B",
    "tiny": "Qwen/Qwen3-ASR-0.6B",
}

_PUNCTUATION_ENDS = set(".!?。！？；;")


class Qwen3ASR(ASRBase):
    """Qwen3-ASR backend with ForcedAligner word-level timestamps."""

    sep = ""  # tokens include leading spaces, like faster-whisper
    SAMPLING_RATE = 16000

    def __init__(self, lan="auto", model_size=None, cache_dir=None,
                 model_dir=None, logfile=sys.stderr, **kwargs):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None if lan == "auto" else lan
        self.model = self.load_model(model_size, cache_dir, model_dir)

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        import torch
        from qwen_asr import Qwen3ASRModel

        if model_dir:
            model_id = model_dir
        elif model_size:
            model_id = QWEN3_MODEL_MAPPING.get(model_size.lower(), model_size)
        else:
            model_id = "Qwen/Qwen3-ASR-1.7B"

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading Qwen3-ASR: {model_id} ({dtype}, {device})")
        model = Qwen3ASRModel.from_pretrained(
            model_id,
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(dtype=dtype, device_map=device),
            dtype=dtype,
            device_map=device,
        )
        logger.info("Qwen3-ASR loaded with ForcedAligner")
        return model

    def _qwen3_language(self) -> Optional[str]:
        if self.original_language is None:
            return None
        return WHISPER_TO_QWEN3_LANGUAGE.get(self.original_language)

    def transcribe(self, audio: np.ndarray, init_prompt: str = ""):
        try:
            results = self.model.transcribe(
                audio=(audio, 16000),
                language=self._qwen3_language(),
                context=init_prompt or "",
                return_time_stamps=True,
            )
        except Exception:
            logger.warning("Qwen3 timestamp alignment failed, falling back to no timestamps", exc_info=True)
            results = self.model.transcribe(
                audio=(audio, 16000),
                language=self._qwen3_language(),
                context=init_prompt or "",
                return_time_stamps=False,
            )
        result = results[0]
        # Stash audio length for timestamp estimation fallback
        result._audio_duration = len(audio) / 16000
        return result

    @staticmethod
    def _detected_language(result) -> Optional[str]:
        """Extract Whisper-style language code from Qwen3 result."""
        lang = getattr(result, 'language', None)
        if lang:
            return QWEN3_TO_WHISPER_LANGUAGE.get(lang, lang.lower())
        return None

    def ts_words(self, result) -> List[ASRToken]:
        detected = self._detected_language(result)
        if result.time_stamps:
            tokens = []
            for i, item in enumerate(result.time_stamps):
                # Prepend space to match faster-whisper convention (tokens carry
                # their own whitespace so ''.join works in Segment.from_tokens)
                text = item.text if i == 0 else " " + item.text
                tokens.append(ASRToken(
                    start=item.start_time, end=item.end_time, text=text,
                    detected_language=detected,
                ))
            return tokens
        # Fallback: estimate timestamps from word count
        if not result.text:
            return []
        words = result.text.split()
        duration = getattr(result, '_audio_duration', 5.0)
        step = duration / max(len(words), 1)
        return [
            ASRToken(
                start=round(i * step, 3), end=round((i + 1) * step, 3),
                text=w if i == 0 else " " + w,
                detected_language=detected,
            )
            for i, w in enumerate(words)
        ]

    def segments_end_ts(self, result) -> List[float]:
        if not result.time_stamps:
            duration = getattr(result, '_audio_duration', 5.0)
            return [duration]
        # Create segment boundaries at punctuation marks
        ends = []
        for item in result.time_stamps:
            if item.text and item.text.rstrip()[-1:] in _PUNCTUATION_ENDS:
                ends.append(item.end_time)
        last_end = result.time_stamps[-1].end_time
        if not ends or ends[-1] != last_end:
            ends.append(last_end)
        return ends

    def use_vad(self):
        return False
