import logging
import re
import sys
from typing import List, Optional

import numpy as np

from whisperlivekit.local_agreement.backends import ASRBase
from whisperlivekit.timed_objects import ASRToken

logger = logging.getLogger(__name__)


def _patch_transformers_compat():
    """Patch transformers for qwen_asr 0.0.6 + transformers >= 5.3 compatibility."""
    import torch

    # 1. check_model_inputs was removed
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

    # 2. 'default' rope type was removed from ROPE_INIT_FUNCTIONS
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if "default" not in ROPE_INIT_FUNCTIONS:
            def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
                head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
                partial = getattr(config, "partial_rotary_factor", 1.0)
                dim = int(head_dim * partial)
                base = config.rope_theta
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
                return inv_freq, 1.0
            ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    except ImportError:
        pass

    # 3. pad_token_id missing on thinker config
    try:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
            Qwen3ASRThinkerConfig,
        )
        if not hasattr(Qwen3ASRThinkerConfig, "pad_token_id"):
            Qwen3ASRThinkerConfig.pad_token_id = None
    except ImportError:
        pass

    # 4. fix_mistral_regex kwarg not accepted by newer transformers
    try:
        from transformers.models.auto import processing_auto
        _orig_ap_from_pretrained = processing_auto.AutoProcessor.from_pretrained.__func__

        @classmethod
        def _patched_ap_from_pretrained(cls, *args, **kwargs):
            kwargs.pop("fix_mistral_regex", None)
            return _orig_ap_from_pretrained(cls, *args, **kwargs)

        processing_auto.AutoProcessor.from_pretrained = _patched_ap_from_pretrained
    except Exception:
        pass

    # 5. compute_default_rope_parameters missing on RotaryEmbedding
    try:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRThinkerTextRotaryEmbedding,
        )
        if not hasattr(Qwen3ASRThinkerTextRotaryEmbedding, "compute_default_rope_parameters"):
            @staticmethod
            def _rope_params(config=None, device=None, seq_len=None, **kwargs):
                head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
                partial = getattr(config, "partial_rotary_factor", 1.0)
                dim = int(head_dim * partial)
                base = config.rope_theta
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
                return inv_freq, 1.0
            Qwen3ASRThinkerTextRotaryEmbedding.compute_default_rope_parameters = _rope_params
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
# Qwen3 raw output starts with "language <Name>" metadata before <asr_text> tag.
# When the tag is missing (silence/noise), this metadata leaks as transcription text.
_GARBAGE_RE = re.compile(r"^language\s+\S+$", re.IGNORECASE)


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

        if torch.cuda.is_available():
            dtype, device = torch.bfloat16, "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dtype, device = torch.float32, "mps"
        else:
            dtype, device = torch.float32, "cpu"

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
        logger.info(
            "Qwen3 result: language=%r text=%r ts=%s",
            result.language, result.text[:80] if result.text else "",
            bool(result.time_stamps),
        )
        return result

    @staticmethod
    def _detected_language(result) -> Optional[str]:
        """Extract Whisper-style language code from Qwen3 result."""
        lang = getattr(result, 'language', None)
        if not lang or lang.lower() == "none":
            return None
        # merge_languages may return comma-separated; take the first
        first = lang.split(",")[0].strip()
        if not first or first.lower() == "none":
            return None
        return QWEN3_TO_WHISPER_LANGUAGE.get(first, first.lower())

    def ts_words(self, result) -> List[ASRToken]:
        # Filter garbage model output (e.g. "language None" for silence/noise)
        text = (result.text or "").strip()
        if not text or _GARBAGE_RE.match(text):
            if text:
                logger.info("Filtered garbage Qwen3 output: %r", text)
            return []
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
