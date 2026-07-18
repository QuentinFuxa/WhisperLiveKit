"""Optional FunASR/SenseVoice adapter for LocalAgreement."""

import math
import re
import sys
import threading
import unicodedata
from typing import Any

from whisperlivekit.config import FUNASR_LANGUAGES
from whisperlivekit.local_agreement.online_asr import OnlineASRProcessor
from whisperlivekit.timed_objects import ASRToken

DEFAULT_FUNASR_MODEL = "iic/SenseVoiceSmall"
_LANGUAGE_TAG = re.compile(r"<\|(?P<language>zh|yue|en|ja|ko)\|>")
_REQUEST_LANGUAGE_KEY = "_wlk_requested_language"
_INSTALL_MESSAGE = (
    "FunASR backend requested but funasr is not installed. Install it with:\n"
    'pip install "whisperlivekit[funasr]"'
)


def _load_funasr_dependencies():
    try:
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
    except ModuleNotFoundError as exc:
        if exc.name and not exc.name.startswith("funasr"):
            raise
        raise ImportError(_INSTALL_MESSAGE) from exc
    return AutoModel, rich_transcription_postprocess


class FunASRASR:
    """Adapt SenseVoice's offline output to WLK's LocalAgreement contract."""

    sep = ""

    def __init__(
        self,
        lan,
        model_size=None,
        cache_dir=None,
        model_dir=None,
        logfile=sys.stderr,
    ):
        del model_size, cache_dir
        auto_model, postprocess = _load_funasr_dependencies()
        self.logfile = logfile
        self.original_language = None if lan == "auto" else lan
        self.transcribe_kargs = {}
        self._postprocess = postprocess
        self._session_lock = threading.RLock()
        self.model_name_or_path = model_dir or DEFAULT_FUNASR_MODEL
        try:
            self.model = auto_model(
                model=self.model_name_or_path,
                trust_remote_code=False,
                disable_update=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FunASR model from {self.model_name_or_path!r}."
            ) from exc

    def transcribe(self, audio, init_prompt=""):
        del init_prompt
        with self._session_lock:
            language = self.original_language or "auto"
            if language not in FUNASR_LANGUAGES:
                supported = ", ".join(sorted(FUNASR_LANGUAGES))
                raise ValueError(
                    f"FunASR SenseVoiceSmall supports only: {supported}."
                )
            result = self.model.generate(
                input=audio,
                cache={},
                language=language,
                use_itn=True,
                output_timestamp=True,
            )
            if (
                isinstance(result, list)
                and len(result) == 1
                and isinstance(result[0], dict)
            ):
                result = [
                    {
                        **result[0],
                        _REQUEST_LANGUAGE_KEY: None
                        if language == "auto"
                        else language,
                    }
                ]
            return result

    @staticmethod
    def _result_item(result: Any):
        if result is None or result == []:
            return None
        if (
            not isinstance(result, list)
            or len(result) != 1
            or not isinstance(result[0], dict)
        ):
            raise ValueError("FunASR must return exactly one result object.")
        return result[0]

    @staticmethod
    def _validated_spans(timestamps, word_count):
        if not isinstance(timestamps, (list, tuple)) or len(timestamps) != word_count:
            timestamp_count = (
                len(timestamps) if isinstance(timestamps, (list, tuple)) else "invalid"
            )
            raise ValueError(
                "FunASR word/timestamp length mismatch: "
                f"{word_count} words and {timestamp_count} timestamps."
            )

        spans = []
        previous_start = 0.0
        previous_end = 0.0
        for index, pair in enumerate(timestamps):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(
                    f"Invalid FunASR timestamp span at word index {index}."
                )
            try:
                start = float(pair[0]) / 1000.0
                end = float(pair[1]) / 1000.0
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Non-numeric FunASR timestamp at word index {index}."
                ) from exc
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError(
                    f"FunASR timestamps must be finite at word index {index}."
                )
            if (
                start < 0
                or end < start
                or start < previous_start
                or end < previous_end
            ):
                raise ValueError(
                    "Invalid or non-monotonic FunASR timestamp span "
                    f"at word index {index}."
                )
            spans.append((start, end))
            previous_start, previous_end = start, end
        return spans

    @staticmethod
    def _is_decoration(text):
        return all(
            char.isspace()
            or unicodedata.category(char)[0] in {"M", "P", "S", "Z"}
            or char == "\u200d"
            for char in text
        )

    def _project_words(self, text, words):
        positions = []
        cursor = 0
        for index, word in enumerate(words):
            if not isinstance(word, str) or not word:
                raise ValueError(f"Invalid FunASR word at word index {index}.")
            start = text.find(word, cursor)
            if start < 0:
                raise ValueError(
                    f"Cannot map FunASR word index {index} into postprocessed text "
                    f"(text length {len(text)}, word length {len(word)})."
                )
            if not self._is_decoration(text[cursor:start]):
                raise ValueError(
                    f"FunASR left unassigned text before word index {index}."
                )
            end = start + len(word)
            positions.append((start, end))
            cursor = end

        if positions and not self._is_decoration(text[cursor:]):
            raise ValueError(
                f"FunASR left unassigned text after word index {len(words) - 1}."
            )

        boundaries = [0]
        for previous, current in zip(positions, positions[1:]):
            gap = text[previous[1] : current[0]]
            trailing_whitespace = len(gap) - len(gap.rstrip())
            boundaries.append(current[0] - trailing_whitespace)
        boundaries.append(len(text))
        pieces = [
            text[boundaries[index] : boundaries[index + 1]]
            for index in range(len(words))
        ]
        if "".join(pieces) != text:
            raise ValueError(
                "FunASR timestamped text reconstruction failed for "
                f"text length {len(text)}."
            )
        return pieces

    def ts_words(self, result):
        item = self._result_item(result)
        if item is None:
            return []

        required_fields = {"text", "words", "timestamp"}
        missing_fields = sorted(required_fields.difference(item))
        if missing_fields:
            raise ValueError(
                "FunASR result missing required field(s): "
                + ", ".join(missing_fields)
                + "."
            )

        raw_text = item["text"]
        if not isinstance(raw_text, str):
            raise ValueError("FunASR result text must be a string.")
        text = self._postprocess(raw_text)
        if not isinstance(text, str):
            raise ValueError("FunASR postprocessed text must be a string.")

        words = item["words"]
        timestamps = item["timestamp"]
        if not isinstance(words, (list, tuple)):
            raise ValueError("FunASR result words must be a list.")
        if not isinstance(timestamps, (list, tuple)):
            raise ValueError("FunASR result timestamp must be a list.")
        if not text and not words and not timestamps:
            return []
        if not words:
            raise ValueError("FunASR returned text without timestamped words.")

        spans = self._validated_spans(timestamps, len(words))
        pieces = self._project_words(text, words)
        match = _LANGUAGE_TAG.search(raw_text)
        tagged_language = match.group("language") if match else None
        if _REQUEST_LANGUAGE_KEY in item:
            detected_language = item[_REQUEST_LANGUAGE_KEY] or tagged_language
        else:
            detected_language = self.original_language or tagged_language
        tokens = [
            ASRToken(start, end, piece, detected_language=detected_language)
            for piece, (start, end) in zip(pieces, spans)
        ]
        if self.sep.join(token.text for token in tokens) != text:
            raise ValueError(
                "FunASR timestamped text reconstruction failed for "
                f"text length {len(text)}."
            )
        return tokens

    def segments_end_ts(self, result):
        tokens = self.ts_words(result)
        return [tokens[-1].end] if tokens else []

    def use_vad(self):
        """Keep WLK's VAC/VAD pipeline authoritative."""
        return None


class FunASROnlineASRProcessor(OnlineASRProcessor):
    """Prevent SenseVoice's committed EOF batch from remaining pending."""

    def finish(self):
        remaining_tokens, final_processed_upto = super().finish()
        self.transcript_buffer.buffer = []
        return remaining_tokens, final_processed_upto
