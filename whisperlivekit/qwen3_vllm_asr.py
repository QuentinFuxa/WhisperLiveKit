"""
Qwen3-ASR backend using vLLM's in-process GPU runtime.

This backend does not use vLLM's HTTP or WebSocket APIs. It keeps one vLLM
engine alive for Qwen3-ASR transcription and another one for Qwen3-ForcedAligner
timestamp prediction. Streaming is implemented by re-transcribing the current
audio buffer and committing only aligned words outside the last 250 ms.
"""

from __future__ import annotations

import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from whisperlivekit.timed_objects import ASRToken, Transcript

logger = logging.getLogger(__name__)

DEFAULT_QWEN3_VLLM_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_QWEN3_VLLM_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

QWEN3_VLLM_MODEL_MAPPING = {
    "base": "Qwen/Qwen3-ASR-0.6B",
    "tiny": "Qwen/Qwen3-ASR-0.6B",
    "small": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
    "medium": DEFAULT_QWEN3_VLLM_MODEL,
    "large": DEFAULT_QWEN3_VLLM_MODEL,
    "large-v3": DEFAULT_QWEN3_VLLM_MODEL,
    "qwen3-asr-1.7b": DEFAULT_QWEN3_VLLM_MODEL,
    "qwen3-1.7b": DEFAULT_QWEN3_VLLM_MODEL,
    "1.7b": DEFAULT_QWEN3_VLLM_MODEL,
}

WHISPER_TO_QWEN3_LANGUAGE = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}
QWEN3_TO_WHISPER_LANGUAGE = {v: k for k, v in WHISPER_TO_QWEN3_LANGUAGE.items()}

_ASR_TEXT_TAG = "<asr_text>"
_LANG_RE = re.compile(r"(?:^|\s)language\s+([A-Za-z][A-Za-z -]*)", re.IGNORECASE)
_EOT_RE = re.compile(r"<\|endoftext\|>$")


@dataclass
class _AlignedWord:
    text: str
    start: float
    end: float


def _missing_dependency_error(reason: str) -> ImportError:
    return ImportError(
        "qwen3-vllm requires vLLM with Qwen3-ASR ForcedAligner support. "
        "Install it on a CUDA/Linux host with `uv sync --extra qwen3-vllm` "
        "in an environment separate from `cu129`, or with: "
        "pip install 'whisperlivekit[qwen3-vllm]'. "
        f"Details: {reason}"
    )


def _load_vllm_runtime():
    try:
        from transformers import AutoConfig
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.model_executor.models.qwen3_asr_forced_aligner import (
            Qwen3ASRForcedAlignerForTokenClassification,  # noqa: F401
        )
    except ImportError as exc:
        raise _missing_dependency_error(str(exc)) from exc
    return LLM, SamplingParams, TokensPrompt, AutoConfig


def _resolve_model_path(kwargs: dict) -> str:
    model_path = kwargs.get("vllm_model") or kwargs.get("model_dir") or kwargs.get("model_path")
    if model_path:
        return model_path

    model_size = (kwargs.get("model_size") or "").strip()
    if not model_size:
        return DEFAULT_QWEN3_VLLM_MODEL
    lowered = model_size.lower()
    if "/" in model_size or model_size.startswith((".", "/")):
        return model_size
    return QWEN3_VLLM_MODEL_MAPPING.get(lowered, model_size)


def _qwen3_language(language: Optional[str]) -> Optional[str]:
    if not language or language == "auto":
        return None
    return WHISPER_TO_QWEN3_LANGUAGE.get(language, language)


def _clean_asr_text(text: str) -> str:
    text = _EOT_RE.sub("", text or "").strip()
    if _ASR_TEXT_TAG in text:
        _, text = text.rsplit(_ASR_TEXT_TAG, 1)
    return _EOT_RE.sub("", text).strip()


def _detect_qwen3_language(text: str) -> Optional[str]:
    match = _LANG_RE.search(text or "")
    if not match:
        return None
    language = match.group(1).strip()
    if _ASR_TEXT_TAG in language:
        language = language.split(_ASR_TEXT_TAG, 1)[0].strip()
    return language or None


def _token_id(tokenizer, token: str) -> int:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            return int(token_id)
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Tokenizer could not encode required token {token!r}")
    return int(token_ids[0])


def _is_kept_char(ch: str) -> bool:
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def _clean_align_token(token: str) -> str:
    return "".join(ch for ch in token if _is_kept_char(ch))


def _split_align_words(text: str) -> list[str]:
    words: list[str] = []
    for segment in text.split():
        cleaned = _clean_align_token(segment)
        if not cleaned:
            continue
        buf: list[str] = []
        for ch in cleaned:
            if _is_cjk_char(ch):
                if buf:
                    words.append("".join(buf))
                    buf = []
                words.append(ch)
            else:
                buf.append(ch)
        if buf:
            words.append("".join(buf))
    return words


def _fix_timestamps(values) -> list[float]:
    data = [float(v) for v in values]
    n = len(data)
    if n <= 1:
        return data

    dp = [1] * n
    parent = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    max_idx = dp.index(max(dp))
    lis_indices = []
    while max_idx != -1:
        lis_indices.append(max_idx)
        max_idx = parent[max_idx]
    lis_indices.reverse()

    normal = [False] * n
    for idx in lis_indices:
        normal[idx] = True

    result = data.copy()
    i = 0
    while i < n:
        if normal[i]:
            i += 1
            continue

        j = i
        while j < n and not normal[j]:
            j += 1

        count = j - i
        left = next((result[k] for k in range(i - 1, -1, -1) if normal[k]), None)
        right = next((result[k] for k in range(j, n) if normal[k]), None)

        if count <= 2:
            for k in range(i, j):
                if left is None:
                    result[k] = right if right is not None else result[k]
                elif right is None:
                    result[k] = left
                else:
                    result[k] = left if (k - (i - 1)) <= (j - k) else right
        elif left is not None and right is not None:
            step = (right - left) / (count + 1)
            for k in range(i, j):
                result[k] = left + step * (k - i + 1)
        elif left is not None:
            for k in range(i, j):
                result[k] = left
        elif right is not None:
            for k in range(i, j):
                result[k] = right
        i = j

    return result


def _to_numpy(data):
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    if hasattr(data, "cpu"):
        return data.cpu().numpy()
    return np.asarray(data)


class Qwen3VLLMASR:
    """Model holder for Qwen3-ASR + Qwen3-ForcedAligner through vLLM."""

    sep = ""
    SAMPLING_RATE = 16_000
    backend_choice = "qwen3-vllm"

    def __init__(self, logfile=sys.stderr, **kwargs):
        LLM, SamplingParams, TokensPrompt, AutoConfig = _load_vllm_runtime()

        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None if kwargs.get("lan", "auto") == "auto" else kwargs.get("lan")
        self.model_path = _resolve_model_path(kwargs)
        self.aligner_model_path = kwargs.get("vllm_aligner_model") or DEFAULT_QWEN3_VLLM_ALIGNER_MODEL
        self.max_decode_tokens = int(kwargs.get("max_tokens") or 256)
        self._SamplingParams = SamplingParams
        self._TokensPrompt = TokensPrompt

        tensor_parallel_size = int(kwargs.get("vllm_tensor_parallel_size") or 1)
        gpu_memory_utilization = float(kwargs.get("vllm_gpu_memory_utilization") or 0.45)
        dtype = kwargs.get("vllm_dtype") or "auto"

        common_kwargs = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
        }

        logger.info("Loading Qwen3-ASR vLLM model '%s' ...", self.model_path)
        self.asr_llm = LLM(model=self.model_path, runner="generate", **common_kwargs)
        self.tokenizer = self.asr_llm.get_tokenizer()

        logger.info("Loading Qwen3 ForcedAligner vLLM model '%s' ...", self.aligner_model_path)
        self.aligner_llm = LLM(
            model=self.aligner_model_path,
            runner="pooling",
            hf_overrides={
                "architectures": ["Qwen3ASRForcedAlignerForTokenClassification"],
            },
            **common_kwargs,
        )
        self.aligner_tokenizer = self.aligner_llm.get_tokenizer()
        aligner_config = AutoConfig.from_pretrained(self.aligner_model_path)
        timestamp_token_id = getattr(aligner_config, "timestamp_token_id", None)
        if timestamp_token_id is None:
            timestamp_token_id = _token_id(self.aligner_tokenizer, "<timestamp>")
        self.timestamp_token_id = int(timestamp_token_id)
        self.timestamp_segment_time = float(getattr(aligner_config, "timestamp_segment_time", 0.02))

    def _build_asr_prompt(self, audio: np.ndarray):
        language = _qwen3_language(self.original_language)
        audio_placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"
        if language is None:
            prompt = (
                f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            prompt = (
                f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
                f"<|im_start|>assistant\nlanguage {language}{_ASR_TEXT_TAG}"
            )
        return self._TokensPrompt(
            prompt_token_ids=self.tokenizer.encode(prompt),
            multi_modal_data={"audio": audio.astype(np.float32)},
        )

    def transcribe_text(self, audio: np.ndarray) -> tuple[str, Optional[str]]:
        if len(audio) < 400:
            return "", None

        prompt = self._build_asr_prompt(audio)
        params = self._SamplingParams(temperature=0.0, max_tokens=self.max_decode_tokens)
        outputs = self.asr_llm.generate([prompt], params, use_tqdm=False)
        raw_text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        language = _detect_qwen3_language(raw_text)
        text = _clean_asr_text(raw_text)
        return text, language

    def _build_aligner_prompt(self, words: list[str], audio: np.ndarray):
        text = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
        prompt = "<|audio_start|><|audio_pad|><|audio_end|>" + text
        prompt_token_ids = self.aligner_tokenizer.encode(prompt, add_special_tokens=False)
        return self._TokensPrompt(
            prompt_token_ids=prompt_token_ids,
            multi_modal_data={"audio": audio.astype(np.float32)},
        )

    def align_words(
        self,
        audio: np.ndarray,
        text: str,
        language: Optional[str],
    ) -> list[_AlignedWord]:
        words = _split_align_words(text)
        if not words:
            return []

        prompt = self._build_aligner_prompt(words, audio)
        outputs = self.aligner_llm.encode([prompt], pooling_task="token_classify", use_tqdm=False)
        if not outputs:
            return []

        output = outputs[0]
        data = _to_numpy(output.outputs.data)
        if data.ndim == 3:
            data = data[0]
        pred_ids = np.argmax(data, axis=-1).reshape(-1)
        prompt_token_ids = getattr(output, "prompt_token_ids", None)
        if prompt_token_ids is None:
            prompt_token_ids = prompt.get("prompt_token_ids")
        prompt_token_ids = list(prompt_token_ids or [])
        limit = min(len(prompt_token_ids), len(pred_ids))
        timestamp_values = [
            pred_ids[idx] * self.timestamp_segment_time
            for idx in range(limit)
            if int(prompt_token_ids[idx]) == self.timestamp_token_id
        ]
        timestamp_values = _fix_timestamps(timestamp_values)

        aligned = []
        for idx, word in enumerate(words):
            start_idx = idx * 2
            end_idx = start_idx + 1
            if end_idx >= len(timestamp_values):
                break
            aligned.append(
                _AlignedWord(
                    text=word,
                    start=round(float(timestamp_values[start_idx]), 3),
                    end=round(float(timestamp_values[end_idx]), 3),
                )
            )
        return aligned

    def transcribe_aligned(self, audio: np.ndarray) -> tuple[list[_AlignedWord], Optional[str]]:
        text, detected_language = self.transcribe_text(audio)
        if not text:
            return [], detected_language
        language = detected_language or _qwen3_language(self.original_language) or "English"
        return self.align_words(audio, text, language), detected_language

    def transcribe(self, audio: np.ndarray, init_prompt: str = ""):
        text, _ = self.transcribe_text(audio)
        return text

    def use_vad(self):
        return False


class Qwen3VLLMOnlineProcessor:
    """Batch retranscription processor with ForcedAligner timestamp holdback."""

    SAMPLING_RATE = 16_000
    _HOLDBACK_SECONDS = 0.250
    _MIN_NEW_SECONDS = 1.0
    _MAX_BUFFER_SECONDS = 30.0
    _TRIM_BEFORE_COMMITTED_SECONDS = 2.0
    _COMMITTED_EPSILON = 0.05

    def __init__(self, asr: Qwen3VLLMASR, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.end = 0.0
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer = []

        self._buffer_time_offset = 0.0
        self._last_committed_time = 0.0
        self._current_tokens: list[ASRToken] = []
        self._samples_since_last_inference = 0
        self._min_new_samples = int(self._MIN_NEW_SECONDS * self.SAMPLING_RATE)

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        self.end = audio_stream_end_time
        self.audio_buffer = np.append(self.audio_buffer, audio.astype(np.float32))
        self._samples_since_last_inference += len(audio)

    def _audio_duration(self) -> float:
        return len(self.audio_buffer) / self.SAMPLING_RATE

    def _trim_buffer_if_needed(self):
        duration = self._audio_duration()
        if duration <= self._MAX_BUFFER_SECONDS:
            return

        trim_to_time = self._last_committed_time - self._TRIM_BEFORE_COMMITTED_SECONDS
        if trim_to_time <= self._buffer_time_offset:
            return

        cut_samples = int((trim_to_time - self._buffer_time_offset) * self.SAMPLING_RATE)
        if cut_samples <= 0:
            return

        self.audio_buffer = self.audio_buffer[cut_samples:]
        self._buffer_time_offset += cut_samples / self.SAMPLING_RATE
        self._samples_since_last_inference = min(self._samples_since_last_inference, len(self.audio_buffer))
        self._current_tokens = []

    def _aligned_tokens(self) -> list[ASRToken]:
        aligned_words, detected_language = self.asr.transcribe_aligned(self.audio_buffer)
        tokens: list[ASRToken] = []
        for idx, word in enumerate(aligned_words):
            text = word.text if idx == 0 else " " + word.text
            tokens.append(
                ASRToken(
                    start=self._buffer_time_offset + word.start,
                    end=self._buffer_time_offset + word.end,
                    text=text,
                    detected_language=QWEN3_TO_WHISPER_LANGUAGE.get(detected_language, detected_language.lower())
                    if detected_language
                    else None,
                )
            )
        self._current_tokens = tokens
        return tokens

    def _commit_available(self, flush: bool = False) -> list[ASRToken]:
        if len(self.audio_buffer) < 400:
            return []

        self._trim_buffer_if_needed()
        cached_tokens = self._current_tokens
        tokens = self._aligned_tokens()
        if not tokens and flush and cached_tokens:
            tokens = cached_tokens
            self._current_tokens = cached_tokens
        if not tokens:
            return []

        cutoff = (
            self._buffer_time_offset + self._audio_duration()
            if flush
            else self._buffer_time_offset + self._audio_duration() - self._HOLDBACK_SECONDS
        )
        start_idx = 0
        while (
            start_idx < len(tokens)
            and tokens[start_idx].end <= self._last_committed_time + self._COMMITTED_EPSILON
        ):
            start_idx += 1

        end_idx = start_idx
        while end_idx < len(tokens) and tokens[end_idx].end <= cutoff:
            end_idx += 1

        committed = tokens[start_idx:end_idx]
        if committed:
            self._last_committed_time = committed[-1].end
        return committed

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        try:
            if not is_last and self._samples_since_last_inference < self._min_new_samples:
                return [], self.end
            self._samples_since_last_inference = 0
            return self._commit_available(flush=is_last), self.end
        except Exception as e:
            logger.warning("[qwen3-vllm] process_iter error: %s", e, exc_info=True)
            return [], self.end

    def get_buffer(self) -> Transcript:
        tokens = [
            token
            for token in self._current_tokens
            if token.end > self._last_committed_time + self._COMMITTED_EPSILON
        ]
        return Transcript.from_tokens(tokens=tokens, sep="")

    def _reset_for_next_utterance(self):
        self._buffer_time_offset += self._audio_duration()
        self._last_committed_time = self._buffer_time_offset
        self.audio_buffer = np.array([], dtype=np.float32)
        self._samples_since_last_inference = 0
        self._current_tokens = []

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        tokens = self._commit_available(flush=True)
        logger.info("[qwen3-vllm] start_silence: flushed %d words", len(tokens))
        self._reset_for_next_utterance()
        return tokens, self.end

    def end_silence(self, silence_duration: float, offset: float):
        self._buffer_time_offset += silence_duration
        self._last_committed_time += silence_duration
        self.end += silence_duration

    def new_speaker(self, change_speaker):
        self.start_silence()

    def warmup(self, audio, init_prompt=""):
        return None

    def finish(self) -> Tuple[List[ASRToken], float]:
        tokens = self._commit_available(flush=True)
        logger.info("[qwen3-vllm] finish: flushed %d words", len(tokens))
        return tokens, self.end
