"""
Qwen3-ASR backend using vllm-metal's in-process STT runtime.

This backend does not use vLLM's HTTP or WebSocket APIs. It loads the
vllm-metal MLX model directly, re-transcribes the current audio buffer, and
streams by committing every word except the last two.
"""

from __future__ import annotations

import logging
import platform
import queue
import sys
import threading
import time
from typing import Any, Callable, List, Tuple

import numpy as np

from whisperlivekit.timed_objects import ASRToken, Transcript

logger = logging.getLogger(__name__)

DEFAULT_QWEN3_VLLM_METAL_MODEL = "Qwen/Qwen3-ASR-0.6B"
QWEN3_VLLM_METAL_1_7B_MODEL = "Qwen/Qwen3-ASR-1.7B"

QWEN3_VLLM_METAL_MODEL_MAPPING = {
    "base": DEFAULT_QWEN3_VLLM_METAL_MODEL,
    "tiny": DEFAULT_QWEN3_VLLM_METAL_MODEL,
    "small": DEFAULT_QWEN3_VLLM_METAL_MODEL,
    "qwen3-asr-0.6b": DEFAULT_QWEN3_VLLM_METAL_MODEL,
    "qwen3-0.6b": DEFAULT_QWEN3_VLLM_METAL_MODEL,
    "0.6b": DEFAULT_QWEN3_VLLM_METAL_MODEL,
    "qwen3-asr-1.7b": QWEN3_VLLM_METAL_1_7B_MODEL,
    "qwen3-1.7b": QWEN3_VLLM_METAL_1_7B_MODEL,
    "1.7b": QWEN3_VLLM_METAL_1_7B_MODEL,
}

_UNSUPPORTED_QWEN3_VLLM_METAL_ALIASES = {
    "medium",
    "large",
    "large-v3",
}

_SENTENCE_ENDINGS = (".", "!", "?")

_VLLM_METAL_INSTALL_HINT = (
    "Install vLLM first with the official vllm-metal install script, then "
    "install the vllm-metal STT extra. The WhisperLiveKit extra only adds "
    "the vllm-metal wheel on supported Apple Silicon/Python 3.12 builds."
)


class _Qwen3MetalWorker:
    """Run all MLX/vllm-metal work on one thread."""

    def __init__(self):
        self._tasks: queue.Queue[Any] = queue.Queue()
        self._ready = threading.Event()
        self._init_error: BaseException | None = None
        self._thread_id: int | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="qwen3-vllm-metal-worker",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._init_error:
            raise self._init_error

    def _run(self) -> None:
        self._thread_id = threading.get_ident()
        try:
            import mlx.core as mx

            mx.set_default_device(mx.gpu)
        except BaseException as exc:
            self._init_error = exc
        finally:
            self._ready.set()

        if self._init_error:
            return

        while True:
            task = self._tasks.get()
            if task is None:
                return

            fn, args, kwargs, result_queue = task
            try:
                result_queue.put((True, fn(*args, **kwargs)))
            except BaseException as exc:
                result_queue.put((False, exc))

    def call(self, fn: Callable, *args, **kwargs):
        if threading.get_ident() == self._thread_id:
            return fn(*args, **kwargs)

        result_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._tasks.put((fn, args, kwargs, result_queue))
        ok, value = result_queue.get()
        if ok:
            return value
        raise value


def _missing_dependency_error(exc: ImportError | None = None) -> ImportError:
    missing_name = getattr(exc, "name", "") if exc is not None else ""

    if missing_name == "vllm" or missing_name.startswith("vllm."):
        return ImportError(
            "qwen3-vllm-metal found vllm-metal STT, but the vLLM CPU package "
            f"is missing. {_VLLM_METAL_INSTALL_HINT}"
        )

    if missing_name == "vllm_metal" or missing_name.startswith("vllm_metal."):
        return ImportError(
            "qwen3-vllm-metal requires vllm-metal with STT support. "
            "On Apple Silicon with Python 3.12, install "
            "`whisperlivekit[qwen3-vllm-metal]` or follow the official "
            f"vllm-metal install instructions. {_VLLM_METAL_INSTALL_HINT}"
        )

    if missing_name == "mlx" or missing_name.startswith("mlx."):
        return ImportError(
            "qwen3-vllm-metal requires MLX on Apple Silicon. "
            "Use Darwin arm64 with a supported vllm-metal installation."
        )

    return ImportError(
        "qwen3-vllm-metal requires vllm-metal STT on Apple Silicon. "
        f"{_VLLM_METAL_INSTALL_HINT}"
    )


def _ensure_supported_platform():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ImportError(
            "qwen3-vllm-metal requires Apple Silicon (Darwin arm64) because "
            "vllm-metal runs on MLX/Metal."
        )


def _resolve_model_path(kwargs: dict) -> str:
    model_path = kwargs.get("model_dir") or kwargs.get("model_path")
    if model_path:
        return model_path

    model_size = (kwargs.get("model_size") or "").strip()
    if not model_size:
        return DEFAULT_QWEN3_VLLM_METAL_MODEL

    lowered = model_size.lower()
    if "/" in model_size or model_size.startswith((".", "/")):
        return model_size
    if lowered in QWEN3_VLLM_METAL_MODEL_MAPPING:
        return QWEN3_VLLM_METAL_MODEL_MAPPING[lowered]
    if lowered in _UNSUPPORTED_QWEN3_VLLM_METAL_ALIASES:
        raise ValueError(
            "qwen3-vllm-metal supports Qwen3-ASR 0.6B and 1.7B; "
            f"got unsupported alias {model_size!r}."
        )
    return model_size


def _resolve_mlx_dtype(mx, kwargs: dict):
    """Resolve shared vLLM dtype names to MLX dtype objects."""
    explicit_dtype = kwargs.get("dtype")
    if explicit_dtype is not None:
        return explicit_dtype

    dtype_name = kwargs.get("vllm_dtype") or "auto"
    if dtype_name == "auto":
        return mx.float16
    if not isinstance(dtype_name, str):
        return dtype_name

    dtype_map = {
        "float16": mx.float16,
        "fp16": mx.float16,
        "bfloat16": mx.bfloat16,
        "bf16": mx.bfloat16,
        "float32": mx.float32,
        "fp32": mx.float32,
    }
    try:
        return dtype_map[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(
            "qwen3-vllm-metal vllm_dtype must be one of auto, float16, "
            f"bfloat16, or float32; got {dtype_name!r}"
        ) from exc


def _token_id(tokenizer, token: str) -> int:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None:
            return int(token_id)
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Tokenizer could not encode required token {token!r}")
    return int(token_ids[0])


class Qwen3VLLMMetalASR:
    """Model holder for vllm-metal Qwen3-ASR."""

    sep = ""
    SAMPLING_RATE = 16_000
    backend_choice = "qwen3-vllm-metal"

    def __init__(self, logfile=sys.stderr, **kwargs):
        _ensure_supported_platform()

        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None
        self.tokenizer = None
        self.holdback_words = int(
            kwargs.get("holdback_words")
            if kwargs.get("holdback_words") is not None
            else Qwen3VLLMMetalOnlineProcessor._HOLDBACK_WORDS
        )
        if self.holdback_words < 0:
            raise ValueError("holdback_words must be >= 0")
        self.trim_sentence_buffer = bool(kwargs.get("trim_sentence_buffer", True))
        self.min_chunk_size = float(
            kwargs.get("min_chunk_size")
            if kwargs.get("min_chunk_size") is not None
            else 1.0
        )
        if self.min_chunk_size < 0:
            raise ValueError("min_chunk_size must be >= 0")

        self._worker = _Qwen3MetalWorker()
        self._worker.call(self._init_model, kwargs)

    def _init_model(self, kwargs: dict) -> None:
        try:
            import mlx.core as mx
            from vllm_metal.stt.loader import load_model
            from vllm_metal.stt.qwen3_asr.adapter import Qwen3ASRRuntimeAdapter
            from vllm_metal.stt.qwen3_asr.transcriber import Qwen3ASRTranscriber
        except ImportError as exc:
            raise _missing_dependency_error(exc) from exc

        self._post_process_output = Qwen3ASRTranscriber.post_process_output

        model_path = _resolve_model_path(kwargs)
        dtype = _resolve_mlx_dtype(mx, kwargs)

        t0 = time.time()
        logger.info("Loading Qwen3 vllm-metal model '%s' ...", model_path)
        self.model = load_model(model_path, dtype=dtype)
        self.adapter = Qwen3ASRRuntimeAdapter(self.model, model_path)
        self.adapter.warm_up()
        self.tokenizer = self.adapter.transcriber.tokenizer
        logger.info("Qwen3 vllm-metal model loaded in %.2fs", time.time() - t0)

    def _build_prompt_token_ids(self, n_audio_tokens: int) -> list[int]:
        tokenizer = self.tokenizer

        prompt = []
        prompt.extend(tokenizer.encode("<|im_start|>", add_special_tokens=False))
        prompt.extend(tokenizer.encode("system\n", add_special_tokens=False))
        prompt.extend(tokenizer.encode("<|im_end|>\n", add_special_tokens=False))
        prompt.extend(tokenizer.encode("<|im_start|>", add_special_tokens=False))
        prompt.extend(tokenizer.encode("user\n", add_special_tokens=False))
        prompt.append(_token_id(tokenizer, "<|audio_start|>"))
        prompt.extend([self.model.config.audio_token_id] * n_audio_tokens)
        prompt.append(_token_id(tokenizer, "<|audio_end|>"))
        prompt.extend(tokenizer.encode("<|im_end|>\n", add_special_tokens=False))
        prompt.extend(tokenizer.encode("<|im_start|>", add_special_tokens=False))
        prompt.extend(tokenizer.encode("assistant\n", add_special_tokens=False))
        return prompt

    def _decode_output_tokens(self, output_tokens: list[int]) -> str:
        text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return self._post_process_output(text).strip()

    def _transcribe_text(self, audio: np.ndarray) -> str:
        """Transcribe raw 16 kHz mono float PCM and return cleaned text."""
        if len(audio) < 400:
            return ""

        try:
            from vllm_metal.stt.audio import log_mel_spectrogram
        except ImportError as exc:
            raise _missing_dependency_error(exc) from exc

        mel = log_mel_spectrogram(audio.astype(np.float32), n_mels=128)
        audio_features = self.adapter.extract_audio_features(mel)
        n_audio_tokens = int(audio_features.shape[0])
        prompt_ids = self._build_prompt_token_ids(n_audio_tokens)
        output_tokens = self.adapter.decode_tokens(audio_features, prompt_ids)
        return self._decode_output_tokens(output_tokens)

    def transcribe_text(self, audio: np.ndarray) -> str:
        return self._worker.call(self._transcribe_text, audio)

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> str:
        return self.transcribe_text(audio)

    def use_vad(self):
        return False


class Qwen3VLLMMetalOnlineProcessor:
    """Batch processor committing the current hypothesis except trailing words."""

    SAMPLING_RATE = 16_000
    _HOLDBACK_WORDS = 2

    def __init__(self, asr: Qwen3VLLMMetalASR, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.holdback_words = getattr(asr, "holdback_words", self._HOLDBACK_WORDS)
        self.trim_sentence_buffer = getattr(asr, "trim_sentence_buffer", True)
        self.end = 0.0
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer = []

        self._buffer_time_offset = 0.0
        self._n_committed_words = 0
        self._current_words: list[str] = []
        self._current_text = ""
        self._samples_since_last_inference = 0
        self._min_new_samples = max(
            1,
            int(getattr(asr, "min_chunk_size", 1.0) * self.SAMPLING_RATE),
        )

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        self.end = audio_stream_end_time
        self.audio_buffer = np.append(self.audio_buffer, audio)
        self._samples_since_last_inference += len(audio)

    def _transcribe_words(self) -> list[str]:
        text = self.asr.transcribe_text(self.audio_buffer)
        self._current_text = text
        words = text.split()
        self._current_words = words
        return words

    def _time_for_word(self, word_idx: int, n_words_total: int) -> Tuple[float, float]:
        duration = max(len(self.audio_buffer) / self.SAMPLING_RATE, 0.001)
        n_total = max(n_words_total, 1)
        start = self._buffer_time_offset + (word_idx / n_total) * duration
        end = self._buffer_time_offset + ((word_idx + 1) / n_total) * duration
        return start, end

    def _tokens_for_range(
        self,
        words: list[str],
        start_idx: int,
        end_idx: int,
    ) -> List[ASRToken]:
        tokens: List[ASRToken] = []
        n_total = len(words)
        for idx in range(start_idx, end_idx):
            start, end = self._time_for_word(idx, n_total)
            text = words[idx] if idx == 0 else " " + words[idx]
            tokens.append(ASRToken(start=start, end=end, text=text))
        return tokens

    @staticmethod
    def _sentence_boundary_before(words: list[str], committed_upto: int) -> int | None:
        for idx in range(min(committed_upto, len(words)) - 1, -1, -1):
            if words[idx].rstrip().endswith(_SENTENCE_ENDINGS):
                return idx
        return None

    def _trim_committed_sentence(self, words: list[str]) -> None:
        if not self.trim_sentence_buffer:
            return

        boundary_idx = self._sentence_boundary_before(words, self._n_committed_words)
        if boundary_idx is None:
            return

        _, trim_end = self._time_for_word(boundary_idx, len(words))
        trim_samples = int((trim_end - self._buffer_time_offset) * self.SAMPLING_RATE)
        trim_samples = min(max(trim_samples, 0), len(self.audio_buffer))
        if trim_samples <= 0:
            return

        trimmed_words = boundary_idx + 1
        self.audio_buffer = self.audio_buffer[trim_samples:]
        self._buffer_time_offset += trim_samples / self.SAMPLING_RATE
        self._samples_since_last_inference = min(
            self._samples_since_last_inference,
            len(self.audio_buffer),
        )
        self._n_committed_words = max(0, self._n_committed_words - trimmed_words)
        self._current_words = words[trimmed_words:]
        self._current_text = " ".join(self._current_words)

    def _commit_available(self, flush: bool = False) -> List[ASRToken]:
        words = self._transcribe_words()
        if flush:
            commit_upto = len(words)
        else:
            commit_upto = max(len(words) - self.holdback_words, 0)
        if commit_upto <= self._n_committed_words:
            return []

        tokens = self._tokens_for_range(words, self._n_committed_words, commit_upto)
        self._n_committed_words = commit_upto
        self._trim_committed_sentence(words)
        return tokens

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        try:
            if (
                not is_last
                and self._samples_since_last_inference < self._min_new_samples
            ):
                return [], self.end
            self._samples_since_last_inference = 0
            return self._commit_available(flush=is_last), self.end
        except Exception as e:
            logger.warning("[qwen3-vllm-metal] process_iter error: %s", e, exc_info=True)
            return [], self.end

    def get_buffer(self) -> Transcript:
        if not self._current_words or self._n_committed_words >= len(self._current_words):
            return Transcript(start=None, end=None, text="")

        words = self._current_words[self._n_committed_words:]
        start, _ = self._time_for_word(self._n_committed_words, len(self._current_words))
        _, end = self._time_for_word(len(self._current_words) - 1, len(self._current_words))
        return Transcript(start=start, end=end, text=" ".join(words))

    def _reset_for_next_utterance(self):
        self._buffer_time_offset += len(self.audio_buffer) / self.SAMPLING_RATE
        self.audio_buffer = np.array([], dtype=np.float32)
        self._samples_since_last_inference = 0
        self._n_committed_words = 0
        self._current_words = []
        self._current_text = ""

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        words = self._commit_available(flush=True)
        logger.info("[qwen3-vllm-metal] start_silence: flushed %d words", len(words))
        self._reset_for_next_utterance()
        return words, self.end

    def end_silence(self, silence_duration: float, offset: float):
        self._buffer_time_offset += silence_duration
        self.end += silence_duration

    def new_speaker(self, change_speaker):
        self.start_silence()

    def warmup(self, audio, init_prompt=""):
        return None

    def finish(self) -> Tuple[List[ASRToken], float]:
        words = self._commit_available(flush=True)
        logger.info("[qwen3-vllm-metal] finish: flushed %d words", len(words))
        return words, self.end
