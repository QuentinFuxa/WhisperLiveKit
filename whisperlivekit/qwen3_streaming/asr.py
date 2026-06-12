"""Shared model holder for the qwen3-streaming backend.

Loads the pretrained Qwen3-ASR model once through plain HF Transformers (no
vLLM), wrapped with the bounded-recompute streaming contract from
``model.py``. Sessions get their own ``SegmentedCachedFullHypothesisStreamer``
via :meth:`Qwen3StreamingASR.build_streamer`; GPU decode calls are serialized
across sessions through :attr:`decode_lock`.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

from whisperlivekit.qwen3_vllm_asr import (
    QWEN3_TO_WHISPER_LANGUAGE,
    WHISPER_TO_QWEN3_LANGUAGE,
)

from .features import StreamingMelExtractor
from .model import (
    Qwen3ASRRealtimeQwenAudioSurgeryModel,
    _register_qwen3_asr_transformers,
)
from .model_config import RealtimeAudioConfig
from .streamer import (
    CachedFullHypothesisConfig,
    SegmentedCachedFullHypothesisStreamer,
    added_token_id,
    qwen_asr_prompt_text,
)

logger = logging.getLogger(__name__)

DEFAULT_QWEN3_STREAMING_MODEL = "Qwen/Qwen3-ASR-0.6B"

QWEN3_STREAMING_MODEL_MAPPING = {
    "base": DEFAULT_QWEN3_STREAMING_MODEL,
    "tiny": DEFAULT_QWEN3_STREAMING_MODEL,
    "small": DEFAULT_QWEN3_STREAMING_MODEL,
    "qwen3-asr-0.6b": DEFAULT_QWEN3_STREAMING_MODEL,
    "qwen3-0.6b": DEFAULT_QWEN3_STREAMING_MODEL,
    "0.6b": DEFAULT_QWEN3_STREAMING_MODEL,
    "medium": "Qwen/Qwen3-ASR-1.7B",
    "large": "Qwen/Qwen3-ASR-1.7B",
    "large-v3": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "1.7b": "Qwen/Qwen3-ASR-1.7B",
}


def _missing_dependency_error(reason: str) -> ImportError:
    return ImportError(
        "qwen3-streaming requires torch, transformers and the qwen-asr package. "
        "Install with: pip install 'whisperlivekit[qwen3-streaming]'. "
        f"Details: {reason}"
    )


def _resolve_model_id(kwargs: dict) -> str:
    model_path = kwargs.get("model_dir") or kwargs.get("model_path")
    if model_path:
        return model_path
    model_size = (kwargs.get("model_size") or "").strip()
    if not model_size:
        return DEFAULT_QWEN3_STREAMING_MODEL
    if "/" in model_size or model_size.startswith("."):
        return model_size
    return QWEN3_STREAMING_MODEL_MAPPING.get(model_size.lower(), model_size)


class Qwen3StreamingASR:
    """Qwen3-ASR streaming model holder (HF Transformers, CUDA/MPS/CPU)."""

    sep = ""
    SAMPLING_RATE = 16_000

    def __init__(self, logfile=sys.stderr, **kwargs):
        try:
            import torch
            from transformers import AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise _missing_dependency_error(str(exc)) from exc

        self.logfile = logfile
        self.backend_choice = "qwen3-streaming"
        self.tokenizer = None  # sentence tokenizer slot, unused by this backend
        self.confidence_validation = False
        self.buffer_trimming = "segment"
        self.buffer_trimming_sec = 15.0

        lan = kwargs.get("lan", "auto")
        if not lan or lan == "auto":
            raise ValueError(
                "qwen3-streaming requires an explicit transcription language "
                "(e.g. --language en). Automatic language detection flips "
                "accented audio to the wrong language mid-stream."
            )
        self.original_language = lan

        # Streaming settings (defaults = validated left12/seg200 operating point)
        self.chunk_sec = float(kwargs.get("qwen3_streaming_chunk_sec", 2.0))
        self.left_context_sec = float(kwargs.get("qwen3_streaming_left_context_sec", 12.0))
        self.right_context_ms = int(kwargs.get("qwen3_streaming_right_context_ms", 640))
        self.segment_max_steps = int(kwargs.get("qwen3_streaming_segment_max_steps", 200))
        self.segment_keep_tail_steps = int(
            kwargs.get("qwen3_streaming_segment_keep_tail_steps", 0)
        )
        self.hold_back_words = int(kwargs.get("qwen3_streaming_hold_back_words", 6))
        self.stable_iterations = int(kwargs.get("qwen3_streaming_stable_iterations", 2))
        self.max_new_tokens = int(kwargs.get("qwen3_streaming_max_new_tokens", 256))
        self.base_context = str(kwargs.get("qwen3_streaming_context", "") or "")
        self.prompt_context_words = int(
            kwargs.get("qwen3_streaming_prompt_context_words", 0)
        )

        # Audio backend: "windowed" (bounded-recompute window, default) or
        # "causal" (append-only causal-KV encoder with the fine-tuned tower).
        self.audio_backend = str(
            kwargs.get("qwen3_streaming_audio_backend", "windowed") or "windowed"
        )
        if self.audio_backend not in ("windowed", "causal"):
            raise ValueError(
                "qwen3_streaming_audio_backend must be 'windowed' or 'causal', "
                f"got {self.audio_backend!r}"
            )
        self.tower_checkpoint = str(
            kwargs.get("qwen3_streaming_tower_checkpoint", "") or ""
        )
        self.block_frames = int(kwargs.get("qwen3_streaming_block_frames", 192))

        # Decode/rollover policy. The windowed defaults stay at the validated
        # operating point; causal mode derives the run-D operating point
        # (RUNS.md 2026-06-12: WER 0.1807 human-whisper, RTF 0.107).
        self.decoder_rolling_kv = False
        self.speculative_draft = False
        self.repetition_penalty = 1.0
        self.no_repeat_ngram_size = 0
        self.segment_punct_rollover = False
        self.segment_punct_min_steps = 150
        self.segment_roll_before_generate = False
        self.reset_encoder_on_rollover = False
        if self.audio_backend == "causal":
            if not self.tower_checkpoint:
                raise ValueError(
                    "the causal audio backend requires "
                    "--qwen3-streaming-tower-checkpoint (local .pt/.safetensors "
                    "file, directory, or Hugging Face repo id)"
                )
            if self.left_context_sec == 12.0:
                # Windowed CLI default; the causal tower was trained at 15 s.
                logger.info(
                    "qwen3-streaming causal: using the trained 15 s left context"
                )
                self.left_context_sec = 15.0
            if self.right_context_ms not in (0, 640):
                logger.warning(
                    "qwen3-streaming causal: ignoring right context %d ms "
                    "(the causal encoder has none)",
                    self.right_context_ms,
                )
            self.right_context_ms = 0
            self.decoder_rolling_kv = True
            self.speculative_draft = True
            self.repetition_penalty = 1.15
            self.no_repeat_ngram_size = 3
            self.segment_punct_rollover = True
            self.segment_roll_before_generate = True
            self.reset_encoder_on_rollover = True

        device_setting = str(kwargs.get("qwen3_streaming_device", "auto"))
        dtype_setting = str(kwargs.get("qwen3_streaming_dtype", "auto"))
        self.device, self.dtype = self._resolve_device_dtype(
            torch, device_setting, dtype_setting
        )

        model_id = _resolve_model_id(kwargs)
        self.model_id = model_id

        t = time.time()
        logger.info(
            "Loading Qwen3-ASR streaming model '%s' on %s (%s)...",
            model_id,
            self.device,
            self.dtype,
        )
        _register_qwen3_asr_transformers()
        from transformers import AutoConfig

        self.qwen_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.feature_extractor = AutoProcessor.from_pretrained(model_id).feature_extractor
        hf_config = AutoConfig.from_pretrained(model_id)
        text_config = hf_config.thinker_config.text_config

        causal = self.audio_backend == "causal"
        self.audio_config = RealtimeAudioConfig(
            d_model=int(text_config.hidden_size),
            qwen_audio_left_context_sec=self.left_context_sec,
            qwen_audio_right_context_ms=self.right_context_ms,
            qwen_audio_block_bidirectional=causal,
            qwen_audio_block_frames=(self.block_frames if causal else 0),
        )
        if causal:
            from .causal import Qwen3ASRRealtimeQwenAudioCausalModel

            model_cls = Qwen3ASRRealtimeQwenAudioCausalModel
        else:
            model_cls = Qwen3ASRRealtimeQwenAudioSurgeryModel
        self.model = (
            model_cls.from_qwen_pretrained(
                model_id,
                config=self.audio_config,
                bos_token_id=(
                    int(self.qwen_tokenizer.eos_token_id)
                    if self.qwen_tokenizer.eos_token_id is not None
                    else 0
                ),
                wait_token_id=None,
                dtype=self.dtype,
                device_map="cpu",
            )
            .to(self.device)
            .eval()
        )
        if causal:
            from .causal import load_tower_checkpoint, resolve_tower_checkpoint

            checkpoint_path = resolve_tower_checkpoint(self.tower_checkpoint)
            metadata = load_tower_checkpoint(self.model, checkpoint_path)
            logger.info(
                "qwen3-streaming causal: loaded tower checkpoint %s%s",
                checkpoint_path,
                f" (step {metadata['step']})" if metadata.get("step") else "",
            )
        logger.info("Qwen3-ASR streaming model loaded in %.2fs", time.time() - t)

        # Token ids shared by all sessions
        self.wait_token_id = added_token_id(self.qwen_tokenizer, "[P]")
        self.word_start_token_id = added_token_id(self.qwen_tokenizer, "[W]")
        self.eos_token_id = (
            None
            if self.qwen_tokenizer.eos_token_id is None
            else int(self.qwen_tokenizer.eos_token_id)
        )
        self.audio_placeholder_token_id = int(
            self.qwen_tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        )
        suppress = [self.wait_token_id, self.word_start_token_id]
        for token in ("<|audio_start|>", "<|audio_pad|>", "<|audio_end|>", "<|im_start|>"):
            token_id = self.qwen_tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                suppress.append(int(token_id))
        self.suppress_token_ids = tuple(suppress)

        self.decode_lock = threading.Lock()

    @staticmethod
    def _resolve_device_dtype(torch, device_setting: str, dtype_setting: str):
        if device_setting == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = device_setting
        if dtype_setting == "auto":
            if device == "cuda":
                dtype = torch.bfloat16
            elif device == "mps":
                # bf16 support on MPS is incomplete/slow for several kernels
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = getattr(torch, dtype_setting)
        return torch.device(device), dtype

    def qwen_language(self, whisper_language: str | None) -> str:
        """Map a whisper-style code ('en') to the Qwen prompt language."""
        lang = whisper_language or self.original_language
        if not lang or lang == "auto":
            logger.warning(
                "qwen3-streaming: per-session language 'auto' is not supported; "
                "falling back to server language '%s'",
                self.original_language,
            )
            lang = self.original_language
        return WHISPER_TO_QWEN3_LANGUAGE.get(lang, lang)

    def whisper_language(self, qwen_language: str) -> str | None:
        return QWEN3_TO_WHISPER_LANGUAGE.get(qwen_language)

    def build_streamer(
        self, whisper_language: str | None = None
    ) -> SegmentedCachedFullHypothesisStreamer:
        qwen_lang = self.qwen_language(whisper_language)
        prompt_prefix_template = self.qwen_tokenizer.encode(
            qwen_asr_prompt_text(context=self.base_context, language=qwen_lang),
            add_special_tokens=False,
        )
        config = CachedFullHypothesisConfig(
            wait_token_id=self.wait_token_id,
            word_start_token_id=self.word_start_token_id,
            eos_token_id=self.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            hold_back_words=self.hold_back_words,
            stable_iterations=self.stable_iterations,
            commit_mode="word",
            suppress_token_ids=self.suppress_token_ids,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            prompt_prefix_template=prompt_prefix_template,
            audio_placeholder_token_id=self.audio_placeholder_token_id,
            decoder_rolling_kv=self.decoder_rolling_kv,
            speculative_draft=self.speculative_draft,
        )
        return SegmentedCachedFullHypothesisStreamer(
            self.model,
            self.qwen_tokenizer,
            config,
            segment_max_cached_steps=self.segment_max_steps,
            segment_keep_tail_steps=self.segment_keep_tail_steps,
            segment_finalize_mode="latest",
            segment_prompt_context_words=self.prompt_context_words,
            segment_prompt_base_context=self.base_context,
            segment_prompt_language=qwen_lang,
            segment_punct_rollover=self.segment_punct_rollover,
            segment_punct_min_steps=self.segment_punct_min_steps,
            segment_roll_before_generate=self.segment_roll_before_generate,
            reset_encoder_on_rollover=self.reset_encoder_on_rollover,
        )

    def new_mel_extractor(self) -> StreamingMelExtractor:
        return StreamingMelExtractor(self.feature_extractor, sample_rate=self.SAMPLING_RATE)

    @property
    def right_context_frames(self) -> int:
        return int(getattr(self.model.audio_encoder, "right_context_frames", 0))

    @property
    def n_mels(self) -> int:
        return int(self.audio_config.n_mels)

    def use_vad(self) -> bool:
        return False

    def transcribe(self, audio, init_prompt=""):
        # Streaming-only backend; sessions go through the online processor.
        pass
