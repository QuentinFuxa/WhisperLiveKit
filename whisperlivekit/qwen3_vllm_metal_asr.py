"""
Qwen3-ASR backend using vllm-metal's in-process STT runtime.

This backend does not use vLLM's HTTP or WebSocket APIs. It loads the
vllm-metal MLX model directly, re-transcribes the current audio buffer, and
streams by committing every word except the last two.
"""

from __future__ import annotations

import logging
import platform
import sys
import time
from typing import List, Tuple

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

_MROPE_SECTION = [24, 20, 20]


def _missing_dependency_error() -> ImportError:
    return ImportError(
        "qwen3-vllm-metal requires vllm-metal STT on Apple Silicon. "
        "Install it with: pip install 'whisperlivekit[qwen3-vllm-metal]'. "
        "If that still fails, ensure your vllm-metal build provides "
        "`vllm_metal.stt`."
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


def _token_id(tokenizer, token: str) -> int:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None:
            return int(token_id)
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Tokenizer could not encode required token {token!r}")
    return int(token_ids[0])


class _InterleavedMRoPE:
    """Qwen3-ASR text decoder MRoPE.

    vllm-metal's STT adapter currently exposes a simplified Qwen3 text decode
    path. Qwen3-ASR needs the official interleaved 3D MRoPE assignment, so the
    backend supplies it locally while still using vllm-metal for loading and
    audio feature extraction.
    """

    def __init__(self, head_dim: int, base: float, mrope_section: list[int] | None):
        import mlx.core as mx

        self.half_dim = head_dim // 2
        self.mrope_section = mrope_section or list(_MROPE_SECTION)
        self._inv_freq = 1.0 / (
            base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        )

        masks = []
        for dim, offset in enumerate((1, 2), start=1):
            stop = min(self.mrope_section[dim] * 3, self.half_dim)
            indices = np.arange(offset, stop, 3, dtype=np.int32)
            mask = np.zeros(self.half_dim, dtype=bool)
            mask[indices] = True
            masks.append(mx.array(mask[None, None, :]))
        self._overwrite_masks = masks

    def __call__(self, position_ids, dtype):
        import mlx.core as mx

        pos = position_ids.astype(mx.float32).transpose(1, 0, 2)[..., None]
        freqs = pos * self._inv_freq[None, None, None, :]
        freqs_t = freqs[0]
        for dim, mask in enumerate(self._overwrite_masks, start=1):
            freqs_t = mx.where(mask, freqs[dim], freqs_t)

        emb = mx.concatenate([freqs_t, freqs_t], axis=-1)
        return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def _rotate_half(x):
    import mlx.core as mx

    mid = x.shape[-1] // 2
    return mx.concatenate([-x[..., mid:], x[..., :mid]], axis=-1)


def _causal_mask(seq_len: int, dtype):
    import mlx.core as mx

    mask = mx.full((seq_len, seq_len), -1e9, dtype=dtype)
    return mx.triu(mask, k=1)[None, None, :, :]


def _mrope_section_from_config(text_config) -> list[int]:
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        section = rope_scaling.get("mrope_section")
        if section:
            return list(section)
    return list(_MROPE_SECTION)


class Qwen3VLLMMetalASR:
    """Model holder for vllm-metal Qwen3-ASR."""

    sep = ""
    SAMPLING_RATE = 16_000
    backend_choice = "qwen3-vllm-metal"

    def __init__(self, logfile=sys.stderr, **kwargs):
        _ensure_supported_platform()

        try:
            import mlx.core as mx
            from vllm_metal.stt.loader import load_model
            from vllm_metal.stt.qwen3_asr.adapter import Qwen3ASRRuntimeAdapter
            from vllm_metal.stt.qwen3_asr.config import (
                QWEN3_ASR_MAX_DECODE_TOKENS,
            )
            from vllm_metal.stt.qwen3_asr.transcriber import Qwen3ASRTranscriber
        except ImportError as exc:
            raise _missing_dependency_error() from exc

        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None
        self.tokenizer = None
        self._post_process_output = Qwen3ASRTranscriber.post_process_output

        model_path = _resolve_model_path(kwargs)
        dtype = kwargs.get("dtype", mx.float16)
        self.max_decode_tokens = int(
            kwargs.get("max_tokens") or QWEN3_ASR_MAX_DECODE_TOKENS
        )

        t0 = time.time()
        logger.info("Loading Qwen3 vllm-metal model '%s' ...", model_path)
        self.model = load_model(model_path, dtype=dtype)
        self.adapter = Qwen3ASRRuntimeAdapter(self.model, model_path)
        self.adapter.warm_up()
        self.tokenizer = self.adapter.transcriber.tokenizer
        self._mrope = self._build_mrope()
        logger.info("Qwen3 vllm-metal model loaded in %.2fs", time.time() - t0)

    def _build_mrope(self) -> _InterleavedMRoPE:
        text_config = self.model.config.text_config
        return _InterleavedMRoPE(
            head_dim=text_config.head_dim,
            base=text_config.rope_theta,
            mrope_section=_mrope_section_from_config(text_config),
        )

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

    def _attention_forward(self, attn, x, cos, sin, mask=None, cache=None):
        import mlx.core as mx

        b, seq, _ = x.shape
        q = attn.q_proj(x).reshape(b, seq, attn.n_heads, attn.head_dim)
        k = attn.k_proj(x).reshape(b, seq, attn.n_kv_heads, attn.head_dim)
        v = attn.v_proj(x).reshape(b, seq, attn.n_kv_heads, attn.head_dim)

        q = attn.q_norm(q).transpose(0, 2, 1, 3)
        k = attn.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)

        key = k
        value = v
        if attn.n_rep > 1:
            key = mx.repeat(key, attn.n_rep, axis=1)
            value = mx.repeat(value, attn.n_rep, axis=1)

        weights = (q * attn.scale) @ key.transpose(0, 1, 3, 2)
        if mask is not None:
            weights = weights + mask
        weights = mx.softmax(weights, axis=-1, precise=True)
        output = (weights @ value).transpose(0, 2, 1, 3).reshape(b, seq, -1)
        return attn.o_proj(output), new_cache

    def _language_model_forward(self, embeds, cache=None):
        import mlx.core as mx

        _, seq, _ = embeds.shape
        offset = cache[0][0].shape[2] if cache and cache[0] is not None else 0

        positions = mx.arange(offset, offset + seq)[None, :]
        position_ids = mx.stack([positions, positions, positions], axis=1)
        cos, sin = self._mrope(position_ids, dtype=embeds.dtype)

        mask = None
        if seq > 1:
            total_len = offset + seq
            mask = _causal_mask(total_len, embeds.dtype)[..., -seq:, :total_len]

        lm = self.model.language_model
        if cache is None:
            cache = [None] * len(lm.layers)

        hidden = embeds
        new_cache = []
        for idx, layer in enumerate(lm.layers):
            residual, layer_cache = self._attention_forward(
                layer.self_attn,
                layer.input_layernorm(hidden),
                cos,
                sin,
                mask=mask,
                cache=cache[idx],
            )
            hidden = hidden + residual
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            new_cache.append(layer_cache)

        hidden = lm.norm(hidden)
        if lm.lm_head is not None:
            logits = lm.lm_head(hidden)
        else:
            logits = lm.embed_tokens.as_linear(hidden)
        return logits, new_cache

    def _inject_audio_features(self, prompt_ids: list[int], audio_features):
        import mlx.core as mx

        token_ids = mx.array([prompt_ids], dtype=mx.int32)
        embeds = self.model.language_model.embed(token_ids)

        audio_positions = [
            idx
            for idx, token_id in enumerate(token_ids[0].tolist())
            if token_id == self.model.config.audio_token_id
        ]
        if not audio_positions or audio_features.shape[0] == 0:
            return embeds

        parts = []
        previous = 0
        n_inject = min(len(audio_positions), audio_features.shape[0])
        for feature_idx, token_pos in enumerate(audio_positions[:n_inject]):
            if token_pos > previous:
                parts.append(embeds[0, previous:token_pos, :])
            parts.append(audio_features[feature_idx : feature_idx + 1].astype(embeds.dtype))
            previous = token_pos + 1

        if previous < embeds.shape[1]:
            parts.append(embeds[0, previous:, :])

        return mx.concatenate(parts, axis=0)[None, ...]

    def _decode_tokens(self, audio_features, prompt_ids: list[int]) -> list[int]:
        import mlx.core as mx

        embeds = self._inject_audio_features(prompt_ids, audio_features)
        logits, cache = self._language_model_forward(embeds)

        output_tokens = []
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        for _ in range(self.max_decode_tokens):
            if next_token == self.model.config.eos_token_id:
                break
            output_tokens.append(next_token)
            token_input = mx.array([[next_token]], dtype=mx.int32)
            token_embeds = self.model.language_model.embed(token_input)
            logits, cache = self._language_model_forward(token_embeds, cache)
            next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())

        return self.adapter._extract_asr_text_tokens(output_tokens)

    def transcribe_text(self, audio: np.ndarray) -> str:
        """Transcribe raw 16 kHz mono float PCM and return cleaned text."""
        if len(audio) < 400:
            return ""

        try:
            from vllm_metal.stt.audio import log_mel_spectrogram
        except ImportError as exc:
            raise _missing_dependency_error() from exc

        mel = log_mel_spectrogram(audio.astype(np.float32), n_mels=128)
        audio_features = self.adapter.extract_audio_features(mel)
        n_audio_tokens = int(audio_features.shape[0])
        prompt_ids = self._build_prompt_token_ids(n_audio_tokens)
        output_tokens = self._decode_tokens(audio_features, prompt_ids)
        text = self.tokenizer.decode(output_tokens, skip_special_tokens=False)
        return self._post_process_output(text).strip()

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> str:
        return self.transcribe_text(audio)

    def use_vad(self):
        return False


class Qwen3VLLMMetalOnlineProcessor:
    """Batch processor with a simple two-word holdback streaming policy."""

    SAMPLING_RATE = 16_000
    _HOLDBACK_WORDS = 2

    def __init__(self, asr: Qwen3VLLMMetalASR, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.end = 0.0
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer = []

        self._buffer_time_offset = 0.0
        self._n_committed_words = 0
        self._current_words: list[str] = []
        self._samples_since_last_inference = 0
        self._min_new_samples = self.SAMPLING_RATE

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        self.end = audio_stream_end_time
        self.audio_buffer = np.append(self.audio_buffer, audio)
        self._samples_since_last_inference += len(audio)

    def _transcribe_words(self) -> list[str]:
        text = self.asr.transcribe_text(self.audio_buffer)
        words = text.split()
        self._current_words = words
        return words

    def _time_for_word(self, word_idx: int, n_words_total: int) -> Tuple[float, float]:
        duration = max(len(self.audio_buffer) / self.SAMPLING_RATE, 0.001)
        n_total = max(n_words_total, 1)
        start = self._buffer_time_offset + (word_idx / n_total) * duration
        end = self._buffer_time_offset + ((word_idx + 1) / n_total) * duration
        return start, end

    def _tokens_for_range(self, words: list[str], start_idx: int, end_idx: int) -> List[ASRToken]:
        tokens: List[ASRToken] = []
        n_total = len(words)
        for idx in range(start_idx, end_idx):
            start, end = self._time_for_word(idx, n_total)
            text = words[idx] if idx == 0 else " " + words[idx]
            tokens.append(ASRToken(start=start, end=end, text=text))
        return tokens

    def _commit_available(self, flush: bool = False) -> List[ASRToken]:
        words = self._transcribe_words()
        commit_upto = len(words) if flush else max(len(words) - self._HOLDBACK_WORDS, 0)
        if commit_upto <= self._n_committed_words:
            return []

        tokens = self._tokens_for_range(words, self._n_committed_words, commit_upto)
        self._n_committed_words = commit_upto
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
