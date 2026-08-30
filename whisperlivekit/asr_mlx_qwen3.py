"""Adapt mlx-qwen3-asr's incremental stream to WLK's audio clock.

The native stable prefix is committed without a second full-utterance decode.
Timestamps bound decode chunks; this backend does not provide word alignment.
"""
from __future__ import annotations

import threading

import numpy as np

from whisperlivekit.session_asr_proxy import merge_session_context
from whisperlivekit.timed_objects import ASRToken, Transcript

_MLX_LOCK = threading.RLock()
_QWEN_LANG_ALIASES = {
    "en": "English", "zh": "Chinese", "cmn": "Chinese", "yue": "Cantonese",
    "ja": "Japanese", "ko": "Korean", "de": "German", "fr": "French",
    "zh-yue": "Cantonese", "es": "Spanish", "it": "Italian", "pt": "Portuguese", "ru": "Russian",
}


def _resolve_language(language):
    if not language or language.strip().lower() == "auto":
        return None
    key = language.strip().lower()
    return _QWEN_LANG_ALIASES.get(key, _QWEN_LANG_ALIASES.get(key.split("-")[0], language.strip().title()))


class MlxQwen3ASR:
    sep = ""
    backend_choice = "mlx-qwen3-asr"

    def __init__(self, config):
        from mlx_qwen3_asr import load_model
        from mlx_qwen3_asr.streaming import feed_audio, finish_streaming

        self.model_id = config.mlx_qwen3_asr_model
        self.language = _resolve_language(config.lan)
        self.hotwords = config.mlx_qwen3_asr_context
        self.chunk_size_sec = config.mlx_qwen3_asr_chunk_sec
        self.max_context_sec = config.mlx_qwen3_asr_max_context_sec
        self.finalization_mode = config.mlx_qwen3_asr_finalization_mode
        # Load and compile once at engine startup, not on every connection.
        with _MLX_LOCK:
            self.model, _ = load_model(self.model_id)
            online = MlxQwen3AsrOnlineProcessor(self)
            state = online._new_state()
            state = feed_audio(np.zeros(8000, dtype=np.float32), state, model=self.model)
            finish_streaming(state, model=self.model)


class MlxQwen3AsrOnlineProcessor:
    SAMPLING_RATE = 16_000

    def __init__(self, asr, logfile=None):
        self.asr = asr
        self.sep = ""
        language = (asr._session_language if getattr(asr, "_override_language", False)
                    else asr.language)
        self.language = _resolve_language(language)
        self.context = merge_session_context(asr.hotwords, getattr(asr, "_session_context", None))
        self._state = None
        self._pending = []
        self._committed = ""
        self._audio_end = 0.0
        self._token_start = 0.0
        self._previous_text = ""

    def _new_state(self):
        from mlx_qwen3_asr.streaming import init_streaming

        state = init_streaming(
            model=self.asr.model_id, context=self.context,
            chunk_size_sec=self.asr.chunk_size_sec,
            max_context_sec=self.asr.max_context_sec,
            language=self.language, finalization_mode=self.asr.finalization_mode,
        )
        state.forced_language = self.language
        return state

    def insert_audio_chunk(self, audio, audio_stream_end_time):
        audio = np.asarray(audio, dtype=np.float32)
        if not len(audio):
            return
        if self._state is None and not self._pending:
            self._token_start = max(0.0, audio_stream_end_time - len(audio) / self.SAMPLING_RATE)
        self._pending.append(audio.copy())
        self._audio_end = audio_stream_end_time

    def _feed_pending(self):
        from mlx_qwen3_asr.streaming import feed_audio

        if not self._pending:
            return
        if self._state is None:
            self._state = self._new_state()
        audio = np.concatenate(self._pending)
        self._pending.clear()
        self._state = feed_audio(audio, self._state, model=self.asr.model)

    def _boundary_prefix(self, text):
        # Native text includes spaces within an utterance. Only a reset between
        # utterances needs a separator; never insert one inside a CJK phrase.
        if not self._committed and self._previous_text and text:
            language = self.language or getattr(self._state, "language", "")
            if (language or "").lower() not in {"chinese", "cantonese", "japanese", "korean"}:
                return " "
        return ""

    def _commit(self, text):
        text = (text or "").strip()
        if not text.startswith(self._committed):
            raise RuntimeError("mlx-qwen3-asr revised an already committed prefix")
        delta = text[len(self._committed):]
        if not delta:
            return [], self._audio_end
        token = ASRToken(
            start=self._token_start, end=self._audio_end,
            text=self._boundary_prefix(delta) + delta,
            detected_language=self.language or getattr(self._state, "language", None),
        )
        self._committed = text
        self._previous_text = text
        self._token_start = self._audio_end
        return [token], self._audio_end

    def process_iter(self, is_last=False):
        # AudioProcessor runs this in its counted inference executor. Inserting
        # PCM on the event loop must not launch a decode.
        if is_last:
            return self.finish()
        with _MLX_LOCK:
            self._feed_pending()
            return self._commit(self._state.stable_text) if self._state else ([], self._audio_end)

    def finish(self):
        from mlx_qwen3_asr.streaming import finish_streaming

        with _MLX_LOCK:
            self._feed_pending()
            if self._state is None:
                return [], self._audio_end
            self._state = finish_streaming(self._state, model=self.asr.model)
            result = self._commit(self._state.text)
            self._state = None
            self._committed = ""
            return result

    def start_silence(self):
        return self.finish()

    def new_speaker(self, *args, **kwargs):
        return self.finish()

    def end_silence(self, silence_duration, offset):
        self._audio_end += silence_duration

    def get_buffer(self):
        text = self._state.text if self._state else ""
        if not text.startswith(self._committed):
            raise RuntimeError("mlx-qwen3-asr revised an already committed prefix")
        tail = text[len(self._committed):]
        return Transcript(start=self._token_start, end=self._audio_end,
                          text=self._boundary_prefix(tail) + tail if tail else "")
