"""mlx-qwen3-asr ASR backend for WhisperLiveKit.

This is the pure-MLX qwen3-asr (moona3k's `mlx-qwen3-asr` package) — a ground-up
MLX reimplementation with a real incremental-audio streaming API
(`init_streaming` / `feed_audio` / `finish_streaming`, a rolling decode with a
stabilizing prefix). Unlike WLK's built-in `qwen3-streaming` backend (which
uses `qwen3-asr-causal`, a torch/transformers package that pins
transformers==4.57.6 and conflicts with recent mlx-lm), this backend is pure
MLX: no torch, no transformers version conflict. It coexists cleanly with
mlx-lm on transformers 5.x.

The online processor adapts mlx-qwen3-asr's streaming API to WLK's
`insert_audio_chunk` / `process_iter` / `start_silence` / `finish` /
`get_buffer` contract (the same shape as Qwen3StreamingOnlineProcessor in
qwen3-asr-causal/online.py). WLK's AudioProcessor owns the VAD endpointing, so
this backend does NOT run its own VAD — unlike livecaption's QwenOnlineStream
which carries its own Silero VAD. That VAD logic is dropped here; WLK provides it.

Tokens: mlx-qwen3-asr emits a `stable_text` prefix (monotonically non-decreasing
committed text) plus the full rolling `text`. The decode loop returns the raw
rolling hypothesis via `get_buffer()`; a `StableCommitTransform` in the wrapper
chain (applied by `online_factory`) commits only the stable prefix — Job 1 of
the generalized wrapper layer. The two-pass re-decode at `start_silence` /
`finish` stays here (model-specific: re-decode the utterance audio offline for
clean text). Warmup runs at init.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Language tag -> name mlx-qwen3-asr expects (mirrors livecaption's _QWEN_LANG_ALIASES).
_QWEN_LANG_ALIASES = {
    "en": "English", "en-us": "English", "en-gb": "English", "english": "English",
    "zh": "Chinese", "zh-cn": "Chinese", "zh-hans": "Chinese", "cmn": "Chinese",
    "chinese": "Chinese", "mandarin": "Chinese",
    "yue": "Cantonese", "zh-yue": "Cantonese", "cantonese": "Cantonese",
    "ja": "Japanese", "ja-jp": "Japanese", "japanese": "Japanese",
    "ko": "Korean", "ko-kr": "Korean", "korean": "Korean",
    "de": "German", "de-de": "German", "german": "German",
    "fr": "French", "fr-fr": "French", "french": "French",
    "es": "Spanish", "es-es": "Spanish", "spanish": "Spanish",
}


def _resolve_language(language: str | None) -> str | None:
    if not language or language.strip().lower() == "auto":
        return None
    key = language.strip().lower()
    return _QWEN_LANG_ALIASES.get(key, language.strip().title())


# Hold the loaded (model_obj, model_id) once per process — mlx-qwen3-asr loads
# weights once and reuses the model object across feed/finish calls.
_MODEL_CACHE: dict[str, object] = {}


def _ensure_model(model_id: str):
    if model_id not in _MODEL_CACHE:
        from mlx_qwen3_asr import load_model
        logger.info("Loading mlx-qwen3-asr model %s ...", model_id)
        model_obj, _cfg = load_model(model_id)
        _MODEL_CACHE[model_id] = model_obj
    return _MODEL_CACHE[model_id]


def _compute_finalize_delta(final_text: str, emitted_stable: str) -> str:
    """Return the uncommitted portion of final_text for finalization.

    Deduplicates against text the StableCommitTransform already committed during
    streaming (tracked in the processor's _emitted_stable). Three cases:

    - emitted_stable is empty: no streaming commit happened (very short utterance
      or first utterance) → emit the full final_text.
    - final_text starts with emitted_stable: the re-decode confirmed the streaming
      prefix → emit only the suffix (the uncommitted delta).
    - final_text does NOT start with emitted_stable: the re-decode corrected the
      prefix → emit the full corrected text. Known limitation: the stale prefix
      emitted during streaming remains in the output; the diff protocol treats the
      full corrected text as a new token rather than retracting the stale prefix.
    """
    final = final_text.strip()
    emitted = emitted_stable.strip() if emitted_stable else ""
    if not emitted:
        return final
    if final.startswith(emitted):
        return final[len(emitted):].strip()
    return final


class MlxQwen3AsrOnlineProcessor:
    """WLK online processor wrapping mlx-qwen3-asr's streaming API.

    Contract (mirrors qwen3-asr-causal's Qwen3StreamingOnlineProcessor):
      - insert_audio_chunk(audio, audio_stream_end_time): buffer + feed audio.
      - process_iter(is_last=False) -> (tokens, end_time): emit committed tokens.
      - start_silence() -> (tokens, end_time): mark silence start.
      - finish() -> (tokens, end_time): finalize the utterance.
      - get_buffer(): return the current rolling transcript.
    """

    SAMPLING_RATE = 16_000

    def __init__(self, asr, logfile=None):
        # `asr` is a config bundle carrying model_id / language / hotwords / chunk params.
        # We build it from the online_factory kwargs (see core.py wiring). When `asr`
        # is a SessionASRProxy, prefer its per-session language override over the
        # server-wide default (the proxy delegates __getattr__ to the shared ASR, so
        # getattr(asr, 'language') would return the server-wide value without this).
        self.model_id = getattr(asr, "model_id", "Qwen/Qwen3-ASR-0.6B")
        session_lang = getattr(asr, "_session_language", None)
        self.language = _resolve_language(session_lang or getattr(asr, "language", None))
        self.hotwords = getattr(asr, "hotwords", "") or ""
        self.chunk_size_sec = getattr(asr, "chunk_size_sec", 2.0)
        self.max_context_sec = getattr(asr, "max_context_sec", 30.0)
        self.finalization_mode = getattr(asr, "finalization_mode", "latest")
        self._model_obj = _ensure_model(self.model_id)
        self.asr = asr  # back-ref (audio_processor reads self.asr.sep)
        self.sep = getattr(asr, "sep", "")  # CJK: no space between tokens
        self._state = None
        self._text = ""
        self._stable_text = ""
        self._emitted_stable = ""
        self._audio_end_time = 0.0
        self._started = False
        self._utt_audio: list = []  # retained for the two-pass re-decode at finish
        self._warmup()

    def _warmup(self) -> None:
        """Run one short decode on silence to absorb Metal kernel compilation
        now, so the first real sentence's partial doesn't stall. Mirrors
        livecaption/asr_qwen.py:_warmup."""
        import numpy as np
        from mlx_qwen3_asr.streaming import feed_audio, finish_streaming
        try:
            silence = np.zeros(int(0.5 * self.SAMPLING_RATE), dtype=np.float32)
            state = self._new_state()
            with _MLX_LOCK:
                state = feed_audio(silence, state, model=self._model_obj)
                finish_streaming(state, model=self._model_obj)
        except Exception as exc:  # warmup is non-fatal
            logger.debug("mlx-qwen3-asr warmup failed (non-fatal): %s", exc)

    def _new_state(self):
        from mlx_qwen3_asr.streaming import init_streaming
        state = init_streaming(
            model=self.model_id,
            context=self.hotwords,
            chunk_size_sec=self.chunk_size_sec,
            max_context_sec=self.max_context_sec,
            language=self.language,
            finalization_mode=self.finalization_mode,
        )
        state.forced_language = self.language
        return state

    def _feed(self, audio: np.ndarray):
        from mlx_qwen3_asr.streaming import feed_audio
        with _MLX_LOCK:
            self._state = feed_audio(audio, self._state, model=self._model_obj)
        self._text = (getattr(self._state, "text", "") or "").strip()
        self._stable_text = (getattr(self._state, "stable_text", "") or "").strip()

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        if not self._started:
            self._state = self._new_state()
            self._started = True
        self._audio_end_time = audio_stream_end_time
        self._utt_audio.append(np.asarray(audio, dtype=np.float32))  # retain for two-pass
        self._feed(np.asarray(audio, dtype=np.float32))

    def process_iter(self, is_last=False) -> Tuple[List, float]:
        # Return the raw rolling hypothesis.  The stable_commit wrapper
        # (applied by online_factory) reads get_buffer() for the rolling
        # text and commits only the stable prefix — Job 1 of the wrapper
        # layer.  We do NOT commit here; the wrapper does.
        return [], self._audio_end_time

    def start_silence(self) -> Tuple[List, float]:
        # Utterance boundary (silence detected): do the two-pass re-decode of the
        # accumulated utterance audio and emit ONE clean committed token. This is
        # the per-utterance finalization point during the session — firing here (not
        # only at finish()) is what makes translation stream per-utterance instead
        # of only at session stop.
        return self._finalize_utterance()

    def end_silence(self, silence_duration: float, offset: float):
        # Advance our audio-end clock by the silence duration (mirrors qwen3-asr-causal).
        self._audio_end_time += silence_duration

    def _finalize_utterance(self) -> Tuple[List, float]:
        """Two-pass re-decode the accumulated utterance audio, emit the clean text
        as one committed token, and reset for the next utterance. Falls back to
        the streaming text if the re-decode fails."""
        from whisperlivekit.timed_objects import ASRToken
        final_text = self._text
        utt_audio_s = sum(len(a) for a in self._utt_audio) / self.SAMPLING_RATE if self._utt_audio else 0.0
        if self._utt_audio:
            try:
                from mlx_qwen3_asr import transcribe
                audio = np.concatenate(self._utt_audio)
                with _MLX_LOCK:
                    result = transcribe(audio, model=self._model_obj, language=self.language)
                redecoded = (getattr(result, "text", "") or "").strip()
                if redecoded:
                    final_text = redecoded
                else:
                    logger.warning("mlx-qwen3-asr two-pass returned empty (utt=%.1fs, stream_text=%r); using stream text", utt_audio_s, self._text[:60])
            except Exception as exc:  # two-pass failure is non-fatal
                logger.warning("mlx-qwen3-asr two-pass failed (utt=%.1fs, stream_text=%r): %s", utt_audio_s, self._text[:60], exc)
        # Reset the rolling-decode state for the next utterance.
        # Compute the dedup delta BEFORE resetting _emitted_stable.
        text_to_emit = _compute_finalize_delta(final_text, self._emitted_stable)
        self._utt_audio = []
        self._text = ""
        self._stable_text = ""
        self._emitted_stable = ""
        self._state = self._new_state()
        self._started = True
        if text_to_emit:
            tok = ASRToken(
                start=self._audio_end_time,
                end=self._audio_end_time,
                text=text_to_emit,
                speaker=-1,
                detected_language=self.language,
            )
            return [tok], self._audio_end_time
        return [], self._audio_end_time

    def get_hypothesis(self):
        # Return a Transcript (start, end, text) with the FULL rolling hypothesis
        # (the committed stable prefix PLUS the unstable tail). This is the seam
        # StableCommitTransform reads to compute the stable prefix across decode
        # passes. Contrast get_buffer(), which returns only the unstable tail
        # (the WLK contract for display and the AlignAtt HypothesisTail).
        from whisperlivekit.timed_objects import Transcript
        return Transcript(start=None, end=self._audio_end_time, text=self._text)

    def get_buffer(self):
        # Return a Transcript (start, end, text) — the UNSTABLE tail only (not the
        # committed stable prefix). WLK's contract: get_buffer returns the unstable
        # hypothesis tail (not yet committed); process_iter returns the committed
        # ASRToken list. Returning the full rolling text double-counts the committed
        # prefix for any consumer that reads get_buffer (display, and the AlignAtt
        # translator which drafts over the tail via HypothesisTail).
        from whisperlivekit.timed_objects import Transcript
        stable = self._stable_text
        text = self._text
        tail = text[len(stable):] if stable and text.startswith(stable) else text
        return Transcript(start=None, end=self._audio_end_time, text=tail)

    def finish(self) -> Tuple[List, float]:
        # Session end: finalize any leftover utterance (no trailing silence fired).
        return self._finalize_utterance()


# MLX lock: serialize MLX decode steps (mirrors livecaption's runtime.MLX_LOCK).
# Imported lazily so this module imports even if mlx isn't ready yet.
try:
    from whisperlivekit._mlx_lock import MLX_LOCK as _MLX_LOCK  # if WLK provides one
except Exception:  # noqa: BLE001
    import contextlib
    _MLX_LOCK = contextlib.nullcontext()
