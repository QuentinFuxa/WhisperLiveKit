"""Adapt mlx-qwen3-asr's incremental stream to WLK's audio clock.

The native stream keeps a bounded decode window and re-decodes it on every
chunk, so its text stays revisable until the window is committed. This adapter
commits only the prefix the library considers stable and drains the stream at
pauses, speaker changes and EOF. Timestamps bound decode chunks; this backend
does not provide word alignment.

Two upstream properties shape the adapter. ``stable_text`` stays empty for the
first ``unfixed_chunk_num`` chunks, so a short utterance would commit nothing
before its pause; the adapter derives a prefix from agreement between
consecutive window texts meanwhile. And ``stable_text`` is neither guaranteed to
be a prefix of ``text`` nor monotone across chunks (upstream issue #31), so a
candidate that revises committed text is repaired to an append instead of
aborting the session. That repair tracks the text the last commit came from, not
the text clients hold, so a revised fragment reaches clients once.
"""
from __future__ import annotations

import re
import threading

import numpy as np

from whisperlivekit.backend_support import QWEN_LANGUAGES
from whisperlivekit.session_asr_proxy import merge_session_context
from whisperlivekit.timed_objects import ASRToken, Transcript

_MLX_LOCK = threading.RLock()

# compute_features() reflect-pads by n_fft // 2 = 200 samples and needs two STFT
# frames, so a shorter window raises instead of decoding. Measured on
# mlx-qwen3-asr 0.4.4: stft() rejects 200 samples, 201 samples already decode. A
# pause can arrive with only a fragment of audio left over, so keep the decoded
# window at twice that floor rather than handing the library audio it cannot
# transcribe.
_MIN_DECODE_SAMPLES = 400

# Trailing units of the agreed prefix that stay revisable. Mirrors the
# library's own UNFIXED_TOKEN_NUM, which the native split holds back.
_HOLD_BACK_UNITS = 5

# CJK has no whitespace between characters, so word units are characters there
# and whitespace-delimited words elsewhere. One unit per CJK character lets the
# agreed prefix advance inside a Chinese sentence.
_CJK_UNIT_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD_UNIT_RE = re.compile(r"\S+\s*")


def _resolve_language(language):
    if not language or language.strip().lower() == "auto":
        return None
    from mlx_qwen3_asr.tokenizer import canonicalize_language, language_is_known

    from whisperlivekit.whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

    # The native alias table covers fewer languages than the Qwen model. Reuse
    # WLK's names for the remaining ISO codes instead of sending "Cs" or "Pl".
    if language_is_known(language):
        return canonicalize_language(language)
    key = language.strip().lower().replace("_", "-")
    if key in {"fil", "filipino"}:
        return "Filipino"
    key = TO_LANGUAGE_CODE.get(key, key.split("-")[0])
    if key in QWEN_LANGUAGES and key in LANGUAGES:
        return LANGUAGES[key].title()
    raise ValueError(f"Unsupported Qwen3 ASR language: {language!r}")


def _common_prefix_length(left: str, right: str) -> int:
    """Length of the shared prefix, ignoring case.

    Case matters at a segment start: a window re-decode can capitalise the first
    word of the same sentence, and comparing case-sensitively would report the
    whole segment as new text and append it a second time.
    """
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index].casefold() != right[index].casefold():
            return index
    return limit


def _append_only_delta(source: str, candidate: str) -> tuple[str, str]:
    """Return ``(source, delta)`` for a candidate that may revise earlier text.

    WLK tokens are append-only, so a candidate that does not extend the text a
    commit came from cannot be adopted verbatim. ``source`` is that text, not the
    text clients already hold: the two diverge once a revision was absorbed, and
    computing the next delta against the accumulated text would repeat the
    revised fragment.

    A candidate that extends the source contributes its tail. A candidate at
    least as long as the source that rewrote it becomes the new source and
    contributes its part after the shared prefix, so the rewrite reaches clients
    exactly once. A shorter candidate retracts nothing: clients keep the longer
    text, so it contributes nothing and the source stays put, which also lets the
    text come back later without being appended twice.
    """
    if candidate.startswith(source):
        return candidate, candidate[len(source):]
    if len(candidate) < len(source):
        return source, ""
    return candidate, candidate[_common_prefix_length(source, candidate):]


def _split_text_units(text: str) -> list[str]:
    """Split text into units that re-join to the same string."""
    units: list[str] = []
    index = 0
    while index < len(text):
        if _CJK_UNIT_RE.match(text[index]):
            units.append(text[index])
            index += 1
            continue
        match = _WORD_UNIT_RE.match(text, index)
        if match is None:
            units.append(text[index])
            index += 1
            continue
        units.append(match.group())
        index = match.end()
    return units


def _agreed_prefix(previous_text: str, text: str, hold_back_units: int) -> str:
    """Prefix of ``text`` whose units also appeared in the previous window text.

    Applies the LocalAgreement-1 policy the adapter needs while the library
    reports no stable prefix: only units two consecutive window decodes agree on
    are eligible, and the last ``hold_back_units`` stay revisable.
    """
    units = _split_text_units(text)
    previous_units = _split_text_units(previous_text)
    # Units keep their trailing whitespace so the joined prefix is exact; the
    # agreement check compares them without it, since only the last unit of a
    # decode lacks a separator.
    agreed = _common_prefix_length(
        [unit.strip() for unit in previous_units],
        [unit.strip() for unit in units],
    )
    return "".join(units[:max(0, agreed - hold_back_units)]).strip()


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
        self._committed_source = ""
        self._audio_end = 0.0
        self._token_start = 0.0
        self._previous_text = ""
        self._previous_window_text = ""

    def _new_state(self):
        from mlx_qwen3_asr.streaming import init_streaming

        state = init_streaming(
            model=self.asr.model_id, context=self.context,
            chunk_size_sec=self.asr.chunk_size_sec,
            max_context_sec=self.asr.max_context_sec,
            language=self.language, finalization_mode=self.asr.finalization_mode,
        )
        return state

    def insert_audio_chunk(self, audio, audio_stream_end_time):
        audio = np.asarray(audio, dtype=np.float32)
        if not len(audio):
            return
        if self._state is None and not self._pending:
            self._token_start = max(0.0, audio_stream_end_time - len(audio) / self.SAMPLING_RATE)
        self._pending.append(audio.copy())
        self._audio_end = audio_stream_end_time

    def _pending_samples(self) -> int:
        return int(sum(len(chunk) for chunk in self._pending))

    def _window_samples(self) -> int:
        """Samples the next decode would cover: the live window plus its tail."""
        state = self._state
        if state is None:
            return 0
        return int(len(state.audio_accum) + len(state.buffer))

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

    def _stable_candidate(self) -> str:
        """Text this decode wants committed: the native prefix, else an agreed one."""
        state = self._state
        window_text = state.text or ""
        candidate = (state.stable_text or "").strip()
        if not candidate:
            candidate = _agreed_prefix(
                self._previous_window_text, window_text, _HOLD_BACK_UNITS
            )
        self._previous_window_text = window_text
        return candidate

    def _commit(self, candidate):
        source, delta = _append_only_delta(self._committed_source, (candidate or "").strip())
        if not delta:
            return []
        token = ASRToken(
            start=self._token_start, end=self._audio_end,
            text=self._boundary_prefix(delta) + delta,
            detected_language=self.language or getattr(self._state, "language", None),
        )
        self._committed_source = source
        self._committed += delta
        self._previous_text = self._committed
        self._token_start = self._audio_end
        return [token]

    def process_iter(self, is_last=False):
        # AudioProcessor runs this in its counted inference executor. Inserting
        # PCM on the event loop must not launch a decode.
        if is_last:
            return self.finish()
        with _MLX_LOCK:
            self._feed_pending()
            if self._state is None:
                return [], self._audio_end
            return self._commit(self._stable_candidate()), self._audio_end

    def finish(self):
        from mlx_qwen3_asr.streaming import finish_streaming

        with _MLX_LOCK:
            if self._state is None and self._pending_samples() < _MIN_DECODE_SAMPLES:
                # Nothing decodable: a pause opened before a full chunk arrived.
                self._pending.clear()
                return [], self._audio_end
            self._feed_pending()
            if self._state is None:
                return [], self._audio_end
            if self._window_samples() < _MIN_DECODE_SAMPLES:
                # The live window is below the mel floor. Keep the text earlier
                # chunks produced instead of asking for an impossible decode.
                final_text = self._state.text
            else:
                # The window re-decode with the pending tail is the finalize
                # path in 0.4.x; there is no separate tail-refinement pass.
                self._state = finish_streaming(self._state, model=self.asr.model)
                final_text = self._state.text
            tokens = self._commit(final_text)
            self._state = None
            self._committed = ""
            self._committed_source = ""
            self._previous_window_text = ""
            return tokens, self._audio_end

    def start_silence(self):
        return self.finish()

    def new_speaker(self, *args, **kwargs):
        return self.finish()

    def end_silence(self, silence_duration, offset):
        self._audio_end += silence_duration

    def get_buffer(self):
        # The unstable tail is what the current window says beyond the text the
        # last commit was derived from; measuring it against the committed text
        # would re-show a fragment a revision already carried.
        text = (self._state.text if self._state else "") or ""
        tail = text[_common_prefix_length(self._committed_source, text):]
        return Transcript(start=self._token_start, end=self._audio_end,
                          text=self._boundary_prefix(tail) + tail if tail else "")
