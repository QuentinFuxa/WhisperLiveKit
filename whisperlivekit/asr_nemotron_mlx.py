"""Nemotron-3.5 ASR streaming transducer backend (pure MLX via mlx-audio).

Provides ``NemotronMLXASR`` (model holder) and ``NemotronMLXOnlineProcessor``
(streaming processor) that plug into WhisperLiveKit's audio processing pipeline
via ``insert_audio_chunk`` / ``process_iter`` / ``get_buffer`` / ``finish``.

The model is a cache-aware FastConformer-RNNT transducer
(``nvidia/nemotron-3.5-asr-streaming-0.6b``) loaded via ``mlx_audio.stt.load`` —
no ``nemo_toolkit``, ONNX, torch, or transformers.  The greedy RNN-T decode is
monotonic and append-only: each non-blank emission appends an ``AlignedToken``
with ``start = (global_time + time) * frame_sec`` to the hypothesis DURING the
decode (not just at finalization).  This gives a principled, time-based
accessible boundary — the research-enabling property — that qwen3-asr's
``stable_text`` proxy cannot.

Ported from ``livecaption/livecaption/asr.py`` (``_StreamingEncoder``,
``_decode_chunk``, ``_finalize``, mel grow/holdback, VAD endpointer).  The
wrapper layer (monotonic-enforce + timestamp-inject) is skipped: the transducer
is monotonic by construction and ``AlignedToken.start`` is emitted mid-decode.
"""

from __future__ import annotations

import logging
import re
import sys
from collections import deque
from typing import List, Tuple

import mlx.core as mx
import numpy as np

from whisperlivekit.timed_objects import ASRToken, Transcript

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language normalization (mirrors livecaption/livecaption/languages.py)
# ---------------------------------------------------------------------------
#
# The model's ``prompt_dictionary`` uses full locale tags (``zh-CN``, ``en-US``)
# but the CLI/API commonly passes bare 2-letter primaries (``zh``, ``en``).
# ``_normalize_language`` bridges that gap so ``--language zh`` resolves to the
# model's ``zh-CN`` prompt instead of raising ValueError.

# Preferred full tag for a bare 2-letter primary (mirrors livecaption's
# ``_DEFAULT_TAGS``).  Matching against the model's keys is case-insensitive.
_DEFAULT_LANGUAGE_TAGS = {
    "de": "de-DE",
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "it": "it-IT",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "pt": "pt-PT",
    "zh": "zh-CN",
}


def _normalize_language(lan: str, supported_keys) -> str:
    """Normalize a user-supplied language tag to a model prompt_dictionary key.

    Mirrors livecaption's ``normalize_asr_language``
    (livecaption/livecaption/languages.py).  Steps:

    1. ``auto`` → ``"auto"`` (the caller converts this to ``None``).
    2. Case-insensitive exact match against supported keys.
    3. Default-tag mapping (``zh`` → ``zh-CN``, ``en`` → ``en-US``, …) then
       case-insensitive match.
    4. Primary-prefix match: if the input is a bare primary, find supported keys
       starting with that prefix; use the unique match, or the default-tag
       variant when multiple match.
    5. Raise ``ValueError`` if no mapping is found.
    """
    if not lan or lan.casefold() == "auto":
        return "auto"

    supported = list(supported_keys)

    # 1. Case-insensitive exact match.
    for key in supported:
        if key.casefold() == lan.casefold():
            return key

    primary = lan.split("-", 1)[0].casefold()

    # 2. Default-tag mapping.
    default_tag = _DEFAULT_LANGUAGE_TAGS.get(primary)
    if default_tag is not None:
        for key in supported:
            if key.casefold() == default_tag.casefold():
                return key

    # 3. Primary-prefix match.
    prefix_matches = [
        key
        for key in supported
        if key.casefold() == primary or key.casefold().startswith(f"{primary}-")
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1 and default_tag is not None:
        # Prefer the default-tag variant among the ambiguous matches.
        for key in prefix_matches:
            if key.casefold() == default_tag.casefold():
                return key

    # 4. No mapping found.
    raise ValueError(
        f"ASR language '{lan}' is not supported by this model.\n"
        f"Available: {', '.join(sorted(supported))}"
    )

# ---------------------------------------------------------------------------
# Constants (ported from livecaption/asr.py)
# ---------------------------------------------------------------------------

_VAD_FRAME = 512  # Silero fixed 32 ms @ 16 kHz
_VAD_FRAME_SEC = _VAD_FRAME / 16_000

# Number of mel frames at the buffer tail whose values can still change due to
# STFT center-padding (ceil of n_fft/2 / hop); held back (not fed) until
# is_final so we wait for later audio, keeping every mel frame consistent with
# the offline result.
_MEL_HOLDBACK = 2

# Extra left-context frames the incremental mel (_mel_grow) pulls in when
# recomputing the tail: must be >= _MEL_HOLDBACK and keep the retained frames'
# windows clear of the preemphasis-contaminated first sample of the slice.
_MEL_LCTX = 4

# Sentence end for the soft-max punctuation cut: "." only counts after a word
# of 3+ letters (abbreviations like "U.S." / "Dr." don't trigger a premature
# split); "?" / "!" always count.  Allows trailing closing quotes/brackets.
_SENT_END_RE = re.compile(r"(?:[A-Za-z]{3,}\.|[!?])[\"'”’)\]]*\s*$")

# Audio back-off for the soft-max cut.  RNNT token timestamps lag acoustics
# (emission delay, worst for punctuation), so cutting at the punctuation
# token's own timestamp bleeds the next sentence's first word into the head.
# Anchor on the NEXT token's start and back off by this much.
_SPLIT_BACKOFF_SEC = 0.32

# Default VAD endpointing thresholds (ported from livecaption config).
_DEFAULT_VAD_THRESHOLD = 0.5
_DEFAULT_VAD_PRE_ROLL_MS = 320
_DEFAULT_RULE1_SILENCE = 2.4
_DEFAULT_RULE2_PUNCT_SILENCE = 0.6
_DEFAULT_RULE2_SILENCE = 1.2
_DEFAULT_RULE2_SOFT_MAX = 8.0
_DEFAULT_RULE3_MAX = 20.0


# ---------------------------------------------------------------------------
# Streaming encoder (ported from livecaption _StreamingEncoder)
# ---------------------------------------------------------------------------


class _StreamingEncoder:
    """Cache-aware push-based stepper for the FastConformer encoder.

    State and bookkeeping mirror ``stream_encode`` in
    ``mlx_audio.stt.models.nemotron_asr.streaming`` line by line; the only
    differences are that mel is fed in chunk by chunk from outside, and
    ``is_final`` is decided by the VAD endpointer rather than by "the audio is
    exhausted".
    """

    def __init__(self, model, language: str):  # noqa: ANN001
        enc = model.encoder
        acs = model.default_att_context_size
        self.model = model
        self.enc = enc
        self.language = language
        self.left_cache = int(acs[0])
        self.right = int(acs[1])
        self.sf = enc.args.subsampling_factor
        self.chunk_mel = (self.right + 1) * self.sf  # mel frames consumed per step
        self.conv_left = enc.args.conv_kernel_size - 1
        self.reset()

    def reset(self) -> None:
        n = len(self.enc.layers)
        self._attn_cache = [None] * n
        self._conv_cache = [None] * n
        self._mel_cache = None
        self._emitted = 0
        self._consumed = 0

    def step(self, m: mx.array, is_final: bool):
        """Feed (1, k, F) new mel frames (k <= chunk_mel), return prompted
        encoder frames or None."""
        from mlx_audio.stt.models.nemotron_asr.streaming import (
            _PRE_ENCODE_MEL_CACHE,
            _stream_block,
        )

        enc = self.enc
        cache_len = 0 if self._mel_cache is None else self._mel_cache.shape[1]
        win = m if self._mel_cache is None else mx.concatenate([self._mel_cache, m], axis=1)
        win_len = win.shape[1]
        sub = enc.pre_encode(win, mx.array([win_len], dtype=mx.int32))[0]

        end = self._consumed + m.shape[1]
        base = (self._consumed - cache_len) // self.sf
        lo = self._emitted - base
        hi = sub.shape[1] if is_final else (end // self.sf - base)
        self._consumed = end
        self._mel_cache = win[:, -_PRE_ENCODE_MEL_CACHE:]

        if hi <= lo:
            self._emitted = base + max(lo, hi)
            return None
        self._emitted = base + hi
        h = sub[:, lo:hi]
        for li, block in enumerate(enc.layers):
            h, self._attn_cache[li], self._conv_cache[li] = _stream_block(
                block,
                h,
                enc.pos_enc,
                self._attn_cache[li],
                self._conv_cache[li],
                self.left_cache,
                self.conv_left,
            )
        return self.model.apply_prompt(h, self.language)


# ---------------------------------------------------------------------------
# Model holder
# ---------------------------------------------------------------------------


class NemotronMLXASR:
    """Model holder: loads the nemotron transducer + Silero VAD once and keeps
    them alive for the lifetime of the server.  No decode state."""

    sep = ""  # nemotron emits punctuated text; tokens are subword pieces concatenated
    SAMPLING_RATE = 16_000
    backend_choice = "nemotron-mlx-asr"

    def __init__(self, logfile=sys.stderr, **kwargs):
        self.logfile = logfile
        self.transcribe_kargs = {}

        lan = kwargs.get("lan", "auto")

        self.model_id = kwargs.get("nemotron_mlx_asr_model", "nvidia/nemotron-3.5-asr-streaming-0.6b")
        self.att_context = kwargs.get("nemotron_mlx_asr_att_context", [56, 6])
        self.two_pass = kwargs.get("nemotron_mlx_asr_two_pass", False)

        from mlx_audio.stt import load as load_stt
        from mlx_audio.vad import load as load_vad

        logger.info("Loading Nemotron MLX ASR model '%s' ...", self.model_id)
        self.model = load_stt(self.model_id)

        # Normalize the language against the model's prompt dictionary
        # (mirrors livecaption's normalize_asr_language — the model exposes
        # zh-CN/zh-ZH, not zh; en-US, not en).
        known = getattr(self.model, "prompt_dictionary", None) or {}
        self.original_language = _normalize_language(lan, known) if known else (
            None if lan.casefold() == "auto" else lan
        )
        if self.original_language == "auto":
            self.original_language = None

        # Override the default [56, 13] (a 1.12s refresh is too sluggish); left
        # determines the cache length, right+1 is the feed chunk size.
        self.model.default_att_context_size = list(self.att_context)

        logger.info("Loading Silero VAD ...")
        self.vad = load_vad("mlx-community/silero-vad")

        self._warmup()

    def _warmup(self) -> None:
        """Run each model once on empty input to absorb Metal kernel compilation
        at startup (otherwise the first inference stalls an extra few hundred ms)."""
        silence = np.zeros(int(0.5 * self.SAMPLING_RATE), dtype=np.float32)
        self.model.generate(
            mx.array(silence),
            language=self.original_language,
            att_context_size=list(self.att_context),
        )
        self.vad.feed(
            silence[:_VAD_FRAME],
            self.vad.initial_state(sample_rate=self.SAMPLING_RATE),
            sample_rate=self.SAMPLING_RATE,
        )
        mx.clear_cache()

    def transcribe(self, audio):
        pass  # all work happens in the online processor


# ---------------------------------------------------------------------------
# Online processor
# ---------------------------------------------------------------------------


class NemotronMLXOnlineProcessor:
    """Streaming processor: incrementally encodes audio and greedily decodes
    the RNN-T transducer, with Silero VAD endpointing.

    Lifecycle (called by ``AudioProcessor.transcription_processor``)::

        insert_audio_chunk(pcm, time)  →  process_iter()  →  get_buffer()
                      ... repeat ...
        start_silence() / end_silence()
        finish()

    The VAD endpointer (rule1/2/3 + soft-max) is internal — the processor is
    self-contained and does not depend on WLK's external VAC.  ``start_silence``
    (from the VAC) acts as a secondary finalize trigger.
    """

    SAMPLING_RATE = 16_000

    def __init__(self, asr: NemotronMLXASR, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.end = 0.0
        self.audio_buffer = np.array([], dtype=np.float32)  # diagnostic

        self._model = asr.model
        self._vad = asr.vad
        self._vad_state = self._vad.initial_state(sample_rate=self.SAMPLING_RATE)
        self._vad_leftover = np.empty(0, dtype=np.float32)

        self._pre = self._model.preprocessor_config
        self._frame_sec = (
            self._model.encoder_config.subsampling_factor
            * self._pre.hop_length
            / self._pre.sample_rate
        )
        self._two_pass = asr.two_pass

        # VAD endpointing thresholds (from the model holder's config kwargs).
        self._vad_threshold = _DEFAULT_VAD_THRESHOLD
        self._n_preroll = max(1, round(_DEFAULT_VAD_PRE_ROLL_MS / 1000 / _VAD_FRAME_SEC))
        self._rule1_silence = _DEFAULT_RULE1_SILENCE
        self._rule2_punct_silence = _DEFAULT_RULE2_PUNCT_SILENCE
        self._rule2_silence = _DEFAULT_RULE2_SILENCE
        self._rule2_soft_max = _DEFAULT_RULE2_SOFT_MAX
        self._rule3_max = _DEFAULT_RULE3_MAX

        # Pending audio from insert_audio_chunk not yet VAD-processed.
        self._raw_chunks: list[np.ndarray] = []
        self._raw_len = 0

        self._encoder = _StreamingEncoder(self._model, asr.original_language)
        self._reset_utterance()

    # -- state management --

    def _reset_utterance(self) -> None:
        self._active = False
        self._audio: list[np.ndarray] = []
        self._n_samples = 0
        self._mel_consumed = 0
        self._mel_stable: mx.array | None = None
        self._silence_frames = 0
        # RNN-T decode state (corresponds to stream_generate's local variables).
        self._last_token = self._model.blank_id
        self._decoder_hidden = None
        self._hypothesis: list = []
        self._global_time = 0
        self._text = ""
        self._preroll: deque[np.ndarray] = deque(maxlen=self._n_preroll)

    def _reset(self) -> None:
        """Full reset: encoder caches + utterance state."""
        self._encoder.reset()
        self._reset_utterance()
        mx.clear_cache()

    # -- audio ingestion --

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        self.end = audio_stream_end_time
        self.audio_buffer = audio  # diagnostic only
        self._raw_chunks.append(np.asarray(audio, dtype=np.float32))
        self._raw_len += len(audio)

    # -- core processing --

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        try:
            return self._process(is_last=is_last)
        except Exception as e:
            logger.warning("[nemotron-mlx] process_iter error: %s", e, exc_info=True)
            return [], self.end

    def _process(self, is_last: bool = False) -> Tuple[List[ASRToken], float]:
        if self._raw_len == 0 and not is_last:
            return [], self.end

        # Flatten pending audio with the VAD sub-frame leftover.
        if self._raw_chunks:
            buf = np.concatenate([self._vad_leftover, np.concatenate(self._raw_chunks)])
        else:
            buf = self._vad_leftover
        self._raw_chunks = []
        self._raw_len = 0

        n = len(buf) // _VAD_FRAME
        self._vad_leftover = buf[n * _VAD_FRAME:]
        if not n and not is_last:
            return [], self.end

        # Compute all VAD frames of this block in one pass.
        flags: list[bool] = []
        frames: list[np.ndarray] = []
        for i in range(n):
            frame = buf[i * _VAD_FRAME : (i + 1) * _VAD_FRAME]
            prob, self._vad_state = self._vad.feed(
                frame, self._vad_state, sample_rate=self.SAMPLING_RATE
            )
            flags.append(float(prob.reshape(-1)[0]) >= self._vad_threshold)
            frames.append(frame)

        committed: list[ASRToken] = []
        for i, is_speech in enumerate(flags):
            committed += self._on_frame(frames[i], is_speech)

        if is_last and self._active:
            committed += self._finalize()

        return committed, self.end

    def _on_frame(self, frame: np.ndarray, is_speech: bool) -> list[ASRToken]:
        """VAD state machine: IDLE → ACTIVE → finalize."""
        if not self._active:
            self._preroll.append(frame)
            if not is_speech:
                return []
            # Speech onset: merge the entire pre-roll into this sentence.
            self._active = True
            for f in self._preroll:
                self._audio.append(f)
                self._n_samples += len(f)
            self._preroll.clear()
            self._drive(final=False)
            return []

        self._audio.append(frame)
        self._n_samples += len(frame)
        self._silence_frames = 0 if is_speech else self._silence_frames + 1
        self._drive(final=False)

        utt_sec = self._n_samples / self.SAMPLING_RATE
        # Soft max: once the utterance is this long, cut at the most recent
        # decoded sentence-final punctuation, without waiting for silence.
        if utt_sec >= self._rule2_soft_max:
            cut = self._last_sentence_end()
            if cut is not None:
                return self._finalize(split_token=cut)

        silence_sec = self._silence_frames * _VAD_FRAME_SEC
        ends_sentence = self._text.rstrip()[-1:] in ".?!"
        rule2_silence = (
            self._rule2_punct_silence if ends_sentence else self._rule2_silence
        )
        if self._text and silence_sec >= rule2_silence:
            return self._finalize()
        elif not self._text and silence_sec >= self._rule1_silence:
            # rule1: reset only, no final emitted (matches sherpa's empty-text endpoint).
            self._reset()
        elif utt_sec >= self._rule3_max:
            return self._finalize()
        return []

    def _last_sentence_end(self) -> int | None:
        """Index of the most recent hypothesis token that ends a sentence."""
        hyp = self._hypothesis
        for i in range(len(hyp) - 1, -1, -1):
            if not any(c in ".?!" for c in hyp[i].text):
                continue
            prefix = "".join(t.text for t in hyp[max(0, i - 3) : i + 1])
            if _SENT_END_RE.search(prefix):
                return i
        return None

    def _finalize(self, split_token: int | None = None) -> list[ASRToken]:
        """Utterance close: flush held-back mel, build ASRTokens, reset."""
        from mlx_audio.stt.models.nemo.alignment import (
            sentences_to_result,
            tokens_to_sentences,
        )

        tokens = self._hypothesis
        text = self._text
        tail: np.ndarray | None = None

        if split_token is not None and len(tokens) > split_token:
            # Soft-max cut: everything past the sentence end carries to the next
            # utterance; it re-decodes there from the tail audio.
            punct = tokens[split_token]
            nxt = next((t for t in tokens[split_token + 1 :] if t.text.strip()), None)
            anchor = nxt.start if nxt is not None else punct.end
            sec = max(0.0, anchor - _SPLIT_BACKOFF_SEC)
            audio = np.concatenate(self._audio) if self._audio else None
            if audio is not None:
                n_cut = int(sec * self.SAMPLING_RATE)
                if 0 < n_cut < len(audio):
                    tail = audio[n_cut:]
            tokens = tokens[: split_token + 1]
            text = sentences_to_result(tokens_to_sentences(tokens)).text.strip()
        else:
            # No split: flush the held-back tail mel for the final tokens.
            self._drive(final=True)
            tokens = self._hypothesis
            text = self._text

        # Two-pass re-decode (optional accuracy lever; default off).
        if self._two_pass and not split_token:
            audio = np.concatenate(self._audio) if self._audio else None
            if audio is not None and len(audio) > 0:
                result = self._model.generate(
                    mx.array(audio),
                    language=self._encoder.language,
                    att_context_size=list(self.asr.att_context),
                )
                new_text = result.text.strip()
                if new_text:
                    new_tokens = [t for s in result.sentences for t in s.tokens]
                    tokens = new_tokens
                    text = new_text

        # Build ASRTokens with absolute timestamps.
        offset = self._utt_offset()
        asr_tokens = [
            ASRToken(start=t.start + offset, end=t.end + offset, text=t.text)
            for t in tokens
        ]

        # Reset utterance; seed next with the carried-over tail audio.
        self._reset()
        if tail is not None and len(tail):
            self._active = True
            self._audio = [tail]
            self._n_samples = len(tail)

        return asr_tokens if text else []

    def _utt_offset(self) -> float:
        """Absolute stream-time offset of the current utterance's first sample."""
        return self.end - self._n_samples / self.SAMPLING_RATE

    # -- incremental encode + decode --

    def _drive(self, final: bool) -> None:
        """Run encoder+decode steps once enough stable mel frames have
        accumulated (flush everything on final)."""
        hop = self._pre.hop_length
        chunk = self._encoder.chunk_mel
        if not final:
            est = self._n_samples // hop + 1 - _MEL_HOLDBACK
            if est < self._mel_consumed + chunk:
                return

        mel = self._mel_grow(final)  # (1, T, F)
        avail = mel.shape[1] if final else mel.shape[1] - _MEL_HOLDBACK
        while self._mel_consumed + chunk <= avail:
            out = self._encoder.step(
                mel[:, self._mel_consumed : self._mel_consumed + chunk], False
            )
            self._mel_consumed += chunk
            if out is not None:
                self._decode_chunk(out)
        if final and self._mel_consumed < avail:
            out = self._encoder.step(mel[:, self._mel_consumed : avail], True)
            self._mel_consumed = avail
            if out is not None:
                self._decode_chunk(out)

    def _mel_grow(self, final: bool) -> mx.array:
        """Incrementally maintain the whole-sentence mel: reuse the stable
        prefix, only run STFT on the new tail audio."""
        from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram

        hop = self._pre.hop_length
        audio = np.concatenate(self._audio)
        stable = 0 if self._mel_stable is None else self._mel_stable.shape[1]
        ctx = min(stable, _MEL_LCTX)
        tail = log_mel_spectrogram(mx.array(audio[(stable - ctx) * hop :]), self._pre)
        tail = tail[:, ctx:]
        mel = (
            tail
            if self._mel_stable is None
            else mx.concatenate([self._mel_stable, tail], axis=1)
        )
        if not final:
            n_stable = mel.shape[1] - _MEL_HOLDBACK
            if n_stable > stable:
                self._mel_stable = mel[:, :n_stable]
        return mel

    def _decode_chunk(self, prompted: mx.array) -> None:
        """Greedy RNN-T decode of one block of encoder output.

        Ported from stream_generate's inner loop.  Each non-blank emission
        appends an ``AlignedToken`` with ``start = (global_time + time) *
        frame_sec`` — the load-bearing line that gives a time-based accessible
        boundary mid-decode.
        """
        from mlx_audio.stt.models.nemo.alignment import (
            AlignedToken,
            sentences_to_result,
            tokens_to_sentences,
        )
        from mlx_audio.stt.models.nemotron_asr import tokenizer as tok

        model = self._model
        chunk_len = prompted.shape[1]
        max_symbols = model.max_symbols or 10
        time = 0
        new_symbols = 0
        while time < chunk_len:
            feature = prompted[:, time : time + 1]
            current_token = (
                mx.array([[self._last_token]], dtype=mx.int32)
                if self._last_token != model.blank_id
                else None
            )
            decoder_output, (h, c) = model.decoder(current_token, self._decoder_hidden)
            decoder_output = decoder_output.astype(feature.dtype)
            proposed_hidden = (h.astype(feature.dtype), c.astype(feature.dtype))
            joint_output = model.joint(feature, decoder_output)
            pred_token = int(mx.argmax(joint_output))
            if pred_token != model.blank_id:
                self._last_token = pred_token
                self._decoder_hidden = proposed_hidden
                if not tok.is_special_token(pred_token, model.vocabulary):
                    self._hypothesis.append(
                        AlignedToken(
                            pred_token,
                            start=(self._global_time + time) * self._frame_sec,
                            duration=self._frame_sec,
                            text=tok.decode([pred_token], model.vocabulary),
                        )
                    )
                new_symbols += 1
                if new_symbols >= max_symbols:
                    time += 1
                    new_symbols = 0
            else:
                time += 1
                new_symbols = 0
        self._global_time += chunk_len
        self._text = sentences_to_result(tokens_to_sentences(self._hypothesis)).text.strip()

    # -- interface methods --

    def get_buffer(self) -> Transcript:
        if self._active and self._text:
            return Transcript(
                start=self._utt_offset(),
                end=self.end,
                text=self._text,
            )
        return Transcript(start=None, end=None, text="")

    def finish(self) -> Tuple[List[ASRToken], float]:
        if self._active:
            tokens = self._finalize()
        else:
            tokens = []
        # Also drain any unprocessed raw audio (shouldn't happen normally).
        if self._raw_chunks:
            extra, _ = self._process(is_last=True)
            tokens = tokens + extra
        logger.info("[nemotron-mlx] finish: flushed %d tokens", len(tokens))
        return tokens, self.end

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        """Secondary finalize trigger (from WLK's external VAC)."""
        if self._active:
            tokens = self._finalize()
        else:
            tokens = []
        return tokens, self.end

    def end_silence(self, silence_duration: float, offset: float):
        """Advance the stream clock past a silence gap."""
        self.end += silence_duration

    def new_speaker(self, change_speaker) -> Tuple[List[ASRToken], float]:
        """Flush and return the previous speaker's final tokens before reset."""
        return self.start_silence()

    def warmup(self, audio, init_prompt=""):
        pass
