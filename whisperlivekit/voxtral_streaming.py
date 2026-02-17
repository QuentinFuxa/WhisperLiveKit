"""
Voxtral Mini Realtime streaming backend using voxmlx's incremental encode/decode.

Uses model.encode_step() for incremental audio encoding and token-by-token
autoregressive decoding, matching voxmlx's native streaming pipeline.
"""

import logging
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

from whisperlivekit.timed_objects import ASRToken, Transcript

logger = logging.getLogger(__name__)

N_LEFT_PAD_TOKENS = 32
N_RIGHT_PAD_TOKENS = 17


class VoxtralStreamingASR:
    """Voxtral model holder for the streaming pipeline."""

    sep = " "

    def __init__(self, logfile=sys.stderr, **kwargs):
        from voxmlx import _build_prompt_tokens
        from voxmlx import load_model as vox_load_model

        self.logfile = logfile
        self.transcribe_kargs = {}

        lan = kwargs.get("lan", "auto")
        self.original_language = None if lan == "auto" else lan

        DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"
        model_path = kwargs.get("model_dir") or kwargs.get("model_path")
        if not model_path:
            model_size = kwargs.get("model_size", "")
            # Only use model_size if it looks like a HF repo or a path, not a Whisper size name
            if model_size and ("/" in model_size or model_size.startswith(".")):
                model_path = model_size
            else:
                model_path = DEFAULT_MODEL

        t = time.time()
        logger.info(f"Loading Voxtral model '{model_path}' via voxmlx...")
        self.model, self._tokenizer, self._config = vox_load_model(model_path)
        self._prompt_tokens, self._n_delay_tokens = _build_prompt_tokens(
            self._tokenizer
        )
        logger.info(f"Voxtral model loaded in {time.time() - t:.2f}s")

        self.backend_choice = "voxtral-mlx"
        self.tokenizer = None  # sentence tokenizer — not needed for streaming

    def transcribe(self, audio):
        pass


class VoxtralStreamingOnlineProcessor:
    """
    Online processor for Voxtral streaming ASR.

    Uses voxmlx's incremental encoding (encode_step) and token-by-token
    autoregressive decoding. Each decode step corresponds to 80ms of audio.
    """

    SAMPLING_RATE = 16000

    def __init__(self, asr: VoxtralStreamingASR, logfile=sys.stderr):
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

        self.asr = asr
        self.logfile = logfile
        self.end = 0.0
        self.buffer = []
        self.audio_buffer = np.array([], dtype=np.float32)  # for logging compat
        self._special_token_policy = SpecialTokenPolicy.IGNORE
        self._reset_state()
        logger.info(
            f"[voxtral] Initialized. eos_id={asr._tokenizer.eos_id}, "
            f"prefix_len={len(asr._prompt_tokens)}, "
            f"n_delay={asr._n_delay_tokens}"
        )

    def _reset_state(self):
        from voxmlx.audio import SAMPLES_PER_TOKEN

        self._samples_per_token = SAMPLES_PER_TOKEN

        # Incremental encoder state
        self._audio_tail = None
        self._conv1_tail = None
        self._conv2_tail = None
        self._encoder_cache = None
        self._ds_buf = None

        # Decoder state
        self._decoder_cache = None
        self._y = None  # last sampled token (mx.array scalar)
        self._t_cond = None
        self._text_embeds = None

        # Audio / decode tracking
        self._pending_audio = np.zeros(0, dtype=np.float32)
        self._audio_embeds = None
        self._n_audio_samples_fed = 0
        self._n_total_decoded = 0
        self._first_cycle = True
        self._prefilled = False

        # Word extraction: accumulate token IDs, full-sequence decode for correct spacing
        self._output_token_ids: List[int] = []
        self._token_positions: List[int] = []  # decode position for each token
        self._n_committed_words = 0
        self._global_time_offset = 0.0
        self._y_flushed_to_output = False  # True after start_silence flushes pending _y

    # ── Interface methods (same as SimulStreamingOnlineProcessor) ──

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        self.end = audio_stream_end_time
        self._pending_audio = np.append(self._pending_audio, audio)
        self.audio_buffer = self._pending_audio  # for logging compat

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        try:
            return self._process_iter_inner(is_last)
        except Exception as e:
            logger.warning(f"[voxtral] process_iter exception: {e}", exc_info=True)
            return [], self.end

    def _get_full_text(self) -> str:
        """Decode all accumulated token IDs at once for correct spacing."""
        if not self._output_token_ids:
            return ""
        sp = self.asr._tokenizer
        return sp.decode(self._output_token_ids, special_token_policy=self._special_token_policy)

    def get_buffer(self) -> Transcript:
        """Return all uncommitted text as buffer, including pending _y token."""
        # Temporarily include pending _y for buffer display
        ids = list(self._output_token_ids)
        if self._y is not None and not self._y_flushed_to_output:
            sp = self.asr._tokenizer
            token_id = self._y.item()
            if token_id != sp.eos_id:
                ids.append(token_id)
        if not ids:
            return Transcript(start=None, end=None, text="")
        sp = self.asr._tokenizer
        full_text = sp.decode(ids, special_token_policy=self._special_token_policy)
        words = full_text.split()
        uncommitted = words[self._n_committed_words:]
        if uncommitted:
            text = " ".join(uncommitted)
            return Transcript(start=self.end, end=self.end, text=text)
        return Transcript(start=None, end=None, text="")

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        """Flush all uncommitted words when silence starts."""
        self._flush_last_y()  # Include the pending _y token before flushing
        words = self._flush_all_pending_words()
        logger.info(f"[voxtral] start_silence: flushed {len(words)} words")
        return words, self.end

    def end_silence(self, silence_duration: float, offset: float):
        self._global_time_offset += silence_duration
        self.end += silence_duration

    def new_speaker(self, change_speaker):
        self.start_silence()

    def warmup(self, audio, init_prompt=""):
        pass

    def finish(self) -> Tuple[List[ASRToken], float]:
        """Flush remaining audio with right-padding to let the model finish decoding."""
        right_pad = np.zeros(
            N_RIGHT_PAD_TOKENS * self._samples_per_token, dtype=np.float32
        )
        self._pending_audio = np.append(self._pending_audio, right_pad)
        self._n_audio_samples_fed += len(right_pad)

        final_words, _ = self._process_iter_inner(is_last=True)
        # Flush the last pending self._y token (like voxmlx's finally block)
        self._flush_last_y()
        final_words.extend(self._flush_all_pending_words())
        return final_words, self.end

    # ── Word extraction ──

    def _pos_to_time(self, pos: int) -> float:
        """Convert a decode position to seconds relative to audio start."""
        SPT = self._samples_per_token
        return max(0.0, (pos - N_LEFT_PAD_TOKENS) * SPT / self.SAMPLING_RATE)

    def _flush_last_y(self):
        """Flush the last pending self._y token that hasn't been processed yet."""
        if self._y is None or self._y_flushed_to_output:
            return
        sp = self.asr._tokenizer
        token_id = self._y.item()
        if token_id != sp.eos_id:
            self._output_token_ids.append(token_id)
            self._token_positions.append(self._n_total_decoded)
            self._y_flushed_to_output = True

    def _extract_new_words(self) -> List[ASRToken]:
        """
        Split accumulated text into words and return new complete words
        (all but the last, which may still be growing).
        """
        if not self._output_token_ids:
            return []

        full_text = self._get_full_text()
        words = full_text.split()

        new_words: List[ASRToken] = []
        n_tokens = len(self._output_token_ids)
        # All words except the last are guaranteed complete
        while len(words) > self._n_committed_words + 1:
            word = words[self._n_committed_words]
            word_idx = self._n_committed_words
            n_words_total = len(words)
            # Approximate: assign token range proportionally
            tok_start = int(word_idx / n_words_total * n_tokens)
            tok_end = int((word_idx + 1) / n_words_total * n_tokens)
            tok_start = min(tok_start, len(self._token_positions) - 1)
            tok_end = min(tok_end, len(self._token_positions) - 1)

            start_time = self._pos_to_time(self._token_positions[tok_start]) + self._global_time_offset
            end_time = self._pos_to_time(self._token_positions[tok_end]) + self._global_time_offset

            # Prepend space to match Whisper convention (Segment.from_tokens joins with '')
            text = word if self._n_committed_words == 0 else " " + word
            new_words.append(ASRToken(start=start_time, end=end_time, text=text))
            self._n_committed_words += 1

        return new_words

    def _flush_all_pending_words(self) -> List[ASRToken]:
        """Flush ALL words including the last partial one."""
        if not self._output_token_ids:
            return []

        full_text = self._get_full_text()
        words = full_text.split()

        new_words: List[ASRToken] = []
        n_tokens = len(self._output_token_ids)
        n_words_total = max(len(words), 1)

        while self._n_committed_words < len(words):
            word = words[self._n_committed_words]
            word_idx = self._n_committed_words

            tok_start = int(word_idx / n_words_total * n_tokens)
            tok_end = int((word_idx + 1) / n_words_total * n_tokens)
            tok_start = min(tok_start, max(len(self._token_positions) - 1, 0))
            tok_end = min(tok_end, max(len(self._token_positions) - 1, 0))

            if self._token_positions:
                start_time = self._pos_to_time(self._token_positions[tok_start]) + self._global_time_offset
                end_time = self._pos_to_time(self._token_positions[tok_end]) + self._global_time_offset
            else:
                start_time = self._global_time_offset
                end_time = self._global_time_offset

            # Prepend space to match Whisper convention (Segment.from_tokens joins with '')
            text = word if self._n_committed_words == 0 else " " + word
            new_words.append(ASRToken(start=start_time, end=end_time, text=text))
            self._n_committed_words += 1

        return new_words

    # ── Core streaming logic ──

    def _process_iter_inner(self, is_last: bool) -> Tuple[List[ASRToken], float]:
        import mlx.core as mx

        from voxmlx.audio import log_mel_spectrogram_step
        from voxmlx.cache import RotatingKVCache

        model = self.asr.model
        sp = self.asr._tokenizer
        prompt_tokens = self.asr._prompt_tokens
        prefix_len = len(prompt_tokens)
        SPT = self._samples_per_token

        # ── Phase 1: Encode new audio ──
        if self._first_cycle and len(self._pending_audio) >= SPT:
            left_pad = np.zeros(N_LEFT_PAD_TOKENS * SPT, dtype=np.float32)
            n_feed = (len(self._pending_audio) // SPT) * SPT
            chunk = np.concatenate([left_pad, self._pending_audio[:n_feed]])
            self._pending_audio = self._pending_audio[n_feed:]
            self._n_audio_samples_fed += n_feed

            mel, self._audio_tail = log_mel_spectrogram_step(
                chunk, self._audio_tail
            )
            (
                new_embeds,
                self._conv1_tail,
                self._conv2_tail,
                self._encoder_cache,
                self._ds_buf,
            ) = model.encode_step(
                mel,
                self._conv1_tail,
                self._conv2_tail,
                self._encoder_cache,
                self._ds_buf,
            )
            if new_embeds is not None:
                mx.eval(new_embeds)
                self._audio_embeds = new_embeds
                logger.info(f"[voxtral] first encode: {new_embeds.shape[0]} embeds from {n_feed} samples")
            else:
                logger.info(f"[voxtral] first encode: no embeds from {n_feed} samples")
            self._first_cycle = False

        elif not self._first_cycle and len(self._pending_audio) >= SPT:
            n_feed = (len(self._pending_audio) // SPT) * SPT
            chunk = self._pending_audio[:n_feed]
            self._pending_audio = self._pending_audio[n_feed:]
            self._n_audio_samples_fed += n_feed

            mel, self._audio_tail = log_mel_spectrogram_step(
                chunk, self._audio_tail
            )
            (
                new_embeds,
                self._conv1_tail,
                self._conv2_tail,
                self._encoder_cache,
                self._ds_buf,
            ) = model.encode_step(
                mel,
                self._conv1_tail,
                self._conv2_tail,
                self._encoder_cache,
                self._ds_buf,
            )
            if new_embeds is not None:
                mx.eval(new_embeds)
                if self._audio_embeds is not None:
                    self._audio_embeds = mx.concatenate(
                        [self._audio_embeds, new_embeds]
                    )
                else:
                    self._audio_embeds = new_embeds

        self.audio_buffer = self._pending_audio  # for logging compat

        if self._audio_embeds is None:
            return [], self.end

        # Safety: don't decode ahead of encoded audio
        safe_total = (
            N_LEFT_PAD_TOKENS + self._n_audio_samples_fed // SPT
        )
        n_decodable = min(
            self._audio_embeds.shape[0], safe_total - self._n_total_decoded
        )

        if n_decodable <= 0:
            return [], self.end

        # ── Phase 2: Prefill (once per utterance) ──
        if not self._prefilled:
            if self._n_total_decoded + self._audio_embeds.shape[0] < prefix_len:
                logger.info(
                    f"[voxtral] waiting for prefill: have {self._audio_embeds.shape[0]} embeds, need {prefix_len}"
                )
                return [], self.end

            n_layers = len(model.language_model.layers)
            self._decoder_cache = [RotatingKVCache(8192) for _ in range(n_layers)]

            self._t_cond = model.time_embedding(
                mx.array([self.asr._n_delay_tokens], dtype=mx.float32)
            )

            prompt_ids = mx.array([prompt_tokens])
            self._text_embeds = model.language_model.embed(prompt_ids)[0]

            prefix_embeds = (
                self._text_embeds + self._audio_embeds[:prefix_len]
            )[None, :, :]

            logits = model.decode(
                prefix_embeds, self._t_cond, "causal", self._decoder_cache
            )
            mx.eval(
                logits,
                *[x for c in self._decoder_cache for x in (c.keys, c.values)],
            )

            self._y = mx.argmax(logits[0, -1:], axis=-1).squeeze()
            mx.async_eval(self._y)

            self._audio_embeds = self._audio_embeds[prefix_len:]
            self._n_total_decoded = prefix_len
            self._prefilled = True
            logger.info(f"[voxtral] prefill done, first token y={self._y.item()}")

            n_decodable = min(
                self._audio_embeds.shape[0], safe_total - self._n_total_decoded
            )

        if n_decodable <= 0:
            return [], self.end

        # ── Phase 3: Decode new positions ──
        eos_id = sp.eos_id
        hit_eos = False
        n_consumed = 0

        for i in range(n_decodable):
            token_embed = model.language_model.embed(self._y.reshape(1, 1))[0, 0]
            step_embed = (self._audio_embeds[i] + token_embed)[None, None, :]
            logits = model.decode(
                step_embed, self._t_cond, mask=None, cache=self._decoder_cache
            )
            next_y = mx.argmax(logits[0, -1:], axis=-1).squeeze()
            mx.async_eval(next_y)

            token_id = self._y.item()
            n_consumed = i + 1

            if token_id == eos_id:
                hit_eos = True
                logger.info("[voxtral] hit EOS")
                break

            # Accumulate token ID — full-sequence decode produces correct spacing
            # Skip if this _y was already flushed by start_silence()
            if self._y_flushed_to_output:
                self._y_flushed_to_output = False
            else:
                self._output_token_ids.append(token_id)
                # Track position for timestamp estimation
                pos = self._n_total_decoded + i
                self._token_positions.append(pos)

            if i > 0 and i % 256 == 0:
                mx.clear_cache()

            self._y = next_y

        self._n_total_decoded += n_consumed

        # Trim consumed embeddings
        if self._audio_embeds.shape[0] > n_consumed:
            self._audio_embeds = self._audio_embeds[n_consumed:]
        else:
            self._audio_embeds = None

        # Log decode results
        full_text = self._get_full_text()
        logger.info(
            f"[voxtral] decoded {n_consumed} tokens | "
            f"total_decoded={self._n_total_decoded} | "
            f"text='{full_text[-80:]}' | "
            f"n_words={len(full_text.split())} committed={self._n_committed_words}"
        )

        # Extract complete words from the decoded token sequence
        new_words = self._extract_new_words()

        if hit_eos:
            new_words.extend(self._flush_all_pending_words())
            self._reset_state()

        if new_words:
            logger.info(f"[voxtral] returning {len(new_words)} words: {[w.text for w in new_words]}")

        self.buffer = []
        return new_words, self.end
