"""
Pure-MLX Voxtral Realtime ASR backend for WhisperLiveKit.

Provides ``VoxtralMLXASR`` (model holder) and ``VoxtralMLXOnlineProcessor``
(streaming processor) that plug into WhisperLiveKit's audio processing
pipeline via ``insert_audio_chunk`` / ``process_iter`` / ``get_buffer`` etc.

Unlike the HuggingFace backend, this runs the full inference loop in-process
(no background thread / queue) — MLX operations on Apple Silicon are fast
enough to run synchronously inside ``asyncio.to_thread(process_iter)``.
"""

import logging
import sys
import time
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from whisperlivekit.timed_objects import ASRToken, Transcript
from whisperlivekit.voxtral_mlx.loader import load_voxtral_model, DEFAULT_MODEL_ID
from whisperlivekit.voxtral_mlx.model import SlidingKVCache
from whisperlivekit.voxtral_mlx.spectrogram import (
    SAMPLES_PER_TOKEN,
    LEFT_PAD_TOKENS,
    RIGHT_PAD_TOKENS,
    compute_mel_streaming,
)

logger = logging.getLogger(__name__)

# Decoder sliding-window size (matches the model's training configuration).
_DECODER_WINDOW = 8192


def _prompt_tokens(tokenizer, n_left_pad=LEFT_PAD_TOKENS, n_delay=6):
    """Build the prompt token sequence and return ``(token_ids, n_delay)``."""
    pad_id = tokenizer.get_special_token("[STREAMING_PAD]")
    ids = [tokenizer.bos_id] + [pad_id] * (n_left_pad + n_delay)
    return ids, n_delay


# ---------------------------------------------------------------------------
# Model holder
# ---------------------------------------------------------------------------


class VoxtralMLXASR:
    """Lightweight model holder — loads the MLX Voxtral model once and keeps
    it alive for the lifetime of the server."""

    sep = " "
    SAMPLING_RATE = 16_000

    def __init__(self, logfile=sys.stderr, **kwargs):
        self.logfile = logfile
        self.transcribe_kargs = {}

        lan = kwargs.get("lan", "auto")
        self.original_language = None if lan == "auto" else lan

        model_path = kwargs.get("model_dir") or kwargs.get("model_path")
        if not model_path:
            model_size = kwargs.get("model_size", "")
            if model_size and ("/" in model_size or model_size.startswith(".")):
                model_path = model_size
            else:
                model_path = DEFAULT_MODEL_ID

        t0 = time.time()
        logger.info("Loading Voxtral MLX model '%s' ...", model_path)
        self.model, self.tokenizer, self.config = load_voxtral_model(model_path)
        logger.info("Voxtral MLX model loaded in %.2fs", time.time() - t0)

        self.backend_choice = "voxtral-mlx"

    def transcribe(self, audio):
        pass  # all work happens in the online processor


# ---------------------------------------------------------------------------
# Online processor
# ---------------------------------------------------------------------------


class VoxtralMLXOnlineProcessor:
    """Streaming processor that incrementally encodes audio and decodes text
    using the MLX Voxtral model.

    Lifecycle (called by ``AudioProcessor.transcription_processor``):

        insert_audio_chunk(pcm, time)  →  process_iter()  →  get_buffer()
                      ... repeat ...
        start_silence() / end_silence()
        finish()
    """

    SAMPLING_RATE = 16_000

    def __init__(self, asr: VoxtralMLXASR, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.end = 0.0
        self.buffer: list = []
        self.audio_buffer = np.array([], dtype=np.float32)

        self._model = asr.model
        self._tokenizer = asr.tokenizer

        # Pre-compute prompt tokens and delay conditioning (constant across utterances).
        self._prompt_ids, self._n_delay = _prompt_tokens(self._tokenizer)
        self._prefix_len = len(self._prompt_ids)

        self._delay_cond = self._model.delay_embedding(
            mx.array([self._n_delay], dtype=mx.float32)
        )
        mx.eval(self._delay_cond)

        self._prompt_embeds = self._model.decoder.embed(
            mx.array([self._prompt_ids])
        )[0]  # [prefix_len, dim]
        mx.eval(self._prompt_embeds)

        self._eos_id = self._tokenizer.eos_id
        self._secs_per_token = SAMPLES_PER_TOKEN / self.SAMPLING_RATE
        # The streaming model has an inherent delay: text for audio at position P
        # is generated at decoder position P + n_delay. Compensate timestamps.
        self._delay_secs = self._n_delay * self._secs_per_token

        self._reset_state()

    # -- state management --

    def _reset_state(self):
        """Reset all incremental state for a fresh utterance."""
        # Audio accumulation
        self._pending = np.zeros(0, dtype=np.float32)
        # Mel overlap
        self._mel_overlap: np.ndarray | None = None
        # Encoder incremental state
        self._conv_tail1 = None
        self._conv_tail2 = None
        self._enc_cache = None
        self._ds_remainder = None
        # Audio embeddings not yet decoded
        self._audio_embeds: mx.array | None = None
        # Decoder state
        self._dec_cache: list[SlidingKVCache] | None = None
        self._last_token: mx.array | None = None
        # Bookkeeping
        self._samples_encoded = 0
        self._positions_decoded = 0
        self._prefilled = False
        self._first_chunk = True
        # Text state
        self._full_text = ""
        self._n_text_tokens = 0
        self._n_committed_words = 0
        self._time_offset = 0.0
        # Per-word audio position tracking: decoder position (relative to prefix)
        # where each word in _full_text started and ended
        self._word_audio_starts: list[int] = []   # audio pos where word i started
        self._word_audio_ends: list[int] = []     # audio pos where word i last produced a token
        self._current_word_pos: Optional[int] = None  # audio pos of current (incomplete) word's first token

    # -- audio ingestion --

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        self.end = audio_stream_end_time
        self._pending = np.append(self._pending, audio)
        self.audio_buffer = self._pending

    # -- core processing --

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        try:
            return self._step(is_last)
        except Exception as e:
            logger.warning("[voxtral-mlx] process_iter error: %s", e, exc_info=True)
            return [], self.end

    def _step(self, is_last: bool) -> Tuple[List[ASRToken], float]:
        # 1. Encode any new audio
        self._encode_pending()

        if self._audio_embeds is None:
            return [], self.end

        # 2. Compute how many positions we can safely decode
        total_safe = LEFT_PAD_TOKENS + self._samples_encoded // SAMPLES_PER_TOKEN
        n_available = self._audio_embeds.shape[0]
        n_decodable = min(n_available, total_safe - self._positions_decoded)

        if n_decodable <= 0:
            return [], self.end

        # 3. Prefill if needed
        if not self._prefilled:
            if self._positions_decoded + n_available < self._prefix_len:
                return [], self.end
            self._do_prefill()
            # Re-check after consuming prefix embeddings
            n_available = self._audio_embeds.shape[0] if self._audio_embeds is not None else 0
            n_decodable = min(n_available, total_safe - self._positions_decoded)

        if n_decodable <= 0 or self._audio_embeds is None:
            return [], self.end

        # 4. Decode available positions
        hit_eos = self._decode_positions(n_decodable)

        if hit_eos:
            # Flush words, reset for next utterance
            words = self._flush_all_words()
            logger.debug(
                "[voxtral-mlx] EOS hit during stream: flushed %d words, "
                "samples_encoded=%d (%.2fs), text='%s'",
                len(words), self._samples_encoded,
                self._samples_encoded / self.SAMPLING_RATE,
                self._full_text[-60:] if self._full_text else "",
            )
            saved_offset = self._time_offset
            self._reset_state()
            self._time_offset = saved_offset
            return words, self.end

        # 5. Extract committed words (all but the last, which may still grow)
        return self._extract_committed_words(), self.end

    def _encode_pending(self):
        """Feed pending audio through the incremental encoder."""
        available = len(self._pending)
        if available < SAMPLES_PER_TOKEN:
            return

        if self._first_chunk:
            # First chunk: prepend silence for left-padding
            n_take = (available // SAMPLES_PER_TOKEN) * SAMPLES_PER_TOKEN
            left_pad = np.zeros(LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN, dtype=np.float32)
            chunk = np.concatenate([left_pad, self._pending[:n_take]])
            self._pending = self._pending[n_take:]
            self._samples_encoded += n_take
            self._first_chunk = False
        else:
            n_take = (available // SAMPLES_PER_TOKEN) * SAMPLES_PER_TOKEN
            chunk = self._pending[:n_take]
            self._pending = self._pending[n_take:]
            self._samples_encoded += n_take

        mel, self._mel_overlap = compute_mel_streaming(chunk, self._mel_overlap)

        embeds, self._conv_tail1, self._conv_tail2, self._enc_cache, self._ds_remainder = (
            self._model.encode_incremental(
                mel, self._conv_tail1, self._conv_tail2, self._enc_cache, self._ds_remainder
            )
        )

        if embeds is not None:
            mx.eval(embeds)
            if self._audio_embeds is not None:
                self._audio_embeds = mx.concatenate([self._audio_embeds, embeds])
            else:
                self._audio_embeds = embeds

        self.audio_buffer = self._pending

    def _do_prefill(self):
        """Run the decoder prefill pass over the prompt + first audio embeddings."""
        n_dec_layers = len(self._model.decoder.blocks)
        self._dec_cache = [SlidingKVCache(_DECODER_WINDOW) for _ in range(n_dec_layers)]

        prefix_embeds = self._prompt_embeds + self._audio_embeds[: self._prefix_len]
        prefix_embeds = prefix_embeds[None, :, :]  # [1, prefix_len, dim]

        logits = self._model.decode(prefix_embeds, self._delay_cond, "causal", self._dec_cache)
        mx.eval(logits, *[x for c in self._dec_cache for x in (c.keys, c.values)])

        self._last_token = self._sample(logits)
        mx.async_eval(self._last_token)

        # Remove consumed prefix embeddings
        self._audio_embeds = self._audio_embeds[self._prefix_len :]
        if self._audio_embeds.shape[0] == 0:
            self._audio_embeds = None
        self._positions_decoded = self._prefix_len
        self._prefilled = True

    def _decode_positions(self, n: int) -> bool:
        """Autoregressively decode *n* positions.  Returns True on EOS."""
        base_pos = self._positions_decoded  # absolute position before this batch
        for i in range(n):
            tok_embed = self._model.decoder.embed(self._last_token.reshape(1, 1))[0, 0]
            combined = (self._audio_embeds[i] + tok_embed)[None, None, :]
            logits = self._model.decode(combined, self._delay_cond, mask=None, cache=self._dec_cache)
            next_tok = self._sample(logits)
            mx.async_eval(next_tok)

            token_id = self._last_token.item()
            if token_id == self._eos_id:
                # Close the current word if one is being built
                if self._current_word_pos is not None:
                    self._word_audio_ends.append(base_pos + i - self._prefix_len)
                    self._current_word_pos = None
                self._trim_embeds(i)
                self._positions_decoded += i
                return True

            text = self._tokenizer.decode(
                [token_id], special_token_policy=SpecialTokenPolicy.IGNORE
            )

            if text:
                audio_pos = base_pos + i - self._prefix_len

                # Detect word boundary: new word starts with space or is the very first text
                if text.lstrip() != text or not self._full_text:
                    # Close previous word if exists
                    if self._current_word_pos is not None:
                        self._word_audio_ends.append(audio_pos)
                    # Start new word
                    self._word_audio_starts.append(audio_pos)
                    self._current_word_pos = audio_pos
                elif self._current_word_pos is None:
                    # First token of first word (no leading space)
                    self._word_audio_starts.append(audio_pos)
                    self._current_word_pos = audio_pos

                self._full_text += text
                self._n_text_tokens += 1

            if i > 0 and i % 256 == 0:
                mx.clear_cache()

            self._last_token = next_tok

        self._positions_decoded += n
        self._trim_embeds(n)
        return False

    def _trim_embeds(self, n_consumed: int):
        if self._audio_embeds is not None and self._audio_embeds.shape[0] > n_consumed:
            self._audio_embeds = self._audio_embeds[n_consumed:]
        else:
            self._audio_embeds = None

    def _sample(self, logits: mx.array) -> mx.array:
        return mx.argmax(logits[0, -1:], axis=-1).squeeze()

    # -- word extraction --

    def _audio_pos_to_time(self, pos: int) -> float:
        """Convert an audio position (relative to prefix end) to seconds."""
        return max(0.0, pos * self._secs_per_token - self._delay_secs + self._time_offset)

    def _word_time_range(self, word_idx: int, n_words: int) -> Tuple[float, float]:
        """Compute (start, end) time for a word using tracked word positions."""
        starts = self._word_audio_starts
        ends = self._word_audio_ends

        if not starts:
            return self._time_offset, self._time_offset

        # Get start position for this word
        if word_idx < len(starts):
            t0 = self._audio_pos_to_time(starts[word_idx])
        else:
            # Fallback: estimate from last known position
            last_pos = ends[-1] if ends else starts[-1]
            t0 = self._audio_pos_to_time(last_pos + 1)

        # Get end position: use the start of the next word, or the end of this word
        if word_idx + 1 < len(starts):
            t1 = self._audio_pos_to_time(starts[word_idx + 1])
        elif word_idx < len(ends):
            t1 = self._audio_pos_to_time(ends[word_idx] + 1)
        else:
            # Last word, still being built: use last known position + 1 token
            last_pos = starts[word_idx] if word_idx < len(starts) else (ends[-1] if ends else 0)
            t1 = self._audio_pos_to_time(last_pos + 1)

        return t0, t1

    def _extract_committed_words(self) -> List[ASRToken]:
        """Return complete words (all except the last which may still grow)."""
        if not self._full_text:
            return []
        words = self._full_text.split()
        tokens: List[ASRToken] = []
        n_total = max(len(words), 1)

        while len(words) > self._n_committed_words + 1:
            w = words[self._n_committed_words]
            idx = self._n_committed_words
            t0, t1 = self._word_time_range(idx, n_total)
            label = w if idx == 0 else " " + w
            tokens.append(ASRToken(start=t0, end=t1, text=label))
            self._n_committed_words += 1

        return tokens

    def _flush_all_words(self) -> List[ASRToken]:
        """Flush every word including the last partial one."""
        if not self._full_text:
            return []
        words = self._full_text.split()
        tokens: List[ASRToken] = []
        n_total = max(len(words), 1)

        while self._n_committed_words < len(words):
            w = words[self._n_committed_words]
            idx = self._n_committed_words
            t0, t1 = self._word_time_range(idx, n_total)
            label = w if idx == 0 else " " + w
            tokens.append(ASRToken(start=t0, end=t1, text=label))
            self._n_committed_words += 1

        return tokens

    # -- interface methods --

    def get_buffer(self) -> Transcript:
        if not self._full_text:
            return Transcript(start=None, end=None, text="")
        words = self._full_text.split()
        remaining = words[self._n_committed_words :]
        if remaining:
            return Transcript(start=self.end, end=self.end, text=" ".join(remaining))
        return Transcript(start=None, end=None, text="")

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        words = self._flush_all_words()
        logger.info("[voxtral-mlx] start_silence: flushed %d words", len(words))
        return words, self.end

    def end_silence(self, silence_duration: float, offset: float):
        self._time_offset += silence_duration
        self.end += silence_duration

    def new_speaker(self, change_speaker):
        self.start_silence()

    def warmup(self, audio, init_prompt=""):
        pass

    def finish(self) -> Tuple[List[ASRToken], float]:
        logger.debug(
            "[voxtral-mlx] finish: pending=%d samples, audio_embeds=%s, "
            "samples_encoded=%d, positions_decoded=%d, prefilled=%s, text so far='%s'",
            len(self._pending),
            self._audio_embeds.shape if self._audio_embeds is not None else None,
            self._samples_encoded,
            self._positions_decoded,
            self._prefilled,
            self._full_text[-80:] if self._full_text else "",
        )

        # Align pending audio to SAMPLES_PER_TOKEN boundary so nothing is lost
        remainder = len(self._pending) % SAMPLES_PER_TOKEN
        if remainder > 0:
            align_pad = SAMPLES_PER_TOKEN - remainder
        else:
            align_pad = 0

        # Add alignment + right-padding silence
        total_pad = align_pad + RIGHT_PAD_TOKENS * SAMPLES_PER_TOKEN
        if total_pad > 0:
            self._pending = np.append(
                self._pending, np.zeros(total_pad, dtype=np.float32)
            )

        # Encode remaining audio (including right-padding)
        self._encode_pending()

        logger.debug(
            "[voxtral-mlx] finish after encode: audio_embeds=%s, pending=%d",
            self._audio_embeds.shape if self._audio_embeds is not None else None,
            len(self._pending),
        )

        hit_eos = False

        # Decode everything that's left from right-padding
        if self._audio_embeds is not None and self._prefilled:
            hit_eos = self._decode_positions(self._audio_embeds.shape[0])
            logger.debug(
                "[voxtral-mlx] finish decode: hit_eos=%s, text='%s'",
                hit_eos, self._full_text[-80:] if self._full_text else "",
            )

        # Flush last token if it wasn't EOS
        if self._last_token is not None:
            tid = self._last_token.item()
            if tid != self._eos_id:
                text = self._tokenizer.decode(
                    [tid], special_token_policy=SpecialTokenPolicy.IGNORE
                )
                if text:
                    last_pos = self._positions_decoded - self._prefix_len
                    # Check if this starts a new word
                    if text.lstrip() != text or not self._full_text:
                        if self._current_word_pos is not None:
                            self._word_audio_ends.append(last_pos)
                        self._word_audio_starts.append(last_pos)
                        self._current_word_pos = last_pos
                    elif self._current_word_pos is None:
                        self._word_audio_starts.append(last_pos)
                        self._current_word_pos = last_pos
                    self._full_text += text
                    self._n_text_tokens += 1

        # Close the last word if still open
        if self._current_word_pos is not None:
            last_pos = self._positions_decoded - self._prefix_len
            self._word_audio_ends.append(last_pos)
            self._current_word_pos = None

        words = self._flush_all_words()
        logger.info("[voxtral-mlx] finish: flushed %d words", len(words))
        return words, self.end
