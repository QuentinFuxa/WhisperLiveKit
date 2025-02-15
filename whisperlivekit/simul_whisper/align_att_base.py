"""Abstract base class for AlignAtt streaming decoders (PyTorch & MLX)."""
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.whisper import DecodingOptions, tokenizer

from .config import AlignAttConfig

DEC_PAD = 50257
logger = logging.getLogger(__name__)


class AlignAttBase(ABC):
    """
    Abstract base class for AlignAtt streaming decoders.

    Provides shared logic for both PyTorch and MLX implementations:
    - Properties (speaker, global_time_offset)
    - Pure-Python methods (warmup, trim_context, refresh_segment, etc.)
    - Template infer() with abstract hooks for tensor-specific operations
    - Post-decode logic (token splitting, timestamped word building)

    Subclasses must implement ~20 abstract methods for tensor-specific ops.
    """

    # === Properties ===

    @property
    def speaker(self):
        return self.state.speaker

    @speaker.setter
    def speaker(self, value):
        self.state.speaker = value

    @property
    def global_time_offset(self):
        return self.state.global_time_offset

    @global_time_offset.setter
    def global_time_offset(self, value):
        self.state.global_time_offset = value

    # === Constructor helpers ===

    def _base_init(self, cfg: AlignAttConfig, model):
        """Common initialization — call from subclass __init__."""
        self.model = model
        self.cfg = cfg
        self.decode_options = DecodingOptions(
            language=cfg.language,
            without_timestamps=True,
            task=cfg.task,
        )
        self.tokenizer_is_multilingual = cfg.tokenizer_is_multilingual
        self.max_text_len = model.dims.n_text_ctx
        self.num_decoder_layers = len(model.decoder.blocks)
        if cfg.max_context_tokens is None:
            self.max_context_tokens = self.max_text_len
        else:
            self.max_context_tokens = cfg.max_context_tokens

    def _init_state_common(self, cfg: AlignAttConfig):
        """Common state initialization — call from subclass _init_state."""
        self.create_tokenizer(cfg.language if cfg.language != "auto" else None)
        self.state.tokenizer = self.tokenizer
        self.state.detected_language = cfg.language if cfg.language != "auto" else None
        self.state.global_time_offset = 0.0
        self.state.last_attend_frame = -cfg.rewind_threshold
        self.state.speaker = -1

    # === Shared concrete methods ===

    def warmup(self, audio):
        try:
            self.insert_audio(audio)
            self.infer(is_last=True)
            self.refresh_segment(complete=True)
            logger.info("Model warmed up successfully")
        except Exception as e:
            logger.exception(f"Model warmup failed: {e}")

    def create_tokenizer(self, language=None):
        self.tokenizer = tokenizer.get_tokenizer(
            multilingual=self.tokenizer_is_multilingual,
            language=language,
            num_languages=self.model.num_languages,
            task=self.decode_options.task,
        )
        self.state.tokenizer = self.tokenizer

    def trim_context(self):
        logger.info("Trimming context")
        c = len(self.state.context.as_token_ids()) - len(self.state.context.prefix_token_ids)
        logger.info(f"Context text: {self.state.context.as_text()}")
        l = sum(t.shape[1] for t in self.state.tokens) + c
        after = 0 if self.cfg.static_init_prompt is None else len(self.cfg.static_init_prompt)
        while c > self.max_context_tokens or l > self.max_text_len - 20:
            t = self.state.context.trim_words(after=after)
            l -= t
            c -= t
            logger.debug(f"len {l}, c {c}, max_context_tokens {self.max_context_tokens}")
            if t == 0:
                break
        logger.info(f"Context after trim: {self.state.context.text} (len: {l})")

    def refresh_segment(self, complete=False):
        logger.debug("Refreshing segment:")
        self.init_tokens()
        self.state.last_attend_frame = -self.cfg.rewind_threshold
        self.state.cumulative_time_offset = 0.0
        self.init_context()
        logger.debug(f"Context: {self.state.context}")
        if not complete and len(self.state.segments) > 2:
            self.state.segments = self.state.segments[-2:]
        else:
            logger.debug("removing all segments.")
            self.state.segments = []
        self.state.log_segments += 1
        self.state.pending_incomplete_tokens = []

    def segments_len(self):
        return sum(s.shape[0] for s in self.state.segments) / 16000

    def _apply_minseglen(self):
        segments_len = self.segments_len()
        if segments_len < self.cfg.audio_min_len:
            logger.debug("waiting for next segment")
            return False
        return True

    def _clean_cache(self):
        self.state.clean_cache()

    def debug_print_tokens(self, tokens):
        for i in range(min(self.cfg.beam_size, tokens.shape[0])):
            logger.debug(self.tokenizer.decode_with_timestamps(tokens[i].tolist()))

    # === Language detection ===

    def _detect_language_if_needed(self, encoder_feature):
        if (
            self.cfg.language == "auto"
            and self.state.detected_language is None
            and self.state.first_timestamp
        ):
            seconds_since_start = self.segments_len() - self.state.first_timestamp
            if seconds_since_start >= 2.0:
                language_tokens, language_probs = self.lang_id(encoder_feature)
                top_lan, p = max(language_probs[0].items(), key=lambda x: x[1])
                print(f"Detected language: {top_lan} with p={p:.4f}")
                self.create_tokenizer(top_lan)
                self.state.last_attend_frame = -self.cfg.rewind_threshold
                self.state.cumulative_time_offset = 0.0
                self.init_tokens()
                self.init_context()
                self.state.detected_language = top_lan
                logger.info(f"Tokenizer language: {self.tokenizer.language}")

    # === Template infer() ===

    def infer(self, is_last=False):
        """Main inference — template method calling abstract hooks for tensor ops."""
        new_segment = True

        if len(self.state.segments) == 0:
            logger.debug("No segments, nothing to do")
            return []
        if not self._apply_minseglen():
            logger.debug(f"applied minseglen {self.cfg.audio_min_len} > {self.segments_len()}.")
            return []

        input_segments = self._concat_segments()
        encoder_feature, content_mel_len = self._encode(input_segments)
        self._evaluate(encoder_feature)

        self._detect_language_if_needed(encoder_feature)
        self.trim_context()
        current_tokens = self._current_tokens()

        fire_detected = self.fire_at_boundary(encoder_feature[:, :content_mel_len, :])

        sum_logprobs = self._init_sum_logprobs()
        completed = False
        token_len_before = current_tokens.shape[1]
        l_absolute_timestamps = []
        accumulated_cross_attns = []

        audio_duration_s = self.segments_len()
        max_tokens = max(50, int(audio_duration_s * 15 * 1.5))
        tokens_produced = 0
        most_attended_frame = None

        while not completed and current_tokens.shape[1] < self.max_text_len:
            tokens_produced += 1
            if tokens_produced > max_tokens:
                logger.warning(
                    f"[Loop Detection] Too many tokens ({tokens_produced}) "
                    f"for {audio_duration_s:.2f}s audio. Breaking."
                )
                current_tokens = current_tokens[:, :token_len_before]
                break

            tokens_for_logits = current_tokens if new_segment else current_tokens[:, -1:]
            logits, cross_attns = self._get_logits_and_cross_attn(
                tokens_for_logits, encoder_feature
            )
            self._evaluate(logits)

            accumulated_cross_attns.append(cross_attns)
            if len(accumulated_cross_attns) > 16:
                accumulated_cross_attns = accumulated_cross_attns[-16:]

            if new_segment and self._check_no_speech(logits):
                break

            logits = logits[:, -1, :]

            if new_segment:
                logits = self._suppress_blank_tokens(logits)
            new_segment = False

            logits = self._apply_token_suppression(logits)
            current_tokens, completed = self._update_tokens(
                current_tokens, logits, sum_logprobs
            )
            self._evaluate(current_tokens)

            logger.debug(f"Decoding completed: {completed}")
            self.debug_print_tokens(current_tokens)

            attn = self._process_cross_attention(accumulated_cross_attns, content_mel_len)
            frames_list, most_attended_frame = self._get_attended_frames(attn)

            absolute_timestamps = [
                (frame * 0.02 + self.state.cumulative_time_offset)
                for frame in frames_list
            ]
            l_absolute_timestamps.append(absolute_timestamps[0])
            logger.debug(f"Absolute timestamps: {absolute_timestamps}")

            if completed:
                current_tokens = current_tokens[:, :-1]
                break

            # Rewind check
            if (
                not is_last
                and self.state.last_attend_frame - most_attended_frame
                > self.cfg.rewind_threshold
            ):
                if current_tokens.shape[1] > 1 and self._is_special_token(current_tokens):
                    logger.debug("omit rewinding from special tokens")
                    self.state.last_attend_frame = most_attended_frame
                else:
                    logger.debug(
                        f"[rewind detected] current: {most_attended_frame}, "
                        f"last: {self.state.last_attend_frame}"
                    )
                    self.state.last_attend_frame = -self.cfg.rewind_threshold
                    current_tokens = self._rewind_tokens()
                    break
            else:
                self.state.last_attend_frame = most_attended_frame

            if content_mel_len - most_attended_frame <= (
                4 if is_last else self.cfg.frame_threshold
            ):
                logger.debug(
                    f"attention reaches the end: {most_attended_frame}/{content_mel_len}"
                )
                current_tokens = current_tokens[:, :-1]
                break

        # Post-decode: split tokens and build timestamped words
        tokens_to_split = self._tokens_to_list(current_tokens, token_len_before)
        if self.state.pending_incomplete_tokens:
            logger.debug(
                f"[UTF-8 Fix] Prepending {len(self.state.pending_incomplete_tokens)} "
                f"pending tokens: {self.state.pending_incomplete_tokens}"
            )
            tokens_to_split = self.state.pending_incomplete_tokens + tokens_to_split

        new_hypothesis, split_words, split_tokens = self._split_tokens(
            tokens_to_split, fire_detected, is_last
        )

        new_tokens_tensor = self._make_new_tokens_tensor(new_hypothesis)
        self.state.tokens.append(new_tokens_tensor)
        logger.info(f"Output: {self.tokenizer.decode(new_hypothesis)}")

        self._clean_cache()

        if len(l_absolute_timestamps) >= 2 and self.state.first_timestamp is None:
            self.state.first_timestamp = l_absolute_timestamps[0]

        timestamped_words = self._build_timestamped_words(
            split_words, split_tokens, l_absolute_timestamps
        )
        self._handle_pending_tokens(split_words, split_tokens)

        return timestamped_words

    # === Post-decode shared helpers ===

    def _split_tokens(self, tokens_list, fire_detected, is_last):
        """Split token list into words. Returns (hypothesis, split_words, split_tokens)."""
        if fire_detected or is_last:
            new_hypothesis = tokens_list
            split_words, split_tokens = self.tokenizer.split_to_word_tokens(new_hypothesis)
        else:
            split_words, split_tokens = self.tokenizer.split_to_word_tokens(tokens_list)
            if len(split_words) > 1:
                new_hypothesis = [i for sublist in split_tokens[:-1] for i in sublist]
            else:
                new_hypothesis = []
        return new_hypothesis, split_words, split_tokens

    def _build_timestamped_words(self, split_words, split_tokens, l_absolute_timestamps):
        """Build list of timestamped ASRToken from split words."""
        timestamped_words = []
        timestamp_idx = 0
        replacement_char = "\ufffd"

        for word, word_tokens in zip(split_words, split_tokens):
            if replacement_char in word:
                logger.warning(f"[UTF-8 Filter] Skipping: {repr(word)}")
                timestamp_idx += len(word_tokens)
                continue

            try:
                current_timestamp = l_absolute_timestamps[timestamp_idx]
            except IndexError:
                logger.warning(
                    f"Timestamp index {timestamp_idx} out of range, using last timestamp"
                )
                current_timestamp = (
                    l_absolute_timestamps[-1] if l_absolute_timestamps else 0.0
                )
            timestamp_idx += len(word_tokens)

            timestamp_entry = ASRToken(
                start=round(current_timestamp, 2),
                end=round(current_timestamp + 0.1, 2),
                text=word,
                speaker=self.state.speaker,
                detected_language=self.state.detected_language,
            ).with_offset(self.state.global_time_offset)
            timestamped_words.append(timestamp_entry)

        return timestamped_words

    def _handle_pending_tokens(self, split_words, split_tokens):
        """Handle incomplete UTF-8 tokens for next chunk."""
        self.state.pending_incomplete_tokens = []
        MAX_PENDING_TOKENS = 10
        replacement_char = "\ufffd"
        if split_words and replacement_char in split_words[-1]:
            if len(split_tokens[-1]) <= MAX_PENDING_TOKENS:
                self.state.pending_incomplete_tokens = split_tokens[-1]
                logger.debug(
                    f"[UTF-8 Fix] Holding {len(self.state.pending_incomplete_tokens)} "
                    f"incomplete tokens for next chunk"
                )
            else:
                logger.warning(
                    f"[UTF-8 Fix] Skipping {len(split_tokens[-1])} tokens "
                    f"(exceeds limit of {MAX_PENDING_TOKENS}, likely hallucination)"
                )

    # === Abstract methods — subclass must implement ===

    @abstractmethod
    def _init_state(self, cfg: AlignAttConfig):
        """Initialize per-session decoder state."""
        ...

    @abstractmethod
    def init_tokens(self):
        """Initialize token sequence with framework-specific tensors."""
        ...

    @abstractmethod
    def init_context(self):
        """Initialize context buffer with framework-specific TokenBuffer."""
        ...

    @abstractmethod
    def insert_audio(self, segment=None):
        """Insert audio segment into buffer."""
        ...

    @abstractmethod
    def _current_tokens(self):
        """Build current token tensor for decoding."""
        ...

    @abstractmethod
    def fire_at_boundary(self, feature):
        """Check if we should fire at word boundary."""
        ...

    @abstractmethod
    def lang_id(self, encoder_features):
        """Language detection from encoder features. Returns (tokens, probs)."""
        ...

    @abstractmethod
    def _concat_segments(self):
        """Concatenate audio segments into single array/tensor."""
        ...

    @abstractmethod
    def _encode(self, input_segments):
        """Encode audio. Returns (encoder_feature, content_mel_len)."""
        ...

    @abstractmethod
    def _init_sum_logprobs(self):
        """Create zero sum_logprobs tensor for beam search."""
        ...

    @abstractmethod
    def _get_logits_and_cross_attn(self, tokens, encoder_feature):
        """Get logits and cross-attention from decoder. Returns (logits, cross_attns)."""
        ...

    @abstractmethod
    def _check_no_speech(self, logits):
        """Check no_speech probability at start of segment. Returns True to break."""
        ...

    @abstractmethod
    def _suppress_blank_tokens(self, logits):
        """Suppress blank/EOT tokens at segment start. Returns modified logits."""
        ...

    @abstractmethod
    def _apply_token_suppression(self, logits):
        """Apply general token suppression. Returns modified logits."""
        ...

    @abstractmethod
    def _update_tokens(self, current_tokens, logits, sum_logprobs):
        """Update tokens via decoder. Returns (current_tokens, completed)."""
        ...

    @abstractmethod
    def _process_cross_attention(self, accumulated_cross_attns, content_mel_len):
        """Process cross-attention for alignment. Returns attention tensor."""
        ...

    @abstractmethod
    def _get_attended_frames(self, attn):
        """Get most attended frames. Returns (frames_as_python_list, first_frame_int)."""
        ...

    @abstractmethod
    def _is_special_token(self, current_tokens):
        """Check if second-to-last token is a special token (>= DEC_PAD)."""
        ...

    @abstractmethod
    def _rewind_tokens(self):
        """Concatenate state tokens for rewind. Returns token tensor."""
        ...

    @abstractmethod
    def _tokens_to_list(self, current_tokens, start_col):
        """Extract tokens as Python list from start_col onwards."""
        ...

    @abstractmethod
    def _make_new_tokens_tensor(self, hypothesis):
        """Create tensor from hypothesis token list, repeated for beam search."""
        ...

    @abstractmethod
    def _evaluate(self, tensor):
        """Evaluate lazy tensor (mx.eval for MLX, no-op for PyTorch)."""
        ...
