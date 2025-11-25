"""
MLX whisper AlignAtt streaming decoder
"""
import logging
from time import time
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from mlx_whisper.audio import log_mel_spectrogram as mlx_log_mel_spectrogram
from mlx_whisper.transcribe import pad_or_trim as mlx_pad_or_trim

from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.whisper import DecodingOptions, tokenizer
from whisperlivekit.whisper.audio import N_FRAMES, N_SAMPLES, TOKENS_PER_SECOND

from ..config import AlignAttConfig
from .decoder_state import MLXDecoderState
from .decoders import MLXBeamSearchDecoder, MLXGreedyDecoder, MLXInference

DEC_PAD = 50257
logger = logging.getLogger(__name__)


class MLXTokenBuffer: #should try to make it heritate from classic simul whisper class
    """Token buffer for MLX-based decoding."""

    def __init__(self, text="", tokenizer=None, prefix_token_ids=None):
        self.text = text
        self.prefix_token_ids = prefix_token_ids or []
        self.tokenizer = tokenizer
        self.pending_token_ids = []

    def as_token_ids(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer is not set.")
        return self.prefix_token_ids + tokenizer.encode(self.text)

    def as_mlx_array(self) -> mx.array:
        """Return tokens as MLX array."""
        tok_ids = self.as_token_ids()
        return mx.array([tok_ids], dtype=mx.int32)

    def as_mlx_array_beam(self, beam: int) -> mx.array:
        """Return tokens as MLX array repeated for beam search."""
        t = self.as_mlx_array()
        return mx.repeat(t, beam, axis=0)

    def as_text(self):
        return self.text

    @staticmethod
    def empty(*a, **kw):
        return MLXTokenBuffer(*a, **kw)

    @staticmethod
    def from_text(text, *a, **kw):
        return MLXTokenBuffer(*a, text=text, **kw)

    def is_empty(self):
        return self.text is None or self.text == ""

    def trim_words(self, num=1, after=0):
        """Trim words from the beginning of the context."""
        tokenizer = self.tokenizer
        assert tokenizer is not None, "Tokenizer is not set."

        ids = tokenizer.encode(self.text[after:])
        words, wids = self.tokenizer.split_to_word_tokens(ids)
        if not words:
            return 0
        self.text = self.text[:after] + "".join(words[num:])
        return sum(len(wi) for wi in wids[:num])

    def append_token_ids(self, token_ids):
        """Append token IDs to the buffer, handling incomplete UTF-8."""
        tokenizer = self.tokenizer
        assert tokenizer is not None, "Tokenizer is not set."

        all_tokens = self.pending_token_ids + token_ids
        decoded = tokenizer.decode(all_tokens)
        replacement_char = "\ufffd"

        if replacement_char in decoded:
            if len(all_tokens) > 1:
                decoded_partial = tokenizer.decode(all_tokens[:-1])
                if replacement_char not in decoded_partial:
                    self.text += decoded_partial
                    self.pending_token_ids = [all_tokens[-1]]
                else:
                    self.pending_token_ids = all_tokens
            else:
                self.pending_token_ids = all_tokens
        else:
            self.text += decoded
            self.pending_token_ids = []


def mlx_median_filter(x: mx.array, filter_width: int) -> mx.array:
    """
    Apply median filter along the last axis.
    
    Args:
        x: Input array of shape (..., T)
        filter_width: Width of the median filter (should be odd)
        
    Returns:
        Filtered array of same shape
    """
    if filter_width <= 1:
        return x
    
    pad_width = filter_width // 2
    shape = x.shape
    
    left_pad = mx.repeat(x[..., :1], pad_width, axis=-1)
    right_pad = mx.repeat(x[..., -1:], pad_width, axis=-1)
    x_padded = mx.concatenate([left_pad, x, right_pad], axis=-1)
    
    result_shape = list(shape)
    result = []
    
    for i in range(shape[-1]):
        window = x_padded[..., i:i + filter_width]
        sorted_window = mx.sort(window, axis=-1)
        median_val = sorted_window[..., filter_width // 2:filter_width // 2 + 1]
        result.append(median_val)
    
    return mx.concatenate(result, axis=-1)


class MLXAlignAtt:
    """
    MLX-native Alignment-based Attention decoder for SimulStreaming.
    
    This class runs entirely on MLX, with no PyTorch dependencies for inference.
    """

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

    def __init__(
        self,
        cfg: AlignAttConfig,
        mlx_model: Any,
    ) -> None:
        """
        Initialize MLX AlignAtt decoder.
        
        Args:
            cfg: AlignAtt configuration
            mlx_model: MLX Whisper model (full model, not just encoder)
        """
        self.model = mlx_model
        self.cfg = cfg
        
        logger.info(f"MLX Model dimensions: {self.model.dims}")
        
        self.decode_options = DecodingOptions(
            language=cfg.language,
            without_timestamps=True,
            task=cfg.task
        )
        self.tokenizer_is_multilingual = cfg.tokenizer_is_multilingual
        
        self.max_text_len = self.model.dims.n_text_ctx
        self.num_decoder_layers = len(self.model.decoder.blocks)

        if self.cfg.max_context_tokens is None:
            self.max_context_tokens = self.max_text_len
        else:
            self.max_context_tokens = self.cfg.max_context_tokens

        # Initialize per-session state
        self.state = MLXDecoderState()
        self._init_state(cfg)

    def _init_state(self, cfg: AlignAttConfig):
        """Initialize the per-session decoder state."""
        self.create_tokenizer(cfg.language if cfg.language != "auto" else None)
        self.state.tokenizer = self.tokenizer
        self.state.detected_language = cfg.language if cfg.language != "auto" else None
        self.state.global_time_offset = 0.0
        self.state.last_attend_frame = -cfg.rewind_threshold
        self.state.speaker = -1

        if cfg.cif_ckpt_path is None or not cfg.cif_ckpt_path:
            if cfg.never_fire:
                self.state.never_fire = True
                self.state.always_fire = False
            else:
                self.state.always_fire = True
                self.state.never_fire = False
        else:
            logger.warning("CIF checkpoint provided but MLX CIF not implemented. Using always_fire=True")
            self.state.always_fire = True
            self.state.never_fire = cfg.never_fire

        self._build_alignment_source()

        suppress_tokens = [
            self.tokenizer.transcribe,
            self.tokenizer.translate,
            self.tokenizer.sot,
            self.tokenizer.sot_prev,
            self.tokenizer.sot_lm,
            self.tokenizer.no_timestamps,
        ] + list(self.tokenizer.all_language_tokens)
        if self.tokenizer.no_speech is not None:
            suppress_tokens.append(self.tokenizer.no_speech)
        self.state.suppress_tokens = tuple(sorted(set(suppress_tokens)))
        logger.debug(f"Suppress tokens: {self.state.suppress_tokens}")

        self.init_tokens()
        self.init_context()

        self.state.decoder_type = cfg.decoder_type
        if cfg.decoder_type == "greedy":
            logger.info("Using MLX greedy decoder")
            self.state.token_decoder = MLXGreedyDecoder(0.0, self.tokenizer.eot)
        elif cfg.decoder_type == "beam":
            logger.info("Using MLX beam decoder")
            self.state.inference = MLXInference(self.model, self.state.initial_token_length)
            self.state.token_decoder = MLXBeamSearchDecoder(
                inference=self.state.inference,
                eot=self.tokenizer.eot,
                beam_size=cfg.beam_size
            )

    def _build_alignment_source(self):
        """Build alignment source mapping from model's alignment_heads."""
        self.state.align_source = {}
        self.state.num_align_heads = 0
        
        alignment_heads = self.model.alignment_heads
        
        if alignment_heads is None:
            logger.warning("No alignment heads found in model")
            return
            
        if hasattr(alignment_heads, 'tolist'):
            heads_list = alignment_heads.tolist()
        else:
            heads_list = np.array(alignment_heads).tolist()
            
        for layer_rank, head_id in heads_list:
            layer_rank = int(layer_rank)
            head_id = int(head_id)
            heads = self.state.align_source.get(layer_rank, [])
            heads.append((self.state.num_align_heads, head_id))
            self.state.align_source[layer_rank] = heads
            self.state.num_align_heads += 1

    def warmup(self, audio: np.ndarray):
        """Warmup the model with sample audio."""
        try:
            self.insert_audio(audio)
            self.infer(is_last=True)
            self.refresh_segment(complete=True)
            logger.info("MLX model warmed up successfully")
        except Exception as e:
            logger.exception(f"MLX model warmup failed: {e}")

    def create_tokenizer(self, language=None):
        """Create tokenizer for the given language."""
        self.tokenizer = tokenizer.get_tokenizer(
            multilingual=self.tokenizer_is_multilingual,
            language=language,
            num_languages=self.model.num_languages,
            task=self.decode_options.task
        )
        self.state.tokenizer = self.tokenizer

    def init_context(self):
        """Initialize context buffer."""
        kw = {
            'tokenizer': self.tokenizer,
            'prefix_token_ids': [self.tokenizer.sot_prev]
        }
        self.state.context = MLXTokenBuffer.empty(**kw)
        if self.cfg.static_init_prompt is not None:
            self.state.context = MLXTokenBuffer.from_text(self.cfg.static_init_prompt, **kw)
        if self.cfg.init_prompt is not None:
            self.state.context.text += self.cfg.init_prompt

    def init_tokens(self):
        """Initialize token sequence."""
        logger.debug(f"init tokens, {len(self.state.segments)}")
        self.state.initial_tokens = mx.array(
            [self.tokenizer.sot_sequence_including_notimestamps],
            dtype=mx.int32
        )
        self.state.initial_token_length = self.state.initial_tokens.shape[1]
        self.state.sot_index = self.tokenizer.sot_sequence.index(self.tokenizer.sot)
        logger.debug(f"init tokens after, {len(self.state.segments)}")
        self.state.tokens = [self.state.initial_tokens]

    def trim_context(self):
        """Trim context if too long."""
        logger.info("Trimming context")
        c = len(self.state.context.as_token_ids()) - len(self.state.context.prefix_token_ids)
        logger.info(f"Context text: {self.state.context.as_text()}")
        l = sum(t.shape[1] for t in self.state.tokens) + c
        if self.cfg.static_init_prompt is None:
            after = 0
        else:
            after = len(self.cfg.static_init_prompt)
        while c > self.max_context_tokens or l > self.max_text_len - 20:
            t = self.state.context.trim_words(after=after)
            l -= t
            c -= t
            logger.debug(f"len {l}, c {c}, max_context_tokens {self.max_context_tokens}")
            if t == 0:
                break
        logger.info(f"Context after trim: {self.state.context.text} (len: {l})")

    def refresh_segment(self, complete=False):
        """Refresh segment state."""
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

    def fire_at_boundary(self, chunked_encoder_feature: mx.array) -> bool:
        """Check if we should fire at word boundary (CIF-based)."""
        if self.state.always_fire:
            return True
        if self.state.never_fire:
            return False
        return True

    def _current_tokens(self) -> mx.array:
        """Get current token sequence for decoding."""
        toks = self.state.tokens
        
        if toks[0].shape[0] == 1:
            toks[0] = mx.repeat(toks[0], self.cfg.beam_size, axis=0)

        if not self.state.context.is_empty():
            context_toks = self.state.context.as_mlx_array_beam(self.cfg.beam_size)
            toks = [context_toks] + toks

        # Concatenate all tokens
        if len(toks) > 1:
            current_tokens = mx.concatenate(toks, axis=1)
        else:
            current_tokens = toks[0]
            
        logger.debug("debug print current_tokens:")
        self.debug_print_tokens(current_tokens)
        return current_tokens

    def debug_print_tokens(self, tokens: mx.array):
        """Debug print token sequences."""
        tokens_np = np.array(tokens)
        for i in range(min(self.cfg.beam_size, tokens_np.shape[0])):
            logger.debug(self.tokenizer.decode_with_timestamps(tokens_np[i].tolist()))

    def segments_len(self) -> float:
        """Get total length of audio segments in seconds."""
        return sum(s.shape[0] for s in self.state.segments) / 16000

    def _apply_minseglen(self) -> bool:
        """Check if we have enough audio to process."""
        segments_len = self.segments_len()
        if segments_len < self.cfg.audio_min_len:
            logger.debug("waiting for next segment")
            return False
        return True

    def insert_audio(self, segment: np.ndarray = None):
        """Insert audio segment into buffer."""
        if segment is not None:
            if hasattr(segment, 'numpy'):
                segment = segment.numpy()
            self.state.segments.append(segment)

        removed_len = 0
        segments_len = self.segments_len()
        
        while len(self.state.segments) > 1 and segments_len > self.cfg.audio_max_len:
            removed_len = self.state.segments[0].shape[0] / 16000
            segments_len -= removed_len
            self.state.last_attend_frame -= int(TOKENS_PER_SECOND * removed_len)
            self.state.cumulative_time_offset += removed_len
            self.state.segments = self.state.segments[1:]
            logger.debug(f"remove segments: {len(self.state.segments)} {len(self.state.tokens)}, cumulative offset: {self.state.cumulative_time_offset:.2f}s")
            
            if len(self.state.tokens) > 1:
                # Convert MLX array to list for context
                token_list = np.array(self.state.tokens[1][0, :]).tolist()
                self.state.context.append_token_ids(token_list)
                self.state.tokens = [self.state.initial_tokens] + self.state.tokens[2:]
                
        return removed_len

    def _clean_cache(self):
        """Clean the kv_cache after each inference step."""
        self.state.clean_cache()

    def _suppress_tokens(self, logits: mx.array) -> mx.array:
        """Apply token suppression to logits."""
        if self.state.suppress_tokens:
            suppress_indices = mx.array(list(self.state.suppress_tokens), dtype=mx.int32)
            logits = logits.at[:, suppress_indices].add(-float('inf'))
        return logits

    def lang_id(self, encoder_features: mx.array) -> Tuple[mx.array, List[dict]]:
        """Language detection from encoder features."""
        n_audio = encoder_features.shape[0]
        x = mx.array([[self.tokenizer.sot]] * n_audio, dtype=mx.int32)
        
        logits, _, _ = self.model.decoder(x, encoder_features, kv_cache=None)
        logits = logits[:, 0]
        
        mask = mx.ones(logits.shape[-1], dtype=mx.bool_)
        language_token_indices = mx.array(list(self.tokenizer.all_language_tokens), dtype=mx.int32)
        mask = mask.at[language_token_indices].add(False)
        
        logits = mx.where(mask, mx.array(-float('inf')), logits)
        
        language_tokens = mx.argmax(logits, axis=-1)
        language_token_probs = mx.softmax(logits, axis=-1)
        
        probs_np = np.array(language_token_probs)
        
        language_probs = [
            {
                c: float(probs_np[i, j])
                for j, c in zip(self.tokenizer.all_language_tokens, self.tokenizer.all_language_codes)
            }
            for i in range(n_audio)
        ]

        self._clean_cache()
        return language_tokens, language_probs

    def infer(self, is_last: bool = False) -> List[ASRToken]:
        """
        Main inference method.
        
        Args:
            is_last: Whether this is the final chunk
            
        Returns:
            List of timestamped ASR tokens
        """
        new_segment = True
        
        if len(self.state.segments) == 0:
            logger.debug("No segments, nothing to do")
            return []
            
        if not self._apply_minseglen():
            logger.debug(f"applied minseglen {self.cfg.audio_min_len} > {self.segments_len()}.")
            return []

        if len(self.state.segments) > 1:
            input_segments = np.concatenate(self.state.segments, axis=0)
        else:
            input_segments = self.state.segments[0]

        beg_encode = time()
        
        mlx_mel_padded = mlx_log_mel_spectrogram(
            audio=input_segments,
            n_mels=self.model.dims.n_mels,
            padding=N_SAMPLES
        )
        mlx_mel = mlx_pad_or_trim(mlx_mel_padded, N_FRAMES, axis=-2)
        encoder_feature = self.model.encoder(mlx_mel[None])
        content_mel_len = int((mlx_mel_padded.shape[0] - mlx_mel.shape[0]) / 2)
        
        mx.eval(encoder_feature)
        
        end_encode = time()
        logger.debug(f'MLX Encoder duration: {end_encode - beg_encode:.3f}s')

        if self.cfg.language == "auto" and self.state.detected_language is None and self.state.first_timestamp:
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

        self.trim_context()
        current_tokens = self._current_tokens()

        fire_detected = self.fire_at_boundary(encoder_feature[:, :content_mel_len, :])

        sum_logprobs = mx.zeros((self.cfg.beam_size,), dtype=mx.float32)
        completed = False

        attn_of_alignment_heads = None
        most_attended_frame = None

        token_len_before_decoding = current_tokens.shape[1]

        l_absolute_timestamps = []
        accumulated_cross_attns = []

        audio_duration_s = self.segments_len()
        max_tokens_per_chunk = max(50, int(audio_duration_s * TOKENS_PER_SECOND * 2.0))
        tokens_produced_this_chunk = 0

        while not completed and current_tokens.shape[1] < self.max_text_len:
            tokens_produced_this_chunk += 1

            if tokens_produced_this_chunk > max_tokens_per_chunk:
                logger.warning(f"[Loop Detection] Too many tokens ({tokens_produced_this_chunk}) for {audio_duration_s:.2f}s audio. Breaking.")
                current_tokens = current_tokens[:, :token_len_before_decoding]
                break

            if new_segment:
                tokens_for_logits = current_tokens
            else:
                tokens_for_logits = current_tokens[:, -1:]

            if self.state.decoder_type == "greedy":
                logits, self.state.kv_cache, cross_qk = self.model.decoder(
                    tokens_for_logits, encoder_feature, kv_cache=self.state.kv_cache
                )
            else:
                logits, cross_qk = self.state.inference.logits(tokens_for_logits, encoder_feature)

            mx.eval(logits)
            
            accumulated_cross_attns.append(cross_qk)

            if new_segment and self.tokenizer.no_speech is not None:
                probs_at_sot = mx.softmax(logits[:, self.state.sot_index, :], axis=-1)
                no_speech_probs = np.array(probs_at_sot[:, self.tokenizer.no_speech]).tolist()
                if no_speech_probs[0] > self.cfg.nonspeech_prob:
                    logger.info("no speech, stop")
                    break

            logits = logits[:, -1, :]  # Last token logits

            # Suppress tokens at segment start
            if new_segment:
                blank_tokens = self.tokenizer.encode(" ") + [self.tokenizer.eot]
                logits = logits.at[:, blank_tokens].add(-float('inf'))
            new_segment = False
            
            logits = self._suppress_tokens(logits)
            
            current_tokens, completed = self.state.token_decoder.update(
                current_tokens, logits, sum_logprobs
            )
            mx.eval(current_tokens)

            logger.debug(f"Decoding completed: {completed}")
            self.debug_print_tokens(current_tokens)

            attn_of_alignment_heads = self._process_cross_attention(
                accumulated_cross_attns, content_mel_len
            )

            most_attended_frames = mx.argmax(attn_of_alignment_heads[:, -1, :], axis=-1)
            most_attended_frames_np = np.array(most_attended_frames)

            absolute_timestamps = [
                (frame * 0.02 + self.state.cumulative_time_offset)
                for frame in most_attended_frames_np.tolist()
            ]

            logger.debug(str(most_attended_frames_np.tolist()) + " most att frames")
            logger.debug(f"Absolute timestamps: {absolute_timestamps}")

            most_attended_frame = int(most_attended_frames_np[0])
            l_absolute_timestamps.append(absolute_timestamps[0])

            if completed:
                current_tokens = current_tokens[:, :-1]
                break
            if not is_last and self.state.last_attend_frame - most_attended_frame > self.cfg.rewind_threshold:
                current_tokens_np = np.array(current_tokens)
                if current_tokens.shape[1] > 1 and current_tokens_np[0, -2] >= DEC_PAD:
                    logger.debug("omit rewinding from special tokens")
                    self.state.last_attend_frame = most_attended_frame
                else:
                    logger.debug(f"[rewind detected] current: {most_attended_frame}, last: {self.state.last_attend_frame}")
                    self.state.last_attend_frame = -self.cfg.rewind_threshold
                    current_tokens = mx.concatenate(self.state.tokens, axis=1) if len(self.state.tokens) > 0 else self.state.tokens[0]
                    break
            else:
                self.state.last_attend_frame = most_attended_frame
            if content_mel_len - most_attended_frame <= (4 if is_last else self.cfg.frame_threshold):
                logger.debug(f"attention reaches the end: {most_attended_frame}/{content_mel_len}")
                current_tokens = current_tokens[:, :-1]
                break
        tokens_to_split = np.array(current_tokens[0, token_len_before_decoding:]).tolist()
        if self.state.pending_incomplete_tokens:
            logger.debug(f"[UTF-8 Fix] Prepending pending tokens: {self.state.pending_incomplete_tokens}")
            tokens_to_split = self.state.pending_incomplete_tokens + tokens_to_split

        if fire_detected or is_last:
            new_hypothesis = tokens_to_split
            split_words, split_tokens = self.tokenizer.split_to_word_tokens(new_hypothesis)
        else:
            split_words, split_tokens = self.tokenizer.split_to_word_tokens(tokens_to_split)
            if len(split_words) > 1:
                new_hypothesis = [i for sublist in split_tokens[:-1] for i in sublist]
            else:
                new_hypothesis = []

        logger.debug(f"new_hypothesis: {new_hypothesis}")
        new_tokens = mx.array([new_hypothesis], dtype=mx.int32)
        new_tokens = mx.repeat(new_tokens, self.cfg.beam_size, axis=0)
        self.state.tokens.append(new_tokens)

        logger.info(f"Output: {self.tokenizer.decode(new_hypothesis)}")

        self._clean_cache()

        if len(l_absolute_timestamps) >= 2 and self.state.first_timestamp is None:
            self.state.first_timestamp = l_absolute_timestamps[0]
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
                pass
            timestamp_idx += len(word_tokens)

            timestamp_entry = ASRToken(
                start=round(current_timestamp, 2),
                end=round(current_timestamp + 0.1, 2),
                text=word,
                speaker=self.state.speaker,
                detected_language=self.state.detected_language
            ).with_offset(self.state.global_time_offset)
            timestamped_words.append(timestamp_entry)
        self.state.pending_incomplete_tokens = []
        MAX_PENDING_TOKENS = 10
        if split_words and replacement_char in split_words[-1]:
            if len(split_tokens[-1]) <= MAX_PENDING_TOKENS:
                self.state.pending_incomplete_tokens = split_tokens[-1]
                logger.debug(f"[UTF-8 Fix] Holding incomplete tokens")
            else:
                logger.warning(f"[UTF-8 Fix] Skipping too many tokens")

        return timestamped_words

    def _process_cross_attention(
        self,
        cross_attns: List[List[mx.array]],
        content_mel_len: int
    ) -> mx.array:
        """
        Process cross-attention weights for alignment.
        
        Args:
            cross_attns: List of cross-attention from each forward pass
                        Each element is a list of mx.arrays per layer
            content_mel_len: Length of actual audio content
            
        Returns:
            Processed attention tensor, shape (batch, seq_len, content_mel_len)
        """
        attn_of_alignment_heads = [[] for _ in range(self.state.num_align_heads)]
        num_decoder_layers = self.num_decoder_layers

        if cross_attns and isinstance(cross_attns[0], list):
            flattened_attns = [attn for layer_list in cross_attns for attn in layer_list]
        else:
            flattened_attns = cross_attns

        for idx, attn_mat in enumerate(flattened_attns):
            if attn_mat is None:
                continue
                
            layer_rank = idx % num_decoder_layers
            align_heads_in_layer = self.state.align_source.get(layer_rank, [])
            
            if len(align_heads_in_layer) == 0:
                continue
            attn_mat = mx.softmax(attn_mat, axis=-1)

            for align_head_rank, head_id in align_heads_in_layer:
                if self.cfg.beam_size == 1:
                    if attn_mat.ndim == 4:
                        a = attn_mat[0, head_id, :, :]
                    else:
                        a = attn_mat[head_id, :, :]
                    a = a[None, :, :]
                else:
                    a = attn_mat[:, head_id, :, :]
                attn_of_alignment_heads[align_head_rank].append(a)
        tmp = []
        for mat in attn_of_alignment_heads:
            if mat:
                t = mx.concatenate(mat, axis=1)
                tmp.append(t)

        if not tmp:
            return mx.zeros((self.cfg.beam_size, 1, content_mel_len))
        attn_of_alignment_heads = mx.stack(tmp, axis=1)

        std = mx.std(attn_of_alignment_heads, axis=-2, keepdims=True)
        mean = mx.mean(attn_of_alignment_heads, axis=-2, keepdims=True)
        attn_of_alignment_heads = (attn_of_alignment_heads - mean) / (std + 1e-8)

        attn_of_alignment_heads = mlx_median_filter(attn_of_alignment_heads, 7)

        attn_of_alignment_heads = mx.mean(attn_of_alignment_heads, axis=1)

        attn_of_alignment_heads = attn_of_alignment_heads[:, :, :content_mel_len]

        mx.eval(attn_of_alignment_heads)
        return attn_of_alignment_heads

