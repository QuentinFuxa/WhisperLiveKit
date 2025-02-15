"""MLX whisper AlignAtt streaming decoder."""
import logging
from typing import Any, List, Tuple

import mlx.core as mx
import numpy as np

from mlx_whisper.audio import log_mel_spectrogram as mlx_log_mel_spectrogram
from mlx_whisper.transcribe import pad_or_trim as mlx_pad_or_trim

from whisperlivekit.whisper.audio import N_FRAMES, N_SAMPLES, TOKENS_PER_SECOND

from ..align_att_base import DEC_PAD, AlignAttBase
from ..config import AlignAttConfig
from .decoder_state import MLXDecoderState
from .decoders import MLXBeamSearchDecoder, MLXGreedyDecoder, MLXInference


logger = logging.getLogger(__name__)


class MLXTokenBuffer:
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
        tok_ids = self.as_token_ids()
        return mx.array([tok_ids], dtype=mx.int32)

    def as_mlx_array_beam(self, beam: int) -> mx.array:
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
        tokenizer = self.tokenizer
        assert tokenizer is not None, "Tokenizer is not set."
        ids = tokenizer.encode(self.text[after:])
        words, wids = self.tokenizer.split_to_word_tokens(ids)
        if not words:
            return 0
        self.text = self.text[:after] + "".join(words[num:])
        return sum(len(wi) for wi in wids[:num])

    def append_token_ids(self, token_ids):
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
    """Apply median filter along the last axis."""
    if filter_width <= 1:
        return x
    pad_width = filter_width // 2
    shape = x.shape
    left_pad = mx.repeat(x[..., :1], pad_width, axis=-1)
    right_pad = mx.repeat(x[..., -1:], pad_width, axis=-1)
    x_padded = mx.concatenate([left_pad, x, right_pad], axis=-1)
    result = []
    for i in range(shape[-1]):
        window = x_padded[..., i:i + filter_width]
        sorted_window = mx.sort(window, axis=-1)
        median_val = sorted_window[..., filter_width // 2:filter_width // 2 + 1]
        result.append(median_val)
    return mx.concatenate(result, axis=-1)


class MLXAlignAtt(AlignAttBase):
    """
    MLX-native Alignment-based Attention decoder for SimulStreaming.

    Runs entirely on MLX, with no PyTorch dependencies for inference.
    """

    def __init__(
        self,
        cfg: AlignAttConfig,
        mlx_model: Any,
    ) -> None:
        # Common init (sets self.model, self.cfg, decode_options, etc.)
        self._base_init(cfg, mlx_model)
        logger.info(f"MLX Model dimensions: {self.model.dims}")

        # Per-session state
        self.state = MLXDecoderState()
        self._init_state(cfg)

    def _init_state(self, cfg: AlignAttConfig):
        self._init_state_common(cfg)

        # CIF: MLX doesn't support CIF checkpoint loading
        if cfg.cif_ckpt_path is None or not cfg.cif_ckpt_path:
            if cfg.never_fire:
                self.state.never_fire = True
                self.state.always_fire = False
            else:
                self.state.always_fire = True
                self.state.never_fire = False
        else:
            logger.warning(
                "CIF checkpoint provided but MLX CIF not implemented. "
                "Using always_fire=True"
            )
            self.state.always_fire = True
            self.state.never_fire = cfg.never_fire

        self._build_alignment_source()

        # Suppress tokens
        suppress_tokens = [
            self.tokenizer.transcribe, self.tokenizer.translate,
            self.tokenizer.sot, self.tokenizer.sot_prev,
            self.tokenizer.sot_lm, self.tokenizer.no_timestamps,
        ] + list(self.tokenizer.all_language_tokens)
        if self.tokenizer.no_speech is not None:
            suppress_tokens.append(self.tokenizer.no_speech)
        self.state.suppress_tokens = tuple(sorted(set(suppress_tokens)))
        logger.debug(f"Suppress tokens: {self.state.suppress_tokens}")

        self.init_tokens()
        self.init_context()

        # Decoder type
        self.state.decoder_type = cfg.decoder_type
        if cfg.decoder_type == "greedy":
            logger.info("Using MLX greedy decoder")
            self.state.token_decoder = MLXGreedyDecoder(0.0, self.tokenizer.eot)
        elif cfg.decoder_type == "beam":
            logger.info("Using MLX beam decoder")
            self.state.inference = MLXInference(
                self.model, self.state.initial_token_length,
            )
            self.state.token_decoder = MLXBeamSearchDecoder(
                inference=self.state.inference,
                eot=self.tokenizer.eot,
                beam_size=cfg.beam_size,
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

    # === Abstract method implementations ===

    def init_tokens(self):
        logger.debug(f"init tokens, {len(self.state.segments)}")
        self.state.initial_tokens = mx.array(
            [self.tokenizer.sot_sequence_including_notimestamps],
            dtype=mx.int32,
        )
        self.state.initial_token_length = self.state.initial_tokens.shape[1]
        self.state.sot_index = self.tokenizer.sot_sequence.index(self.tokenizer.sot)
        logger.debug(f"init tokens after, {len(self.state.segments)}")
        self.state.tokens = [self.state.initial_tokens]

    def init_context(self):
        kw = {
            'tokenizer': self.tokenizer,
            'prefix_token_ids': [self.tokenizer.sot_prev],
        }
        self.state.context = MLXTokenBuffer.empty(**kw)
        if self.cfg.static_init_prompt is not None:
            self.state.context = MLXTokenBuffer.from_text(self.cfg.static_init_prompt, **kw)
        if self.cfg.init_prompt is not None:
            self.state.context.text += self.cfg.init_prompt

    def insert_audio(self, segment=None):
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
            logger.debug(
                f"remove segments: {len(self.state.segments)} {len(self.state.tokens)}, "
                f"cumulative offset: {self.state.cumulative_time_offset:.2f}s"
            )
            if len(self.state.tokens) > 1:
                token_list = np.array(self.state.tokens[1][0, :]).tolist()
                self.state.context.append_token_ids(token_list)
                self.state.tokens = [self.state.initial_tokens] + self.state.tokens[2:]
        return removed_len

    def _current_tokens(self) -> mx.array:
        toks = self.state.tokens
        if toks[0].shape[0] == 1:
            toks[0] = mx.repeat(toks[0], self.cfg.beam_size, axis=0)
        if not self.state.context.is_empty():
            context_toks = self.state.context.as_mlx_array_beam(self.cfg.beam_size)
            toks = [context_toks] + toks
        if len(toks) > 1:
            current_tokens = mx.concatenate(toks, axis=1)
        else:
            current_tokens = toks[0]
        logger.debug("debug print current_tokens:")
        self.debug_print_tokens(current_tokens)
        return current_tokens

    def fire_at_boundary(self, chunked_encoder_feature: mx.array) -> bool:
        if self.state.always_fire:
            return True
        if self.state.never_fire:
            return False
        return True  # MLX CIF not implemented

    def lang_id(self, encoder_features: mx.array) -> Tuple[mx.array, List[dict]]:
        n_audio = encoder_features.shape[0]
        x = mx.array([[self.tokenizer.sot]] * n_audio, dtype=mx.int32)
        logits, _, _ = self.model.decoder(x, encoder_features, kv_cache=None)
        logits = logits[:, 0]

        mask = mx.ones(logits.shape[-1], dtype=mx.bool_)
        language_token_indices = mx.array(
            list(self.tokenizer.all_language_tokens), dtype=mx.int32,
        )
        mask = mask.at[language_token_indices].add(False)
        logits = mx.where(mask, mx.array(-float('inf')), logits)

        language_tokens = mx.argmax(logits, axis=-1)
        language_token_probs = mx.softmax(logits, axis=-1)
        probs_np = np.array(language_token_probs)
        language_probs = [
            {
                c: float(probs_np[i, j])
                for j, c in zip(
                    self.tokenizer.all_language_tokens,
                    self.tokenizer.all_language_codes,
                )
            }
            for i in range(n_audio)
        ]
        self._clean_cache()
        return language_tokens, language_probs

    def _concat_segments(self):
        if len(self.state.segments) > 1:
            return np.concatenate(self.state.segments, axis=0)
        return self.state.segments[0]

    def _encode(self, input_segments):
        mlx_mel_padded = mlx_log_mel_spectrogram(
            audio=input_segments,
            n_mels=self.model.dims.n_mels,
            padding=N_SAMPLES,
        )
        mlx_mel = mlx_pad_or_trim(mlx_mel_padded, N_FRAMES, axis=-2)
        encoder_feature = self.model.encoder(mlx_mel[None])
        content_mel_len = int((mlx_mel_padded.shape[0] - mlx_mel.shape[0]) / 2)
        return encoder_feature, content_mel_len

    def _init_sum_logprobs(self):
        return mx.zeros((self.cfg.beam_size,), dtype=mx.float32)

    def _get_logits_and_cross_attn(self, tokens, encoder_feature):
        if self.state.decoder_type == "greedy":
            logits, self.state.kv_cache, cross_qk = self.model.decoder(
                tokens, encoder_feature, kv_cache=self.state.kv_cache,
            )
            return logits, cross_qk
        else:
            return self.state.inference.logits(tokens, encoder_feature)

    def _check_no_speech(self, logits):
        if self.tokenizer.no_speech is not None:
            probs_at_sot = mx.softmax(logits[:, self.state.sot_index, :], axis=-1)
            no_speech_probs = np.array(
                probs_at_sot[:, self.tokenizer.no_speech],
            ).tolist()
            if no_speech_probs[0] > self.cfg.nonspeech_prob:
                logger.info("no speech, stop")
                return True
        return False

    def _suppress_blank_tokens(self, logits):
        blank_tokens = self.tokenizer.encode(" ") + [self.tokenizer.eot]
        logits = logits.at[:, blank_tokens].add(-float('inf'))
        return logits

    def _apply_token_suppression(self, logits):
        if self.state.suppress_tokens:
            suppress_indices = mx.array(
                list(self.state.suppress_tokens), dtype=mx.int32,
            )
            logits = logits.at[:, suppress_indices].add(-float('inf'))
        return logits

    def _update_tokens(self, current_tokens, logits, sum_logprobs):
        return self.state.token_decoder.update(current_tokens, logits, sum_logprobs)

    def _process_cross_attention(
        self, cross_attns: List, content_mel_len: int,
    ) -> mx.array:
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
            if not align_heads_in_layer:
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
                tmp.append(mx.concatenate(mat, axis=1))
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

    def _get_attended_frames(self, attn):
        most_attended_frames = mx.argmax(attn[:, -1, :], axis=-1)
        frames_np = np.array(most_attended_frames)
        return frames_np.tolist(), int(frames_np[0])

    def _is_special_token(self, current_tokens):
        return int(np.array(current_tokens[0, -2])) >= DEC_PAD

    def _rewind_tokens(self):
        if len(self.state.tokens) > 0:
            return mx.concatenate(self.state.tokens, axis=1)
        return self.state.tokens[0]

    def _tokens_to_list(self, current_tokens, start_col):
        return np.array(current_tokens[0, start_col:]).tolist()

    def _make_new_tokens_tensor(self, hypothesis):
        new_tokens = mx.array([hypothesis], dtype=mx.int32)
        return mx.repeat(new_tokens, self.cfg.beam_size, axis=0)

    def _evaluate(self, tensor):
        mx.eval(tensor)
