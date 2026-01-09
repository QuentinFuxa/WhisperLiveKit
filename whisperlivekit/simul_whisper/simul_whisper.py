import logging
import os
from time import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from whisperlivekit.backend_support import (faster_backend_available,
                                            mlx_backend_available)
from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.whisper import DecodingOptions, tokenizer
from whisperlivekit.whisper.audio import (N_FRAMES, N_SAMPLES,
                                          TOKENS_PER_SECOND,
                                          log_mel_spectrogram, pad_or_trim)
from whisperlivekit.whisper.decoding import (BeamSearchDecoder, GreedyDecoder,
                                             SuppressTokens)
from whisperlivekit.whisper.timing import median_filter

from ..timed_objects import PUNCTUATION_MARKS
from .beam import BeamPyTorchInference
from .config import AlignAttConfig
from .decoder_state import DecoderState
from .eow_detection import fire_at_boundary, load_cif
from .token_buffer import TokenBuffer

DEC_PAD = 50257
logger = logging.getLogger(__name__)

if mlx_backend_available():
    from mlx_whisper.audio import \
        log_mel_spectrogram as mlx_log_mel_spectrogram
    from mlx_whisper.transcribe import pad_or_trim as mlx_pad_or_trim

if faster_backend_available():
    from faster_whisper.audio import pad_or_trim as fw_pad_or_trim
    from faster_whisper.feature_extractor import FeatureExtractor

USE_MLCORE = False


def load_coreml_encoder():
    try:
        from coremltools.models import MLModel
    except ImportError:
        logger.warning("coremltools is not installed")
        return None
    COREML_ENCODER_PATH = os.environ.get("MLCORE_ENCODER_PATH", "whisperlivekit/whisper/whisper_encoder.mlpackage")
    _coreml_encoder = MLModel(COREML_ENCODER_PATH)
    spec = _coreml_encoder.get_spec()
    _coreml_input_name = spec.description.input[0].name if spec.description.input else "mel"
    _coreml_output_name = spec.description.output[0].name if spec.description.output else None
    return _coreml_encoder, _coreml_input_name, _coreml_output_name


class AlignAtt:
    """
    Alignment-based Attention decoder for SimulStreaming.
    
    This class is now hookless - the model can be shared across multiple
    sessions, with each session maintaining its own DecoderState.
    """
    
    # Property accessors for backward compatibility
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
            loaded_model=None,
            mlx_encoder=None,
            fw_encoder=None,
        ) -> None:
        # Shared model reference (can be shared across sessions)
        self.model = loaded_model
        self.mlx_encoder = mlx_encoder
        self.fw_encoder = fw_encoder            
        if fw_encoder:
            self.fw_feature_extractor = FeatureExtractor(feature_size=self.model.dims.n_mels)
        self.coreml_encoder_tuple = None
        if USE_MLCORE:
            self.coreml_encoder_tuple = load_coreml_encoder()
        self.use_mlcore = self.coreml_encoder_tuple is not None
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Model dimensions: {self.model.dims}")
        self.decode_options = DecodingOptions(
            language=cfg.language, 
            without_timestamps=True,
            task=cfg.task
        )
        self.tokenizer_is_multilingual = cfg.tokenizer_is_multilingual
        
        self.max_text_len = self.model.dims.n_text_ctx
        self.num_decoder_layers = len(self.model.decoder.blocks)
        self.cfg = cfg

        if self.cfg.max_context_tokens is None:
            self.max_context_tokens = self.max_text_len
        else:
            self.max_context_tokens = self.cfg.max_context_tokens

        # Initialize per-session state
        self.state = DecoderState()
        self._init_state(cfg)
        
    def _init_state(self, cfg: AlignAttConfig):
        """Initialize the per-session decoder state."""
        # Create tokenizer
        self.create_tokenizer(cfg.language if cfg.language != "auto" else None)
        self.state.tokenizer = self.tokenizer
        self.state.detected_language = cfg.language if cfg.language != "auto" else None
        
        # Timing state
        self.state.global_time_offset = 0.0
        self.state.last_attend_frame = -cfg.rewind_threshold
        self.state.speaker = -1
        
        # CIF helpers for end-of-word boundary detection
        self.state.CIFLinear, self.state.always_fire, self.state.never_fire = load_cif(
            cfg,
            n_audio_state=self.model.dims.n_audio_state,
            device=self.model.device
        )

        # Build alignment source mapping from model's alignment_heads
        self.state.align_source = {}
        self.state.num_align_heads = 0
        for layer_rank, head_id in self.model.alignment_heads.indices().T:
            layer_rank = layer_rank.item()
            heads = self.state.align_source.get(layer_rank, [])
            heads.append((self.state.num_align_heads, head_id.item()))
            self.state.align_source[layer_rank] = heads
            self.state.num_align_heads += 1

        # Build suppress tokens function
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
        suppress_tokens = tuple(sorted(set(suppress_tokens)))
        logger.debug(f"Suppress tokens: {suppress_tokens}")
        sup_tokens = SuppressTokens(suppress_tokens)
        self.state.suppress_tokens_fn = lambda logits: sup_tokens.apply(logits, None)

        # Initialize tokens
        self.init_tokens()
        self.init_context()

        # Set up decoder type
        self.state.decoder_type = cfg.decoder_type
        if cfg.decoder_type == "greedy":
            logger.info("Using greedy decoder")
            self.state.token_decoder = GreedyDecoder(0.0, self.tokenizer.eot)
        elif cfg.decoder_type == "beam":
            logger.info("Using beam decoder")
            self.state.inference = BeamPyTorchInference(self.model, self.state.initial_token_length)
            self.state.inference.kv_cache = self.state.kv_cache
            self.state.token_decoder = BeamSearchDecoder(
                inference=self.state.inference, 
                eot=self.tokenizer.eot, 
                beam_size=cfg.beam_size
            )

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
            task=self.decode_options.task
        )
        self.state.tokenizer = self.tokenizer

    def init_context(self):
        kw = {'tokenizer': self.tokenizer, 
              'device': self.model.device, 
              'prefix_token_ids': [self.tokenizer.sot_prev]}
        self.state.context = TokenBuffer.empty(**kw)
        if self.cfg.static_init_prompt is not None:
            self.state.context = TokenBuffer.from_text(self.cfg.static_init_prompt, **kw)
        if self.cfg.init_prompt is not None:
            self.state.context.text += self.cfg.init_prompt

    def init_tokens(self):
        logger.debug(f"init tokens, {len(self.state.segments)}")
        # init tokens (mandatory prompt)
        self.state.initial_tokens = torch.tensor(
            self.tokenizer.sot_sequence_including_notimestamps, 
            dtype=torch.long, 
            device=self.model.device).unsqueeze(0)
        self.state.initial_token_length = self.state.initial_tokens.shape[1]
        self.state.sot_index = self.tokenizer.sot_sequence.index(self.tokenizer.sot)
        logger.debug(f"init tokens after, {len(self.state.segments)}")
        self.state.tokens = [self.state.initial_tokens]

    def trim_context(self):
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


    def logits(
        self, 
        tokens: torch.Tensor, 
        audio_features: torch.Tensor,
        return_cross_attn: bool = False
    ):
        """Get logits from decoder, optionally returning cross-attention weights."""
        if self.state.decoder_type == "greedy":
            return self.model.decoder(
                tokens, audio_features, 
                kv_cache=self.state.kv_cache,
                return_cross_attn=return_cross_attn
            )
        else:
            logger.debug(f"Logits shape: {tokens.shape}")
            return self.state.inference.logits(
                tokens, audio_features,
                return_cross_attn=return_cross_attn
            )
    

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

    def fire_at_boundary(self, chunked_encoder_feature: torch.Tensor):
        if self.state.always_fire: 
            return True
        if self.state.never_fire: 
            return False
        return fire_at_boundary(chunked_encoder_feature, self.state.CIFLinear)

    def _current_tokens(self):
        toks = self.state.tokens
        # very first infer: duplicate start of seq to beam_size
        if toks[0].shape[0] == 1:
            toks[0] = toks[0].repeat_interleave(self.cfg.beam_size, dim=0)

        if not self.state.context.is_empty():
            context_toks = self.state.context.as_tensor_beam(self.cfg.beam_size, device=self.model.device)
            toks = [context_toks] + toks

        # make it one tensor
        if len(toks) > 1:
            current_tokens = torch.cat(toks, dim=1)
        else:
            current_tokens = toks[0]
        logger.debug("debug print current_tokens:")
        self.debug_print_tokens(current_tokens)
        return current_tokens


    def debug_print_tokens(self, tokens):
        for i in range(self.cfg.beam_size):
            logger.debug(self.tokenizer.decode_with_timestamps(tokens[i].tolist()))

    ### audio buffer 

    def segments_len(self):
        segments_len = sum(s.shape[0] for s in self.state.segments) / 16000
        return segments_len

    def _apply_minseglen(self):
        segments_len = self.segments_len()
        # wait for long enough audio to start
        if segments_len < self.cfg.audio_min_len: 
            logger.debug("waiting for next segment")
            return False
        return True

    def insert_audio(self, segment=None):
        if segment is not None:
            self.state.segments.append(segment)

        removed_len = 0
        # len of audio is bigger than buffer_len. Going to remove the first segment
        segments_len = self.segments_len()
        while len(self.state.segments) > 1 and segments_len > self.cfg.audio_max_len:
            removed_len = self.state.segments[0].shape[0] / 16000
            segments_len -= removed_len
            self.state.last_attend_frame -= int(TOKENS_PER_SECOND * removed_len)
            self.state.cumulative_time_offset += removed_len  # Track cumulative time removed
            self.state.segments = self.state.segments[1:]
            logger.debug(f"remove segments: {len(self.state.segments)} {len(self.state.tokens)}, cumulative offset: {self.state.cumulative_time_offset:.2f}s")
            if len(self.state.tokens) > 1:
                self.state.context.append_token_ids(self.state.tokens[1][0, :].tolist())
                self.state.tokens = [self.state.initial_tokens] + self.state.tokens[2:]
        return removed_len

    def _clean_cache(self):
        """Clean the kv_cache after each inference step."""
        self.state.clean_cache()

    @torch.no_grad()
    def lang_id(self, encoder_features):
        """Language detection from encoder features.
        This code is trimmed and copy-pasted from whisper.decoding.detect_language.
        """
        # forward pass using a single token, startoftranscript
        n_audio = encoder_features.shape[0]
        x = torch.tensor([[self.tokenizer.sot]] * n_audio).to(self.model.device)  # [n_audio, 1]
        # Note: don't use kv_cache for language detection
        logits = self.model.logits(x, encoder_features)[:, 0]

        # collect detected languages; suppress all non-language tokens
        mask = torch.ones(logits.shape[-1], dtype=torch.bool)
        mask[list(self.tokenizer.all_language_tokens)] = False
        logits[:, mask] = -np.inf
        language_tokens = logits.argmax(dim=-1)
        language_token_probs = logits.softmax(dim=-1).cpu()
        language_probs = [
            {
                c: language_token_probs[i, j].item()
                for j, c in zip(self.tokenizer.all_language_tokens, self.tokenizer.all_language_codes)
            }
            for i in range(n_audio)
        ]

        single = encoder_features.ndim == 2
        if single:
            language_tokens = language_tokens[0]
            language_probs = language_probs[0]

        self._clean_cache()
        return language_tokens, language_probs

    ### transcription / translation

    @torch.no_grad()
    def infer(self, is_last=False):
        new_segment = True
        if len(self.state.segments) == 0:
            logger.debug("No segments, nothing to do")
            return []
        if not self._apply_minseglen():
            logger.debug(f"applied minseglen {self.cfg.audio_min_len} > {self.segments_len()}.")
            input_segments = torch.cat(self.state.segments, dim=0)
            return []

        # input_segments is concatenation of audio, it's one array
        if len(self.state.segments) > 1:
            input_segments = torch.cat(self.state.segments, dim=0)
        else:
            input_segments = self.state.segments[0]

        beg_encode = time()
        if self.use_mlcore:
            coreml_encoder, coreml_input_name, coreml_output_name = self.coreml_encoder_tuple
            mel_padded = log_mel_spectrogram(
                input_segments,
                n_mels=self.model.dims.n_mels,
                padding=N_SAMPLES,
                device="cpu",
            ).unsqueeze(0)
            mel = pad_or_trim(mel_padded, N_FRAMES)
            content_mel_len = int((mel_padded.shape[2] - mel.shape[2]) / 2)
            mel_np = np.ascontiguousarray(mel.numpy())
            ml_inputs = {coreml_input_name or "mel": mel_np}
            coreml_outputs = coreml_encoder.predict(ml_inputs)
            if coreml_output_name and coreml_output_name in coreml_outputs:
                encoder_feature_np = coreml_outputs[coreml_output_name]
            else:
                encoder_feature_np = next(iter(coreml_outputs.values()))
            encoder_feature = torch.as_tensor(
                np.array(encoder_feature_np),
                device=self.device,
            )
        if self.mlx_encoder:
            mlx_mel_padded = mlx_log_mel_spectrogram(audio=input_segments.detach(), n_mels=self.model.dims.n_mels, padding=N_SAMPLES)
            mlx_mel = mlx_pad_or_trim(mlx_mel_padded, N_FRAMES, axis=-2)
            mlx_encoder_feature = self.mlx_encoder.encoder(mlx_mel[None])
            encoder_feature = torch.as_tensor(mlx_encoder_feature)
            content_mel_len = int((mlx_mel_padded.shape[0] - mlx_mel.shape[0])/2)
        elif self.fw_encoder:
            audio_length_seconds = len(input_segments) / 16000   
            content_mel_len = int(audio_length_seconds * 100)//2      
            mel_padded_2 = self.fw_feature_extractor(waveform=input_segments.numpy(), padding=N_SAMPLES)[None, :]
            mel = fw_pad_or_trim(mel_padded_2, N_FRAMES, axis=-1)
            encoder_feature_ctranslate = self.fw_encoder.encode(mel)
            if self.device == 'cpu': #it seems that on gpu, passing StorageView to torch.as_tensor fails and wrapping in the array works
                encoder_feature_ctranslate = np.array(encoder_feature_ctranslate)
            try:
                encoder_feature = torch.as_tensor(encoder_feature_ctranslate, device=self.device)
            except TypeError: # Normally the cpu condition should prevent having exceptions, but just in case:
                encoder_feature = torch.as_tensor(np.array(encoder_feature_ctranslate), device=self.device)
        else:
            # mel + padding to 30s
            mel_padded = log_mel_spectrogram(input_segments, n_mels=self.model.dims.n_mels, padding=N_SAMPLES, 
                                                device=self.device).unsqueeze(0)
            # trim to 3000
            mel = pad_or_trim(mel_padded, N_FRAMES)
            # the len of actual audio
            content_mel_len = int((mel_padded.shape[2] - mel.shape[2])/2)
            encoder_feature = self.model.encoder(mel)
        end_encode = time()
        # print('Encoder duration:', end_encode-beg_encode)
                
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
                logger.info(f"Tokenizer language: {self.tokenizer.language}, {self.tokenizer.sot_sequence_including_notimestamps}")

        self.trim_context()
        current_tokens = self._current_tokens()
   
        fire_detected = self.fire_at_boundary(encoder_feature[:, :content_mel_len, :])


        sum_logprobs = torch.zeros(self.cfg.beam_size, device=self.device)
        completed = False
        # punctuation_stop = False

        attn_of_alignment_heads = None
        most_attended_frame = None

        token_len_before_decoding = current_tokens.shape[1]
        
        l_absolute_timestamps = []
        
        accumulated_cross_attns = []
        
        audio_duration_s = self.segments_len()
        max_tokens_per_chunk = max(50, int(audio_duration_s * TOKENS_PER_SECOND * 2.0))  # 2x margin, min 50
        tokens_produced_this_chunk = 0
        
        while not completed and current_tokens.shape[1] < self.max_text_len:  # bos is 3 tokens
            tokens_produced_this_chunk += 1
            
            if tokens_produced_this_chunk > max_tokens_per_chunk:
                logger.warning(f"[Loop Detection] Too many tokens ({tokens_produced_this_chunk}) for {audio_duration_s:.2f}s audio. Breaking.")
                current_tokens = current_tokens[:, :token_len_before_decoding]  # Discard all new tokens
                break

            if new_segment:
                tokens_for_logits = current_tokens
            else:
                # only need to use the last token except in the first forward pass
                tokens_for_logits = current_tokens[:, -1:]

            # Get logits and cross-attention weights from decoder
            result = self.logits(tokens_for_logits, encoder_feature, return_cross_attn=True)
            logits, cross_attns = result
            
            # Accumulate cross-attention from this forward pass
            accumulated_cross_attns.append(cross_attns)

            if new_segment and self.tokenizer.no_speech is not None:
                probs_at_sot = logits[:, self.state.sot_index, :].float().softmax(dim=-1)
                no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()
                if no_speech_probs[0] > self.cfg.nonspeech_prob:
                    logger.info("no speech, stop")
                    break

            logits = logits[:, -1, :]  # logits for the last token

            # suppress blank tokens only at the beginning of the segment
            if new_segment:
                logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf
            new_segment = False
            self.state.suppress_tokens_fn(logits)
            current_tokens, completed = self.state.token_decoder.update(current_tokens, logits, sum_logprobs)

            logger.debug(f"Decoding completed: {completed}, sum_logprobs: {sum_logprobs.tolist()}, tokens: ")
            self.debug_print_tokens(current_tokens)

            # Process accumulated cross-attention weights for alignment
            attn_of_alignment_heads = self._process_cross_attention(accumulated_cross_attns, content_mel_len)

            # for each beam, the most attended frame is:
            most_attended_frames = torch.argmax(attn_of_alignment_heads[:, -1, :], dim=-1)
            
            # Calculate absolute timestamps accounting for cumulative offset
            absolute_timestamps = [
                (frame * 0.02 + self.state.cumulative_time_offset) 
                for frame in most_attended_frames.tolist()
            ]
            
            logger.debug(str(most_attended_frames.tolist()) + " most att frames")
            logger.debug(f"Absolute timestamps: {absolute_timestamps} (offset: {self.state.cumulative_time_offset:.2f}s)")

            most_attended_frame = most_attended_frames[0].item()
            l_absolute_timestamps.append(absolute_timestamps[0])

            logger.debug("current tokens" + str(current_tokens.shape))
            if completed:
                # stripping the last token, the eot
                current_tokens = current_tokens[:, :-1]
                break
            
            # for some rare cases where the attention fails
            if not is_last and self.state.last_attend_frame - most_attended_frame > self.cfg.rewind_threshold:
                if current_tokens.shape[1] > 1 and current_tokens[0, -2] >= DEC_PAD:
                    logger.debug("omit rewinding from special tokens")
                    self.state.last_attend_frame = most_attended_frame
                else:
                    logger.debug(
                        f"[rewind detected] current attention pos: {most_attended_frame}, "
                        f"last attention pos: {self.state.last_attend_frame}; omit this segment")
                    self.state.last_attend_frame = -self.cfg.rewind_threshold
                    current_tokens = torch.cat(self.state.tokens, dim=1) if len(self.state.tokens) > 0 else self.state.tokens[0]
                    break
            else:
                self.state.last_attend_frame = most_attended_frame

            if content_mel_len - most_attended_frame <= (4 if is_last else self.cfg.frame_threshold):
                logger.debug(f"attention reaches the end: {most_attended_frame}/{content_mel_len}")
                # stripping the last token, the one that is attended too close to the end
                current_tokens = current_tokens[:, :-1]
                break
        
            # debug print
            for i in range(self.cfg.beam_size):
                logger.debug("attn: {}, current pos: {}, current token: {}({})".format(
                    attn_of_alignment_heads.shape if attn_of_alignment_heads is not None else None,
                    most_attended_frames[i], 
                    current_tokens[i, -1].item(),
                    self.tokenizer.decode([current_tokens[i, -1].item()])
                ))

        tokens_to_split = current_tokens[0, token_len_before_decoding:]

        # Prepend pending tokens from previous chunk if any
        if self.state.pending_incomplete_tokens:
            logger.debug(f"[UTF-8 Fix] Prepending {len(self.state.pending_incomplete_tokens)} pending tokens: {self.state.pending_incomplete_tokens}")
            pending_tensor = torch.tensor(self.state.pending_incomplete_tokens, dtype=torch.long, device=self.device)
            tokens_to_split = torch.cat([pending_tensor, tokens_to_split])

        if fire_detected or is_last:
            new_hypothesis = tokens_to_split.flatten().tolist()
            split_words, split_tokens = self.tokenizer.split_to_word_tokens(new_hypothesis)
        else:
            # going to truncate the tokens after the last space
            split_words, split_tokens = self.tokenizer.split_to_word_tokens(tokens_to_split.tolist())
            if len(split_words) > 1:
                new_hypothesis = [i for sublist in split_tokens[:-1] for i in sublist]  
            else:
                new_hypothesis = []

        logger.debug(f"new_hypothesis: {new_hypothesis}")
        new_tokens = torch.tensor([new_hypothesis], dtype=torch.long).repeat_interleave(self.cfg.beam_size, dim=0).to(
            device=self.device,
        )
        self.state.tokens.append(new_tokens)

        logger.info(f"Output: {self.tokenizer.decode(new_hypothesis)}")
        
        self._clean_cache()

        if len(l_absolute_timestamps) >= 2 and self.state.first_timestamp is None:
            self.state.first_timestamp = l_absolute_timestamps[0]

        timestamped_words = []
        timestamp_idx = 0
        replacement_char = "\ufffd"
        for word, word_tokens in zip(split_words, split_tokens):
            # Skip words containing incomplete UTF-8 from client output
            if replacement_char in word:
                logger.warning(f"[UTF-8 Filter] Skipping incomplete word from client output: {repr(word)}")
                timestamp_idx += len(word_tokens)
                continue

            try:
                current_timestamp = l_absolute_timestamps[timestamp_idx]
            except IndexError:
                # Use last timestamp if index out of range
                logger.warning(f"Timestamp index {timestamp_idx} out of range, using last timestamp")
                current_timestamp = l_absolute_timestamps[-1] if l_absolute_timestamps else 0.0
            timestamp_idx += len(word_tokens)

            timestamp_entry = ASRToken(
                start=round(current_timestamp, 2),
                end=round(current_timestamp + 0.1, 2),
                text=word,
                speaker=self.state.speaker,
                detected_language=self.state.detected_language
            ).with_offset(
                self.state.global_time_offset
            )
            timestamped_words.append(timestamp_entry)

        # Hold incomplete tokens for next chunk (with limit to prevent hallucination accumulation)
        self.state.pending_incomplete_tokens = []
        MAX_PENDING_TOKENS = 10  # Real incomplete UTF-8 chars are at most a few tokens
        if split_words and replacement_char in split_words[-1]:
            if len(split_tokens[-1]) <= MAX_PENDING_TOKENS:
                self.state.pending_incomplete_tokens = split_tokens[-1]
                logger.debug(f"[UTF-8 Fix] Holding {len(self.state.pending_incomplete_tokens)} incomplete tokens for next chunk")
            else:
                logger.warning(f"[UTF-8 Fix] Skipping {len(split_tokens[-1])} tokens (exceeds limit of {MAX_PENDING_TOKENS}, likely hallucination)")

        return timestamped_words

    def _process_cross_attention(
        self, 
        cross_attns: List[torch.Tensor], 
        content_mel_len: int
    ) -> torch.Tensor:
        """
        Process cross-attention weights from decoder layers for alignment.
        
        Args:
            cross_attns: List of cross-attention tensors from each decoder layer.
                         Each tensor has shape (batch, n_head, seq_len, audio_len)
            content_mel_len: Length of actual audio content in mel frames
            
        Returns processed attention tensor for alignment, shape (batch, seq_len, content_mel_len)
        """
        attn_of_alignment_heads = [[] for _ in range(self.state.num_align_heads)]
        num_decoder_layers = len(self.model.decoder.blocks)

        if cross_attns and isinstance(cross_attns[0], list):
            flattened_attns: List[torch.Tensor] = [attn for layer_list in cross_attns for attn in layer_list]
        else:
            flattened_attns = cross_attns
        
        for idx, attn_mat in enumerate(flattened_attns):
            layer_rank = idx % num_decoder_layers
            # attn_mat shape: (batch, n_head, seq_len, audio_len) or (n_head, seq_len, audio_len) for batch=1
            align_heads_in_layer = self.state.align_source.get(layer_rank, [])
            if len(align_heads_in_layer) == 0:
                continue
            
            attn_mat = F.softmax(attn_mat, dim=-1)
            
            for align_head_rank, head_id in align_heads_in_layer:
                if self.cfg.beam_size == 1:
                    # (n_head, seq_len, audio_len) when squeezed
                    if attn_mat.dim() == 4:
                        a = attn_mat[0, head_id, :, :]  # (seq_len, audio_len)
                    else:
                        a = attn_mat[head_id, :, :]
                    a = a.unsqueeze(0)  # (1, seq_len, audio_len)
                else:
                    # attn_mat: (batch, n_head, seq_len, audio_len)
                    a = attn_mat[:, head_id, :, :]  # (batch, seq_len, audio_len)
                attn_of_alignment_heads[align_head_rank].append(a)
        
        tmp = []
        for mat in attn_of_alignment_heads:
            if mat:
                t = torch.cat(mat, dim=1)  # (batch, total_seq_len, audio_len)
                tmp.append(t)
        
        if not tmp:
            return torch.zeros(self.cfg.beam_size, 1, content_mel_len, device=self.device)
        
        # stck al heads: (batch, num_align_heads, seq_len, audio_len)
        attn_of_alignment_heads = torch.stack(tmp, dim=1)
        
        std, mean = torch.std_mean(attn_of_alignment_heads, dim=-2, keepdim=True, unbiased=False)
        attn_of_alignment_heads = (attn_of_alignment_heads - mean) / (std + 1e-8)
        
        attn_of_alignment_heads = median_filter(attn_of_alignment_heads, 7)
        attn_of_alignment_heads = attn_of_alignment_heads.mean(dim=1)
        attn_of_alignment_heads = attn_of_alignment_heads[:, :, :content_mel_len]
        return attn_of_alignment_heads