import gc
import logging
import os
import platform
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from whisperlivekit.backend_support import (faster_backend_available,
                                            mlx_backend_available)
from whisperlivekit.model_paths import detect_model_format, resolve_model_path
from whisperlivekit.simul_whisper.config import AlignAttConfig
from whisperlivekit.simul_whisper.simul_whisper import AlignAtt
from whisperlivekit.timed_objects import ASRToken, ChangeSpeaker, Transcript
from whisperlivekit.warmup import load_file
from whisperlivekit.whisper import load_model, tokenizer
from whisperlivekit.whisper.audio import TOKENS_PER_SECOND

logger = logging.getLogger(__name__)


HAS_MLX_WHISPER = mlx_backend_available(warn_on_missing=True)
if HAS_MLX_WHISPER:
    from .mlx_encoder import load_mlx_encoder, load_mlx_model, mlx_model_mapping
    from .mlx import MLXAlignAtt
else:
    mlx_model_mapping = {}
    MLXAlignAtt = None
HAS_FASTER_WHISPER = faster_backend_available(warn_on_missing=not HAS_MLX_WHISPER)
if HAS_FASTER_WHISPER:
    from faster_whisper import WhisperModel
else:
    WhisperModel = None

MIN_DURATION_REAL_SILENCE = 5

class SimulStreamingOnlineProcessor:
    """Online processor for SimulStreaming ASR."""
    SAMPLING_RATE = 16000

    def __init__(self, asr, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.end = 0.0
        self.buffer = []
        self.committed: List[ASRToken] = []
        self.last_result_tokens: List[ASRToken] = []        
        self.model = self._create_alignatt()
        
        if asr.tokenizer:
            self.model.tokenizer = asr.tokenizer
            self.model.state.tokenizer = asr.tokenizer

    def _create_alignatt(self):
        """Create the AlignAtt decoder instance based on ASR mode."""
        if self.asr.use_full_mlx and HAS_MLX_WHISPER:
            return MLXAlignAtt(cfg=self.asr.cfg, mlx_model=self.asr.mlx_model)
        else:
            return AlignAtt(
                cfg=self.asr.cfg,
                loaded_model=self.asr.shared_model,
                mlx_encoder=self.asr.mlx_encoder,
                fw_encoder=self.asr.fw_encoder,
            )

    def start_silence(self):
        tokens, processed_upto = self.process_iter(is_last=True)
        return tokens, processed_upto

    def end_silence(self, silence_duration, offset):
        """Handle silence period."""
        self.end += silence_duration
        long_silence = silence_duration >= MIN_DURATION_REAL_SILENCE
        if not long_silence:
            gap_len = int(16000 * silence_duration)
            if gap_len > 0:
                if self.asr.use_full_mlx:
                    gap_silence = np.zeros(gap_len, dtype=np.float32)
                else:
                    gap_silence = torch.zeros(gap_len)
                self.model.insert_audio(gap_silence)
        if long_silence:
            self.model.refresh_segment(complete=True)
            self.model.global_time_offset = silence_duration + offset

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time):
        """Append an audio chunk to be processed by SimulStreaming."""
        self.end = audio_stream_end_time
        if self.asr.use_full_mlx:
            self.model.insert_audio(audio)
        else:
            audio_tensor = torch.from_numpy(audio).float()
            self.model.insert_audio(audio_tensor)

    def new_speaker(self, change_speaker: ChangeSpeaker):
        """Handle speaker change event."""
        self.process_iter(is_last=True)
        self.model.refresh_segment(complete=True)
        self.model.speaker = change_speaker.speaker
        self.model.global_time_offset = change_speaker.start
            
    def get_buffer(self):
        concat_buffer = Transcript.from_tokens(tokens= self.buffer, sep='')
        return concat_buffer

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        """
        Process accumulated audio chunks using SimulStreaming.
        
        Returns a tuple: (list of committed ASRToken objects, float representing the audio processed up to time).
        """
        try:
            timestamped_words = self.model.infer(is_last=is_last)
            
            if not timestamped_words:
                return [], self.end
            
            if self.model.cfg.language == "auto" and timestamped_words[0].detected_language is None:
                self.buffer.extend(timestamped_words)
                return [], self.end
            
            self.committed.extend(timestamped_words)
            self.buffer = []
            return timestamped_words, self.end
        except Exception as e:
            logger.exception(f"SimulStreaming processing error: {e}")
            return [], self.end

    def warmup(self, audio, init_prompt=""):
        """Warmup the SimulStreaming model."""
        try:
            if self.asr.use_full_mlx:
                # MLX mode: ensure numpy array
                if hasattr(audio, 'numpy'):
                    audio = audio.numpy()
            self.model.insert_audio(audio)
            self.model.infer(True)
            self.model.refresh_segment(complete=True)
            logger.info("SimulStreaming model warmed up successfully")
        except Exception as e:
            logger.exception(f"SimulStreaming warmup failed: {e}")

    def __del__(self):
        gc.collect()
        if not getattr(self.asr, 'use_full_mlx', True) and torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


class SimulStreamingASR:
    """SimulStreaming backend with AlignAtt policy."""
    sep = ""

    def __init__(self, logfile=sys.stderr, **kwargs):
        self.logfile = logfile
        self.transcribe_kargs = {}
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.decoder_type is None:
            self.decoder_type = 'greedy' if self.beams == 1 else 'beam'

        self.fast_encoder = False
        self._resolved_model_path = None
        self.encoder_backend = "whisper"
        self.use_full_mlx = getattr(self, "use_full_mlx", False)
        preferred_backend = getattr(self, "backend", "auto")
        compatible_whisper_mlx, compatible_faster_whisper = True, True
        
        if self.model_path:
            resolved_model_path = resolve_model_path(self.model_path)
            self._resolved_model_path = resolved_model_path
            self.model_path = str(resolved_model_path)
            
            model_info = detect_model_format(resolved_model_path)
            compatible_whisper_mlx = model_info.compatible_whisper_mlx
            compatible_faster_whisper = model_info.compatible_faster_whisper
            
            if not self.use_full_mlx and not model_info.has_pytorch:
                raise FileNotFoundError(
                    f"No PyTorch checkpoint (.pt/.bin/.safetensors) found under {self.model_path}"
                )            
            self.model_name = resolved_model_path.name if resolved_model_path.is_dir() else resolved_model_path.stem
        elif self.model_size is not None:
            self.model_name = self.model_size
        else:
            raise ValueError("Either model_size or model_path must be specified for SimulStreaming.")

        is_multilingual = not self.model_name.endswith(".en")

        self.encoder_backend = self._resolve_encoder_backend(
            preferred_backend,
            compatible_whisper_mlx,
            compatible_faster_whisper,
        )
        self.fast_encoder = self.encoder_backend in ("mlx-whisper", "faster-whisper")
        if self.encoder_backend == "whisper":
            self.disable_fast_encoder = True
        
        if self.encoder_backend == "mlx-whisper" and platform.system() == "Darwin":
            if not hasattr(self, '_full_mlx_disabled'):
                self.use_full_mlx = True
                    
        self.cfg = AlignAttConfig(
                tokenizer_is_multilingual= is_multilingual,
                segment_length=self.min_chunk_size,
                frame_threshold=self.frame_threshold,
                language=self.lan,
                audio_max_len=self.audio_max_len,
                audio_min_len=self.audio_min_len,
                cif_ckpt_path=self.cif_ckpt_path,
                decoder_type="beam",
                beam_size=self.beams,
                task=self.direct_english_translation,
                never_fire=self.never_fire,
                init_prompt=self.init_prompt,
                max_context_tokens=self.max_context_tokens,
                static_init_prompt=self.static_init_prompt,
        )  
        
        # Set up tokenizer for translation if needed
        if self.direct_english_translation:
            self.tokenizer = self.set_translate_task()
        else:
            self.tokenizer = None

        self.mlx_encoder, self.fw_encoder, self.mlx_model = None, None, None
        self.shared_model = None
        
        if self.use_full_mlx and HAS_MLX_WHISPER:
            logger.info('MLX Whisper backend used.')
            if self._resolved_model_path is not None:
                mlx_model_path = str(self._resolved_model_path)
            else:
                mlx_model_path = mlx_model_mapping.get(self.model_name)
            if not mlx_model_path:
                raise FileNotFoundError(
                    f"MLX Whisper backend requested but no compatible weights found for model '{self.model_name}'."
                )
            self.mlx_model = load_mlx_model(path_or_hf_repo=mlx_model_path)
            self._warmup_mlx_model()
        elif self.encoder_backend == "mlx-whisper":
            # hybrid mode: mlx encoder + pytorch decoder
            logger.info('SimulStreaming will use MLX Whisper encoder with PyTorch decoder.')
            if self._resolved_model_path is not None:
                mlx_model_path = str(self._resolved_model_path)
            else:
                mlx_model_path = mlx_model_mapping.get(self.model_name)
            if not mlx_model_path:
                raise FileNotFoundError(
                    f"MLX Whisper backend requested but no compatible weights found for model '{self.model_name}'."
                )
            self.mlx_encoder = load_mlx_encoder(path_or_hf_repo=mlx_model_path)
            self.shared_model = self.load_model()
        elif self.encoder_backend == "faster-whisper":
            print('SimulStreaming will use Faster Whisper for the encoder.')
            if self._resolved_model_path is not None:
                fw_model = str(self._resolved_model_path)
            else:
                fw_model = self.model_name
            self.fw_encoder = WhisperModel(
                fw_model,
                device='auto',
                compute_type='auto',
            )
            self.shared_model = self.load_model()
        else:
            self.shared_model = self.load_model()
    
    def _warmup_mlx_model(self):
        """Warmup the full MLX model."""
        warmup_audio = load_file(self.warmup_file)
        if warmup_audio is not None:
            temp_model = MLXAlignAtt(
                cfg=self.cfg,
                mlx_model=self.mlx_model,
            )
            temp_model.warmup(warmup_audio)
            logger.info("Full MLX model warmed up successfully")


    def _resolve_encoder_backend(self, preferred_backend, compatible_whisper_mlx, compatible_faster_whisper):
        choice = preferred_backend or "auto"
        if self.disable_fast_encoder:
            return "whisper"
        if choice == "whisper":
            return "whisper"
        if choice == "mlx-whisper":
            if not self._can_use_mlx(compatible_whisper_mlx):
                raise RuntimeError("mlx-whisper backend requested but MLX Whisper is unavailable or incompatible with the provided model.")
            return "mlx-whisper"
        if choice == "faster-whisper":
            if not self._can_use_faster(compatible_faster_whisper):
                raise RuntimeError("faster-whisper backend requested but Faster-Whisper is unavailable or incompatible with the provided model.")
            return "faster-whisper"
        if choice == "openai-api":
            raise ValueError("openai-api backend is only supported with the LocalAgreement policy.")
        # auto mode
        if platform.system() == "Darwin" and self._can_use_mlx(compatible_whisper_mlx):
            return "mlx-whisper"
        if self._can_use_faster(compatible_faster_whisper):
            return "faster-whisper"
        return "whisper"

    def _has_custom_model_path(self):
        return self._resolved_model_path is not None

    def _can_use_mlx(self, compatible_whisper_mlx):
        if not HAS_MLX_WHISPER:
            return False
        if self._has_custom_model_path():
            return compatible_whisper_mlx
        return self.model_name in mlx_model_mapping

    def _can_use_faster(self, compatible_faster_whisper):
        if not HAS_FASTER_WHISPER:
            return False
        if self._has_custom_model_path():
            return compatible_faster_whisper
        return True

    def load_model(self):
        model_ref = str(self._resolved_model_path) if self._resolved_model_path else self.model_name
        lora_path = getattr(self, 'lora_path', None)
        whisper_model = load_model(
            name=model_ref,
            download_root=None,
            decoder_only=self.fast_encoder,
            custom_alignment_heads=self.custom_alignment_heads,
            lora_path=lora_path,
        )
        warmup_audio = load_file(self.warmup_file)
        if warmup_audio is not None:
            warmup_audio = torch.from_numpy(warmup_audio).float()
            if self.fast_encoder:
                temp_model = AlignAtt(
                    cfg=self.cfg,
                    loaded_model=whisper_model,
                    mlx_encoder=self.mlx_encoder,
                    fw_encoder=self.fw_encoder,
                )
                temp_model.warmup(warmup_audio)
            else:
                whisper_model.transcribe(warmup_audio, language=self.lan if self.lan != 'auto' else None)
        return whisper_model

    def set_translate_task(self):
        """Set up translation task."""
        if self.cfg.language == 'auto':
            raise Exception('Translation cannot be done with language = auto')
        return tokenizer.get_tokenizer(
            multilingual=True,
            language=self.cfg.language,
            num_languages=99,
            task="translate"
        )

    def transcribe(self, audio):
        """
        Warmup is done directly in load_model
        """
        pass
