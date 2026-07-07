"""Typed configuration for the WhisperLiveKit pipeline."""
import logging
from dataclasses import dataclass, fields
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WhisperLiveKitConfig:
    """Single source of truth for all WhisperLiveKit configuration.

    Replaces the previous dict-based parameter system in TranscriptionEngine.
    All fields have defaults matching the prior behaviour.
    """

    # Server / global
    host: str = "localhost"
    port: int = 8000
    diarization: bool = False
    punctuation_split: bool = False
    target_language: str = ""
    vac: bool = True
    vac_chunk_size: float = 0.04
    log_level: str = "DEBUG"
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    forwarded_allow_ips: Optional[str] = None
    transcription: bool = True
    vad: bool = True
    pcm_input: bool = False
    disable_punctuation_split: bool = False
    diarization_backend: str = "sortformer"
    backend_policy: str = "simulstreaming"
    backend: str = "auto"

    # Transcription common
    warmup_file: Optional[str] = None
    min_chunk_size: float = 0.1
    model_size: str = "base"
    model_cache_dir: Optional[str] = None
    model_dir: Optional[str] = None
    model_path: Optional[str] = None
    encoder_model_path: Optional[str] = None
    decoder_model_path: Optional[str] = None
    lora_path: Optional[str] = None
    lan: str = "auto"
    direct_english_translation: bool = False

    # LocalAgreement-specific
    buffer_trimming: str = "segment"
    confidence_validation: bool = False
    buffer_trimming_sec: float = 15.0

    # SimulStreaming-specific
    disable_fast_encoder: bool = False
    custom_alignment_heads: Optional[str] = None
    frame_threshold: int = 25
    beams: int = 1
    decoder_type: Optional[str] = None
    audio_max_len: float = 30.0
    audio_min_len: float = 0.0
    cif_ckpt_path: Optional[str] = None
    never_fire: bool = False
    init_prompt: Optional[str] = None
    static_init_prompt: Optional[str] = None
    max_context_tokens: Optional[int] = None

    # Diarization (diart)
    segmentation_model: str = "pyannote/segmentation-3.0"
    embedding_model: str = "pyannote/embedding"

    # Translation
    nllb_backend: str = "transformers"
    nllb_size: str = "600M"

    # vLLM Qwen3 backends
    vllm_model: str = ""
    vllm_aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.45
    vllm_dtype: str = "auto"
    vllm_max_model_len: int = 0
    qwen3_vllm_audio_backend: str = "standard"
    qwen3_vllm_causal_decoder_backend: str = "vllm-text"
    qwen3_vllm_causal_attn_implementation: str = "auto"
    qwen3_vllm_text_decoder_model: str = ""
    qwen3_vllm_live_idle_timeout_ms: float = 50.0
    qwen3_vllm_tower_checkpoint: str = ""
    qwen3_vllm_left_context_sec: float = 15.0
    qwen3_vllm_block_frames: int = 192
    qwen3_vllm_cache_block_size: int = 0
    qwen3_vllm_segment_max_steps: int = 150
    qwen3_vllm_segment_min_sec: float = 0.0
    qwen3_vllm_prompt_context_words: int = 0
    # None keeps the deprecated WLK_QWEN3_VLLM_*_MULTIPROCESSING env fallback.
    qwen3_vllm_live_multiprocessing: Optional[bool] = None
    qwen3_vllm_aligner_multiprocessing: Optional[bool] = None
    holdback_words: Optional[int] = None
    trim_sentence_buffer: bool = True
    qwen3_vllm_metal_audio_backend: str = "standard"
    qwen3_vllm_metal_tower_checkpoint: str = ""
    qwen3_vllm_metal_left_context_sec: float = 15.0
    qwen3_vllm_metal_block_frames: int = 192

    # Qwen3 streaming backend (HF Transformers, bounded-recompute cache)
    qwen3_streaming_chunk_sec: float = 2.0
    qwen3_streaming_left_context_sec: float = 12.0
    qwen3_streaming_right_context_ms: int = 640
    qwen3_streaming_segment_max_steps: int = 200
    qwen3_streaming_segment_keep_tail_steps: int = 0
    qwen3_streaming_hold_back_words: int = 6
    # None = auto (2 windowed, 1 causal — see qwen3_asr_causal asr.py).
    qwen3_streaming_stable_iterations: Optional[int] = None
    qwen3_streaming_max_new_tokens: int = 256
    qwen3_streaming_device: str = "auto"
    qwen3_streaming_dtype: str = "auto"
    qwen3_streaming_attn_implementation: str = "auto"
    qwen3_streaming_context: str = ""
    qwen3_streaming_prompt_context_words: int = 0
    qwen3_streaming_audio_backend: str = "windowed"
    qwen3_streaming_tower_checkpoint: str = ""
    qwen3_streaming_block_frames: int = 192

    def __post_init__(self):
        # .en model suffix forces English
        if self.model_size and self.model_size.endswith(".en"):
            self.lan = "en"
        # Normalize backend_policy aliases
        if self.backend_policy == "1":
            self.backend_policy = "simulstreaming"
        elif self.backend_policy == "2":
            self.backend_policy = "localagreement"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_namespace(cls, ns) -> "WhisperLiveKitConfig":
        """Create config from an argparse Namespace, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in vars(ns).items() if k in known})

    @classmethod
    def from_kwargs(cls, **kwargs) -> "WhisperLiveKitConfig":
        """Create config from keyword arguments; warns on unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = set(kwargs.keys()) - known
        if unknown:
            logger.warning("Unknown config keys ignored: %s", unknown)
        return cls(**{k: v for k, v in kwargs.items() if k in known})
