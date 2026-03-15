import logging
import threading
from argparse import Namespace
from dataclasses import asdict

from whisperlivekit.config import WhisperLiveKitConfig
from whisperlivekit.local_agreement.online_asr import OnlineASRProcessor
from whisperlivekit.local_agreement.whisper_online import backend_factory
from whisperlivekit.simul_whisper import SimulStreamingASR

logger = logging.getLogger(__name__)

class TranscriptionEngine:
    _instance = None
    _initialized = False
    _lock = threading.Lock()  # Thread-safe singleton lock

    def __new__(cls, *args, **kwargs):
        # Double-checked locking pattern for thread-safe singleton
        if cls._instance is None:
            with cls._lock:
                # Check again inside lock to prevent race condition
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton so a new instance can be created.

        For testing only — allows switching backends between test runs.
        In production, the singleton should never be reset.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False

    def __init__(self, config=None, **kwargs):
        # Thread-safe initialization check
        with TranscriptionEngine._lock:
            if TranscriptionEngine._initialized:
                return

        try:
            self._do_init(config, **kwargs)
        except Exception:
            # Reset singleton so a retry is possible
            with TranscriptionEngine._lock:
                TranscriptionEngine._instance = None
                TranscriptionEngine._initialized = False
            raise

        with TranscriptionEngine._lock:
            TranscriptionEngine._initialized = True

    def _do_init(self, config=None, **kwargs):
        # Handle negated kwargs from programmatic API
        if 'no_transcription' in kwargs:
            kwargs['transcription'] = not kwargs.pop('no_transcription')
        if 'no_vad' in kwargs:
            kwargs['vad'] = not kwargs.pop('no_vad')
        if 'no_vac' in kwargs:
            kwargs['vac'] = not kwargs.pop('no_vac')

        if config is None:
            if isinstance(kwargs.get('config'), WhisperLiveKitConfig):
                config = kwargs.pop('config')
            else:
                config = WhisperLiveKitConfig.from_kwargs(**kwargs)
        self.config = config

        # Backward compat: expose as self.args (Namespace-like) for AudioProcessor etc.
        self.args = Namespace(**asdict(config))

        self.asr = None
        self.tokenizer = None
        self.diarization = None
        self.vac_session = None

        if config.vac:
            from whisperlivekit.silero_vad_iterator import is_onnx_available

            if is_onnx_available():
                from whisperlivekit.silero_vad_iterator import load_onnx_session
                self.vac_session = load_onnx_session()
            else:
                logger.warning(
                    "onnxruntime not installed. VAC will use JIT model which is loaded per-session. "
                    "For multi-user scenarios, install onnxruntime: pip install onnxruntime"
                )

        transcription_common_params = {
            "warmup_file": config.warmup_file,
            "min_chunk_size": config.min_chunk_size,
            "model_size": config.model_size,
            "model_cache_dir": config.model_cache_dir,
            "model_dir": config.model_dir,
            "model_path": config.model_path,
            "lora_path": config.lora_path,
            "lan": config.lan,
            "direct_english_translation": config.direct_english_translation,
        }

        if config.transcription:
            if config.backend == "vllm-realtime":
                from whisperlivekit.vllm_realtime import VLLMRealtimeASR
                self.tokenizer = None
                self.asr = VLLMRealtimeASR(
                    vllm_url=config.vllm_url,
                    model_name=config.vllm_model or "Qwen/Qwen3-ASR-1.7B",
                    lan=config.lan,
                )
                logger.info("Using vLLM Realtime streaming backend at %s", config.vllm_url)
            elif config.backend == "voxtral-mlx":
                from whisperlivekit.voxtral_mlx_asr import VoxtralMLXASR
                self.tokenizer = None
                self.asr = VoxtralMLXASR(**transcription_common_params)
                logger.info("Using Voxtral MLX native backend")
            elif config.backend == "voxtral":
                from whisperlivekit.voxtral_hf_streaming import VoxtralHFStreamingASR
                self.tokenizer = None
                self.asr = VoxtralHFStreamingASR(**transcription_common_params)
                logger.info("Using Voxtral HF Transformers streaming backend")
            elif config.backend == "qwen3-mlx":
                from whisperlivekit.qwen3_mlx_asr import Qwen3MLXASR
                self.tokenizer = None
                self.asr = Qwen3MLXASR(**transcription_common_params)
                logger.info("Using Qwen3 MLX native backend")
            elif config.backend == "qwen3-simul-kv":
                from whisperlivekit.qwen3_simul_kv import Qwen3SimulKVASR
                self.tokenizer = None
                self.asr = Qwen3SimulKVASR(
                    **transcription_common_params,
                    alignment_heads_path=config.custom_alignment_heads,
                    border_fraction=getattr(config, 'border_fraction', 0.25),
                )
                logger.info("Using Qwen3-ASR backend with SimulStreaming+KV policy")
            elif config.backend == "qwen3-simul":
                from whisperlivekit.qwen3_simul import Qwen3SimulStreamingASR
                self.tokenizer = None
                self.asr = Qwen3SimulStreamingASR(
                    **transcription_common_params,
                    alignment_heads_path=config.custom_alignment_heads,
                )
                logger.info("Using Qwen3-ASR backend with SimulStreaming policy")
            elif config.backend == "qwen3":
                from whisperlivekit.qwen3_asr import Qwen3ASR
                self.asr = Qwen3ASR(**transcription_common_params)
                self.asr.confidence_validation = config.confidence_validation
                self.asr.tokenizer = None
                self.asr.buffer_trimming = config.buffer_trimming
                self.asr.buffer_trimming_sec = config.buffer_trimming_sec
                self.asr.backend_choice = "qwen3"
                from whisperlivekit.warmup import warmup_asr
                warmup_asr(self.asr, config.warmup_file)
                logger.info("Using Qwen3-ASR backend with LocalAgreement policy")
            elif config.backend_policy == "simulstreaming":
                simulstreaming_params = {
                    "disable_fast_encoder": config.disable_fast_encoder,
                    "custom_alignment_heads": config.custom_alignment_heads,
                    "frame_threshold": config.frame_threshold,
                    "beams": config.beams,
                    "decoder_type": config.decoder_type,
                    "audio_max_len": config.audio_max_len,
                    "audio_min_len": config.audio_min_len,
                    "cif_ckpt_path": config.cif_ckpt_path,
                    "never_fire": config.never_fire,
                    "init_prompt": config.init_prompt,
                    "static_init_prompt": config.static_init_prompt,
                    "max_context_tokens": config.max_context_tokens,
                }

                self.tokenizer = None
                self.asr = SimulStreamingASR(
                    **transcription_common_params,
                    **simulstreaming_params,
                    backend=config.backend,
                )
                logger.info(
                    "Using SimulStreaming policy with %s backend",
                    getattr(self.asr, "encoder_backend", "whisper"),
                )
            else:
                whisperstreaming_params = {
                    "buffer_trimming": config.buffer_trimming,
                    "confidence_validation": config.confidence_validation,
                    "buffer_trimming_sec": config.buffer_trimming_sec,
                }

                self.asr = backend_factory(
                    backend=config.backend,
                    **transcription_common_params,
                    **whisperstreaming_params,
                )
                logger.info(
                    "Using LocalAgreement policy with %s backend",
                    getattr(self.asr, "backend_choice", self.asr.__class__.__name__),
                )

        if config.diarization:
            if config.diarization_backend == "diart":
                from whisperlivekit.diarization.diart_backend import DiartDiarization
                self.diarization_model = DiartDiarization(
                    block_duration=config.min_chunk_size,
                    segmentation_model=config.segmentation_model,
                    embedding_model=config.embedding_model,
                )
            elif config.diarization_backend == "sortformer":
                from whisperlivekit.diarization.sortformer_backend import SortformerDiarization
                self.diarization_model = SortformerDiarization()

        self.translation_model = None
        if config.target_language:
            if config.lan == 'auto' and config.backend_policy != "simulstreaming":
                raise ValueError('Translation cannot be set with language auto when transcription backend is not simulstreaming')
            else:
                try:
                    from nllw import load_model
                except ImportError:
                    raise ImportError('To use translation, you must install nllw: `pip install nllw`')
                self.translation_model = load_model(
                    [config.lan],
                    nllb_backend=config.nllb_backend,
                    nllb_size=config.nllb_size,
                )


def online_factory(args, asr, language=None):
    """Create an online ASR processor for a session.

    Args:
        args: Configuration namespace.
        asr: Shared ASR backend instance.
        language: Optional per-session language override (e.g. "en", "fr", "auto").
            If provided and the backend supports it, transcription will use
            this language instead of the server-wide default.
    """
    # Wrap the shared ASR with a per-session language if requested
    if language is not None:
        from whisperlivekit.session_asr_proxy import SessionASRProxy
        asr = SessionASRProxy(asr, language)

    backend = getattr(args, 'backend', None)
    if backend == "vllm-realtime":
        from whisperlivekit.vllm_realtime import VLLMRealtimeOnlineProcessor
        return VLLMRealtimeOnlineProcessor(asr)
    if backend == "qwen3-simul-kv":
        from whisperlivekit.qwen3_simul_kv import Qwen3SimulKVOnlineProcessor
        return Qwen3SimulKVOnlineProcessor(asr)
    if backend == "qwen3-mlx":
        from whisperlivekit.qwen3_mlx_asr import Qwen3MLXOnlineProcessor
        return Qwen3MLXOnlineProcessor(asr)
    if backend == "qwen3-simul":
        from whisperlivekit.qwen3_simul import Qwen3SimulStreamingOnlineProcessor
        return Qwen3SimulStreamingOnlineProcessor(asr)
    if backend == "voxtral-mlx":
        from whisperlivekit.voxtral_mlx_asr import VoxtralMLXOnlineProcessor
        return VoxtralMLXOnlineProcessor(asr)
    if backend == "voxtral":
        from whisperlivekit.voxtral_hf_streaming import VoxtralHFStreamingOnlineProcessor
        return VoxtralHFStreamingOnlineProcessor(asr)
    if backend == "qwen3":
        return OnlineASRProcessor(asr)
    if args.backend_policy == "simulstreaming":
        from whisperlivekit.simul_whisper import SimulStreamingOnlineProcessor
        return SimulStreamingOnlineProcessor(asr)
    return OnlineASRProcessor(asr)


def online_diarization_factory(args, diarization_backend):
    if args.diarization_backend == "diart":
        online = diarization_backend
        # Not the best here, since several user/instances will share the same backend, but diart is not SOTA anymore and sortformer is recommended
    elif args.diarization_backend == "sortformer":
        from whisperlivekit.diarization.sortformer_backend import SortformerDiarizationOnline
        online = SortformerDiarizationOnline(shared_model=diarization_backend)
    else:
        raise ValueError(f"Unknown diarization backend: {args.diarization_backend}")
    return online


def online_translation_factory(args, translation_model):
    #should be at speaker level in the future:
    #one shared nllb model for all speaker
    #one tokenizer per speaker/language
    from nllw import OnlineTranslation
    return OnlineTranslation(translation_model, [args.lan], [args.target_language])
