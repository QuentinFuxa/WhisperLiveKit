import logging
import sys
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
            if config.backend_policy == "simulstreaming":
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


def online_factory(args, asr):
    if args.backend_policy == "simulstreaming":
        from whisperlivekit.simul_whisper import SimulStreamingOnlineProcessor
        return SimulStreamingOnlineProcessor(asr)
    return OnlineASRProcessor(asr)
  
  
def online_diarization_factory(args, diarization_backend):
    if args.diarization_backend == "diart":
        online = diarization_backend
        # Not the best here, since several user/instances will share the same backend, but diart is not SOTA anymore and sortformer is recommended
    elif args.diarization_backend == "sortformer":
        from whisperlivekit.diarization.sortformer_backend import \
            SortformerDiarizationOnline
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
