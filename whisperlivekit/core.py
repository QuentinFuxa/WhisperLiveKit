import logging
import threading
from argparse import Namespace
from dataclasses import asdict

from whisperlivekit.config import WhisperLiveKitConfig
from whisperlivekit.local_agreement.online_asr import OnlineASRProcessor
from whisperlivekit.local_agreement.whisper_online import backend_factory
from whisperlivekit.simul_whisper import SimulStreamingASR
from whisperlivekit.timed_objects import ASRToken, TimedText

logger = logging.getLogger(__name__)


_NLLW_LANGUAGE_ALIASES = {
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh-hans": "zh-CN",
    "zh-sg": "zh-CN",
    "cmn": "zh-CN",
    "cmn-hans": "zh-CN",
    "zh-tw": "zh-TW",
    "zh-hant": "zh-TW",
    "zh-hk": "zh-TW",
    "cmn-hant": "zh-TW",
}


def _nllw_language_code(language):
    """Return a language identifier accepted by NLLW without changing ASR config."""
    if not language:
        return language
    normalized = str(language).strip()
    lookup_key = normalized.replace("_", "-").lower()
    return _NLLW_LANGUAGE_ALIASES.get(lookup_key, normalized)


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
            "encoder_model_path": config.encoder_model_path,
            "decoder_model_path": config.decoder_model_path,
            "lora_path": config.lora_path,
            "lan": config.lan,
            "direct_english_translation": config.direct_english_translation,
            "vllm_model": config.vllm_model,
            "vllm_aligner_model": config.vllm_aligner_model,
            "vllm_tensor_parallel_size": config.vllm_tensor_parallel_size,
            "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
            "vllm_dtype": config.vllm_dtype,
            "vllm_max_model_len": config.vllm_max_model_len,
            "qwen3_vllm_audio_backend": config.qwen3_vllm_audio_backend,
            "qwen3_vllm_causal_decoder_backend": config.qwen3_vllm_causal_decoder_backend,
            "qwen3_vllm_causal_attn_implementation": config.qwen3_vllm_causal_attn_implementation,
            "qwen3_vllm_text_decoder_model": config.qwen3_vllm_text_decoder_model,
            "qwen3_vllm_live_idle_timeout_ms": config.qwen3_vllm_live_idle_timeout_ms,
            "qwen3_vllm_tower_checkpoint": config.qwen3_vllm_tower_checkpoint,
            "qwen3_vllm_left_context_sec": config.qwen3_vllm_left_context_sec,
            "qwen3_vllm_block_frames": config.qwen3_vllm_block_frames,
            "qwen3_vllm_cache_block_size": config.qwen3_vllm_cache_block_size,
            "qwen3_vllm_segment_max_steps": config.qwen3_vllm_segment_max_steps,
            "qwen3_vllm_segment_min_sec": config.qwen3_vllm_segment_min_sec,
            "qwen3_vllm_prompt_context_words": config.qwen3_vllm_prompt_context_words,
            "qwen3_vllm_live_multiprocessing": config.qwen3_vllm_live_multiprocessing,
            "qwen3_vllm_aligner_multiprocessing": config.qwen3_vllm_aligner_multiprocessing,
            "holdback_words": config.holdback_words,
            "trim_sentence_buffer": config.trim_sentence_buffer,
        }

        if config.transcription:
            if config.backend == "qwen3-streaming":
                from whisperlivekit.qwen3_streaming import Qwen3StreamingASR
                qwen3_streaming_params = {
                    "qwen3_streaming_chunk_sec": config.qwen3_streaming_chunk_sec,
                    "qwen3_streaming_left_context_sec": config.qwen3_streaming_left_context_sec,
                    "qwen3_streaming_right_context_ms": config.qwen3_streaming_right_context_ms,
                    "qwen3_streaming_segment_max_steps": config.qwen3_streaming_segment_max_steps,
                    "qwen3_streaming_segment_keep_tail_steps": config.qwen3_streaming_segment_keep_tail_steps,
                    "qwen3_streaming_hold_back_words": config.qwen3_streaming_hold_back_words,
                    "qwen3_streaming_stable_iterations": config.qwen3_streaming_stable_iterations,
                    "qwen3_streaming_max_new_tokens": config.qwen3_streaming_max_new_tokens,
                    "qwen3_streaming_device": config.qwen3_streaming_device,
                    "qwen3_streaming_dtype": config.qwen3_streaming_dtype,
                    "qwen3_streaming_attn_implementation": config.qwen3_streaming_attn_implementation,
                    "qwen3_streaming_context": config.qwen3_streaming_context,
                    "qwen3_streaming_prompt_context_words": config.qwen3_streaming_prompt_context_words,
                    "qwen3_streaming_audio_backend": config.qwen3_streaming_audio_backend,
                    "qwen3_streaming_tower_checkpoint": config.qwen3_streaming_tower_checkpoint,
                    "qwen3_streaming_block_frames": config.qwen3_streaming_block_frames,
                }
                self.tokenizer = None
                self.asr = Qwen3StreamingASR(
                    **transcription_common_params, **qwen3_streaming_params
                )
                logger.info("Using Qwen3-ASR streaming (HF Transformers) backend")
            elif config.backend == "qwen3-vllm":
                from whisperlivekit.qwen3_vllm_asr import Qwen3VLLMASR
                self.tokenizer = None
                self.asr = Qwen3VLLMASR(**transcription_common_params)
                logger.info("Using Qwen3-ASR vLLM in-process backend")
            elif config.backend == "qwen3-vllm-metal":
                from whisperlivekit.qwen3_vllm_metal_asr import Qwen3VLLMMetalASR
                qwen3_vllm_metal_params = {
                    "qwen3_vllm_metal_audio_backend": config.qwen3_vllm_metal_audio_backend,
                    "qwen3_vllm_metal_tower_checkpoint": config.qwen3_vllm_metal_tower_checkpoint,
                    "qwen3_vllm_metal_left_context_sec": config.qwen3_vllm_metal_left_context_sec,
                    "qwen3_vllm_metal_block_frames": config.qwen3_vllm_metal_block_frames,
                }
                self.tokenizer = None
                self.asr = Qwen3VLLMMetalASR(
                    **transcription_common_params, **qwen3_vllm_metal_params
                )
                logger.info("Using Qwen3-ASR vllm-metal in-process backend")
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
            elif config.backend == "canary":
                from whisperlivekit.canary_backend import CANARY_LANGS, CanaryASR, CanaryLID
                # Config-time language validation: reject unsupported codes at
                # startup rather than letting a session silently fall back or die.
                if config.canary_default_lang not in CANARY_LANGS:
                    raise ValueError(
                        f"--canary-default-lang {config.canary_default_lang!r} is not one "
                        f"of Canary's 25 supported codes: {', '.join(sorted(CANARY_LANGS))}."
                    )
                if config.lan not in (None, "auto") and config.lan not in CANARY_LANGS:
                    raise ValueError(
                        f"--language {config.lan!r} is not supported by Canary. Use 'auto' "
                        f"or one of: {', '.join(sorted(CANARY_LANGS))}."
                    )
                self.tokenizer = None
                self.asr = CanaryASR(
                    lan=config.lan,
                    canary_model=config.canary_model,
                    canary_default_lang=config.canary_default_lang,
                    buffer_trimming=config.buffer_trimming,
                    buffer_trimming_sec=config.buffer_trimming_sec,
                    confidence_validation=config.confidence_validation,
                )
                # Load the LID model so any session may request auto-detection.
                # A failure here (e.g. the LID model cannot be downloaded) must
                # not stop the server from serving transcription; degrade to no
                # auto-detect (sessions fall back to --canary-default-lang).
                try:
                    self.asr.lid_model = CanaryLID(lid_model=config.canary_lid_model)
                except Exception as e:
                    logger.warning(
                        "Canary LID model %r failed to load (%s); auto language "
                        "detection disabled, sessions use the default language.",
                        config.canary_lid_model, e,
                    )
                    self.asr.lid_model = None
                from whisperlivekit.warmup import warmup_asr
                warmup_asr(self.asr, config.warmup_file)
                logger.info("Using LocalAgreement policy with Canary backend")
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
                self.diarization_model = SortformerDiarization(model_path=config.sortformer_model_path)

        self.translation_model = None
        if config.target_language:
            if config.lan == 'auto' and config.backend_policy != "simulstreaming":
                raise ValueError('Translation cannot be set with language auto when transcription backend is not simulstreaming')
            if getattr(config, "translation_backend", "nllb") == "alignatt":
                from whisperlivekit.translation_alignatt import AlignAttRemoteEngine
                self.translation_model = AlignAttRemoteEngine(
                    url=config.alignatt_url,
                    source_language=config.lan,
                    preset=config.alignatt_preset,
                    latency=config.alignatt_latency,
                    context_text=config.alignatt_context,
                )
            elif getattr(config, "translation_backend", "nllb") in ("mlx-llm-mt", "hunyuan-mlx"):
                from whisperlivekit.translation_mlx_llm_mt import MlxLlmTranslation
                model_id = getattr(config, "mlx_llm_mt_model", "hy-mt2-1.8b-8bit")
                self.translation_model = MlxLlmTranslation(
                    model_id=model_id,
                    target_language=config.target_language,
                    source_language=config.lan,
                )
            else:
                if config.backend in {"qwen3-vllm", "qwen3-vllm-metal", "qwen3-streaming"}:
                    raise ValueError(
                        f"{config.backend} does not support in-process NLLB translation; "
                        "use --translation-backend alignatt with an alignatt-mt-server sidecar."
                    )
                try:
                    from nllw import load_model
                except ImportError:
                    raise ImportError('To use translation, you must install nllw: `pip install nllw`')
                source_language = _nllw_language_code(config.lan)
                self.translation_model = load_model(
                    [source_language],
                    nllb_backend=config.nllb_backend,
                    nllb_size=config.nllb_size,
                )

def _to_wlk_token(tok):
    """Convert a qwen3_asr_causal token into WhisperLiveKit's ASRToken.

    qwen3's ASRToken is a separate class that doesn't derive from TimedText, so
    it lacks helpers (has_punctuation) the diarization alignment needs.
    """
    if isinstance(tok, TimedText):
        return tok
    is_silence = getattr(tok, "is_silence", None)
    if callable(is_silence) and is_silence():
        return tok
    # start/end/text accessed directly on purpose: a token missing them is a real
    # incompatibility that should raise here, not be masked with defaults.
    return ASRToken(
        start=tok.start,
        end=tok.end,
        text=tok.text or "",
        speaker=getattr(tok, "speaker", -1),
        detected_language=getattr(tok, "detected_language", None),
        probability=getattr(tok, "probability", None),
    )


class _ASRTokenNormalizer:
    """Wraps a qwen3 online processor, converting emitted tokens to WhisperLiveKit
    ASRTokens. finish is wrapped via __getattr__ (not an explicit method) so that
    hasattr(proc, "finish") stays honest for the loop's fallback probe.
    """

    _WRAP = {"finish"}

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    @staticmethod
    def _convert(result):
        tokens, *rest = result # (tokens, end_time)
        converted = [_to_wlk_token(t) for t in (tokens or [])]
        return (converted, *rest)

    def process_iter(self, *args, **kwargs):
        return self._convert(self._inner.process_iter(*args, **kwargs))

    def start_silence(self, *args, **kwargs):
        return self._convert(self._inner.start_silence(*args, **kwargs))

    def new_speaker(self, *args, **kwargs):
        """Preserve Qwen boundary tokens discarded by its compatibility API.

        The current qwen3 processors implement new_speaker() as a bare call to
        start_silence() and drop its return value. Calling start_silence()
        directly keeps the identical reset behavior while exposing the tokens
        and processed position required by AudioProcessor.
        """
        return self.start_silence()

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name in self._WRAP and callable(attr):
            def wrapped(*args, **kwargs):
                return self._convert(attr(*args, **kwargs))
            return wrapped
        return attr

def online_factory(args, asr, language=None, context=None):
    """Create an online ASR processor for a session.

    Args:
        args: Configuration namespace.
        asr: Shared ASR backend instance.
        language: Optional per-session language override (e.g. "en", "fr", "auto").
            If provided and the backend supports it, transcription will use
            this language instead of the server-wide default.
        context: Optional terminology, names, or other text conditioning for
            this session. Unsupported backends reject it explicitly.
    """
    from whisperlivekit.session_asr_proxy import (
        SessionASRProxy,
        validate_session_context,
    )

    backend = getattr(args, 'backend', None)
    context = validate_session_context(args, asr, context)
    # Canary carries its own per-session wrapper (CanarySessionASR with auto-detect),
    # so it returns here before the generic SessionASRProxy wrap to avoid double-wrapping.
    if backend == "canary":
        from whisperlivekit.canary_backend import CanarySessionASR
        effective = language if language is not None else getattr(args, 'lan', 'auto')
        wrapped = CanarySessionASR(
            asr,
            effective,
            lid=getattr(asr, 'lid_model', None),
            default_lang=getattr(args, 'canary_default_lang', 'en'),
            lid_min_sec=getattr(args, 'canary_lid_min_sec', 2.0),
            lid_min_conf=getattr(args, 'canary_lid_min_conf', 0.5),
        )
        return OnlineASRProcessor(wrapped)

    # Wrap the shared ASR with per-session language and decoder context. For
    # SimulStreaming, the proxy also exposes an isolated cfg copy consumed by
    # SimulStreamingOnlineProcessor at construction time.
    if language is not None or context is not None:
        if getattr(args, "backend", None) == "funasr":
            from whisperlivekit.config import FUNASR_LANGUAGES

            if language not in FUNASR_LANGUAGES:
                supported = ", ".join(sorted(FUNASR_LANGUAGES))
                raise ValueError(
                    f"FunASR SenseVoiceSmall supports only: {supported}."
                )
        asr = SessionASRProxy(
            asr,
            language,
            context=context,
            simulstreaming=(
                getattr(args, "backend_policy", None) == "simulstreaming"
            ),
        )

    if backend == "qwen3-streaming":
        from whisperlivekit.qwen3_streaming import Qwen3StreamingOnlineProcessor
        return _ASRTokenNormalizer(Qwen3StreamingOnlineProcessor(asr))
    if backend == "qwen3-vllm":
        from whisperlivekit.qwen3_vllm_asr import (
            Qwen3VLLMCausalOnlineProcessor,
            Qwen3VLLMOnlineProcessor,
        )
        if getattr(asr, "audio_backend", "standard") == "causal":
            return _ASRTokenNormalizer(Qwen3VLLMCausalOnlineProcessor(asr))
        return _ASRTokenNormalizer(Qwen3VLLMOnlineProcessor(asr))
    if backend == "qwen3-vllm-metal":
        from whisperlivekit.qwen3_vllm_metal_asr import (
            Qwen3VLLMMetalCausalOnlineProcessor,
            Qwen3VLLMMetalOnlineProcessor,
        )
        if getattr(asr, "audio_backend", "standard") == "causal":
            return _ASRTokenNormalizer(Qwen3VLLMMetalCausalOnlineProcessor(asr))
        return _ASRTokenNormalizer(Qwen3VLLMMetalOnlineProcessor(asr))
    if backend == "voxtral-mlx":
        from whisperlivekit.voxtral_mlx_asr import VoxtralMLXOnlineProcessor
        return VoxtralMLXOnlineProcessor(asr)
    if backend == "voxtral":
        from whisperlivekit.voxtral_hf_streaming import VoxtralHFStreamingOnlineProcessor
        return VoxtralHFStreamingOnlineProcessor(asr)
    if backend == "funasr":
        from whisperlivekit.funasr_backend import FunASROnlineASRProcessor
        if not isinstance(asr, SessionASRProxy):
            asr = SessionASRProxy(asr)
        return FunASROnlineASRProcessor(asr)
    if getattr(args, "backend_policy", None) == "simulstreaming":
        from whisperlivekit.simul_whisper import SimulStreamingOnlineProcessor
        return SimulStreamingOnlineProcessor(asr)
    if not isinstance(asr, SessionASRProxy):
        # Every shared LocalAgreement backend participates in the same lock,
        # including sessions that use the server-wide language. Otherwise a
        # plain session could race with a language-overriding proxy and observe
        # its temporary ``original_language`` value.
        asr = SessionASRProxy(asr)
    return OnlineASRProcessor(asr)


def online_diarization_factory(args, diarization_backend):
    if args.diarization_backend == "diart":
        online = diarization_backend
        # Not the best here, since several user/instances will share the same backend, but diart is not SOTA anymore and sortformer is recommended
    elif args.diarization_backend == "sortformer":
        from whisperlivekit.diarization.sortformer_backend import SortformerDiarizationOnline
        online = SortformerDiarizationOnline(
            shared_model=diarization_backend,
            max_speakers=getattr(args, "sortformer_max_speakers", None),
        )
    else:
        raise ValueError(f"Unknown diarization backend: {args.diarization_backend}")
    return online


def online_translation_factory(args, translation_model):
    from whisperlivekit.translation_alignatt import AlignAttRemoteEngine
    if isinstance(translation_model, AlignAttRemoteEngine):
        return translation_model.new_session(args.target_language)
    # mlx-llm-mt: create a per-session client (fresh state) sharing the model cache.
    from whisperlivekit.translation_mlx_llm_mt import MlxLlmTranslation
    if isinstance(translation_model, MlxLlmTranslation):
        return translation_model.new_session(args.target_language)
    #should be at speaker level in the future:
    #one shared nllb model for all speaker
    #one tokenizer per speaker/language
    from nllw import OnlineTranslation
    source_language = _nllw_language_code(args.lan)
    target_language = _nllw_language_code(args.target_language)
    return OnlineTranslation(translation_model, [source_language], [target_language])
