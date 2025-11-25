import logging
import sys
from argparse import Namespace

from whisperlivekit.local_agreement.online_asr import OnlineASRProcessor
from whisperlivekit.local_agreement.whisper_online import backend_factory
from whisperlivekit.simul_whisper import SimulStreamingASR


def update_with_kwargs(_dict, kwargs):
    _dict.update({
        k: v for k, v in kwargs.items() if k in _dict
    })
    return _dict


logger = logging.getLogger(__name__)

class TranscriptionEngine:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, **kwargs):
        if TranscriptionEngine._initialized:
            return

        global_params = {
            "host": "localhost",
            "port": 8000,
            "diarization": False,
            "punctuation_split": False,
            "target_language": "",
            "vac": True,
            "vac_chunk_size": 0.04,
            "log_level": "DEBUG",
            "ssl_certfile": None,
            "ssl_keyfile": None,
            "forwarded_allow_ips": None,
            "transcription": True,
            "vad": True,
            "pcm_input": False,
            "disable_punctuation_split" : False,
            "diarization_backend": "sortformer",
            "backend_policy": "simulstreaming",
            "backend": "auto",
        }
        global_params = update_with_kwargs(global_params, kwargs)

        transcription_common_params = {
            "warmup_file": None,
            "min_chunk_size": 0.1,
            "model_size": "base",
            "model_cache_dir": None,
            "model_dir": None,
            "model_path": None,
            "lora_path": None,
            "lan": "auto",
            "direct_english_translation": False,
        }
        transcription_common_params = update_with_kwargs(transcription_common_params, kwargs)                                            

        if transcription_common_params['model_size'].endswith(".en"):
            transcription_common_params["lan"] = "en"
        if 'no_transcription' in kwargs:
            global_params['transcription'] = not global_params['no_transcription']
        if 'no_vad' in kwargs:
            global_params['vad'] = not kwargs['no_vad']
        if 'no_vac' in kwargs:
            global_params['vac'] = not kwargs['no_vac']

        self.args = Namespace(**{**global_params, **transcription_common_params})
        
        self.asr = None
        self.tokenizer = None
        self.diarization = None
        self.vac_session = None
        
        if self.args.vac:
            from whisperlivekit.silero_vad_iterator import is_onnx_available
            
            if is_onnx_available():
                from whisperlivekit.silero_vad_iterator import load_onnx_session
                self.vac_session = load_onnx_session()
            else:
                logger.warning(
                    "onnxruntime not installed. VAC will use JIT model which is loaded per-session. "
                    "For multi-user scenarios, install onnxruntime: pip install onnxruntime"
                )
        backend_policy = self.args.backend_policy
        if self.args.transcription:
            if backend_policy == "simulstreaming":                 
                simulstreaming_params = {
                    "disable_fast_encoder": False,
                    "custom_alignment_heads": None,
                    "frame_threshold": 25,
                    "beams": 1,
                    "decoder_type": None,
                    "audio_max_len": 20.0,
                    "audio_min_len": 0.0,
                    "cif_ckpt_path": None,
                    "never_fire": False,
                    "init_prompt": None,
                    "static_init_prompt": None,
                    "max_context_tokens": None,
                }
                simulstreaming_params = update_with_kwargs(simulstreaming_params, kwargs)
                
                self.tokenizer = None        
                self.asr = SimulStreamingASR(
                    **transcription_common_params,
                    **simulstreaming_params,
                    backend=self.args.backend,
                )
                logger.info(
                    "Using SimulStreaming policy with %s backend",
                    getattr(self.asr, "encoder_backend", "whisper"),
                )
            else:
                
                whisperstreaming_params = {
                    "buffer_trimming": "segment",
                    "confidence_validation": False,
                    "buffer_trimming_sec": 15,
                }
                whisperstreaming_params = update_with_kwargs(whisperstreaming_params, kwargs)
                
                self.asr = backend_factory(
                    backend=self.args.backend,
                    **transcription_common_params,
                    **whisperstreaming_params,
                )
                logger.info(
                    "Using LocalAgreement policy with %s backend",
                    getattr(self.asr, "backend_choice", self.asr.__class__.__name__),
                )

        if self.args.diarization:
            if self.args.diarization_backend == "diart":
                from whisperlivekit.diarization.diart_backend import \
                    DiartDiarization
                diart_params = {
                    "segmentation_model": "pyannote/segmentation-3.0",
                    "embedding_model": "pyannote/embedding",
                }
                diart_params = update_with_kwargs(diart_params, kwargs)
                self.diarization_model = DiartDiarization(
                    block_duration=self.args.min_chunk_size,
                    **diart_params
                )
            elif self.args.diarization_backend == "sortformer":
                from whisperlivekit.diarization.sortformer_backend import \
                    SortformerDiarization
                self.diarization_model = SortformerDiarization()
        
        self.translation_model = None
        if self.args.target_language:
            if self.args.lan == 'auto' and backend_policy != "simulstreaming":
                raise Exception('Translation cannot be set with language auto when transcription backend is not simulstreaming')
            else:
                try:
                    from nllw import load_model
                except:
                    raise Exception('To use translation, you must install nllw: `pip install nllw`')
                translation_params = { 
                    "nllb_backend": "transformers",
                    "nllb_size": "600M"
                }
                translation_params = update_with_kwargs(translation_params, kwargs)
                self.translation_model = load_model([self.args.lan], **translation_params) #in the future we want to handle different languages for different speakers
        TranscriptionEngine._initialized = True


def online_factory(args, asr):
    if args.backend_policy == "simulstreaming":
        from whisperlivekit.simul_whisper import SimulStreamingOnlineProcessor
        return SimulStreamingOnlineProcessor(asr)
    return OnlineASRProcessor(asr)
  
  
def online_diarization_factory(args, diarization_backend):
    if args.diarization_backend == "diart":
        online = diarization_backend
        # Not the best here, since several user/instances will share the same backend, but diart is not SOTA anymore and sortformer is recommended
    
    if args.diarization_backend == "sortformer":
        from whisperlivekit.diarization.sortformer_backend import \
            SortformerDiarizationOnline
        online = SortformerDiarizationOnline(shared_model=diarization_backend)
    return online


def online_translation_factory(args, translation_model):
    #should be at speaker level in the future:
    #one shared nllb model for all speaker
    #one tokenizer per speaker/language
    from nllw import OnlineTranslation
    return OnlineTranslation(translation_model, [args.lan], [args.target_language])
