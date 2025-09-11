
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Whisper FastAPI Online Server")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="The host address to bind the server to.",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="The port number to bind the server to."
    )
    parser.add_argument(
        "--warmup-file",
        type=str,
        default=None,
        dest="warmup_file",
        help="""
        The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast.
        If not set, uses https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav.
        If False, no warmup is performed.
        """,
    )

    parser.add_argument(
        "--confidence-validation",
        action="store_true",
        help="Accelerates validation of tokens using confidence scores. Transcription will be faster but punctuation might be less accurate.",
    )

    parser.add_argument(
        "--diarization",
        action="store_true",
        default=False,
        help="Enable speaker diarization.",
    )

    parser.add_argument(
        "--punctuation-split",
        action="store_true",
        default=False,
        help="Use punctuation marks from transcription to improve speaker boundary detection. Requires both transcription and diarization to be enabled.",
    )

    parser.add_argument(
        "--segmentation-model",
        type=str,
        default="pyannote/segmentation-3.0",
        help="Hugging Face model ID for pyannote.audio segmentation model.",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="pyannote/embedding",
        help="Hugging Face model ID for pyannote.audio embedding model.",
    )

    parser.add_argument(
        "--diarization-backend",
        type=str,
        default="sortformer",
        choices=["sortformer", "diart"],
        help="The diarization backend to use.",
    )

    parser.add_argument(
        "--no-transcription",
        action="store_true",
        help="Disable transcription to only see live diarization results.",
    )
    
    parser.add_argument(
        "--min-chunk-size",
        type=float,
        default=0.5,
        help="Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        help="Name size of the Whisper model to use (default: tiny). Suggested values: tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo. The model is automatically downloaded from the model hub if not present in model cache dir.",
    )
    
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Overriding the default model cache dir where models downloaded from the hub are saved",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.",
    )
    parser.add_argument(
        "--lan",
        "--language",
        type=str,
        default="auto",
        help="Source language code, e.g. en,de,cs, or 'auto' for language detection.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Transcribe or translate.",
    )
    
    parser.add_argument(
        "--target-language",
        type=str,
        default="",
        dest="target_language",
        help="Target language for translation. Not functional yet.",
    )    

    parser.add_argument(
        "--backend",
        type=str,
        default="simulstreaming",
        choices=["faster-whisper", "whisper_timestamped", "mlx-whisper", "openai-api", "simulstreaming"],
        help="Load only this backend for Whisper processing.",
    )
    parser.add_argument(
        "--no-vac",
        action="store_true",
        default=False,
        help="Disable VAC = voice activity controller.",
    )
    parser.add_argument(
        "--vac-chunk-size", type=float, default=0.04, help="VAC sample size in seconds."
    )

    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD (voice activity detection).",
    )
    
    parser.add_argument(
        "--buffer_trimming",
        type=str,
        default="segment",
        choices=["sentence", "segment"],
        help='Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.',
    )
    parser.add_argument(
        "--buffer_trimming_sec",
        type=float,
        default=15,
        help="Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the log level",
        default="DEBUG",
    )
    parser.add_argument("--ssl-certfile", type=str, help="Path to the SSL certificate file.", default=None)
    parser.add_argument("--ssl-keyfile", type=str, help="Path to the SSL private key file.", default=None)
    parser.add_argument(
        "--pcm-input",
        action="store_true",
        default=False,
        help="If set, raw PCM (s16le) data is expected as input and FFmpeg will be bypassed."
    )

    # SimulStreaming-specific arguments
    simulstreaming_group = parser.add_argument_group('SimulStreaming arguments (only used with --backend simulstreaming)')

    simulstreaming_group.add_argument(
        "--disable-fast-encoder",
        action="store_true",
        default=False,
        dest="disable_fast_encoder",
        help="Disable Faster Whisper or MLX Whisper backends for encoding (if installed). Slower but helpful when GPU memory is limited",
    )
    
    simulstreaming_group.add_argument(
        "--frame-threshold",
        type=int,
        default=25,
        dest="frame_threshold",
        help="Threshold for the attention-guided decoding. The AlignAtt policy will decode only until this number of frames from the end of audio. In frames: one frame is 0.02 seconds for large-v3 model.",
    )
    
    simulstreaming_group.add_argument(
        "--beams",
        "-b",
        type=int,
        default=1,
        help="Number of beams for beam search decoding. If 1, GreedyDecoder is used.",
    )
    
    simulstreaming_group.add_argument(
        "--decoder",
        type=str,
        default=None,
        dest="decoder_type",
        choices=["beam", "greedy"],
        help="Override automatic selection of beam or greedy decoder. If beams > 1 and greedy: invalid.",
    )
    
    simulstreaming_group.add_argument(
        "--audio-max-len",
        type=float,
        default=30.0,
        dest="audio_max_len",
        help="Max length of the audio buffer, in seconds.",
    )
    
    simulstreaming_group.add_argument(
        "--audio-min-len",
        type=float,
        default=0.0,
        dest="audio_min_len",
        help="Skip processing if the audio buffer is shorter than this length, in seconds. Useful when the --min-chunk-size is small.",
    )
    
    simulstreaming_group.add_argument(
        "--cif-ckpt-path",
        type=str,
        default=None,
        dest="cif_ckpt_path",
        help="The file path to the Simul-Whisper's CIF model checkpoint that detects whether there is end of word at the end of the chunk. If not, the last decoded space-separated word is truncated because it is often wrong -- transcribing a word in the middle. The CIF model adapted for the Whisper model version should be used. Find the models in https://github.com/backspacetg/simul_whisper/tree/main/cif_models . Note that there is no model for large-v3.",
    )
    
    simulstreaming_group.add_argument(
        "--never-fire",
        action="store_true",
        default=False,
        dest="never_fire",
        help="Override the CIF model. If True, the last word is NEVER truncated, no matter what the CIF model detects. If False: if CIF model path is set, the last word is SOMETIMES truncated, depending on the CIF detection. Otherwise, if the CIF model path is not set, the last word is ALWAYS trimmed.",
    )
    
    simulstreaming_group.add_argument(
        "--init-prompt",
        type=str,
        default=None,
        dest="init_prompt",
        help="Init prompt for the model. It should be in the target language.",
    )
    
    simulstreaming_group.add_argument(
        "--static-init-prompt",
        type=str,
        default=None,
        dest="static_init_prompt",
        help="Do not scroll over this text. It can contain terminology that should be relevant over all document.",
    )
    
    simulstreaming_group.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        dest="max_context_tokens",
        help="Max context tokens for the model. Default is 0.",
    )
    
    simulstreaming_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        dest="model_path",
        help="Direct path to the SimulStreaming Whisper .pt model file. Overrides --model for SimulStreaming backend.",
    )
    
    simulstreaming_group.add_argument(
        "--preload-model-count",
        type=int,
        default=1,
        dest="preload_model_count",
        help="Optional. Number of models to preload in memory to speed up loading (set up to the expected number of concurrent instances).",
    )

    args = parser.parse_args()
    
    args.transcription = not args.no_transcription
    args.vad = not args.no_vad    
    delattr(args, 'no_transcription')
    delattr(args, 'no_vad')
    
    return args
