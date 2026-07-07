
from argparse import ArgumentParser, BooleanOptionalAction


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
        If empty, no warmup is performed.
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
        "--sortformer-model-path",
        type=str,
        default=None,
        dest="sortformer_model_path",
        help="Path to a local Sortformer .nemo file, a directory containing exactly one .nemo file, or a NeMo/Hugging Face model ID.",
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
        "--disable-punctuation-split",
        action="store_true",
        help="Disable the split parameter.",
    )

    parser.add_argument(
        "--min-chunk-size",
        type=float,
        default=0.1,
        help="Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.",
    )

    parser.add_argument(
        "--retention-seconds",
        type=float,
        default=None,
        dest="retention_seconds",
        help=(
            "Transcript history kept in server memory per session. Default: "
            "unlimited for mode=full sessions (the client is sent the whole "
            "transcript each update), 300 for diff-mode sessions. "
            "0 = unlimited."
        ),
    )

    parser.add_argument(
        "--rest-timeout",
        type=float,
        default=0.0,
        dest="rest_timeout",
        help=(
            "Processing budget in seconds for /v1/audio/transcriptions. "
            "0 = auto: max(120, 2.5x the audio duration). On expiry the "
            "endpoint returns HTTP 408 instead of a silent empty result."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default="base",
        dest='model_size',
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
        "--lora-path",
        type=str,
        default=None,
        dest="lora_path",
        help="Path or Hugging Face repo ID for LoRA adapter weights (e.g., QuentinFuxa/whisper-base-french-lora). Only works with native Whisper backend.",
    )
    parser.add_argument(
        "--lan",
        "--language",
        type=str,
        default="auto",
        dest='lan',
        help="Source language code, e.g. en,de,cs, or 'auto' for language detection.",
    )
    parser.add_argument(
        "--direct-english-translation",
        action="store_true",
        default=False,
        help="Use Whisper to directly translate to english.",
    )

    parser.add_argument(
        "--target-language",
        type=str,
        default="",
        dest="target_language",
        help="Target language for translation. Not functional yet.",
    )

    parser.add_argument(
        "--backend-policy",
        type=str,
        default="simulstreaming",
        choices=["1", "2", "simulstreaming", "localagreement"],
        help="Select the streaming policy: 1 or 'simulstreaming' for AlignAtt, 2 or 'localagreement' for LocalAgreement.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "mlx-whisper", "faster-whisper", "whisper", "openai-api", "voxtral", "voxtral-mlx", "qwen3-vllm", "qwen3-vllm-metal", "qwen3-streaming"],
        help="Select the ASR backend implementation. Use 'qwen3-vllm' for Qwen3-ASR through in-process vLLM with ForcedAligner on GPU. Use 'qwen3-vllm-metal' for Qwen3-ASR through vllm-metal in-process STT on Apple Silicon. Use 'qwen3-streaming' for Qwen3-ASR through plain HF Transformers with a bounded-recompute audio cache (CUDA/MPS/CPU, no vLLM; requires an explicit --language).",
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
    parser.add_argument("--forwarded-allow-ips", type=str, help="Allowed ips for reverse proxying.", default=None)
    parser.add_argument(
        "--cors-origins",
        type=str,
        default="",
        dest="cors_origins",
        help="Comma-separated list of allowed CORS origins. Empty disables CORS; use '*' to allow all origins.",
    )
    parser.add_argument(
        "--pcm-input",
        action="store_true",
        default=False,
        help="If set, raw PCM (s16le) data is expected as input and FFmpeg will be bypassed. Frontend will use AudioWorklet instead of MediaRecorder."
    )
    # vLLM Qwen3 backend arguments
    parser.add_argument(
        "--vllm-model",
        type=str,
        default="",
        dest="vllm_model",
        help="Model name to use with vLLM (e.g. Qwen/Qwen3-ASR-1.7B).",
    )
    parser.add_argument(
        "--vllm-aligner-model",
        type=str,
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        dest="vllm_aligner_model",
        help="ForcedAligner model name to use with the qwen3-vllm backend.",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        dest="vllm_tensor_parallel_size",
        help="Tensor parallel size for the qwen3-vllm in-process backend.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.45,
        dest="vllm_gpu_memory_utilization",
        help="GPU memory utilization fraction for qwen3-vllm vLLM engines.",
    )
    parser.add_argument(
        "--vllm-dtype",
        type=str,
        default="auto",
        dest="vllm_dtype",
        help="dtype passed to vLLM for qwen3-vllm engines, e.g. auto, bfloat16, float16.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=0,
        dest="vllm_max_model_len",
        help=(
            "Optional max_model_len passed to qwen3-vllm engines. "
            "0 keeps the model default."
        ),
    )
    parser.add_argument(
        "--qwen3-vllm-audio-backend",
        choices=["standard", "causal"],
        default="standard",
        dest="qwen3_vllm_audio_backend",
        help=(
            "Qwen3 CUDA/vLLM audio backend. 'standard' re-encodes the current "
            "buffer; 'causal' uses the append-only causal audio tower. The "
            "causal decoder path is selected by "
            "--qwen3-vllm-causal-decoder-backend."
        ),
    )
    parser.add_argument(
        "--qwen3-vllm-causal-decoder-backend",
        choices=["append-kv", "rolling", "vllm", "vllm-live", "vllm-text"],
        default="vllm-text",
        dest="qwen3_vllm_causal_decoder_backend",
        help=(
            "Decoder used by qwen3-vllm when --qwen3-vllm-audio-backend=causal. "
            "'append-kv' keeps a persistent decoder KV over the prompt head "
            "and audio prefix and uses vLLM only for ForcedAligner timestamps; "
            "'rolling' is the older name for the same experimental path; "
            "'vllm' feeds causal audio embeddings to a fresh Qwen3-ASR vLLM "
            "request per chunk; 'vllm-live' keeps one streaming vLLM text "
            "decoder request open and appends prompt embeddings with a "
            "request-local KV prefix; 'vllm-text' exports Qwen3-ASR's text "
            "decoder as Qwen3ForCausalLM and feeds it causal audio prompt "
            "embeddings through fresh vLLM requests with prefix caching."
        ),
    )
    parser.add_argument(
        "--qwen3-vllm-text-decoder-model",
        type=str,
        default="",
        dest="qwen3_vllm_text_decoder_model",
        help=(
            "Optional local Qwen3ForCausalLM export for "
            "--qwen3-vllm-causal-decoder-backend vllm-text or vllm-live. "
            "If unset, the decoder is exported into the WhisperLiveKit cache."
        ),
    )
    parser.add_argument(
        "--qwen3-vllm-live-idle-timeout-ms",
        type=float,
        default=50.0,
        dest="qwen3_vllm_live_idle_timeout_ms",
        help=(
            "Idle gap used to close one partial vLLM-live decode update after "
            "the last streamed token. Lower values reduce latency; higher "
            "values are more conservative on slow GPUs."
        ),
    )
    parser.add_argument(
        "--qwen3-vllm-causal-attn-implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="auto",
        dest="qwen3_vllm_causal_attn_implementation",
        help=(
            "Transformers attention implementation for the qwen3-vllm causal "
            "rolling ASR decoder. Ignored by the legacy vLLM decoder path."
        ),
    )
    parser.add_argument(
        "--qwen3-vllm-tower-checkpoint",
        type=str,
        default="",
        dest="qwen3_vllm_tower_checkpoint",
        help=(
            "Local path or Hugging Face repo id for the qwen3-vllm causal "
            "audio tower checkpoint. Defaults to qfuxa/qwen3-asr-0.6b-streaming "
            "when causal mode is enabled."
        ),
    )
    parser.add_argument(
        "--qwen3-vllm-left-context-sec",
        type=float,
        default=15.0,
        dest="qwen3_vllm_left_context_sec",
        help="Left context retained in the qwen3-vllm causal audio KV cache.",
    )
    parser.add_argument(
        "--qwen3-vllm-block-frames",
        type=int,
        default=192,
        dest="qwen3_vllm_block_frames",
        help="Fixed mel-frame block size for qwen3-vllm causal audio encoding.",
    )
    parser.add_argument(
        "--qwen3-vllm-cache-block-size",
        type=int,
        default=0,
        dest="qwen3_vllm_cache_block_size",
        help=(
            "Optional vLLM KV/prefix-cache block size for qwen3-vllm engines. "
            "0 keeps vLLM's platform default."
        ),
    )
    parser.add_argument(
        "--qwen3-vllm-segment-max-steps",
        type=int,
        default=150,
        dest="qwen3_vllm_segment_max_steps",
        help=(
            "Maximum cached causal audio decoder steps before qwen3-vllm "
            "rolls to a fresh no-past-rewrite segment. 0 disables rollover."
        ),
    )
    parser.add_argument(
        "--qwen3-vllm-segment-min-sec",
        type=float,
        default=0.0,
        dest="qwen3_vllm_segment_min_sec",
        help=(
            "Minimum active causal segment duration before qwen3-vllm may "
            "roll. Increase this to avoid rollover on short clips."
        ),
    )
    parser.add_argument(
        "--qwen3-vllm-live-multiprocessing",
        action=BooleanOptionalAction,
        default=None,
        dest="qwen3_vllm_live_multiprocessing",
        help=(
            "Run the qwen3-vllm vllm-live decoder engine in a separate vLLM "
            "worker process. Default keeps the deprecated "
            "WLK_QWEN3_VLLM_LIVE_MULTIPROCESSING env fallback (off)."
        ),
    )
    parser.add_argument(
        "--qwen3-vllm-aligner-multiprocessing",
        action=BooleanOptionalAction,
        default=None,
        dest="qwen3_vllm_aligner_multiprocessing",
        help=(
            "Run the ForcedAligner engine in a separate vLLM worker process "
            "when vllm-live multiprocessing is on. Default keeps the "
            "deprecated WLK_QWEN3_VLLM_ALIGNER_MULTIPROCESSING env fallback."
        ),
    )
    parser.add_argument(
        "--qwen3-vllm-prompt-context-words",
        type=int,
        default=0,
        dest="qwen3_vllm_prompt_context_words",
        help=(
            "Number of previously committed transcript words injected into "
            "the next qwen3-vllm causal segment prompt after rollover."
        ),
    )
    parser.add_argument(
        "--holdback-words",
        type=int,
        default=None,
        dest="holdback_words",
        help="For Qwen3 vllm-metal, keep this many trailing words uncommitted.",
    )
    parser.add_argument(
        "--no-trim-sentence-buffer",
        action="store_false",
        default=True,
        dest="trim_sentence_buffer",
        help="Disable Qwen3 vllm-metal buffer trimming at committed sentence boundaries.",
    )
    parser.add_argument(
        "--qwen3-vllm-metal-audio-backend",
        choices=["standard", "causal"],
        default="standard",
        dest="qwen3_vllm_metal_audio_backend",
        help=(
            "Qwen3 vllm-metal audio backend. 'standard' re-encodes the current "
            "buffer; 'causal' uses the experimental append-only causal MLX audio "
            "tower."
        ),
    )
    parser.add_argument(
        "--qwen3-vllm-metal-tower-checkpoint",
        type=str,
        default="",
        dest="qwen3_vllm_metal_tower_checkpoint",
        help=(
            "Local path or Hugging Face repo id for the qwen3-vllm-metal causal "
            "audio tower checkpoint. Defaults to qfuxa/qwen3-asr-0.6b-streaming "
            "when causal mode is enabled."
        ),
    )
    parser.add_argument(
        "--qwen3-vllm-metal-left-context-sec",
        type=float,
        default=15.0,
        dest="qwen3_vllm_metal_left_context_sec",
        help="Left context retained in the qwen3-vllm-metal causal audio KV cache.",
    )
    parser.add_argument(
        "--qwen3-vllm-metal-block-frames",
        type=int,
        default=192,
        dest="qwen3_vllm_metal_block_frames",
        help="Fixed mel-frame block size for qwen3-vllm-metal causal audio encoding.",
    )

    # Qwen3 streaming backend arguments
    qwen3_streaming_group = parser.add_argument_group(
        'Qwen3 streaming backend arguments (only used with --backend qwen3-streaming)'
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-chunk-sec",
        type=float,
        default=2.0,
        dest="qwen3_streaming_chunk_sec",
        help="Minimum seconds of new audio per decode update. Decodes self-pace upward on slow hardware.",
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-left-context-sec",
        type=float,
        default=12.0,
        dest="qwen3_streaming_left_context_sec",
        help="Audio tower left context window in seconds (bounded recompute). Quality saturates around 12s on long-form audio.",
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-right-context-ms",
        type=int,
        default=640,
        dest="qwen3_streaming_right_context_ms",
        help="Audio tower right context in milliseconds before an encoder step is finalized.",
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-segment-max-steps",
        type=int,
        default=200,
        dest="qwen3_streaming_segment_max_steps",
        help="Cached decoder steps before the active segment is finalized and rolled (200 steps = ~15s of audio).",
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-segment-keep-tail-steps",
        type=int,
        default=0,
        dest="qwen3_streaming_segment_keep_tail_steps",
        help="Audio embedding steps carried over after a segment roll (0 = hard boundary, validated default).",
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-hold-back-words",
        type=int,
        default=6,
        dest="qwen3_streaming_hold_back_words",
        help="Trailing words held back from commitment until stable.",
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-stable-iterations",
        type=int,
        default=None,
        dest="qwen3_streaming_stable_iterations",
        help=(
            "Consecutive identical hypothesis prefixes required before "
            "committing. Default: 2 for the windowed backend, 1 for causal "
            "(measured p50 commit latency 4.0 s vs 5.9 s for +0.5 pt WER)."
        ),
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-max-new-tokens",
        type=int,
        default=256,
        dest="qwen3_streaming_max_new_tokens",
        help="Max tokens per full-hypothesis decode (bounds one segment's text).",
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        dest="qwen3_streaming_device",
        help="Device for the Qwen3 streaming model.",
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        dest="qwen3_streaming_dtype",
        help="Model dtype. auto = bfloat16 on CUDA, float16 on MPS, float32 on CPU.",
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-attn-implementation",
        type=str,
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        dest="qwen3_streaming_attn_implementation",
        help="Attention implementation passed to Transformers for Qwen3 streaming.",
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-context",
        type=str,
        default="",
        dest="qwen3_streaming_context",
        help="Optional system-prompt context (terminology, names) for the Qwen ASR prompt.",
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-prompt-context-words",
        type=int,
        default=0,
        dest="qwen3_streaming_prompt_context_words",
        help="Trailing transcript words injected into the next segment prompt (0 = disabled; enabling measured worse).",
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-audio-backend",
        choices=["windowed", "causal"],
        default="windowed",
        dest="qwen3_streaming_audio_backend",
        help=(
            "Audio encoder execution. 'windowed' (default) re-encodes a bounded "
            "window per update (best WER). 'causal' runs the append-only "
            "causal-KV encoder with a fine-tuned tower checkpoint: each audio "
            "block is encoded exactly once (lowest compute per chunk; requires "
            "--qwen3-streaming-tower-checkpoint)."
        ),
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-tower-checkpoint",
        type=str,
        default="",
        dest="qwen3_streaming_tower_checkpoint",
        help=(
            "Fine-tuned audio-tower weights for the causal backend: local "
            ".pt/.safetensors file, directory, or Hugging Face repo id "
            "(e.g. qfuxa/qwen3-asr-0.6b-streaming)."
        ),
    )
    qwen3_streaming_group.add_argument(
        "--qwen3-streaming-block-frames",
        type=int,
        default=192,
        dest="qwen3_streaming_block_frames",
        help=(
            "Fixed attention-block size in mel frames for the causal backend "
            "(96 or 192; must match the tower checkpoint training regime)."
        ),
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
        "--custom-alignment-heads",
        type=str,
        default=None,
        help="Use your own alignment heads, useful when `--model-dir` is used",
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
        help="Legacy alias for --decoder-model-path. Direct path to the SimulStreaming PyTorch Whisper decoder/alignment model.",
    )

    simulstreaming_group.add_argument(
        "--encoder-model-path",
        type=str,
        default=None,
        dest="encoder_model_path",
        help="Direct path or Hugging Face repo ID for the fast encoder weights used by SimulStreaming hybrid mode (CT2 for faster-whisper, MLX for mlx-whisper).",
    )

    simulstreaming_group.add_argument(
        "--decoder-model-path",
        type=str,
        default=None,
        dest="decoder_model_path",
        help="Direct path or Hugging Face repo ID for the PyTorch Whisper decoder/alignment weights used by SimulStreaming.",
    )

    simulstreaming_group.add_argument(
        "--nllb-backend",
        type=str,
        default="transformers",
        help="transformers or ctranslate2",
    )

    simulstreaming_group.add_argument(
        "--nllb-size",
        type=str,
        default="600M",
        help="600M or 1.3B",
    )

    args = parser.parse_args()
    args.transcription = not args.no_transcription
    args.vad = not args.no_vad
    args.vac = not args.no_vac
    delattr(args, 'no_transcription')
    delattr(args, 'no_vad')
    delattr(args, 'no_vac')

    from whisperlivekit.config import WhisperLiveKitConfig
    return WhisperLiveKitConfig.from_namespace(args)
