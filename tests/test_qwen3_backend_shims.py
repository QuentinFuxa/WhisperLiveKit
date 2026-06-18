from argparse import Namespace


def test_qwen3_streaming_shim_exposes_standalone_private_helpers():
    import whisperlivekit.qwen3_streaming.model as wlk_model
    import qwen3_asr_causal.model as qwen_model

    assert wlk_model.Qwen3ASRRealtimeQwenAudioSurgeryModel is qwen_model.Qwen3ASRRealtimeQwenAudioSurgeryModel
    assert wlk_model._split_prompt_template is qwen_model._split_prompt_template


def test_qwen3_vllm_shim_preserves_whisperlivekit_default_model():
    import whisperlivekit.qwen3_vllm_asr as qwen_vllm

    assert qwen_vllm._resolve_model_path({}) == "Qwen/Qwen3-ASR-1.7B"
    assert qwen_vllm._resolve_model_path({"model_size": "0.6b"}) == "Qwen/Qwen3-ASR-0.6B"


def test_qwen3_vllm_metal_shim_imports_standalone_backend():
    import whisperlivekit.qwen3_vllm_metal_asr as wlk_metal
    import qwen3_asr_causal.metal as qwen_metal

    assert wlk_metal.Qwen3VLLMMetalASR is qwen_metal.Qwen3VLLMMetalASR
    assert wlk_metal._resolve_audio_backend({"qwen3_vllm_metal_audio_backend": "causal"}) == "causal"


def test_parse_args_accepts_qwen3_causal_options(monkeypatch):
    from whisperlivekit.parse_args import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "whisperlivekit-server",
            "--backend",
            "qwen3-vllm",
            "--qwen3-vllm-audio-backend",
            "causal",
            "--qwen3-vllm-causal-decoder-backend",
            "vllm-live",
            "--qwen3-vllm-live-idle-timeout-ms",
            "25",
            "--qwen3-vllm-cache-block-size",
            "8",
            "--qwen3-vllm-metal-audio-backend",
            "causal",
            "--qwen3-vllm-metal-tower-checkpoint",
            "tower",
        ],
    )

    args = parse_args()

    assert args.backend == "qwen3-vllm"
    assert args.qwen3_vllm_audio_backend == "causal"
    assert args.qwen3_vllm_causal_decoder_backend == "vllm-live"
    assert args.qwen3_vllm_live_idle_timeout_ms == 25
    assert args.qwen3_vllm_cache_block_size == 8
    assert args.qwen3_vllm_metal_audio_backend == "causal"
    assert args.qwen3_vllm_metal_tower_checkpoint == "tower"


def test_online_factory_routes_qwen3_streaming():
    from qwen3_asr_causal.online import Qwen3StreamingOnlineProcessor
    from whisperlivekit.core import online_factory

    class FakeASR:
        sep = ""
        SAMPLING_RATE = 16_000
        chunk_sec = 2.0
        original_language = "en"

        def build_streamer(self, language=None):
            return object()

        def new_mel_extractor(self):
            return object()

    processor = online_factory(Namespace(backend="qwen3-streaming"), FakeASR())

    assert isinstance(processor, Qwen3StreamingOnlineProcessor)


def test_transcription_engine_routes_qwen3_vllm_metal(monkeypatch):
    import whisperlivekit.qwen3_vllm_metal_asr as qwen_metal
    from whisperlivekit.core import TranscriptionEngine

    seen = {}

    class FakeASR:
        sep = ""

        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def use_vad(self):
            return False

    monkeypatch.setattr(qwen_metal, "Qwen3VLLMMetalASR", FakeASR)
    TranscriptionEngine.reset()
    try:
        engine = TranscriptionEngine(
            backend="qwen3-vllm-metal",
            model_size="0.6b",
            lan="auto",
            vac=False,
            vad=False,
            diarization=False,
            qwen3_vllm_metal_audio_backend="causal",
            qwen3_vllm_metal_tower_checkpoint="tower",
        )
    finally:
        TranscriptionEngine.reset()

    assert isinstance(engine.asr, FakeASR)
    assert seen["kwargs"]["qwen3_vllm_metal_audio_backend"] == "causal"
    assert seen["kwargs"]["qwen3_vllm_metal_tower_checkpoint"] == "tower"
