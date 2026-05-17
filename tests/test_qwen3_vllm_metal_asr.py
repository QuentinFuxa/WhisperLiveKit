from argparse import Namespace

import numpy as np
import pytest


def _processor_for(text: str, holdback_words: int = 2):
    from whisperlivekit.qwen3_vllm_metal_asr import Qwen3VLLMMetalOnlineProcessor

    class FakeASR:
        sep = ""
        trim_sentence_buffer = False
        min_chunk_size = 0.0

        def __init__(self):
            self.holdback_words = holdback_words

        def transcribe_text(self, audio):
            return text

    processor = Qwen3VLLMMetalOnlineProcessor(FakeASR())
    processor.insert_audio_chunk(np.zeros(16000, dtype=np.float32), 1.0)
    return processor


def _texts(tokens):
    return [token.text for token in tokens]


def test_qwen3_vllm_metal_empty_text_commits_nothing():
    processor = _processor_for("")

    tokens, _ = processor.process_iter()

    assert tokens == []
    assert processor.get_buffer().text == ""


def test_qwen3_vllm_metal_two_words_are_buffered():
    processor = _processor_for("one two")

    tokens, _ = processor.process_iter()

    assert tokens == []
    assert processor.get_buffer().text == "one two"


def test_qwen3_vllm_metal_three_words_commits_one_and_buffers_two():
    processor = _processor_for("one two three")

    tokens, _ = processor.process_iter()

    assert _texts(tokens) == ["one"]
    assert processor.get_buffer().text == "two three"


def test_qwen3_vllm_metal_start_silence_flushes_buffered_words():
    processor = _processor_for("one two three")
    tokens, _ = processor.process_iter()
    assert _texts(tokens) == ["one"]

    tokens, _ = processor.start_silence()

    assert _texts(tokens) == [" two", " three"]
    assert processor.get_buffer().text == ""
    assert len(processor.audio_buffer) == 0


def test_qwen3_vllm_metal_finish_flushes_buffered_words():
    processor = _processor_for("one two three")
    tokens, _ = processor.process_iter()
    assert _texts(tokens) == ["one"]

    tokens, _ = processor.finish()

    assert _texts(tokens) == [" two", " three"]
    assert processor.get_buffer().text == ""


def test_qwen3_vllm_metal_holdback_words_is_configurable():
    processor = _processor_for("one two three", holdback_words=1)

    tokens, _ = processor.process_iter()

    assert _texts(tokens) == ["one", " two"]
    assert processor.get_buffer().text == "three"


def test_qwen3_vllm_metal_model_path_aliases():
    from whisperlivekit.qwen3_vllm_metal_asr import (
        DEFAULT_QWEN3_VLLM_METAL_MODEL,
        QWEN3_VLLM_METAL_1_7B_MODEL,
        _resolve_model_path,
    )

    assert _resolve_model_path({}) == DEFAULT_QWEN3_VLLM_METAL_MODEL
    assert _resolve_model_path({"model_size": "0.6b"}) == DEFAULT_QWEN3_VLLM_METAL_MODEL
    assert _resolve_model_path({"model_size": "qwen3-asr-0.6b"}) == DEFAULT_QWEN3_VLLM_METAL_MODEL
    assert _resolve_model_path({"model_size": "1.7b"}) == QWEN3_VLLM_METAL_1_7B_MODEL
    assert _resolve_model_path({"model_size": "qwen3-asr-1.7b"}) == QWEN3_VLLM_METAL_1_7B_MODEL
    assert _resolve_model_path({"model_size": "Qwen/Qwen3-ASR-0.6B"}) == "Qwen/Qwen3-ASR-0.6B"
    assert _resolve_model_path({"model_size": "./local-model"}) == "./local-model"
    assert _resolve_model_path({"model_path": "/models/qwen"}) == "/models/qwen"
    assert _resolve_model_path({"model_dir": "/models/qwen-dir"}) == "/models/qwen-dir"


def test_qwen3_vllm_metal_rejects_unsupported_model_alias():
    from whisperlivekit.qwen3_vllm_metal_asr import _resolve_model_path

    with pytest.raises(ValueError, match="supports Qwen3-ASR 0.6B and 1.7B"):
        _resolve_model_path({"model_size": "large-v3"})


def test_qwen3_vllm_metal_resolves_shared_vllm_dtype_names():
    from whisperlivekit.qwen3_vllm_metal_asr import _resolve_mlx_dtype

    class FakeMX:
        float16 = "float16"
        bfloat16 = "bfloat16"
        float32 = "float32"

    assert _resolve_mlx_dtype(FakeMX, {"vllm_dtype": "auto"}) == "float16"
    assert _resolve_mlx_dtype(FakeMX, {"vllm_dtype": "float16"}) == "float16"
    assert _resolve_mlx_dtype(FakeMX, {"vllm_dtype": "bf16"}) == "bfloat16"
    assert _resolve_mlx_dtype(FakeMX, {"vllm_dtype": "float32"}) == "float32"
    assert _resolve_mlx_dtype(FakeMX, {"dtype": "custom", "vllm_dtype": "bf16"}) == "custom"

    with pytest.raises(ValueError, match="vllm_dtype must be one of"):
        _resolve_mlx_dtype(FakeMX, {"vllm_dtype": "int8"})


def test_qwen3_vllm_metal_dependency_errors_are_specific():
    from whisperlivekit.qwen3_vllm_metal_asr import _missing_dependency_error

    assert "vLLM CPU package is missing" in str(
        _missing_dependency_error(ImportError("missing", name="vllm"))
    )
    assert "requires vllm-metal with STT support" in str(
        _missing_dependency_error(ImportError("missing", name="vllm_metal.stt"))
    )
    assert "requires MLX" in str(
        _missing_dependency_error(ImportError("missing", name="mlx.core"))
    )


def test_qwen3_vllm_metal_rejects_unsupported_platform(monkeypatch):
    import whisperlivekit.qwen3_vllm_metal_asr as qwen_metal

    monkeypatch.setattr(qwen_metal.platform, "system", lambda: "Linux")
    monkeypatch.setattr(qwen_metal.platform, "machine", lambda: "x86_64")

    with pytest.raises(ImportError, match="Apple Silicon"):
        qwen_metal._ensure_supported_platform()


def test_qwen3_vllm_metal_decodes_without_special_tokens():
    from whisperlivekit.qwen3_vllm_metal_asr import Qwen3VLLMMetalASR

    class FakeTokenizer:
        def decode(self, tokens, skip_special_tokens=False):
            assert tokens == [1, 2, 3]
            assert skip_special_tokens is True
            return "hello"

    asr = object.__new__(Qwen3VLLMMetalASR)
    asr.tokenizer = FakeTokenizer()
    asr._post_process_output = lambda text: text

    assert asr._decode_output_tokens([1, 2, 3]) == "hello"


def test_parse_args_accepts_qwen3_vllm_metal(monkeypatch):
    from whisperlivekit.parse_args import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "whisperlivekit-server",
            "--backend",
            "qwen3-vllm-metal",
            "--model",
            "0.6b",
            "--holdback-words",
            "3",
            "--no-trim-sentence-buffer",
        ],
    )

    args = parse_args()

    assert args.backend == "qwen3-vllm-metal"
    assert args.model_size == "0.6b"
    assert args.holdback_words == 3
    assert args.trim_sentence_buffer is False


def test_transcription_engine_routes_to_qwen3_vllm_metal(monkeypatch):
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
            holdback_words=3,
            trim_sentence_buffer=False,
        )
    finally:
        TranscriptionEngine.reset()

    assert isinstance(engine.asr, FakeASR)
    assert seen["kwargs"]["holdback_words"] == 3
    assert seen["kwargs"]["trim_sentence_buffer"] is False


def test_online_factory_routes_to_qwen3_vllm_metal_processor():
    from whisperlivekit.core import online_factory
    from whisperlivekit.qwen3_vllm_metal_asr import Qwen3VLLMMetalOnlineProcessor

    class FakeASR:
        sep = ""
        holdback_words = 2
        trim_sentence_buffer = False
        min_chunk_size = 0.0

    processor = online_factory(
        Namespace(backend="qwen3-vllm-metal"),
        FakeASR(),
    )

    assert isinstance(processor, Qwen3VLLMMetalOnlineProcessor)
