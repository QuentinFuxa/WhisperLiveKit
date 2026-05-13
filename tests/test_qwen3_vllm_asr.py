from types import SimpleNamespace

import numpy as np

from whisperlivekit.core import online_factory
from whisperlivekit.qwen3_vllm_asr import (
    Qwen3VLLMOnlineProcessor,
    _AlignedWord,
    _fix_timestamps,
    _load_vllm_runtime,
    _split_align_words,
)


class MockQwen3VLLM:
    def __init__(self, aligned):
        self.aligned = aligned
        self.calls = 0

    def transcribe_aligned(self, audio):
        self.calls += 1
        return self.aligned, "English"


def _audio(seconds):
    return np.zeros(int(seconds * 16_000), dtype=np.float32)


def test_qwen3_vllm_commits_only_before_last_250ms():
    asr = MockQwen3VLLM(
        [
            _AlignedWord("one", 0.0, 1.0),
            _AlignedWord("two", 1.0, 9.70),
            _AlignedWord("three", 9.70, 9.80),
        ]
    )
    processor = Qwen3VLLMOnlineProcessor(asr)
    processor.insert_audio_chunk(_audio(10), 10.0)

    committed, _ = processor.process_iter()

    assert [token.text.strip() for token in committed] == ["one", "two"]
    assert processor.get_buffer().text.strip() == "three"


def test_qwen3_vllm_finish_flushes_last_250ms():
    asr = MockQwen3VLLM(
        [
            _AlignedWord("one", 0.0, 1.0),
            _AlignedWord("two", 1.0, 9.80),
        ]
    )
    processor = Qwen3VLLMOnlineProcessor(asr)
    processor.insert_audio_chunk(_audio(10), 10.0)

    first, _ = processor.process_iter()
    final, _ = processor.finish()

    assert [token.text.strip() for token in first] == ["one"]
    assert [token.text.strip() for token in final] == ["two"]
    assert processor.get_buffer().text == ""


def test_qwen3_vllm_no_duplicate_on_same_buffer():
    asr = MockQwen3VLLM(
        [
            _AlignedWord("one", 0.0, 1.0),
            _AlignedWord("two", 1.0, 2.0),
            _AlignedWord("three", 2.0, 3.0),
        ]
    )
    processor = Qwen3VLLMOnlineProcessor(asr)
    processor.insert_audio_chunk(_audio(4), 4.0)

    first, _ = processor.process_iter()
    second, _ = processor.process_iter(is_last=True)

    assert [token.text.strip() for token in first] == ["one", "two", "three"]
    assert second == []


def test_qwen3_vllm_online_factory_routing():
    args = SimpleNamespace(backend="qwen3-vllm")
    asr = MockQwen3VLLM([])

    processor = online_factory(args, asr)

    assert isinstance(processor, Qwen3VLLMOnlineProcessor)


def test_qwen3_vllm_aligner_helpers():
    assert _split_align_words("Hello, 世界!") == ["Hello", "世", "界"]
    assert _fix_timestamps([0, 3, 2, 4]) == [0.0, 3.0, 3.0, 4.0]


def test_qwen3_vllm_lazy_import_error_is_clear():
    try:
        _load_vllm_runtime()
    except ImportError as exc:
        assert "qwen3-vllm requires vLLM" in str(exc)
