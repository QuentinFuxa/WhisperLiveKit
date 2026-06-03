import pytest

from qwen3_streaming.realtime_config import RealtimeAudioConfig


def test_qwen_audio_default_context_is_pragmatic_streaming_v1_candidate():
    config = RealtimeAudioConfig()

    assert config.qwen_audio_left_context_sec == 2.0
    assert config.qwen_audio_left_context_frames == 200
    assert config.qwen_audio_right_context_ms == 640
    assert config.qwen_audio_right_context_frames == 64


def test_qwen_audio_context_frame_conversions_use_mel_hop():
    config = RealtimeAudioConfig(
        qwen_audio_left_context_sec=2.5,
        qwen_audio_right_context_ms=640,
        mel_hop_ms=10,
    )

    assert config.qwen_audio_left_context_frames == 250
    assert config.qwen_audio_right_context_frames == 64


def test_qwen_audio_strict_causal_removes_right_context():
    config = RealtimeAudioConfig(
        qwen_audio_right_context_ms=640,
        qwen_audio_strict_causal=True,
    )

    assert config.qwen_audio_right_context_frames == 0


def test_qwen_audio_context_validation_rejects_invalid_values():
    with pytest.raises(ValueError, match="right_context"):
        RealtimeAudioConfig(qwen_audio_right_context_ms=-1)
    with pytest.raises(ValueError, match="left_context"):
        RealtimeAudioConfig(qwen_audio_left_context_sec=0.0)
