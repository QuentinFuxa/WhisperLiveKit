from src.stt_core.config import STTConfig


def test_config_defaults() -> None:
    cfg = STTConfig()
    assert cfg.model_backend == "faster-whisper"
    assert cfg.vad_enabled is True
    assert 0 <= cfg.vad_sensitivity <= 1
    assert cfg.redis_url.startswith("redis://")
    assert cfg.redis_stream == "daymind:transcripts"
    assert cfg.buffer_path.endswith("transcripts.jsonl")
    assert cfg.buffer_max_mb == 32
