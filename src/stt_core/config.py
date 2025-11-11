"""Configuration helpers for the STT core."""

import os
from pydantic import BaseModel, Field


class STTConfig(BaseModel):
    """Runtime configuration for the WhisperLiveKit STT pipeline."""

    model_backend: str = Field(default=os.getenv("STT_MODEL", "faster-whisper"))
    vad_enabled: bool = True
    vad_sensitivity: float = 0.6
    language: str = "auto"
    sample_rate: int = 16000
    redis_url: str = Field(default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    redis_stream: str = Field(default=os.getenv("REDIS_STREAM", "daymind:transcripts"))
    buffer_path: str = Field(default=os.getenv("BUFFER_PATH", "data/transcripts.jsonl"))
    buffer_max_mb: int = Field(default=int(os.getenv("BUFFER_MAX_MB", "32")))
