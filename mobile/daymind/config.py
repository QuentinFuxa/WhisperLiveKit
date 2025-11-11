"""Shared configuration for the DayMind mobile client."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AppConfig:
    chunk_seconds: int = 6
    sample_rate: int = 16000
    channels: int = 1
    queue_file: str = "chunk_queue.json"
    settings_file: str = "settings.json"
    log_history: int = 200


CONFIG = AppConfig()

