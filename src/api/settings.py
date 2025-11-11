"""API settings resolved from the environment."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic import BaseModel, Field


class APISettings(BaseModel):
    app_name: str = Field(default="Symbioza DayMind API")
    version: str = Field(default="1.0.0")
    data_dir: str = Field(default=os.getenv("DATA_DIR", "data"))
    transcript_path: str = Field(
        default=os.getenv("TRANSCRIPT_PATH", "data/transcripts.jsonl")
    )
    ledger_path: str = Field(default=os.getenv("LEDGER_PATH", "data/ledger.jsonl"))
    api_keys: List[str] = Field(default_factory=lambda: _split_keys())
    redis_url: str | None = Field(default=os.getenv("REDIS_URL"))
    redis_stream: str = Field(default=os.getenv("REDIS_STREAM", "daymind:transcripts"))
    summary_dir: str = Field(default=os.getenv("SUMMARY_DIR", "data"))
    session_gap_sec: float = Field(float(os.getenv("SESSION_GAP_SEC", "45")))
    finance_ledger_path: str = Field(default=os.getenv("FINANCE_LEDGER_PATH", "finance/ledger.beancount"))
    finance_default_currency: str = Field(default=os.getenv("FINANCE_DEFAULT_CURRENCY", "CZK"))
    fava_host: str = Field(default=os.getenv("FAVA_HOST", "127.0.0.1"))
    fava_port: int = Field(default=int(os.getenv("FAVA_PORT", "5000")))
    fava_base_url: str | None = Field(default=os.getenv("FAVA_BASE_URL"))


def _split_keys() -> List[str]:
    raw = os.getenv("API_KEYS") or os.getenv("API_KEY") or ""
    return [key.strip() for key in raw.split(",") if key.strip()]


@lru_cache()
def get_settings() -> APISettings:
    return APISettings()
