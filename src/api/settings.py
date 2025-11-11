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
    api_key_store_path: str = Field(
        default=os.getenv("API_KEY_STORE_PATH", "data/api_keys.json")
    )
    api_rate_limit_per_minute: int = Field(
        default=int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "120"))
    )
    ip_rate_limit_per_minute: int = Field(
        default=int(os.getenv("IP_RATE_LIMIT_PER_MINUTE", "240"))
    )
    redis_url: str | None = Field(default=os.getenv("REDIS_URL"))
    redis_stream: str = Field(default=os.getenv("REDIS_STREAM", "daymind:transcripts"))
    summary_dir: str = Field(default=os.getenv("SUMMARY_DIR", "data"))
    session_gap_sec: float = Field(float(os.getenv("SESSION_GAP_SEC", "45")))
    finance_ledger_path: str = Field(default=os.getenv("FINANCE_LEDGER_PATH", "finance/ledger.beancount"))
    finance_default_currency: str = Field(default=os.getenv("FINANCE_DEFAULT_CURRENCY", "CZK"))
    fava_host: str = Field(default=os.getenv("FAVA_HOST", "127.0.0.1"))
    fava_port: int = Field(default=int(os.getenv("FAVA_PORT", "5000")))
    fava_base_url: str | None = Field(default=os.getenv("FAVA_BASE_URL"))
    openai_api_key: str | None = Field(default=os.getenv("OPENAI_API_KEY"))
    openai_health_model: str | None = Field(default=os.getenv("OPENAI_HEALTH_MODEL", "gpt-4o-mini"))
    billing_mode: str = Field(default=os.getenv("BILLING_MODE", "local"))
    stripe_secret_key: str | None = Field(default=os.getenv("STRIPE_SECRET_KEY"))
    tls_required: bool = Field(
        default=os.getenv("TLS_REQUIRED", "false").lower() in {"1", "true", "yes"}
    )
    tls_proxy_host: str | None = Field(default=os.getenv("TLS_PROXY_HOST"))


def _split_keys() -> List[str]:
    raw = os.getenv("API_KEYS") or os.getenv("API_KEY") or ""
    return [key.strip() for key in raw.split(",") if key.strip()]


@lru_cache()
def get_settings() -> APISettings:
    return APISettings()
