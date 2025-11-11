"""Configuration for GPT post-processing."""

from __future__ import annotations

import os
from pydantic import BaseModel


class GPTConfig(BaseModel):
    """Environment-driven configuration for GPT-ledger processing."""

    model: str = os.getenv("GPT_MODEL", "gpt-4o-mini")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    input_path: str = os.getenv("TRANSCRIPT_PATH", "data/transcripts.jsonl")
    ledger_path: str = os.getenv("LEDGER_PATH", "data/ledger.jsonl")
    temperature: float = float(os.getenv("GPT_TEMP", "0.2"))
    session_gap_sec: float = float(os.getenv("SESSION_GAP_SEC", "45"))
