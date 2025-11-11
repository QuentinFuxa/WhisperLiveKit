"""Pydantic schemas for API contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TranscribeResponse(BaseModel):
    text: str
    lang: str = Field(default="auto")
    start: float = 0.0
    end: float = 0.0
    confidence: float | None = None
    session_id: int | None = None


class IngestRequest(BaseModel):
    text: str
    start: float = 0.0
    end: float = 0.0
    lang: str = "auto"
    meta: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    status: str = "ok"
    stored_at: float


class LedgerEntry(BaseModel):
    session_id: int | None = None
    input: str
    start: float | None = None
    end: float | None = None
    gpt_output: Any | None = None
    gap: float | None = None
    ts: float | None = None


class LedgerResponse(BaseModel):
    date: str
    count: int
    entries: list[LedgerEntry]


class SummaryResponse(BaseModel):
    date: str
    summary_md: str


class HealthResponse(BaseModel):
    ok: bool
    redis: str
    disk: str
    openai: str
    tls: str
    timestamp: datetime


class FinanceSummaryItem(BaseModel):
    date: str
    category: str
    total: float
    currency: str


class FinanceSummaryResponse(BaseModel):
    count: int
    items: list[FinanceSummaryItem]


class UsageResponse(BaseModel):
    owner: str
    created_at: float
    usage_count: int
    requests_today: int
    last_used: float | None = None
