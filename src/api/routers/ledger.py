"""Ledger retrieval endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..deps.auth import get_api_key
from ..schemas import LedgerEntry, LedgerResponse
from ..settings import APISettings, get_settings

router = APIRouter(prefix="/v1", tags=["ledger"])


@router.get("/ledger", response_model=LedgerResponse)
async def get_ledger(
    date: str,
    limit: int = Query(default=100, le=1000),
    offset: int = 0,
    _: str = Depends(get_api_key),
    settings: APISettings = Depends(get_settings),
):
    entries = _load_entries(date, settings)
    if entries is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No data for date")

    sliced = entries[offset : offset + limit]
    return LedgerResponse(date=date, count=len(entries), entries=[LedgerEntry(**e) for e in sliced])


def _load_entries(date: str, settings: APISettings):
    candidates = [
        Path(settings.summary_dir) / f"ledger_{date}.jsonl",
        Path(settings.ledger_path),
    ]
    for path in candidates:
        if not path.exists():
            continue
        entries = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if path.name.startswith("ledger_"):
                    entries.append(data)
                else:
                    if _matches_date(data, date):
                        entries.append(data)
        if entries:
            return entries
    return None


def _matches_date(entry, date: str) -> bool:
    from datetime import datetime

    ts = entry.get("start") or entry.get("ts")
    if ts is None:
        return False
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d") == date
    except Exception:
        return False
