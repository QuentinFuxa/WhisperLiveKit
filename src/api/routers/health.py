"""Health and diagnostics endpoints."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends

from redis.asyncio import from_url

from ..deps.auth import get_api_key
from ..schemas import HealthResponse
from ..settings import APISettings, get_settings

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz(
    _: str = Depends(get_api_key),
    settings: APISettings = Depends(get_settings),
) -> HealthResponse:
    disk_state = _check_disk(Path(settings.data_dir))
    redis_state = await _check_redis(settings.redis_url)
    return HealthResponse(
        ok=True,
        redis=redis_state,
        disk=disk_state,
        timestamp=datetime.now(timezone.utc),
    )


async def _check_redis(url: str | None) -> str:
    if not url:
        return "skip"
    try:
        client = from_url(url)
        await asyncio.wait_for(client.ping(), timeout=1)
        await client.close()
        return "ok"
    except Exception:
        return "error"


def _check_disk(path: Path) -> str:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".health"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return "ok"
    except Exception:
        return "error"
