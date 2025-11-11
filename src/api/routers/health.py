"""Health and diagnostics endpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends
from openai import AsyncOpenAI
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
    openai_state = await _check_openai(settings.openai_api_key, settings.openai_health_model)
    tls_state = _check_tls(settings)
    ok = _all_green(disk_state, redis_state, openai_state, tls_state)
    return HealthResponse(
        ok=ok,
        redis=redis_state,
        disk=disk_state,
        openai=openai_state,
        tls=tls_state,
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


async def _check_openai(api_key: str | None, model_name: str | None) -> str:
    if not api_key:
        return "skip"
    try:
        client = AsyncOpenAI(api_key=api_key)
        await asyncio.wait_for(client.models.retrieve(model_name or "gpt-4o-mini"), timeout=2)
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


def _check_tls(settings: APISettings) -> str:
    if not settings.tls_required:
        return "skip"
    if settings.tls_proxy_host:
        return "ok"
    return "error"


def _all_green(disk: str, redis: str, openai: str, tls: str) -> bool:
    failures = []
    if disk != "ok":
        failures.append("disk")
    if redis not in {"ok", "skip"}:
        failures.append("redis")
    if openai not in {"ok", "skip"}:
        failures.append("openai")
    if tls not in {"ok", "skip"}:
        failures.append("tls")
    return len(failures) == 0
