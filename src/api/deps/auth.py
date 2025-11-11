"""API key authentication dependency with usage tracking and rate limiting."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import Depends, Header, HTTPException, Request, status

from ..services.auth_service import APIKeyRecord, AuthService, RateLimitError, build_auth_service
from ..settings import APISettings, get_settings


@lru_cache()
def _get_service(
    store_path: str,
    redis_url: str | None,
    api_keys: tuple[str, ...],
    rate_limit: int,
) -> AuthService:
    return build_auth_service(Path(store_path), api_keys, redis_url, rate_limit)


def reset_auth_service_cache() -> None:
    _get_service.cache_clear()


async def get_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: APISettings = Depends(get_settings),
) -> str:
    store_path = settings.api_key_store_path
    if not settings.api_keys and not Path(store_path).exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API keys not configured",
        )

    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")

    service = _get_service(
        store_path,
        settings.redis_url,
        tuple(settings.api_keys),
        settings.api_rate_limit_per_minute,
    )
    try:
        record = await service.validate_and_track(x_api_key)
    except RateLimitError:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded") from None
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key") from None

    request.state.api_key_metadata = record
    return x_api_key


def get_api_key_metadata(request: Request) -> APIKeyRecord:
    metadata = getattr(request.state, "api_key_metadata", None)
    if metadata is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Auth metadata missing")
    return metadata
