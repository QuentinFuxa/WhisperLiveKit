"""API key authentication dependency."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from ..settings import APISettings, get_settings


def get_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: APISettings = Depends(get_settings),
) -> str:
    if not settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API keys not configured",
        )

    if not x_api_key or x_api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return x_api_key
