"""Usage endpoint reporting API key statistics."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps.auth import get_api_key, get_api_key_metadata
from ..schemas import UsageResponse

router = APIRouter(prefix="/v1", tags=["usage"])


@router.get("/usage", response_model=UsageResponse)
async def usage_endpoint(
    _: str = Depends(get_api_key),
    metadata = Depends(get_api_key_metadata),
) -> UsageResponse:
    return UsageResponse(
        owner=metadata.owner,
        created_at=metadata.created_at,
        usage_count=metadata.usage_count,
        requests_today=metadata.requests_today,
        last_used=metadata.last_used,
    )
