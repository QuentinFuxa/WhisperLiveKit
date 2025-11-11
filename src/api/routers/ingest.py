"""Transcript ingestion endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ..deps.auth import get_api_key
from ..schemas import IngestRequest, IngestResponse
from ..services.transcript_service import TranscriptService
from ..settings import APISettings, get_settings

router = APIRouter(prefix="/v1", tags=["ingest"])


def get_service(settings: APISettings = Depends(get_settings)) -> TranscriptService:
    return TranscriptService(settings)


@router.post("/ingest-transcript", response_model=IngestResponse)
async def ingest_transcript(
    payload: IngestRequest,
    _: str = Depends(get_api_key),
    service: TranscriptService = Depends(get_service),
):
    ts = await service.ingest_text(payload.model_dump())
    return IngestResponse(status="ok", stored_at=ts)
