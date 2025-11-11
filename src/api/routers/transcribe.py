"""Transcription endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from ..deps.auth import get_api_key
from ..schemas import TranscribeResponse
from ..services.transcript_service import TranscriptService
from ..settings import APISettings, get_settings

router = APIRouter(prefix="/v1", tags=["transcribe"])


def get_service(settings: APISettings = Depends(get_settings)) -> TranscriptService:
    return TranscriptService(settings)


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    lang: str | None = None,
    _: str = Depends(get_api_key),
    service: TranscriptService = Depends(get_service),
):
    record = await service.save_audio(file, lang)
    return TranscribeResponse(
        text=record["text"],
        lang=record.get("lang", "auto"),
        start=record.get("start", 0.0),
        end=record.get("end", 0.0),
        confidence=record.get("confidence"),
        session_id=record.get("session_id"),
    )
