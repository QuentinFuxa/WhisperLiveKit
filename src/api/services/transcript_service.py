"""Helpers for ingesting transcripts and audio files."""

from __future__ import annotations

import asyncio
import io
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import UploadFile

from src.stt_core.buffer_store import BufferStore
from src.stt_core.redis_io import RedisPublisher

from ..settings import APISettings


class TranscriptService:
    def __init__(self, settings: APISettings) -> None:
        self.settings = settings
        self.buffer = BufferStore(settings.transcript_path)
        self._redis: Optional[RedisPublisher] = None
        if settings.redis_url:
            self._redis = RedisPublisher(settings.redis_url, settings.redis_stream)

    async def save_audio(self, file: UploadFile, lang: str | None = None) -> Dict[str, Any]:
        data = await file.read()
        lang = lang or "auto"
        tmp_dir = Path(self.settings.data_dir) / "uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{int(time.time()*1000)}_{file.filename}"
        tmp_path.write_bytes(data)

        # Placeholder transcription until full STT integration is wired in.
        text = f"audio-bytes:{len(data)} from {file.filename}"
        entry = {
            "text": text,
            "lang": lang,
            "start": 0.0,
            "end": 0.0,
            "ts": time.time(),
            "source": str(tmp_path),
        }
        self.buffer.append(entry)
        await self._publish(entry)
        return entry

    async def ingest_text(self, payload: Dict[str, Any]) -> float:
        payload.setdefault("ts", time.time())
        self.buffer.append(payload)
        await self._publish(payload)
        return payload["ts"]

    async def _publish(self, payload: Dict[str, Any]) -> None:
        if not self._redis:
            return
        try:
            await self._redis.publish(payload)
        except Exception:
            pass
