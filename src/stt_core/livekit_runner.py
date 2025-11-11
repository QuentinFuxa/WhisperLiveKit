"""Simple entrypoint for the WhisperLiveKit streaming loop."""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from redis.exceptions import RedisError

try:  # pragma: no cover - optional dependency during tests
    from whisper_livekit import LiveKit  # type: ignore
except ImportError:  # pragma: no cover
    LiveKit = None  # patched in tests

from .buffer_store import BufferStore
from .config import STTConfig
from .redis_io import RedisPublisher


async def run_realtime_stt() -> None:
    """Boot the LiveKit engine, stream mic audio, and log transcripts."""

    cfg = STTConfig()
    buffer_store = BufferStore(cfg.buffer_path, cfg.buffer_max_mb)
    redis_publisher = RedisPublisher(cfg.redis_url, cfg.redis_stream)

    print(
        f"[DayMind] Starting STT with backend={cfg.model_backend}, "
        f"VAD={cfg.vad_enabled}"
    )

    if LiveKit is None:
        raise RuntimeError(
            "whisper_livekit is not installed. Install the optional 'stt_livekit' "
            "extra or vendor LiveKit before running the realtime STT loop."
        )

    live = LiveKit(
        backend=cfg.model_backend,
        vad=cfg.vad_enabled,
        vad_sensitivity=cfg.vad_sensitivity,
        language=cfg.language,
    )

    async for segment in live.listen():
        payload = _segment_payload(segment, cfg)
        await _publish_with_retry(redis_publisher, payload)
        buffer_store.append(payload)
        print(f"[Sink][Buffer] appended {cfg.buffer_path}")
        print(f"[Transcript] {payload['text']}")


def _segment_payload(segment: Any, cfg: STTConfig) -> Dict[str, Any]:
    """Normalize LiveKit segment objects into dict payloads."""

    return {
        "text": getattr(segment, "text", ""),
        "lang": getattr(segment, "lang", None) or cfg.language,
        "start": getattr(segment, "start", None),
        "end": getattr(segment, "end", None),
        "confidence": getattr(segment, "confidence", None),
    }


async def _publish_with_retry(
    publisher: RedisPublisher, payload: Dict[str, Any], attempts: int = 3
) -> None:
    """Attempt to publish with exponential backoff on failures."""

    for attempt in range(1, attempts + 1):
        try:
            entry_id = await publisher.publish(payload)
            print(f"[Sink][Redis] published id={entry_id}")
            return
        except RedisError as exc:
            delay = min(2 ** (attempt - 1) * 0.5, 5)
            print(
                f"[Sink][Redis] publish failed (attempt {attempt}/{attempts}): {exc}"
            )
            if attempt == attempts:
                print("[Sink][Redis] giving up for this segment; continuing")
                return
            await asyncio.sleep(delay)


if __name__ == "__main__":
    asyncio.run(run_realtime_stt())
