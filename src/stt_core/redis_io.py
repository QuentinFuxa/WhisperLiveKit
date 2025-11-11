"""Async Redis publisher for transcript payloads."""

from __future__ import annotations

from typing import Any, Dict

from redis.asyncio import Redis, from_url


class RedisPublisher:
    """Publish transcript payloads to a Redis stream."""

    def __init__(self, url: str, stream: str) -> None:
        self.url = url
        self.stream = stream
        self._client: Redis = from_url(url, decode_responses=True)

    async def publish(self, payload: Dict[str, Any]) -> str:
        """Write payload to the configured Redis stream via XADD."""

        return await self._client.xadd(
            self.stream,
            payload,
            maxlen=10000,
            approximate=True,
        )

