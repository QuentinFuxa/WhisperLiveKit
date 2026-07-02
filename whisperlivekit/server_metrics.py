"""Process-local metrics and admission tracking for the HTTP server."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional


class ServerMetrics:
    def __init__(self, max_concurrent_transcriptions: Optional[int] = None):
        self.max_concurrent_transcriptions = (
            max_concurrent_transcriptions
            if max_concurrent_transcriptions and max_concurrent_transcriptions > 0
            else None
        )
        self.waiting_requests = 0
        self.active_requests = 0
        self._condition = asyncio.Condition()

    @asynccontextmanager
    async def transcription_admission(self) -> AsyncIterator[None]:
        if self.max_concurrent_transcriptions is None:
            yield
            return

        async with self._condition:
            if self.active_requests >= self.max_concurrent_transcriptions:
                self.waiting_requests += 1
                try:
                    await self._condition.wait_for(
                        lambda: self.active_requests < self.max_concurrent_transcriptions
                    )
                finally:
                    self.waiting_requests -= 1
            self.active_requests += 1

        try:
            yield
        finally:
            async with self._condition:
                self.active_requests -= 1
                self._condition.notify()

    async def snapshot(self) -> dict[str, int]:
        async with self._condition:
            return {
                "waiting_requests": self.waiting_requests,
                "active_requests": self.active_requests,
            }

    async def render_openmetrics(self) -> str:
        metrics = await self.snapshot()
        return (
            "# TYPE vllm:num_requests_waiting gauge\n"
            f"vllm:num_requests_waiting {metrics['waiting_requests']}\n"
            "# EOF\n"
        )
