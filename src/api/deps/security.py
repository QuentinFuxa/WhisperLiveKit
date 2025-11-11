"""Security helpers: middleware for IP throttling and log anonymization."""

from __future__ import annotations

import re
import time
from threading import Lock
from typing import Callable

from fastapi import Request


class _IPRateLimiter:
    def __init__(self, limit_per_minute: int):
        self.limit = max(0, limit_per_minute)
        self._hits: dict[str, tuple[int, int]] = {}
        self._lock = Lock()

    def allow(self, ip: str) -> bool:
        if self.limit == 0:
            return True
        bucket = int(time.time() // 60)
        with self._lock:
            previous_bucket, count = self._hits.get(ip, (bucket, 0))
            if previous_bucket != bucket:
                previous_bucket, count = bucket, 0
            count += 1
            self._hits[ip] = (previous_bucket, count)
            return count <= self.limit


def install_security_middleware(app, limit_per_minute: int) -> None:
    limiter = _IPRateLimiter(limit_per_minute)

    @app.middleware("http")
    async def _ip_guard(request: Request, call_next: Callable):  # type: ignore
        client_ip = request.client.host if request.client else "anonymous"
        if not limiter.allow(client_ip):
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=429, content={"detail": "Too many requests"})
        return await call_next(request)


MASK_PATTERN = re.compile(r"([\d]{4})([\d]{2,})")


def anonymize_text(payload: str, mask: str = "****") -> str:
    """Best-effort masking for personally identifiable number-like strings."""

    return MASK_PATTERN.sub(lambda match: f"{match.group(1)}{mask}", payload)
