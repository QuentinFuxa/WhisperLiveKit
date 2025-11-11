"""Prometheus metrics helpers."""

from __future__ import annotations

import time
from typing import Callable

from fastapi import APIRouter, Depends, Response
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest

from .deps.auth import get_api_key

REQUEST_COUNTER = Counter(
    "api_requests_total",
    "Total API requests",
    labelnames=("path", "method", "status"),
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    labelnames=("path", "method"),
)

router = APIRouter()


@router.get("/metrics")
async def metrics_endpoint(_: str = Depends(get_api_key)) -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


def instrument_app(app):
    @app.middleware("http")
    async def prometheus_middleware(request, call_next: Callable):  # type: ignore
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        method = request.method
        REQUEST_COUNTER.labels(path=path, method=method, status=response.status_code).inc()
        REQUEST_LATENCY.labels(path=path, method=method).observe(duration)
        return response

    return app
