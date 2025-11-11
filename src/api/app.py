"""FastAPI application bootstrap."""

from __future__ import annotations

from fastapi import FastAPI

from .deps.security import install_security_middleware
from .metrics import instrument_app, router as metrics_router
from .routers import finance, health, ingest, ledger, summary, transcribe, usage, welcome
from .settings import get_settings


def create_app() -> FastAPI:
    app = FastAPI(title="Symbioza DayMind API", version="1.0.0")
    instrument_app(app)

    settings = get_settings()
    install_security_middleware(app, settings.ip_rate_limit_per_minute)

    app.include_router(health.router)
    app.include_router(metrics_router)
    app.include_router(welcome.router)
    app.include_router(transcribe.router)
    app.include_router(ingest.router)
    app.include_router(ledger.router)
    app.include_router(summary.router)
    app.include_router(usage.router)
    app.include_router(finance.api_router)
    app.include_router(finance.ui_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)
