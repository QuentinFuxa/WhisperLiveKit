"""Finance endpoints for Beancount summaries and Fava redirect."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse

from ..deps.auth import get_api_key
from ..schemas import FinanceSummaryItem, FinanceSummaryResponse
from ..services.finance import summarize_ledger
from ..settings import APISettings, get_settings

api_router = APIRouter(prefix="/v1", tags=["finance"])
ui_router = APIRouter(tags=["finance"])


@api_router.get("/finance", response_model=FinanceSummaryResponse)
async def get_finance_summary(
    date: str | None = None,
    _: str = Depends(get_api_key),
    settings: APISettings = Depends(get_settings),
):
    ledger_path = Path(settings.finance_ledger_path)
    try:
        rows = summarize_ledger(ledger_path, date=date)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Finance ledger not ready") from None

    items = [
        FinanceSummaryItem(
            date=row.date,
            category=row.category,
            total=float(row.amount),
            currency=row.currency or settings.finance_default_currency,
        )
        for row in rows
    ]
    return FinanceSummaryResponse(count=len(items), items=items)


@ui_router.get("/finance")
async def redirect_to_fava(
    _: str = Depends(get_api_key),
    settings: APISettings = Depends(get_settings),
):
    target = settings.fava_base_url or f"http://{settings.fava_host}:{settings.fava_port}"
    return RedirectResponse(target)
