"""Summary endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from ..deps.auth import get_api_key
from ..schemas import SummaryResponse
from ..settings import APISettings, get_settings
from src.gpt_postproc.ledger_store import LedgerStore
from src.gpt_postproc.daily_summary import summarize_day
from src.gpt_postproc.config import GPTConfig

router = APIRouter(prefix="/v1", tags=["summary"])


@router.get("/summary", response_model=SummaryResponse)
async def get_summary(
    date: str,
    force: bool = False,
    _: str = Depends(get_api_key),
    settings: APISettings = Depends(get_settings),
):
    summary_path = Path(settings.summary_dir) / f"summary_{date}.md"
    if summary_path.exists() and not force:
        content = summary_path.read_text(encoding="utf-8")
        return SummaryResponse(date=date, summary_md=content)

    store = LedgerStore(settings.ledger_path)
    entries = store.group_by_day().get(date)
    if not entries:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No ledger entries")

    cfg = GPTConfig()
    await summarize_day(date, entries, cfg=cfg, output_dir=Path(settings.summary_dir))
    if not summary_path.exists():
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Summary generation failed")
    return SummaryResponse(date=date, summary_md=summary_path.read_text(encoding="utf-8"))
