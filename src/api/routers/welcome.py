"""Public onboarding endpoint returning documentation links."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(tags=["welcome"])


@router.get("/welcome")
async def welcome():
    return {
        "message": "DayMind server is running. Create an API key to continue.",
        "docs": {
            "api": "./API_REFERENCE.md",
            "security": "./SECURITY.md",
            "billing": "./BILLING.md",
            "onboarding": "./ONBOARDING.md",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
