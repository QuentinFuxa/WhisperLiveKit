from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _write_beancount(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """option "title" "Test Ledger"
option "operating_currency" "CZK"
1970-01-01 open Assets:Cash:DayMind
1970-01-01 open Expenses:Food
1970-01-01 open Income:Salary

2024-11-10 * "Bistro" "Lunch"
  Expenses:Food         120 CZK
  Assets:Cash:DayMind

2024-11-11 * "Acme" "Salary"
  Assets:Cash:DayMind   5000 CZK
  Income:Salary
""",
        encoding="utf-8",
    )


@pytest.fixture()
def finance_client(tmp_path):
    ledger = tmp_path / "finance" / "ledger.beancount"
    _write_beancount(ledger)

    from src.api.app import create_app
    from src.api.settings import APISettings, get_settings

    get_settings.cache_clear()  # type: ignore
    settings = APISettings(
        api_keys=["k"],
        ledger_path=str(tmp_path / "data" / "ledger.jsonl"),
        transcript_path=str(tmp_path / "data" / "transcripts.jsonl"),
        summary_dir=str(tmp_path / "data"),
        finance_ledger_path=str(ledger),
        fava_base_url="http://example.com:5000",
    )
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)
    return client


def _auth():
    return {"X-API-Key": "k"}


def test_finance_summary(finance_client):
    resp = finance_client.get("/v1/finance", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    items = {item["category"]: item for item in body["items"]}
    assert items["Expenses:Food"]["total"] == 120.0
    assert items["Income:Salary"]["total"] == 5000.0


def test_finance_redirect(finance_client):
    resp = finance_client.get("/finance", headers=_auth(), follow_redirects=False)
    assert resp.status_code in (301, 302, 307)
    assert resp.headers["location"].startswith("http://example.com:5000")
