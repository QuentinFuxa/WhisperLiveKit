import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.deps.auth import reset_auth_service_cache


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    transcripts = tmp_path / "transcripts.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    summary_dir = tmp_path

    from src.api.settings import APISettings, get_settings
    from src.api.app import create_app

    get_settings.cache_clear()  # type: ignore
    reset_auth_service_cache()
    settings = APISettings(
        api_keys=["test-key"],
        api_key_store_path=str(tmp_path / "keys.json"),
        transcript_path=str(transcripts),
        ledger_path=str(ledger),
        summary_dir=str(summary_dir),
        data_dir=str(tmp_path),
        api_rate_limit_per_minute=120,
        ip_rate_limit_per_minute=0,
    )

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings

    client = TestClient(app)
    return client, transcripts, ledger, summary_dir


def _auth_headers():
    return {"X-API-Key": "test-key"}


def test_auth_required(api_client):
    client, *_ = api_client
    resp = client.get("/healthz")
    assert resp.status_code == 401


def test_health_ok(api_client):
    client, *_ = api_client
    resp = client.get("/healthz", headers=_auth_headers())
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["openai"] == "skip"
    assert "tls" in body


def test_ingest_transcript_appends(api_client):
    client, transcripts, *_ = api_client
    payload = {"text": "hello", "start": 0.0, "end": 1.0}
    resp = client.post("/v1/ingest-transcript", json=payload, headers=_auth_headers())
    assert resp.status_code == 200
    assert transcripts.exists()
    line = transcripts.read_text().strip()
    data = json.loads(line)
    assert data["text"] == "hello"


def test_ledger_endpoint(api_client):
    client, _, ledger, _ = api_client
    ts = datetime(2024, 11, 1).timestamp()
    entries = [
        {"input": "a", "start": ts, "session_id": 1},
        {"input": "b", "start": ts + 5, "session_id": 1},
    ]
    with open(ledger, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    resp = client.get("/v1/ledger?date=2024-11-01", headers=_auth_headers())
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2


def test_summary_returns_file(api_client):
    client, _, _, summary_dir = api_client
    target = summary_dir / "summary_2024-11-01.md"
    target.write_text("## summary", encoding="utf-8")
    resp = client.get("/v1/summary?date=2024-11-01", headers=_auth_headers())
    assert resp.status_code == 200
    assert "summary" in resp.json()["summary_md"]


def test_transcribe_endpoint(api_client):
    client, transcripts, *_ = api_client
    audio_path = Path("tests/assets/sample_cs.wav")
    resp = client.post(
        "/v1/transcribe",
        headers=_auth_headers(),
        files={"file": (audio_path.name, audio_path.read_bytes(), "audio/wav")},
    )
    assert resp.status_code == 200
    assert "text" in resp.json()
    assert transcripts.exists()


def test_metrics_secured(api_client):
    client, *_ = api_client
    resp = client.get("/metrics", headers=_auth_headers())
    assert resp.status_code == 200
    assert "api_requests_total" in resp.text


def test_usage_endpoint(api_client):
    client, *_ = api_client
    resp = client.get("/v1/usage", headers=_auth_headers())
    assert resp.status_code == 200
    body = resp.json()
    assert body["usage_count"] >= 1


def test_welcome_public(api_client):
    client, *_ = api_client
    resp = client.get("/welcome")
    assert resp.status_code == 200
    assert "docs" in resp.json()


def test_rate_limit_enforced(tmp_path):
    transcripts = tmp_path / "transcripts.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    summary_dir = tmp_path

    from src.api.settings import APISettings, get_settings
    from src.api.app import create_app

    get_settings.cache_clear()  # type: ignore
    reset_auth_service_cache()
    settings = APISettings(
        api_keys=["test-key"],
        api_key_store_path=str(tmp_path / "keys.json"),
        transcript_path=str(transcripts),
        ledger_path=str(ledger),
        summary_dir=str(summary_dir),
        data_dir=str(tmp_path),
        api_rate_limit_per_minute=1,
        ip_rate_limit_per_minute=0,
    )

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}

    assert client.get("/healthz", headers=headers).status_code == 200
    resp = client.get("/healthz", headers=headers)
    assert resp.status_code == 429
