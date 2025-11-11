import httpx
from pathlib import Path

from mobile.daymind.services.network import ApiClient, ApiError
from mobile.daymind.store.settings_store import SettingsStore


def make_client(tmp_path, transport):
    settings = SettingsStore(tmp_path / "settings.json")
    settings.update(server_url="https://api.example.com", api_key="k")
    client = ApiClient(settings, client=httpx.Client(transport=transport, base_url="https://api.example.com"))
    return client


def test_upload_chunk_success(tmp_path):
    def handler(request):
        if request.method == "POST" and request.url.path.endswith("/v1/transcribe"):
            return httpx.Response(200, json={"text": "ok"})
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        raise AssertionError("Unexpected request")

    transport = httpx.MockTransport(handler)
    client = make_client(tmp_path, transport)
    audio = tmp_path / "chunk.wav"
    audio.write_bytes(b"123")
    client.upload_chunk(str(audio))
    assert client.test_connection() is True


def test_summary_error(tmp_path):
    def handler(request):
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = make_client(tmp_path, transport)
    try:
        client.fetch_summary("2024-11-01")
    except ApiError as exc:
        assert "Summary" in str(exc)
    else:
        raise AssertionError("Expected ApiError")
