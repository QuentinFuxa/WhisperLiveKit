"""Optional fastapi-guard layer: real HTTP requests and WebSocket handshakes through create_app()."""

import asyncio
import dataclasses
import importlib.util
import itertools
import subprocess
import sys
from types import SimpleNamespace

import pytest

from whisperlivekit.audio_processor import AudioProcessor
from whisperlivekit.basic_server import create_app
from whisperlivekit.core import TranscriptionEngine
from whisperlivekit.ffmpeg_manager import FFmpegManager
from whisperlivekit.security_guard import trusted_proxies, uvicorn_proxy_kwargs

requires_guard = pytest.mark.skipif(
    importlib.util.find_spec("guard") is None, reason="fastapi-guard is not installed (whisperlivekit[security])"
)
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_octets = itertools.count(1)


@pytest.fixture
def client_ip():
    """Guard state (bans, budgets) is per client address and process wide: one address per test."""
    n = next(_octets)
    return f"10.{n // 250 + 1}.{n % 250 + 1}.7"


@pytest.fixture
def decoders(monkeypatch):
    """Counts every AudioProcessor (and so every decoder) a test starts."""
    started = SimpleNamespace(processors=0, ffmpeg=0)
    init, start = AudioProcessor.__init__, FFmpegManager.start

    def counting_init(self, *args, **kwargs):
        started.processors += 1
        init(self, *args, **kwargs)

    async def counting_start(self, *args, **kwargs):
        started.ffmpeg += 1
        return await start(self, *args, **kwargs)

    monkeypatch.setattr(AudioProcessor, "__init__", counting_init)
    monkeypatch.setattr(FFmpegManager, "start", counting_start)
    return started


@pytest.fixture
def base_config(monkeypatch):
    # Real queues and ingestion without loading an inference model.
    monkeypatch.delenv("WLK_SECURITY_GUARD", raising=False)
    monkeypatch.setattr(TranscriptionEngine, "_instance", None)
    monkeypatch.setattr(TranscriptionEngine, "_initialized", False)
    engine = TranscriptionEngine(transcription=False, vac=False, pcm_input=True)
    engine.args.transcription = True
    monkeypatch.setattr(
        "whisperlivekit.audio_processor.online_factory",
        lambda *args, **kwargs: SimpleNamespace(asr=SimpleNamespace(sep=" ")),
    )
    return engine.config


@pytest.fixture
def make_client(base_config, client_ip):
    from fastapi.testclient import TestClient

    opened = []

    def make(peer=None, **settings):
        settings.setdefault("security_guard", True)
        app = create_app(dataclasses.replace(base_config, **settings))
        client = TestClient(app, client=(peer or client_ip, 50000))
        opened.append(client.__enter__())
        return client

    yield make
    for client in opened:
        client.__exit__(None, None, None)


def _refused_handshake(client, url, **kwargs):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect(url, **kwargs):
            pass
    return closed.value


def _upload(client, **kwargs):
    kwargs.setdefault("data", {"response_format": "xml"})
    return client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"audio")}, **kwargs)


# --- off by default -------------------------------------------------------------------------


def test_off_adds_nothing_and_never_imports_guard():
    probe = (
        "import sys; sys.argv = ['x']\n"
        "from whisperlivekit.basic_server import create_app\n"
        "from whisperlivekit.config import WhisperLiveKitConfig\n"
        "app = create_app(WhisperLiveKitConfig())\n"
        "assert [m.cls.__name__ for m in app.user_middleware] == ['CORSMiddleware'], app.user_middleware\n"
        "assert not any(name.split('.')[0] in ('guard', 'guard_core') for name in sys.modules)\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr


def test_missing_package_fails_at_startup_with_install_hint(base_config, monkeypatch):
    monkeypatch.setitem(sys.modules, "guard", None)
    with pytest.raises(RuntimeError, match=r"whisperlivekit\[security\]"):
        create_app(dataclasses.replace(base_config, security_guard=True))


def test_environment_variable_enables_the_guard(base_config, monkeypatch):
    pytest.importorskip("guard")
    monkeypatch.setenv("WLK_SECURITY_GUARD", "1")
    app = create_app(base_config)
    assert [m.cls.__name__ for m in app.user_middleware] == ["CORSMiddleware", "SecurityMiddleware"]


# --- HTTP -------------------------------------------------------------------------------------


@requires_guard
def test_cors_is_outermost_so_guard_rejections_carry_cors_headers(make_client, client_ip):
    client = make_client(
        security_guard_rate_limit=2, security_guard_block_ips="192.0.2.99", cors_origins="https://ui.example",
    )
    assert [m.cls.__name__ for m in client.app.user_middleware] == ["CORSMiddleware", "SecurityMiddleware"]
    origin = {"Origin": "https://ui.example"}
    assert [client.get("/health", headers=origin).status_code for _ in range(3)] == [200, 200, 429]
    limited = client.get("/health", headers=origin)
    assert limited.status_code == 429
    assert limited.headers["access-control-allow-origin"] == "https://ui.example"
    blocked = make_client(peer="192.0.2.99", security_guard_block_ips="192.0.2.99", cors_origins="https://ui.example")
    refused = blocked.get("/health", headers=origin)
    assert refused.status_code == 403
    assert refused.headers["access-control-allow-origin"] == "https://ui.example"


@requires_guard
def test_attack_pattern_in_the_query_is_refused_and_repeats_ban_the_client(make_client):
    client = make_client(security_guard_rate_limit=1000)
    assert client.get("/health").status_code == 200
    assert client.get("/health?next=../../../etc/passwd").status_code == 400
    statuses = {client.get("/health?next=../../../etc/passwd").status_code for _ in range(12)}
    assert 403 in statuses
    assert client.get("/health").status_code == 403


@requires_guard
def test_browser_ui_referer_is_allowed_but_a_script_referer_is_not(make_client):
    client = make_client(security_guard_rate_limit=1000)
    for referer in ("http://localhost:3000/", "http://127.0.0.1:8000/", "http://192.168.1.20:8000/", "http://10.0.0.5/"):
        page = client.get("/", headers={"Referer": referer})
        assert page.status_code == 200, referer
        assert "<html" in page.text.lower()
    assert client.get("/", headers={"Referer": "<script>alert(1)</script>"}).status_code == 400


@requires_guard
def test_free_text_context_and_api_tokens_are_not_screened(make_client):
    client = make_client(security_guard_rate_limit=1000)
    context = "select name from users where id = 1 or 1=1 -- Dr. O'Brien's glossary"
    assert client.get("/health", params={"context": context}).status_code == 200
    assert client.get("/health", params={"other": context}).status_code == 400


@requires_guard
def test_upload_route_has_its_own_budget_size_limit_and_content_type_filter(make_client):
    client = make_client(security_guard_upload_rate_limit=2, security_guard_max_upload_mb=1)
    assert [_upload(client).status_code for _ in range(3)] == [400, 400, 429]
    other = make_client(security_guard_max_upload_mb=1)
    big = other.post("/v1/audio/transcriptions", files={"file": ("a.wav", bytes(3 * 1024 * 1024))})
    assert big.status_code == 413
    wrong_type = other.post(
        "/v1/audio/transcriptions", content=b"{}", headers={"Content-Type": "application/json"},
    )
    assert wrong_type.status_code == 415
    assert other.get("/health").status_code == 200  # the other routes are not limited by it


@requires_guard
def test_repeated_401s_ban_the_client_on_http_and_at_the_websocket_handshake(make_client, decoders):
    client = make_client(api_token="secret")
    wrong = {"Authorization": "Bearer wrong"}
    assert [_upload(client, headers=wrong).status_code for _ in range(6)] == [401] * 6
    assert _upload(client, headers={"Authorization": "Bearer secret"}).status_code == 403
    assert client.get("/health").status_code == 403
    _refused_handshake(client, "/asr?token=secret")
    assert decoders.processors == 0
    # Another client is unaffected.
    assert make_client(peer="198.51.100.77", api_token="secret").get("/health").status_code == 200


@requires_guard
def test_allowlist_blocklist_and_user_agent_rules_apply_over_http(make_client, client_ip):
    allowed = make_client(security_guard_allow_ips="10.0.0.0/8")
    assert allowed.get("/health").status_code == 200
    outsider = make_client(peer="203.0.113.5", security_guard_allow_ips="10.0.0.0/8")
    assert outsider.get("/health").status_code == 403
    blocked = make_client(security_guard_block_ips=client_ip)
    assert blocked.get("/health").status_code == 403
    agent = make_client(security_guard_block_user_agents="evilscanner")
    assert agent.get("/health", headers={"User-Agent": "evilscanner/1.0"}).status_code == 403
    assert agent.get("/health", headers={"User-Agent": "curl/8"}).status_code == 200


@requires_guard
def test_emergency_mode_admits_only_its_allowlist(make_client):
    with pytest.raises(ValueError, match="emergency-allow-ips"):
        make_client(security_guard_emergency=True)
    inside = make_client(security_guard_emergency=True, security_guard_emergency_allow_ips="10.0.0.0/8")
    assert inside.get("/health").status_code == 200
    outside = make_client(
        peer="203.0.113.9", security_guard_emergency=True, security_guard_emergency_allow_ips="10.0.0.0/8"
    )
    assert outside.get("/health").status_code == 503
    _refused_handshake(outside, "/asr")


@requires_guard
def test_passive_mode_logs_attacks_without_refusing_them(make_client, decoders):
    client = make_client(security_guard_passive=True, security_guard_block_user_agents="evilscanner")
    assert client.get("/health?next=../../../etc/passwd").status_code == 200
    with client.websocket_connect("/asr?language=../../../etc/passwd", headers={"User-Agent": "evilscanner"}) as ws:
        assert ws.receive_json()["type"] == "config"
    assert decoders.processors == 1


# --- WebSocket handshake ------------------------------------------------------------------------


@requires_guard
@pytest.mark.parametrize("endpoint", ["/asr", "/v1/listen?encoding=linear16&sample_rate=16000&endpointing=false"])
def test_banned_and_blocklisted_clients_are_refused_before_accept_and_start_no_decoder(
    make_client, client_ip, decoders, endpoint
):
    from guard import ip_ban_manager

    asyncio.run(ip_ban_manager.ban_ip(client_ip, 60))
    banned = make_client()
    assert _refused_handshake(banned, endpoint).code == 1008
    blocked = make_client(peer="198.51.100.31", security_guard_block_ips="198.51.100.0/24")
    assert _refused_handshake(blocked, endpoint).code == 1008
    assert (decoders.processors, decoders.ffmpeg) == (0, 0)


@requires_guard
def test_attack_in_the_handshake_query_is_refused_before_any_decoder_starts(make_client, decoders):
    client = make_client(security_guard_rate_limit=1000)
    for url in ("/asr?language=../../../etc/passwd", "/v1/listen?language=<script>alert(1)</script>"):
        assert _refused_handshake(client, url).code == 1008
    assert (decoders.processors, decoders.ffmpeg) == (0, 0)


@requires_guard
def test_user_agent_rules_and_allowlist_apply_at_the_handshake(make_client, decoders):
    agent = make_client(security_guard_block_user_agents="evilscanner")
    _refused_handshake(agent, "/asr", headers={"User-Agent": "evilscanner/1.0"})
    outsider = make_client(peer="203.0.113.5", security_guard_allow_ips="10.0.0.0/8")
    _refused_handshake(outsider, "/asr")
    assert decoders.processors == 0


@requires_guard
def test_new_session_budget_trips_per_client_and_normal_streaming_still_works(make_client, decoders):
    client = make_client(security_guard_rate_limit=3)
    for _ in range(3):
        with client.websocket_connect("/asr?language=en&context=Dr.+O%27Brien%2C+select+1") as ws:
            assert ws.receive_json()["type"] == "config"
            ws.send_bytes(bytes(3200))
    assert _refused_handshake(client, "/asr").code == 1008
    assert decoders.processors == 3
    other = make_client(peer="198.51.100.78", security_guard_rate_limit=3)
    with other.websocket_connect("/asr") as ws:
        assert ws.receive_json()["type"] == "config"


# --- client identity ------------------------------------------------------------------------------


@requires_guard
def test_spoofed_forwarded_for_from_an_untrusted_peer_is_ignored(make_client):
    client = make_client(security_guard_rate_limit=3)
    statuses = [client.get("/health", headers={"X-Forwarded-For": f"203.0.113.{n}"}).status_code for n in range(5)]
    assert statuses == [200, 200, 200, 429, 429]


@requires_guard
def test_forwarded_for_is_honored_from_a_trusted_proxy(make_client, client_ip):
    client = make_client(security_guard_rate_limit=2, forwarded_allow_ips=client_ip)
    first = [client.get("/health", headers={"X-Forwarded-For": "203.0.113.50"}).status_code for _ in range(3)]
    assert first == [200, 200, 429]
    assert client.get("/health", headers={"X-Forwarded-For": "203.0.113.51"}).status_code == 200


def test_guard_trusts_exactly_what_uvicorn_trusts():
    assert trusted_proxies("10.0.0.1, 10.1.0.0/16") == ("10.0.0.1", "10.1.0.0/16")
    assert trusted_proxies("*") == ("0.0.0.0/0", "::/0")
    assert trusted_proxies("/run/wlk.sock") == ("unix",)
    assert trusted_proxies(None) == ()


@requires_guard
@pytest.mark.asyncio
@pytest.mark.parametrize("forwarded", [None, "198.51.100.200"])
async def test_uvicorn_proxy_headers_cannot_change_the_guard_identity(base_config, monkeypatch, forwarded, client_ip):
    import httpx
    import uvicorn

    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")  # what Uvicorn would otherwise trust
    settings = dataclasses.replace(
        base_config, security_guard=True, security_guard_rate_limit=3, forwarded_allow_ips=forwarded,
    )
    server_config = uvicorn.Config(create_app(settings), log_config=None, **uvicorn_proxy_kwargs(settings))
    server_config.load()
    transport = httpx.ASGITransport(app=server_config.loaded_app, client=(client_ip, 50000))
    async with httpx.AsyncClient(transport=transport, base_url="http://wlk.test") as http:
        statuses = [
            (await http.get("/health", headers={"X-Forwarded-For": f"203.0.113.{n}"})).status_code for n in range(5)
        ]
    assert statuses == [200, 200, 200, 429, 429]


@requires_guard
@pytest.mark.asyncio
async def test_proxy_headers_are_off_when_the_guard_is_on_without_trusted_proxies(base_config, monkeypatch):
    import uvicorn
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")
    guarded = dataclasses.replace(base_config, security_guard=True, forwarded_allow_ips=None)
    assert uvicorn_proxy_kwargs(guarded) == {"proxy_headers": False}
    server_config = uvicorn.Config(create_app(guarded), log_config=None, **uvicorn_proxy_kwargs(guarded))
    server_config.load()
    assert not isinstance(server_config.loaded_app, ProxyHeadersMiddleware)
    plain = dataclasses.replace(base_config, security_guard=False, forwarded_allow_ips=None)
    assert uvicorn_proxy_kwargs(plain) == {}


# --- real socket ---------------------------------------------------------------------------------


@requires_guard
@pytest.mark.asyncio
async def test_real_server_answers_a_refused_handshake_with_403_and_starts_no_decoder(base_config, decoders):
    import socket

    import uvicorn
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidStatus

    settings = dataclasses.replace(base_config, security_guard=True, security_guard_block_ips="127.0.0.1")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    server = uvicorn.Server(uvicorn.Config(create_app(settings), log_level="error", lifespan="on", **uvicorn_proxy_kwargs(settings)))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                await asyncio.sleep(0.01)
        with pytest.raises(InvalidStatus) as refused:
            async with connect(f"ws://127.0.0.1:{listener.getsockname()[1]}/asr"):
                pass
        assert refused.value.response.status_code == 403
        assert decoders.processors == 0
    finally:
        server.should_exit = True
        await task
