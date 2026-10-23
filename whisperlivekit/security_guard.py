"""Optional fastapi-guard security layer. Nothing here imports `guard` unless it is enabled."""
import logging
import os
from ipaddress import ip_address, ip_network
from typing import Callable

from fastapi import Depends, WebSocket
from starlette.exceptions import WebSocketException

from whisperlivekit.config import WhisperLiveKitConfig

logger = logging.getLogger(__name__)

INSTALL_HINT = "pip install 'whisperlivekit[security]'"
# More than five 401s from one client within five minutes ban it (HTTP API only;
# WebSocket token failures close with 4401 after the handshake and are not counted).
UNAUTHORIZED_BAN_THRESHOLD = 5
UNAUTHORIZED_BAN_WINDOW_SECONDS = 300
# Multipart framing around the audio itself.
UPLOAD_OVERHEAD_BYTES = 1024 * 1024
# The browser UI is served at "/" and legitimately sends a localhost or LAN Referer.
DISABLED_DETECTION_CATEGORIES = frozenset({"ssrf"})
# Free text; it trips the SQL injection patterns.
FREE_TEXT_QUERY_PARAMS = frozenset({"context"})


def guard_enabled(config: WhisperLiveKitConfig) -> bool:
    return bool(config.security_guard) or os.environ.get("WLK_SECURITY_GUARD", "").strip().lower() in {"1", "true", "yes", "on"}


def parse_csv(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in (value or "").split(",") if item.strip())


def trusted_proxies(forwarded_allow_ips: str | None) -> tuple[str, ...]:
    """The guard trusts exactly the proxies Uvicorn is told to trust."""
    entries = []
    for entry in parse_csv(forwarded_allow_ips):
        if entry == "*":
            logger.warning(
                "forwarded_allow_ips '*' trusts X-Forwarded-For from every peer, so any client can choose "
                "the address the security guard rate limits and bans; list your proxy addresses instead"
            )
            entries += ["0.0.0.0/0", "::/0"]
        elif entry.startswith("/"):
            entries.append("unix")  # a socket path, the guard's name for a Unix socket peer
        else:
            entries.append(entry)
    return tuple(dict.fromkeys(entries))


def uvicorn_proxy_kwargs(config: WhisperLiveKitConfig) -> dict:
    """Uvicorn rewrites the client address from X-Forwarded-For before any middleware runs."""
    if config.forwarded_allow_ips:
        return {"forwarded_allow_ips": config.forwarded_allow_ips}
    # FORWARDED_ALLOW_IPS in the environment would otherwise still apply.
    return {"proxy_headers": False} if guard_enabled(config) else {}


def _import_guard():
    try:
        import guard
        from guard.lifespan import make_lifespan
        from guard_core.handlers.suspatterns_handler import ALL_DETECTION_CATEGORIES
    except ImportError as exc:
        raise RuntimeError(
            f"The security guard is enabled (--security-guard or WLK_SECURITY_GUARD) but fastapi-guard is not installed. Install it with: {INSTALL_HINT}"
        ) from exc
    return guard, make_lifespan, ALL_DETECTION_CATEGORIES


def build_security_config(settings: WhisperLiveKitConfig):
    guard, _, all_categories = _import_guard()
    for name in ("security_guard_rate_limit", "security_guard_rate_window", "security_guard_upload_rate_limit", "security_guard_max_upload_mb"):
        if getattr(settings, name) < 1:
            raise ValueError(f"{name} must be at least 1.")
    emergency_ips = parse_csv(settings.security_guard_emergency_allow_ips)
    if settings.security_guard_emergency and not emergency_ips:
        raise ValueError("--security-guard-emergency needs --security-guard-emergency-allow-ips, otherwise it would lock everyone out.")
    return guard.SecurityConfig(
        # State is in memory: nothing else has to be deployed.
        enable_redis=False,
        enable_agent=False,
        # The app's own CORSMiddleware stays in charge, and no headers are added.
        enable_cors=False,
        security_headers={"enabled": False},
        # Audio and free text are never scanned.
        detection_scan_body=False,
        enabled_detection_categories=frozenset(all_categories) - DISABLED_DETECTION_CATEGORIES,
        excluded_detection_params=set(FREE_TEXT_QUERY_PARAMS),
        rate_limit=settings.security_guard_rate_limit,
        rate_limit_window=settings.security_guard_rate_window,
        trusted_proxies=trusted_proxies(settings.forwarded_allow_ips),
        whitelist=parse_csv(settings.security_guard_allow_ips) or None,
        blacklist=parse_csv(settings.security_guard_block_ips),
        whitelist_countries=frozenset(c.upper() for c in parse_csv(settings.security_guard_allow_countries)),
        blocked_countries=frozenset(c.upper() for c in parse_csv(settings.security_guard_block_countries)),
        block_cloud_providers=frozenset(parse_csv(settings.security_guard_block_clouds)) or None,
        blocked_user_agents=list(parse_csv(settings.security_guard_block_user_agents)),
        ipinfo_token=os.environ.get("IPINFO_TOKEN") or None,
        emergency_mode=settings.security_guard_emergency,
        emergency_whitelist=list(emergency_ips),
        passive_mode=settings.security_guard_passive,
    )


def _ip_listed(client_ip: str, entries) -> bool:
    try:
        address = ip_address(client_ip)
        return any(address in ip_network(entry, strict=False) for entry in entries)
    except ValueError:
        return False


def make_handshake_dependency(guard, security_config):
    """Handshake checks for the WebSocket routes, run before `accept()`.

    fastapi-guard 8.1.2 checks bans, IP and country rules, cloud providers, the session
    budget and attack patterns on a handshake; emergency mode and user-agent rules are
    HTTP-pipeline checks there, so they are applied here.
    """
    from guard_core.utils import is_user_agent_allowed

    async def guard_handshake(websocket: WebSocket) -> None:
        await guard.guard_websocket(websocket)
        reason = None
        if security_config.emergency_mode and not _ip_listed(
            websocket.state.client_ip, security_config.emergency_whitelist
        ):
            reason = "Emergency mode"
        elif not await is_user_agent_allowed(websocket.headers.get("user-agent", ""), security_config):
            reason = "User agent not allowed"
        if reason is None:
            return
        if security_config.passive_mode:
            logger.warning("Passive mode: would refuse WebSocket handshake (%s) from %s", reason, websocket.state.client_ip)
            return
        raise WebSocketException(code=guard.WS_CLOSE_IP_NOT_ALLOWED.code, reason=reason)

    return guard_handshake


def install_security_guard(application, upload_endpoint: Callable) -> list:
    """Add the guard to `application`; return the dependencies for the WebSocket routes."""
    guard, make_lifespan, _ = _import_guard()
    settings = application.state.config
    security_config = build_security_config(settings)
    # Classic middleware: the pure-ASGI variant does not run behavior rules.
    application.add_middleware(guard.SecurityMiddleware, config=security_config)

    decorator = guard.SecurityDecorator(security_config)
    for apply in (
        decorator.rate_limit(settings.security_guard_upload_rate_limit, settings.security_guard_rate_window),
        decorator.max_request_size(settings.security_guard_max_upload_mb * 1024 * 1024 + UPLOAD_OVERHEAD_BYTES),
        decorator.content_type_filter(["multipart/form-data"]),
        decorator.return_monitor(
            "status:401", UNAUTHORIZED_BAN_THRESHOLD, UNAUTHORIZED_BAN_WINDOW_SECONDS, action="ban"
        ),
    ):
        apply(upload_endpoint)
    application.state.guard_decorator = decorator
    # Initialize at startup instead of on the first HTTP request, so a WebSocket
    # handshake that arrives first is already covered.
    application.router.lifespan_context = make_lifespan(application.router.lifespan_context)
    return [Depends(make_handshake_dependency(guard, security_config))]
