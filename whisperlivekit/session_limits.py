"""Admission control before WebSocket handlers allocate session resources."""

import logging

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


class WebSocketSessionLimitMiddleware:
    """Bound WebSocket handlers, including their cleanup, within one app/worker."""

    def __init__(self, app: ASGIApp, max_sessions: int):
        self.app = app
        self.max_sessions = max_sessions
        self.active_sessions = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "websocket":
            await self.app(scope, receive, send)
            return
        # No await between admission and reservation: handlers on this event
        # loop cannot reserve the same capacity. Each app/worker owns its count.
        if self.active_sessions >= self.max_sessions:
            client = scope.get("client")
            logger.warning(
                "WebSocket rejected: maximum concurrent sessions reached (client=%s)",
                client[0] if client else "unknown",
            )
            # Closing before accept rejects the HTTP upgrade (HTTP 403).
            await send({
                "type": "websocket.close", "code": 1013,
                "reason": "Maximum concurrent sessions reached",
            })
            return
        self.active_sessions += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.active_sessions -= 1
