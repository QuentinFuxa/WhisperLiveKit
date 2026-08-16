"""Shared authentication helpers for HTTP and WebSocket APIs."""

from typing import Optional

from fastapi import WebSocket


def websocket_token(websocket: WebSocket) -> Optional[str]:
    """Read WLK query auth plus Bearer or Deepgram-style Token headers."""
    candidate = websocket.query_params.get("token")
    if candidate:
        return candidate
    auth = websocket.headers.get("authorization") or ""
    scheme, _, value = auth.partition(" ")
    if scheme.lower() in {"bearer", "token"}:
        return value.strip() or None
    return None
