"""Lightweight ring buffer for log events."""

from __future__ import annotations

import time
from typing import Deque, List
from collections import deque


class LogBuffer:
    def __init__(self, max_entries: int = 200) -> None:
        self._entries: Deque[str] = deque(maxlen=max_entries)

    def add(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self._entries.appendleft(f"[{timestamp}] {message}")

    def get(self) -> List[str]:
        return list(self._entries)

