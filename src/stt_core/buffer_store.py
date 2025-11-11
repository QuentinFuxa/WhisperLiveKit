"""Append-only JSONL buffer with rolling truncation."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict


class BufferStore:
    """Persist transcript payloads locally with a soft size limit."""

    def __init__(self, path: str, max_mb: int = 32) -> None:
        self.path = path
        self.max_bytes = max_mb * 1024 * 1024
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        """Append a record to the JSONL buffer and enforce size limits."""

        record.setdefault("ts", time.time())
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._truncate_if_needed()

    def _truncate_if_needed(self) -> None:
        """Drop the oldest lines if the buffer exceeds the allowed size."""

        try:
            if os.path.getsize(self.path) <= self.max_bytes:
                return
            with open(self.path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            if not lines:
                return
            keep = max(1, int(len(lines) * 0.7))
            with open(self.path, "w", encoding="utf-8") as fh:
                fh.writelines(lines[-keep:])
        except FileNotFoundError:
            return

