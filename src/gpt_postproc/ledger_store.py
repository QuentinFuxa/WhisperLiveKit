"""Append-only ledger store for GPT outputs."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import GPTConfig


class LedgerStore:
    """Persist GPT outputs in a JSONL ledger."""

    def __init__(self, path: Optional[str] = None) -> None:
        cfg = GPTConfig() if path is None else None
        self.path = path or cfg.ledger_path  # type: ignore[assignment]
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        record.setdefault("ts", time.time())
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def group_by_day(self) -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        if not os.path.exists(self.path):
            return groups

        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = data.get("start") or data.get("ts") or time.time()
                try:
                    day = datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    day = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d")
                groups[day].append(data)
        return groups
