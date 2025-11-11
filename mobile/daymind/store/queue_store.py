"""Persistent queue for audio chunks awaiting upload."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional


class ChunkQueue:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def enqueue(self, file_path: str, lang: str = "auto") -> str:
        entry_id = uuid.uuid4().hex
        entry = {
            "id": entry_id,
            "path": file_path,
            "lang": lang,
            "created_at": time.time(),
            "attempts": 0,
            "next_retry": time.time(),
            "last_error": None,
        }
        self._data.append(entry)
        self._persist()
        return entry_id

    def list(self) -> List[Dict]:
        return list(self._data)

    def peek(self) -> Optional[Dict]:
        now = time.time()
        ready = [item for item in self._data if item["next_retry"] <= now]
        if not ready:
            return None
        ready.sort(key=lambda item: item["created_at"])
        return ready[0]

    def mark_sent(self, entry_id: str) -> None:
        self._data = [item for item in self._data if item["id"] != entry_id]
        self._persist()

    def mark_failed(self, entry_id: str, error: str) -> None:
        for item in self._data:
            if item["id"] == entry_id:
                item["attempts"] += 1
                delay = min(60.0, 2 ** item["attempts"])
                item["next_retry"] = time.time() + delay
                item["last_error"] = error[-200:]
                break
        self._persist()

    def clear(self) -> None:
        for item in self._data:
            try:
                Path(item["path"]).unlink(missing_ok=True)
            except Exception:
                pass
        self._data = []
        self._persist()

    def __len__(self) -> int:
        return len(self._data)

    def _load(self) -> List[Dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _persist(self) -> None:
        self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")

