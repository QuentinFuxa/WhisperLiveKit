"""Persistent settings storage for server URL and API key."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class AppSettings:
    server_url: str = ""
    api_key: str = ""


class SettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._settings = self._load()

    @property
    def _path(self) -> Path:
        return self.path

    def _load(self) -> AppSettings:
        if not self._path.exists():
            return AppSettings()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return AppSettings(**{k: data.get(k, "") for k in ("server_url", "api_key")})
        except Exception:
            return AppSettings()

    def get(self) -> AppSettings:
        return self._settings

    def update(self, **kwargs) -> AppSettings:
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value or "")
        self._persist()
        return self._settings

    def _persist(self) -> None:
        self._path.write_text(json.dumps(asdict(self._settings)), encoding="utf-8")

