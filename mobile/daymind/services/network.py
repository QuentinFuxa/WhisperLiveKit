"""HTTP client helpers for DayMind mobile."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import httpx

from ..store.settings_store import SettingsStore


class ApiError(Exception):
    pass


class ApiClient:
    def __init__(self, settings: SettingsStore, *, timeout: float = 15.0, client: Optional[httpx.Client] = None) -> None:
        self.settings_store = settings
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def _headers(self) -> dict:
        api_key = self.settings_store.get().api_key
        if not api_key:
            raise ApiError("API key missing")
        return {"X-API-Key": api_key}

    def _url(self, path: str) -> str:
        base = self.settings_store.get().server_url.rstrip("/")
        if not base:
            raise ApiError("Server URL missing")
        return f"{base}{path}"

    def test_connection(self) -> bool:
        try:
            resp = self._client.get(self._url("/healthz"), headers=self._headers())
            return resp.status_code == 200
        except Exception as exc:
            raise ApiError(str(exc))

    def upload_chunk(self, file_path: str, lang: str = "auto") -> None:
        try:
            with open(file_path, "rb") as fh:
                files = {"file": (Path(file_path).name, fh, "audio/wav")}
                data = {"lang": lang}
                resp = self._client.post(
                    self._url("/v1/transcribe"),
                    headers=self._headers(),
                    files=files,
                    data=data,
                )
            if resp.status_code == 401:
                raise ApiError("Unauthorized: check API key")
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ApiError(f"Upload failed: {exc.response.status_code}")
        except Exception as exc:
            raise ApiError(str(exc))

    def fetch_summary(self, date: Optional[str] = None) -> str:
        date = date or dt.datetime.utcnow().strftime("%Y-%m-%d")
        try:
            resp = self._client.get(
                self._url("/v1/summary"),
                headers=self._headers(),
                params={"date": date},
            )
            if resp.status_code == 404:
                raise ApiError("Summary not available yet")
            resp.raise_for_status()
            data = resp.json()
            return data.get("summary_md", "") or resp.text
        except httpx.HTTPStatusError as exc:
            raise ApiError(f"Summary error: {exc.response.status_code}")
        except Exception as exc:
            raise ApiError(str(exc))

    def close(self) -> None:
        self._client.close()

