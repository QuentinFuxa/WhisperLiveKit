"""Background worker that drains the chunk queue."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from .logger import LogBuffer
from .network import ApiClient, ApiError
from ..store.queue_store import ChunkQueue


class UploadWorker:
    def __init__(self, queue: ChunkQueue, client: ApiClient, logger: LogBuffer) -> None:
        self.queue = queue
        self.client = client
        self.logger = logger
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def wake(self) -> None:
        self._wake_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            entry = self.queue.peek()
            if not entry:
                self._wake_event.wait(timeout=2)
                self._wake_event.clear()
                continue
            entry_id = entry["id"]
            try:
                self.logger.add(f"Uploading chunk {entry_id[:6]}...")
                self.client.upload_chunk(entry["path"], entry.get("lang", "auto"))
            except ApiError as exc:
                self.logger.add(f"Upload failed ({entry_id[:6]}): {exc}")
                self.queue.mark_failed(entry_id, str(exc))
                time.sleep(1)
            else:
                try:
                    Path(entry["path"]).unlink(missing_ok=True)
                except Exception:
                    pass
                self.queue.mark_sent(entry_id)
                self.logger.add(f"Chunk {entry_id[:6]} sent")

