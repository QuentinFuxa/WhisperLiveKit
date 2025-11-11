"""Audio recording helper with chunked output."""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional

from ..config import CONFIG
from ..services.logger import LogBuffer


class AudioRecorder:
    def __init__(self, output_dir: Path, logger: LogBuffer, on_chunk: Callable[[str], None]) -> None:
        self.output_dir = output_dir
        self.logger = logger
        self.on_chunk = on_chunk
        self.chunk_seconds = CONFIG.chunk_seconds
        self.sample_rate = CONFIG.sample_rate
        self.channels = CONFIG.channels
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._sd = self._try_import_sounddevice()

    def _try_import_sounddevice(self):
        try:
            import sounddevice as sd  # type: ignore

            return sd
        except Exception:
            return None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.logger.add("Recording started")
        self._stop.clear()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.logger.add("Recording stopped")

    def _loop(self) -> None:
        while not self._stop.is_set():
            path = self._record_chunk()
            if path:
                self.on_chunk(path)
            time.sleep(0.1)

    def _record_chunk(self) -> Optional[str]:
        filename = f"chunk_{int(time.time()*1000)}.wav"
        path = self.output_dir / filename
        frames = self.sample_rate * self.chunk_seconds
        if self._sd:
            try:
                data = self._sd.rec(frames, samplerate=self.sample_rate, channels=self.channels, dtype="int16")
                self._sd.wait()
                self._write_wave(path, data)
                return str(path)
            except Exception as exc:
                self.logger.add(f"sounddevice error: {exc}")
        # fallback: silence
        import array

        buffer = array.array("h", [0] * frames)
        self._write_wave(path, buffer)
        return str(path)

    def _write_wave(self, path: Path, data) -> None:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(data.tobytes() if hasattr(data, "tobytes") else data)

