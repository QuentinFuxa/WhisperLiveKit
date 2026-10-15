"""Beeps and desktop notifications for dictation."""

import logging
import subprocess
import threading

import numpy as np

logger = logging.getLogger(__name__)

BEEP_RATE = 44100


class Feedback:
    """Short tones for start/stop/error and notify-send messages."""

    def start(self):
        self._beep(freq=880, duration=0.08)

    def stop(self):
        self._beep(freq=440, duration=0.12)

    def error(self):
        self._beep(freq=220, duration=0.2)

    def ready(self):
        self._beep(freq=660, duration=0.08)

    def notify(self, title: str, body: str = ""):
        threading.Thread(target=self._notify, args=(title, body), daemon=True).start()

    def _beep(self, freq: float, duration: float, volume: float = 0.12):
        import sounddevice as sd

        t = np.linspace(0, duration, int(BEEP_RATE * duration), False)
        wave = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        try:
            sd.play(wave, BEEP_RATE)
        except Exception:
            logger.debug("Beep failed", exc_info=True)

    def _notify(self, title: str, body: str):
        try:
            subprocess.run(["notify-send", title, body, "-t", "5000"], timeout=30, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            logger.debug("notify-send failed", exc_info=True)
