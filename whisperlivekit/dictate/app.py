"""`wlk dictate`: press a hotkey, speak, press again, and the text is pasted at the cursor.

Key presses and microphone blocks arrive on their own threads. Everything that
touches the transcription pipeline runs on the asyncio loop.
"""

import argparse
import asyncio
import fcntl
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from whisperlivekit.dictate.audio import SAMPLE_RATE, has_speech, is_hallucination, to_pcm16
from whisperlivekit.dictate.session import TranscriptionFailed

logger = logging.getLogger(__name__)

IDLE, RECORDING, FINISHING = "idle", "recording", "finishing"
BLOCK_SIZE = 1600  # 0.1 s of audio per microphone block
HOTKEY_DEBOUNCE = 0.3  # Seconds; ignore repeated presses of the same key
MIN_FINISH_TIMEOUT = 30.0
LOCK_FILE = Path.home() / ".cache" / "wlk-dictate" / "lock"


class Dictation:
    """Record on the first hotkey press, transcribe and paste on the second."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        session_factory: Callable[[], object],
        open_microphone: Callable[[Callable[[np.ndarray], None]], object],
        paste: Callable[..., None],
        active_window: Callable[[], bytes | None],
        feedback,
    ):
        self._loop = loop
        self._session_factory = session_factory
        self._open_microphone = open_microphone
        self._paste = paste
        self._active_window = active_window
        self._feedback = feedback
        self._lock = threading.Lock()
        self.state = IDLE
        self.last_text: str | None = None
        self._microphone = None
        self._queue: asyncio.Queue | None = None
        self._audio: list[np.ndarray] = []

    def on_record_key(self):
        microphone = None
        with self._lock:
            if self.state == IDLE:
                self._start_recording()
            elif self.state == RECORDING:
                microphone = self._stop_recording()
            else:
                self._feedback.error()
        # Closed outside the lock: closing waits for the audio callback, which takes the lock
        _close(microphone)

    def on_repaste_key(self):
        if not self.last_text:
            self._feedback.error()
            return
        text, window = self.last_text, self._active_window()
        self._loop.call_soon_threadsafe(
            lambda: self._loop.run_in_executor(None, lambda: self._paste(text=text, window=window))
        )

    def _start_recording(self):
        window = self._active_window()
        self._feedback.start()
        self._audio = []
        self._queue = asyncio.Queue()
        self.state = RECORDING
        try:
            self._microphone = self._open_microphone(self._on_audio)
        except Exception as e:
            logger.exception("Could not open the microphone")
            self.state = IDLE
            self._feedback.error()
            self._feedback.notify("Dictation failed", f"Could not open the microphone: {e}")
            return
        future = asyncio.run_coroutine_threadsafe(self._dictate(queue=self._queue, window=window), self._loop)
        future.add_done_callback(_log_unexpected_error)

    def _stop_recording(self):
        """Switch to finishing; return the microphone for the caller to close."""
        self.state = FINISHING
        self._feedback.stop()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        microphone, self._microphone = self._microphone, None
        return microphone

    def _on_audio(self, chunk: np.ndarray):
        with self._lock:
            if self.state != RECORDING:
                return
            self._audio.append(chunk)
            self._loop.call_soon_threadsafe(self._queue.put_nowait, to_pcm16(chunk))

    async def _dictate(self, queue: asyncio.Queue, window: bytes | None):
        """Run one recording from first audio block to paste."""
        session = None
        try:
            session = self._session_factory()
            await session.start()
            feed_error = None
            # Keep reading until the stop press even if the pipeline fails, so
            # that press ends this recording instead of starting a new one.
            while (pcm := await queue.get()) is not None:
                if feed_error is None:
                    try:
                        await session.feed(pcm)
                    except Exception as e:
                        logger.exception("Feeding audio failed")
                        feed_error = e
            if feed_error is not None:
                raise feed_error

            audio = np.concatenate(self._audio) if self._audio else np.zeros(0, dtype=np.float32)
            if not has_speech(audio):
                self._feedback.notify("No speech detected")
                return
            timeout = max(MIN_FINISH_TIMEOUT, 2.5 * len(audio) / SAMPLE_RATE)
            text = await session.finish(timeout=timeout)
            if not text:
                self._feedback.notify("No speech detected")
                return
            self.last_text = text
            if is_hallucination(text):
                logger.info("Dropped likely hallucination: %r", text)
                self._feedback.notify("No speech detected", f"Heard {text!r}. Press F8 to paste it anyway.")
                return
            await asyncio.to_thread(self._paste, text=text, window=window)
            logger.info("Pasted %d characters", len(text))
        except Exception as e:
            logger.exception("Dictation failed")
            self._feedback.error()
            self._feedback.notify("Dictation failed", _error_message(e))
        finally:
            try:
                if session is not None:
                    await session.close()
            except Exception:
                logger.exception("Cleaning up the session failed")
            finally:
                with self._lock:
                    microphone, self._microphone = self._microphone, None
                    self.state = IDLE
                _close(microphone)


def _close(microphone):
    if microphone is not None:
        try:
            microphone.close()
        except Exception:
            logger.exception("Closing the microphone failed")


def _log_unexpected_error(future):
    if not future.cancelled() and future.exception() is not None:
        logger.error("Dictation task failed", exc_info=future.exception())


def _error_message(error: Exception) -> str:
    message = str(error) or type(error).__name__
    if "out of memory" in message.lower():
        return f"GPU out of memory: {message}"
    if not isinstance(error, TranscriptionFailed):
        return f"{type(error).__name__}: {message}"
    return message


def parse_dictate_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split dictation flags from the server/model flags."""
    parser = argparse.ArgumentParser(prog="wlk dictate", add_help=False)
    parser.add_argument("--hotkey", default="f9", help="Key that starts and stops a recording (default: f9)")
    parser.add_argument("--repaste-hotkey", default="f8", help="Key that pastes the last text again (default: f8)")
    parser.add_argument("-h", "--help", action="store_true")
    return parser.parse_known_args(argv)


def _print_help():
    from whisperlivekit.parse_args import build_parser

    print("usage: wlk dictate [--hotkey KEY] [--repaste-hotkey KEY] [model options]\n")
    print("Push-to-talk dictation: press the hotkey, speak, press it again; the text is pasted at the cursor.\n")
    print("dictation options:")
    print("  --hotkey KEY           start/stop recording (default: f9)")
    print("  --repaste-hotkey KEY   paste the last text again (default: f8)\n")
    print("Model and streaming options (same as `wlk serve`; host/port are ignored):\n")
    build_parser().print_help()


def _acquire_lock():
    """Exit if another `wlk dictate` is running; keep the returned file open."""
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK_FILE, "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.seek(0)
        print(f"wlk dictate is already running (PID {lock.read().strip() or '?'})")
        sys.exit(1)
    lock.truncate(0)
    lock.write(str(os.getpid()))
    lock.flush()
    return lock


def _open_microphone(callback: Callable[[np.ndarray], None]):
    import sounddevice as sd

    def on_block(indata, frames, time_info, status):
        callback(indata[:, 0].copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SIZE, callback=on_block)
    stream.start()
    return stream


def _key(name: str):
    from pynput import keyboard

    try:
        return getattr(keyboard.Key, name.lower())
    except AttributeError:
        raise SystemExit(f"Unknown hotkey: {name}") from None


def main(argv: list[str] | None = None):
    args, rest = parse_dictate_args(sys.argv[1:] if argv is None else argv)
    if args.help:
        _print_help()
        return

    from whisperlivekit import AudioProcessor, TranscriptionEngine
    from whisperlivekit.parse_args import parse_args

    config = parse_args(rest)
    logging.basicConfig(level=config.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    record_key, repaste_key = _key(args.hotkey), _key(args.repaste_hotkey)
    lock = _acquire_lock()  # noqa: F841 - held open for the life of the process

    logger.info("Loading model %s", config.model_size)
    engine = TranscriptionEngine(config=config)
    asyncio.run(_run(engine=engine, processor_class=AudioProcessor, record_key=record_key, repaste_key=repaste_key))


async def _run(engine, processor_class, record_key, repaste_key):
    from pynput import keyboard

    from whisperlivekit.dictate import paste
    from whisperlivekit.dictate.feedback import Feedback
    from whisperlivekit.dictate.session import DictationSession

    feedback = Feedback()
    dictation = Dictation(
        loop=asyncio.get_running_loop(),
        session_factory=lambda: DictationSession(
            processor_factory=lambda: processor_class(transcription_engine=engine, pcm_input=True)
        ),
        open_microphone=_open_microphone,
        paste=paste.paste,
        active_window=paste.active_window,
        feedback=feedback,
    )
    handlers = {record_key: dictation.on_record_key, repaste_key: dictation.on_repaste_key}
    last_press: dict = {}

    def on_press(key):
        handler = handlers.get(key)
        if handler is None:
            return
        now = time.monotonic()
        if now - last_press.get(key, 0.0) < HOTKEY_DEBOUNCE:
            return
        last_press[key] = now
        try:
            handler()
        except Exception:
            logger.exception("Hotkey handler failed")

    with keyboard.Listener(on_press=on_press):
        feedback.ready()
        logger.info("Ready. Press %s to dictate, %s to paste the last text again.",
                    record_key.name.upper(), repaste_key.name.upper())
        await asyncio.Event().wait()
