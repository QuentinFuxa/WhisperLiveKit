"""Tests for the Dictation state machine with fake session, microphone, paste and feedback."""

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from whisperlivekit.dictate.app import FINISHING, IDLE, RECORDING, Dictation
from whisperlivekit.dictate.session import TranscriptionFailed

LOUD = np.full(1600, 0.2, dtype=np.float32)
QUIET = np.zeros(1600, dtype=np.float32)


class FakeMicrophone:
    def __init__(self, callback, lock=None):
        self.callback = callback
        self.open = True
        self.lock = lock

    def push(self, chunk):
        self.callback(chunk)

    def close(self):
        self.open = False
        self.closed_under_lock = self.lock.locked() if self.lock else None


class FakeSession:
    def __init__(self, text="Hello world.", error=None, feed_error=None, close_error=None):
        self.text = text
        self.error = error
        self.feed_error = feed_error
        self.close_error = close_error
        self.fed = []
        self.closed = False
        self.release = asyncio.Event()
        self.release.set()

    async def start(self):
        pass

    async def feed(self, pcm):
        if self.feed_error:
            raise RuntimeError(self.feed_error)
        self.fed.append(pcm)

    async def finish(self, timeout):
        await self.release.wait()
        if self.error:
            raise TranscriptionFailed(self.error)
        return self.text

    async def close(self):
        self.closed = True
        if self.close_error:
            raise RuntimeError(self.close_error)


class FakeFeedback:
    def __init__(self):
        self.events = []

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")

    def error(self):
        self.events.append("error")

    def ready(self):
        self.events.append("ready")

    def notify(self, title, body=""):
        self.events.append(("notify", title, body))


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)


@pytest.fixture
def harness(loop):
    """Dictation wired to fakes; session_kwargs set before a recording shape the next FakeSession."""
    h = SimpleNamespace(sessions=[], microphones=[], pastes=[], feedback=FakeFeedback(), session_kwargs={}, session_error=None, paste_error=None)

    def make_session():
        if h.session_error:
            raise RuntimeError(h.session_error)
        session = FakeSession(**h.session_kwargs)
        h.sessions.append(session)
        return session

    def open_microphone(callback):
        microphone = FakeMicrophone(callback, lock=h.dictation._lock)
        h.microphones.append(microphone)
        return microphone

    def fake_paste(text, window):
        if h.paste_error:
            raise RuntimeError(h.paste_error)
        h.pastes.append((text, window))

    h.dictation = Dictation(
        loop=loop,
        session_factory=make_session,
        open_microphone=open_microphone,
        paste=fake_paste,
        active_window=lambda: b"7",
        feedback=h.feedback,
    )
    h.loop = loop
    return h


def wait_for(condition, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


def record(h, chunks):
    h.dictation.on_record_key()
    for chunk in chunks:
        h.microphones[-1].push(chunk)
    h.dictation.on_record_key()


def test_two_presses_paste_text(harness):
    record(harness, [LOUD, LOUD])

    wait_for(lambda: harness.dictation.state == IDLE and harness.pastes)
    assert harness.pastes == [("Hello world.", b"7")]
    assert len(harness.sessions[0].fed) == 2
    assert harness.sessions[0].closed
    assert not harness.microphones[0].open
    assert harness.feedback.events[:2] == ["start", "stop"]


def test_f9_while_finishing_is_ignored(harness):
    harness.dictation.on_record_key()
    wait_for(lambda: harness.sessions)
    harness.loop.call_soon_threadsafe(harness.sessions[0].release.clear)
    harness.microphones[0].push(LOUD)
    harness.dictation.on_record_key()
    wait_for(lambda: harness.dictation.state == FINISHING)

    harness.dictation.on_record_key()

    assert len(harness.sessions) == 1
    assert harness.feedback.events[-1] == "error"
    harness.loop.call_soon_threadsafe(harness.sessions[0].release.set)
    wait_for(lambda: harness.dictation.state == IDLE)


def test_silent_recording_is_not_pasted(harness):
    record(harness, [QUIET, QUIET])

    wait_for(lambda: harness.dictation.state == IDLE and harness.sessions[0].closed)
    assert harness.pastes == []
    assert ("notify", "No speech detected", "") in harness.feedback.events


def test_hallucination_is_not_pasted(harness):
    harness.session_kwargs = {"text": "Thank you."}
    record(harness, [LOUD])

    wait_for(lambda: harness.dictation.state == IDLE and harness.sessions[0].closed)
    assert harness.pastes == []


def test_failed_finish_notifies_and_returns_to_idle(harness):
    harness.session_kwargs = {"error": "CUDA failed with error out of memory"}
    record(harness, [LOUD])

    wait_for(lambda: harness.dictation.state == IDLE and harness.sessions[0].closed)
    assert harness.pastes == []
    assert "error" in harness.feedback.events
    notes = [e for e in harness.feedback.events if isinstance(e, tuple)]
    assert "GPU out of memory" in notes[-1][2]


def test_audio_after_stop_is_ignored(harness):
    record(harness, [LOUD])
    harness.microphones[0].push(LOUD)

    wait_for(lambda: harness.dictation.state == IDLE and harness.pastes)
    assert len(harness.sessions[0].fed) == 1


def test_repaste_without_text_beeps(harness):
    harness.dictation.on_repaste_key()

    assert harness.feedback.events == ["error"]
    assert harness.pastes == []


def test_repaste_pastes_last_text_into_focused_window(harness):
    record(harness, [LOUD])
    wait_for(lambda: harness.dictation.state == IDLE and harness.pastes)

    harness.dictation.on_repaste_key()

    wait_for(lambda: len(harness.pastes) == 2)
    assert harness.pastes[1] == ("Hello world.", b"7")


def test_state_is_recording_after_first_press(harness):
    harness.dictation.on_record_key()

    assert harness.dictation.state == RECORDING
    harness.dictation.on_record_key()
    wait_for(lambda: harness.dictation.state == IDLE)


def test_parse_dictate_args_leaves_model_flags():
    from whisperlivekit.dictate.app import parse_dictate_args

    args, rest = parse_dictate_args(["--hotkey", "f12", "--model", "large-v3", "--no-vac"])

    assert args.hotkey == "f12"
    assert args.repaste_hotkey == "f8"
    assert rest == ["--model", "large-v3", "--no-vac"]


def test_session_factory_failure_returns_to_idle(harness):
    harness.session_error = "CUDA failed with error out of memory"
    harness.dictation.on_record_key()

    wait_for(lambda: harness.dictation.state == IDLE)
    assert not harness.microphones[0].open
    assert "error" in harness.feedback.events


def test_close_failure_returns_to_idle(harness):
    harness.session_kwargs = {"close_error": "cleanup failed"}
    record(harness, [LOUD])

    wait_for(lambda: harness.dictation.state == IDLE and harness.pastes)


def test_feed_error_waits_for_stop_press(harness):
    harness.session_kwargs = {"feed_error": "pipeline overloaded"}
    harness.dictation.on_record_key()
    harness.microphones[0].push(LOUD)
    time.sleep(0.1)

    assert harness.dictation.state == RECORDING
    harness.dictation.on_record_key()
    wait_for(lambda: harness.dictation.state == IDLE and harness.sessions[0].closed)
    assert harness.pastes == []
    notes = [e for e in harness.feedback.events if isinstance(e, tuple)]
    assert "pipeline overloaded" in notes[-1][2]


def test_paste_failure_keeps_text_for_repaste(harness):
    harness.paste_error = "xclip not found"
    record(harness, [LOUD])
    wait_for(lambda: harness.dictation.state == IDLE and harness.sessions[0].closed)

    assert harness.dictation.last_text == "Hello world."


def test_dropped_filler_kept_for_repaste(harness):
    harness.session_kwargs = {"text": "Thank you."}
    record(harness, [LOUD])
    wait_for(lambda: harness.dictation.state == IDLE and harness.sessions[0].closed)

    assert harness.pastes == []
    assert harness.dictation.last_text == "Thank you."


def test_microphone_closed_outside_lock(harness):
    record(harness, [LOUD])
    wait_for(lambda: harness.dictation.state == IDLE and harness.pastes)

    assert harness.microphones[0].closed_under_lock is False
