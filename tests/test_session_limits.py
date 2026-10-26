"""Streaming admission and cleanup over real sockets and FFmpeg pipes."""

import asyncio
import json
import socket
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from whisperlivekit import AudioProcessor, TranscriptionEngine
from whisperlivekit.basic_server import create_app
from whisperlivekit.config import WhisperLiveKitConfig
from whisperlivekit.parse_args import parse_args


def _app(**options):
    config = WhisperLiveKitConfig(transcription=False, vac=False, **options)
    app = create_app(config)
    # No model is needed to exercise admission, actual decoder startup and
    # cleanup. Avoid the engine singleton and its model-loading side effects.
    engine = object.__new__(TranscriptionEngine)
    engine.args = engine.config = config
    engine.asr = engine.translation_model = None
    app.state.transcription_engine = engine
    return app


@asynccontextmanager
async def _serve(app):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await _until(lambda: server.started or task.done())
        if task.done():
            await task
        assert server.started
        yield f"127.0.0.1:{listener.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        listener.close()


async def _until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


async def _reconnect(uri):
    # A close frame can reach the client before FFmpeg cleanup has finished.
    async with asyncio.timeout(3):
        while True:
            try:
                return await connect(uri)
            except InvalidStatus as exc:
                assert exc.response.status_code == 403
                await asyncio.sleep(0.01)


@pytest.fixture
def decoders(monkeypatch):
    monkeypatch.delenv("WLK_API_TOKEN", raising=False)
    processes = []
    spawn = asyncio.create_subprocess_exec

    async def record(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", record)
    return processes


@pytest.mark.asyncio
@pytest.mark.parametrize("first_path", ["/asr", "/v1/listen?endpointing=false"])
@pytest.mark.parametrize("second_path", ["/asr", "/v1/listen?endpointing=false"])
async def test_streaming_routes_share_capacity_and_reject_before_decoder_start(
    first_path, second_path, decoders,
):
    async with _serve(_app(max_concurrent_sessions=1)) as origin:
        async with connect(f"ws://{origin}{first_path}") as first:
            await first.recv()
            await _until(lambda: len(decoders) == 1)
            with pytest.raises(InvalidStatus) as rejected:
                async with connect(f"ws://{origin}{second_path}"):
                    pytest.fail("An excess session was accepted")
            assert rejected.value.response.status_code == 403
            assert len(decoders) == 1
            async with httpx.AsyncClient() as client:
                assert (await client.get(f"http://{origin}/health")).status_code == 200
        replacement = await _reconnect(f"ws://{origin}{second_path}")
        try:
            await replacement.recv()
            await _until(lambda: len(decoders) == 2)
        finally:
            await replacement.close()
    assert all(process.returncode is not None for process in decoders)


@pytest.mark.asyncio
async def test_idle_native_session_reaps_decoder_and_releases_capacity(decoders):
    async with _serve(_app(max_concurrent_sessions=1, session_idle_timeout=0.15)) as origin:
        async with connect(f"ws://{origin}/asr") as idle:
            assert json.loads(await idle.recv())["type"] == "config"
            # Protocol pings keep the transport alive, but are not audio.
            for _ in range(3):
                await idle.ping()
                await asyncio.sleep(0.03)
            await asyncio.wait_for(idle.wait_closed(), timeout=3)
            assert idle.close_code == 1001
            assert idle.close_reason == "session idle timeout"
        replacement = await _reconnect(f"ws://{origin}/v1/listen?endpointing=false")
        try:
            assert json.loads(await replacement.recv())["type"] == "Metadata"
        finally:
            await replacement.close()
    assert len(decoders) == 2
    assert all(process.returncode is not None for process in decoders)


@pytest.mark.asyncio
async def test_capacity_stays_reserved_until_cleanup_finishes(monkeypatch, decoders):
    entered = asyncio.Event()
    resume = asyncio.Event()
    cleanup = AudioProcessor.cleanup

    async def held_cleanup(self):
        await cleanup(self)
        entered.set()
        await resume.wait()

    monkeypatch.setattr(AudioProcessor, "cleanup", held_cleanup)
    async with _serve(_app(max_concurrent_sessions=1)) as origin:
        try:
            async with connect(f"ws://{origin}/asr") as first:
                await first.recv()
                await _until(lambda: len(decoders) == 1)
            await asyncio.wait_for(entered.wait(), timeout=3)
            assert decoders[0].returncode is not None
            with pytest.raises(InvalidStatus) as rejected:
                async with connect(f"ws://{origin}/asr"):
                    pytest.fail("Capacity was released before cleanup finished")
            assert rejected.value.response.status_code == 403
        finally:
            resume.set()
        replacement = await _reconnect(f"ws://{origin}/asr")
        await replacement.close()


@pytest.mark.asyncio
async def test_activity_and_disabled_timeout_keep_native_sessions_open(decoders):
    async with _serve(_app(pcm_input=True, session_idle_timeout=0.2)) as origin:
        async with connect(f"ws://{origin}/asr") as active:
            await active.recv()
            for _ in range(6):
                await active.send(b"\x00\x00" * 1600)
                await asyncio.sleep(0.06)
            await active.send(b"")
            messages = [json.loads(message) async for message in active]
            assert messages[-1] == {"type": "ready_to_stop"}
            assert active.close_code == 1000
    async with _serve(_app(pcm_input=True, session_idle_timeout=0)) as origin:
        async with connect(f"ws://{origin}/asr") as idle:
            await idle.recv()
            await asyncio.sleep(0.25)
            await idle.send(b"")
            messages = [json.loads(message) async for message in idle]
            assert messages[-1] == {"type": "ready_to_stop"}
    assert decoders == []


@pytest.mark.asyncio
async def test_eof_drains_final_result_beyond_idle_timeout(monkeypatch, decoders):
    from whisperlivekit import basic_server
    from whisperlivekit.timed_objects import FrontData

    class DelayedFinalProcessor:
        def __init__(self, **kwargs):
            self.finished = asyncio.Event()

        async def create_tasks(self):
            async def results():
                await self.finished.wait()
                await asyncio.sleep(0.2)
                yield FrontData(status="active_transcription", buffer_transcription="final words")
            return results()

        async def process_audio(self, message):
            self.finished.set()

        async def cleanup(self):
            pass

    monkeypatch.setattr(basic_server, "AudioProcessor", DelayedFinalProcessor)
    async with _serve(_app(session_idle_timeout=0.1)) as origin:
        async with connect(f"ws://{origin}/asr") as stream:
            await stream.recv()
            await stream.send(b"")
            messages = [json.loads(message) async for message in stream]
            assert messages[0]["buffer_transcription"] == "final words"
            assert messages[-1] == {"type": "ready_to_stop"}
            assert stream.close_code == 1000


@pytest.mark.asyncio
async def test_disconnect_during_final_drain_releases_capacity(monkeypatch, decoders):
    from whisperlivekit import basic_server

    cleaned = asyncio.Event()

    class HeldFinalProcessor:
        def __init__(self, **kwargs):
            pass

        async def create_tasks(self):
            async def results():
                await asyncio.Event().wait()
                yield None
            return results()

        async def process_audio(self, message):
            pass

        async def cleanup(self):
            cleaned.set()

    monkeypatch.setattr(basic_server, "AudioProcessor", HeldFinalProcessor)
    async with _serve(_app(max_concurrent_sessions=1)) as origin:
        async with connect(f"ws://{origin}/asr") as stream:
            await stream.recv()
            await stream.send(b"")
        await asyncio.wait_for(cleaned.wait(), timeout=2)
        replacement = await _reconnect(f"ws://{origin}/asr")
        await replacement.close()


@pytest.mark.asyncio
async def test_rejected_and_failed_sessions_return_capacity(monkeypatch, decoders):
    async with _serve(_app(max_concurrent_sessions=1, api_token="secret")) as origin:
        for path in ("/asr", "/v1/listen"):
            with pytest.raises(InvalidStatus):
                async with connect(f"ws://{origin}{path}"):
                    pytest.fail("Unauthenticated connection accepted")
        assert decoders == []
        # A handled parameter error must also return its reservation.
        async with connect(f"ws://{origin}/v1/listen?token=secret&encoding=invalid") as invalid:
            assert json.loads(await invalid.recv())["type"] == "Error"
            await invalid.wait_closed()

        async def fail_startup(self):
            raise RuntimeError("decoder startup failed")

        with monkeypatch.context() as patch:
            patch.setattr(AudioProcessor, "create_tasks", fail_startup)
            async with connect(f"ws://{origin}/asr?token=secret") as failed:
                await failed.recv()
                assert json.loads(await failed.recv())["type"] == "error"
                await failed.wait_closed()
        assert decoders == []
        recovered = await _reconnect(f"ws://{origin}/asr?token=secret")
        try:
            assert json.loads(await recovered.recv())["type"] == "config"
        finally:
            await recovered.close()
    assert len(decoders) == 1
    assert decoders[0].returncode is not None


@pytest.mark.asyncio
async def test_embedded_apps_have_independent_capacity(decoders):
    parent = FastAPI()
    parent.mount("/first", _app(max_concurrent_sessions=1))
    parent.mount("/second", _app(max_concurrent_sessions=1))
    async with _serve(parent) as origin:
        async with connect(f"ws://{origin}/first/asr") as first:
            await first.recv()
            async with connect(f"ws://{origin}/second/v1/listen?endpointing=false") as second:
                await second.recv()
                await _until(lambda: len(decoders) == 2)
    assert all(process.returncode is not None for process in decoders)


@pytest.mark.parametrize("name,value", [
    ("max_concurrent_sessions", 0), ("max_concurrent_sessions", -1),
    ("max_concurrent_sessions", 1.5), ("max_concurrent_sessions", True),
    ("session_idle_timeout", -1), ("session_idle_timeout", float("inf")),
    ("session_idle_timeout", float("nan")),
])
def test_invalid_limits_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        WhisperLiveKitConfig(**{name: value})


def test_cli_configures_session_limits():
    settings = parse_args(["--max-concurrent-sessions", "3", "--session-idle-timeout", "0"])
    assert settings.max_concurrent_sessions == 3
    assert settings.session_idle_timeout == 0
