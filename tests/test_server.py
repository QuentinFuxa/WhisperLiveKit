"""HTTP/WebSocket contracts, export formats, and request cleanup."""

import asyncio
import importlib
import json
import sys
from types import SimpleNamespace

import pytest


def _import_basic_server(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["whisperlivekit-server"])
    return importlib.import_module("whisperlivekit.basic_server")


@pytest.mark.asyncio
async def test_native_websocket_passes_context_and_advertises_capability(monkeypatch):
    from fastapi import WebSocketDisconnect

    basic_server = _import_basic_server(monkeypatch)
    processor_kwargs = []
    cleaned = []

    class RecordingSocket:
        query_params = {
            "language": "fr",
            "context": "WhisperLiveKit, Qwen3-ASR",
        }
        headers = {}

        def __init__(self):
            self.sent = []
            self.accepted = False

        async def accept(self):
            self.accepted = True

        async def send_json(self, message):
            self.sent.append(message)

        async def receive_bytes(self):
            raise WebSocketDisconnect()

        async def close(self, code, **kwargs):
            self.closed = code

    class EmptyProcessor:
        def __init__(self, **kwargs):
            processor_kwargs.append(kwargs)

        async def create_tasks(self):
            async def results():
                if False:
                    yield None

            return results()

        async def cleanup(self):
            cleaned.append(True)

    asr = SimpleNamespace(backend_choice="faster-whisper")
    monkeypatch.setattr(
        basic_server,
        "transcription_engine",
        SimpleNamespace(
            args=SimpleNamespace(
                backend="faster-whisper",
                backend_policy="localagreement",
            ),
            asr=asr,
        ),
    )
    monkeypatch.setattr(basic_server, "_API_TOKEN", None)
    monkeypatch.setattr(basic_server, "AudioProcessor", EmptyProcessor)
    websocket = RecordingSocket()

    await basic_server.websocket_endpoint(websocket)

    assert websocket.accepted
    assert processor_kwargs[0]["language"] == "fr"
    assert processor_kwargs[0]["context"] == "WhisperLiveKit, Qwen3-ASR"
    config_message = next(m for m in websocket.sent if m["type"] == "config")
    assert config_message["context"] == {
        "supported": True,
        "maxCharacters": 1000,
        "backend": "faster-whisper",
    }
    assert cleaned == [True]

    async def fail_startup(self):
        raise RuntimeError("model failed during startup")
    monkeypatch.setattr(EmptyProcessor, "create_tasks", fail_startup)
    failed = RecordingSocket()
    await basic_server.websocket_endpoint(failed)
    assert failed.closed == 1011
    assert failed.sent[-1]["type"] == "error"
    assert cleaned == [True, True]

    async def disconnect_during_accept(self):
        raise WebSocketDisconnect()
    monkeypatch.setattr(RecordingSocket, "accept", disconnect_during_accept)
    await basic_server.websocket_endpoint(RecordingSocket())
    assert cleaned == [True, True, True]


@pytest.mark.asyncio
async def test_openai_rest_rejects_unsupported_context_before_expensive_work(
    monkeypatch,
):
    basic_server = _import_basic_server(monkeypatch)
    expensive_calls = []

    class UnreadUpload:
        async def read(self, size=-1):
            expensive_calls.append("read")
            return b"encoded audio"

    async def forbidden_conversion(audio_bytes):
        expensive_calls.append("convert")
        return audio_bytes

    class ForbiddenAudioProcessor:
        def __init__(self, **kwargs):
            expensive_calls.append("asr")

    monkeypatch.setattr(
        basic_server,
        "transcription_engine",
        SimpleNamespace(
            args=SimpleNamespace(
                backend="qwen3-streaming",
                backend_policy="simulstreaming",
            ),
            asr=SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(basic_server, "_API_TOKEN", None)
    monkeypatch.setattr(basic_server, "_convert_to_pcm", forbidden_conversion)
    monkeypatch.setattr(basic_server, "AudioProcessor", ForbiddenAudioProcessor)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await basic_server.create_transcription(
            request=SimpleNamespace(headers={}),
            file=UnreadUpload(),
            model="ignored",
            language="en",
            prompt="must not be ignored",
            response_format="json",
            timestamp_granularities=None,
        )

    assert exc_info.value.status_code == 400
    assert "not supported by backend 'qwen3-streaming'" in exc_info.value.detail
    assert expensive_calls == []


def test_openai_rest_diarized_json_preserves_speaker_labels(monkeypatch):
    basic_server = _import_basic_server(monkeypatch)

    front_data = SimpleNamespace(
        to_dict=lambda: {
            "lines": [
                {"text": "hello", "start": "0:00:00.00", "end": "0:00:02.10", "speaker": 1},
                {"text": "", "start": "0:00:02.10", "end": "0:00:05.10", "speaker": -2},
                {"text": "hi", "start": "0:00:05.10", "end": "0:00:06.00", "speaker": 2},
            ]
        }
    )

    payload = basic_server._format_openai_response(front_data, "diarized_json", "en", 6.0)

    assert payload == {
        "task": "transcribe",
        "duration": 6.0,
        "text": "A: hello\nB: hi",
        "segments": [
            {
                "type": "transcript.text.segment",
                "id": "seg_001",
                "start": 0.0,
                "end": 2.1,
                "text": "hello",
                "speaker": "A",
            },
            {
                "type": "transcript.text.segment",
                "id": "seg_002",
                "start": 5.1,
                "end": 6.0,
                "text": "hi",
                "speaker": "B",
            },
        ],
        "usage": {
            "type": "duration",
            "seconds": 6,
        },
    }


def test_openai_rest_verbose_json_shape_remains_without_speaker(monkeypatch):
    basic_server = _import_basic_server(monkeypatch)

    front_data = SimpleNamespace(
        to_dict=lambda: {
            "lines": [
                {"text": "hello world", "start": "0:00:00.00", "end": "0:00:02.00", "speaker": 1},
            ]
        }
    )

    payload = basic_server._format_openai_response(front_data, "verbose_json", "en", 2.0)

    assert payload["segments"] == [
        {
            "id": 0,
            "start": 0.0,
            "end": 2.0,
            "text": "hello world",
        }
    ]
    assert "speaker" not in payload["segments"][0]
    assert payload["usage"] == {
        "type": "duration",
        "seconds": 2,
    }


def test_openai_rest_verbose_json_uses_real_asr_token_timestamps(monkeypatch):
    """verbose_json should emit real per-word timestamps from ASRToken, not fabricated."""
    from whisperlivekit.timed_objects import ASRToken, FrontData, Segment

    basic_server = _import_basic_server(monkeypatch)

    tokens = [
        ASRToken(start=0.0, end=0.5, text="hello"),
        ASRToken(start=0.5, end=1.2, text="world"),
        ASRToken(start=1.2, end=2.0, text="there"),
    ]
    seg = Segment(start=0.0, end=2.0, text="hello world there", speaker=1, tokens=tokens)
    front_data = FrontData(lines=[seg])

    payload = basic_server._format_openai_response(front_data, "verbose_json", "en", 2.0)

    assert payload["words"] == [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.2},
        {"word": "there", "start": 1.2, "end": 2.0},
    ]


def test_openai_rest_verbose_json_falls_back_to_interpolation_without_tokens(monkeypatch):
    """When Segment.tokens is None, verbose_json falls back to interpolated timestamps."""
    from whisperlivekit.timed_objects import FrontData, Segment

    basic_server = _import_basic_server(monkeypatch)

    seg = Segment(start=0.0, end=2.0, text="hello world", speaker=1, tokens=None)
    front_data = FrontData(lines=[seg])

    payload = basic_server._format_openai_response(front_data, "verbose_json", "en", 2.0)

    # Two words, evenly distributed across 2 seconds
    assert payload["words"] == [
        {"word": "hello", "start": 0.0, "end": 1.0},
        {"word": "world", "start": 1.0, "end": 2.0},
    ]


def test_openai_rest_diarized_json_rejects_when_diarization_disabled(monkeypatch):
    """diarized_json without diarization enabled should raise 400, not fake speaker labels."""
    basic_server = _import_basic_server(monkeypatch)

    front_data = SimpleNamespace(
        to_dict=lambda: {
            "lines": [
                {"text": "hello", "start": "0:00:00.00", "end": "0:00:02.00", "speaker": 1},
            ]
        }
    )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        basic_server._format_openai_response(
            front_data, "diarized_json", "en", 2.0, diarization_enabled=False
        )
    assert exc_info.value.status_code == 400
    assert "diarization" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_openai_rest_rejects_diarized_json_before_expensive_work(monkeypatch):
    basic_server = _import_basic_server(monkeypatch)
    expensive_calls = []

    class UnreadUpload:
        async def read(self, size=-1):
            expensive_calls.append("read")
            return b"encoded audio"

    async def forbidden_conversion(audio_bytes):
        expensive_calls.append("convert")
        return audio_bytes

    class ForbiddenAudioProcessor:
        def __init__(self, **kwargs):
            expensive_calls.append("asr")

    monkeypatch.setattr(basic_server, "_API_TOKEN", None)
    monkeypatch.setattr(basic_server.config, "diarization", False)
    monkeypatch.setattr(basic_server, "_convert_to_pcm", forbidden_conversion)
    monkeypatch.setattr(basic_server, "AudioProcessor", ForbiddenAudioProcessor)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await basic_server.create_transcription(
            request=SimpleNamespace(headers={}),
            file=UnreadUpload(),
            model="ignored",
            language="en",
            prompt="",
            response_format="diarized_json",
            timestamp_granularities=None,
        )

    assert exc_info.value.status_code == 400
    assert expensive_calls == []


@pytest.mark.asyncio
async def test_openai_rest_rejects_unknown_response_format_before_expensive_work(monkeypatch):
    basic_server = _import_basic_server(monkeypatch)
    expensive_calls = []

    class UnreadUpload:
        async def read(self, size=-1):
            expensive_calls.append("read")
            return b"encoded audio"

    async def forbidden_conversion(audio_bytes):
        expensive_calls.append("convert")
        return audio_bytes

    class ForbiddenAudioProcessor:
        def __init__(self, **kwargs):
            expensive_calls.append("asr")

    monkeypatch.setattr(basic_server, "_API_TOKEN", None)
    monkeypatch.setattr(basic_server, "_convert_to_pcm", forbidden_conversion)
    monkeypatch.setattr(basic_server, "AudioProcessor", ForbiddenAudioProcessor)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await basic_server.create_transcription(
            request=SimpleNamespace(headers={}),
            file=UnreadUpload(),
            model="ignored",
            language="en",
            prompt="",
            response_format="xml",
            timestamp_granularities=None,
        )

    assert exc_info.value.status_code == 400
    assert "Unsupported response_format='xml'" in exc_info.value.detail
    assert expensive_calls == []


@pytest.mark.parametrize(
    ("response_format", "expected_body"),
    [
        (
            "json",
            {
                "text": "",
                "usage": {"type": "duration", "seconds": 1},
            },
        ),
        (
            "verbose_json",
            {
                "task": "transcribe",
                "language": "en",
                "duration": 1.0,
                "text": "",
                "words": [],
                "segments": [],
                "usage": {"type": "duration", "seconds": 1},
            },
        ),
        (
            "diarized_json",
            {
                "task": "transcribe",
                "duration": 1.0,
                "text": "",
                "segments": [],
                "usage": {"type": "duration", "seconds": 1},
            },
        ),
        ("text", ""),
        ("srt", ""),
        ("vtt", "WEBVTT\n"),
    ],
)
@pytest.mark.asyncio
async def test_openai_rest_empty_result_uses_requested_formatter(
    monkeypatch,
    response_format,
    expected_body,
):
    from whisperlivekit.timed_objects import FrontData

    basic_server = _import_basic_server(monkeypatch)
    formatted_inputs = []
    processor_kwargs = []
    real_formatter = basic_server._format_openai_response

    class EncodedUpload:
        async def read(self, size=-1):
            return b"encoded audio"

    class EmptyResults:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class EmptyAudioProcessor:
        def __init__(self, **kwargs):
            processor_kwargs.append(kwargs)
            self.is_pcm_input = False

        async def create_tasks(self):
            return EmptyResults()

        async def process_audio(self, audio_bytes):
            pass

        async def cleanup(self):
            pass

    async def fake_conversion(audio_bytes):
        assert audio_bytes == b"encoded audio"
        return b"\0" * 32_000

    def tracking_formatter(*args, **kwargs):
        formatted_inputs.append(args[0])
        return real_formatter(*args, **kwargs)

    monkeypatch.setattr(basic_server, "_API_TOKEN", None)
    monkeypatch.setattr(basic_server.config, "diarization", True)
    monkeypatch.setattr(basic_server, "_convert_to_pcm", fake_conversion)
    monkeypatch.setattr(basic_server, "AudioProcessor", EmptyAudioProcessor)
    monkeypatch.setattr(basic_server, "_format_openai_response", tracking_formatter)

    response = await basic_server.create_transcription(
        request=SimpleNamespace(headers={}),
        file=EncodedUpload(),
        model="ignored",
        language="en",
        prompt="WhisperLiveKit, Qwen3-ASR",
        response_format=response_format,
        timestamp_granularities=None,
    )

    assert len(formatted_inputs) == 1
    assert processor_kwargs[0]["context"] == "WhisperLiveKit, Qwen3-ASR"
    assert isinstance(formatted_inputs[0], FrontData)
    assert formatted_inputs[0].lines == []
    if isinstance(expected_body, dict):
        assert json.loads(response.body) == expected_body
    else:
        assert response.body.decode() == expected_body


def test_openai_rest_json_includes_duration_usage(monkeypatch):
    basic_server = _import_basic_server(monkeypatch)

    front_data = SimpleNamespace(
        to_dict=lambda: {
            "lines": [
                {"text": "hello world", "start": "0:00:00.00", "end": "0:00:02.00", "speaker": 1},
            ]
        }
    )

    payload = basic_server._format_openai_response(front_data, "json", "en", 2.4)

    assert payload == {
        "text": "hello world",
        "usage": {
            "type": "duration",
            "seconds": 2,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["startup", "feed", "cancel", "timeout", "result"])
async def test_rest_request_always_releases_its_pipeline(monkeypatch, failure):
    from fastapi import HTTPException

    from whisperlivekit.timed_objects import FrontData

    server = _import_basic_server(monkeypatch)
    started = asyncio.Event()
    cleaned = []
    collectors = []

    class Processor:
        def __init__(self, **kwargs):
            assert kwargs["pcm_input"] is True

        async def create_tasks(self):
            if failure == "startup":
                raise RuntimeError("startup failed")
            async def results():
                collectors.append(asyncio.current_task())
                if failure == "result":
                    yield FrontData(status="error", error="backend failed")
                else:
                    await asyncio.Event().wait()
            return results()

        async def process_audio(self, data):
            started.set()
            if failure == "feed":
                raise RuntimeError("feed failed")
            if failure != "result":
                await asyncio.Event().wait()

        async def cleanup(self):
            cleaned.append(True)

    async def convert(_): return bytes(3200)
    async def read(size): return b"encoded"
    monkeypatch.setattr(server, "AudioProcessor", Processor)
    monkeypatch.setattr(server, "_convert_to_pcm", convert)
    monkeypatch.setattr(server, "_API_TOKEN", None)
    monkeypatch.setattr(server.config, "rest_timeout", 0.05)
    request = asyncio.create_task(server.create_transcription(
        SimpleNamespace(headers={}), SimpleNamespace(read=read), model="",
        language="en", prompt="", response_format="json", timestamp_granularities=None,
    ))
    if failure == "cancel":
        await asyncio.wait_for(started.wait(), 2)
        request.cancel()
    expected = asyncio.CancelledError if failure == "cancel" else (RuntimeError, HTTPException)
    with pytest.raises(expected) as error:
        await request
    if failure in ("timeout", "result"):
        assert error.value.status_code == (408 if failure == "timeout" else 500)
    assert cleaned == [True]
    assert all(task.done() for task in collectors)


@pytest.mark.asyncio
async def test_audio_conversion_reaps_process_on_cancel_and_output_limit(monkeypatch):
    import shutil

    from fastapi import HTTPException

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required for subprocess lifecycle scenario")
    server = _import_basic_server(monkeypatch)
    real_exec = asyncio.create_subprocess_exec
    processes = []
    async def record_exec(*args, **kwargs):
        process = await real_exec(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(asyncio, "create_subprocess_exec", record_exec)
    # A real WAV conversion, followed by the same request over the decoded cap.
    import io
    import wave
    wav = io.BytesIO()
    with wave.open(wav, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(3200))
    assert len(await server._convert_to_pcm(wav.getvalue())) == 3200
    monkeypatch.setattr(server, "_MAX_AUDIO_BYTES", 100)
    with pytest.raises(HTTPException) as error:
        await server._convert_to_pcm(wav.getvalue())
    assert error.value.status_code == 413
    # Replace the decoder command with a process that never completes. Exercise
    # real pipes and cancellation, without relying on a particular codec's speed.
    async def stalled_exec(*args, **kwargs):
        return await record_exec(sys.executable, "-c", "import time; time.sleep(30)", **kwargs)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", stalled_exec)
    task = asyncio.create_task(server._convert_to_pcm(b"audio"))
    async with asyncio.timeout(2):
        while len(processes) < 3:
            await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(process.returncode is not None for process in processes)


def test_embedded_apps_keep_authentication_and_cors_separate(monkeypatch):
    from fastapi.testclient import TestClient

    from whisperlivekit.config import WhisperLiveKitConfig

    # Server import must ignore the embedding program's argv.
    monkeypatch.setattr(sys, "argv", ["host-program", "--unrelated-option"])
    server = importlib.reload(importlib.import_module("whisperlivekit.basic_server"))
    monkeypatch.setattr(server, "TranscriptionEngine", lambda **kwargs: SimpleNamespace(
        args=kwargs["config"], asr=None,
    ))
    for token, origin in (("first", "https://first.example"), ("second", "https://second.example")):
        app = server.create_app(WhisperLiveKitConfig(api_token=token, cors_origins=origin))
        with TestClient(app, raise_server_exceptions=True) as client:
            response = client.options("/v1/audio/transcriptions", headers={
                "Origin": origin, "Access-Control-Request-Method": "POST",
            })
            assert response.headers["access-control-allow-origin"] == origin
            assert client.options("/v1/audio/transcriptions", headers={
                "Origin": "https://untrusted.example", "Access-Control-Request-Method": "POST",
            }).status_code == 400
            files = {"file": ("audio.wav", b"encoded")}
            assert client.post("/v1/audio/transcriptions", files=files).status_code == 401
            response = client.post("/v1/audio/transcriptions", files=files,
                                   headers={"Authorization": f"Bearer {token}"},
                                   data={"response_format": "xml"})
            assert response.status_code == 400
            assert "Unsupported response_format" in response.json()["detail"]
