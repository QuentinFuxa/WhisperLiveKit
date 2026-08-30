"""Real subprocess teardown and bounded, lossless ingestion under load."""

import asyncio
import io
import shutil
import sys
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from whisperlivekit.audio_processor import SENTINEL, AudioProcessor, get_all_from_queue
from whisperlivekit.core import TranscriptionEngine
from whisperlivekit.ffmpeg_manager import FFmpegManager, FFmpegState
from whisperlivekit.processing_queue import PipelineClosed, PipelineOverloaded, ProcessingQueue


def _wav(pcm):
    out = io.BytesIO()
    with wave.open(out, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(pcm)
    return out.getvalue()


@pytest.mark.asyncio
async def test_ffmpeg_drains_eof_and_reaps_abandoned_or_cancelled_sessions(monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required for real process lifecycle coverage")
    manager = FFmpegManager()
    monkeypatch.setattr(manager, "_STOP_TIMEOUT", .2)
    monkeypatch.setattr(manager, "_TERMINATE_TIMEOUT", .2)
    writer = None
    process = None
    try:
        # Normal EOF must retain every sample, and the manager must restart.
        assert await manager.start()
        process = manager.process
        pcm = np.arange(16000, dtype=np.int16).tobytes()
        assert await manager.write_data(_wav(pcm))
        await manager.close_stdin()
        decoded = bytearray()
        async with asyncio.timeout(5):
            while chunk := await manager.read_data(4096):
                decoded.extend(chunk)
            await manager.stop()
        assert decoded == pcm
        assert process.returncode == 0

        # A client disconnects while the decoder's output pipe is full.
        assert await manager.restart()
        process = manager.process
        writer = asyncio.create_task(manager.write_data(_wav(pcm * 30)))
        assert await asyncio.wait_for(manager.read_data(1), 5)
        await asyncio.sleep(.05)
        await asyncio.wait_for(manager.stop(), 5)
        await writer
        assert process.returncode is not None
        assert manager.process is None

        # Cancellation and a second stop must still reap a child ignoring TERM.
        spawn = asyncio.create_subprocess_exec

        async def stubborn_child(*args, **kwargs):
            return await spawn(
                sys.executable, "-c",
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ready', flush=True); time.sleep(30)",
                **kwargs,
            )

        monkeypatch.setattr(asyncio, "create_subprocess_exec", stubborn_child)
        assert await manager.start()
        process = manager.process
        assert await asyncio.wait_for(manager.read_data(6), 5) == b"ready\n"
        stop = asyncio.create_task(manager.stop())
        await asyncio.sleep(0)
        stop.cancel()
        async with asyncio.timeout(5):
            await manager.stop()
            with pytest.raises(asyncio.CancelledError):
                await stop
        assert process.returncode is not None
        assert manager.process is None
        assert await manager.get_state() == FFmpegState.STOPPED
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.communicate()
        if writer is not None:
            await asyncio.gather(writer, return_exceptions=True)


@pytest.fixture
def engine(monkeypatch):
    # Exercise real ingestion and queues without loading an inference model.
    monkeypatch.setattr(TranscriptionEngine, "_instance", None)
    monkeypatch.setattr(TranscriptionEngine, "_initialized", False)
    shared = TranscriptionEngine(
        transcription=False, vac=False, pcm_input=True,
        max_buffered_audio=.2, backpressure_timeout=2,
    )
    shared.args.transcription = True
    monkeypatch.setattr(
        "whisperlivekit.audio_processor.online_factory",
        lambda *args, **kwargs: SimpleNamespace(asr=SimpleNamespace(sep=" ")),
    )
    return shared


@pytest.mark.asyncio
async def test_bounded_ingestion_preserves_samples_and_isolates_slow_sessions(engine):
    slow = AudioProcessor(transcription_engine=engine)
    fast = AudioProcessor(transcription_engine=engine)
    released = asyncio.Event()
    seen = {slow: [], fast: []}

    async def consume(processor, gate=None):
        if gate:
            await gate.wait()
        while True:
            item = await get_all_from_queue(processor.transcription_queue)
            if item is SENTINEL:
                return
            if isinstance(item, np.ndarray):
                seen[processor].append(item)
            await asyncio.sleep(0)

    pcm = np.arange(24001, dtype=np.int16).tobytes()

    async def feed(processor):
        # The first sample straddles messages; the second message exceeds the
        # queue's capacity, so the producer must split and wait without loss.
        await processor.process_audio(pcm[:1])
        await processor.process_audio(pcm[1:])
        await processor.process_audio(b"")

    consumers = [asyncio.create_task(consume(slow, released)), asyncio.create_task(consume(fast))]
    producers = [asyncio.create_task(feed(slow)), asyncio.create_task(feed(fast))]
    try:
        await asyncio.wait_for(producers[1], 2)
        await asyncio.wait_for(consumers[1], 2)
        assert not producers[0].done()
        assert slow.transcription_queue.queued_samples <= 3200
        assert len(slow.pcm_buffer) <= slow.max_bytes_per_sec
        released.set()
        async with asyncio.timeout(2):
            await producers[0]
            await consumers[0]
        expected = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
        for processor in (slow, fast):
            np.testing.assert_array_equal(np.concatenate(seen[processor]), expected)
            assert processor.total_pcm_samples == len(expected)
            assert processor.transcription_queue.peak_samples <= 3200
            assert not processor.pcm_buffer
    finally:
        for task in producers + consumers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*producers, *consumers, return_exceptions=True)
        await slow.cleanup()
        await fast.cleanup()
    assert slow.metrics.backpressure_wait_s > 0
    assert slow.metrics.peak_queued_audio_s <= .2


@pytest.mark.asyncio
async def test_full_pipeline_timeout_and_cleanup_release_blocked_feeds(engine):
    queue = ProcessingQueue("Audio", max_samples=8)
    await queue.put(np.ones(4))
    await queue.put(np.ones(4))
    large = asyncio.create_task(queue.put(np.ones(8)))
    small = asyncio.create_task(queue.put(np.ones(2)))
    try:
        await asyncio.sleep(0)
        await queue.get()
        queue.task_done()
        await asyncio.wait_for(small, 1)
        assert not large.done()
    finally:
        large.cancel()
        small.cancel()
        await asyncio.gather(large, small, return_exceptions=True)
        queue.close()
    await asyncio.wait_for(queue.join(), 1)

    for timeout in (True, False):
        processor = AudioProcessor(transcription_engine=engine)
        processor.transcription_queue.timeout = .05 if timeout else 30
        producer = asyncio.create_task(processor.process_audio(bytes(32000)))
        try:
            if timeout:
                with pytest.raises(PipelineOverloaded, match="backlog"):
                    await asyncio.wait_for(producer, 2)
                response = await anext(processor.results_formatter())
                assert response.status == "error" and "backlog" in response.error
            else:
                await asyncio.sleep(.01)
                assert not producer.done()
                await processor.cleanup()
                with pytest.raises(PipelineClosed):
                    await asyncio.wait_for(producer, 2)
            assert processor.transcription_queue.closed
            assert processor.transcription_queue.queued_samples == 0
        finally:
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
            await processor.cleanup()

    if shutil.which("ffmpeg"):
        # Encoded input can block on FFmpeg stdin while decoded output waits
        # on a full queue. Overload must release that writer as well.
        processor = AudioProcessor(transcription_engine=engine, pcm_input=False)
        processor.transcription_queue.timeout = .05
        assert await processor.ffmpeg_manager.start()
        process = processor.ffmpeg_manager.process
        reader = asyncio.create_task(processor.ffmpeg_stdout_reader())
        processor.all_tasks_for_cleanup.append(reader)
        producer = asyncio.create_task(processor.process_audio(_wav(bytes(32000 * 30))))
        try:
            async with asyncio.timeout(5):
                await producer
                await reader
            assert processor.overload_error and "backlog" in processor.overload_error
            assert process.returncode is not None
            assert processor.ffmpeg_manager.process is None
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
            await processor.cleanup()


def test_overload_reaches_http_and_websocket_clients(engine, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required for REST audio conversion")
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from whisperlivekit.basic_server import create_app

    engine.args.backpressure_timeout = .05
    processors = []

    async def stalled_inference(processor):
        processors.append(processor)
        await asyncio.Event().wait()

    monkeypatch.setattr(AudioProcessor, "transcription_processor", stalled_inference)
    with TestClient(create_app(engine.config)) as client:
        response = client.post(
            "/v1/audio/transcriptions", files={"file": ("audio.wav", _wav(bytes(32000)), "audio/wav")},
        )
        assert response.status_code == 503, response.text
        assert "backlog" in response.json()["detail"]
        for endpoint, greeting, error_field in (
            ("/asr", "config", "error"),
            ("/v1/listen?encoding=linear16&sample_rate=16000&endpointing=false", "Metadata", "description"),
        ):
            with client.websocket_connect(endpoint) as ws:
                assert ws.receive_json()["type"] == greeting
                ws.send_bytes(bytes(32000))
                while True:
                    message = ws.receive_json()
                    if message.get(error_field):
                        assert "backlog" in message[error_field]
                        break
                with pytest.raises(WebSocketDisconnect) as closed:
                    ws.receive_json()
                assert closed.value.code == 1011
    assert len(processors) == 3
    assert all(task.done() for processor in processors for task in processor.all_tasks_for_cleanup)
