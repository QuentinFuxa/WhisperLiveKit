import asyncio
import importlib
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _server_metrics_class():
    try:
        from whisperlivekit.server_metrics import ServerMetrics

        return ServerMetrics
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
        module_path = Path(__file__).parents[1] / "whisperlivekit" / "server_metrics.py"
        spec = importlib.util.spec_from_file_location("server_metrics_direct", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.ServerMetrics


def _import_basic_server(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("numpy")
    monkeypatch.setattr(sys, "argv", ["wlk"])
    import whisperlivekit.basic_server as basic_server

    return importlib.reload(basic_server)


async def _wait_for_snapshot(metrics, predicate):
    for _ in range(20):
        snapshot = await metrics.snapshot()
        if predicate(snapshot):
            return snapshot
        await asyncio.sleep(0.01)
    return await metrics.snapshot()


def _front_data(lines):
    return SimpleNamespace(to_dict=lambda: {"lines": lines})


def test_metrics_route_returns_openmetrics(monkeypatch):
    testclient = pytest.importorskip("fastapi.testclient")
    basic_server = _import_basic_server(monkeypatch)
    client = testclient.TestClient(basic_server.app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"] == (
        "application/openmetrics-text; version=1.0.0; charset=utf-8"
    )
    assert "# TYPE vllm:num_requests_waiting gauge" in response.text
    assert re.search(r"^vllm:num_requests_waiting \d+$", response.text, re.MULTILINE)
    assert len(re.findall(r"^vllm:num_requests_waiting \d+$", response.text, re.MULTILINE)) == 1
    assert response.text.endswith("# EOF\n")


def test_metrics_unlimited_mode_reports_no_waiting_requests():
    asyncio.run(_test_metrics_unlimited_mode_reports_no_waiting_requests())


async def _test_metrics_unlimited_mode_reports_no_waiting_requests():
    ServerMetrics = _server_metrics_class()
    metrics = ServerMetrics()

    async with metrics.transcription_admission():
        snapshot = await metrics.snapshot()

    assert snapshot["waiting_requests"] == 0


def test_metrics_capacity_tracks_waiting_and_restores_on_success():
    asyncio.run(_test_metrics_capacity_tracks_waiting_and_restores_on_success())


async def _test_metrics_capacity_tracks_waiting_and_restores_on_success():
    ServerMetrics = _server_metrics_class()
    metrics = ServerMetrics(max_concurrent_transcriptions=1)
    release_holder = asyncio.Event()
    holder_entered = asyncio.Event()

    async def hold_capacity():
        async with metrics.transcription_admission():
            holder_entered.set()
            await release_holder.wait()

    async def wait_for_capacity():
        async with metrics.transcription_admission():
            pass

    holder_task = asyncio.create_task(hold_capacity())
    await holder_entered.wait()
    waiter_task = asyncio.create_task(wait_for_capacity())

    snapshot = await _wait_for_snapshot(
        metrics,
        lambda current: current["waiting_requests"] == 1 and current["active_requests"] == 1,
    )
    assert snapshot == {"waiting_requests": 1, "active_requests": 1}

    release_holder.set()
    await asyncio.gather(holder_task, waiter_task)

    assert await metrics.snapshot() == {"waiting_requests": 0, "active_requests": 0}


def test_metrics_capacity_restores_on_exception():
    asyncio.run(_test_metrics_capacity_restores_on_exception())


async def _test_metrics_capacity_restores_on_exception():
    ServerMetrics = _server_metrics_class()
    metrics = ServerMetrics(max_concurrent_transcriptions=1)

    with pytest.raises(RuntimeError):
        async with metrics.transcription_admission():
            raise RuntimeError("boom")

    assert await metrics.snapshot() == {"waiting_requests": 0, "active_requests": 0}


def test_openai_json_response_includes_usage(monkeypatch):
    basic_server = _import_basic_server(monkeypatch)
    front_data = _front_data([
        {"speaker": 1, "text": "hello world", "start": "0:00:00", "end": "0:00:02"},
    ])

    payload = basic_server._format_openai_response(front_data, "json", "en", 2.345)

    assert payload == {
        "text": "hello world",
        "usage": {
            "seconds": 2.35,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }


def test_openai_verbose_json_response_preserves_duration_and_adds_usage(monkeypatch):
    basic_server = _import_basic_server(monkeypatch)
    front_data = _front_data([
        {"speaker": 1, "text": "hello world", "start": "0:00:00", "end": "0:00:02"},
    ])

    payload = basic_server._format_openai_response(front_data, "verbose_json", "en", 2.345)

    assert payload["duration"] == 2.35
    assert payload["usage"]["seconds"] == 2.35
    assert payload["text"] == "hello world"
    assert payload["segments"]


@pytest.mark.parametrize("response_format", ["text", "srt", "vtt"])
def test_openai_text_srt_vtt_responses_remain_strings(monkeypatch, response_format):
    basic_server = _import_basic_server(monkeypatch)
    front_data = _front_data([
        {"speaker": 1, "text": "hello world", "start": "0:00:00", "end": "0:00:02"},
    ])

    payload = basic_server._format_openai_response(front_data, response_format, "en", 2.345)

    assert isinstance(payload, str)


def test_openai_empty_response_includes_usage_seconds(monkeypatch):
    basic_server = _import_basic_server(monkeypatch)

    payload = basic_server._format_empty_openai_response(30.004)

    assert payload == {
        "text": "",
        "usage": {
            "seconds": 30.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }
