import json
from pathlib import Path

import pytest

from src.stt_core import livekit_runner
from src.stt_core.config import STTConfig


class FakeSegment:
    text = "toto je test přepisu řeči"
    lang = "cs"
    start = 0.0
    end = 3.0
    confidence = 0.92


@pytest.mark.asyncio
async def test_full_pipeline(monkeypatch, tmp_path):
    audio_path = Path("tests/assets/sample_cs_noisy.wav")
    assert audio_path.exists()

    class FakeLiveKit:
        def __init__(self, *args, **kwargs):
            self.audio = audio_path.read_bytes()

        async def listen(self):
            yield FakeSegment()

    class DummyPublisher:
        def __init__(self):
            self.data = []

        async def publish(self, payload):
            self.data.append(payload)
            return "ok"

    dummy_pub = DummyPublisher()

    cfg = STTConfig(
        language="cs",
        buffer_path=str(tmp_path / "out.jsonl"),
    )

    monkeypatch.setattr(livekit_runner, "LiveKit", FakeLiveKit)
    monkeypatch.setattr(livekit_runner, "RedisPublisher", lambda *args, **kwargs: dummy_pub)
    monkeypatch.setattr(livekit_runner, "STTConfig", lambda: cfg)

    await livekit_runner.run_realtime_stt()

    assert dummy_pub.data, "Redis publisher should capture payloads"
    assert "test" in dummy_pub.data[0]["text"].lower()

    assert Path(cfg.buffer_path).exists()
    with open(cfg.buffer_path, "r", encoding="utf-8") as fh:
        last_line = json.loads(fh.readlines()[-1])

    assert last_line["lang"] == "cs"
    assert last_line["start"] == pytest.approx(0.0)
    assert last_line["end"] == pytest.approx(3.0)
    assert "ts" in last_line
