import asyncio

from src.stt_core.redis_io import RedisPublisher


class _FakeRedisClient:
    def __init__(self) -> None:
        self.calls = []

    async def xadd(self, stream, payload, maxlen=None, approximate=None):
        self.calls.append((stream, payload, maxlen, approximate))
        return "1-0"


def test_redis_publisher(monkeypatch) -> None:
    fake_client = _FakeRedisClient()

    def _fake_from_url(url, decode_responses=True):
        assert url == "redis://example:6379/0"
        assert decode_responses is True
        return fake_client

    monkeypatch.setattr("src.stt_core.redis_io.from_url", _fake_from_url)

    publisher = RedisPublisher("redis://example:6379/0", "daymind:transcripts")
    payload = {"text": "hello"}
    result = asyncio.run(publisher.publish(payload))

    assert result == "1-0"
    assert fake_client.calls
    stream, recorded_payload, maxlen, approximate = fake_client.calls[0]
    assert stream == "daymind:transcripts"
    assert recorded_payload == payload
    assert maxlen == 10000
    assert approximate is True
