"""Tests for the AlignAtt translation client against an in-process fake sidecar.

The fake speaks protocol v1 (see Alignatt4LLM docs/mt_server_protocol.md)
with a deterministic word-mapper: committed words uppercase into the
accepted text, tail words uppercase into the draft buffer, finals get an
"[F]" marker so quality passes are distinguishable from streamed partials.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

websockets = pytest.importorskip("websockets")
from websockets.asyncio.server import serve as ws_serve  # noqa: E402

from whisperlivekit.timed_objects import ASRToken, HypothesisTail, Translation  # noqa: E402
from whisperlivekit.translation_alignatt import (  # noqa: E402
    AlignAttRemoteEngine,
    AlignAttTranslationClient,
)


class FakeSidecar:
    """Minimal protocol-v1 server on a background thread."""

    def __init__(self):
        self.inits: list[dict] = []
        self.updates: list[dict] = []
        self.port: int | None = None
        self._loop = asyncio.new_event_loop()
        self._stop = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._ready.wait(5), "fake sidecar did not start"

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    def stop(self):
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(timeout=5)

    async def _handler(self, ws):
        async for raw in ws:
            message = json.loads(raw)
            if message.get("type") == "init":
                self.inits.append(message)
                if message.get("target_lang") not in ("de", "zh"):
                    await ws.send(json.dumps({
                        "type": "error", "code": "unsupported_direction",
                        "message": "nope", "supported": ["en-de", "en-zh"],
                    }))
                    await ws.close()
                    return
                await ws.send(json.dumps({
                    "type": "init_ok", "protocol_version": 1,
                    "direction": f"en-{message['target_lang']}",
                    "preset": message.get("preset") or "fake",
                    "model": "fake",
                }))
            elif message.get("type") == "update":
                self.updates.append(message)
                words = [w[0] for w in message.get("words") or []]
                tail_words = [
                    w[0] for w in (message.get("tail") or {}).get("words") or []
                ]
                committed = " ".join(w.upper() for w in words)
                if message.get("is_final"):
                    response = {
                        "type": "translation", "seq": message.get("seq"),
                        "utterance_id": message.get("utterance_id"),
                        "final": True, "forced_final": False,
                        "committed_text": (committed + " [F]").strip(),
                        "committed_delta": "", "buffer_text": "",
                        "covered_source_units": None,
                    }
                else:
                    response = {
                        "type": "translation", "seq": message.get("seq"),
                        "utterance_id": message.get("utterance_id"),
                        "final": False,
                        "committed_text": committed,
                        "committed_delta": "",
                        "buffer_text": " ".join(w.upper() for w in tail_words),
                        "covered_source_units": len(words) or None,
                    }
                await ws.send(json.dumps(response))

    def _run(self):
        asyncio.set_event_loop(self._loop)

        async def main():
            self._stop = asyncio.Event()
            async with ws_serve(self._handler, "127.0.0.1", 0) as server:
                self.port = server.sockets[0].getsockname()[1]
                self._ready.set()
                await self._stop.wait()

        self._loop.run_until_complete(main())


def token(text: str, start: float, end: float) -> ASRToken:
    return ASRToken(start=start, end=end, text=text)


def make_client(sidecar: FakeSidecar, latency: str = "balanced") -> AlignAttTranslationClient:
    engine = AlignAttRemoteEngine(
        url=sidecar.url, source_language="en", preset=None, latency=latency,
    )
    return engine.new_session("de")


@pytest.fixture()
def sidecar():
    server = FakeSidecar()
    yield server
    server.stop()


def test_partial_process_fills_buffer_not_segments(sidecar):
    client = make_client(sidecar)
    client.insert_tokens([token("Hello", 0.0, 0.4), token("world", 0.5, 0.9)])
    validated, buffer = client.process()
    assert validated is None
    assert buffer.text == "HELLO WORLD"
    assert buffer.start == 0.0
    assert buffer.end == 0.9
    update = sidecar.updates[-1]
    assert update["words"][0][0] == "Hello"
    assert update["words"][0][2] == pytest.approx(400.0)
    assert update["is_final"] is False


def test_punctuation_triggers_final_segment_with_span(sidecar):
    client = make_client(sidecar)
    client.insert_tokens([token("Hello", 0.0, 0.4), token("world.", 0.5, 0.9)])
    validated, buffer = client.process()
    assert isinstance(validated, Translation)
    assert validated.text == "HELLO WORLD. [F]"
    assert validated.start == 0.0
    assert validated.end == 0.9
    assert buffer.text == ""
    assert sidecar.updates[-1]["is_final"] is True
    # next utterance opens cleanly and spans start after the previous segment
    client.insert_tokens([token("Again", 2.0, 2.4)])
    validated2, buffer2 = client.process()
    assert validated2 is None
    assert buffer2.start == 0.9  # continues from the previous translation end
    assert buffer2.text == "AGAIN"


def test_tail_words_are_sent_untimestamped_and_buffered(sidecar):
    client = make_client(sidecar)  # balanced: tail on
    assert client.wants_hypothesis_tail
    client.insert_tokens([
        token("Hello", 0.0, 0.4),
        HypothesisTail(start=0.5, end=1.8, text="how are"),
    ])
    validated, buffer = client.process()
    assert validated is None
    update = sidecar.updates[-1]
    assert [w[0] for w in update["tail"]["words"]] == ["how", "are"]
    assert all(w[1] is None and w[2] is None for w in update["tail"]["words"])
    assert update["clock_ms"] == pytest.approx(1800.0)
    assert buffer.text == "HELLO"  # accepted only; the draft stays server-side


def test_quality_latency_preset_disables_tail(sidecar):
    client = make_client(sidecar, latency="quality")
    assert client.wants_hypothesis_tail is False


def test_pacing_skips_tail_only_updates_but_not_commits(sidecar):
    client = make_client(sidecar)
    client.insert_tokens([token("One", 0.0, 0.4)])
    client.process()
    n_after_first = len(sidecar.updates)
    # tail-only refresh right after a call: paced out
    client.insert_tokens([HypothesisTail(start=0.5, end=1.0, text="two")])
    client.process()
    assert len(sidecar.updates) == n_after_first
    # a new commit bypasses pacing (server-side release is cheap)
    client.insert_tokens([token("two", 0.5, 1.0)])
    client.process()
    assert len(sidecar.updates) == n_after_first + 1


def test_validate_buffer_and_reset_returns_streamed_partial(sidecar):
    client = make_client(sidecar)
    client.insert_tokens([token("Hello", 0.0, 0.4), token("world", 0.5, 0.9)])
    client.process()
    validated, buffer = client.validate_buffer_and_reset()
    assert validated.text == "HELLO WORLD"
    assert validated.start == 0.0
    assert validated.end == 0.9
    assert buffer.text == ''
    # the utterance rolls server-side on the next process()
    client.insert_tokens([token("Next", 2.0, 2.4)])
    client.process()
    finals = [u for u in sidecar.updates if u["is_final"]]
    assert len(finals) == 1
    assert [w[0] for w in finals[0]["words"]] == ["Hello", "world"]


def test_translation_segments_are_monotone_nonoverlapping(sidecar):
    client = make_client(sidecar)
    spans = []
    for i, sentence in enumerate([
        ["Sentence", "one."], ["Sentence", "two."], ["And", "three."],
    ]):
        base = i * 2.0
        client.insert_tokens([
            token(w, base + j * 0.5, base + j * 0.5 + 0.4)
            for j, w in enumerate(sentence)
        ])
        validated, _ = client.process()
        assert validated is not None
        spans.append((validated.start, validated.end))
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert e1 <= s2 or abs(s2 - e1) < 1e-9
        assert s2 >= e1


def test_sidecar_down_degrades_gracefully(caplog):
    engine = AlignAttRemoteEngine(
        url="ws://127.0.0.1:9", source_language="en", preset=None, latency="balanced",
    )
    client = engine.new_session("de")
    client.insert_tokens([token("Hello", 0.0, 0.4)])
    with caplog.at_level("WARNING"):
        validated, buffer = client.process()
    assert validated is None
    assert buffer.text == ""
    warnings = [r for r in caplog.records if "sidecar unavailable" in r.message]
    assert len(warnings) == 1
    # second failure does not re-warn
    with caplog.at_level("WARNING"):
        client.process()
    warnings = [r for r in caplog.records if "sidecar unavailable" in r.message]
    assert len(warnings) == 1


def test_reconnect_resumes_with_accepted_prefix(sidecar):
    client = make_client(sidecar)
    client.insert_tokens([token("Hello", 0.0, 0.4), token("world", 0.5, 0.9)])
    client.process()
    assert client._emitted_partial == "HELLO WORLD"
    # sidecar dies mid-session
    sidecar.stop()
    client.insert_tokens([token("again", 1.0, 1.4)])
    validated, buffer = client.process()
    assert validated is None
    assert buffer.text == "HELLO WORLD"  # what was shown stays shown
    # a new sidecar comes up at the same port: not guaranteed in tests, so
    # simulate recovery by pointing the engine at a fresh server
    replacement = FakeSidecar()
    try:
        client.engine.url = replacement.url
        client._retry_at = 0.0
        client.insert_tokens([token("more", 1.5, 1.9)])
        client.process()
        assert replacement.inits, "client did not reconnect"
        assert replacement.inits[0].get("accepted_target_prefix") == "HELLO WORLD"
    finally:
        replacement.stop()


def test_unsupported_direction_warns_and_stays_empty(sidecar, caplog):
    engine = AlignAttRemoteEngine(
        url=sidecar.url, source_language="en", preset=None, latency="balanced",
    )
    client = engine.new_session("xx")
    client.insert_tokens([token("Hello", 0.0, 0.4)])
    with caplog.at_level("WARNING"):
        validated, buffer = client.process()
    assert validated is None
    assert buffer.text == ""
    assert any("unsupported_direction" in r.message for r in caplog.records)


def test_factory_routes_alignatt_engine():
    from argparse import Namespace

    from whisperlivekit.core import online_translation_factory

    engine = AlignAttRemoteEngine(
        url="ws://127.0.0.1:9", source_language="en", preset=None, latency="balanced",
    )
    args = Namespace(target_language="de", lan="en")
    client = online_translation_factory(args, engine)
    assert isinstance(client, AlignAttTranslationClient)
    assert client.target_language == "de"

    from whisperlivekit.translation import session_translation_factory

    session_client = session_translation_factory(args, engine, "zh")
    assert isinstance(session_client, AlignAttTranslationClient)
    assert session_client.target_language == "zh"


def test_translation_processor_plumbing_with_fake_sidecar(sidecar):
    """Drive AudioProcessor.translation_processor as a bound coroutine over a
    real queue: tokens and tails in, Translation segments and buffer out."""
    from types import SimpleNamespace

    from whisperlivekit.audio_processor import SENTINEL, AudioProcessor
    from whisperlivekit.timed_objects import State

    processor = SimpleNamespace(
        translation_queue=asyncio.Queue(),
        translation=make_client(sidecar),
        state=State(),
        lock=asyncio.Lock(),
    )

    async def scenario():
        task = asyncio.create_task(
            AudioProcessor.translation_processor(processor)
        )
        await processor.translation_queue.put(token("Hello", 0.0, 0.4))
        await processor.translation_queue.put(token("world.", 0.5, 0.9))
        await processor.translation_queue.put(
            HypothesisTail(start=1.0, end=1.5, text="next words")
        )
        await asyncio.sleep(0.5)
        await processor.translation_queue.put(SENTINEL)
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(scenario())
    assert processor.state.new_translation, "no validated translation produced"
    assert processor.state.new_translation[0].text == "HELLO WORLD. [F]"
