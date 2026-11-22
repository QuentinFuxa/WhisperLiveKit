"""The drafting predicate against each MT backend's own buffer.

`drafts_provisional_text` decides whether an audio-to-translation run reports a
provisional lead. That is a claim about the second element each backend's
`process()` returns, so every case here drives the backend and reads that
buffer, then asserts the predicate answers the same way. A backend whose buffer
changes without a matching answer fails here, and so does a backend added to
`--translation-backend` without a case.
"""

from __future__ import annotations

import pytest


def test_an_alignatt_sidecar_partial_is_a_draft():
    """The sidecar's committed text arrives as the buffer, so alignatt drafts."""
    from tests.test_translation_alignatt import FakeSidecar, make_client, token
    from whisperlivekit.benchmark.metrics import drafts_provisional_text

    server = FakeSidecar()
    try:
        client = make_client(server)
        client.insert_tokens([token("Hello", 0.0, 0.4), token("world", 0.5, 0.9)])
        validated, buffer = client.process()
    finally:
        server.stop()

    assert validated is None
    assert buffer.text == "HELLO WORLD"
    assert drafts_provisional_text({"translation_backend": "alignatt"})


def test_the_segment_mlx_backend_returns_an_empty_buffer(monkeypatch):
    """The segment path translates completed sentences and drafts nothing."""
    from whisperlivekit.benchmark.metrics import drafts_provisional_text
    from whisperlivekit.timed_objects import ASRToken
    from whisperlivekit.translation_mlx_llm_mt import MlxLlmTranslation

    client = MlxLlmTranslation(source_language="en", target_language="zh", warmup=False)
    monkeypatch.setattr(client, "_translate_text", lambda text: "你好")
    client.insert_tokens([ASRToken(0, 1, "Hello.")])
    validated, buffer = client.process()

    assert validated is not None and validated.text == "你好"
    assert not buffer.text
    # Both spellings of the MLX segment path resolve to that same answer.
    for backend in ("mlx-llm-mt", "hunyuan-mlx"):
        assert not drafts_provisional_text({"translation_backend": backend})


def test_the_simultaneous_mlx_backend_returns_a_drafted_buffer(tmp_path, monkeypatch):
    """Drafting is the simultaneous switch on one class, not a second backend."""
    from tests.test_translation_mlx import _test_calibration
    from whisperlivekit.benchmark.metrics import drafts_provisional_text
    from whisperlivekit.timed_objects import ASRToken, HypothesisTail
    from whisperlivekit.translation_mlx_llm_mt_simul import MlxLlmTranslationSimul

    engine = MlxLlmTranslationSimul(source_language="en", target_language="zh", warmup=False,
                                   calibration_file=_test_calibration(tmp_path))
    monkeypatch.setattr(engine, "_translate_simul", lambda source, committed: "您好")
    engine.insert_tokens([ASRToken(0, 1, "Hello"), HypothesisTail(1, 2, "world")])
    validated, buffer = engine.process()

    assert validated is None
    assert buffer.text == "您好"
    for backend in ("mlx-llm-mt", "hunyuan-mlx"):
        assert drafts_provisional_text({"translation_backend": backend,
                                        "mlx_llm_mt_simultaneous": True})


@pytest.mark.asyncio
async def test_the_nllb_backend_publishes_its_draft_buffer():
    """The default backend drafts: nllw returns its uncommitted target tokens.

    The stand-in below is the surface `run_translation` calls and the shape
    `nllw.OnlineTranslation.process()` returns (nllw 0.1.6, `translation.py:51-72`):
    a `TimedText` of the stable target prefix and, as the second element, a
    `TimedText` of the target tokens outside that prefix. It stands in for the
    optional `nllw` extra, which the tests cannot load without its NLLB
    checkpoint; the publisher it is driven through is the real one.
    """
    import asyncio

    from whisperlivekit.audio_processor import SENTINEL
    from whisperlivekit.benchmark.metrics import drafts_provisional_text
    from whisperlivekit.timed_objects import ASRToken, State, TimedText
    from whisperlivekit.translation_processor import run_translation

    class OnlineTranslation:
        def insert_tokens(self, tokens):
            pass

        def insert_silence(self, duration):
            pass

        def process(self):
            stable = TimedText(start=0.0, end=1.0, text="")
            buffer = TimedText(start=0.0, end=1.0, text="欢迎来到我的")
            return stable, buffer

        def validate_buffer_and_reset(self):
            return TimedText(start=0.0, end=1.0, text="欢迎来到我的"), TimedText()

        def finish(self):
            return self.validate_buffer_and_reset()

    # The pipeline publishes the second element of process() as the buffer the
    # harness reads, so record what it published rather than the last value.
    published = []

    class RecordingState(State):
        def __setattr__(self, name, value):
            if name == "new_translation_buffer" and getattr(value, "text", ""):
                published.append(value.text)
            super().__setattr__(name, value)

    queue, lock, state = asyncio.Queue(), asyncio.Lock(), RecordingState()
    for event in [ASRToken(0, 1, "Welcome to my"), SENTINEL]:
        await queue.put(event)
    await asyncio.wait_for(run_translation(queue, OnlineTranslation(), state, lock), 5)

    assert published[0] == "欢迎来到我的"
    for backend in ("nllb", "nllw"):
        assert drafts_provisional_text({"translation_backend": backend})


def test_every_translation_backend_the_cli_accepts_is_classified():
    """A backend added to `--translation-backend` without a case fails here."""
    from whisperlivekit.benchmark.metrics import drafts_provisional_text
    from whisperlivekit.parse_args import build_parser

    parser = build_parser()
    choices = set(parser._option_string_actions["--translation-backend"].choices)

    # The answer each case above establishes by driving that backend.
    drilled = {"nllb": True, "alignatt": True, "mlx-llm-mt": False, "hunyuan-mlx": False}
    assert choices == set(drilled)
    for backend, drafts in drilled.items():
        assert drafts_provisional_text({"translation_backend": backend}) is drafts
