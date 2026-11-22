"""The drafting predicate against each MT backend's own buffer.

`drafts_provisional_text` decides whether an audio-to-translation run reports a
provisional lead. That is a claim about the second element each backend's
`process()` returns, so every case here drives the backend and reads that
buffer, then asserts the predicate answers the same way. A backend whose buffer
changes without a matching answer fails here, and so does a backend added to
`--translation-backend` without a case.

Driving happens in the module-scoped `driven` fixture that every case requests,
and each driver records what it established in `_DRIVEN`; the completeness case
reads that record. So the classified set is what running a backend produced, and
it does not depend on the order the cases run in: alone, under `-k`, under
`--lf`, or split across workers.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, NamedTuple

import pytest

# The answer each driver established by running a backend, keyed by the name
# `--translation-backend` accepts: the drafting answer for a run that names only
# that backend.
_DRIVEN: Dict[str, bool] = {}


def drove(backend: str, drafts: bool) -> None:
    """Record the answer driving this backend's buffer established."""
    _DRIVEN[backend] = drafts


class Driven(NamedTuple):
    """What one `process()`-style backend returned: validated tokens and buffer."""

    validated: Any
    buffer: Any


def drive_alignatt() -> Driven:
    """Drive the sidecar client, whose committed text arrives as the buffer."""
    from tests.test_translation_alignatt import FakeSidecar, make_client, token

    server = FakeSidecar()
    try:
        client = make_client(server)
        client.insert_tokens([token("Hello", 0.0, 0.4), token("world", 0.5, 0.9)])
        validated, buffer = client.process()
    finally:
        server.stop()
    drove("alignatt", True)
    return Driven(validated, buffer)


def drive_mlx_segment(monkeypatch) -> Driven:
    """Drive the segment MLX path, which translates completed sentences only."""
    from whisperlivekit.timed_objects import ASRToken
    from whisperlivekit.translation_mlx_llm_mt import MlxLlmTranslation

    client = MlxLlmTranslation(source_language="en", target_language="zh", warmup=False)
    monkeypatch.setattr(client, "_translate_text", lambda text: "你好")
    client.insert_tokens([ASRToken(0, 1, "Hello.")])
    validated, buffer = client.process()
    # Both spellings of the MLX segment path resolve to that same answer.
    for backend in ("mlx-llm-mt", "hunyuan-mlx"):
        drove(backend, False)
    return Driven(validated, buffer)


def drive_mlx_simul(tmp_path, monkeypatch) -> Driven:
    """Drive the simultaneous switch on the same class, which does draft."""
    from tests.test_translation_mlx import _test_calibration
    from whisperlivekit.timed_objects import ASRToken, HypothesisTail
    from whisperlivekit.translation_mlx_llm_mt_simul import MlxLlmTranslationSimul

    engine = MlxLlmTranslationSimul(source_language="en", target_language="zh", warmup=False,
                                   calibration_file=_test_calibration(tmp_path))
    monkeypatch.setattr(engine, "_translate_simul", lambda source, committed: "您好")
    engine.insert_tokens([ASRToken(0, 1, "Hello"), HypothesisTail(1, 2, "world")])
    validated, buffer = engine.process()
    # The switch is a second arrangement on the same backend, so the record keeps
    # the bare name's answer and the switch is asserted in its own case.
    return Driven(validated, buffer)


async def drive_nllb() -> list:
    """Drive the publisher and return the buffers it published, in order.

    The stand-in below is the surface `run_translation` calls and the shape
    `nllw.OnlineTranslation.process()` returns (nllw 0.1.6, `translation.py:51-72`):
    a `TimedText` of the stable target prefix and, as the second element, a
    `TimedText` of the target tokens outside that prefix. It stands in for the
    optional `nllw` extra, which the tests cannot load without its NLLB
    checkpoint; the publisher it is driven through is the real one.
    """
    from whisperlivekit.audio_processor import SENTINEL
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
    # `nllw` is the library name behind this backend, not a second backend: only
    # the name the flag accepts is recorded.
    drove("nllb", True)
    return published


@pytest.fixture(scope="module")
def driven(tmp_path_factory):
    """Drive every backend once for this module, whichever case runs first.

    Module scope, because the record the completeness case reads has to be full
    before that case runs even when it is the only one selected.
    """
    calibration_dir = tmp_path_factory.mktemp("simul_calibration")
    with pytest.MonkeyPatch.context() as monkeypatch:
        return {
            "alignatt": drive_alignatt(),
            "mlx_segment": drive_mlx_segment(monkeypatch),
            "mlx_simul": drive_mlx_simul(calibration_dir, monkeypatch),
            "nllb": asyncio.run(drive_nllb()),
        }


def test_an_alignatt_sidecar_partial_is_a_draft(driven):
    """The sidecar's committed text arrives as the buffer, so alignatt drafts."""
    from whisperlivekit.benchmark.metrics import drafts_provisional_text

    validated, buffer = driven["alignatt"]
    assert validated is None
    assert buffer.text == "HELLO WORLD"
    assert drafts_provisional_text({"translation_backend": "alignatt"})


def test_the_segment_mlx_backend_returns_an_empty_buffer(driven):
    """The segment path translates completed sentences and drafts nothing."""
    from whisperlivekit.benchmark.metrics import drafts_provisional_text

    validated, buffer = driven["mlx_segment"]
    assert validated is not None and validated.text == "你好"
    assert not buffer.text
    for backend in ("mlx-llm-mt", "hunyuan-mlx"):
        assert not drafts_provisional_text({"translation_backend": backend})


def test_the_simultaneous_mlx_backend_returns_a_drafted_buffer(driven):
    """Drafting is the simultaneous switch on one class, not a second backend."""
    from whisperlivekit.benchmark.metrics import drafts_provisional_text

    validated, buffer = driven["mlx_simul"]
    assert validated is None
    assert buffer.text == "您好"
    for backend in ("mlx-llm-mt", "hunyuan-mlx"):
        assert drafts_provisional_text({"translation_backend": backend,
                                        "mlx_llm_mt_simultaneous": True})


def test_the_nllb_backend_publishes_its_draft_buffer(driven):
    """The default backend drafts: nllw returns its uncommitted target tokens."""
    from whisperlivekit.benchmark.metrics import drafts_provisional_text

    assert driven["nllb"][0] == "欢迎来到我的"
    for backend in ("nllb", "nllw"):
        assert drafts_provisional_text({"translation_backend": backend})


def test_every_translation_backend_the_cli_accepts_has_a_case_that_drives_it(driven):
    """A backend added to `--translation-backend` without a case fails here.

    The answers come from `_DRIVEN`, which only the drivers above fill by running
    a backend, so an entry added to this check without one to drive it leaves
    the backend unrecorded and fails rather than passing as a listed name.
    """
    from whisperlivekit.benchmark.metrics import drafts_provisional_text
    from whisperlivekit.parse_args import build_parser

    parser = build_parser()
    choices = set(parser._option_string_actions["--translation-backend"].choices)
    undriven = choices - set(_DRIVEN)
    assert not undriven, f"--translation-backend accepts {sorted(undriven)} with no case that drives it"
    assert set(_DRIVEN) <= choices
    for backend, drafts in _DRIVEN.items():
        assert drafts_provisional_text({"translation_backend": backend}) is drafts
