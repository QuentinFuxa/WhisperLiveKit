"""Standardized caption event stream — the contract between ASR+MT generation
and the display/test layers.

Parallel tap (spike): the pipeline emits these events alongside building
``FrontData``. ``FrontData`` (the web UI wire protocol) is untouched — it
stays built by ``audio_processor``. The event stream is the testable seam:
it carries the caption event types so generation can be tested for coherence
independently of how the display renders it.

Event types (named by what the viewer sees, not which subsystem produced it,
aligning with the overlay's partial/preview/translation vocabulary and the
FrontData buffer_transcription/buffer_translation fields):

  - ``transcription_provisional`` : the unstable ASR tail (rolling hypothesis,
    not committed) — what the viewer sees as the in-progress source line.
  - ``transcription_final``   : committed ASR tokens (a finalized segment).
  - ``translation_provisional``: provisional translation (AlignAtt release
    against committed source; held target is NOT released yet).
    ``fresh=True`` if this is a freshly drafted translation (a new MT forward
    pass); ``fresh=False`` if it is replayed from the cached attention (a
    free release — no MT call). Exposes the hysteresis distinction (call
    count vs emission count).
  - ``translation_final``     : finalized translation (quality pass at a
    boundary).

The display layer derives its own state from this stream; the web UI keeps
using ``FrontData`` snapshots unchanged.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol


@dataclass
class CaptionEvent:
    """One event in the caption stream."""
    t: float          # wall clock (seconds since stream start)
    audio_t: float    # audio position (seconds)
    type: str         # transcription_provisional | transcription_final |
                      # translation_provisional | translation_final
    text: str
    # translation_provisional only: what AlignAtt released against (committed source)
    committed: str = ""
    # translation_provisional only: the full source the MT saw (committed + tail)
    source: str = ""
    # translation_provisional only: True if a freshly drafted translation (a
    # new MT forward pass); False if replayed from cached attention (a free
    # release — no MT call). Exposes the hysteresis lever (call count vs
    # emission count).
    fresh: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class EventSink(Protocol):
    """Anything that consumes caption events."""
    def emit(self, event: CaptionEvent) -> None: ...


@dataclass
class EventLog:
    """In-memory event log + JSONL save. Use as the sink for capture."""
    events: List[CaptionEvent] = field(default_factory=list)

    def emit(self, event: CaptionEvent) -> None:
        self.events.append(event)

    def save(self, path: str) -> None:
        Path(path).write_text(
            "\n".join(json.dumps(e.to_dict(), ensure_ascii=False) for e in self.events)
            + "\n"
        )

    @classmethod
    def load(cls, path: str) -> "EventLog":
        log = cls()
        for line in Path(path).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                log.events.append(CaptionEvent(**d))
        return log


class FanOutSink:
    """Dispatch each event to multiple sinks (e.g. DisplayAdapter + EventLog)."""
    def __init__(self, sinks: List[EventSink]) -> None:
        self.sinks = list(sinks)

    def emit(self, event: CaptionEvent) -> None:
        for s in self.sinks:
            s.emit(event)


class EventTap:
    """Tap attached to the ASR+MT driver loop.

    The driver calls the emit methods at the points where each event is
    produced; the tap forwards to its sink (if any). Kept stateless so a
    no-op tap (sink=None) is zero-cost and the pipeline runs unchanged.
    """
    def __init__(self, sink: Optional[EventSink] = None, clock=None):
        self._sink = sink
        self._clock = clock  # callable returning wall seconds; None = perf_counter

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        import time
        return time.perf_counter()

    def transcription_provisional(self, audio_t: float, text: str) -> None:
        if self._sink is None or not text or not text.strip():
            return
        self._sink.emit(CaptionEvent(self._now(), audio_t, "transcription_provisional", text))

    def transcription_final(self, audio_t: float, text: str) -> None:
        if self._sink is None or not text or not text.strip():
            return
        self._sink.emit(CaptionEvent(self._now(), audio_t, "transcription_final", text))

    def translation_provisional(self, audio_t: float, text: str, committed: str = "", source: str = "", fresh: bool = False) -> None:
        if self._sink is None or not text or not text.strip():
            return
        self._sink.emit(CaptionEvent(self._now(), audio_t, "translation_provisional", text, committed, source, fresh))

    def translation_final(self, audio_t: float, text: str) -> None:
        if self._sink is None or not text or not text.strip():
            return
        self._sink.emit(CaptionEvent(self._now(), audio_t, "translation_final", text))
