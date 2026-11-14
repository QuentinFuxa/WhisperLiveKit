"""Standardized caption event stream — the contract between ASR+MT generation
and the display/test layers.

The pipeline emits these events alongside building ``FrontData``. ``FrontData``
(the web UI wire protocol) is untouched — it stays built by ``audio_processor``.
The event stream is the testable seam: it carries the caption event types so
generation can be tested for coherence independently of how the display renders
it.

Event types (named by what the viewer sees, not which subsystem produced it,
aligning with the FrontData partial/preview/translation vocabulary and the
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

Recording the stream is opt-in (``--event-log``); display views and tests
consume the same stream. The web UI keeps using ``FrontData`` snapshots
unchanged.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

# Only translation_provisional compares its draft against source text; the
# other event types carry no source columns (see CaptionEvent.to_dict).
_SOURCE_COLUMNS = ("committed", "source", "fresh")


@dataclass
class CaptionEvent:
    """One event in the caption stream."""
    t: float          # seconds since this stream started (monotonic)
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
        """The recorded payload. ``committed``/``source``/``fresh`` describe a
        draft's source gating, so they are dropped from the three event types
        that have no draft."""
        row = asdict(self)
        if self.type != "translation_provisional":
            for column in _SOURCE_COLUMNS:
                del row[column]
        return row


class EventSink(Protocol):
    """Anything that consumes caption events."""
    def emit(self, event: CaptionEvent) -> None: ...


def _open_event_log(path: Path):
    """Open the JSONL log, reporting an unusable path as a config error.

    A bad ``--event-log`` path is a startup problem: ``ValueError`` is what the
    server's session factories already report to the client, whereas the raw
    ``OSError`` would escape as an unexplained per-connection failure.
    """
    try:
        return path.open("a", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"--event-log path {path} is not writable: {exc}") from exc


def validate_event_log_path(path: Optional[str]) -> None:
    """Fail at startup rather than on the first connection when the log path
    cannot be written. Sessions each open their own handle, so without this
    check the path would first be touched per session."""
    if not path:
        return
    _open_event_log(Path(path)).close()


class EventLog:
    """JSONL sink: one line per event, appended and flushed as it is emitted.

    The file is opened in append mode, so a restart adds to an existing log.
    Each line carries this sink's short ``session`` id: sessions that share one
    path interleave their lines, and the id is what separates them. Each line
    is written by a single ``emit()`` call, so the file stays parseable.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.session = uuid.uuid4().hex[:8]
        self._fh = _open_event_log(self.path)

    def emit(self, event: CaptionEvent) -> None:
        if self._fh is None:  # closed by session cleanup
            return
        row = {"session": self.session, **event.to_dict()}
        try:
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fh.flush()
        except OSError as exc:
            # A recording instrument must not end a live session: report once
            # and stop writing.
            logger.warning("Caption event log %s disabled after write error: %s", self.path, exc)
            self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class EventTap:
    """Tap attached to the ASR+MT driver loop.

    The driver calls the emit methods at the points where each event is
    produced; the tap forwards to its sink (if any). A tap with no sink is
    inert: ``active`` is False, callers skip it, and the pipeline runs
    unchanged. ``t`` is measured from the tap's creation, so a stream's
    timestamps start at zero regardless of the process clock.
    """
    def __init__(self, sink: Optional[EventSink] = None, clock=None):
        self._sink = sink
        self._clock = clock  # callable returning monotonic seconds; None = perf_counter
        self._origin = self._read_clock()

    @property
    def active(self) -> bool:
        """True when a sink consumes the stream. Callers check this before
        building an event's payload, so no sink means no work."""
        return self._sink is not None

    def _read_clock(self) -> float:
        if self._clock is not None:
            return self._clock()
        return perf_counter()

    def _now(self) -> float:
        """Seconds since this tap (this stream) was created."""
        return self._read_clock() - self._origin

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
