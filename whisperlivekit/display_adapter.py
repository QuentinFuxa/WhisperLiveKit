"""Display adapter — derives display state from a caption event stream.

The display layer (overlay, and any future view) consumes ``CaptionEvent``s and derives what
to render: a rolling partial (ASR draft + provisional translation) and
finalized lines (source + translation pairs). This is the layer we test
independently of generation: feed a golden event sequence, assert the
rendered state.

The shipped web UI keeps using ``FrontData`` snapshots; this adapter is for
the display paths (our code), which previously reconstructed display
semantics from ``TestState`` fields with text-equality heuristics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from whisperlivekit.caption_events import CaptionEvent


@dataclass
class DisplayState:
    """What the display should show at any point in the stream."""
    partial_transcription: str = ""   # current rolling ASR (transcription_provisional)
    partial_translation: str = ""     # current provisional MT (translation_provisional)
    # ASR commits since the last translation final — the source text that will
    # pair with the next translation_final to form a finalized line.
    committed_transcription: str = ""
    # finalized lines: (source, translation) pairs, one per translation_final
    final_lines: List[Tuple[str, str]] = field(default_factory=list)
    # bookkeeping
    _last_final: str = ""

    def render_summary(self) -> str:
        """Human-readable one-line state, for tests/debug."""
        parts = []
        if self.final_lines:
            parts.append(f"finals={len(self.final_lines)}")
        if self.partial_translation:
            parts.append(f"prov={self.partial_translation!r}")
        if self.partial_transcription:
            parts.append(f"asr={self.partial_transcription!r}")
        if self.committed_transcription:
            parts.append(f"committed={self.committed_transcription!r}")
        return " | ".join(parts) or "(empty)"


class DisplayAdapter:
    """Stateful reducer: feed CaptionEvents, read DisplayState.

    Rules:
      - transcription_provisional    -> set partial_transcription (overwrites; rolling)
      - transcription_final          -> append to committed_transcription, clear the
                                        rolling draft (it is now committed)
      - translation_provisional      -> set partial_translation (overwrites; provisional)
      - translation_final            -> close the line: (committed_transcription, text)
                                        appended to final_lines; clear both partials
    """

    def __init__(self) -> None:
        self.state = DisplayState()

    def emit(self, event: CaptionEvent) -> DisplayState:
        """EventSink protocol — alias for feed()."""
        return self.feed(event)

    def feed(self, event: CaptionEvent) -> DisplayState:
        t = event.type
        if t == "transcription_provisional":
            self.state.partial_transcription = event.text
        elif t == "transcription_final":
            if event.text:
                sep = "" if not self.state.committed_transcription else " "
                self.state.committed_transcription += sep + event.text
            self.state.partial_transcription = ""  # committed; draft no longer rolling
        elif t == "translation_provisional":
            self.state.partial_translation = event.text
        elif t == "translation_final":
            if event.text and event.text != self.state._last_final:
                self.state.final_lines.append((self.state.committed_transcription, event.text))
                self.state._last_final = event.text
            self.state.partial_translation = ""  # final replaces the provisional
            self.state.committed_transcription = ""
        return self.state
