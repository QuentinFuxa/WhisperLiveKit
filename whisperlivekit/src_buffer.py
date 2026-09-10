"""Pure state machine for the overlay's source reading buffer (no AppKit).

The src row semantics — accumulated committed clauses + rolling tail, with
the float-to-history happening when new words need the line — are extracted
here so they are unit-testable without AppKit. ``OverlayRenderer`` delegates
its src-row bookkeeping to this class; tests drive it directly.
"""
from __future__ import annotations


def _is_cjk(ch: str) -> bool:
    return ("\u4e00" <= ch <= "\u9fff"
            or ch in "\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\uff09\u300d\u300f\u3011")


def src_join(a: str, b: str) -> str:
    """Join accumulated source clauses: no space across a CJK boundary (zh has
    none), a single space for Latin."""
    if not a:
        return b
    if not b:
        return a
    if a[-1].isspace() or b[:1].isspace():
        return a + b
    if _is_cjk(a[-1]) and _is_cjk(b[0]):
        return a + b
    return a + " " + b


def ends_sentence(text: str) -> bool:
    """True when the text ends a sentence (CJK/Latin/quote-final terminators)."""
    return text.rstrip().endswith(
        ("。", "！", "？", ".", "!", "?", '."', '!"', '?"'))


class SrcReadingBuffer:
    """Src reading buffer: committed clauses accumulate; a sentence-final
    terminator freezes the buffer; the NEXT utterance's words promote it to
    the zh history field. The display is monotonic within a sentence.

    The rolling tail is RETAINED across commits: a commit absorbs the tail's
    committed prefix, so the display text never changes at commit time (no
    vanish/re-appear flicker at mid-sentence commits).

    Usage:
        display = buf.tail(text)      # rolling update -> display string
        if buf.consume_promotion():   # a completed sentence floated up
            zh_field = buf.last_promoted
        display = buf.commit(text)    # a clause committed -> display string
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._committed = ""          # committed clauses of the current sentence
        self._tail = ""               # rolling hypothesis tail being displayed
        self._sentence_complete = False
        self.last_promoted: str | None = None
        # right after a promotion, backends' rolling buffers often still carry
        # the promoted sentence — strip it from the tail until it's gone
        self._suppress_prefix: str = ""

    @property
    def committed(self) -> str:
        return self._committed

    @property
    def tail_text(self) -> str:
        """The rolling hypothesis tail currently held for display."""
        return self._tail

    @property
    def sentence_complete(self) -> bool:
        return self._sentence_complete

    def consume_promotion(self) -> str:
        """Return and clear the sentence promoted to history (or None)."""
        p = self.last_promoted
        self.last_promoted = None
        return p

    def tail(self, text: str) -> str:
        """A rolling update: stores the new hypothesis tail and returns the src
        row display string. If the buffer holds a completed sentence, it
        promotes — the float happens when new words need the line, not at
        commit."""
        if self._sentence_complete:
            self.last_promoted = self._committed
            self._suppress_prefix = self._committed
            self._committed = ""
            self._sentence_complete = False
            self._tail = ""
        # suppress the promoted sentence if the hypothesis still carries it —
        # otherwise the vanished bright text re-appears as a dim draft
        if self._suppress_prefix:
            if text.startswith(self._suppress_prefix):
                text = text[len(self._suppress_prefix):]
            else:
                self._suppress_prefix = ""  # hypothesis moved past it
        # defensive: some backends' rolling buffer still carries the committed
        # prefix — never render it twice
        if self._committed and text.startswith(self._committed):
            text = text[len(self._committed):]
        self._tail = text
        return src_join(self._committed, self._tail)

    def commit(self, text: str) -> str:
        """A committed clause accumulates into the buffer; a sentence-final
        terminator freezes the buffer (promotable). Returns the new display
        string: committed + the tail the commit did NOT absorb — the text on
        screen must not change at commit time (CL: flickering prefix)."""
        self._committed = src_join(self._committed, text)
        if self._tail.startswith(text):
            self._tail = self._tail[len(text):]
        elif text.startswith(self._tail) or text == self._tail:
            self._tail = ""  # the commit absorbed the whole tail
        else:
            # reworded hypothesis: the tail text is stale — drop it rather
            # than render text the ASR no longer claims
            self._tail = ""
        if ends_sentence(text):
            self._sentence_complete = True
        return src_join(self._committed, self._tail)

    @property
    def display(self) -> str:
        """The full src row display string."""
        return src_join(self._committed, self._tail)
