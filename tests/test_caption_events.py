"""Tests for the caption event stream + display adapter + diff (spike).

Two independent test surfaces, decoupled:
  1. CaptionLineAccumulator: feed golden events, assert the rendered state is coherent
     (draft grows into final, no flicker, finals accumulate). Tests the DISPLAY
     layer with no ASR/MT running.
  2. diff_event_streams: assert a known-bad captured stream (fragment finals,
     empty provisionals) diverges from the golden; assert a known-good stream
     matches. Tests the DIFF metric, not generation.
  3. CaptionEvent round-trip: save/load via EventLog.
"""
import os

from whisperlivekit.caption_display import CaptionLineAccumulator
from whisperlivekit.caption_events import CaptionEvent, EventLog, EventTap

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "zh_long_ideal.jsonl")


def _load_golden():
    return EventLog.load(GOLDEN_PATH).events


# ---------------------------------------------------------------------------
# CaptionLineAccumulator — line state tested with golden events (no ASR/MT)
# ---------------------------------------------------------------------------

class TestCaptionLineAccumulator:
    def test_golden_renders_one_final_per_sentence(self):
        """The golden stream has 6 translation_finals; the adapter should accumulate all 6."""
        events = _load_golden()
        acc = CaptionLineAccumulator()
        for e in events:
            acc.feed(e)
        assert len(acc.state.final_lines) == 6, (
            f"expected 6 finals, got {len(acc.state.final_lines)}: {acc.state.final_lines}"
        )

    def test_provisional_grows_then_clears_at_final(self):
        """Each translation_provisional sets partial_translation; the following translation_final clears it."""
        events = _load_golden()
        acc = CaptionLineAccumulator()
        seen_prov = False
        cleared_after_final = True
        for e in events:
            before = acc.state.partial_translation
            acc.feed(e)
            if e.type == "translation_provisional":
                assert acc.state.partial_translation, (
                    f"translation_provisional did not set provisional: {e}"
                )
                seen_prov = True
            elif e.type == "translation_final":
                if before and acc.state.partial_translation:
                    cleared_after_final = False
        assert seen_prov, "no translation_provisional observed"
        assert cleared_after_final, "translation_final did not clear the provisional"

    def test_transcription_final_clears_partial_transcription(self):
        """transcription_final commits the draft; partial_transcription should clear."""
        events = [CaptionEvent(0, 2.0, "transcription_provisional", "hello"),
                  CaptionEvent(0, 2.1, "transcription_final", "hello")]
        acc = CaptionLineAccumulator()
        acc.feed(events[0])
        assert acc.state.partial_transcription == "hello"
        acc.feed(events[1])
        assert acc.state.partial_transcription == ""

    def test_no_flicker_on_repeated_draft(self):
        """Repeated identical translation_provisionals should not append to final_lines."""
        events = [CaptionEvent(0, 1.0, "translation_provisional", "Hello"),
                  CaptionEvent(0, 1.1, "translation_provisional", "Hello"),
                  CaptionEvent(0, 1.2, "translation_provisional", "Hello world")]
        acc = CaptionLineAccumulator()
        for e in events:
            acc.feed(e)
        assert acc.state.final_lines == [], "drafts must not become finals"


# ---------------------------------------------------------------------------
# stream comparison (inline): a captured stream "matches" the golden when it
# has the same number of finals with no fragmentation or starvation signals

def _stream_report(captured, golden):
    """Compare a captured stream against the golden: count finals, drafts with
    no committed context, finals with no preceding draft, and fragmentation."""
    cap_f = [e for e in captured if e.type == "translation_final"]
    gold_f = [e for e in golden if e.type == "translation_final"]
    drafts = []
    seen_draft = False
    finals_without_draft = 0
    for e in captured:
        if e.type == "translation_provisional":
            drafts.append(e)
            seen_draft = True
        elif e.type == "translation_final":
            if not seen_draft:
                finals_without_draft += 1
            seen_draft = False
    empty_committed = sum(1 for e in drafts if not getattr(e, "committed", ""))
    return {
        "captured_finals": len(cap_f),
        "golden_finals": len(gold_f),
        "empty_committed_drafts": empty_committed,
        "finals_without_preceding_draft": finals_without_draft,
        "fragment_finals": len(cap_f) >= 2 * len(gold_f),
    }


class TestStreamComparison:
    def test_golden_matches_itself(self):
        golden = _load_golden()
        report = _stream_report(golden, golden)
        assert report["captured_finals"] == report["golden_finals"]
        assert not report["fragment_finals"]
        assert report["finals_without_preceding_draft"] == 0

    def test_fragment_finals_diverge(self):
        """A captured stream with 12 translation_finals (fragmentation) vs golden's 6."""
        golden = _load_golden()
        captured = []
        for i in range(6):
            captured.append(CaptionEvent(0, i, "translation_provisional", "frag", committed=""))
            captured.append(CaptionEvent(0, i, "translation_final", f"frag {i}"))
            captured.append(CaptionEvent(0, i, "translation_final", f"frag {i}b"))
        report = _stream_report(captured, golden)
        assert report["fragment_finals"], "should detect fragmentation"
        assert report["empty_committed_drafts"] == 6, "should detect empty committed"
        assert report["captured_finals"] == 12

    def test_starved_provisionals_diverge(self):
        """translation_final with no preceding translation_provisional (provisional starved)."""
        golden = _load_golden()
        captured = [CaptionEvent(0, 1, "translation_final", "final with no draft")]
        report = _stream_report(captured, golden)
        assert report["finals_without_preceding_draft"] == 1


# ---------------------------------------------------------------------------
# EventLog round-trip + EventTap no-op
# ---------------------------------------------------------------------------

class TestEventStream:
    def test_eventlog_roundtrip(self, tmp_path):
        log = EventLog()
        log.emit(CaptionEvent(1.0, 2.0, "transcription_provisional", "hello"))
        log.emit(CaptionEvent(1.1, 2.1, "translation_final", "你好", committed="", source=""))
        p = str(tmp_path / "ev.jsonl")
        log.save(p)
        loaded = EventLog.load(p)
        assert len(loaded.events) == 2
        assert loaded.events[0].type == "transcription_provisional"
        assert loaded.events[1].text == "你好"

    def test_tap_noop_when_no_sink(self):
        """A tap with no sink is zero-cost: nothing emitted, no crash."""
        tap = EventTap()
        tap.transcription_provisional(1.0, "hello")
        tap.translation_final(2.0, "你好")
        # no assertion needed — just must not raise

    def test_tap_forwards_to_sink(self):
        log = EventLog()
        tap = EventTap(sink=log, clock=lambda: 0.0)
        tap.transcription_provisional(1.0, "rolling")
        tap.translation_provisional(2.0, "prov", committed="c", source="s")
        assert len(log.events) == 2
        assert log.events[1].committed == "c"
        assert log.events[1].source == "s"
