"""Tests for the caption event stream + display adapter + diff (spike).

Two independent test surfaces, decoupled:
  1. DisplayAdapter: feed golden events, assert the rendered state is coherent
     (draft grows into final, no flicker, finals accumulate). Tests the DISPLAY
     layer with no ASR/MT running.
  2. diff_event_streams: assert a known-bad captured stream (fragment finals,
     empty provisionals) diverges from the golden; assert a known-good stream
     matches. Tests the DIFF metric, not generation.
  3. CaptionEvent round-trip: save/load via EventLog.
"""
import os

from whisperlivekit.caption_events import CaptionEvent, EventLog, EventTap
from whisperlivekit.display_adapter import DisplayAdapter
from whisperlivekit.event_diff import diff_event_streams

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "zh_long_ideal.jsonl")


def _load_golden():
    return EventLog.load(GOLDEN_PATH).events


# ---------------------------------------------------------------------------
# DisplayAdapter — display layer tested with golden events (no ASR/MT)
# ---------------------------------------------------------------------------

class TestDisplayAdapter:
    def test_golden_renders_one_final_per_sentence(self):
        """The golden stream has 6 translation_finals; the adapter should accumulate all 6."""
        events = _load_golden()
        adapter = DisplayAdapter()
        for e in events:
            adapter.feed(e)
        assert len(adapter.state.final_lines) == 6, (
            f"expected 6 finals, got {len(adapter.state.final_lines)}: {adapter.state.final_lines}"
        )

    def test_provisional_grows_then_clears_at_final(self):
        """Each translation_provisional sets partial_translation; the following translation_final clears it."""
        events = _load_golden()
        adapter = DisplayAdapter()
        seen_prov = False
        cleared_after_final = True
        for e in events:
            before = adapter.state.partial_translation
            adapter.feed(e)
            if e.type == "translation_provisional":
                assert adapter.state.partial_translation, (
                    f"translation_provisional did not set provisional: {e}"
                )
                seen_prov = True
            elif e.type == "translation_final":
                if before and adapter.state.partial_translation:
                    cleared_after_final = False
        assert seen_prov, "no translation_provisional observed"
        assert cleared_after_final, "translation_final did not clear the provisional"

    def test_transcription_final_clears_partial_transcription(self):
        """transcription_final commits the draft; partial_transcription should clear."""
        events = [CaptionEvent(0, 2.0, "transcription_provisional", "hello"),
                  CaptionEvent(0, 2.1, "transcription_final", "hello")]
        adapter = DisplayAdapter()
        adapter.feed(events[0])
        assert adapter.state.partial_transcription == "hello"
        adapter.feed(events[1])
        assert adapter.state.partial_transcription == ""

    def test_no_flicker_on_repeated_draft(self):
        """Repeated identical translation_provisionals should not append to final_lines."""
        events = [CaptionEvent(0, 1.0, "translation_provisional", "Hello"),
                  CaptionEvent(0, 1.1, "translation_provisional", "Hello"),
                  CaptionEvent(0, 1.2, "translation_provisional", "Hello world")]
        adapter = DisplayAdapter()
        for e in events:
            adapter.feed(e)
        assert adapter.state.final_lines == [], "drafts must not become finals"


# ---------------------------------------------------------------------------
# diff_event_streams — the "how far from ideal" metric
# ---------------------------------------------------------------------------

class TestEventDiff:
    def test_golden_matches_itself(self):
        golden = _load_golden()
        report = diff_event_streams(golden, golden)
        assert report.verdict == "matches", report.summary()

    def test_fragment_finals_diverge(self):
        """A captured stream with 12 translation_finals (fragmentation) vs golden's 6."""
        golden = _load_golden()
        # synthetic capture: 12 fragment finals, 6 drafts (some empty committed)
        captured = []
        for i in range(6):
            captured.append(CaptionEvent(0, i, "translation_provisional", "frag", committed=""))
            captured.append(CaptionEvent(0, i, "translation_final", f"frag {i}"))
            captured.append(CaptionEvent(0, i, "translation_final", f"frag {i}b"))
        report = diff_event_streams(captured, golden)
        assert report.verdict == "diverges", report.summary()
        assert report.fragment_finals, "should detect fragmentation"
        assert report.empty_committed_drafts == 6, "should detect empty committed"
        assert report.captured_finals == 12

    def test_starved_provisionals_diverge(self):
        """translation_final with no preceding translation_provisional (provisional starved)."""
        golden = _load_golden()
        captured = [CaptionEvent(0, 1, "translation_final", "final with no draft")]
        report = diff_event_streams(captured, golden)
        assert report.finals_without_preceding_draft == 1
        assert report.verdict == "diverges"


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
