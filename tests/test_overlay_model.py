"""Tests for the overlay display model (pure, AppKit-free).

Drives OverlayDisplayModel with a deterministic event stream + fake clock and
asserts on the DisplayState DOM — the same events the OverlayRenderer receives,
without AppKit or real time. Uses the segment-based API (list of (speaker, text)
tuples) matching the live callback contract.
"""
import os
from datetime import datetime, timedelta

from whisperlivekit.caption_events import EventLog
from whisperlivekit.overlay_model import (
    FINAL_ADD,
    FINAL_SAME,
    PROVISIONAL,
    OverlayDisplayModel,
)

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "zh_long_ideal.jsonl")

U1 = datetime(2026, 1, 1, 0, 0, 1)
U2 = datetime(2026, 1, 1, 0, 0, 2)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, s: float) -> None:
        self.now += s


def make(hold=3.5):
    clk = FakeClock()
    return OverlayDisplayModel(hold_sec=hold, clock=clk), clk


def segs(text, diff=None):
    """Build a single-speaker segment list (None speaker = no marker)."""
    return [(None, text, diff)] if diff else [(None, text)]


def plain(state):
    return "".join(s.text for s in state.current)


def prev_plain(state):
    return "".join(s.text for s in state.prev)


# ---- provisional ----

def test_preview_is_dimmed_and_does_not_expire():
    m, clk = make()
    m.preview(segs("Hello there"), started_at=U1)
    m.tick()
    state = m.state()
    assert state.current and state.current[0].style == PROVISIONAL


def test_preview_persists_after_hold_no_final():
    """A provisional never expires on its own — a vanishing draft reads as dropped."""
    m, clk = make()
    m.preview(segs("draft text"), started_at=U1)
    m.tick()
    clk.advance(10 * 3.5)
    m.tick()
    assert plain(m.state()) == "draft text"


def test_preview_rewrite_replaces_frozen_draft():
    """The MT re-commits a different prefix — the display must hard-replace."""
    m, clk = make()
    m.preview(segs("We are here today to discuss laser"), started_at=U1)
    m.tick()
    m.preview(segs("Today we will discuss laser"), started_at=U1)
    m.tick()
    assert plain(m.state()) == "Today we will discuss laser"


def test_preview_unchanged_skips():
    m, _ = make()
    m.preview(segs("same text"), started_at=U1)
    m.tick()
    m.preview(segs("same text"), started_at=U1)
    assert m.tick() is None  # no churn


# ---- final ----

def test_final_single_sentence_shows_immediately():
    m, clk = make()
    m.preview(segs("draft"), started_at=U1)
    m.tick()
    m.translation(segs("The real sentence."), started_at=U1)
    m.tick()  # drain the queued delta
    state = m.state()
    assert plain(state) == "The real sentence."
    assert state.current[0].style == FINAL_SAME
    assert state.prev == []  # own provisional never scrolls up


def test_two_consecutive_finals_scroll_up():
    """Speak two sentences; when the second shows, the first must be on prev.
    A final landing while the first is inside its hold queues behind it and
    pops after MIN_SHOW (half the hold) — no flash, no loss."""
    m, clk = make()
    m.translation(segs("First caption here."), started_at=U1)
    m.tick()
    m.translation(segs("Second caption arrives."), started_at=U1)
    # queued behind the shown final's minimum show (hold/2)
    clk.advance(2.0)
    m.tick()
    state = m.state()
    assert plain(state) == "Second caption arrives."
    assert prev_plain(state) == "First caption here."


def test_final_replaces_provisional_in_place():
    """A same-utterance final correcting a provisional: the common-prefix amend keeps
    the shown prefix (flips to final style) and only enqueues the delta. The committed
    prefix scrolls up when the delta shows (multi-sentence pacing). No cross-utterance
    scroll — the prev row holds the committed prefix, not an old caption."""
    m, clk = make()
    m.preview(segs("draft grows"), started_at=U1)
    m.tick()
    m.preview(segs("draft grows more"), started_at=U1)
    m.tick()
    m.translation(segs("draft grows more now final"), started_at=U1)
    m.tick()  # drain
    state = m.state()
    # the delta appends to the line — one coherent caption, prefix + delta
    assert plain(state) == "draft grows more now final"
    assert prev_plain(state) == ""  # nothing split off to history


def test_final_expires_after_hold():
    m, clk = make()
    m.translation(segs("Temporary caption."), started_at=U1)
    m.tick()
    clk.advance(3.6)
    m.tick()  # hold elapsed
    assert plain(m.state()) == ""


def test_prev_expires_on_own_timer():
    m, clk = make()
    m.translation(segs("First."), started_at=U1)
    m.tick()
    m.translation(segs("Second."), started_at=U1)
    clk.advance(2.0)  # queued final pops after MIN_SHOW
    m.tick()
    assert prev_plain(m.state()) == "First."
    clk.advance(3.6)
    m.tick()  # prev hold elapsed
    assert prev_plain(m.state()) == ""
    clk.advance(3.6)
    m.tick()  # prev hold elapsed
    assert prev_plain(m.state()) == ""


# ---- diff rendering ----

def test_final_with_diff_shows_add_green():
    """A final with diff spans renders 'add' as FINAL_ADD, 'same' as FINAL_SAME,
    'del' omitted."""
    m, clk = make()
    diff = [("same", "Hello "), ("del", "world"), ("add", "there")]
    m.translation([(None, "Hello there", diff)], started_at=U1)
    m.tick()
    state = m.state()
    styles = [s.style for s in state.current]
    texts = [s.text for s in state.current]
    assert FINAL_ADD in styles
    assert "there" in texts
    assert "world" not in texts  # del omitted


def test_provisional_with_diff_renders_all_dimmed():
    m, clk = make()
    diff = [("same", "Hello "), ("add", "world")]
    m.preview([(None, "Hello world", diff)], started_at=U1)
    m.tick()
    state = m.state()
    assert all(s.style == PROVISIONAL for s in state.current)


# ---- partial ----

def test_partial_in_state():
    m, clk = make()
    m.set_partial("源语言 partial")
    m.tick()
    state = m.state()
    assert state.partial == "源语言 partial"


# ---- append effect (provisional growth) ----

def test_preview_growth_appends_in_place_no_scroll():
    """The append effect: when a provisional extends the shown draft, the line
    grows in place — no scroll-up, no queue, no retype. The reader sees one
    growing line, not a new caption per update."""
    m, clk = make()
    m.preview(segs("Today we will discuss"), started_at=U1)
    m.tick()
    assert plain(m.state()) == "Today we will discuss"
    clk.advance(0.2)
    m.preview(segs("Today we will discuss laser"), started_at=U1)
    st = m.tick()
    assert st is not None
    assert plain(st) == "Today we will discuss laser"
    assert st.current[0].style == PROVISIONAL
    # the old draft must NOT have scrolled into history
    assert prev_plain(st) == ""


def test_preview_growth_does_not_churn_history():
    """Rapid provisional growth (the 42-drafts case) leaves history untouched."""
    m, clk = make()
    texts = ["Today we will discuss",
             "Today we will discuss laser",
             "Today we will discuss laser applications",
             "Today we will discuss laser applications in medicine"]
    for t in texts:
        m.preview(segs(t), started_at=U1)
        m.tick()
        clk.advance(0.1)
    st = m.state()
    assert plain(st) == texts[-1]
    assert prev_plain(st) == ""  # no draft ever scrolled up


def test_preview_rewrite_discards_draft_not_history():
    """A rewrite (different prefix) replaces the draft in place; history stays clean."""
    m, clk = make()
    m.preview(segs("We are here today to discuss laser"), started_at=U1)
    m.tick()
    m.preview(segs("Today we will discuss laser"), started_at=U1)
    m.tick()
    assert plain(m.state()) == "Today we will discuss laser"
    assert prev_plain(m.state()) == ""


def test_finals_scroll_but_drafts_never():
    """final→final scrolls the old final to history; draft→final does not."""
    m, clk = make()
    # draft shown
    m.preview(segs("Today we will discuss laser applications"), started_at=U1)
    m.tick()
    # final arrives (amend path: prefix to prev, delta becomes current)
    m.translation(segs("Today we will discuss laser applications in medicine."), started_at=U1)
    m.tick()
    st = m.state()
    # amend appends the delta to the line: one coherent final sentence
    assert plain(st) == "Today we will discuss laser applications in medicine."
    assert prev_plain(st) == ""  # the discarded draft never scrolled
    # next utterance's final scrolls the previous final up
    clk.advance(4.0)
    m.translation(segs("Dentists also use lasers for oral surgery."), started_at=U2)
    m.tick()
    st = m.state()
    assert "Dentists" in plain(st)
    assert "Today we will discuss laser applications in medicine." in prev_plain(st)  # the full final scrolled up


# ---- golden-stream display sanity (the "what shows in overlay makes sense" bar) ----

def _feed_golden_to_overlay(hold=3.5):
    """Drive the real overlay model with the golden CaptionEvents (the OverlaySink mapping)."""
    events = EventLog.load(GOLDEN_PATH).events
    EPOCH = datetime(2026, 1, 1)

    class C:
        now = 0.0
        def __call__(self): return self.now
        def advance(self, s): self.now += s
    c = C()
    m = OverlayDisplayModel(hold_sec=1.2, clock=c)  # short hold so pacing progresses
    trace = []
    for e in events:
        t = e.type
        if t == "transcription_provisional":
            m.set_partial(e.text)
        elif t == "transcription_final":
            m.clear_partial()
        elif t == "translation_provisional":
            m.preview([(None, e.text)], EPOCH + timedelta(seconds=e.audio_t))
        elif t == "translation_final":
            m.translation([(None, e.text)], EPOCH + timedelta(seconds=e.audio_t))
        c.advance(0.3)
        m.tick()
        snap = m.state()
        trace.append((e, m._en_plain, [sp.style for sp in snap.current],
                      "".join(s.text for s in snap.prev)))
    return events, trace


def test_golden_final_survives_its_hold():
    """After a translation_final lands, the row holds the polished sentence for
    the hold duration — the next sentence's in-flight draft must not wipe it."""
    events = EventLog.load(GOLDEN_PATH).events
    EPOCH = datetime(2026, 1, 1)

    class C:
        now = 0.0

        def __call__(self):
            return self.now

        def advance(self, s):
            self.now += s

    c = C()
    m = OverlayDisplayModel(hold_sec=1.0, clock=c)
    finals_seen = 0
    for e in events:
        t = e.type
        if t == "transcription_provisional":
            m.set_partial(e.text)
        elif t == "transcription_final":
            m.clear_partial()
        elif t == "translation_provisional":
            m.preview([(None, e.text)], EPOCH + timedelta(seconds=e.audio_t))
        elif t == "translation_final":
            m.translation([(None, e.text)], EPOCH + timedelta(seconds=e.audio_t))
        c.advance(0.2)
        m.tick()
        cur = m._en_plain
        if e.type == "translation_final":
            finals_seen += 1
            assert cur == e.text, f"final not displayed when it lands: {cur!r}"
            assert m._en_is_final, "final must be final-styled"
            c.advance(0.1)
            m.tick()
            assert m._en_plain == e.text, "final wiped before its hold elapsed"
    assert finals_seen == 6


def test_preempting_final_clears_stale_queued_draft():
    """A final that takes over the line while its own draft is still queued
    must clear the queue: on a >hold pause, the stale dim draft must not pop
    and regress the bright final (bright->dim on same content)."""
    m, clk = make()
    # draft queued behind nothing (first show)
    m.preview(segs("Sentence two"), started_at=U1)
    m.tick()
    clk.advance(0.1)
    # the final lands while the draft is still queued (preempt)
    m.translation(segs("Sentence two."), started_at=U1)
    m.tick()
    assert m._en_plain == "Sentence two."
    assert m._en_is_final
    assert m._queue == [], "stale draft left queued after the final took over"
    # speaker pauses past the hold: the final must persist (or expire cleanly),
    # never regress to the stale dim draft
    clk.advance(4.0)
    m.tick()
    assert m._en_plain != "Sentence two", "stale draft popped over the bright final"


# ---- monotonic src display (no vanish/reappear on hypothesis revision) ----

def test_set_partial_holds_on_pure_shrink():
    """A streaming revision that shrinks the tail (same committed split) is
    held — the display never vanishes/reappears on hypothesis churn."""
    from whisperlivekit.overlay_model import OverlayDisplayModel
    m = OverlayDisplayModel(hold_sec=3.5)
    m.set_partial("我們今天來討論鐳射在醫學上的應用", committed_len=0)
    m.set_partial("我們今天來討論鐳射在", committed_len=0)  # revision shrank
    assert m.state().partial == "我們今天來討論鐳射在醫學上的應用", "shrink must be held"
    m.set_partial("我們今天來討論鐳射在醫學上的應用，未來", committed_len=0)  # growth shows
    assert m.state().partial == "我們今天來討論鐳射在醫學上的應用，未來"


def test_set_partial_commit_shrink_passes_through():
    """A legit commit shrinks the tail but grows committed_len — passes through."""
    m = OverlayDisplayModel(hold_sec=3.5)
    m.set_partial("我們今天來討論鐳射在醫學上的應用", committed_len=0)
    m.set_partial("鐳射在醫學上的應用", committed_len=6)  # commit landed, tail stripped
    assert m.state().partial == "鐳射在醫學上的應用"
    assert m.state().partial_committed_len == 6


def test_set_partial_new_sentence_reset_passes_through():
    """A promote resets committed to empty — the shorter new-sentence tail must show."""
    m = OverlayDisplayModel(hold_sec=3.5)
    m.set_partial("我們今天來討論鐳射在醫學上的應用。", committed_len=12)
    m.set_partial("鐳射技術", committed_len=0)  # new sentence after promote
    assert m.state().partial == "鐳射技術"
    assert m.state().partial_committed_len == 0


# ---- SrcReadingBuffer (the src row state machine, pure) ----

def test_src_buffer_full_zh_long_sequence():
    """Drive the real src buffer through the golden sentence-2 sequence:
    clause commits accumulate, the sentence completes at the terminator,
    the next sentence's words promote it to history."""
    from whisperlivekit.src_buffer import SrcReadingBuffer
    b = SrcReadingBuffer()
    # sentence 1
    assert b.tail("我们今天来讨论") == "我们今天来讨论"
    b.commit("我们今天来讨论")
    assert b.tail("镭射在医学上的应用") == "我们今天来讨论镭射在医学上的应用"
    assert b.commit("镭射在医学上的应用。") == "我们今天来讨论镭射在医学上的应用。"
    assert b.sentence_complete
    # sentence 2 starts: sentence 1 promotes
    assert b.tail("镭射技术可以精确的切除肿瘤组织") == "镭射技术可以精确的切除肿瘤组织"
    assert b.consume_promotion() == "我们今天来讨论镭射在医学上的应用。"
    b.commit("镭射技术可以精确的切除肿瘤组织")
    assert b.tail("减少对周围健康组织的伤害") == "镭射技术可以精确的切除肿瘤组织减少对周围健康组织的伤害"
    assert b.commit("减少对周围健康组织的伤害。") == "镭射技术可以精确的切除肿瘤组织减少对周围健康组织的伤害。"
    assert b.sentence_complete


def test_src_join_cjk_no_space_latin_space():
    from whisperlivekit.src_buffer import src_join
    assert src_join("我们今天来讨论", "镭射在医学上的应用。") == "我们今天来讨论镭射在医学上的应用。"
    assert src_join("Hello everyone.", "My name is Ihab Bilad") == "Hello everyone. My name is Ihab Bilad"
    assert src_join("", "text") == "text"
    assert src_join("text", "") == "text"


def test_src_buffer_suppresses_promoted_sentence_in_tail():
    """After a promote, backends' rolling buffers still carry the promoted
    sentence — it must NOT re-appear as a dim draft in the same row."""
    from whisperlivekit.src_buffer import SrcReadingBuffer
    b = SrcReadingBuffer()
    b.commit("我们今天来讨论镭射在医学上的应用。")   # sentence 1 complete
    # next sentence's words arrive: promote fires, but the raw hypothesis
    # still carries the promoted sentence + the new tail
    display = b.tail("我们今天来讨论镭射在医学上的应用。镭射技术可")
    assert "我们今天来讨论镭射在医学上的应用。" not in display, (
        f"promoted sentence re-appeared in the draft: {display!r}")
    assert display == "镭射技术可", display
    assert b.consume_promotion() == "我们今天来讨论镭射在医学上的应用。"
    # once the hypothesis moves past the promoted text, suppression ends
    display = b.tail("镭射技术可以精确")
    assert display == "镭射技术可以精确"


# ---- sentence-partitioned commit display (the captain's sentence queue) ----

def drain(m, clk, steps=1):
    for _ in range(steps):
        clk.advance(0.1)
        st = m.tick()
        if st is not None:
            pass
    return m.state()


def test_multi_sentence_final_queues_each_sentence():
    """A multi-sentence final queues each sentence as its own bright item —
    each earns its own hold instead of one long line being replaced whole."""
    m, clk = make()
    m.translation(segs("Dentists also use lasers. This reduces bleeding."), started_at=U1)
    st = m.tick()
    st = m.state()
    assert plain(st) == "Dentists also use lasers.", plain(st)
    # the second sentence is queued and paces after the first's hold
    clk.advance(1.0)
    st = drain(m, clk)
    assert plain(st) == "Dentists also use lasers.", "second sentence preempted the first's hold"
    clk.advance(2.0)
    st = m.tick()
    st = m.state()
    assert plain(st) == "This reduces bleeding.", plain(st)
    assert prev_plain(st) == "Dentists also use lasers.", prev_plain(st)


def test_commit_crossing_terminator_freezes_completed_sentence():
    """When the growing commit/draft crosses '.', the completed sentence
    freezes (bright) and the line carries only the next sentence's fragment —
    the completed sentence is never extended by the next sentence's words."""
    m, clk = make()
    m.preview(segs("Dentists also use lasers for oral surgery."), started_at=U1)
    m.tick()
    clk.advance(0.2)
    # the draft crosses the terminator and starts the next sentence
    m.preview(segs("Dentists also use lasers for oral surgery. This reduces"), started_at=U1)
    m.tick()
    st = m.state()
    assert "Dentists also use lasers for oral surgery." in plain(st) or \
           "Dentists also use lasers for oral surgery." in prev_plain(st), \
        f"completed sentence not frozen: {plain(st)!r} / {prev_plain(st)!r}"
    # the completed sentence's text is stable while the next sentence grows
    text_now = prev_plain(st) or plain(st)
    assert text_now.endswith("surgery.")
    m.preview(segs("Dentists also use lasers for oral surgery. This reduces bleeding"), started_at=U1)
    m.tick()
    st = m.state()
    shown = plain(st) + " " + prev_plain(st)
    assert "Dentists also use lasers for oral surgery." in shown, \
        f"completed sentence lost on fragment growth: {shown!r}"


def test_reworded_final_amends_only_current_sentence():
    """A reworded final amends the sentence on the line; a completed bright
    sentence already scrolled to history is never retracted."""
    m, clk = make()
    # commit stream: sentence 1 completes and freezes (bright on the line)
    m.preview(segs("Dentists also use lasers for oral surgery. This reduces"), started_at=U1)
    m.tick()
    st = m.state()
    assert "Dentists also use lasers for oral surgery." in plain(st) + prev_plain(st)
    clk.advance(1.0)
    m.tick()
    assert "Dentists also use lasers for oral surgery." in plain(m.state()), \
        "the bright sentence did not survive its hold"
    # the final re-words the sentence: amend in place (green adds), never a
    # blank flash between the old wording and the new
    m.translation(segs("Dentists also use laser technology for oral surgery, reducing bleeding."), started_at=U1)
    m.tick()
    st = m.state()
    cur, prev = plain(st), prev_plain(st)
    shown = cur + " " + prev
    assert "surgery" in shown and "reducing bleeding" in shown, f"current sentence lost: {cur!r} / {prev!r}"
    assert cur != "", "the line blanked on the final (retraction)"


def test_dermatology_tail_displays_before_next_caption():
    """The captain's scenario: the commit stream showed only
    'Dermatologists use lasers to remove' when the segment's final arrived —
    the tail ('spots and tattoos') must display as part of the completed
    sentence before the next sentence's caption takes the line."""
    m, clk = make()
    m.preview(segs("This reduces bleeding, sweating, pain. Dermatologists use lasers to remove"), started_at=U1)
    m.tick()
    # the final lands while the reader is mid-'remove'
    m.translation(segs("Dentists also use laser technology for oral surgery, reducing bleeding, sweating, and pain. "
                       "Dermatologists use lasers to remove spots and tattoos."), started_at=U1)
    m.tick()
    st = m.state()
    clk.advance(2.0)
    m.tick()
    st = m.state()
    shown = plain(st) + " | " + prev_plain(st)
    assert "spots and tattoos" in shown, f"the tail never displayed: {shown!r}"
    # and both sentences pace: the second sentence holds before the next final
    clk.advance(1.0)
    m.tick()
    st = m.state()
    assert plain(st) == "Dentists also use laser technology for oral surgery, reducing bleeding, sweating, and pain." or \
           "spots and tattoos" in plain(st) + prev_plain(st), plain(st)


# ---- end-of-stream drained replay (round-2: queued sentences must display) ----

TIME_FRONTIER_GOLDEN = os.path.join(os.path.dirname(__file__), "golden",
                                    "zh_long_time_frontier.jsonl")


def _drained_replay(path, hold=3.5, drain_seconds=30.0):
    """Replay a captured event stream through the REAL display model with the
    drainer pumped past the last event — what a viewer sees when the stream
    ends and real time keeps running. Returns the ordered list of distinct
    lines that reached the current line."""
    events = EventLog.load(path).events
    t0 = events[0].t
    m, clk = make(hold=hold)
    base = datetime(2026, 1, 1)
    seen = []

    for e in events:
        clk.advance(0.8)  # ~real-time pacing per event
        when = base + timedelta(seconds=e.t - t0)
        if e.type == "transcription_provisional":
            m.set_partial(e.text)
        elif e.type == "transcription_final":
            m.clear_partial()
        elif e.type == "translation_provisional":
            m.preview([(None, e.text)], when)
        elif e.type == "translation_final":
            m.translation([(None, e.text)], when)
        st = m.tick()
        if st is not None:
            cur = "".join(s.text for s in st.current)
            if cur and cur != (seen[-1] if seen else None):
                seen.append(cur)
    # pump the drainer past the end of the stream (the reader's clock keeps
    # running): every completed sentence must reach the line
    for _ in range(int(drain_seconds / 0.1)):
        clk.advance(0.1)
        st = m.tick()
        if st is not None:
            cur = "".join(s.text for s in st.current)
            if cur and cur != (seen[-1] if seen else None):
                seen.append(cur)
    return seen


def test_drained_replay_displays_every_completed_sentence():
    """End-of-stream guarantee: replaying the captured zh_long time-frontier
    stream through the display model and draining past the last event, EVERY
    completed sentence of every final displays — including the dermatology
    tail ('spots and tattoos') that motivated the sentence queue. Fails if a
    queued sentence is dropped by a later draft or a queue-jumping final."""
    assert os.path.exists(TIME_FRONTIER_GOLDEN), "golden stream missing"
    seen = _drained_replay(TIME_FRONTIER_GOLDEN)
    for sentence in (
        "Dermatologists use lasers to remove spots and tattoos.",
        "It reduces bleeding, sweating, and pain.",
        "In summary, laser plays an increasingly important role in modern medicine.",
    ):
        assert any(sentence in s for s in seen), \
            f"never displayed: {sentence!r}; saw: {seen!r}"


def test_pending_sentences_survive_next_utterance_draft():
    """A new utterance's DRAFT must not drop queued committed sentences
    (round-2: the 'In short.' draft wiped the dentist final's backlog)."""
    m, clk = make()
    m.translation(segs("Dentists also use lasers for oral surgery. It reduces bleeding, "
                       "sweating, and pain. Dermatologists use lasers to remove spots and tattoos."),
                  started_at=U1)
    m.tick()
    st = m.state()
    assert plain(st) == "Dentists also use lasers for oral surgery.", plain(st)
    # the next utterance's fragment draft arrives while two sentences are queued
    m.preview(segs("In short."), started_at=U2)
    m.tick()
    seen = [plain(m.state())]
    for _ in range(60):
        clk.advance(0.25)
        m.tick()
        seen.append(plain(m.state()))
    joined = "\n".join(seen)
    assert "It reduces bleeding, sweating, and pain." in joined, f"draft dropped a queued sentence: {joined!r}"
    assert "spots and tattoos" in joined, f"draft dropped the tail sentence: {joined!r}"


def test_new_final_enqueues_behind_pending_sentences_fifo():
    """A new final enqueues BEHIND pending completed sentences and pops in
    order — it never jumps the queue (round-2: 'In summary' went straight to
    bright over the dentist final's pending sentences)."""
    m, clk = make()
    m.translation(segs("First sentence here. Second sentence here."), started_at=U1)
    m.tick()
    assert plain(m.state()) == "First sentence here."
    # second final arrives while the second sentence is still queued
    m.translation(segs("Third sentence entirely different."), started_at=U2)
    m.tick()
    # FIFO: the pending second sentence displays next, NOT the new final
    clk.advance(2.0)
    m.tick()
    st = m.state()
    shown = plain(st) + "|" + prev_plain(st)
    assert "Second sentence here" in shown, f"queue jumped: {shown!r}"
    clk.advance(3.0)
    m.tick()
    st = m.state()
    shown = plain(st) + "|" + prev_plain(st)
    assert "Third sentence" in shown, f"the new final never displayed: {shown!r}"


# ---- promote-in-place (round 3: no identical-content retype) ----

def test_promote_in_place_flips_bright_without_retype():
    """The sentence the reader is watching type promotes at its terminator as
    a single style flip (bright, scrolled to the prev line) — never a
    clear+re-render of identical text (round 3: the dentist retype, where the
    line reset and the queue re-rendered the words the reader had just read)."""
    m, clk = make()
    S1 = "Dentists also use laser technology for oral surgery."
    m.preview(segs("Dentists also use laser technology for "), started_at=U1)
    m.tick()
    # the draft crosses the terminator: the completed sentence promotes
    m.preview(segs(S1 + " This reduces"), started_at=U1)
    m.tick()
    st = m.state()
    # the completed sentence is bright exactly once — on the prev line
    assert prev_plain(st) == S1, prev_plain(st)
    assert prev_plain(st) not in plain(st), "promoted sentence re-rendered on the current line"
    # the line keeps only the in-progress fragment (a layout split, not a clear)
    assert plain(st) == "This reduces", plain(st)
    # no queue round-trip: the completed sentence was never enqueued
    assert all(it.plain != S1 for it in m._queue), \
        [it.plain for it in m._queue]
    # the current line never went clear between the dim and bright frames of
    # the same content (the clear+retype signature)
    assert plain(st) != "" , "line cleared at the boundary"


def test_promote_in_place_final_amends_without_retype():
    """The final amends the promoted sentence in its scrolled position and the
    never-typed tail sentence enqueues — no whole-line re-render."""
    m, clk = make()
    S1 = "Dentists also use laser technology for oral surgery."
    m.preview(segs("Dentists also use laser technology for "), started_at=U1)
    m.tick()
    m.preview(segs(S1 + " This reduces"), started_at=U1)
    m.tick()
    st = m.state()
    assert prev_plain(st) == S1
    # the final's authoritative wording amends the promoted sentence in place
    m.translation(segs(S1 + " It reduces bleeding, sweating, and pain. "
                       "Dermatologists use lasers to remove spots and tattoos."),
                  started_at=U1)
    st = m.state()
    assert prev_plain(st) == S1, "promoted sentence was re-rendered instead of amended"
    # the reworded sentence is on the line (bright) or queued — never a full
    # re-render of S1
    cur = plain(st)
    assert cur != S1, "S1 re-typed on the current line"
    # the tail sentence the reader never saw is queued (or on the line), not lost
    all_pending = cur + "".join(it.plain for it in m._queue)
    assert "Dermatologists use lasers to remove spots and tattoos." in all_pending or \
           "Dermatologists use lasers to remove spots and tattoos." in prev_plain(st), \
           (cur, [it.plain for it in m._queue])
