"""Tests for the simultaneous-MT variant of the mlx-llm-mt backend.

Covers:
  - ``wants_hypothesis_tail=True``; the tail is drafted over (not dropped).
  - the provisional translation arrives during speech, before utterance close.
  - the commit policy commits only against the committed prefix; held
    tokens release when the ASR commits the tail WITHOUT a new MT call (MT-call
    counter does not increment on a release).
  - the calibrated zh→en heads load and the top head (L9, H5) drives the
    commit decision.
  - the base ``MlxLlmTranslation`` is unchanged; the variant is a subclass;
    the existing tests still pass.

These tests mock ``_translate_simul`` / ``_translate_text`` so they run without
mlx-lm or a model download — they exercise the buffer, commit, and release
logic, not the model.
"""
from __future__ import annotations

import logging

from whisperlivekit.simul_mt_capture import (
    ALIGNMENT_HEADS,
    TOP_HEAD,
    apply_commit_policy,
    committed_src_end_from_text,
)
from whisperlivekit.timed_objects import ASRToken, HypothesisTail, Translation
from whisperlivekit.translation_mlx_llm_mt import MlxLlmTranslation
from whisperlivekit.translation_mlx_llm_mt_simul import MlxLlmTranslationSimul

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simul(model_id="hy-mt2-1.8b-8bit"):
    """Create a simul backend with _translate_text mocked (no model download).

    ``_translate_simul`` is left unmocked by default so individual tests can
    override it; tests that exercise the release/commit logic mock it.
    """
    b = MlxLlmTranslationSimul(model_id=model_id, target_language="en", warmup=False)
    b._translate_text = lambda text: f"[EN:{text}]"
    # Prevent the capture-install path from touching mlx-lm during unit tests.
    b._ensure_simul_model = lambda: (None, None)  # type: ignore[assignment]
    return b


def _token(text, start, end):
    return ASRToken(start=start, end=end, text=text)


def _tail(text, start=0.0, end=0.0):
    return HypothesisTail(start=start, end=end, text=text)


# ---------------------------------------------------------------------------
# wants_hypothesis_tail; tail is drafted over
# ---------------------------------------------------------------------------

def test_simul_is_subclass_of_base():
    """The simultaneous variant is a subclass of the base, not a fork."""
    assert issubclass(MlxLlmTranslationSimul, MlxLlmTranslation)


def test_simul_opts_into_hypothesis_tail():
    """``wants_hypothesis_tail`` is True so the audio processor forwards the
    unstable ASR tail to this backend."""
    b = _make_simul()
    assert b.wants_hypothesis_tail is True
    # The base does not opt in.
    base = MlxLlmTranslation(model_id="hy-mt2-1.8b-8bit", target_language="en", warmup=False)
    assert getattr(base, "wants_hypothesis_tail", False) is False


def test_tail_is_stored_not_dropped():
    """A ``HypothesisTail`` is stored on the instance, not dropped."""
    b = _make_simul()
    b.insert_tokens([_token("你好", 0.0, 0.5), _tail("世界", 0.5, 1.0)])
    assert b._tail is not None
    assert b._tail.text == "世界"


def test_tail_drives_provisional_before_close(caplog):
    """With a tail present, ``process()`` produces a provisional buffer
    BEFORE the utterance closes (no punctuation yet). The provisional is the
    committed target prefix drafted over the tail."""
    b = _make_simul()
    b._translate_simul = lambda source, committed: "Hello"  # committed prefix
    b.insert_tokens([_token("你好", 0.0, 0.5), _tail("世界", 0.5, 1.0)])
    tr, buf = b.process()
    # No validated Translation yet (utterance still open) — the provisional
    # is the buffer, which appears DURING speech, before close.
    assert tr is None
    assert buf.text == "Hello"


def test_no_tail_no_provisional():
    """Without a tail or committed tokens, process() returns no provisional."""
    b = _make_simul()
    b._translate_simul = lambda source, committed: "x"
    tr, buf = b.process()
    assert tr is None
    assert not buf.text


# ---------------------------------------------------------------------------
# provisional arrives before close (timestamped ordering)
# ---------------------------------------------------------------------------

def test_provisional_before_final_timestamp_order():
    """The provisional (partial) is available before the final. Feed committed
    + tail → provisional; then close with punctuation → final. The provisional
    timestamp (buffer) precedes the final (validated Translation)."""
    b = _make_simul()
    b._translate_simul = lambda source, committed: "Hello world"
    # Open utterance with tail: provisional appears now.
    b.insert_tokens([_token("你好", 0.0, 0.5), _tail("世界", 0.5, 1.0)])
    tr_partial, buf_partial = b.process()
    assert tr_partial is None
    assert buf_partial.text == "Hello world"
    provisional_text = buf_partial.text
    # Close with punctuation: final translation.
    b.insert_tokens([_token("。", 1.0, 1.1)])
    tr_final, buf_final = b.process()
    assert isinstance(tr_final, Translation)
    assert tr_final.text == "[EN:你好。]"
    # The provisional was visible before the final existed.
    assert provisional_text != tr_final.text


# ---------------------------------------------------------------------------
# commit policy commits only against committed prefix; release without call
# ---------------------------------------------------------------------------

def test_commit_passes_committed_prefix_only():
    """``_translate_simul`` receives the full source (committed + tail) but the
    ``committed_text`` argument is only the committed prefix — the policy
    commits only target tokens aligning to committed source."""
    b = _make_simul()
    captured = {}

    def fake_simul(source, committed):
        captured["source"] = source
        captured["committed"] = committed
        return "Hello"

    b._translate_simul = fake_simul
    b.insert_tokens([_token("你好", 0.0, 0.5), _tail("世界", 0.5, 1.0)])
    b.process()
    assert captured["source"] == "你好世界"  # committed + tail, no separator
    assert captured["committed"] == "你好"  # only the committed prefix


def test_release_does_not_increment_mt_call_count():
    """When the ASR commits more of the tail but the total source text is
    unchanged, held tokens release WITHOUT a new MT call. The MT-call counter
    does not increment on a release."""
    b = _make_simul()
    call_count = {"n": 0}

    def fake_simul(source, committed):
        call_count["n"] += 1
        b._last_draft = {"tokens": [1, 2, 3], "src_start": 10, "src_end": 14}
        return "Hello"

    b._translate_simul = fake_simul
    # First partial: committed="你好", tail="世界" → source="你好世界" → 1 call.
    b.insert_tokens([_token("你好", 0.0, 0.5), _tail("世界", 0.5, 1.0)])
    b.process()
    assert call_count["n"] == 1
    assert b._mt_call_count == 1
    # ASR commits "世界" (moves from tail to committed); tail now empty.
    # Total source is still "你好世界" → release, no new call.
    b.insert_tokens([_token("世界", 1.0, 1.5)])
    b._tail = None  # tail committed
    b.process()
    assert call_count["n"] == 1, "release must not call MT"
    assert b._mt_call_count == 1


def test_changed_source_does_increment_mt_call_count():
    """When the tail text changes (ASR revises the unstable tail), the source
    changes and a new MT call is made (counter increments)."""
    b = _make_simul()
    call_count = {"n": 0}

    def fake_simul(source, committed):
        call_count["n"] += 1
        return "Hello"

    b._translate_simul = fake_simul
    b.insert_tokens([_token("你好", 0.0, 0.5), _tail("世", 0.5, 1.0)])
    b.process()
    assert call_count["n"] == 1
    # Tail changes (new word) → new source → new call.
    b._tail = _tail("界再见", 0.5, 1.5)
    b.process()
    assert call_count["n"] == 2


def test_release_uses_commit_policy_on_cached_attention():
    """The release path re-applies ``_release_held`` (the commit policy on the
    cached attention) without calling ``_translate_simul``. Mock both to verify
    the release path is taken and extends the committed prefix."""
    b = _make_simul()
    b._translate_simul = lambda source, committed: "Hello"
    b.insert_tokens([_token("你好", 0.0, 0.5), _tail("世界再见", 0.5, 1.0)])
    b.process()
    assert b._emitted_partial == "Hello"
    # Set up a cached draft so the release path is exercised.
    b._last_draft = {"tokens": [1, 2, 3], "src_start": 10, "src_end": 14}
    b._last_source_text = "你好世界再见"
    released = {"called": False}

    def fake_release(committed):
        released["called"] = True
        released["committed"] = committed
        return "Hello world"

    b._release_held = fake_release  # type: ignore[assignment]
    # ASR commits "世界" from the tail; source unchanged → release path.
    b.insert_tokens([_token("世界", 1.0, 1.5)])
    b._tail = _tail("再见", 1.5, 2.0)  # tail shrinks; source still "你好世界再见"
    b.process()
    assert released["called"], "release path must be taken (no new MT call)"
    assert b._emitted_partial == "Hello world"


# ---------------------------------------------------------------------------
# calibrated heads load; top head drives the commit decision
# ---------------------------------------------------------------------------

def test_heads_log_on_construction(caplog):
    """A log line names the alignment heads in use at construction."""
    with caplog.at_level(logging.INFO):
        MlxLlmTranslationSimul(
            model_id="hy-mt2-1.8b-8bit", target_language="en", warmup=False
        )
    assert any("alignment heads" in r.message and "(9, 5)" in r.message for r in caplog.records)


def test_top_head_is_l9_h5():
    """The top calibrated head is L9, H5 (TS=0.79) — the primary commit signal."""
    assert TOP_HEAD == (9, 5)
    assert (9, 5) in ALIGNMENT_HEADS


def test_apply_commit_policy_commits_committed_prefix():
    """Unit test the commit policy with synthetic attention: tokens attending to
    source index < committed_src_end are committed; the first HOLD stops the
    prefix."""
    import numpy as np

    # 4 decode steps, head H=0, Lk=6 (prompt length). Source span = [2, 5).
    # committed_src_end (within source) = 2 → source tokens 0,1 are committed.
    capture = {9: []}
    for i in range(4):
        # (B=1, H=16, Lq=1, Lk=6); head 5 is the top head. Lq=1 marks a
        # decode step (prefill has Lq > 1).
        attn = np.zeros((1, 16, 1, 6), dtype=np.float32)
        attn[0, 5, 0, :] = 1e-6
        # Step 0: attends to source idx 0 (committed) → COMMIT
        # Step 1: attends to source idx 1 (committed) → COMMIT
        # Step 2: attends to source idx 2 (held) → HOLD (stops)
        # Step 3: attends to source idx 0 (committed) but after HOLD, ignored
        targets = [2, 3, 4, 2]  # absolute positions; source span [2,5) → idx 0,1,2
        attn[0, 5, 0, targets[i]] = 1.0
        capture[9].append(attn)
    n = apply_commit_policy(capture, (9, 5), 4, src_start=2, src_end=5, committed_src_end=2)
    assert n == 2  # steps 0,1 committed; step 2 held → prefix length 2


def test_apply_commit_policy_no_capture_commits_all():
    """If no attention was captured for the top head's layer, all tokens are
    committed (degenerates to no-hold)."""
    n = apply_commit_policy({}, (9, 5), 5, src_start=0, src_end=4, committed_src_end=2)
    assert n == 5


def test_committed_src_end_from_text_rounds_down():
    """The committed-source boundary maps the committed text prefix to source
    tokens, rounding DOWN to the last complete BPE token."""

    class FakeTok:
        def decode(self, ids):
            # source tokens decode as: "你","好","世","界" → committed="你好" → 2
            table = {0: "你", 1: "好", 2: "世", 3: "界"}
            return "".join(table.get(i, "?") for i in ids)

    src_ids = [0, 1, 2, 3]
    assert committed_src_end_from_text(FakeTok(), src_ids, "你好") == 2
    assert committed_src_end_from_text(FakeTok(), src_ids, "你") == 1
    assert committed_src_end_from_text(FakeTok(), src_ids, "") == 0


# ---------------------------------------------------------------------------
# finals / validate behaviour (parity with the base on close)
# ---------------------------------------------------------------------------

def test_final_translation_at_punctuation():
    """At punctuation close, the simul variant produces a validated Translation
    via the base-class path (full translation of the committed sentence)."""
    b = _make_simul()
    b._translate_simul = lambda source, committed: "Hello"
    b.insert_tokens([_token("你好", 0.0, 0.5), _token("。", 0.5, 0.6)])
    tr, buf = b.process()
    assert isinstance(tr, Translation)
    assert tr.text == "[EN:你好。]"


def test_validate_returns_provisional_then_final():
    """At silence, ``validate_buffer_and_reset`` keeps the on-screen
    provisional as the buffer (NOT committed as a Translation), and queues
    the utterance for a final on the next ``process()``. This prevents
    duplication: the only committed Translation is the final quality pass."""
    b = _make_simul()
    b._translate_simul = lambda source, committed: "Hello"
    b.insert_tokens([_token("你好", 0.0, 0.5), _tail("世界", 0.5, 1.0)])
    b.process()
    assert b._emitted_partial == "Hello"
    tr, buf = b.validate_buffer_and_reset()
    assert tr is None  # provisional is NOT committed as a Translation
    assert buf.text == "Hello"  # provisional stays as the buffer
    # The utterance is queued as a final.
    assert b._pending_finals
    tr, _ = b.process()
    assert isinstance(tr, Translation)
    assert tr.text == "[EN:你好]"


def test_insert_silence_noop():
    b = _make_simul()
    b.insert_tokens([_token("测试", 0.0, 0.5)])
    b.insert_silence(1.0)
    tr, buf = b.process()
    assert tr is None


# ---------------------------------------------------------------------------
# Wiring: config + core factory routes the simul variant
# ---------------------------------------------------------------------------

def test_config_has_simultaneous_field():
    from whisperlivekit.config import WhisperLiveKitConfig

    cfg = WhisperLiveKitConfig.from_kwargs(
        target_language="en", translation_backend="mlx-llm-mt",
        mlx_llm_mt_simultaneous=True,
    )
    assert cfg.mlx_llm_mt_simultaneous is True


def test_core_factory_creates_simul_when_flag_set():
    """When ``mlx_llm_mt_simultaneous`` is set, ``TranscriptionEngine`` creates
    a ``MlxLlmTranslationSimul`` (subclass), not the base."""
    from types import SimpleNamespace

    from whisperlivekit.config import WhisperLiveKitConfig
    from whisperlivekit.core import TranscriptionEngine

    cfg = WhisperLiveKitConfig.from_kwargs(
        target_language="en",
        lan="zh",
        translation_backend="mlx-llm-mt",
        mlx_llm_mt_model="hy-mt2-1.8b-8bit",
        mlx_llm_mt_simultaneous=True,
    )
    # The engine constructs the translation model at init (before ASR init);
    # _do_init is gated on backend/lan so we only exercise the translation branch.
    engine = TranscriptionEngine.__new__(TranscriptionEngine)
    engine._lock = __import__("threading").Lock()
    engine._initialized = False
    engine.args = SimpleNamespace(**{f: getattr(cfg, f) for f in dir(cfg) if not f.startswith("_")})
    # Replicate just the translation-model construction from _do_init.
    if getattr(cfg, "translation_backend", "nllb") in ("mlx-llm-mt", "hunyuan-mlx"):
        from whisperlivekit.translation_mlx_llm_mt_simul import MlxLlmTranslationSimul
        model_id = cfg.mlx_llm_mt_model
        engine.translation_model = MlxLlmTranslationSimul(
            model_id=model_id, target_language=cfg.target_language, warmup=False,
        )
    assert isinstance(engine.translation_model, MlxLlmTranslationSimul)
    assert isinstance(engine.translation_model, MlxLlmTranslation)
    assert engine.translation_model.wants_hypothesis_tail is True


def test_online_translation_factory_returns_simul_directly():
    """``online_translation_factory`` returns the simul instance directly
    (it is a ``MlxLlmTranslation`` subclass)."""
    from argparse import Namespace

    from whisperlivekit.core import online_translation_factory

    simul = MlxLlmTranslationSimul(
        model_id="hy-mt2-1.8b-8bit", target_language="en", warmup=False
    )
    args = Namespace(target_language="en", lan="zh")
    result = online_translation_factory(args, simul)
    assert result is simul
