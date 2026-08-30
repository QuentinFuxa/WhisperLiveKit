"""Unit tests for the Nemotron MLX ASR transducer backend.

These tests mock the mlx_audio model/alignment/tokenizer modules so they run
without mlx_audio installed or a model download.  They exercise the decode
loop + AlignedToken population (the time-based accessible boundary that is
the research value), the monotonic append-only property, finalize flushing,
and the online-processor lifecycle contract.

Falsification guide (one line per claim):
  - emission: start is non-None and monotonic non-decreasing — breaks if
    _decode_chunk uses global_time+time wrong or skips the start.
  - monotonicity: hypothesis only grows — breaks if _decode_chunk ever
    replaces or deletes a prior token.
  - finalize-flush: _finalize runs a final step and resets — breaks if it
    skips the final drive or doesn't reset hypothesis/active.
  - lifecycle: insert→process→buffer→finish returns ASRTokens with
    timestamps — breaks if the VAD state machine or contract diverges.
"""

from __future__ import annotations

import sys
import types
from collections import deque

import mlx.core as mx
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Inject fake mlx_audio modules so the backend imports without the package.
# ---------------------------------------------------------------------------


def _install_fake_mlx_audio():
    """Inject minimal fake mlx_audio sub-modules into sys.modules."""
    from dataclasses import dataclass

    @dataclass
    class AlignedToken:
        id: int
        text: str
        start: float
        duration: float
        end: float = 0.0

        def __post_init__(self):
            self.end = self.start + self.duration

    @dataclass
    class AlignedSentence:
        text: str
        tokens: list
        start: float = 0.0
        end: float = 0.0
        duration: float = 0.0

        def __post_init__(self):
            self.tokens = list(sorted(self.tokens, key=lambda x: x.start))
            self.start = self.tokens[0].start
            self.end = self.tokens[-1].end
            self.duration = self.end - self.start

    @dataclass
    class AlignedResult:
        text: str
        sentences: list

        def __post_init__(self):
            self.text = self.text.strip()

    def tokens_to_sentences(tokens):
        sentences = []
        current = []
        for idx, token in enumerate(tokens):
            current.append(token)
            if (
                "!" in token.text
                or "?" in token.text
                or "." in token.text
            ):
                sentences.append(AlignedSentence("".join(t.text for t in current), current))
                current = []
        if current:
            sentences.append(AlignedSentence("".join(t.text for t in current), current))
        return sentences

    def sentences_to_result(sentences):
        return AlignedResult("".join(s.text for s in sentences), sentences)

    # alignment module
    align_mod = types.ModuleType("mlx_audio.stt.models.nemo.alignment")
    align_mod.AlignedToken = AlignedToken
    align_mod.AlignedSentence = AlignedSentence
    align_mod.AlignedResult = AlignedResult
    align_mod.tokens_to_sentences = tokens_to_sentences
    align_mod.sentences_to_result = sentences_to_result

    # tokenizer module
    tok_mod = types.ModuleType("mlx_audio.stt.models.nemotron_asr.tokenizer")
    _SPECIAL = {"<unk>", "<pad>", "<s>", "</s>"}

    def is_special_token(token_id, vocabulary):
        if token_id < 0 or token_id >= len(vocabulary):
            return False
        return vocabulary[token_id] in _SPECIAL

    def decode(tokens, vocabulary, strip_lang_tags=True):
        parts = []
        for t in tokens:
            if t < 0 or t >= len(vocabulary):
                continue
            piece = vocabulary[t]
            if piece in _SPECIAL:
                continue
            parts.append(piece.replace("▁", " "))
        return "".join(parts)

    tok_mod.is_special_token = is_special_token
    tok_mod.decode = decode

    # Parent packages
    for pkg in [
        "mlx_audio",
        "mlx_audio.stt",
        "mlx_audio.stt.models",
        "mlx_audio.stt.models.nemo",
        "mlx_audio.stt.models.nemotron_asr",
        "mlx_audio.vad",
    ]:
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)

    sys.modules["mlx_audio.stt.models.nemo.alignment"] = align_mod
    sys.modules["mlx_audio.stt.models.nemotron_asr.tokenizer"] = tok_mod

    return align_mod, tok_mod


_fake_align, _fake_tok = _install_fake_mlx_audio()

# Now import the backend.
from whisperlivekit.asr_nemotron_mlx import (  # noqa: E402
    NemotronMLXOnlineProcessor,
    _normalize_language,
)
from whisperlivekit.timed_objects import ASRToken  # noqa: E402

# ---------------------------------------------------------------------------
# Fake model for _decode_chunk tests
# ---------------------------------------------------------------------------


class FakeModel:
    """Minimal model mock for the greedy RNN-T decode loop.

    ``joint_sequence`` is a list of token ids the joint network will emit
    in order (one per ``joint()`` call).  The decoder and joint return mlx
    arrays so ``int(mx.argmax(...))`` works.
    """

    def __init__(self, joint_sequence, vocabulary, blank_id=100, max_symbols=10, d_model=8):
        self.joint_sequence = joint_sequence
        self.vocabulary = vocabulary
        self.blank_id = blank_id
        self.max_symbols = max_symbols
        self.d_model = d_model
        self._joint_idx = 0

    def decoder(self, current_token, hidden):
        out = mx.zeros((1, 1, self.d_model))
        h = mx.zeros((1, self.d_model))
        c = mx.zeros((1, self.d_model))
        return out, (h, c)

    def joint(self, feature, decoder_output):
        token = self.joint_sequence[self._joint_idx % len(self.joint_sequence)]
        self._joint_idx += 1
        vocab_size = max(self.blank_id + 1, len(self.vocabulary))
        logits = mx.zeros((1, 1, vocab_size))
        logits[0, 0, token] = 1.0
        return logits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeEncoder:
    """Minimal encoder mock for _reset()."""
    def reset(self):
        pass


def _make_processor(model, frame_sec=0.08):
    """Build a minimal NemotronMLXOnlineProcessor bypassing __init__."""
    proc = NemotronMLXOnlineProcessor.__new__(NemotronMLXOnlineProcessor)
    proc._model = model
    proc._frame_sec = frame_sec
    proc._global_time = 0
    proc._last_token = model.blank_id
    proc._decoder_hidden = None
    proc._hypothesis = []
    proc._text = ""
    proc._active = True
    proc._n_samples = 0
    proc._audio = []
    proc._mel_consumed = 0
    proc._mel_stable = None
    proc._silence_frames = 0
    proc._encoder = FakeEncoder()
    proc.end = 0.0
    proc.SAMPLING_RATE = 16000
    proc._two_pass = False
    proc.asr = None
    proc._vad_threshold = 0.5
    proc._n_preroll = 10
    proc._rule1_silence = 2.4
    proc._rule2_punct_silence = 0.6
    proc._rule2_silence = 1.2
    proc._rule2_soft_max = 8.0
    proc._rule3_max = 20.0
    proc._raw_chunks = []
    proc._raw_len = 0
    proc._vad_leftover = np.empty(0, dtype=np.float32)
    proc._vad = None
    proc._vad_state = None
    proc._pre = None
    proc._preroll = deque(maxlen=10)
    proc.logfile = sys.stderr
    proc.audio_buffer = np.array([], dtype=np.float32)
    return proc


# ---------------------------------------------------------------------------
# Test 1: _decode_chunk emits AlignedTokens with monotonic start times
# ---------------------------------------------------------------------------


def test_decode_chunk_emits_aligned_tokens():
    """Non-blank emissions append AlignedToken with start = (global_time +
    time) * frame_sec, monotonically non-decreasing."""
    vocab = ["<unk>", "<pad>", "hello", "world", "."]
    # Sequence: blank(t=0), token2(hello)(t=0, emit), blank(t=1), token3(world)(t=1, emit), blank(t=2)
    # But the loop stays at the same time for non-blank (no time advance unless max_symbols).
    # So: t=0: emit "hello" (token 2), t=0: blank → t=1, t=1: emit "world" (token 3), t=1: blank → t=2
    model = FakeModel(
        joint_sequence=[2, 100, 3, 100],  # hello, blank, world, blank
        vocabulary=vocab,
        blank_id=100,
    )
    proc = _make_processor(model, frame_sec=0.08)
    # prompted: (1, 2, d) — 2 encoder time steps
    prompted = mx.zeros((1, 2, 8))

    proc._decode_chunk(prompted)

    assert len(proc._hypothesis) == 2, f"expected 2 tokens, got {len(proc._hypothesis)}"
    t0, t1 = proc._hypothesis
    # First token at time=0: start = (0 + 0) * 0.08 = 0.0
    assert t0.start == 0.0, f"first token start={t0.start}, expected 0.0"
    # Second token at time=1: start = (0 + 1) * 0.08 = 0.08
    assert t1.start == 0.08, f"second token start={t1.start}, expected 0.08"
    # Monotonic non-decreasing
    assert t1.start >= t0.start, "start times must be non-decreasing"
    # Text is decoded
    assert "hello" in t0.text, f"first token text={t0.text!r}"
    assert "world" in t1.text, f"second token text={t1.text!r}"


# ---------------------------------------------------------------------------
# Test 2: hypothesis is append-only (monotonic — no replacement/deletion)
# ---------------------------------------------------------------------------


def test_hypothesis_append_only():
    """Multiple _decode_chunk calls only grow the hypothesis; prior tokens
    are never mutated or dropped."""
    vocab = ["<unk>", "<pad>", "foo", "bar", "baz"]
    model = FakeModel(
        joint_sequence=[2, 100, 3, 100],  # foo, blank, bar, blank
        vocabulary=vocab,
        blank_id=100,
    )
    proc = _make_processor(model, frame_sec=0.08)

    # First chunk
    proc._decode_chunk(mx.zeros((1, 2, 8)))
    first_snapshot = list(proc._hypothesis)
    assert len(first_snapshot) == 2

    # Advance global time (as _decode_chunk would)
    proc._global_time += 2

    # Second chunk
    model._joint_idx = 0
    model.joint_sequence = [4, 100]  # baz, blank
    proc._decode_chunk(mx.zeros((1, 1, 8)))

    # Prior tokens must be unchanged
    assert len(proc._hypothesis) == 3, f"expected 3 tokens, got {len(proc._hypothesis)}"
    for i, orig in enumerate(first_snapshot):
        assert proc._hypothesis[i] is orig, f"token {i} was replaced"
        assert proc._hypothesis[i].start == orig.start, f"token {i} start mutated"
        assert proc._hypothesis[i].text == orig.text, f"token {i} text mutated"
    # New token appended at the end
    assert "baz" in proc._hypothesis[2].text


# ---------------------------------------------------------------------------
# Test 3: _finalize flushes held-back mel and resets state
# ---------------------------------------------------------------------------


def test_finalize_flushes_held_mel():
    """_finalize calls _drive(final=True) to flush held-back mel, returns
    ASRTokens, and resets decode state."""
    vocab = ["<unk>", "<pad>", "hello", "world"]
    model = FakeModel(
        joint_sequence=[100],  # not used (we mock _drive)
        vocabulary=vocab,
        blank_id=100,
    )
    proc = _make_processor(model, frame_sec=0.08)

    # Seed the hypothesis with two pre-existing tokens (simulating mid-decode).
    proc._hypothesis = [
        _fake_align.AlignedToken(2, start=0.0, duration=0.08, text="hello"),
        _fake_align.AlignedToken(3, start=0.08, duration=0.08, text="world"),
    ]
    proc._text = "hello world"
    proc._n_samples = 16000  # 1 second of audio
    proc._audio = [np.zeros(16000, dtype=np.float32)]
    proc.end = 1.0
    proc._active = True

    # Mock _drive(final=True) to append a flushed tail token.
    flush_called = []

    def mock_drive(final):
        flush_called.append(final)
        if final:
            proc._hypothesis.append(
                _fake_align.AlignedToken(2, start=0.16, duration=0.08, text="!")
            )
            proc._text = "hello world!"

    proc._drive = mock_drive

    tokens = proc._finalize()

    # _finalize ran a final=True drive
    assert True in flush_called, "finalize must call _drive(final=True)"

    # Returns ASRTokens with absolute timestamps
    assert len(tokens) == 3, f"expected 3 ASRTokens, got {len(tokens)}"
    assert all(isinstance(t, ASRToken) for t in tokens)
    # Timestamps are non-None and non-decreasing
    starts = [t.start for t in tokens]
    assert all(s is not None for s in starts), "start must not be None"
    assert starts == sorted(starts), "starts must be non-decreasing"

    # State is reset
    assert proc._hypothesis == [], "hypothesis must be reset after finalize"
    assert not proc._active, "active must be False after finalize (no tail)"
    assert proc._text == "", "text must be reset"


# ---------------------------------------------------------------------------
# Test 4: online processor lifecycle (insert → process → buffer → finish)
# ---------------------------------------------------------------------------


class FakeVad:
    """VAD mock that produces speech for the first ``n_speech`` frames then
    silence."""

    def __init__(self, n_speech=10):
        self.n_speech = n_speech
        self._frame = 0

    def initial_state(self, sample_rate=16000):
        return None

    def feed(self, chunk, state, sample_rate=16000):
        self._frame += 1
        prob = 0.9 if self._frame <= self.n_speech else 0.0
        return mx.array([prob]), state


def test_online_processor_lifecycle():
    """insert_audio_chunk → process_iter → get_buffer → finish returns
    committed ASRTokens with timestamps, following the online-processor
    contract shape."""
    vocab = ["<unk>", "<pad>", "hello", "world", "."]
    model = FakeModel(
        joint_sequence=[2, 100, 3, 100, 4, 100],  # hello, blank, world, blank, ".", blank
        vocabulary=vocab,
        blank_id=100,
    )

    proc = _make_processor(model, frame_sec=0.08)
    # Equip with a fake VAD that produces speech for 10 frames then silence.
    proc._vad = FakeVad(n_speech=10)
    proc._vad_state = None
    proc._active = False  # start in IDLE

    # Mock _drive to append a token to the hypothesis (simulating decode).
    drive_count = [0]

    def mock_drive(final):
        drive_count[0] += 1
        if not final and not proc._hypothesis:
            proc._hypothesis.append(
                _fake_align.AlignedToken(2, start=0.0, duration=0.08, text="hello")
            )
            proc._text = "hello"
        elif final and proc._hypothesis:
            proc._hypothesis.append(
                _fake_align.AlignedToken(3, start=0.08, duration=0.08, text="world")
            )
            proc._text = "hello world"

    proc._drive = mock_drive

    # Feed 5 VAD frames of audio (speech) — onset + decode.
    # Each VAD frame is 512 samples.
    audio_chunk = np.zeros(512 * 5, dtype=np.float32)
    proc.insert_audio_chunk(audio_chunk, 5 * 512 / 16000)
    tokens, end_time = proc.process_iter()

    # Speech onset → active, drive called, but no final yet.
    assert proc._active, "should be active after speech onset"
    assert drive_count[0] > 0, "_drive should have been called"
    # Buffer should show the partial text.
    buf = proc.get_buffer()
    assert buf.text == "hello", f"buffer should show partial text, got {buf.text!r}"

    # Feed silence frames to trigger rule2 finalize.
    # Need enough silence: rule2_silence=1.2s = 1.2/0.032 ≈ 38 frames.
    # 5 speech frames (from VAD mock's remaining budget) + 45 silence = 50 total.
    silence_chunk = np.zeros(512 * 50, dtype=np.float32)
    proc.insert_audio_chunk(silence_chunk, (5 * 512 + 512 * 50) / 16000)
    tokens, end_time = proc.process_iter()

    # Finalize should have produced ASRTokens.
    assert len(tokens) >= 1, f"expected finalized tokens, got {len(tokens)}"
    assert all(isinstance(t, ASRToken) for t in tokens)
    assert all(t.start is not None for t in tokens), "all tokens must have timestamps"
    # After finalize, buffer should be empty.
    buf = proc.get_buffer()
    assert buf.text == "", f"buffer should be empty after finalize, got {buf.text!r}"

    # finish() on an idle processor returns no tokens.
    final_tokens, _ = proc.finish()
    assert final_tokens == [], "finish on idle processor should return no tokens"


# ---------------------------------------------------------------------------
# Test 5: language normalization (mirrors livecaption normalize_asr_language)
# ---------------------------------------------------------------------------


def test_normalize_language_zh_to_zh_cn():
    """Bare 'zh' resolves to the model's 'zh-CN' prompt key, not a ValueError."""
    supported = ["zh-CN", "zh-ZH", "en-US", "ja-JP"]
    result = _normalize_language("zh", supported)
    assert result == "zh-CN", f"expected zh-CN, got {result!r}"


def test_normalize_language_en_to_en_us():
    """Bare 'en' resolves to the model's 'en-US' prompt key."""
    supported = ["zh-CN", "zh-ZH", "en-US", "ja-JP"]
    result = _normalize_language("en", supported)
    assert result == "en-US", f"expected en-US, got {result!r}"


def test_normalize_language_exact_match_passthrough():
    """An exact prompt_dictionary key passes through unchanged."""
    supported = ["zh-CN", "zh-ZH", "en-US", "ja-JP"]
    assert _normalize_language("zh-CN", supported) == "zh-CN"
    assert _normalize_language("en-US", supported) == "en-US"


def test_normalize_language_case_insensitive():
    """Matching is case-insensitive (model keys are mixed-case)."""
    supported = ["zh-CN", "en-US"]
    assert _normalize_language("zh-cn", supported) == "zh-CN"
    assert _normalize_language("EN-us", supported) == "en-US"


def test_normalize_language_auto():
    """'auto' returns 'auto' (the caller converts to None)."""
    assert _normalize_language("auto", ["zh-CN", "en-US"]) == "auto"
    assert _normalize_language("", ["zh-CN"]) == "auto"


def test_normalize_language_unknown_raises():
    """An unknown language with no mapping raises ValueError."""
    supported = ["zh-CN", "zh-ZH", "en-US", "ja-JP"]
    with pytest.raises(ValueError, match="not supported"):
        _normalize_language("xx", supported)


def test_normalize_language_prefix_unique_match():
    """When only one supported key matches the primary prefix, it is used
    even without a default-tag mapping."""
    # 'fr' has no default-tag match here, but only one fr-* key exists.
    supported = ["fr-FR", "en-US"]
    assert _normalize_language("fr", supported) == "fr-FR"


def test_normalize_language_prefers_default_on_ambiguity():
    """When multiple prefix matches exist, the default-tag variant wins."""
    # Both 'pt-PT' and 'pt-BR' exist; default is pt-PT.
    supported = ["pt-BR", "pt-PT"]
    assert _normalize_language("pt", supported) == "pt-PT"
