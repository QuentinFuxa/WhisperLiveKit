"""Tests for the mlx-llm-mt translation backend.

Covers:
  - The config registry has multiple model families with distinct repos
    and prompt templates. A second config loads with a different repo and
    prompt without new code.
  - ``validate_buffer_and_reset`` does not double the output: it returns
    the translation once at a silence boundary, and returns empty when
    there is nothing to flush.

These tests mock ``_translate_text`` so they run without mlx-lm or a model
download — they exercise the buffer and contract logic, not the model.
"""
from __future__ import annotations

import sys

import pytest

from whisperlivekit.timed_objects import ASRToken, Translation
from whisperlivekit.translation_mlx_llm_mt import (
    MTX_MODEL_CONFIGS,
    MlxLlmMtModelConfig,
    MlxLlmTranslation,
)

# ---------------------------------------------------------------------------
# Config registry: multiple model families from config, not code
# ---------------------------------------------------------------------------

def test_registry_has_multiple_config_families():
    """The registry has at least two config families with distinct repos and
    prompt kinds."""
    hy = MTX_MODEL_CONFIGS["hy-mt2-1.8b-8bit"]
    tg = MTX_MODEL_CONFIGS["translategemma-4b-it-4bit"]
    assert hy.repo != tg.repo, "repos must differ across config families"
    assert hy.prompt_kind != tg.prompt_kind, "prompt kinds must differ"


def test_second_config_constructs_without_new_code():
    """Constructing the backend with a second config (warmup=False, no model
    download) resolves the right config — different repo and prompt kind — purely
    from the config dict, with no new code."""
    backend = MlxLlmTranslation(
        model_id="translategemma-4b-it-4bit", target_language="en", warmup=False
    )
    assert backend._config.repo == "mlx-community/translategemma-4b-it-4bit"
    assert backend._config.prompt_kind == "structured_chat"
    hy = MlxLlmTranslation(
        model_id="hy-mt2-1.8b-8bit", target_language="en", warmup=False
    )
    assert hy._config.repo == "mlx-community/Hy-MT2-1.8B-8bit"
    assert backend._config.repo != hy._config.repo
    assert backend._config.prompt_kind != hy._config.prompt_kind


def test_unknown_model_raises():
    """An unknown model id raises a clear error listing available configs."""
    with pytest.raises(ValueError, match="Unknown mlx-llm-mt model"):
        MlxLlmTranslation(model_id="nonexistent-model", warmup=False)


def test_config_is_data_not_code():
    """Adding a new model is a config-dict entry, not a subclass."""
    custom = MlxLlmMtModelConfig(
        repo="mlx-community/some-other-mt-4bit",
        prompt_template="Translate to {target_lang}: {text}",
        eos_token="<|end|>",
    )
    assert custom.repo == "mlx-community/some-other-mt-4bit"
    assert custom.eos_token == "<|end|>"


def test_registry_populated_by_profiles_module():
    """The registry is populated by the profiles module at import time."""
    # All six Hunyuan-MT entries must be present.
    for name in (
        "hy-mt2-1.8b-8bit", "hy-mt2-1.8b-4bit",
        "hy-mt2-7b-4bit", "hy-mt2-7b-8bit",
        "hunyuan-mt-7b-4bit", "hunyuan-mt-7b-8bit",
    ):
        assert name in MTX_MODEL_CONFIGS, f"{name} missing from registry"
    # TranslateGemma entry must be present.
    assert "translategemma-4b-it-4bit" in MTX_MODEL_CONFIGS
    # Hunyuan prompt lives in the config, not as a module-level constant.
    hy = MTX_MODEL_CONFIGS["hy-mt2-1.8b-8bit"]
    assert "把下面的文本翻译成" in hy.prompt_template
    # TranslateGemma uses structured_chat, not a text template.
    tg = MTX_MODEL_CONFIGS["translategemma-4b-it-4bit"]
    assert tg.prompt_kind == "structured_chat"


# ---------------------------------------------------------------------------
# validate_buffer_and_reset does not double the output
# ---------------------------------------------------------------------------

def _make_backend(model_id="hy-mt2-1.8b-8bit"):
    """Create a backend with _translate_text mocked (no model download)."""
    backend = MlxLlmTranslation(model_id=model_id, target_language="en", warmup=False)
    backend._translate_text = lambda text: f"[EN:{text}]"
    return backend


def _token(text, start, end):
    return ASRToken(start=start, end=end, text=text)


def test_process_emits_translation_for_closed_segment():
    """``process()`` translates a punctuation-closed segment and returns it."""
    b = _make_backend()
    b.insert_tokens([_token("你好", 0.0, 0.5), _token("。", 0.5, 0.6)])
    tr, buf = b.process()
    assert tr is not None
    assert tr.text == "[EN:你好。]"
    assert isinstance(tr, Translation)


def test_validate_flushes_open_segment_once():
    """At a silence boundary, ``validate_buffer_and_reset`` flushes the open
    segment and returns the translation once."""
    b = _make_backend()
    b.insert_tokens([_token("你好", 0.0, 0.5)])
    tr, buf = b.validate_buffer_and_reset()
    assert tr.text == "[EN:你好]"
    assert buf.text == "[EN:你好]"


def test_validate_does_not_double_after_process():
    """After ``process()`` has already emitted a translation for a closed
    segment, ``validate_buffer_and_reset`` must not return it again. It
    returns empty, not the stale buffer."""
    b = _make_backend()
    b.insert_tokens([_token("你好", 0.0, 0.5), _token("。", 0.5, 0.6)])
    tr, buf = b.process()
    assert tr is not None
    assert tr.text == "[EN:你好。]"
    tr2, buf2 = b.validate_buffer_and_reset()
    assert not tr2.text, "validate must return empty, not a duplicate"
    assert not buf2.text, "buffer must be empty, not the stale translation"


def test_validate_empty_when_nothing_buffered():
    """``validate_buffer_and_reset`` on a fresh backend returns empty."""
    b = _make_backend()
    tr, buf = b.validate_buffer_and_reset()
    assert not tr.text
    assert not buf.text


def test_process_returns_none_when_no_closed_segment():
    """``process()`` returns (None, buffer) when there is no closed segment.
    The buffer is the running untranslated partial."""
    b = _make_backend()
    b.insert_tokens([_token("你好", 0.0, 0.5)])
    tr, buf = b.process()
    assert tr is None
    assert buf.text == "你好"


def test_insert_silence_noop():
    """``insert_silence`` is a no-op."""
    b = _make_backend()
    b.insert_tokens([_token("测试", 0.0, 0.5)])
    b.insert_silence(1.0)
    tr, buf = b.process()
    assert tr is None
    assert buf.text == "测试"



# ---------------------------------------------------------------------------
# Alias (backward-compat for --translation-backend hunyuan-mlx)
# ---------------------------------------------------------------------------

def test_hunyuan_mlx_reexports():
    """The ``translation_hunyuan_mlx`` module re-exports the class so existing
    imports keep working."""
    from whisperlivekit.translation_hunyuan_mlx import HunyuanMlxTranslation

    assert HunyuanMlxTranslation is MlxLlmTranslation


# ---------------------------------------------------------------------------
# In-loop early stop at the Hunyuan placeholder token (mlx decode layer)
# ---------------------------------------------------------------------------

_HY_PLACEHOLDER = "<｜hy_place▁holder▁no▁2｜>"


class _Chunk:
    """Minimal stand-in for mlx_lm's GenerationResponse."""

    def __init__(self, token, text):
        self.token = token
        self.text = text


def _install_fake_mlx_lm(monkeypatch, chunks, consumed):
    """Inject stub ``mlx_lm`` / ``mlx_lm.sample_utils`` whose ``stream_generate``
    yields ``chunks`` and appends each yielded chunk to ``consumed`` (mutable),
    so the test can assert how much of the stream the decode loop consumed
    before breaking out."""
    import types

    def fake_stream_generate(model, tokenizer, prompt=None, max_tokens=None, **kw):
        for c in chunks:
            consumed.append(c)
            yield c

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.stream_generate = fake_stream_generate
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **kw: None
    sample_utils.make_logits_processors = lambda **kw: []
    mlx_lm.sample_utils = sample_utils
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)


class _Tokenizer:
    """Stub tokenizer: configurable placeholder id sequence, no real vocab."""

    def __init__(self, placeholder_ids, eos_token=""):
        self._placeholder_ids = placeholder_ids
        self.eos_token = eos_token

    def encode(self, text, add_special_tokens=True):
        if text == _HY_PLACEHOLDER:
            return list(self._placeholder_ids)
        return [1]  # any non-placeholder prompt encodes to one id

    def decode(self, ids, skip_special_tokens=True):
        return ""

    def apply_chat_template(self, messages, add_generation_prompt=False):
        return [1, 2, 3]


def _backend_with_tokenizer(monkeypatch, tokenizer, chunks):
    """Backend whose model/tokenizer come from the stub registry; returns
    (backend, consumed) where consumed lists the chunks the decode loop
    actually pulled before stopping."""
    backend = MlxLlmTranslation(model_id="hy-mt2-1.8b-8bit", target_language="en", warmup=False)
    backend._eos_token = ""  # exercise only the placeholder-stop path
    consumed = []
    _install_fake_mlx_lm(monkeypatch, chunks, consumed)
    MlxLlmTranslation._MODEL_CACHE["stub-repo"] = (object(), tokenizer)
    monkeypatch.setattr(
        type(backend), "_ensure_model", classmethod(lambda cls, config: (object(), tokenizer))
    )
    return backend, consumed


def test_early_stops_on_single_id_placeholder(monkeypatch):
    """Hy-MT2-1.8B case: placeholder is one token id — decode stops at that id
    and the hallucinated tail is never consumed."""
    tail_chunks = [_Chunk(9000 + i, "废" if i % 2 else "话") for i in range(20)]
    chunks = [
        _Chunk(500, "Hello"),
        _Chunk(501, " world"),
        _Chunk(120020, _HY_PLACEHOLDER),  # the placeholder, one id
        *tail_chunks,
    ]
    tok = _Tokenizer(placeholder_ids=[120020])
    backend, consumed = _backend_with_tokenizer(monkeypatch, tok, chunks)

    out = backend._translate_text("你好。")

    assert out == "Hello world"
    # Stop fired at the placeholder: only 3 chunks were consumed, not the tail.
    assert len(consumed) == 3


def test_early_stops_on_fragmented_placeholder(monkeypatch):
    """Fragmented (multi-id) placeholder: the rolling id window fires once the
    full sequence has been emitted."""
    frag_ids = [27, 15755, 250]
    frag_ids = [27, 15755, 250]
    chunks = [
        _Chunk(500, "Hi"),
        _Chunk(frag_ids[0], "<"),
        _Chunk(frag_ids[1], "｜hy"),
        _Chunk(frag_ids[2], "_place▁holder▁no▁2｜>"),
        _Chunk(7000, "junk"),
        _Chunk(9001, "更多"),
    ]
    tok = _Tokenizer(placeholder_ids=frag_ids)
    backend, consumed = _backend_with_tokenizer(monkeypatch, tok, chunks)

    out = backend._translate_text("你好。")

    assert out == "Hi"
    assert len(consumed) == 4  # stopped right after the last fragment


def test_no_stop_id_falls_back_to_full_decode_and_string_strip(monkeypatch):
    """Tokenizer that fragments the placeholder beyond the window cap: no
    in-loop stop (stream runs to completion), and the post-hoc string strip
    still truncates the hallucinated tail."""
    chunks = [
        _Chunk(500, "Hi "),
        _Chunk(700, "there "),
        _Chunk(901, _HY_PLACEHOLDER),
        _Chunk(902, "hallucinated tail"),
    ]
    tok = _Tokenizer(placeholder_ids=list(range(40)))  # >16 ids → no stop check
    backend, consumed = _backend_with_tokenizer(monkeypatch, tok, chunks)

    out = backend._translate_text("你好。")

    assert out == "Hi there"  # string strip still cut at the placeholder
    assert len(consumed) == len(chunks)  # nothing stopped in-loop (expected)


def test_early_stop_does_not_affect_clean_output(monkeypatch):
    """No placeholder in the stream: the loop runs to EOS-ish end and the
    output is untouched by the strip."""
    chunks = [_Chunk(500, "Hello "), _Chunk(501, "world")]
    tok = _Tokenizer(placeholder_ids=[120020])
    backend, consumed = _backend_with_tokenizer(monkeypatch, tok, chunks)

    out = backend._translate_text("你好。")

    assert out == "Hello world"
    assert len(consumed) == 2


# ---------------------------------------------------------------------------
# Per-session state isolation: new_session shares the model cache but not
# mutable translation state (buffer, pending finals, metrics).
# ---------------------------------------------------------------------------

def test_new_session_shares_model_cache_not_state(monkeypatch):
    """Two sessions created via new_session must not share per-instance state
    (_buffer_tokens, _pending_finals, _last_buffer, metrics)."""
    # Stub the model load so we don't need real MLX.
    monkeypatch.setattr(MlxLlmTranslation, "_warmup", lambda self: None)
    monkeypatch.setattr(MlxLlmTranslation, "_ensure_model", lambda cls, config: ("model", "tokenizer"))

    server = MlxLlmTranslation(model_id="hy-mt2-1.8b-8bit", warmup=False)
    session_a = server.new_session("en")
    session_b = server.new_session("ja")

    # Different target languages
    assert session_a._target_language == "en"
    assert session_b._target_language == "ja"

    # Different per-instance state objects
    assert session_a._buffer_tokens is not session_b._buffer_tokens
    assert session_a._pending_finals is not session_b._pending_finals
    assert session_a._last_buffer is not session_b._last_buffer

    # Mutating session A's state must not affect session B
    session_a._buffer_tokens.append("token_a")
    session_a._pending_finals.append(("text_a", 0.0, 1.0))
    assert len(session_b._buffer_tokens) == 0
    assert len(session_b._pending_finals) == 0


def test_new_session_shares_model_cache(monkeypatch):
    """new_session must share the model cache (_MODEL_CACHE), not reload."""
    cache = {}
    monkeypatch.setattr(MlxLlmTranslation, "_warmup", lambda self: None)

    MlxLlmTranslation(model_id="hy-mt2-1.8b-8bit", warmup=False)._config
    def fake_ensure(cls, config):
        key = config.repo
        if key not in cache:
            cache[key] = ("model", "tokenizer")
        return cache[key]
    monkeypatch.setattr(MlxLlmTranslation, "_MODEL_CACHE", cache)
    monkeypatch.setattr(MlxLlmTranslation, "_ensure_model", classmethod(fake_ensure))

    server = MlxLlmTranslation(model_id="hy-mt2-1.8b-8bit", warmup=False)
    # Populate the cache via _ensure_model
    MlxLlmTranslation._ensure_model(server._config)

    session = server.new_session("en")
    assert session._model_id == server._model_id
    # The cache is shared — new_session didn't reload
    assert len(cache) == 1


# ---------------------------------------------------------------------------
# Session translation factory: the MLX backend routes through new_session,
# not the nllw fallback.
# ---------------------------------------------------------------------------

def test_session_translation_factory_mlx(monkeypatch):
    """session_translation_factory must route MlxLlmTranslation through
    new_session (not fall through to nllw)."""
    import types

    from whisperlivekit.translation import session_translation_factory

    monkeypatch.setattr(MlxLlmTranslation, "_warmup", lambda self: None)
    monkeypatch.setattr(MlxLlmTranslation, "_ensure_model",
                        classmethod(lambda cls, config: ("model", "tokenizer")))

    server_model = MlxLlmTranslation(model_id="hy-mt2-1.8b-8bit", warmup=False)
    args = types.SimpleNamespace(lan="zh", target_language="en")

    session = session_translation_factory(args, server_model, "ja")

    assert isinstance(session, MlxLlmTranslation)
    assert session._target_language == "ja"
    assert session is not server_model  # per-session, not shared
    assert session._buffer_tokens is not server_model._buffer_tokens


def test_session_translation_factory_mlx_default_target(monkeypatch):
    """Default target (empty string) falls back to the server-wide target."""
    import types

    from whisperlivekit.translation import session_translation_factory

    monkeypatch.setattr(MlxLlmTranslation, "_warmup", lambda self: None)
    monkeypatch.setattr(MlxLlmTranslation, "_ensure_model",
                        classmethod(lambda cls, config: ("model", "tokenizer")))

    server_model = MlxLlmTranslation(
        model_id="hy-mt2-1.8b-8bit", target_language="en", warmup=False)
    args = types.SimpleNamespace(lan="zh", target_language="en")

    session = session_translation_factory(args, server_model, "")
    assert session._target_language == "en"


# ---------------------------------------------------------------------------
# has_wer is a property, not a method — 'if report.has_wer' must evaluate to
# bool, not a truthy method object.
# ---------------------------------------------------------------------------

def test_has_wer_is_property():
    """has_wer must be a property (not a method) so `if report.has_wer` tests
    the bool, not the method object (which is always truthy)."""
    from whisperlivekit.benchmark.metrics import BenchmarkReport, SampleResult

    # All-N/A: no applicable WER
    results_na = [SampleResult(sample_name="s1", language="en", category="test",
                            duration_s=1.0, wer=0.0, wer_details={},
                            processing_time_s=0.1, rtf=0.5, wer_applicable=False)]
    report_na = BenchmarkReport(backend="test", model_size="0", results=results_na)
    assert not report_na.has_wer  # property → bool, not method

    # Mixed: some applicable, some not
    results_mixed = [
        SampleResult(sample_name="s1", language="en", category="test",
                     duration_s=1.0, wer=0.15, wer_details={},
                     processing_time_s=0.1, rtf=0.5, wer_applicable=True),
        SampleResult(sample_name="s2", language="en", category="test",
                     duration_s=1.0, wer=0.0, wer_details={},
                     processing_time_s=0.1, rtf=0.5, wer_applicable=False),
    ]
    report_mixed = BenchmarkReport(backend="test", model_size="0", results=results_mixed)
    assert report_mixed.has_wer  # at least one applicable

    # All applicable
    results_all = [
        SampleResult(sample_name="s1", language="en", category="test",
                     duration_s=1.0, wer=0.10, wer_details={},
                     processing_time_s=0.1, rtf=0.5, wer_applicable=True),
        SampleResult(sample_name="s2", language="en", category="test",
                     duration_s=1.0, wer=0.20, wer_details={},
                     processing_time_s=0.1, rtf=0.5, wer_applicable=True),
    ]
    report_all = BenchmarkReport(backend="test", model_size="0", results=results_all)
    assert report_all.has_wer
