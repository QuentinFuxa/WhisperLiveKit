"""Local MT lifecycle through the same worker used by WebSocket sessions."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from whisperlivekit.timed_objects import ASRToken, ChangeSpeaker, Silence, State
from whisperlivekit.translation_mlx_llm_mt import MlxLlmTranslation


def _test_calibration(tmp_path):
    engine = MlxLlmTranslation(source_language="en", target_language="zh", warmup=False)
    data = {
        "model": engine._config.repo, "direction": "en-zh", "num_layers": 2, "num_heads": 4,
        "used_pairs": 100, "token_alignment_heads": [
            {"layer": layer, "head": head, "ts": 0.5} for layer in range(2) for head in range(4)
        ],
        "promotion_gate": {"eligible_for_promotion": True},
        "stability_checks": [{"stable_vs_full": True, "max_abs_ts_delta_vs_full": 0.01}] * 3,
        "runtime": {"backend": "mlx-lm", "model_revision": "a" * 40, "source_sha256": "0" * 64,
                    "quantization": {"bits": 8}, "prompt": engine._prompt},
    }
    path = tmp_path / "heads.json"
    path.write_text(json.dumps(data))
    return path


def test_simultaneous_requires_matching_calibration_and_keeps_session_options(tmp_path):
    from whisperlivekit.translation_mlx_llm_mt_simul import MlxLlmTranslationSimul

    with pytest.raises(ValueError, match="calibration"):
        MlxLlmTranslationSimul(source_language="fr", target_language="en", warmup=False)
    path = _test_calibration(tmp_path)
    engine = MlxLlmTranslationSimul(source_language="en", target_language="zh", warmup=False,
                                   calibration_file=path, simul_soft_max_s=1.25, simul_hard_max_s=7)
    session = engine.new_session()
    assert (session._simul_soft_max_s, session._simul_hard_max_s) == (1.25, 7)
    assert session._config.revision == "a" * 40
    with pytest.raises(ValueError, match="direction"):
        engine.new_session("fr")
    data = json.loads(path.read_text())
    data["model"] = "unrelated/Hy-MT2-1.8B-8bit"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="exact model"):
        engine.new_session()


@pytest.mark.asyncio
async def test_simultaneous_partial_failure_pause_and_eof(tmp_path, monkeypatch):
    from whisperlivekit.audio_processor import SENTINEL, AudioProcessor
    from whisperlivekit.timed_objects import HypothesisTail
    from whisperlivekit.translation_mlx_llm_mt_simul import MlxLlmTranslationSimul

    engine = MlxLlmTranslationSimul(source_language="en", target_language="zh", warmup=False,
                                   calibration_file=_test_calibration(tmp_path))
    monkeypatch.setattr(engine, "_translate_simul", lambda source, committed: "您好")
    monkeypatch.setattr(engine, "_translate_text", lambda text: text.upper())
    engine.insert_tokens([ASRToken(0, 1, "Hello"), HypothesisTail(1, 2, "world")])
    _, buffer = engine.process()
    assert buffer.text == "您好"

    def unavailable(*args):
        raise RuntimeError("missing attention")

    monkeypatch.setattr(engine, "_translate_simul", unavailable)
    engine.insert_tokens([ASRToken(1, 2, "world")])
    _, buffer = engine.process()
    assert buffer.text == "您好" and "missing attention" in engine.error
    processor = SimpleNamespace(translation_queue=asyncio.Queue(), translation=engine,
                                state=State(), lock=asyncio.Lock())
    for event in [Silence(start=2, end=3, is_starting=True, has_ended=True),
                  ASRToken(3, 7, "One."), ASRToken(7, 11, "Two."), ASRToken(11, 12, "Tail"), SENTINEL]:
        await processor.translation_queue.put(event)
    await asyncio.wait_for(AudioProcessor.translation_processor(processor), 5)
    assert " ".join(part.text for part in processor.state.new_translation) == "HELLO WORLD ONE. TWO. TAIL"
    assert not processor.state.new_translation_buffer.text and not engine.error
    assert engine.finish()[0] is None


def test_capture_cached_prefill_isolation_and_missing_attention():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    import numpy as np
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.hunyuan_v1_dense import Model, ModelArgs

    from whisperlivekit.simul_mt_capture import apply_commit_policy, capture_attention, snapshot_capture

    mx.random.seed(4)
    args = ModelArgs(model_type="hunyuan_v1_dense", vocab_size=32, hidden_size=32, num_hidden_layers=2,
                     intermediate_size=64, num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6)
    model = Model(args)
    ids = mx.array([[1, 2, 3, 4, 5]])
    full = np.array(model(ids))
    cache = make_prompt_cache(model)
    mx.eval(model(ids[:, :3], cache=cache))
    originals = [layer.self_attn for layer in model.model.layers]
    heads = ((0, 1), (1, 2))
    with capture_attention(model, heads, target_start=3) as capture:
        actual = np.array(model(ids[:, 3:], cache=cache))
        snapshot = snapshot_capture(capture)
    np.testing.assert_allclose(actual, full[:, 3:], rtol=1e-5, atol=1e-5)
    for entries in snapshot.values():
        assert entries[0].query_start == 3
        weights = entries[0].weights
        assert weights[0, 4] == 0 and np.all(weights[0, :4] > 0)
        assert np.all(weights[1, :5] > 0)
        assert not weights.flags.writeable
    before = snapshot[(0, 1)][0].weights.copy()
    with pytest.raises(RuntimeError, match="interrupted"):
        with capture_attention(model, [(0, 2)]) as other:
            mx.eval(model(ids))
            assert set(other) == {(0, 2)}
            raise RuntimeError("interrupted")
    assert all(original is layer.self_attn for original, layer in zip(originals, model.model.layers))
    np.testing.assert_array_equal(snapshot[(0, 1)][0].weights, before)
    assert apply_commit_policy(snapshot, heads, 2, 3, 0, 3, 3) == 2
    with pytest.raises(RuntimeError, match="Missing alignment"):
        apply_commit_policy({}, heads, 2, 3, 0, 3, 0)


@pytest.mark.asyncio
async def test_mlx_translation_drains_batches_boundaries_and_eof(monkeypatch):
    from whisperlivekit.audio_processor import SENTINEL, AudioProcessor

    calls = []

    def translate(self, text):
        calls.append((self._source_language, self._target_language, text))
        return text.upper()

    monkeypatch.setattr(MlxLlmTranslation, "_translate_text", translate)
    engine = MlxLlmTranslation(source_language="zh", target_language="en", warmup=False)
    from whisperlivekit.translation import session_translation_factory

    client = session_translation_factory(SimpleNamespace(lan="zh"), engine, "de", "fr")
    other = engine.new_session()
    processor = SimpleNamespace(
        translation_queue=asyncio.Queue(), translation=client,
        state=State(), lock=asyncio.Lock(),
    )
    for event in [
        ASRToken(0, 1, "One."), ASRToken(1, 2, " Two."), ASRToken(2, 3, " Three."),
        ASRToken(3, 4, "Pause"),
        Silence(start=4, end=5, is_starting=True, has_ended=True),
        ASRToken(5, 6, "Speaker"),
        ChangeSpeaker(start=6, speaker=2),
        ASRToken(6, 6.5, "Last"), ASRToken(6.5, 7, "words"),
        SENTINEL,
    ]:
        await processor.translation_queue.put(event)
    await asyncio.wait_for(AudioProcessor.translation_processor(processor), 5)
    assert " ".join(t.text for t in processor.state.new_translation) == "ONE. TWO. THREE. PAUSE SPEAKER LAST WORDS"
    assert not processor.state.new_translation_buffer.text
    assert client.finish()[0] is None  # no duplicate on a repeated drain
    assert all(source == "fr" and target == "de" for source, target, _ in calls)
    assert other.finish()[0] is None
    assert (other._source_language, other._target_language) == ("zh", "en")


def test_mlx_translation_keeps_failed_sentence_for_retry(monkeypatch):
    client = MlxLlmTranslation(warmup=False)
    failed = False

    def translate(text):
        nonlocal failed
        if text == "Two." and not failed:
            failed = True
            raise RuntimeError("decode failed")
        return text.upper()

    monkeypatch.setattr(client, "_translate_text", translate)
    client.insert_tokens([ASRToken(0, 1, "One."), ASRToken(1, 2, "Two."), ASRToken(2, 3, "Tail")])
    result, buffer = client.process()
    assert result.text == "ONE."
    assert "decode failed" in client.error
    assert not buffer.text
    result, buffer = client.finish()
    assert result.text == "TWO. TAIL"
    assert (result.start, result.end) == (1, 3)
    assert not client.error and not buffer.text
    assert client.finish()[0] is None


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
