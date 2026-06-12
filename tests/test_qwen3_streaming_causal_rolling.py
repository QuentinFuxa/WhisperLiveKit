"""Rolling decoder KV + speculative draft: parity on history-dependent fakes.

Ported from experiments/qwen3-causal/tests/test_decoder_rolling_kv.py @ 9d4b99a
(Part 2). The causal model under test comes from whisperlivekit.qwen3_streaming.
"""

import pytest

torch = pytest.importorskip("torch")

from whisperlivekit.qwen3_streaming.causal import (  # noqa: E402
    Qwen3ASRRealtimeQwenAudioCausalModel,
)
from whisperlivekit.qwen3_streaming.model import _split_prompt_template  # noqa: E402
from whisperlivekit.qwen3_streaming.streamer import trim_at_stop  # noqa: E402

from qwen3_streaming_fakes import (  # noqa: E402
    D_MODEL,
    VOCAB as KV_VOCAB,
    FakeCachingQwenTextModel,
    TinyQwenAudioTower,
    _Out,
    tiny_config,
)


PH = 7
TEMPLATE = [10, 11, PH, 12, 13]
EOS = 5
CONTROL_KW = dict(
    max_new_tokens=8,
    eos_token_id=EOS,
    suppress_token_ids=[2, 3],
    repetition_penalty=1.3,
    no_repeat_ngram_size=2,
)
PLAIN_KW = dict(max_new_tokens=8, eos_token_id=EOS)


class ConcatCache:
    """Stores the raw embedded prefix so crop is exact."""

    def __init__(self):
        self.embeds = None

    def get_seq_length(self):
        return 0 if self.embeds is None else int(self.embeds.shape[1])

    def crop(self, max_length):
        if self.embeds is not None:
            self.embeds = self.embeds[:, : int(max_length), :]


class FakeCroppableQwenTextModel(torch.nn.Module):
    """hidden[t] = mean(inputs[0..t]) like FakeCachingQwenTextModel, but the
    cache keeps the concatenated inputs so cropping is exact."""

    def __init__(self, vocab_size: int = KV_VOCAB, hidden_size: int = D_MODEL):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList()
        self.norm = torch.nn.Identity()
        self.forwarded_positions = 0
        self.calls = 0

    def forward(self, inputs_embeds, past_key_values=None, use_cache=False, **_kw):
        self.calls += 1
        self.forwarded_positions += int(inputs_embeds.shape[1])
        cache = past_key_values
        if cache is None and use_cache:
            cache = ConcatCache()
        previous = (
            cache.embeds
            if cache is not None and cache.embeds is not None
            else inputs_embeds.new_zeros(inputs_embeds.shape[0], 0, inputs_embeds.shape[2])
        )
        full = torch.cat([previous, inputs_embeds], dim=1)
        counts = torch.arange(
            1, full.shape[1] + 1, dtype=full.dtype, device=full.device
        ).view(1, -1, 1)
        means = full.cumsum(dim=1) / counts
        out = means[:, previous.shape[1] :, :]
        if cache is not None:
            cache.embeds = full
        return _Out(out, past_key_values=cache if use_cache else None)


def make_rolling_model(text_model=None) -> Qwen3ASRRealtimeQwenAudioCausalModel:
    torch.manual_seed(0)
    return Qwen3ASRRealtimeQwenAudioCausalModel(
        tiny_config(),
        qwen_model_id="fake",
        audio_tower=TinyQwenAudioTower(),
        text_model=text_model if text_model is not None else FakeCroppableQwenTextModel(),
        lm_head=torch.nn.Linear(D_MODEL, KV_VOCAB, bias=False),
        bos_token_id=1,
        wait_token_id=0,
    ).eval()


def expanded_prefix(audio_steps: int, template=None) -> list[int]:
    head, tail = _split_prompt_template(template or TEMPLATE, PH)
    return head + [PH] * audio_steps + tail


def reference_tokens(model, frame_hidden, template=None, **kw) -> list[int]:
    out = model.generate_full_hypothesis_from_cached_audio(
        frame_hidden,
        prefix_token_ids=expanded_prefix(int(frame_hidden.shape[1]), template),
        audio_placeholder_token_id=PH,
        use_decoder_kv_cache=True,
        **kw,
    )
    return out[0].tolist()


def rolling_tokens(model, state, frame_hidden, draft=None, template=None, **kw):
    tokens, stats = model.generate_full_hypothesis_rolling(
        frame_hidden,
        state=state,
        template_token_ids=template or TEMPLATE,
        audio_placeholder_token_id=PH,
        draft_token_ids=draft,
        **kw,
    )
    return tokens[0].tolist(), stats


def test_split_prompt_template():
    assert _split_prompt_template(TEMPLATE, PH) == ([10, 11], [12, 13])
    with pytest.raises(ValueError):
        _split_prompt_template([10, 11], PH)
    with pytest.raises(ValueError):
        _split_prompt_template([PH, 10, PH], PH)


@pytest.mark.parametrize("kw", [PLAIN_KW, CONTROL_KW], ids=["plain", "controls"])
def test_rolling_matches_full_reprefill_across_chunks(kw):
    model = make_rolling_model()
    torch.manual_seed(8)
    base = torch.randn(1, 9, D_MODEL)
    state = model.init_cached_audio_decode_state()
    head_len = 2
    audio_steps = 0
    for delta_steps in [3, 2, 0, 4]:
        audio_steps += delta_steps
        if audio_steps == 0:
            continue
        frame_hidden = base[:, :audio_steps, :]
        got, stats = rolling_tokens(model, state, frame_hidden, **kw)
        want = reference_tokens(model, frame_hidden, **kw)
        assert got == want, f"S={audio_steps}"
        assert stats["decoder_path"] == "rolling"
        # Cache restored to exactly [head + audio] between chunks.
        assert state.decoder is not None and not state.decoder.disabled
        assert state.decoder.audio_steps == audio_steps
        assert state.decoder.cache.get_seq_length() == head_len + audio_steps


def test_rolling_steady_state_forwards_far_fewer_positions():
    kw = dict(max_new_tokens=6, eos_token_id=EOS)
    model = make_rolling_model()
    torch.manual_seed(9)
    base = torch.randn(1, 12, D_MODEL)
    state = model.init_cached_audio_decode_state()

    rolling_tokens(model, state, base[:, :6, :], **kw)
    model.text_model.forwarded_positions = 0
    got, stats = rolling_tokens(model, state, base, **kw)

    tail_len = 2
    sequential = len(got) - (1 if got and got[-1] == EOS else 0)
    # Combined pass: 6 new audio + tail; then one forward per non-final pick.
    assert stats["prefill_positions"] == 6 + tail_len
    assert (
        model.text_model.forwarded_positions
        <= stats["prefill_positions"] + len(got)
    )
    # The legacy path would have re-forwarded head + all 12 audio + tail.
    assert stats["prefill_positions"] < 12 + 2 + tail_len


def test_rolling_rebuilds_on_template_change():
    kw = dict(max_new_tokens=6, eos_token_id=EOS)
    other_template = [9, 11, PH, 12, 13]
    model = make_rolling_model()
    torch.manual_seed(10)
    base = torch.randn(1, 8, D_MODEL)
    state = model.init_cached_audio_decode_state()

    _, first = rolling_tokens(model, state, base[:, :4, :], **kw)
    assert first["decoder_rebuilt"] is True
    _, second = rolling_tokens(model, state, base[:, :6, :], **kw)
    assert second["decoder_rebuilt"] is False
    got, third = rolling_tokens(model, state, base, template=other_template, **kw)
    assert third["decoder_rebuilt"] is True
    assert got == reference_tokens(model, base, template=other_template, **kw)


def test_rolling_falls_back_without_croppable_cache_and_disables():
    model = make_rolling_model(FakeCachingQwenTextModel())
    torch.manual_seed(11)
    frame_hidden = torch.randn(1, 4, D_MODEL)
    state = model.init_cached_audio_decode_state()

    got, stats = rolling_tokens(model, state, frame_hidden, **PLAIN_KW)
    assert stats["decoder_path"] == "full"
    assert state.decoder is not None and state.decoder.disabled
    assert got == reference_tokens(model, frame_hidden, **PLAIN_KW)

    # Disabled state short-circuits: the next chunk must not probe again, so
    # call count equals the legacy path's own count exactly.
    twin = make_rolling_model(FakeCachingQwenTextModel())
    reference_tokens(twin, frame_hidden, **PLAIN_KW)
    legacy_calls = twin.text_model.calls
    model.text_model.calls = 0
    rolling_tokens(model, state, frame_hidden, **PLAIN_KW)
    assert model.text_model.calls == legacy_calls


def test_rolling_falls_back_when_model_returns_no_cache():
    class NoCacheModel(FakeCroppableQwenTextModel):
        def forward(self, inputs_embeds, past_key_values=None, use_cache=False, **kw):
            out = super().forward(inputs_embeds, past_key_values, use_cache=False)
            out.past_key_values = None
            return out

    model = make_rolling_model(NoCacheModel())
    torch.manual_seed(12)
    frame_hidden = torch.randn(1, 3, D_MODEL)
    state = model.init_cached_audio_decode_state()
    got, stats = rolling_tokens(model, state, frame_hidden, max_new_tokens=5)
    assert stats["decoder_path"] == "full"
    assert state.decoder.disabled
    assert len(got) == 5  # legacy uncached loop produced the hypothesis


def nonstop_token(*avoid):
    banned = set(avoid) | {EOS}
    for token_id in range(KV_VOCAB):
        if token_id not in banned:
            return token_id
    raise AssertionError("no spare token")


@pytest.mark.parametrize("kw", [PLAIN_KW, CONTROL_KW], ids=["plain", "controls"])
def test_draft_cases_are_lossless(kw):
    model = make_rolling_model()
    torch.manual_seed(13)
    base = torch.randn(1, 10, D_MODEL)

    def fresh_state_tokens(draft, frame_hidden, **extra):
        state = model.init_cached_audio_decode_state()
        merged = dict(kw)
        merged.update(extra)
        # Warm the cache on a prior chunk so drafts ride the steady-state path.
        rolling_tokens(model, state, frame_hidden[:, :-2, :], **merged)
        return rolling_tokens(model, state, frame_hidden, draft=draft, **merged)

    reference = reference_tokens(model, base, **kw)
    trimmed = trim_at_stop(list(reference), EOS)

    # Full accept: the draft IS the (trimmed) reference.
    got, stats = fresh_state_tokens(trimmed, base)
    assert got == reference
    assert stats["draft_tokens"] == len(trimmed)
    assert stats["draft_accepted"] == len(trimmed)
    if len(trimmed) < kw["max_new_tokens"]:
        assert stats["draft_all_accepted"] is True

    # Divergence mid-draft.
    if len(trimmed) >= 2:
        perturbed = list(trimmed)
        perturbed[1] = nonstop_token(perturbed[1])
        got, stats = fresh_state_tokens(perturbed, base)
        assert got == reference
        assert stats["draft_accepted"] <= 1

    # Draft fully contradicted.
    junk = [nonstop_token(reference[0])] * 4
    got, stats = fresh_state_tokens(junk, base)
    assert got == reference
    assert stats["draft_accepted"] == 0

    # Empty draft.
    got, stats = fresh_state_tokens([], base)
    assert got == reference
    assert stats["draft_tokens"] == 0

    # Draft longer than the budget: only budget tokens are considered.
    small_budget = dict(kw)
    small_budget["max_new_tokens"] = 3
    short_reference = reference_tokens(model, base, **small_budget)
    state = model.init_cached_audio_decode_state()
    rolling_tokens(model, state, base[:, :-2, :], **small_budget)
    got, stats = rolling_tokens(
        model, state, base, draft=trimmed, **small_budget
    )
    assert got == short_reference
    assert stats["draft_tokens"] <= 3

    # Draft with a stop token inside is trimmed defensively.
    if trimmed:
        with_stop = trimmed[:1] + [EOS] + trimmed[1:]
        got, stats = fresh_state_tokens(with_stop, base)
        assert got == reference
        assert stats["draft_tokens"] == 1


def test_draft_corrected_pick_can_be_eos():
    model = make_rolling_model()
    found = False
    for seed in range(20):
        torch.manual_seed(40 + seed)
        base = torch.randn(1, 6, D_MODEL)
        reference = reference_tokens(model, base, **PLAIN_KW)
        trimmed = trim_at_stop(list(reference), EOS)
        if reference and reference[-1] == EOS and len(trimmed) >= 1:
            found = True
            break
    assert found, "no seed produced an eos-terminated reference"
    # Draft extends past where the model wants to stop: at position
    # len(trimmed) the corrected pick IS eos and generation must end there.
    draft = trimmed + [nonstop_token(), nonstop_token()]
    state = model.init_cached_audio_decode_state()
    got, stats = rolling_tokens(model, state, base, draft=draft, **PLAIN_KW)
    assert got == reference
    assert stats["draft_accepted"] == len(trimmed)
