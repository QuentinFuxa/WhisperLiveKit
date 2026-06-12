"""Decoder-side per-chunk compute optimizations.

Part 1 — `_GreedyControlSession` must be value-equivalent to the legacy
control pipeline (`_apply_repetition_controls_to_logits` kept in source as the
reference spec, plus the deleted `_control_logits_and_pick` preamble/stop
handling reproduced here verbatim).

Part 2 — rolling audio-prefix decoder KV + lossless speculative draft verify:
parity with the full re-prefill path on history-dependent fakes.
"""

import pytest

torch = pytest.importorskip("torch")

from qwen3_streaming.native_realtime_model import (  # noqa: E402
    _apply_repetition_controls_to_logits,
    _GreedyControlSession,
)

VOCAB = 32


def legacy_controls(
    logits,
    histories,
    *,
    suppress_ids,
    wait_token_id,
    repetition_penalty,
    no_repeat_ngram_size,
    max_consecutive_text_tokens,
):
    """Verbatim preamble of the deleted `_control_logits_and_pick`."""
    if suppress_ids:
        logits = logits.clone()
        logits[:, suppress_ids] = -torch.inf
    token_history = [[int(t) for t in row] for row in histories]
    consecutive = (
        torch.tensor(
            [len(row) for row in token_history],
            dtype=torch.long,
            device=logits.device,
        )
        if max_consecutive_text_tokens > 0
        else None
    )
    return _apply_repetition_controls_to_logits(
        logits,
        token_history=token_history,
        consecutive_text_tokens=consecutive,
        repetition_penalty=float(repetition_penalty),
        no_repeat_ngram_size=int(no_repeat_ngram_size),
        max_consecutive_text_tokens=int(max_consecutive_text_tokens),
        wait_token_id=wait_token_id,
    )


def legacy_pick(
    logits,
    histories,
    finished,
    *,
    stop_ids,
    suppress_ids,
    wait_token_id,
    eos_token_id,
    repetition_penalty,
    no_repeat_ngram_size,
    max_consecutive_text_tokens,
):
    """Verbatim argmax/stop handling of the deleted `_control_logits_and_pick`."""
    device = logits.device
    logits = legacy_controls(
        logits,
        histories,
        suppress_ids=suppress_ids,
        wait_token_id=wait_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        max_consecutive_text_tokens=max_consecutive_text_tokens,
    )
    next_token = logits.argmax(dim=-1)
    if stop_ids:
        if bool(finished.any().item()):
            stop_fill_id = (
                int(eos_token_id) if eos_token_id is not None else min(stop_ids)
            )
            next_token = torch.where(
                finished, torch.full_like(next_token, stop_fill_id), next_token
            )
        finished_next = torch.tensor(
            [int(t) in stop_ids for t in next_token.tolist()],
            dtype=torch.bool,
            device=device,
        )
        finished = finished | finished_next
    return next_token, finished


def make_session(histories, **overrides):
    kwargs = dict(
        batch_size=len(histories),
        device=torch.device("cpu"),
        vocab_size=VOCAB,
        stop_ids=set(),
        suppress_ids=[],
        control_wait_token_id=None,
        eos_token_id=None,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        max_consecutive_text_tokens=0,
        initial_histories=histories,
    )
    kwargs.update(overrides)
    return _GreedyControlSession(**kwargs)


CONTROL_CASES = [
    # (suppress, wait, penalty, ngram, maxcons, histories)
    pytest.param([], None, 1.0, 0, 0, [[], []], id="all-off"),
    pytest.param([2, 3], None, 1.0, 0, 0, [[2, 4], []], id="suppress-only"),
    pytest.param(
        [2, 3], None, 1.3, 0, 0, [[2, 4, 4, 9], [5]], id="penalty-overlaps-suppress"
    ),
    pytest.param(
        [], None, 1.3, 2, 0, [[4, 9, 4, 9, 4], [7, 7, 7]], id="penalty-plus-ngram"
    ),
    pytest.param(
        [9], None, 1.0, 2, 0, [[4, 9, 4, 9, 4], []], id="banned-equals-suppressed"
    ),
    pytest.param([], 0, 1.15, 3, 3, [[4, 5, 6, 8], [1]], id="maxcons-trigger"),
    pytest.param([], None, 1.15, 3, 3, [[4, 5, 6, 8], [1]], id="maxcons-wait-none"),
    pytest.param(
        [], None, 1.3, 0, 0, [[-100, 4, VOCAB + 5], []], id="out-of-vocab-history"
    ),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "suppress,wait,penalty,ngram,maxcons,histories", CONTROL_CASES
)
def test_controlled_logits_matches_legacy_spec(
    dtype, suppress, wait, penalty, ngram, maxcons, histories
):
    torch.manual_seed(0)
    logits = torch.randn(len(histories), VOCAB, dtype=dtype)
    logits[0, 1] = -torch.inf  # pre-suppressed value flowing into penalty
    expected = legacy_controls(
        logits,
        histories,
        suppress_ids=suppress,
        wait_token_id=wait,
        repetition_penalty=penalty,
        no_repeat_ngram_size=ngram,
        max_consecutive_text_tokens=maxcons,
    )
    session = make_session(
        histories,
        suppress_ids=suppress,
        control_wait_token_id=wait,
        repetition_penalty=penalty,
        no_repeat_ngram_size=ngram,
        max_consecutive_text_tokens=maxcons,
    )
    # Cached-index path (sequential decode) and explicit-history path (draft
    # verification) must both match the legacy spec bit for bit.
    assert torch.equal(session.controlled_logits(logits), expected)
    assert torch.equal(session.controlled_logits(logits, histories), expected)


def test_controlled_logits_no_controls_returns_input():
    logits = torch.randn(1, VOCAB)
    session = make_session([[]])
    assert session.controlled_logits(logits) is logits


def test_pick_sequence_matches_legacy_pick():
    torch.manual_seed(1)
    batch = 2
    histories = [[4, 9], []]
    stop_ids = {5, 6}
    common = dict(
        stop_ids=stop_ids,
        suppress_ids=[2],
        wait_token_id=0,
        eos_token_id=5,
        repetition_penalty=1.3,
        no_repeat_ngram_size=2,
        max_consecutive_text_tokens=6,
    )
    session = make_session(
        histories,
        stop_ids=stop_ids,
        suppress_ids=[2],
        control_wait_token_id=0,
        eos_token_id=5,
        repetition_penalty=1.3,
        no_repeat_ngram_size=2,
        max_consecutive_text_tokens=6,
    )
    legacy_histories = [list(row) for row in histories]
    finished = torch.zeros(batch, dtype=torch.bool)
    for step in range(10):
        logits = torch.randn(batch, VOCAB)
        if step == 3:
            logits[0, 5] = 100.0  # force row 0 to finish -> stop-fill kicks in
        expected_token, finished = legacy_pick(
            logits, legacy_histories, finished, **common
        )
        for row, token_id in enumerate(expected_token.tolist()):
            legacy_histories[row].append(int(token_id))
        got_token, all_finished = session.pick(logits)
        assert torch.equal(got_token, expected_token), f"step {step}"
        assert all_finished == bool(finished.all().item())
        assert session.histories == legacy_histories
        if all_finished:
            break


def test_pick_single_row_finishes_immediately():
    session = make_session([[]], stop_ids={5}, eos_token_id=5)
    logits = torch.full((1, VOCAB), -1.0)
    logits[0, 5] = 10.0
    token, all_finished = session.pick(logits)
    assert token.tolist() == [5]
    assert all_finished


# ---------------------------------------------------------------------------
# Part 2 — rolling audio-prefix decoder KV + speculative draft verification.
# ---------------------------------------------------------------------------

from qwen3_streaming.cached_full_hypothesis import (  # noqa: E402
    CachedFullHypothesisConfig,
    SegmentedCachedFullHypothesisStreamer,
    trim_at_stop,
)
from qwen3_streaming.native_realtime_model import (  # noqa: E402
    Qwen3ASRRealtimeQwenAudioCausalModel,
    _split_prompt_template,
)
from test_decoder_kv_cache import (  # noqa: E402
    VOCAB as KV_VOCAB,
    FakeCachingQwenTextModel,
    _Out,
)
from test_mutable_tail import D_MODEL, N_MELS, TinyQwenAudioTower, tiny_config  # noqa: E402

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


class IntTokenizer:
    """Tokenizer over small int vocab; encode returns the test template."""

    def __init__(self):
        self.encode_calls = 0

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"t{int(t)}" for t in token_ids)

    def encode(self, text, add_special_tokens=False):
        self.encode_calls += 1
        return list(TEMPLATE)


def make_streamer(model, tokenizer, *, rolling: bool, **streamer_kwargs):
    config = CachedFullHypothesisConfig(
        wait_token_id=0,
        word_start_token_id=98,
        eos_token_id=EOS,
        max_new_tokens=8,
        hold_back_words=0,
        stable_iterations=1,
        suppress_token_ids=(2, 3),
        repetition_penalty=1.3,
        no_repeat_ngram_size=2,
        prompt_prefix_template=tuple(TEMPLATE),
        audio_placeholder_token_id=PH,
        decoder_rolling_kv=rolling,
        speculative_draft=rolling,
    )
    return SegmentedCachedFullHypothesisStreamer(
        model,
        tokenizer,
        config,
        segment_max_cached_steps=5,
        segment_finalize_mode="latest",
        reset_encoder_on_rollover=True,
        **streamer_kwargs,
    )


def test_streamer_rolling_parity_end_to_end_with_rollovers():
    torch.manual_seed(20)
    mel_chunks = [torch.randn(1, 24, N_MELS) for _ in range(6)]

    old = make_streamer(make_rolling_model(), IntTokenizer(), rolling=False)
    new = make_streamer(make_rolling_model(), IntTokenizer(), rolling=True)

    saw_rollover = False
    saw_draft = False
    for chunk in mel_chunks:
        event_old = old.append_mel_chunk(chunk.clone())
        event_new = new.append_mel_chunk(chunk.clone())
        assert event_new["hypothesis"] == event_old["hypothesis"]
        assert event_new["committed"] == event_old["committed"]
        assert "generate_ms" in event_new
        if event_new.get("segment_rollover"):
            saw_rollover = True
            assert getattr(new.state, "decoder", None) is None
        elif event_new.get("decoder_path", "").startswith("rolling"):
            assert new.state.decoder is not None
        if event_new.get("draft_tokens"):
            saw_draft = True
    assert saw_rollover, "test must cross at least one segment rollover"
    assert saw_draft, "speculative draft never engaged"
    assert (
        new.finalize(finalize_mode="latest").final_text
        == old.finalize(finalize_mode="latest").final_text
    )


def test_roll_before_generate_skips_over_limit_generation():
    streamer = make_streamer(
        make_rolling_model(),
        IntTokenizer(),
        rolling=True,
        segment_roll_before_generate=True,
    )
    torch.manual_seed(22)
    # 24 mel frames = 3 steps per chunk on the tiny tower; cap is 5 steps, so
    # chunk 2 would land at 6 steps: the roll must happen BEFORE its decode.
    events = [streamer.append_mel_chunk(torch.randn(1, 24, N_MELS)) for _ in range(4)]

    pre_rolls = [e for e in events if e.get("segment_rolled_before_generate")]
    assert pre_rolls, "pre-generate rollover never triggered"
    # No generation ever ran on an over-cap cache, and the boundary chunk's
    # audio opened the next segment.
    assert all(e["cached_steps"] <= 5 for e in events)
    assert all(e["segment_rollover"] is False for e in events)
    assert streamer.segments_finalized >= 1
    assert streamer.finalize(finalize_mode="latest").final_text


def test_segment_prompt_template_encoded_once_per_segment():
    tokenizer = IntTokenizer()
    streamer = make_streamer(
        make_rolling_model(),
        tokenizer,
        rolling=True,
        segment_prompt_context_words=4,
        segment_prompt_language="English",
    )
    torch.manual_seed(21)
    rollovers = 0
    for _ in range(4):
        event = streamer.append_mel_chunk(torch.randn(1, 16, N_MELS))
        rollovers += int(bool(event.get("segment_rollover")))
    # One encode for the first segment's template plus one per rollover.
    assert tokenizer.encode_calls == 1 + rollovers
