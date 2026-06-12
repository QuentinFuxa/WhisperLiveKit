"""Decoder-side decode-control equivalence for the qwen3-streaming backend.

Part 1 — `_GreedyControlSession` must be value-equivalent to the legacy
control pipeline (`_apply_repetition_controls_to_logits` kept in source as the
reference spec, plus the deleted `_control_logits_and_pick` preamble/stop
handling reproduced here verbatim).

Part 2 — rolling audio-prefix decoder KV + lossless speculative draft verify:
parity with the full re-prefill path on history-dependent fakes.
"""

import pytest

torch = pytest.importorskip("torch")

from whisperlivekit.qwen3_streaming.model import (  # noqa: E402
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
