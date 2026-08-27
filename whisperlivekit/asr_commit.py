"""Job 1: stable/unstable split for ASR backends that revise their hypothesis.

Models that re-decode a window (rolling-decode attention encoder-decoders)
can change their recent output on every decode pass. This module commits
only the prefix that stays stable across consecutive hypotheses, using
token agreement across N decode passes and a hold-back token count.

Factored from ``qwen3_asr_causal.stable_commit`` so WLK backends can use it
without depending on the ``qwen3-asr-causal`` package (which pins
transformers==4.57.6). Two variants are provided:

- **Token-ID** (``update_stable_prefix_commit``): for backends that emit
  raw token IDs. The commit boundary is the longest common prefix of the
  previous and current hypothesis token sequences, minus a hold-back count,
  confirmed across ``stable_iterations`` passes.

- **Text-unit** (``update_stable_text_commit``): for backends that emit
  text (e.g. mlx-qwen3-asr's ``stable_text`` / ``text``). The commit
  boundary operates on whitespace-split text units with the same
  agreement + hold-back logic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

# ---------------------------------------------------------------------------
# Token-ID variant
# ---------------------------------------------------------------------------


def longest_common_prefix_length(
    left: Sequence[int],
    right: Sequence[int],
) -> int:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if int(left[idx]) != int(right[idx]):
            return idx
    return limit


@dataclass
class StablePrefixCommitState:
    committed_tokens: list[int] = field(default_factory=list)
    previous_hypothesis: list[int] = field(default_factory=list)
    stable_candidate: list[int] = field(default_factory=list)
    stable_candidate_count: int = 0


@dataclass(frozen=True)
class StablePrefixCommitUpdate:
    delta_tokens: list[int]
    committed_tokens: list[int]
    candidate_tokens: list[int]
    committed_token_count: int


def update_stable_prefix_commit(
    state: StablePrefixCommitState,
    hypothesis_tokens: Sequence[int],
    *,
    hold_back_tokens: int = 0,
    stable_iterations: int = 2,
    allow_commit: bool = True,
    final: bool = False,
) -> StablePrefixCommitUpdate:
    """Commit only the prefix that stayed stable across consecutive hypotheses."""
    if hold_back_tokens < 0:
        raise ValueError("hold_back_tokens must be >= 0")
    if stable_iterations <= 0:
        raise ValueError("stable_iterations must be > 0")

    tokens = [int(token_id) for token_id in hypothesis_tokens]
    committed = state.committed_tokens
    previous_committed_len = len(committed)
    committed_still_matches = tokens[:previous_committed_len] == committed

    if final and committed_still_matches:
        candidate = tokens
        commit_len = len(tokens)
    elif not committed_still_matches:
        candidate = committed
        commit_len = previous_committed_len
        state.stable_candidate = candidate.copy()
        state.stable_candidate_count = 0
    else:
        lcp_len = longest_common_prefix_length(state.previous_hypothesis, tokens)
        candidate_len = max(previous_committed_len, lcp_len - hold_back_tokens)
        candidate = tokens[:candidate_len]
        if candidate == state.stable_candidate:
            state.stable_candidate_count += 1
        else:
            state.stable_candidate = candidate.copy()
            state.stable_candidate_count = 1
        if state.stable_candidate_count >= stable_iterations:
            commit_len = candidate_len
        else:
            commit_len = previous_committed_len

    if not allow_commit and not final:
        commit_len = previous_committed_len

    if commit_len > previous_committed_len:
        state.committed_tokens = tokens[:commit_len]
        delta = state.committed_tokens[previous_committed_len:commit_len]
    else:
        delta = []

    state.previous_hypothesis = tokens
    return StablePrefixCommitUpdate(
        delta_tokens=delta,
        committed_tokens=state.committed_tokens.copy(),
        candidate_tokens=candidate.copy(),
        committed_token_count=len(state.committed_tokens),
    )


# ---------------------------------------------------------------------------
# Text-unit variant
# ---------------------------------------------------------------------------

_TEXT_UNIT_RE = re.compile(r"\S+\s*")
_TEXT_UNIT_EDGE_PUNCT_RE = re.compile(r"^[^\w']+|[^\w']+$")


def split_text_units(text: str) -> list[str]:
    return [match.group(0) for match in _TEXT_UNIT_RE.finditer(text or "")]


def join_text_units(units: Sequence[str]) -> str:
    return "".join(units).strip()


def normalize_text_unit_for_match(
    unit: str,
    *,
    case_sensitive: bool = True,
) -> str:
    normalized = _TEXT_UNIT_EDGE_PUNCT_RE.sub("", unit.strip())
    return normalized if case_sensitive else normalized.lower()


def longest_common_text_prefix_length(
    left: Sequence[str],
    right: Sequence[str],
    *,
    case_sensitive: bool = True,
    normalize_for_match: bool = False,
) -> int:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if normalize_for_match:
            left_unit = normalize_text_unit_for_match(
                left[idx],
                case_sensitive=case_sensitive,
            )
            right_unit = normalize_text_unit_for_match(
                right[idx],
                case_sensitive=case_sensitive,
            )
        else:
            left_unit = left[idx] if case_sensitive else left[idx].lower()
            right_unit = right[idx] if case_sensitive else right[idx].lower()
        if left_unit != right_unit:
            return idx
    return limit


def text_prefix_matches(
    prefix: Sequence[str],
    units: Sequence[str],
    *,
    case_sensitive: bool = True,
    normalize_for_match: bool = False,
) -> bool:
    if len(prefix) > len(units):
        return False
    return (
        longest_common_text_prefix_length(
            prefix,
            units,
            case_sensitive=case_sensitive,
            normalize_for_match=normalize_for_match,
        )
        == len(prefix)
    )


@dataclass
class StableTextCommitState:
    committed_units: list[str] = field(default_factory=list)
    previous_hypothesis_units: list[str] = field(default_factory=list)
    stable_candidate_units: list[str] = field(default_factory=list)
    stable_candidate_count: int = 0


@dataclass(frozen=True)
class StableTextCommitUpdate:
    delta_text: str
    committed_text: str
    display_text: str
    unstable_text: str
    candidate_text: str
    committed_unit_count: int
    hypothesis_unit_count: int


def update_stable_text_commit(
    state: StableTextCommitState,
    hypothesis_text: str,
    *,
    hold_back_units: int = 6,
    stable_iterations: int = 2,
    case_sensitive: bool = True,
    normalize_for_match: bool = False,
    allow_commit: bool = True,
    final: bool = False,
    final_revises_committed: bool = True,
) -> StableTextCommitUpdate:
    """Commit stable word-like text units while keeping a revisable display tail."""
    if hold_back_units < 0:
        raise ValueError("hold_back_units must be >= 0")
    if stable_iterations <= 0:
        raise ValueError("stable_iterations must be > 0")

    units = split_text_units(hypothesis_text)
    previous_committed = state.committed_units
    previous_committed_len = len(previous_committed)
    committed_still_matches = text_prefix_matches(
        previous_committed,
        units,
        case_sensitive=case_sensitive,
        normalize_for_match=normalize_for_match,
    )

    if final and (final_revises_committed or committed_still_matches):
        candidate = units
        commit_len = len(units)
    elif not committed_still_matches:
        candidate = previous_committed
        commit_len = previous_committed_len
        state.stable_candidate_units = candidate.copy()
        state.stable_candidate_count = 0
    else:
        lcp_len = longest_common_text_prefix_length(
            state.previous_hypothesis_units,
            units,
            case_sensitive=case_sensitive,
            normalize_for_match=normalize_for_match,
        )
        candidate_len = max(previous_committed_len, lcp_len - hold_back_units)
        candidate = units[:candidate_len]
        if candidate == state.stable_candidate_units:
            state.stable_candidate_count += 1
        else:
            state.stable_candidate_units = candidate.copy()
            state.stable_candidate_count = 1
        if state.stable_candidate_count >= stable_iterations:
            commit_len = candidate_len
        else:
            commit_len = previous_committed_len

    if not allow_commit and not final:
        commit_len = previous_committed_len

    if commit_len > previous_committed_len or (
        final and final_revises_committed and units != previous_committed
    ):
        if final and final_revises_committed:
            delta_units = units[previous_committed_len:] if committed_still_matches else units
            state.committed_units = units.copy()
        else:
            state.committed_units = units[:commit_len]
            delta_units = state.committed_units[previous_committed_len:commit_len]
    else:
        delta_units = []

    committed_units = state.committed_units
    if units[: len(committed_units)] == committed_units:
        unstable_units = units[len(committed_units):]
        display_units = committed_units + unstable_units
    else:
        unstable_units = units
        display_units = units
    state.previous_hypothesis_units = units

    return StableTextCommitUpdate(
        delta_text=join_text_units(delta_units),
        committed_text=join_text_units(committed_units),
        display_text=join_text_units(display_units),
        unstable_text=join_text_units(unstable_units),
        candidate_text=join_text_units(candidate),
        committed_unit_count=len(committed_units),
        hypothesis_unit_count=len(units),
    )


# ---------------------------------------------------------------------------
# Transform for the AsrWrapper chain
# ---------------------------------------------------------------------------


class StableCommitTransform:
    """Stateful transform that applies Job 1 (stable/unstable split).

    Designed for use in ``AsrWrapper``'s transform chain. On each
    ``process_iter`` call it reads the rolling hypothesis from
    ``inner.get_buffer()`` and commits only the prefix that stays stable
    across decode passes.

    At utterance boundaries (``start_silence`` / ``finish``) the wrapper
    calls ``reset()`` so the state aligns with the next utterance.
    """

    def __init__(
        self,
        hold_back_units: int = 6,
        stable_iterations: int = 2,
    ):
        self._hold_back = hold_back_units
        self._stable_iterations = stable_iterations
        self._state = StableTextCommitState()

    def __call__(self, result, inner):
        from whisperlivekit.timed_objects import ASRToken

        _, end_time = result
        buffer = inner.get_buffer()
        hypothesis = getattr(buffer, "text", "") or ""

        update = update_stable_text_commit(
            self._state,
            hypothesis,
            hold_back_units=self._hold_back,
            stable_iterations=self._stable_iterations,
        )

        if update.delta_text:
            tok = ASRToken(
                start=end_time,
                end=end_time,
                text=update.delta_text,
                speaker=-1,
            )
            return [tok], end_time
        return [], end_time

    def reset(self):
        self._state = StableTextCommitState()
