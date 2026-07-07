"""History-retention policy (issue #372).

mode=full clients receive the whole transcript on every update, so pruning
server-side history permanently deletes their earliest lines. The policy:
explicit --retention-seconds wins (0 = unlimited), otherwise full mode keeps
everything and diff mode keeps the bounded default.
"""

import math
from types import SimpleNamespace

from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.tokens_alignment import (
    _DEFAULT_RETENTION_SECONDS,
    TokensAlignment,
    resolve_retention_seconds,
)


def test_resolve_retention_policy():
    assert math.isinf(resolve_retention_seconds(None, "full"))
    assert resolve_retention_seconds(None, "diff") == _DEFAULT_RETENTION_SECONDS
    assert resolve_retention_seconds(60, "full") == 60.0
    assert resolve_retention_seconds(60, "diff") == 60.0
    assert math.isinf(resolve_retention_seconds(0, "diff"))
    assert math.isinf(resolve_retention_seconds(-1, "full"))


def _alignment(retention_seconds):
    state = SimpleNamespace(
        new_tokens=[],
        new_diarization=[],
        new_translation=[],
        new_tokens_buffer=[],
        new_translation_buffer=None,
    )
    args = SimpleNamespace(diarization=False)
    return TokensAlignment(state, args, " ", retention_seconds=retention_seconds)


def _tokens(until_sec, step=10.0):
    return [
        ASRToken(start=beg, end=beg + step, text=f"w{int(beg)}")
        for beg in [i * step for i in range(int(until_sec / step))]
    ]


def test_prune_drops_old_tokens_with_bounded_retention():
    alignment = _alignment(retention_seconds=300.0)
    alignment.all_tokens = _tokens(600.0)
    alignment._prune()
    assert alignment.all_tokens
    # Everything older than latest_end - 300 s is gone.
    assert alignment.all_tokens[0].end >= 600.0 - 300.0


def test_unlimited_retention_keeps_full_history():
    alignment = _alignment(retention_seconds=math.inf)
    alignment.all_tokens = _tokens(3600.0)  # one hour
    before = list(alignment.all_tokens)
    alignment._prune()
    assert alignment.all_tokens == before
