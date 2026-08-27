"""Composable wrapper layer for ASR online processors.

Generalizes the existing ``_ASRTokenNormalizer`` from a single
token-normalize step to a composable transform chain. A backend declares
which jobs it needs (stable-commit, timestamp manufacture, etc.), and
``online_factory`` builds the chain.

The wrapper forwards the online-processor contract methods
(``insert_audio_chunk``, ``process_iter``, ``start_silence``,
``finish``, ``get_buffer``, etc.) to an inner processor. Methods that
return ``(tokens, end_time)`` — ``process_iter``, ``start_silence``,
``finish`` — are intercepted so transforms can modify the token list
before it reaches the AudioProcessor.

Transform list semantics:
  - **Iter transforms** (``transforms`` arg): applied to ``process_iter``
    results only. These are the job-specific transforms (stable-commit,
    timestamp manufacture). Stateful transforms are reset at utterance
    boundaries (``start_silence`` / ``finish``).
  - **Token normalize**: always applied last, on every intercepted
    method. This is the existing ``_ASRTokenNormalizer`` behaviour —
    convert foreign token objects to WLK ``ASRToken`` instances.

A transform is a callable with the signature::

    transform(result: (tokens, end_time), inner: object) -> (tokens, end_time)

Stateful transforms may expose a ``reset()`` method; the wrapper calls
it after ``start_silence`` and ``finish`` so the transform's state
aligns with the utterance lifecycle.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

from whisperlivekit.timed_objects import ASRToken, TimedText

TransformResult = Tuple[List, float]
Transform = Callable[[TransformResult, object], TransformResult]


def _to_wlk_token(tok) -> ASRToken:
    """Convert an arbitrary token object into a WLK ASRToken.

    qwen3's ASRToken is a separate class that doesn't derive from
    TimedText, so it lacks helpers (has_punctuation) the diarization
    alignment needs.
    """
    if isinstance(tok, TimedText):
        return tok
    is_silence = getattr(tok, "is_silence", None)
    if callable(is_silence) and is_silence():
        return tok
    return ASRToken(
        start=tok.start,
        end=tok.end,
        text=tok.text or "",
        speaker=getattr(tok, "speaker", -1),
        detected_language=getattr(tok, "detected_language", None),
        probability=getattr(tok, "probability", None),
    )


def token_normalize_transform(result: TransformResult, inner: object) -> TransformResult:
    """Convert emitted tokens to WLK ASRTokens.

    This is the generalized form of ``_ASRTokenNormalizer._convert``.
    """
    tokens, *rest = result
    converted = [_to_wlk_token(t) for t in (tokens or [])]
    return (converted, *rest)


class AsrWrapper:
    """Wrap an online processor and apply a chain of transforms.

    ``transforms`` are applied to ``process_iter`` results only.
    Token normalization is always applied last on every intercepted
    method. Stateful transforms are reset after ``start_silence`` and
    ``finish``.
    """

    _WRAP = {"finish"}

    def __init__(self, inner, transforms: List[Transform] | None = None):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_transforms", transforms or [])

    def _apply_iter_transforms(self, result: TransformResult) -> TransformResult:
        for t in self._transforms:
            result = t(result, self._inner)
        return token_normalize_transform(result, self._inner)

    def _apply_boundary_transforms(self, result: TransformResult) -> TransformResult:
        result = token_normalize_transform(result, self._inner)
        for t in self._transforms:
            reset = getattr(t, "reset", None)
            if callable(reset):
                reset()
        return result

    def process_iter(self, *args, **kwargs):
        return self._apply_iter_transforms(self._inner.process_iter(*args, **kwargs))

    def start_silence(self, *args, **kwargs):
        return self._apply_boundary_transforms(self._inner.start_silence(*args, **kwargs))

    def new_speaker(self, *args, **kwargs):
        """Preserve Qwen boundary tokens discarded by its compatibility API.

        The current qwen3 processors implement ``new_speaker()`` as a bare
        call to ``start_silence()`` and drop its return value. Calling
        ``start_silence()`` directly keeps the identical reset behavior
        while exposing the tokens and processed position required by
        AudioProcessor.
        """
        return self.start_silence()

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name in self._WRAP and callable(attr):
            def wrapped(*args, **kwargs):
                return self._apply_boundary_transforms(attr(*args, **kwargs))
            return wrapped
        return attr


class _ASRTokenNormalizer(AsrWrapper):
    """Backward-compatible alias: token-normalize only, no iter transforms."""

    def __init__(self, inner):
        super().__init__(inner, transforms=[])
