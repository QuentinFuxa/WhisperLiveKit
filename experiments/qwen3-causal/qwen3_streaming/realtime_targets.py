from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...


@dataclass(frozen=True)
class WordAlignment:
    text: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class ScheduledEmission:
    frame_index: int
    token_id: int
    text: str
    word_index: int
    is_word_start: bool = False


@dataclass(frozen=True)
class FrameTargets:
    labels: list[int]
    previous_input_ids: list[int]
    frame_times_sec: list[float]
    emissions: list[ScheduledEmission]
    frame_sec: float
    delay_sec: float


def heuristic_word_alignments(text: str, duration_sec: float) -> list[WordAlignment]:
    """Approximate word timings when a dataset has only utterance-level text.

    This is not good enough for final training, but it is useful for a first
    end-to-end causal ASR smoke model before WhisperX/MFA alignments are ready.
    Durations are distributed by word character length so long words get a
    little more time than short words.
    """

    words = [word.strip() for word in text.split() if word.strip()]
    if not words or duration_sec <= 0.0:
        return []
    weights = [max(1, len(word)) for word in words]
    total = float(sum(weights))
    cursor = 0.0
    alignments: list[WordAlignment] = []
    for word, weight in zip(words, weights, strict=True):
        end = min(duration_sec, cursor + duration_sec * (float(weight) / total))
        alignments.append(WordAlignment(text=word, start_sec=cursor, end_sec=end))
        cursor = end
    if alignments:
        last = alignments[-1]
        alignments[-1] = WordAlignment(
            text=last.text,
            start_sec=last.start_sec,
            end_sec=duration_sec,
        )
    return alignments


def frame_index_for_time(time_sec: float, frame_sec: float) -> int:
    if frame_sec <= 0.0:
        raise ValueError("frame_sec must be > 0")
    return max(0, int(math.ceil((time_sec / frame_sec) - 1e-9)))


def _encode_word(tokenizer: TokenizerLike, word: str, *, prepend_space: bool = False) -> list[int]:
    text = f" {word}" if prepend_space else word
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def build_frame_targets(
    *,
    words: list[WordAlignment],
    tokenizer: TokenizerLike,
    duration_sec: float,
    wait_token_id: int,
    word_start_token_id: int,
    bos_token_id: int | None,
    frame_sec: float = 0.080,
    delay_sec: float = 0.800,
    include_word_start: bool = True,
    include_trailing_delay: bool = True,
) -> FrameTargets:
    """Build one decoder target per realtime frame.

    A target frame contains either the wait token ``[P]``, the word-start token
    ``[W]``, or exactly one text token. Word tokens are scheduled at
    ``word.start_sec + delay_sec`` and shifted right on collisions so the target
    stream remains one token per 80 ms decoder step.
    """

    if duration_sec <= 0.0:
        raise ValueError("duration_sec must be > 0")
    if delay_sec < 0.0:
        raise ValueError("delay_sec must be >= 0")
    if frame_sec <= 0.0:
        raise ValueError("frame_sec must be > 0")

    target_duration = duration_sec + (delay_sec if include_trailing_delay else 0.0)
    n_frames = max(1, frame_index_for_time(target_duration, frame_sec))
    labels = [wait_token_id for _ in range(n_frames)]
    emissions: list[ScheduledEmission] = []

    for word_index, word in enumerate(words):
        text = word.text.strip()
        if not text:
            continue
        token_ids = _encode_word(tokenizer, text, prepend_space=word_index > 0)
        if not token_ids:
            continue

        cursor = frame_index_for_time(max(0.0, word.start_sec) + delay_sec, frame_sec)
        pieces: list[tuple[int, str, bool]] = []
        if include_word_start:
            pieces.append((word_start_token_id, "[W]", True))
        pieces.extend((token_id, text, False) for token_id in token_ids)

        for token_id, piece_text, is_word_start in pieces:
            while cursor >= len(labels):
                labels.append(wait_token_id)
            while cursor < len(labels) and labels[cursor] != wait_token_id:
                cursor += 1
                if cursor >= len(labels):
                    labels.append(wait_token_id)
            labels[cursor] = token_id
            emissions.append(
                ScheduledEmission(
                    frame_index=cursor,
                    token_id=token_id,
                    text=piece_text,
                    word_index=word_index,
                    is_word_start=is_word_start,
                )
            )
            cursor += 1

    first_input = wait_token_id if bos_token_id is None else bos_token_id
    previous_input_ids = [first_input] + labels[:-1]
    frame_times = [idx * frame_sec for idx in range(len(labels))]
    return FrameTargets(
        labels=labels,
        previous_input_ids=previous_input_ids,
        frame_times_sec=frame_times,
        emissions=emissions,
        frame_sec=frame_sec,
        delay_sec=delay_sec,
    )
