from __future__ import annotations

import re
from dataclasses import dataclass

from jiwer import wer

_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _SPACE_RE.sub(" ", text.strip().lower())


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    if not reference:
        return None
    return float(wer(reference, hypothesis))


def _as_int_tokens(token_ids) -> list[int]:
    if hasattr(token_ids, "detach"):
        token_ids = token_ids.detach().cpu().reshape(-1).tolist()
    return [int(token_id) for token_id in token_ids]


def token_repetition_stats(
    token_ids,
    *,
    ignored_token_ids: set[int] | tuple[int, ...] | list[int] = (),
    max_ngram: int = 3,
) -> dict[str, int | float]:
    ignored = {int(token_id) for token_id in ignored_token_ids}
    tokens = [
        token_id for token_id in _as_int_tokens(token_ids) if token_id not in ignored
    ]
    stats: dict[str, int | float] = {
        "text_token_count": len(tokens),
        "unique_text_token_count": len(set(tokens)),
    }
    for n in range(1, max_ngram + 1):
        prefix = {1: "unigram", 2: "bigram", 3: "trigram"}.get(n, f"{n}gram")
        total = max(0, len(tokens) - n + 1)
        if total == 0:
            unique = 0
        elif n == 1:
            unique = len(set(tokens))
        else:
            unique = len(
                {
                    tuple(tokens[start : start + n])
                    for start in range(0, len(tokens) - n + 1)
                }
            )
        repeated = total - unique
        stats[f"{prefix}_total"] = total
        stats[f"{prefix}_repeated"] = repeated
        stats[f"{prefix}_repetition_ratio"] = (
            float(repeated / total) if total else 0.0
        )
    return stats


def merge_token_repetition_stats(
    items: list[dict[str, int | float]],
    *,
    max_ngram: int = 3,
) -> dict[str, int | float]:
    merged: dict[str, int | float] = {
        "text_token_count": sum(int(item.get("text_token_count", 0)) for item in items),
        "unique_text_token_count_sum": sum(
            int(item.get("unique_text_token_count", 0)) for item in items
        ),
    }
    for n in range(1, max_ngram + 1):
        prefix = {1: "unigram", 2: "bigram", 3: "trigram"}.get(n, f"{n}gram")
        total = sum(int(item.get(f"{prefix}_total", 0)) for item in items)
        repeated = sum(int(item.get(f"{prefix}_repeated", 0)) for item in items)
        merged[f"{prefix}_total"] = total
        merged[f"{prefix}_repeated"] = repeated
        merged[f"{prefix}_repetition_ratio"] = (
            float(repeated / total) if total else 0.0
        )
    return merged


@dataclass(frozen=True)
class StablePrefixStats:
    reference_words: int
    hypothesis_words: int
    common_prefix_words: int
    revision_words: int

    @property
    def common_prefix_ratio(self) -> float:
        if self.reference_words == 0:
            return 1.0 if self.hypothesis_words == 0 else 0.0
        return self.common_prefix_words / self.reference_words


def stable_prefix_stats(reference: str, hypothesis: str) -> StablePrefixStats:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    common = 0
    for ref_word, hyp_word in zip(ref_words, hyp_words):
        if ref_word != hyp_word:
            break
        common += 1
    revision = max(0, len(hyp_words) - common)
    return StablePrefixStats(
        reference_words=len(ref_words),
        hypothesis_words=len(hyp_words),
        common_prefix_words=common,
        revision_words=revision,
    )


def normalized_words(text: str) -> list[str]:
    return normalize_text(text).split()


def text_revision_stats(texts: list[str]) -> dict[str, int | float]:
    revision_events = 0
    revision_words = 0
    max_revision_words = 0
    previous: list[str] = []
    nonempty_updates = 0

    for text in texts:
        words = normalized_words(text)
        if words:
            nonempty_updates += 1
        common = 0
        for previous_word, current_word in zip(previous, words):
            if previous_word != current_word:
                break
            common += 1
        revised = max(0, len(previous) - common)
        if revised:
            revision_events += 1
            revision_words += revised
            max_revision_words = max(max_revision_words, revised)
        previous = words

    update_count = max(0, len(texts) - 1)
    return {
        "updates": len(texts),
        "nonempty_updates": nonempty_updates,
        "revision_events": revision_events,
        "revision_words": revision_words,
        "max_revision_words": max_revision_words,
        "revision_event_ratio": (
            float(revision_events / update_count) if update_count else 0.0
        ),
    }


def streaming_text_event_stats(
    events: list[dict],
    *,
    final_text: str,
    stable_text: str,
) -> dict[str, int | float | None]:
    first_display_sec = None
    first_commit_sec = None
    for event in events:
        event_sec = float(event.get("audio_sec", 0.0))
        if first_display_sec is None and str(event.get("display", "")).strip():
            first_display_sec = event_sec
        if first_commit_sec is None and str(event.get("committed", "")).strip():
            first_commit_sec = event_sec

    final_words = normalized_words(final_text)
    stable_words = normalized_words(stable_text)
    display_revision = text_revision_stats(
        [str(event.get("display", "")) for event in events]
    )
    hypothesis_revision = text_revision_stats(
        [str(event.get("hypothesis", "")) for event in events]
    )
    committed_revision = text_revision_stats(
        [str(event.get("committed", "")) for event in events]
    )

    return {
        "first_display_sec": first_display_sec,
        "first_commit_sec": first_commit_sec,
        "final_word_count": len(final_words),
        "stable_word_count": len(stable_words),
        "stable_coverage_ratio": (
            float(len(stable_words) / len(final_words)) if final_words else 0.0
        ),
        "display_revision_events": int(display_revision["revision_events"]),
        "display_revision_words": int(display_revision["revision_words"]),
        "display_max_revision_words": int(display_revision["max_revision_words"]),
        "display_revision_event_ratio": float(
            display_revision["revision_event_ratio"]
        ),
        "hypothesis_revision_events": int(hypothesis_revision["revision_events"]),
        "hypothesis_revision_words": int(hypothesis_revision["revision_words"]),
        "committed_revision_events": int(committed_revision["revision_events"]),
        "committed_revision_words": int(committed_revision["revision_words"]),
    }
