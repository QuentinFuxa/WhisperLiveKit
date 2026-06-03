from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from .audio_io import audio_duration_seconds, prefix_audio, write_pcm16_wav


LANGUAGE_PREFIX = {
    "en": "language English ",
    "fr": "language French ",
    "none": "language None ",
}


@dataclass(frozen=True)
class PrefixExample:
    audio: str
    text: str
    raw_text: str
    source: str
    language: str
    duration_sec: float
    prefix_end_sec: float
    right_context_sec: float
    is_prefix: bool

    def sft_record(self) -> dict[str, str]:
        return {"audio": self.audio, "text": self.text}

    def manifest_record(self) -> dict[str, object]:
        return {
            "audio": self.audio,
            "text": self.text,
            "raw_text": self.raw_text,
            "source": self.source,
            "language": self.language,
            "duration_sec": round(self.duration_sec, 4),
            "prefix_end_sec": round(self.prefix_end_sec, 4),
            "right_context_sec": round(self.right_context_sec, 4),
            "is_prefix": self.is_prefix,
        }


def stable_prefix_text(
    text: str,
    *,
    prefix_end_sec: float,
    duration_sec: float,
    right_context_sec: float,
) -> str:
    """Heuristic streaming label when word timestamps are not available.

    The target contains the transcript fraction that should be stable before
    ``prefix_end_sec - right_context_sec``. A later forced-aligner pass can
    replace this with exact word times without changing the JSONL interface.
    """
    words = text.strip().split()
    if not words or duration_sec <= 0.0:
        return ""

    stable_until = max(0.0, prefix_end_sec - right_context_sec)
    ratio = min(1.0, stable_until / duration_sec)
    if ratio <= 0.0:
        return ""

    n_words = int(math.floor(len(words) * ratio))
    if n_words == 0 and ratio > 0.0:
        n_words = 1
    return " ".join(words[:n_words])


def prefix_points(
    duration_sec: float,
    *,
    min_prefix_sec: float,
    stride_sec: float,
    include_full: bool = True,
) -> list[float]:
    if duration_sec <= 0.0:
        return []
    if stride_sec <= 0.0:
        raise ValueError("stride_sec must be > 0")

    points: list[float] = []
    current = min_prefix_sec
    while current < duration_sec:
        points.append(round(current, 3))
        current += stride_sec
    if include_full and (not points or points[-1] < duration_sec):
        points.append(round(duration_sec, 3))
    return points


def prefixed_text(text: str, language: str) -> str:
    prefix = LANGUAGE_PREFIX.get(language.lower(), LANGUAGE_PREFIX["none"])
    return f"{prefix}{text.strip()}".strip()


def stable_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def make_examples(
    *,
    audio,
    sample_rate: int,
    text: str,
    source: str,
    language: str,
    output_dir: Path,
    item_id: str,
    prefix_mode: bool,
    min_prefix_sec: float,
    stride_sec: float,
    right_context_sec: float,
    include_empty_prefix: bool = False,
) -> list[PrefixExample]:
    duration = audio_duration_seconds(audio, sample_rate)
    if duration <= 0.0 or not text.strip():
        return []

    if not prefix_mode:
        audio_path = output_dir / source / f"{item_id}.wav"
        write_pcm16_wav(audio_path, audio, sample_rate)
        return [
            PrefixExample(
                audio=str(audio_path.resolve()),
                text=prefixed_text(text, language),
                raw_text=text.strip(),
                source=source,
                language=language,
                duration_sec=duration,
                prefix_end_sec=duration,
                right_context_sec=0.0,
                is_prefix=False,
            )
        ]

    examples: list[PrefixExample] = []
    for idx, end_sec in enumerate(
        prefix_points(duration, min_prefix_sec=min_prefix_sec, stride_sec=stride_sec)
    ):
        target = stable_prefix_text(
            text,
            prefix_end_sec=end_sec,
            duration_sec=duration,
            right_context_sec=right_context_sec,
        )
        is_full = end_sec >= duration
        if is_full:
            target = text.strip()
        if not target and not include_empty_prefix:
            continue

        prefix = prefix_audio(audio, sample_rate, end_sec)
        audio_path = output_dir / source / f"{item_id}__p{idx:04d}.wav"
        write_pcm16_wav(audio_path, prefix, sample_rate)
        examples.append(
            PrefixExample(
                audio=str(audio_path.resolve()),
                text=prefixed_text(target, language),
                raw_text=target,
                source=source,
                language=language,
                duration_sec=duration,
                prefix_end_sec=end_sec,
                right_context_sec=right_context_sec,
                is_prefix=not is_full,
            )
        )
    return examples
