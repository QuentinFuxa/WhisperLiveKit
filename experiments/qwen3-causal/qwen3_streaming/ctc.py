from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .realtime_targets import WordAlignment


@dataclass(frozen=True)
class CTCDecodeResult:
    token_ids: list[int]
    raw_token_ids: list[int]
    blank_count: int
    raw_text_token_count: int


@dataclass(frozen=True)
class CompactCTCVocab:
    token_ids: list[int]
    blank_index: int = 0

    @property
    def token_to_index(self) -> dict[int, int]:
        return {int(token_id): idx for idx, token_id in enumerate(self.token_ids)}

    def encode(self, token_ids) -> list[int]:
        mapping = self.token_to_index
        missing = [int(token_id) for token_id in token_ids if int(token_id) not in mapping]
        if missing:
            raise KeyError(f"compact CTC vocab missing token ids: {missing[:10]}")
        return [mapping[int(token_id)] for token_id in token_ids]

    def decode(self, compact_indices) -> list[int]:
        output: list[int] = []
        for index in compact_indices:
            idx = int(index)
            if not 0 <= idx < len(self.token_ids):
                raise IndexError(f"compact CTC index out of range: {idx}")
            output.append(int(self.token_ids[idx]))
        return output


def _encode_word(tokenizer, word: str, *, prepend_space: bool) -> list[int]:
    text = f" {word}" if prepend_space else word
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def build_ctc_token_targets(
    *,
    words: list[WordAlignment],
    tokenizer,
    blank_token_id: int,
    ignored_token_ids: set[int] | tuple[int, ...] | list[int] = (),
) -> list[int]:
    ignored = {int(token_id) for token_id in ignored_token_ids}
    ignored.add(int(blank_token_id))
    token_ids: list[int] = []
    emitted_word_index = 0
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        encoded = _encode_word(
            tokenizer,
            text,
            prepend_space=emitted_word_index > 0,
        )
        token_ids.extend(int(token_id) for token_id in encoded if int(token_id) not in ignored)
        emitted_word_index += 1
    return token_ids


def build_compact_ctc_vocab(
    target_sequences,
    *,
    blank_token_id: int,
    max_tokens: int = 0,
) -> CompactCTCVocab:
    counter: Counter[int] = Counter()
    blank = int(blank_token_id)
    for sequence in target_sequences:
        for token_id in sequence:
            token = int(token_id)
            if token != blank:
                counter[token] += 1
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if max_tokens > 0:
        items = items[: max(0, int(max_tokens) - 1)]
    return CompactCTCVocab(
        token_ids=[blank] + [int(token_id) for token_id, _ in items],
        blank_index=0,
    )


def ctc_greedy_decode(
    token_ids,
    *,
    blank_token_id: int,
    ignored_token_ids: set[int] | tuple[int, ...] | list[int] = (),
    previous_token_id: int | None = None,
) -> CTCDecodeResult:
    ignored = {int(token_id) for token_id in ignored_token_ids}
    blank = int(blank_token_id)
    raw = [int(token_id) for token_id in token_ids]
    output: list[int] = []
    last = previous_token_id
    blank_count = 0
    raw_text_token_count = 0
    for token_id in raw:
        if token_id == blank:
            blank_count += 1
        elif token_id not in ignored:
            raw_text_token_count += 1
        if token_id == last:
            continue
        last = token_id
        if token_id == blank or token_id in ignored:
            continue
        output.append(token_id)
    return CTCDecodeResult(
        token_ids=output,
        raw_token_ids=raw,
        blank_count=blank_count,
        raw_text_token_count=raw_text_token_count,
    )


def ctc_decode_text(
    tokenizer,
    token_ids,
    *,
    blank_token_id: int,
    ignored_token_ids: set[int] | tuple[int, ...] | list[int] = (),
    previous_token_id: int | None = None,
) -> tuple[str, CTCDecodeResult]:
    from .realtime_features import clean_decoded_text

    result = ctc_greedy_decode(
        token_ids,
        blank_token_id=blank_token_id,
        ignored_token_ids=ignored_token_ids,
        previous_token_id=previous_token_id,
    )
    if not result.token_ids:
        return "", result
    return (
        clean_decoded_text(tokenizer.decode(result.token_ids, skip_special_tokens=True)),
        result,
    )


def ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    blank_token_id: int,
) -> torch.Tensor:
    import torch

    log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
    return torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lengths.to(dtype=torch.long).cpu(),
        target_lengths.to(dtype=torch.long).cpu(),
        blank=int(blank_token_id),
        zero_infinity=True,
    )
