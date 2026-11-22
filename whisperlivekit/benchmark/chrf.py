"""chrF (Popović 2015) scoring for the translation half of a benchmark run.

The definition follows sacrebleu: character n-gram order 6, word n-gram order
0, beta 2, and whitespace removed before n-grams are counted. The corpus
figure sums the per-order [hypothesis, reference, match] counts over every row
of a language pair and computes one score from those totals; averaging the
per-row scores would weight short rows too heavily. sacrebleu's default
effective-order smoothing is used (no epsilon smoothing), which ignores orders
that neither side has n-grams for.
"""

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CHAR_ORDER = 6
WORD_ORDER = 0
BETA = 2

_EPS = 1e-16


def _char_ngrams(text: str, order: int) -> List[Counter]:
    """Count character n-grams of every order up to ``order``.

    Whitespace is dropped first, as sacrebleu does with ``whitespace=False``.
    """
    text = "".join(text.split())
    return [
        Counter(text[i:i + n] for i in range(len(text) - n + 1))
        for n in range(1, order + 1)
    ]


def _match_statistics(hypothesis_ngrams: Counter, reference_ngrams: Counter) -> List[int]:
    """[hypothesis count, reference count, matching count] for one order."""
    matches = sum(
        min(count, reference_ngrams[ngram])
        for ngram, count in hypothesis_ngrams.items()
        if ngram in reference_ngrams
    )
    return [sum(hypothesis_ngrams.values()), sum(reference_ngrams.values()), matches]


def chrf_statistics(hypothesis: str, reference: str,
                    char_order: int = CHAR_ORDER) -> List[int]:
    """Per-order match statistics for one row, in the order chrF consumes them."""
    return [
        token
        for hyp, ref in zip(_char_ngrams(hypothesis, char_order),
                            _char_ngrams(reference, char_order))
        for token in _match_statistics(hyp, ref)
    ]


def chrf_from_statistics(statistics: Sequence[int],
                         char_order: int = CHAR_ORDER, beta: int = BETA) -> float:
    """chrF in percent from statistics, matching sacrebleu's effective order."""
    factor = beta ** 2
    average_precision = 0.0
    average_recall = 0.0
    effective_order = 0

    for index in range(char_order):
        n_hyp, n_ref, n_match = statistics[3 * index:3 * index + 3]
        precision = n_match / n_hyp if n_hyp > 0 else _EPS
        recall = n_match / n_ref if n_ref > 0 else _EPS
        if n_hyp > 0 and n_ref > 0:
            average_precision += precision
            average_recall += recall
            effective_order += 1

    if effective_order == 0:
        return 0.0
    average_precision /= effective_order
    average_recall /= effective_order
    if average_precision + average_recall == 0:
        return 0.0
    score = (1 + factor) * average_precision * average_recall
    score /= (factor * average_precision) + average_recall
    return 100 * score


def compute_chrf(hypothesis: str, reference: str,
                 char_order: int = CHAR_ORDER) -> float:
    """chrF for a single row, in percent."""
    return chrf_from_statistics(chrf_statistics(hypothesis, reference, char_order), char_order)


def corpus_chrf(pairs: Iterable[Tuple[str, str]],
                char_order: int = CHAR_ORDER) -> Optional[float]:
    """chrF over all rows of one language pair, from their summed statistics.

    ``None`` only when no row carries a reference: a translated row scored
    against a reference is a measurement even when the hypothesis is empty.
    """
    totals = [0] * (3 * char_order)
    scored = 0
    for hypothesis, reference in pairs:
        if not reference:
            continue
        statistics = chrf_statistics(hypothesis, reference, char_order)
        totals = [total + value for total, value in zip(totals, statistics)]
        scored += 1
    if not scored:
        return None
    return chrf_from_statistics(totals, char_order)


def corpus_chrf_by_language(results: Sequence) -> Dict[str, Dict[str, object]]:
    """Corpus chrF per source language, with the row count that entered it.

    Rows with a ``translation_reference`` are grouped by the row's source
    ``language`` (the ASR side); the target language is one per run and is
    recorded next to this summary.
    """
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for result in results:
        reference = getattr(result, "translation_reference", "") or ""
        if not reference:
            continue
        language = getattr(result, "language", "") or ""
        groups.setdefault(language, []).append(
            (getattr(result, "translation_hypothesis", "") or "", reference)
        )
    return {
        language: {"corpus_chrf": corpus_chrf(pairs), "n_rows": len(pairs)}
        for language, pairs in groups.items()
    }
