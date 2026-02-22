"""Lightweight ASR evaluation metrics — no external dependencies.

Provides WER (Word Error Rate) computation via word-level Levenshtein distance,
text normalization, and word-level timestamp accuracy metrics with greedy alignment.
"""

import re
import unicodedata
from typing import Dict, List, Optional


def normalize_text(text: str) -> str:
    """Normalize text for WER comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    # Normalize unicode (e.g., accented chars to composed form)
    text = unicodedata.normalize("NFC", text)
    # Remove punctuation (keep letters, numbers, spaces, hyphens within words)
    text = re.sub(r"[^\w\s\-']", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> Dict:
    """Compute Word Error Rate using word-level Levenshtein edit distance.

    Args:
        reference: Ground truth transcription.
        hypothesis: Predicted transcription.

    Returns:
        Dict with keys: wer, substitutions, insertions, deletions, ref_words, hyp_words.
        WER can exceed 1.0 if there are more errors than reference words.
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        return {
            "wer": 0.0 if m == 0 else float(m),
            "substitutions": 0,
            "insertions": m,
            "deletions": 0,
            "ref_words": 0,
            "hyp_words": m,
        }

    # DP table: dp[i][j] = (edit_distance, substitutions, insertions, deletions)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0, i)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1]
                ins = dp[i][j - 1]
                dele = dp[i - 1][j]

                sub_cost = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])
                ins_cost = (ins[0] + 1, ins[1], ins[2] + 1, ins[3])
                del_cost = (dele[0] + 1, dele[1], dele[2], dele[3] + 1)

                dp[i][j] = min(sub_cost, del_cost, ins_cost, key=lambda x: x[0])

    dist, subs, ins, dels = dp[n][m]
    return {
        "wer": dist / n,
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "ref_words": n,
        "hyp_words": m,
    }


def compute_timestamp_accuracy(
    predicted: List[Dict],
    reference: List[Dict],
) -> Dict:
    """Compute timestamp accuracy by aligning predicted words to reference words.

    Uses greedy left-to-right alignment on normalized text. For each matched pair,
    computes the start-time delta (predicted - reference).

    Args:
        predicted: List of dicts with keys: word, start, end.
        reference: List of dicts with keys: word, start, end.

    Returns:
        Dict with keys: mae_start, max_delta_start, median_delta_start,
        n_matched, n_ref, n_pred. Returns None values if no matches found.
    """
    if not predicted or not reference:
        return {
            "mae_start": None,
            "max_delta_start": None,
            "median_delta_start": None,
            "n_matched": 0,
            "n_ref": len(reference),
            "n_pred": len(predicted),
        }

    # Normalize words for matching
    pred_norm = [normalize_text(p["word"]) for p in predicted]
    ref_norm = [normalize_text(r["word"]) for r in reference]

    # Greedy left-to-right alignment
    deltas_start = []
    ref_idx = 0
    for p_idx, p_word in enumerate(pred_norm):
        if not p_word:
            continue
        # Scan forward in reference to find a match (allow small skips)
        search_limit = min(ref_idx + 3, len(ref_norm))
        for r_idx in range(ref_idx, search_limit):
            if ref_norm[r_idx] == p_word:
                delta = predicted[p_idx]["start"] - reference[r_idx]["start"]
                deltas_start.append(delta)
                ref_idx = r_idx + 1
                break

    if not deltas_start:
        return {
            "mae_start": None,
            "max_delta_start": None,
            "median_delta_start": None,
            "n_matched": 0,
            "n_ref": len(reference),
            "n_pred": len(predicted),
        }

    abs_deltas = [abs(d) for d in deltas_start]
    sorted_abs = sorted(abs_deltas)
    n = len(sorted_abs)
    if n % 2 == 1:
        median = sorted_abs[n // 2]
    else:
        median = (sorted_abs[n // 2 - 1] + sorted_abs[n // 2]) / 2

    return {
        "mae_start": sum(abs_deltas) / len(abs_deltas),
        "max_delta_start": max(abs_deltas),
        "median_delta_start": median,
        "n_matched": len(deltas_start),
        "n_ref": len(reference),
        "n_pred": len(predicted),
    }
