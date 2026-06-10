#!/usr/bin/env python3
"""Re-score per-item eval JSONLs against human MCIF references.

Historical evals scored hypotheses against Qwen3-ASR-1.7B teacher transcripts
with minimal normalization (lowercase + whitespace). This tool recomputes WER
for existing per-item JSONL prediction files under two reference sets
(teacher = the row's own ``reference`` field, human = MCIF manifest) and two
normalizers (legacy, Whisper EnglishTextNormalizer), so conclusions drawn from
the old numbers can be re-checked without re-running any GPU inference.

Audio ids are matched on the wav basename. Chunked items named
``{id}_{idx}_{start_ms}_{end_ms}`` get a time-windowed human reference built
from MCIF segments whose midpoint falls inside the window.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from jiwer import wer as jiwer_wer
from whisper_normalizer.english import EnglishTextNormalizer

_SPACE_RE = re.compile(r"\s+")
_CHUNK_RE = re.compile(r"^(?P<id>.+?)_(?P<idx>\d{4})_(?P<start>\d{8})_(?P<end>\d{8})$")
# Chunk filenames encode the time window in milliseconds, e.g.
# EqmWoxNDIr_0000_00000000_00020000 covers 0-20s.
_MS_PER_SEC = 1000.0

_english_normalizer = EnglishTextNormalizer()


def legacy_normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", text.strip().lower())


def whisper_normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", _english_normalizer(text)).strip()


def safe_wer(reference: str, hypothesis: str) -> float | None:
    if not reference:
        return None
    return float(jiwer_wer(reference, hypothesis))


def load_human_manifest(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rows[row["audio_id"]] = row
    return rows


def human_reference_for(stem: str, manifest: dict[str, dict]) -> str | None:
    """Full-file or time-windowed human reference for a wav basename stem."""
    if stem in manifest:
        return manifest[stem]["human_text"]
    match = _CHUNK_RE.match(stem)
    if not match or match.group("id") not in manifest:
        return None
    start_sec = int(match.group("start")) / _MS_PER_SEC
    end_sec = int(match.group("end")) / _MS_PER_SEC
    texts = []
    for seg in manifest[match.group("id")]["segments"]:
        midpoint = seg["offset_sec"] + seg["duration_sec"] / 2.0
        if start_sec <= midpoint < end_sec:
            texts.append(seg["text"])
    return " ".join(texts) if texts else None


def rescore_file(jsonl_path: Path, manifest: dict[str, dict]) -> dict:
    per_item = []
    with jsonl_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                continue
            stem = Path(row["audio"]).stem
            hyp = row.get("final_text") or ""
            teacher_ref = row.get("reference") or ""
            human_ref = human_reference_for(stem, manifest)
            item = {
                "audio_id": stem,
                "wer_teacher_legacy": safe_wer(legacy_normalize(teacher_ref), legacy_normalize(hyp)),
                "wer_teacher_whisper": safe_wer(whisper_normalize(teacher_ref), whisper_normalize(hyp)),
                "wer_human_legacy": None,
                "wer_human_whisper": None,
                "wer_final_original": row.get("wer_final"),
            }
            if human_ref:
                item["wer_human_legacy"] = safe_wer(legacy_normalize(human_ref), legacy_normalize(hyp))
                item["wer_human_whisper"] = safe_wer(whisper_normalize(human_ref), whisper_normalize(hyp))
            stable = row.get("stable_committed_text")
            if stable and human_ref:
                item["wer_stable_human_whisper"] = safe_wer(
                    whisper_normalize(human_ref), whisper_normalize(stable)
                )
            per_item.append(item)

    def mean(key: str) -> float | None:
        values = [item[key] for item in per_item if item.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "jsonl": str(jsonl_path),
        "items": len(per_item),
        "items_with_human_ref": sum(1 for item in per_item if item["wer_human_whisper"] is not None),
        "wer_final_original_mean": mean("wer_final_original"),
        "wer_teacher_legacy_mean": mean("wer_teacher_legacy"),
        "wer_teacher_whisper_mean": mean("wer_teacher_whisper"),
        "wer_human_legacy_mean": mean("wer_human_legacy"),
        "wer_human_whisper_mean": mean("wer_human_whisper"),
        "wer_stable_human_whisper_mean": mean("wer_stable_human_whisper"),
        "per_item": per_item,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonls", nargs="+", type=Path, help="Per-item prediction JSONL files.")
    parser.add_argument(
        "--human-manifest",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "mcif_refs" / "manifest.human.jsonl",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Write full results here.")
    args = parser.parse_args()

    manifest = load_human_manifest(args.human_manifest)
    results = [rescore_file(path, manifest) for path in args.jsonls]

    header = (
        f"{'file':58s} {'n':>3s} {'orig':>6s} {'t-leg':>6s} {'t-whi':>6s} {'h-leg':>6s} {'h-whi':>6s}"
    )
    print(header)
    print("-" * len(header))
    for res in results:
        def fmt(value):
            return f"{value:6.4f}" if value is not None else "     -"

        name = Path(res["jsonl"]).name
        print(
            f"{name[:58]:58s} {res['items']:3d} {fmt(res['wer_final_original_mean'])} "
            f"{fmt(res['wer_teacher_legacy_mean'])} {fmt(res['wer_teacher_whisper_mean'])} "
            f"{fmt(res['wer_human_legacy_mean'])} {fmt(res['wer_human_whisper_mean'])}"
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
