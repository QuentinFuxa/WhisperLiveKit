#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Qwen3 forced word alignments to an existing audio/text manifest."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--language", default="English")
    parser.add_argument("--language-code", default="en")
    parser.add_argument("--aligner-model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument(
        "--failure-mode",
        choices=["error", "skip", "heuristic"],
        default="skip",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    from tqdm import tqdm

    from qwen3_streaming.audio_io import audio_duration_seconds, load_audio_mono
    from scripts.prepare_qwen_aligned_jsonl import (
        align_words,
        load_aligner,
        serialize_words,
    )

    args = parse_args()
    aligner = load_aligner(args)
    outputs: list[dict[str, object]] = []
    reject_reasons: Counter[str] = Counter()
    for record in tqdm(read_jsonl(args.input_jsonl), desc="align-manifest"):
        text = str(record.get(args.text_field) or "").strip()
        if not text:
            reject_reasons["missing_text"] += 1
            continue
        audio_path = Path(str(record["audio"]))
        audio, sample_rate = load_audio_mono(audio_path)
        duration_sec = audio_duration_seconds(audio, sample_rate)
        words, alignment_source = align_words(
            aligner,
            audio_path=audio_path,
            text=text,
            language=str(record.get("language") or args.language),
            duration_sec=duration_sec,
            failure_mode=args.failure_mode,
        )
        if not words:
            reject_reasons["alignment_failed"] += 1
            continue
        output = dict(record)
        output["text"] = text
        output["language"] = str(record.get("language") or args.language)
        output["language_code"] = str(record.get("language_code") or args.language_code)
        output["duration_sec"] = round(duration_sec, 4)
        output["alignment_source"] = alignment_source
        output["alignment_text_field"] = args.text_field
        output["word_alignments"] = serialize_words(words)
        outputs.append(output)

    write_jsonl(args.output_jsonl, outputs)
    summary = {
        "input": len(read_jsonl(args.input_jsonl)),
        "kept": len(outputs),
        "rejected": sum(reject_reasons.values()),
        "reject_reasons": dict(reject_reasons),
        "text_field": args.text_field,
        "aligner_model": args.aligner_model,
        "alignment_sources": dict(Counter(str(record["alignment_source"]) for record in outputs)),
    }
    args.output_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
