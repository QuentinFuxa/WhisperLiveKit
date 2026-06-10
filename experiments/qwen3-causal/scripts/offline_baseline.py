#!/usr/bin/env python3
"""Offline full-file Qwen3-ASR baseline on a manifest of WAVs.

Uses the official ``qwen_asr.Qwen3ASRModel.transcribe`` (no streaming, no
window) to establish the quality upper bound the streaming paths are compared
against. Scores against a chosen reference field with both the legacy and
Whisper-normalized WER (same scorers as scripts/rescore_jsonl.py).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rescore_jsonl import legacy_normalize, safe_wer, whisper_normalize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        required=True,
        help="Manifest with one row per item; needs an audio path or wav name.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Resolve relative 'wav'/'audio' fields against this directory.",
    )
    parser.add_argument("--reference-field", default="human_text")
    parser.add_argument("--language", required=True, help="e.g. English")
    parser.add_argument("--context", default="")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()
    return args


def resolve_audio(record: dict, audio_dir: Path | None) -> Path:
    raw = record.get("audio") or record.get("wav")
    if raw is None:
        raise ValueError(f"manifest row has no audio/wav field: {list(record)}")
    path = Path(raw)
    if not path.is_absolute() and audio_dir is not None:
        path = audio_dir / path.name
    return path


def main() -> None:
    args = parse_args()
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        args.model_id,
        max_new_tokens=args.max_new_tokens,
    )

    rows = [
        json.loads(line)
        for line in args.manifest_jsonl.read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.output_jsonl.open("w") as out:
        for record in rows:
            audio_path = resolve_audio(record, args.audio_dir)
            reference = record.get(args.reference_field) or ""
            started = time.perf_counter()
            transcription = model.transcribe(
                str(audio_path),
                context=args.context,
                language=args.language,
            )[0]
            latency = time.perf_counter() - started
            hypothesis = transcription.text or ""
            item = {
                "audio": str(audio_path),
                "audio_id": audio_path.stem,
                "model_id": args.model_id,
                "language": args.language,
                "latency_sec": latency,
                "reference_field": args.reference_field,
                "reference": reference,
                "hypothesis": hypothesis,
                "wer_legacy": safe_wer(
                    legacy_normalize(reference), legacy_normalize(hypothesis)
                ),
                "wer_whisper": safe_wer(
                    whisper_normalize(reference), whisper_normalize(hypothesis)
                ),
            }
            results.append(item)
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"{item['audio_id']:16s} wer_whisper="
                f"{item['wer_whisper'] if item['wer_whisper'] is not None else float('nan'):.4f} "
                f"latency={latency:.1f}s"
            )

    def mean(key: str) -> float | None:
        values = [item[key] for item in results if item.get(key) is not None]
        return sum(values) / len(values) if values else None

    summary = {
        "model_id": args.model_id,
        "items": len(results),
        "wer_legacy_mean": mean("wer_legacy"),
        "wer_whisper_mean": mean("wer_whisper"),
        "latency_mean_sec": mean("latency_sec"),
    }
    summary_path = args.output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
