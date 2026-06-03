#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a simple eval manifest from audio files.")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--glob", default="*.wav")
    parser.add_argument("--source", default="audio_dir")
    parser.add_argument("--language", default=None)
    parser.add_argument("--language-code", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def audio_manifest_records(
    audio_dir: Path,
    *,
    glob: str,
    source: str,
    language: str | None = None,
    language_code: str | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    paths = sorted(path for path in audio_dir.glob(glob) if path.is_file())
    if limit is not None:
        paths = paths[:limit]
    records: list[dict[str, object]] = []
    for path in paths:
        record: dict[str, object] = {
            "audio": str(path.resolve()),
            "id": path.stem,
            "source": source,
        }
        if language:
            record["language"] = language
        if language_code:
            record["language_code"] = language_code
        records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    records = audio_manifest_records(
        args.audio_dir,
        glob=args.glob,
        source=args.source,
        language=args.language,
        language_code=args.language_code,
        limit=args.limit,
    )
    write_jsonl(args.output_jsonl, records)
    print(json.dumps({"output_jsonl": str(args.output_jsonl), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
