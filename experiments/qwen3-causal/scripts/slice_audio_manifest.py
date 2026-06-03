#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen3_streaming.audio_io import (
    audio_duration_seconds,
    load_audio_mono,
    write_pcm16_wav,
)


def chunk_spans(
    duration_sec: float,
    *,
    chunk_sec: float,
    stride_sec: float,
    min_chunk_sec: float,
    max_chunks: int | None = None,
) -> list[tuple[float, float]]:
    if chunk_sec <= 0.0:
        raise ValueError("chunk_sec must be > 0")
    if stride_sec <= 0.0:
        raise ValueError("stride_sec must be > 0")
    if min_chunk_sec <= 0.0:
        raise ValueError("min_chunk_sec must be > 0")
    spans: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_sec:
        end = min(duration_sec, start + chunk_sec)
        if end - start >= min_chunk_sec:
            spans.append((start, end))
        if end >= duration_sec:
            break
        if max_chunks is not None and len(spans) >= max_chunks:
            break
        start += stride_sec
    if max_chunks is not None:
        spans = spans[:max_chunks]
    return spans


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice audio manifest records into short WAV chunks.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-audio-dir", type=Path, required=True)
    parser.add_argument("--chunk-sec", type=float, default=20.0)
    parser.add_argument("--stride-sec", type=float, default=None)
    parser.add_argument("--min-chunk-sec", type=float, default=5.0)
    parser.add_argument("--max-chunks-per-audio", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stride_sec = float(args.stride_sec if args.stride_sec is not None else args.chunk_sec)
    records = read_jsonl(args.input_jsonl)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_audio_dir.mkdir(parents=True, exist_ok=True)

    output_records: list[dict[str, object]] = []
    for record in records:
        audio_path = Path(str(record["audio"]))
        audio, sample_rate = load_audio_mono(audio_path, target_sr=args.sample_rate)
        duration_sec = audio_duration_seconds(audio, sample_rate)
        stem = str(record.get("id") or audio_path.stem)
        for index, (start_sec, end_sec) in enumerate(
            chunk_spans(
                duration_sec,
                chunk_sec=float(args.chunk_sec),
                stride_sec=stride_sec,
                min_chunk_sec=float(args.min_chunk_sec),
                max_chunks=args.max_chunks_per_audio,
            )
        ):
            start_sample = int(round(start_sec * sample_rate))
            end_sample = int(round(end_sec * sample_rate))
            chunk_audio = audio[start_sample:end_sample]
            chunk_id = f"{stem}_{index:04d}_{int(round(start_sec * 1000)):08d}_{int(round(end_sec * 1000)):08d}"
            chunk_path = (args.output_audio_dir / f"{chunk_id}.wav").resolve()
            write_pcm16_wav(chunk_path, chunk_audio, sample_rate)
            output_records.append(
                {
                    "audio": str(chunk_path),
                    "id": chunk_id,
                    "source": str(record.get("source", "audio_chunks")),
                    "parent_audio": str(audio_path),
                    "parent_id": stem,
                    "chunk_index": index,
                    "chunk_start_sec": start_sec,
                    "chunk_end_sec": end_sec,
                    "duration_sec": end_sec - start_sec,
                    "eval_only": True,
                }
            )

    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for record in output_records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "input_records": len(records),
        "output_records": len(output_records),
        "chunk_sec": float(args.chunk_sec),
        "stride_sec": stride_sec,
        "min_chunk_sec": float(args.min_chunk_sec),
        "max_chunks_per_audio": args.max_chunks_per_audio,
    }
    args.output_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
