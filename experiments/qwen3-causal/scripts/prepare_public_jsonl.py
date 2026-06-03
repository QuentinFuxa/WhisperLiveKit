#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from qwen3_streaming.audio_io import audio_duration_seconds
from qwen3_streaming.prefix_manifest import make_examples, stable_id


SOURCE_SPECS = {
    "fleurs_en": {
        "dataset": "google/fleurs",
        "config": "en_us",
        "train_split": "train",
        "eval_split": "validation",
        "text_columns": ("transcription", "raw_transcription", "text"),
        "language": "en",
    },
    "fleurs_fr": {
        "dataset": "google/fleurs",
        "config": "fr_fr",
        "train_split": "train",
        "eval_split": "validation",
        "text_columns": ("transcription", "raw_transcription", "text"),
        "language": "fr",
    },
    "librispeech_clean_100": {
        "dataset": "openslr/librispeech_asr",
        "config": "clean",
        "train_split": "train.100",
        "eval_split": "validation",
        "text_columns": ("text", "transcription"),
        "language": "en",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Qwen3-ASR SFT JSONL files from public FR/EN datasets."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["fleurs_en", "fleurs_fr", "librispeech_clean_100"],
        choices=sorted(SOURCE_SPECS),
    )
    parser.add_argument("--max-train-per-source", type=int, default=5000)
    parser.add_argument("--max-eval-per-source", type=int, default=500)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prefix-mode", action="store_true")
    parser.add_argument("--min-prefix-sec", type=float, default=2.0)
    parser.add_argument("--prefix-stride-sec", type=float, default=1.0)
    parser.add_argument("--right-context-sec", type=float, default=1.0)
    parser.add_argument("--include-empty-prefix", action="store_true")
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def pick_text(row: dict, columns: Iterable[str]) -> str:
    for column in columns:
        value = row.get(column)
        if value:
            return clean_text(str(value))
    return ""


def row_audio(row: dict) -> tuple[np.ndarray, int]:
    audio = row["audio"]
    array = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio["sampling_rate"])
    if array.ndim == 2:
        array = array.mean(axis=1)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:
        array = array / peak
    return array, sr


def resample_if_needed(audio: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return audio.astype(np.float32, copy=False), sr
    import librosa

    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32), target_sr


def shuffled_indices(n_rows: int, limit: int, rng: random.Random) -> list[int]:
    indices = list(range(n_rows))
    rng.shuffle(indices)
    return indices[: min(limit, n_rows)]


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_split(
    *,
    source_name: str,
    split_name: str,
    output_split: str,
    limit: int,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    spec = SOURCE_SPECS[source_name]
    dataset = load_dataset(spec["dataset"], spec["config"], split=split_name)
    selected = shuffled_indices(len(dataset), limit, rng)

    sft_records: list[dict] = []
    manifest_records: list[dict] = []
    wav_root = args.out_dir / "wavs" / output_split

    for idx in tqdm(selected, desc=f"{source_name}:{output_split}"):
        row = dataset[int(idx)]
        text = pick_text(row, spec["text_columns"])
        if not text:
            continue
        audio, sr = row_audio(row)
        audio, sr = resample_if_needed(audio, sr, args.sample_rate)
        if audio_duration_seconds(audio, sr) <= 0.0:
            continue

        item_id = stable_id(source_name, output_split, idx, text)
        examples = make_examples(
            audio=audio,
            sample_rate=sr,
            text=text,
            source=source_name,
            language=spec["language"],
            output_dir=wav_root,
            item_id=item_id,
            prefix_mode=args.prefix_mode,
            min_prefix_sec=args.min_prefix_sec,
            stride_sec=args.prefix_stride_sec,
            right_context_sec=args.right_context_sec,
            include_empty_prefix=args.include_empty_prefix,
        )
        for example in examples:
            sft_records.append(example.sft_record())
            manifest_records.append(example.manifest_record())

    return sft_records, manifest_records


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    all_train: list[dict] = []
    all_train_manifest: list[dict] = []
    all_eval: list[dict] = []
    all_eval_manifest: list[dict] = []

    for source_name in args.sources:
        spec = SOURCE_SPECS[source_name]
        train, train_manifest = build_split(
            source_name=source_name,
            split_name=spec["train_split"],
            output_split="train",
            limit=args.max_train_per_source,
            args=args,
            rng=rng,
        )
        eval_, eval_manifest = build_split(
            source_name=source_name,
            split_name=spec["eval_split"],
            output_split="eval",
            limit=args.max_eval_per_source,
            args=args,
            rng=rng,
        )
        all_train.extend(train)
        all_train_manifest.extend(train_manifest)
        all_eval.extend(eval_)
        all_eval_manifest.extend(eval_manifest)

    rng.shuffle(all_train)
    rng.shuffle(all_train_manifest)
    rng.shuffle(all_eval)
    rng.shuffle(all_eval_manifest)

    n_train = write_jsonl(args.out_dir / "train.jsonl", all_train)
    n_eval = write_jsonl(args.out_dir / "eval.jsonl", all_eval)
    write_jsonl(args.out_dir / "train_manifest.jsonl", all_train_manifest)
    write_jsonl(args.out_dir / "eval_manifest.jsonl", all_eval_manifest)

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir.resolve()),
                "train_records": n_train,
                "eval_records": n_eval,
                "prefix_mode": args.prefix_mode,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
