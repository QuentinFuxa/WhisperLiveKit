#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

from qwen3_streaming.audio_io import (
    audio_duration_seconds,
    load_audio_mono,
    write_pcm16_wav,
)
from qwen3_streaming.prefix_manifest import stable_id
from qwen3_streaming.realtime_targets import WordAlignment, heuristic_word_alignments


SOURCE_SPECS = {
    "fleurs_en": {
        "dataset": "google/fleurs",
        "config": "en_us",
        "train_split": "train",
        "eval_split": "validation",
        "text_columns": ("transcription", "raw_transcription", "text"),
        "language": "English",
        "language_code": "en",
    },
    "fleurs_fr": {
        "dataset": "google/fleurs",
        "config": "fr_fr",
        "train_split": "train",
        "eval_split": "validation",
        "text_columns": ("transcription", "raw_transcription", "text"),
        "language": "French",
        "language_code": "fr",
    },
    "librispeech_clean_100": {
        "dataset": "parquet",
        "config": None,
        "data_files": {
            "train.100": "hf://datasets/openslr/librispeech_asr/clean/train.100/*.parquet",
            "validation": "hf://datasets/openslr/librispeech_asr/clean/validation/*.parquet",
        },
        "train_split": "train.100",
        "eval_split": "validation",
        "text_columns": ("text", "transcription"),
        "language": "English",
        "language_code": "en",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build realtime ASR JSONL manifests with Qwen3 forced alignments."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["fleurs_en", "fleurs_fr"],
        choices=sorted(SOURCE_SPECS),
    )
    parser.add_argument("--max-train-per-source", type=int, default=128)
    parser.add_argument("--max-eval-per-source", type=int, default=16)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-audio-sec", type=float, default=30.0)
    parser.add_argument(
        "--drop-long-audio",
        action="store_true",
        help=(
            "Skip utterances longer than --max-audio-sec instead of truncating "
            "audio while keeping the full transcript."
        ),
    )
    parser.add_argument("--aligner-model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument(
        "--alignment-failure-mode",
        choices=["error", "skip", "heuristic"],
        default="error",
        help=(
            "error raises on forced-alignment failure, skip drops the row, "
            "heuristic uses linear word timings."
        ),
    )
    parser.add_argument(
        "--allow-heuristic-fallback",
        action="store_true",
        help="Deprecated alias for --alignment-failure-mode heuristic.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def pick_text(row: dict, columns: Iterable[str]) -> str:
    for column in columns:
        value = row.get(column)
        if value:
            return clean_text(str(value))
    return ""


def row_audio(row: dict) -> tuple[np.ndarray, int]:
    audio = row["audio"]
    if "array" in audio and audio["array"] is not None:
        array = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
    elif audio.get("path") and Path(str(audio["path"])).exists():
        array, sr = sf.read(str(audio["path"]), always_2d=False)
        array = np.asarray(array, dtype=np.float32)
    elif audio.get("bytes"):
        array, sr = sf.read(io.BytesIO(audio["bytes"]), always_2d=False)
        array = np.asarray(array, dtype=np.float32)
    else:
        raise ValueError("Audio row does not contain decoded samples, path, or bytes.")
    if array.ndim == 2:
        array = array.mean(axis=1)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:
        array = array / peak
    return array.astype(np.float32, copy=False), sr


def resample_if_needed(audio: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return audio.astype(np.float32, copy=False), sr
    import librosa

    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32), target_sr


def shuffled_indices(n_rows: int, rng: random.Random) -> list[int]:
    indices = list(range(n_rows))
    rng.shuffle(indices)
    return indices


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_aligner(args: argparse.Namespace):
    from qwen_asr import Qwen3ForcedAligner

    kwargs: dict[str, object] = {
        "dtype": torch_dtype(args.dtype),
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    return Qwen3ForcedAligner.from_pretrained(args.aligner_model, **kwargs)


def alignment_item_to_word(item) -> WordAlignment | None:
    if isinstance(item, dict):
        text = item.get("text") or item.get("word") or item.get("unit")
        start = item.get("start_time", item.get("start", item.get("begin")))
        end = item.get("end_time", item.get("end", item.get("finish")))
    else:
        text = getattr(item, "text", None)
        start = getattr(item, "start_time", getattr(item, "start", None))
        end = getattr(item, "end_time", getattr(item, "end", None))
    if text is None or start is None or end is None:
        return None
    text = clean_text(str(text))
    if not text:
        return None
    start_f = float(start)
    end_f = float(end)
    if end_f <= start_f:
        return None
    return WordAlignment(text=text, start_sec=start_f, end_sec=end_f)


def normalize_alignment_scale(
    words: list[WordAlignment],
    duration_sec: float,
) -> list[WordAlignment]:
    if not words:
        return []
    max_end = max(word.end_sec for word in words)
    if duration_sec > 0.0 and max_end > duration_sec * 3.0 and max_end > 100.0:
        words = [
            WordAlignment(
                text=word.text,
                start_sec=word.start_sec / 1000.0,
                end_sec=word.end_sec / 1000.0,
            )
            for word in words
        ]
    return [
        WordAlignment(
            text=word.text,
            start_sec=max(0.0, min(word.start_sec, duration_sec)),
            end_sec=max(0.0, min(word.end_sec, duration_sec)),
        )
        for word in words
        if word.end_sec > word.start_sec
    ]


def serialize_words(words: list[WordAlignment]) -> list[dict[str, object]]:
    return [
        {
            "text": word.text,
            "start_sec": round(float(word.start_sec), 4),
            "end_sec": round(float(word.end_sec), 4),
        }
        for word in words
    ]


def align_words(
    aligner,
    *,
    audio_path: Path,
    text: str,
    language: str,
    duration_sec: float,
    failure_mode: str,
) -> tuple[list[WordAlignment], str]:
    try:
        results = aligner.align(
            audio=str(audio_path),
            text=text,
            language=language,
        )
        first = results[0] if results else []
        words = [word for item in first if (word := alignment_item_to_word(item))]
        words = normalize_alignment_scale(words, duration_sec)
        if words:
            return words, "qwen3_forced_aligner"
        if failure_mode == "error":
            raise RuntimeError(f"Qwen3 forced alignment returned no words for {audio_path}")
    except Exception as exc:  # noqa: BLE001
        if failure_mode == "error":
            raise RuntimeError(f"Qwen3 forced alignment failed for {audio_path}: {exc}") from exc

    if failure_mode == "heuristic":
        return heuristic_word_alignments(text, duration_sec), "heuristic_fallback"
    return [], "failed"


def build_split(
    *,
    aligner,
    source_name: str,
    split_name: str,
    output_split: str,
    limit: int,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, object]]:
    spec = SOURCE_SPECS[source_name]
    if spec.get("data_files"):
        dataset = load_dataset(
            spec["dataset"],
            data_files={split_name: spec["data_files"][split_name]},
            split=split_name,
        )
    else:
        dataset = load_dataset(spec["dataset"], spec["config"], split=split_name)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    selected = shuffled_indices(len(dataset), rng)
    wav_root = args.out_dir / "wavs" / output_split / source_name
    records: list[dict[str, object]] = []

    for idx in tqdm(selected, desc=f"{source_name}:{output_split}:align"):
        if len(records) >= limit:
            break
        row = dataset[int(idx)]
        text = pick_text(row, spec["text_columns"])
        if not text:
            continue
        audio, sr = row_audio(row)
        audio, sr = resample_if_needed(audio, sr, args.sample_rate)
        original_duration = audio_duration_seconds(audio, sr)
        if args.max_audio_sec > 0.0 and original_duration > args.max_audio_sec:
            if args.drop_long_audio:
                continue
            audio = audio[: int(round(args.max_audio_sec * sr))]
        duration = audio_duration_seconds(audio, sr)
        if duration <= 0.5:
            continue

        item_id = stable_id(source_name, output_split, idx, text)
        audio_path = wav_root / f"{item_id}.wav"
        if not audio_path.exists() or not args.skip_existing:
            write_pcm16_wav(audio_path, audio, sr)

        words, alignment_source = align_words(
            aligner,
            audio_path=audio_path,
            text=text,
            language=spec["language"],
            duration_sec=duration,
            failure_mode=args.alignment_failure_mode,
        )
        if not words:
            continue
        records.append(
            {
                "audio": str(audio_path.resolve()),
                "text": text,
                "source": source_name,
                "language": spec["language"],
                "language_code": spec["language_code"],
                "duration_sec": round(duration, 4),
                "alignment_source": alignment_source,
                "word_alignments": serialize_words(words),
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.allow_heuristic_fallback:
        args.alignment_failure_mode = "heuristic"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    aligner = load_aligner(args)

    all_train: list[dict[str, object]] = []
    all_eval: list[dict[str, object]] = []
    for source_name in args.sources:
        spec = SOURCE_SPECS[source_name]
        all_train.extend(
            build_split(
                aligner=aligner,
                source_name=source_name,
                split_name=spec["train_split"],
                output_split="train",
                limit=args.max_train_per_source,
                args=args,
                rng=rng,
            )
        )
        all_eval.extend(
            build_split(
                aligner=aligner,
                source_name=source_name,
                split_name=spec["eval_split"],
                output_split="eval",
                limit=args.max_eval_per_source,
                args=args,
                rng=rng,
            )
        )
        write_jsonl(args.out_dir / "train_manifest.jsonl", all_train)
        write_jsonl(args.out_dir / "eval_manifest.jsonl", all_eval)

    rng.shuffle(all_train)
    rng.shuffle(all_eval)
    write_jsonl(args.out_dir / "train_manifest.jsonl", all_train)
    write_jsonl(args.out_dir / "eval_manifest.jsonl", all_eval)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir.resolve()),
                "train_records": len(all_train),
                "eval_records": len(all_eval),
                "aligner_model": args.aligner_model,
                "sources": args.sources,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
