#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated float")
    return [float(item) for item in items]


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated integer")
    return [int(item) for item in items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep realtime decoding controls on one checkpoint and manifest."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--glob", default="*.wav")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk-ms", type=float, default=320.0)
    parser.add_argument("--emit-thresholds", default="0.5")
    parser.add_argument("--repetition-penalties", default="1.0")
    parser.add_argument("--no-repeat-ngram-sizes", default="0")
    parser.add_argument("--max-consecutive-text-tokens", default="0")
    parser.add_argument("--reference-field", default="teacher_text")
    parser.add_argument("--fallback-reference-field", default="text")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def config_name(config: dict[str, object]) -> str:
    return (
        f"th{float(config['emit_threshold']):.2f}"
        f"_rp{float(config['repetition_penalty']):.2f}"
        f"_ng{int(config['no_repeat_ngram_size'])}"
        f"_mx{int(config['max_consecutive_text_tokens'])}"
    ).replace(".", "p")


def iter_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    return [
        {
            "emit_threshold": emit_threshold,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "max_consecutive_text_tokens": max_consecutive_text_tokens,
        }
        for (
            emit_threshold,
            repetition_penalty,
            no_repeat_ngram_size,
            max_consecutive_text_tokens,
        ) in itertools.product(
            parse_float_list(args.emit_thresholds),
            parse_float_list(args.repetition_penalties),
            parse_int_list(args.no_repeat_ngram_sizes),
            parse_int_list(args.max_consecutive_text_tokens),
        )
    ]


def main() -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from qwen3_streaming.metrics import word_error_rate
    from qwen3_streaming.native_realtime_model import load_realtime_model
    from scripts.eval_realtime_checkpoint import (
        configure_decoding,
        infer_record,
        load_records,
        summarize,
    )

    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    records = load_records(args)
    configs = iter_configs(args)
    device = torch.device(args.device)

    model = load_realtime_model(args.checkpoint, map_location="cpu").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / "tokenizer")
    wait_token_id = int(tokenizer.convert_tokens_to_ids("[P]"))
    word_start_token_id = int(tokenizer.convert_tokens_to_ids("[W]"))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.records_dir:
        args.records_dir.mkdir(parents=True, exist_ok=True)

    with args.output_jsonl.open("w", encoding="utf-8") as summary_out:
        for config in configs:
            config_args = argparse.Namespace(
                emit_threshold=config["emit_threshold"],
                repetition_penalty=config["repetition_penalty"],
                no_repeat_ngram_size=config["no_repeat_ngram_size"],
                max_consecutive_text_tokens=config["max_consecutive_text_tokens"],
            )
            configure_decoding(model, config_args)
            rows: list[dict[str, object]] = []
            name = config_name(config)
            record_out = None
            if args.records_dir:
                record_out = (args.records_dir / f"{name}.jsonl").open(
                    "w",
                    encoding="utf-8",
                )
            try:
                for record in tqdm(records, desc=name):
                    row = dict(record)
                    reference = (
                        record.get(args.reference_field)
                        or record.get(args.fallback_reference_field)
                        or record.get("reference")
                    )
                    row["reference"] = reference
                    row["reference_field"] = (
                        args.reference_field
                        if record.get(args.reference_field)
                        else args.fallback_reference_field
                        if record.get(args.fallback_reference_field)
                        else None
                    )
                    row["sweep_config"] = config
                    row["error"] = None
                    try:
                        row.update(
                            infer_record(
                                model=model,
                                tokenizer=tokenizer,
                                record=record,
                                device=device,
                                chunk_ms=args.chunk_ms,
                                wait_token_id=wait_token_id,
                                word_start_token_id=word_start_token_id,
                            )
                        )
                        row["wer"] = (
                            word_error_rate(str(reference), str(row["hypothesis"]))
                            if reference
                            else None
                        )
                    except Exception as exc:  # noqa: BLE001
                        row["error"] = str(exc)
                        row["wer"] = None
                    rows.append(row)
                    if record_out is not None:
                        record_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        record_out.flush()
            finally:
                if record_out is not None:
                    record_out.close()

            summary = summarize(rows)
            summary.update(
                {
                    "checkpoint": str(args.checkpoint),
                    "sweep_config": config,
                    "chunk_ms": args.chunk_ms,
                    "records_limit": args.limit,
                }
            )
            summary_out.write(json.dumps(summary, ensure_ascii=False) + "\n")
            summary_out.flush()
            print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
