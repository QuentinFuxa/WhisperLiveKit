#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from qwen3_streaming.metrics import normalize_text, word_error_rate


TranscribeFn = Callable[[Path, str | None], tuple[str, dict]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate a manifest with teacher ASR transcriptions from vLLM REST."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--language", default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent REST requests. Keep 1 for deterministic serial annotation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing output records keyed by audio path.",
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


def transcribe_vllm(
    *,
    base_url: str,
    model: str,
    timeout: float,
    path: Path,
    language: str | None,
) -> tuple[str, dict]:
    import requests

    url = base_url.rstrip("/") + "/audio/transcriptions"
    data = {"model": model}
    if language:
        data["language"] = language
    with path.open("rb") as f:
        files = {"file": (path.name, f, "audio/wav")}
        response = requests.post(url, data=data, files=files, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("text") or payload.get("transcript") or ""), payload


def annotate_record(
    record: dict[str, object],
    *,
    model: str,
    transcribe_fn: TranscribeFn,
) -> dict[str, object]:
    output = dict(record)
    t0 = time.perf_counter()
    teacher_text = ""
    teacher_error = None
    try:
        teacher_text, _payload = transcribe_fn(
            Path(str(record["audio"])),
            str(record.get("language_code") or "") or None,
        )
    except Exception as exc:  # noqa: BLE001
        teacher_error = str(exc)
    latency = time.perf_counter() - t0

    reference = str(record.get("text") or record.get("reference") or "")
    output["teacher_text"] = teacher_text
    output["teacher_text_norm"] = normalize_text(teacher_text)
    output["teacher_model"] = model
    output["teacher_wer"] = word_error_rate(reference, teacher_text) if reference else None
    output["teacher_latency_sec"] = latency
    output["teacher_error"] = teacher_error
    return output


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    latencies = [
        float(record["teacher_latency_sec"])
        for record in records
        if record.get("teacher_error") is None and record.get("teacher_latency_sec") is not None
    ]
    wers = [
        float(record["teacher_wer"])
        for record in records
        if record.get("teacher_error") is None and record.get("teacher_wer") is not None
    ]
    errors = sum(1 for record in records if record.get("teacher_error") is not None)
    return {
        "count": len(records),
        "ok": len(records) - errors,
        "errors": errors,
        "error_rate": float(errors / len(records)) if records else 0.0,
        "teacher_wer_mean": statistics.mean(wers) if wers else None,
        "teacher_latency_mean": statistics.mean(latencies) if latencies else None,
    }


def main() -> None:
    from tqdm import tqdm

    args = parse_args()
    records = read_jsonl(args.input_jsonl)
    if args.limit is not None:
        records = records[: args.limit]

    existing_by_audio: dict[str, dict[str, object]] = {}
    if args.resume and args.output_jsonl.exists():
        existing_by_audio = {
            str(record.get("audio")): record for record in read_jsonl(args.output_jsonl)
        }

    def transcribe_fn(path: Path, language: str | None) -> tuple[str, dict]:
        return transcribe_vllm(
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            path=path,
            language=args.language or language,
        )

    def annotate_or_reuse(record: dict[str, object]) -> dict[str, object]:
        existing = existing_by_audio.get(str(record.get("audio")))
        if existing is not None and existing.get("teacher_error") is None:
            return existing
        return annotate_record(
            record,
            model=args.model,
            transcribe_fn=transcribe_fn,
        )

    outputs: list[dict[str, object]] = []
    if args.workers <= 1:
        iterator = (annotate_or_reuse(record) for record in tqdm(records, desc="teacher"))
    else:
        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures = [executor.submit(annotate_or_reuse, record) for record in records]

        def concurrent_iterator():
            try:
                for future in tqdm(as_completed(futures), total=len(futures), desc="teacher"):
                    yield future.result()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        iterator = concurrent_iterator()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for annotated in iterator:
            outputs.append(annotated)
            out.write(json.dumps(annotated, ensure_ascii=False) + "\n")
            out.flush()

    summary = summarize(outputs)
    summary_path = args.output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
