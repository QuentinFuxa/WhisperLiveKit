#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import requests
from tqdm import tqdm

from qwen3_streaming.metrics import normalize_text, word_error_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vLLM /v1/audio/transcriptions.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--manifest-jsonl", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--glob", default="*.wav")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--language", default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_items(args: argparse.Namespace) -> list[dict]:
    if args.manifest_jsonl:
        items = []
        with args.manifest_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    items.append(
                        {
                            "audio": record["audio"],
                            "reference": record.get("raw_text") or record.get("text"),
                            "source": record.get("source", ""),
                        }
                    )
        return items[: args.limit] if args.limit else items

    if not args.audio_dir:
        raise ValueError("Pass --manifest-jsonl or --audio-dir.")
    items = [{"audio": str(path), "reference": None, "source": ""} for path in sorted(args.audio_dir.glob(args.glob))]
    return items[: args.limit] if args.limit else items


def transcribe(base_url: str, model: str, path: Path, language: str | None, timeout: float) -> tuple[str, dict]:
    url = base_url.rstrip("/") + "/audio/transcriptions"
    data = {"model": model}
    if language:
        data["language"] = language
    with path.open("rb") as f:
        files = {"file": (path.name, f, "audio/wav")}
        response = requests.post(url, data=data, files=files, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    text = payload.get("text") or payload.get("transcript") or ""
    return text, payload


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[int(idx)]


def main() -> None:
    args = parse_args()
    items = load_items(args)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    latencies: list[float] = []
    wers: list[float] = []
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for item in tqdm(items, desc="transcriptions"):
            path = Path(item["audio"])
            t0 = time.perf_counter()
            error = None
            text = ""
            payload = {}
            try:
                text, payload = transcribe(
                    args.base_url, args.model, path, args.language, args.timeout
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            latency = time.perf_counter() - t0
            if error is None:
                latencies.append(latency)

            ref = item.get("reference")
            wer_value = word_error_rate(ref, text) if ref else None
            if wer_value is not None:
                wers.append(wer_value)

            out.write(
                json.dumps(
                    {
                        "audio": str(path),
                        "source": item.get("source", ""),
                        "reference": ref,
                        "hypothesis": text,
                        "hypothesis_norm": normalize_text(text),
                        "latency_sec": latency,
                        "wer": wer_value,
                        "error": error,
                        "raw_response": payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()

    summary = {
        "count": len(items),
        "ok": len(latencies),
        "latency_p50": percentile(latencies, 0.50),
        "latency_p95": percentile(latencies, 0.95),
        "latency_mean": statistics.mean(latencies) if latencies else None,
        "wer_mean": statistics.mean(wers) if wers else None,
    }
    summary_path = args.output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
