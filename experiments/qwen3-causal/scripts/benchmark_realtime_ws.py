#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import time
from pathlib import Path

import websockets
from tqdm import tqdm

from qwen3_streaming.audio_io import load_audio_mono, pcm16_bytes
from qwen3_streaming.metrics import stable_prefix_stats, word_error_rate
from qwen3_streaming.realtime import is_silent_pcm16, post_process_realtime_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vLLM /v1/realtime WebSocket STT.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--manifest-jsonl", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--glob", default="*.wav")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--chunk-ms", type=float, default=250.0)
    parser.add_argument("--sleep-realtime", action="store_true")
    parser.add_argument("--receive-timeout", type=float, default=120.0)
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


async def stream_one(args: argparse.Namespace, item: dict) -> dict:
    uri = f"ws://{args.host}:{args.port}/v1/realtime"
    path = Path(item["audio"])
    audio, sr = load_audio_mono(path, target_sr=16_000)
    chunk_bytes = max(2, int(round((args.chunk_ms / 1000.0) * sr)) * 2)
    payload = pcm16_bytes(audio)
    events: list[dict] = []
    transcript = ""
    t0 = time.perf_counter()
    first_delta_sec = None
    error = None

    try:
        async with websockets.connect(uri, max_size=None) as ws:
            await asyncio.wait_for(ws.recv(), timeout=args.receive_timeout)
            await ws.send(json.dumps({"type": "session.update", "model": args.model}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            async def send_audio() -> None:
                for start in range(0, len(payload), chunk_bytes):
                    chunk = payload[start : start + chunk_bytes]
                    if is_silent_pcm16(chunk):
                        continue
                    b64 = base64.b64encode(chunk).decode("ascii")
                    await ws.send(
                        json.dumps({"type": "input_audio_buffer.append", "audio": b64})
                    )
                    if args.sleep_realtime:
                        await asyncio.sleep(args.chunk_ms / 1000.0)
                await ws.send(
                    json.dumps({"type": "input_audio_buffer.commit", "final": True})
                )

            send_task = asyncio.create_task(send_audio())
            while True:
                try:
                    message = await asyncio.wait_for(
                        ws.recv(), timeout=args.receive_timeout
                    )
                except asyncio.TimeoutError:
                    break
                data = json.loads(message)
                event_type = data.get("type")
                now = time.perf_counter()
                events.append({"t": now - t0, "event": data})
                if event_type == "transcription.delta":
                    if first_delta_sec is None:
                        first_delta_sec = now - t0
                    transcript += data.get("delta", "")
                elif event_type == "transcription.done":
                    transcript = (
                        data.get("text")
                        or data.get("transcript")
                        or data.get("final_text")
                        or transcript
                    )
                    break
                elif event_type == "error":
                    error = json.dumps(data)
                    break
                if send_task.done() and event_type in {"transcription.completed", "done"}:
                    break
            await send_task
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    latency = time.perf_counter() - t0
    transcript = post_process_realtime_text(transcript)
    reference = item.get("reference")
    stable = stable_prefix_stats(reference, transcript) if reference else None
    return {
        "audio": str(path),
        "source": item.get("source", ""),
        "reference": reference,
        "hypothesis": transcript,
        "latency_sec": latency,
        "first_delta_sec": first_delta_sec,
        "wer": word_error_rate(reference, transcript) if reference else None,
        "common_prefix_words": stable.common_prefix_words if stable else None,
        "revision_words": stable.revision_words if stable else None,
        "events": events,
        "error": error,
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    return values[int(round((len(values) - 1) * pct))]


async def amain() -> None:
    args = parse_args()
    items = load_items(args)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    latencies: list[float] = []
    first_deltas: list[float] = []
    wers: list[float] = []
    revisions: list[int] = []

    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for item in tqdm(items, desc="realtime"):
            result = await stream_one(args, item)
            if result["error"] is None:
                latencies.append(result["latency_sec"])
                if result["first_delta_sec"] is not None:
                    first_deltas.append(result["first_delta_sec"])
                if result["wer"] is not None:
                    wers.append(result["wer"])
                if result["revision_words"] is not None:
                    revisions.append(int(result["revision_words"]))
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

    summary = {
        "count": len(items),
        "ok": len(latencies),
        "latency_p50": percentile(latencies, 0.50),
        "latency_p95": percentile(latencies, 0.95),
        "first_delta_p50": percentile(first_deltas, 0.50),
        "first_delta_p95": percentile(first_deltas, 0.95),
        "wer_mean": statistics.mean(wers) if wers else None,
        "revision_words_mean": statistics.mean(revisions) if revisions else None,
    }
    summary_path = args.output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
