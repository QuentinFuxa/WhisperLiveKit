#!/usr/bin/env python3
"""Offline Qwen3-ASR baseline, the recommended long-form way: VAD-segment the
file, transcribe each speech segment offline (short generations, no
long-decode drift), concatenate.

This is the fair offline reference for long files: the bare
``transcribe()`` call only splits above MAX_ASR_INPUT_SECONDS=1200 s, so a
5-7 min file is one autoregressive generation and drifts. Every long-form
ASR framework segments first; this reproduces that with Silero VAD.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000


def vad_segments(audio: np.ndarray, get_ts, collect, model, *, max_sec, min_sec=1.0):
    """Silero speech timestamps, merged into <= max_sec segments."""
    import torch

    ts = get_ts(torch.from_numpy(audio), model, sampling_rate=SR)
    merged = []
    cur = None
    for seg in ts:
        s, e = seg["start"], seg["end"]
        if cur is None:
            cur = [s, e]
        elif (e - cur[0]) / SR <= max_sec:
            cur[1] = e
        else:
            merged.append(tuple(cur))
            cur = [s, e]
    if cur is not None:
        merged.append(tuple(cur))
    # absorb tiny trailing fragments into the previous segment
    out = []
    for s, e in merged:
        if out and (e - s) / SR < min_sec:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--max-segment-sec", type=float, default=28.0)
    parser.add_argument("--experiments-dir", type=Path,
                        default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    sys.path.insert(0, str(args.experiments_dir))
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel
    from qwen3_streaming.metrics import word_error_rate

    vad_model, vad_utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True
    )
    get_speech_timestamps = vad_utils[0]

    asr = Qwen3ASRModel.LLM(
        model=args.model_id,
        gpu_memory_utilization=0.6,
        max_inference_batch_size=8,
    )

    rows = [json.loads(l) for l in args.manifest_jsonl.read_text().splitlines() if l.strip()]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    wers = []
    with args.output_jsonl.open("w") as out:
        for i, row in enumerate(rows):
            path = row.get("audio") or row.get("wav")
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            segs = vad_segments(audio, get_speech_timestamps, None, vad_model,
                                max_sec=args.max_segment_sec)
            started = time.perf_counter()
            pieces = []
            for s, e in segs:
                seg_audio = audio[s:e]
                res = asr.transcribe((seg_audio, SR), language=args.language)
                pieces.append(res[0].text if res else "")
            text = " ".join(p.strip() for p in pieces if p.strip()).strip()
            elapsed = time.perf_counter() - started
            ref = row.get("teacher_text") or row.get("text") or row.get("reference") or ""
            wer = word_error_rate(ref, text) if ref else None
            if wer is not None:
                wers.append(wer)
            out.write(json.dumps({
                "audio": str(path), "final_text": text, "reference": ref,
                "wer_final": wer, "n_segments": len(segs),
                "audio_sec": len(audio) / SR, "decode_sec": elapsed,
            }) + "\n")
            out.flush()
            print(f"[{i+1}/{len(rows)}] segs={len(segs)} "
                  f"wer={None if wer is None else round(wer,4)} {Path(path).name}",
                  flush=True)

    summary = {
        "count": len(rows),
        "wer_final_mean": sum(wers) / len(wers) if wers else None,
        "max_segment_sec": args.max_segment_sec,
        "method": "Silero VAD segmentation + per-segment offline transcribe",
    }
    args.output_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
