#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from qwen3_streaming.audio_io import load_audio_mono
from qwen3_streaming.native_realtime_model import Qwen3ASRRealtimeQwenAudioSurgeryModel
from qwen3_streaming.realtime_config import RealtimeAudioConfig
from qwen3_streaming.realtime_features import log_mel_spectrogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 0.5 validation for the Qwen audio surgery backend."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--audio", nargs="*", type=Path, default=[])
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--max-audio-sec", type=float, default=30.0)
    parser.add_argument("--chunk-ms", type=float, default=320.0)
    parser.add_argument("--left-context-sec", type=float, default=2.0)
    parser.add_argument("--right-context-ms", type=int, default=640)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_ms(device: torch.device, fn):
    synchronize(device)
    start = time.perf_counter()
    output = fn()
    synchronize(device)
    return output, (time.perf_counter() - start) * 1000.0


def collect_audio_paths(args: argparse.Namespace) -> list[Path]:
    paths = [path for path in args.audio if path.exists()]
    if args.audio_dir is not None:
        for suffix in ("*.wav", "*.flac", "*.mp3", "*.m4a"):
            paths.extend(sorted(args.audio_dir.glob(suffix)))
    deduped: list[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped[: args.max_files]


def main() -> None:
    args = parse_args()
    audio_paths = collect_audio_paths(args)
    if not audio_paths:
        raise RuntimeError("No audio files found for validation")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[P]", "[W]"]})
    wait_token_id = int(tokenizer.convert_tokens_to_ids("[P]"))
    bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id or wait_token_id
    config = RealtimeAudioConfig(
        d_model=1024,
        qwen_audio_left_context_sec=args.left_context_sec,
        qwen_audio_right_context_ms=args.right_context_ms,
    )
    model = Qwen3ASRRealtimeQwenAudioSurgeryModel.from_qwen_pretrained(
        args.model_id,
        config=config,
        bos_token_id=int(bos_token_id),
        wait_token_id=wait_token_id,
        dtype=torch_dtype(args.dtype),
        device_map="cpu",
    ).to(device)
    model.eval()
    encoder = model.audio_encoder
    chunk_frames = max(1, int(round(args.chunk_ms / config.mel_hop_ms)))

    rows: list[dict[str, object]] = []
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for audio_path in audio_paths:
            audio, sr = load_audio_mono(audio_path, target_sr=config.sample_rate)
            if args.max_audio_sec > 0:
                max_samples = int(round(args.max_audio_sec * sr))
                audio = audio[:max_samples]
            duration_sec = float(len(audio)) / float(sr) if sr else 0.0
            mel = log_mel_spectrogram(audio, sr, config).unsqueeze(0).to(device)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            with torch.no_grad():
                full_hidden, full_ms = timed_ms(
                    device,
                    lambda: encoder.forward_full(mel),
                )
                state = encoder.init_state()
                stream_steps = 0
                chunk_times_ms: list[float] = []
                chunk_step_counts: list[int] = []
                max_recomputed_frames = 0
                max_recomputed_context_frames = 0
                for start in range(0, mel.shape[1], chunk_frames):
                    chunk = mel[:, start : start + chunk_frames, :]
                    chunk_out, chunk_ms_value = timed_ms(
                        device,
                        lambda chunk=chunk: encoder.forward_chunk(chunk, state),
                    )
                    out, state = chunk_out
                    stream_steps += int(out.shape[1])
                    chunk_step_counts.append(int(out.shape[1]))
                    chunk_times_ms.append(float(chunk_ms_value))
                    max_recomputed_frames = max(
                        max_recomputed_frames,
                        int(state.last_recomputed_frames),
                    )
                    max_recomputed_context_frames = max(
                        max_recomputed_context_frames,
                        int(state.last_recomputed_context_frames),
                    )

            peak_mem_gb = (
                float(torch.cuda.max_memory_allocated(device) / 1e9)
                if device.type == "cuda"
                else 0.0
            )
            row = {
                "audio": str(audio_path),
                "duration_sec": duration_sec,
                "mel_frames": int(mel.shape[1]),
                "full_steps": int(full_hidden.shape[1]),
                "stream_steps": int(stream_steps),
                "frames_seen": int(state.frames_seen),
                "emitted_steps": int(state.emitted_steps),
                "max_recomputed_frames": int(max_recomputed_frames),
                "max_recomputed_context_frames": int(max_recomputed_context_frames),
                "max_allowed_recompute_frames": int(encoder.max_recompute_frames),
                "window_start_frame": int(state.window_start_frame),
                "full_audio_ms": float(full_ms),
                "stream_total_audio_ms": float(np.sum(chunk_times_ms)),
                "stream_mean_chunk_ms": float(np.mean(chunk_times_ms)),
                "stream_max_chunk_ms": float(np.max(chunk_times_ms)),
                "chunk_frames": int(chunk_frames),
                "chunk_step_counts": chunk_step_counts,
                "peak_mem_gb": peak_mem_gb,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps(row, ensure_ascii=False))

    summary = {
        "model_id": args.model_id,
        "audio_count": len(rows),
        "chunk_ms": args.chunk_ms,
        "left_context_sec": args.left_context_sec,
        "right_context_ms": args.right_context_ms,
        "max_audio_sec": args.max_audio_sec,
        "max_allowed_recompute_frames": int(encoder.max_recompute_frames),
        "max_observed_recompute_frames": max(
            int(row["max_recomputed_frames"]) for row in rows
        ),
        "max_observed_context_recompute_frames": max(
            int(row["max_recomputed_context_frames"]) for row in rows
        ),
        "mean_stream_chunk_ms": float(
            np.mean([float(row["stream_mean_chunk_ms"]) for row in rows])
        ),
        "max_stream_chunk_ms": float(
            np.max([float(row["stream_max_chunk_ms"]) for row in rows])
        ),
        "mean_peak_mem_gb": float(
            np.mean([float(row["peak_mem_gb"]) for row in rows])
        ),
        "all_recompute_bounded": all(
            int(row["max_recomputed_frames"])
            <= int(row["max_allowed_recompute_frames"])
            for row in rows
        ),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
