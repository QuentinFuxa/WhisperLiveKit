#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from qwen3_streaming.audio_io import load_audio_mono
from qwen3_streaming.metrics import token_repetition_stats
from qwen3_streaming.native_realtime_model import load_realtime_model
from qwen3_streaming.realtime_features import (
    decode_realtime_token_ids,
    log_mel_spectrogram,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy streaming inference for a native realtime ASR checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-ms", type=float, default=320.0)
    parser.add_argument("--emit-threshold", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--max-consecutive-text-tokens", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetition_penalty <= 0.0:
        raise ValueError("--repetition-penalty must be > 0")
    if args.no_repeat_ngram_size < 0:
        raise ValueError("--no-repeat-ngram-size must be >= 0")
    if args.max_consecutive_text_tokens < 0:
        raise ValueError("--max-consecutive-text-tokens must be >= 0")
    device = torch.device(args.device)
    model = load_realtime_model(
        args.checkpoint,
        map_location="cpu",
    ).to(device)
    if args.emit_threshold is not None and hasattr(model, "emit_threshold"):
        model.emit_threshold = float(args.emit_threshold)
    if hasattr(model, "repetition_penalty"):
        model.repetition_penalty = float(args.repetition_penalty)
    if hasattr(model, "no_repeat_ngram_size"):
        model.no_repeat_ngram_size = int(args.no_repeat_ngram_size)
    if hasattr(model, "max_consecutive_text_tokens"):
        model.max_consecutive_text_tokens = int(args.max_consecutive_text_tokens)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / "tokenizer")
    wait_token_id = int(tokenizer.convert_tokens_to_ids("[P]"))
    word_start_token_id = int(tokenizer.convert_tokens_to_ids("[W]"))

    audio, sr = load_audio_mono(args.audio, target_sr=model.config.sample_rate)
    mel = log_mel_spectrogram(audio, sr, model.config).to(device)
    chunk_frames = max(1, int(round(args.chunk_ms / model.config.mel_hop_ms)))

    state = model.init_stream_state(batch_size=1, device=device)
    all_tokens: list[int] = []
    with torch.no_grad():
        for start in range(0, mel.shape[0], chunk_frames):
            chunk = mel[start : start + chunk_frames, :].unsqueeze(0)
            _, tokens, state = model.stream_chunk(chunk, state)
            all_tokens.extend(int(token_id) for token_id in tokens.reshape(-1).tolist())

    hypothesis = decode_realtime_token_ids(
        tokenizer,
        all_tokens,
        wait_token_id=wait_token_id,
        word_start_token_id=word_start_token_id,
    )
    ignored_tokens = {wait_token_id, word_start_token_id, -100}
    repetition = token_repetition_stats(
        all_tokens,
        ignored_token_ids=ignored_tokens,
    )
    total_tokens = len(all_tokens)
    wait_tokens = sum(1 for token_id in all_tokens if token_id == wait_token_id)
    word_start_tokens = sum(
        1 for token_id in all_tokens if token_id == word_start_token_id
    )
    text_tokens = int(repetition["text_token_count"])
    payload = {
        "checkpoint": str(args.checkpoint),
        "audio": str(args.audio),
        "hypothesis": hypothesis,
        "tokens": len(all_tokens),
        "text_tokens": text_tokens,
        "wait_tokens": wait_tokens,
        "word_start_tokens": word_start_tokens,
        "wait_token_ratio": float(wait_tokens / total_tokens) if total_tokens else 0.0,
        "text_token_ratio": float(text_tokens / total_tokens) if total_tokens else 0.0,
        "wait_to_text_ratio": (
            float(wait_tokens / text_tokens) if text_tokens else None
        ),
        "hypothesis_char_length": len(hypothesis),
        "hypothesis_word_count": len(hypothesis.split()),
        "repetition": repetition,
        "audio_frames_seen": state.audio.frames_seen,
        "decoder_steps_seen": state.decoder.steps_seen,
        "chunk_ms": args.chunk_ms,
        "emit_threshold": getattr(model, "emit_threshold", None),
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "max_consecutive_text_tokens": args.max_consecutive_text_tokens,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(hypothesis)


if __name__ == "__main__":
    main()
