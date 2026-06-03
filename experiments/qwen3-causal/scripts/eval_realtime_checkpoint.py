#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from qwen3_streaming.audio_io import load_audio_mono
from qwen3_streaming.ctc import ctc_greedy_decode
from qwen3_streaming.metrics import (
    merge_token_repetition_stats,
    normalize_text,
    token_repetition_stats,
    word_error_rate,
)
from qwen3_streaming.native_realtime_model import load_realtime_model
from qwen3_streaming.realtime_features import decode_realtime_token_ids, log_mel_spectrogram
from qwen3_streaming.rnnt import rnnt_greedy_decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a realtime checkpoint on audio records.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--glob", default="*.wav")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-ms", type=float, default=320.0)
    parser.add_argument("--emit-threshold", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--max-consecutive-text-tokens", type=int, default=0)
    parser.add_argument(
        "--rnnt-max-symbols-per-frame",
        type=int,
        default=8,
        help="Maximum nonblank RNNT symbols emitted before forcibly advancing a frame.",
    )
    parser.add_argument(
        "--decode-mode",
        choices=[
            "autoregressive",
            "ctc",
            "compact_ctc",
            "aligned_window_ce",
            "aligned_window_sampled_ce",
            "rnnt_lite",
            "rnnt_fb",
        ],
        default="autoregressive",
    )
    parser.add_argument(
        "--ctc-blank-logit-adjust",
        type=float,
        default=0.0,
        help=(
            "Add this value to the CTC blank/[P] logit before greedy argmax. "
            "Negative values suppress blank emission for calibration sweeps."
        ),
    )
    parser.add_argument("--reference-field", default="teacher_text")
    parser.add_argument("--fallback-reference-field", default="text")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_records(args: argparse.Namespace) -> list[dict[str, object]]:
    if bool(args.manifest_jsonl) == bool(args.audio_dir):
        raise ValueError("Pass exactly one of --manifest-jsonl or --audio-dir.")
    if args.manifest_jsonl:
        records = read_jsonl(args.manifest_jsonl)
    else:
        records = [
            {"audio": str(path.resolve()), "source": args.audio_dir.name, "id": path.stem}
            for path in sorted(args.audio_dir.glob(args.glob))
            if path.is_file()
        ]
    return records[: args.limit] if args.limit is not None else records


def configure_decoding(model, args: argparse.Namespace) -> None:
    if args.repetition_penalty <= 0.0:
        raise ValueError("--repetition-penalty must be > 0")
    if args.no_repeat_ngram_size < 0:
        raise ValueError("--no-repeat-ngram-size must be >= 0")
    if args.max_consecutive_text_tokens < 0:
        raise ValueError("--max-consecutive-text-tokens must be >= 0")
    if args.rnnt_max_symbols_per_frame <= 0:
        raise ValueError("--rnnt-max-symbols-per-frame must be > 0")
    if args.emit_threshold is not None and hasattr(model, "emit_threshold"):
        model.emit_threshold = float(args.emit_threshold)
    if hasattr(model, "repetition_penalty"):
        model.repetition_penalty = float(args.repetition_penalty)
    if hasattr(model, "no_repeat_ngram_size"):
        model.no_repeat_ngram_size = int(args.no_repeat_ngram_size)
    if hasattr(model, "max_consecutive_text_tokens"):
        model.max_consecutive_text_tokens = int(args.max_consecutive_text_tokens)


def banned_ngram_tokens(history: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 0 or len(history) < ngram_size - 1:
        return set()
    if ngram_size == 1:
        return set(history)
    prefix = tuple(history[-(ngram_size - 1) :])
    banned: set[int] = set()
    for start in range(0, len(history) - ngram_size + 1):
        ngram = tuple(history[start : start + ngram_size])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return banned


def apply_no_repeat_filter(
    token_ids: list[int],
    *,
    history: list[int],
    ngram_size: int,
) -> list[int]:
    if ngram_size <= 0:
        history.extend(token_ids)
        return token_ids
    kept: list[int] = []
    for token_id in token_ids:
        if token_id in banned_ngram_tokens(history, ngram_size):
            continue
        history.append(token_id)
        kept.append(token_id)
    return kept


def infer_record(
    *,
    model,
    tokenizer,
    record: dict[str, object],
    device: torch.device,
    chunk_ms: float,
    wait_token_id: int,
    word_start_token_id: int,
) -> dict[str, object]:
    audio_path = Path(str(record["audio"]))
    t0 = time.perf_counter()
    audio, sr = load_audio_mono(audio_path, target_sr=model.config.sample_rate)
    mel = log_mel_spectrogram(audio, sr, model.config).to(device)
    chunk_frames = max(1, int(round(chunk_ms / model.config.mel_hop_ms)))
    state = model.init_stream_state(batch_size=1, device=device)
    all_tokens: list[int] = []
    with torch.inference_mode():
        for start in range(0, mel.shape[0], chunk_frames):
            chunk = mel[start : start + chunk_frames, :].unsqueeze(0)
            _, tokens, state = model.stream_chunk(chunk, state)
            all_tokens.extend(int(token_id) for token_id in tokens.reshape(-1).tolist())
    latency = time.perf_counter() - t0
    hypothesis = decode_realtime_token_ids(
        tokenizer,
        all_tokens,
        wait_token_id=wait_token_id,
        word_start_token_id=word_start_token_id,
    )
    repetition = token_repetition_stats(
        all_tokens,
        ignored_token_ids={wait_token_id, word_start_token_id, -100},
    )
    total_tokens = len(all_tokens)
    wait_tokens = sum(1 for token_id in all_tokens if token_id == wait_token_id)
    word_start_tokens = sum(1 for token_id in all_tokens if token_id == word_start_token_id)
    text_tokens = int(repetition["text_token_count"])
    return {
        "hypothesis": hypothesis,
        "hypothesis_norm": normalize_text(hypothesis),
        "decode_mode": "autoregressive",
        "latency_sec": latency,
        "tokens": total_tokens,
        "text_tokens": text_tokens,
        "wait_tokens": wait_tokens,
        "word_start_tokens": word_start_tokens,
        "wait_token_ratio": float(wait_tokens / total_tokens) if total_tokens else 0.0,
        "text_token_ratio": float(text_tokens / total_tokens) if total_tokens else 0.0,
        "repetition": repetition,
        "audio_frames_seen": state.audio.frames_seen,
        "decoder_steps_seen": state.decoder.steps_seen,
    }


def infer_record_ctc(
    *,
    model,
    tokenizer,
    record: dict[str, object],
    device: torch.device,
    chunk_ms: float,
    wait_token_id: int,
    word_start_token_id: int,
    ctc_blank_logit_adjust: float = 0.0,
    no_repeat_ngram_size: int = 0,
) -> dict[str, object]:
    if not hasattr(model, "ctc_head"):
        raise ValueError("--decode-mode ctc requires a checkpoint with ctc_head")
    audio_path = Path(str(record["audio"]))
    t0 = time.perf_counter()
    audio, sr = load_audio_mono(audio_path, target_sr=model.config.sample_rate)
    mel = log_mel_spectrogram(audio, sr, model.config).to(device)
    chunk_frames = max(1, int(round(chunk_ms / model.config.mel_hop_ms)))
    state = model.init_stream_state(batch_size=1, device=device)
    emitted_tokens: list[int] = []
    total_tokens = 0
    wait_tokens = 0
    text_tokens = 0
    word_start_tokens = 0
    last_ctc_token_id: int | None = None
    emitted_history: list[int] = []
    with torch.inference_mode():
        for start in range(0, mel.shape[0], chunk_frames):
            chunk = mel[start : start + chunk_frames, :].unsqueeze(0)
            audio_hidden, state.audio = model.audio_encoder.forward_chunk(
                chunk,
                state.audio,
            )
            frame_hidden, state.adapter = model.adapter.forward_chunk(
                audio_hidden,
                state.adapter,
            )
            if frame_hidden.shape[1] == 0:
                continue
            logits = model.ctc_head(frame_hidden.to(dtype=model.ctc_head.weight.dtype))
            if ctc_blank_logit_adjust:
                logits = logits.clone()
                logits[..., wait_token_id] += float(ctc_blank_logit_adjust)
            raw_ids = logits.argmax(dim=-1).reshape(-1).tolist()
            decoded = ctc_greedy_decode(
                raw_ids,
                blank_token_id=wait_token_id,
                ignored_token_ids={word_start_token_id, -100},
                previous_token_id=last_ctc_token_id,
            )
            if raw_ids:
                last_ctc_token_id = int(raw_ids[-1])
            total_tokens += len(decoded.raw_token_ids)
            wait_tokens += int(decoded.blank_count)
            text_tokens += int(decoded.raw_text_token_count)
            word_start_tokens += sum(
                1 for token_id in decoded.raw_token_ids if int(token_id) == word_start_token_id
            )
            emitted_tokens.extend(
                apply_no_repeat_filter(
                    decoded.token_ids,
                    history=emitted_history,
                    ngram_size=no_repeat_ngram_size,
                )
            )
    latency = time.perf_counter() - t0
    hypothesis = decode_realtime_token_ids(
        tokenizer,
        emitted_tokens,
        wait_token_id=wait_token_id,
        word_start_token_id=word_start_token_id,
    )
    repetition = token_repetition_stats(
        emitted_tokens,
        ignored_token_ids={wait_token_id, word_start_token_id, -100},
    )
    return {
        "hypothesis": hypothesis,
        "hypothesis_norm": normalize_text(hypothesis),
        "decode_mode": "ctc",
        "latency_sec": latency,
        "tokens": total_tokens,
        "text_tokens": text_tokens,
        "wait_tokens": wait_tokens,
        "word_start_tokens": word_start_tokens,
        "wait_token_ratio": float(wait_tokens / total_tokens) if total_tokens else 0.0,
        "text_token_ratio": float(text_tokens / total_tokens) if total_tokens else 0.0,
        "repetition": repetition,
        "audio_frames_seen": state.audio.frames_seen,
        "decoder_steps_seen": state.adapter.decoder_steps_seen,
    }


def infer_record_compact_ctc(
    *,
    model,
    tokenizer,
    record: dict[str, object],
    device: torch.device,
    chunk_ms: float,
    wait_token_id: int,
    word_start_token_id: int,
    ctc_blank_logit_adjust: float = 0.0,
    no_repeat_ngram_size: int = 0,
) -> dict[str, object]:
    head = getattr(model, "compact_ctc_head", None)
    token_ids = getattr(model, "compact_ctc_token_ids", None)
    blank_index = int(getattr(model, "compact_ctc_blank_index", 0))
    if head is None or token_ids is None:
        raise ValueError("--decode-mode compact_ctc requires a compact CTC checkpoint")
    compact_to_full = [int(token_id) for token_id in token_ids]

    audio_path = Path(str(record["audio"]))
    t0 = time.perf_counter()
    audio, sr = load_audio_mono(audio_path, target_sr=model.config.sample_rate)
    mel = log_mel_spectrogram(audio, sr, model.config).to(device)
    chunk_frames = max(1, int(round(chunk_ms / model.config.mel_hop_ms)))
    state = model.init_stream_state(batch_size=1, device=device)
    emitted_tokens: list[int] = []
    total_tokens = 0
    wait_tokens = 0
    text_tokens = 0
    last_ctc_token_id: int | None = None
    emitted_history: list[int] = []
    with torch.inference_mode():
        for start in range(0, mel.shape[0], chunk_frames):
            chunk = mel[start : start + chunk_frames, :].unsqueeze(0)
            audio_hidden, state.audio = model.audio_encoder.forward_chunk(
                chunk,
                state.audio,
            )
            frame_hidden, state.adapter = model.adapter.forward_chunk(
                audio_hidden,
                state.adapter,
            )
            if frame_hidden.shape[1] == 0:
                continue
            logits = head(frame_hidden.to(dtype=head.weight.dtype))
            if ctc_blank_logit_adjust:
                logits = logits.clone()
                logits[..., blank_index] += float(ctc_blank_logit_adjust)
            raw_ids = logits.argmax(dim=-1).reshape(-1).tolist()
            decoded = ctc_greedy_decode(
                raw_ids,
                blank_token_id=blank_index,
                previous_token_id=last_ctc_token_id,
            )
            if raw_ids:
                last_ctc_token_id = int(raw_ids[-1])
            full_token_ids = [compact_to_full[int(idx)] for idx in decoded.token_ids]
            total_tokens += len(decoded.raw_token_ids)
            wait_tokens += int(decoded.blank_count)
            text_tokens += int(decoded.raw_text_token_count)
            emitted_tokens.extend(
                apply_no_repeat_filter(
                    full_token_ids,
                    history=emitted_history,
                    ngram_size=no_repeat_ngram_size,
                )
            )
    latency = time.perf_counter() - t0
    hypothesis = decode_realtime_token_ids(
        tokenizer,
        emitted_tokens,
        wait_token_id=wait_token_id,
        word_start_token_id=word_start_token_id,
    )
    repetition = token_repetition_stats(
        emitted_tokens,
        ignored_token_ids={wait_token_id, word_start_token_id, -100},
    )
    return {
        "hypothesis": hypothesis,
        "hypothesis_norm": normalize_text(hypothesis),
        "decode_mode": "compact_ctc",
        "latency_sec": latency,
        "tokens": total_tokens,
        "text_tokens": text_tokens,
        "wait_tokens": wait_tokens,
        "word_start_tokens": 0,
        "wait_token_ratio": float(wait_tokens / total_tokens) if total_tokens else 0.0,
        "text_token_ratio": float(text_tokens / total_tokens) if total_tokens else 0.0,
        "repetition": repetition,
        "audio_frames_seen": state.audio.frames_seen,
        "decoder_steps_seen": state.adapter.decoder_steps_seen,
    }


def infer_record_rnnt_lite(
    *,
    model,
    tokenizer,
    record: dict[str, object],
    device: torch.device,
    chunk_ms: float,
    wait_token_id: int,
    word_start_token_id: int,
    ctc_blank_logit_adjust: float = 0.0,
    no_repeat_ngram_size: int = 0,
) -> dict[str, object]:
    token_ids = getattr(model, "rnnt_lite_token_ids", None)
    blank_index = int(getattr(model, "rnnt_lite_blank_index", 0))
    if token_ids is None or not hasattr(model, "forward_rnnt_lite_logits_from_frames"):
        raise ValueError("--decode-mode rnnt_lite requires an RNNT-lite checkpoint")
    compact_to_full = [int(token_id) for token_id in token_ids]

    audio_path = Path(str(record["audio"]))
    t0 = time.perf_counter()
    audio, sr = load_audio_mono(audio_path, target_sr=model.config.sample_rate)
    mel = log_mel_spectrogram(audio, sr, model.config).to(device)
    chunk_frames = max(1, int(round(chunk_ms / model.config.mel_hop_ms)))
    state = model.init_stream_state(batch_size=1, device=device)
    emitted_tokens: list[int] = []
    emitted_history: list[int] = []
    previous_compact = torch.full(
        (1, 1),
        blank_index,
        dtype=torch.long,
        device=device,
    )
    total_tokens = 0
    wait_tokens = 0
    text_tokens = 0
    with torch.inference_mode():
        for start in range(0, mel.shape[0], chunk_frames):
            chunk = mel[start : start + chunk_frames, :].unsqueeze(0)
            audio_hidden, state.audio = model.audio_encoder.forward_chunk(
                chunk,
                state.audio,
            )
            frame_hidden, state.adapter = model.adapter.forward_chunk(
                audio_hidden,
                state.adapter,
            )
            if frame_hidden.shape[1] == 0:
                continue
            for frame_index in range(frame_hidden.shape[1]):
                logits = model.forward_rnnt_lite_logits_from_frames(
                    frame_hidden[:, frame_index : frame_index + 1, :],
                    previous_compact,
                )
                if ctc_blank_logit_adjust:
                    logits = logits.clone()
                    logits[..., blank_index] += float(ctc_blank_logit_adjust)
                compact_id = int(logits[:, -1, :].argmax(dim=-1).item())
                total_tokens += 1
                if compact_id == blank_index:
                    wait_tokens += 1
                    continue
                text_tokens += 1
                previous_compact.fill_(compact_id)
                full_token_id = compact_to_full[compact_id]
                emitted_tokens.extend(
                    apply_no_repeat_filter(
                        [full_token_id],
                        history=emitted_history,
                        ngram_size=no_repeat_ngram_size,
                    )
                )
    latency = time.perf_counter() - t0
    hypothesis = decode_realtime_token_ids(
        tokenizer,
        emitted_tokens,
        wait_token_id=wait_token_id,
        word_start_token_id=word_start_token_id,
    )
    repetition = token_repetition_stats(
        emitted_tokens,
        ignored_token_ids={wait_token_id, word_start_token_id, -100},
    )
    return {
        "hypothesis": hypothesis,
        "hypothesis_norm": normalize_text(hypothesis),
        "decode_mode": "rnnt_lite",
        "latency_sec": latency,
        "tokens": total_tokens,
        "text_tokens": text_tokens,
        "wait_tokens": wait_tokens,
        "word_start_tokens": 0,
        "wait_token_ratio": float(wait_tokens / total_tokens) if total_tokens else 0.0,
        "text_token_ratio": float(text_tokens / total_tokens) if total_tokens else 0.0,
        "repetition": repetition,
        "audio_frames_seen": state.audio.frames_seen,
        "decoder_steps_seen": state.adapter.decoder_steps_seen,
    }


def infer_record_rnnt_greedy(
    *,
    model,
    tokenizer,
    record: dict[str, object],
    device: torch.device,
    chunk_ms: float,
    wait_token_id: int,
    word_start_token_id: int,
    ctc_blank_logit_adjust: float = 0.0,
    no_repeat_ngram_size: int = 0,
    max_symbols_per_frame: int = 8,
) -> dict[str, object]:
    token_ids = getattr(model, "rnnt_lite_token_ids", None)
    blank_index = int(getattr(model, "rnnt_lite_blank_index", 0))
    if token_ids is None or not hasattr(model, "forward_rnnt_lite_logits_from_frames"):
        raise ValueError("--decode-mode rnnt_fb requires an RNNT checkpoint")
    compact_to_full = [int(token_id) for token_id in token_ids]

    audio_path = Path(str(record["audio"]))
    t0 = time.perf_counter()
    audio, sr = load_audio_mono(audio_path, target_sr=model.config.sample_rate)
    mel = log_mel_spectrogram(audio, sr, model.config).to(device)
    chunk_frames = max(1, int(round(chunk_ms / model.config.mel_hop_ms)))
    state = model.init_stream_state(batch_size=1, device=device)
    emitted_tokens: list[int] = []
    emitted_history: list[int] = []
    previous_compact_id = blank_index
    total_decisions = 0
    wait_tokens = 0
    text_tokens = 0
    forced_advance_count = 0
    with torch.inference_mode():
        for start in range(0, mel.shape[0], chunk_frames):
            chunk = mel[start : start + chunk_frames, :].unsqueeze(0)
            audio_hidden, state.audio = model.audio_encoder.forward_chunk(
                chunk,
                state.audio,
            )
            frame_hidden, state.adapter = model.adapter.forward_chunk(
                audio_hidden,
                state.adapter,
            )
            if frame_hidden.shape[1] == 0:
                continue

            def step_fn(
                frame_index: int,
                previous_index: int,
                symbols_this_frame: int,
            ) -> int:
                del symbols_this_frame
                previous = torch.tensor(
                    [[int(previous_index)]],
                    dtype=torch.long,
                    device=device,
                )
                logits = model.forward_rnnt_lite_logits_from_frames(
                    frame_hidden[:, frame_index : frame_index + 1, :],
                    previous,
                )
                if ctc_blank_logit_adjust:
                    logits = logits.clone()
                    logits[..., blank_index] += float(ctc_blank_logit_adjust)
                return int(logits[:, -1, :].argmax(dim=-1).item())

            decoded = rnnt_greedy_decode(
                frame_count=int(frame_hidden.shape[1]),
                step_fn=step_fn,
                blank_index=blank_index,
                start_prediction_index=previous_compact_id,
                max_symbols_per_frame=max_symbols_per_frame,
            )
            previous_compact_id = int(decoded.last_prediction_index)
            total_decisions += int(decoded.decision_count)
            wait_tokens += int(decoded.blank_count)
            text_tokens += len(decoded.compact_token_ids)
            forced_advance_count += int(decoded.forced_advance_count)
            full_token_ids = [
                compact_to_full[int(compact_id)]
                for compact_id in decoded.compact_token_ids
            ]
            emitted_tokens.extend(
                apply_no_repeat_filter(
                    full_token_ids,
                    history=emitted_history,
                    ngram_size=no_repeat_ngram_size,
                )
            )
    latency = time.perf_counter() - t0
    hypothesis = decode_realtime_token_ids(
        tokenizer,
        emitted_tokens,
        wait_token_id=wait_token_id,
        word_start_token_id=word_start_token_id,
    )
    repetition = token_repetition_stats(
        emitted_tokens,
        ignored_token_ids={wait_token_id, word_start_token_id, -100},
    )
    return {
        "hypothesis": hypothesis,
        "hypothesis_norm": normalize_text(hypothesis),
        "decode_mode": "rnnt_fb",
        "latency_sec": latency,
        "tokens": total_decisions,
        "text_tokens": text_tokens,
        "wait_tokens": wait_tokens,
        "word_start_tokens": 0,
        "wait_token_ratio": float(wait_tokens / total_decisions)
        if total_decisions
        else 0.0,
        "text_token_ratio": float(text_tokens / total_decisions)
        if total_decisions
        else 0.0,
        "rnnt_forced_advance_count": forced_advance_count,
        "repetition": repetition,
        "audio_frames_seen": state.audio.frames_seen,
        "decoder_steps_seen": state.adapter.decoder_steps_seen,
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    return values[int(round((len(values) - 1) * pct))]


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    ok_rows = [row for row in rows if row.get("error") is None]
    latencies = [float(row["latency_sec"]) for row in ok_rows]
    wers = [float(row["wer"]) for row in ok_rows if row.get("wer") is not None]
    repetitions = [row["repetition"] for row in ok_rows if isinstance(row.get("repetition"), dict)]
    tokens = sum(int(row.get("tokens", 0)) for row in ok_rows)
    text_tokens = sum(int(row.get("text_tokens", 0)) for row in ok_rows)
    wait_tokens = sum(int(row.get("wait_tokens", 0)) for row in ok_rows)
    rnnt_forced_advances = sum(
        int(row.get("rnnt_forced_advance_count", 0)) for row in ok_rows
    )
    return {
        "count": len(rows),
        "ok": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "wer_mean": statistics.mean(wers) if wers else None,
        "latency_p50": percentile(latencies, 0.50),
        "latency_p95": percentile(latencies, 0.95),
        "latency_mean": statistics.mean(latencies) if latencies else None,
        "tokens": tokens,
        "text_token_ratio": float(text_tokens / tokens) if tokens else 0.0,
        "wait_token_ratio": float(wait_tokens / tokens) if tokens else 0.0,
        "rnnt_forced_advance_count": rnnt_forced_advances,
        "repetition": merge_token_repetition_stats(repetitions),
    }


def main() -> None:
    args = parse_args()
    records = load_records(args)
    device = torch.device(args.device)
    model = load_realtime_model(args.checkpoint, map_location="cpu").to(device)
    configure_decoding(model, args)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / "tokenizer")
    wait_token_id = int(tokenizer.convert_tokens_to_ids("[P]"))
    word_start_token_id = int(tokenizer.convert_tokens_to_ids("[W]"))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for record in tqdm(records, desc="realtime-eval"):
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
            row["error"] = None
            try:
                if args.decode_mode == "ctc":
                    result = infer_record_ctc(
                        model=model,
                        tokenizer=tokenizer,
                        record=record,
                        device=device,
                        chunk_ms=args.chunk_ms,
                        wait_token_id=wait_token_id,
                        word_start_token_id=word_start_token_id,
                        ctc_blank_logit_adjust=args.ctc_blank_logit_adjust,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                    )
                elif args.decode_mode in {
                    "compact_ctc",
                    "aligned_window_ce",
                    "aligned_window_sampled_ce",
                }:
                    result = infer_record_compact_ctc(
                        model=model,
                        tokenizer=tokenizer,
                        record=record,
                        device=device,
                        chunk_ms=args.chunk_ms,
                        wait_token_id=wait_token_id,
                        word_start_token_id=word_start_token_id,
                        ctc_blank_logit_adjust=args.ctc_blank_logit_adjust,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                    )
                    result["decode_mode"] = args.decode_mode
                elif args.decode_mode == "rnnt_lite":
                    result = infer_record_rnnt_lite(
                        model=model,
                        tokenizer=tokenizer,
                        record=record,
                        device=device,
                        chunk_ms=args.chunk_ms,
                        wait_token_id=wait_token_id,
                        word_start_token_id=word_start_token_id,
                        ctc_blank_logit_adjust=args.ctc_blank_logit_adjust,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                    )
                elif args.decode_mode == "rnnt_fb":
                    result = infer_record_rnnt_greedy(
                        model=model,
                        tokenizer=tokenizer,
                        record=record,
                        device=device,
                        chunk_ms=args.chunk_ms,
                        wait_token_id=wait_token_id,
                        word_start_token_id=word_start_token_id,
                        ctc_blank_logit_adjust=args.ctc_blank_logit_adjust,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        max_symbols_per_frame=args.rnnt_max_symbols_per_frame,
                    )
                else:
                    result = infer_record(
                        model=model,
                        tokenizer=tokenizer,
                        record=record,
                        device=device,
                        chunk_ms=args.chunk_ms,
                        wait_token_id=wait_token_id,
                        word_start_token_id=word_start_token_id,
                    )
                row.update(result)
                row["wer"] = (
                    word_error_rate(str(reference), str(row["hypothesis"]))
                    if reference
                    else None
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = str(exc)
                row["wer"] = None
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    summary = summarize(rows)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "decode_mode": args.decode_mode,
            "ctc_blank_logit_adjust": args.ctc_blank_logit_adjust,
            "emit_threshold": getattr(model, "emit_threshold", None),
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "max_consecutive_text_tokens": args.max_consecutive_text_tokens,
            "rnnt_max_symbols_per_frame": args.rnnt_max_symbols_per_frame,
        }
    )
    summary_path = args.output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
