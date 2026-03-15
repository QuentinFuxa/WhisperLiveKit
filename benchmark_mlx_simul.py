#!/usr/bin/env python3
"""
Benchmark Qwen3-ASR MLX SimulStreaming on LibriSpeech test-clean.

Measures:
  - Word Error Rate (WER) via jiwer
  - Real-Time Factor (RTF) = total_inference_time / total_audio_duration
  - Per-utterance stats

Usage:
  # Per-utterance simul-streaming (default)
  python benchmark_mlx_simul.py --model-size 0.6b

  # Single-shot (batch-like, no streaming chunking)
  python benchmark_mlx_simul.py --model-size 0.6b --single-shot

  # Quick test with 100 utterances
  python benchmark_mlx_simul.py --model-size 0.6b --max-utterances 100

  # Chapter-grouped (matching H100 benchmark methodology)
  python benchmark_mlx_simul.py --model-size 0.6b --chapter-grouped
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from jiwer import wer as compute_wer, cer as compute_cer

# Add WhisperLiveKit to path
WLKIT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WLKIT_DIR))

from whisperlivekit.qwen3_mlx_simul import (
    Qwen3MLXSimulStreamingASR,
    Qwen3MLXSimulStreamingOnlineProcessor,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)

SAMPLE_RATE = 16_000

# Alignment heads paths
ALIGNMENT_HEADS = {
    "0.6b": str(WLKIT_DIR / "scripts" / "alignment_heads_qwen3_asr_0.6B.json"),
    "1.7b": str(WLKIT_DIR / "scripts" / "alignment_heads_qwen3_asr_1.7B_v2.json"),
}


def load_librispeech_utterances(data_dir: str, max_utterances: int = 0):
    """Load LibriSpeech utterances: yields (utt_id, audio_np, reference_text, duration_s)."""
    data_path = Path(data_dir)
    trans_files = sorted(data_path.rglob("*.trans.txt"))

    count = 0
    for trans_file in trans_files:
        chapter_dir = trans_file.parent
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                utt_id = parts[0]
                ref_text = parts[1] if len(parts) > 1 else ""

                flac_path = chapter_dir / f"{utt_id}.flac"
                if not flac_path.exists():
                    logger.warning("Missing FLAC: %s", flac_path)
                    continue

                audio, sr = sf.read(str(flac_path), dtype="float32")
                if sr != SAMPLE_RATE:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

                duration = len(audio) / SAMPLE_RATE
                yield utt_id, audio, ref_text, duration

                count += 1
                if max_utterances > 0 and count >= max_utterances:
                    return


def load_librispeech_chapters(data_dir: str):
    """Load LibriSpeech grouped by speaker-chapter.

    Concatenates all utterances within each speaker/chapter into one long audio.
    Returns list of (chapter_id, audio_np, reference_text, duration_s).
    """
    data_path = Path(data_dir)
    trans_files = sorted(data_path.rglob("*.trans.txt"))

    chapters = []
    for trans_file in trans_files:
        chapter_dir = trans_file.parent
        chapter_id = chapter_dir.name
        speaker_id = chapter_dir.parent.name
        full_id = f"{speaker_id}-{chapter_id}"

        audios = []
        refs = []
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                utt_id = parts[0]
                ref_text = parts[1] if len(parts) > 1 else ""

                flac_path = chapter_dir / f"{utt_id}.flac"
                if not flac_path.exists():
                    continue

                audio, sr = sf.read(str(flac_path), dtype="float32")
                if sr != SAMPLE_RATE:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

                audios.append(audio)
                refs.append(ref_text)

        if audios:
            # Concatenate with 0.5s silence between utterances
            silence = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
            combined = []
            for j, a in enumerate(audios):
                if j > 0:
                    combined.append(silence)
                combined.append(a)
            combined_audio = np.concatenate(combined)
            combined_ref = " ".join(refs)
            duration = len(combined_audio) / SAMPLE_RATE
            chapters.append((full_id, combined_audio, combined_ref, duration))

    return chapters


def transcribe_simul(asr, audio, chunk_seconds=2.0):
    """Transcribe using SimulStreaming with chunked audio feed.

    Returns (transcription_text, inference_time_seconds).
    """
    processor = Qwen3MLXSimulStreamingOnlineProcessor(asr)
    chunk_size = int(chunk_seconds * SAMPLE_RATE)
    total_samples = len(audio)
    offset = 0
    all_tokens = []

    t0 = time.perf_counter()

    while offset < total_samples:
        end = min(offset + chunk_size, total_samples)
        chunk = audio[offset:end]
        stream_time = end / SAMPLE_RATE

        processor.insert_audio_chunk(chunk, stream_time)

        is_last = (end >= total_samples)
        tokens, _ = processor.process_iter(is_last=is_last)
        if tokens:
            all_tokens.extend(tokens)
        offset = end

    # Final flush
    final_tokens, _ = processor.finish()
    if final_tokens:
        all_tokens.extend(final_tokens)

    t1 = time.perf_counter()
    inference_time = t1 - t0

    text = "".join(t.text for t in all_tokens).strip()
    return text, inference_time


def transcribe_single_shot(asr, audio):
    """Transcribe by feeding all audio at once (batch-like).

    Returns (transcription_text, inference_time_seconds).
    """
    processor = Qwen3MLXSimulStreamingOnlineProcessor(asr)

    t0 = time.perf_counter()

    duration = len(audio) / SAMPLE_RATE
    processor.insert_audio_chunk(audio, duration)
    all_tokens, _ = processor.process_iter(is_last=True)

    # Flush
    final_tokens, _ = processor.finish()
    if final_tokens:
        all_tokens.extend(final_tokens)

    t1 = time.perf_counter()
    inference_time = t1 - t0

    text = "".join(t.text for t in all_tokens).strip()
    return text, inference_time


def normalize_text(text: str) -> str:
    """Normalize text for WER computation: uppercase, strip punctuation."""
    text = text.upper()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-ASR MLX SimulStreaming")
    parser.add_argument("--model-size", default="0.6b", choices=["0.6b", "1.7b"],
                        help="Model size (default: 0.6b)")
    parser.add_argument("--max-utterances", type=int, default=0,
                        help="Max utterances to process (0=all). Ignored in chapter mode.")
    parser.add_argument("--librispeech-dir", default="/tmp/LibriSpeech/test-clean",
                        help="Path to LibriSpeech test-clean directory")
    parser.add_argument("--single-shot", action="store_true",
                        help="Feed entire audio at once instead of streaming chunks")
    parser.add_argument("--chunk-seconds", type=float, default=2.0,
                        help="Chunk size in seconds for simul-streaming (default: 2.0)")
    parser.add_argument("--border-fraction", type=float, default=0.25,
                        help="Border fraction for AlignAtt stopping (default: 0.25, matching H100 config)")
    parser.add_argument("--chapter-grouped", action="store_true",
                        help="Group utterances by speaker-chapter (matching H100 methodology)")
    parser.add_argument("--output-json", default=None,
                        help="Save per-utterance results to JSON file")
    args = parser.parse_args()

    # Check alignment heads
    heads_path = ALIGNMENT_HEADS.get(args.model_size)
    if heads_path and os.path.exists(heads_path):
        logger.info("Using alignment heads: %s", heads_path)
        with open(heads_path) as f:
            heads_data = json.load(f)
        n_heads = len(heads_data.get("alignment_heads_compact", []))
        logger.info("  Loaded %d alignment heads for border detection", n_heads)
    else:
        heads_path = None
        logger.warning("No alignment heads file found for %s! Using default heuristic.",
                        args.model_size)

    # Load model
    logger.info("Loading Qwen3-ASR-%s MLX SimulStreaming model...", args.model_size.upper())
    t_load_start = time.perf_counter()
    asr = Qwen3MLXSimulStreamingASR(
        model_size=args.model_size,
        lan="en",
        alignment_heads_path=heads_path,
        border_fraction=args.border_fraction,
    )
    t_load_end = time.perf_counter()
    logger.info("Model loaded in %.2fs", t_load_end - t_load_start)

    # Verify alignment heads
    logger.info("Alignment heads active: %d heads across %d layers",
                len(asr.alignment_heads), len(asr.heads_by_layer))
    if asr.alignment_heads:
        layers = sorted(asr.heads_by_layer.keys())
        logger.info("  Active layers: %s", layers[:10])
        logger.info("  First 5 heads: %s", asr.alignment_heads[:5])

    logger.info("Config: border_fraction=%.2f, chunk_seconds=%.1f",
                args.border_fraction, args.chunk_seconds)

    # Warmup
    logger.info("Running warmup inference...")
    dummy_audio = np.random.randn(SAMPLE_RATE * 3).astype(np.float32) * 0.01
    if args.single_shot:
        _, warmup_time = transcribe_single_shot(asr, dummy_audio)
    else:
        _, warmup_time = transcribe_simul(asr, dummy_audio, args.chunk_seconds)
    logger.info("Warmup done in %.2fs", warmup_time)

    # Determine mode
    mode = "single-shot" if args.single_shot else "simul-streaming"
    if args.chapter_grouped:
        mode += " (chapter-grouped)"

    logger.info("Starting benchmark: model=%s, mode=%s, bf=%.2f, chunk=%.1fs",
                args.model_size, mode, args.border_fraction, args.chunk_seconds)
    logger.info("LibriSpeech dir: %s", args.librispeech_dir)

    # Load data
    if args.chapter_grouped:
        samples = load_librispeech_chapters(args.librispeech_dir)
        logger.info("Loaded %d speaker-chapters", len(samples))
    else:
        samples = list(load_librispeech_utterances(
            args.librispeech_dir, args.max_utterances
        ))
        logger.info("Loaded %d utterances", len(samples))

    # Run benchmark
    references = []
    hypotheses = []
    per_sample_results = []
    total_audio_duration = 0.0
    total_inference_time = 0.0

    for i, (sample_id, audio, ref_text, duration) in enumerate(samples):
        if args.single_shot:
            hyp_text, infer_time = transcribe_single_shot(asr, audio)
        else:
            hyp_text, infer_time = transcribe_simul(asr, audio, args.chunk_seconds)

        ref_norm = normalize_text(ref_text)
        hyp_norm = normalize_text(hyp_text)

        # Per-sample WER
        if ref_norm:
            sample_wer = compute_wer(ref_norm, hyp_norm)
        else:
            sample_wer = 0.0

        total_audio_duration += duration
        total_inference_time += infer_time

        references.append(ref_norm)
        hypotheses.append(hyp_norm)

        result = {
            "id": sample_id,
            "ref": ref_text,
            "hyp": hyp_text,
            "ref_norm": ref_norm,
            "hyp_norm": hyp_norm,
            "duration_s": round(duration, 3),
            "infer_time_s": round(infer_time, 3),
            "rtf": round(infer_time / duration, 4) if duration > 0 else 0,
            "wer": round(sample_wer, 4),
        }
        per_sample_results.append(result)

        # Progress logging
        if (i + 1) % 50 == 0 or (i + 1) <= 5:
            running_wer = compute_wer(references, hypotheses)
            running_rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0
            logger.info(
                "[%d/%d] id=%s dur=%.1fs infer=%.2fs rtf=%.3f wer=%.1f%% "
                "| running: wer=%.2f%% rtf=%.3f",
                i + 1, len(samples), sample_id, duration, infer_time,
                infer_time / duration if duration > 0 else 0,
                sample_wer * 100, running_wer * 100, running_rtf,
            )

        # Show first few transcriptions
        if i < 3:
            logger.info("  REF: %s", ref_text[:120])
            logger.info("  HYP: %s", hyp_text[:120])

    # Final results
    n_samples = len(references)
    if n_samples == 0:
        logger.error("No samples processed!")
        return

    total_wer = compute_wer(references, hypotheses)
    total_cer = compute_cer(references, hypotheses)
    total_rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0

    total_ref_words = sum(len(r.split()) for r in references)
    total_hyp_words = sum(len(h.split()) for h in hypotheses)

    wers = [r["wer"] for r in per_sample_results]
    wers_sorted = sorted(wers)
    median_wer = wers_sorted[len(wers_sorted) // 2]
    p90_wer = wers_sorted[int(len(wers_sorted) * 0.9)]
    p95_wer = wers_sorted[int(len(wers_sorted) * 0.95)]
    zero_wer_count = sum(1 for w in wers if w == 0.0)

    unit = "chapters" if args.chapter_grouped else "utterances"

    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS: Qwen3-ASR-{args.model_size.upper()} MLX SimulStreaming")
    print(f"Mode: {mode}")
    print(f"Config: border_fraction={args.border_fraction}, chunk={args.chunk_seconds}s")
    print("=" * 70)
    print(f"Samples ({unit}):    {n_samples}")
    print(f"Total audio:         {total_audio_duration:.1f}s ({total_audio_duration/60:.1f}min)")
    print(f"Total inference:     {total_inference_time:.1f}s ({total_inference_time/60:.1f}min)")
    print(f"Reference words:     {total_ref_words}")
    print(f"Hypothesis words:    {total_hyp_words}")
    print("-" * 70)
    print(f"WER:                 {total_wer * 100:.2f}%")
    print(f"CER:                 {total_cer * 100:.2f}%")
    print(f"RTF:                 {total_rtf:.4f}")
    if total_rtf > 0:
        print(f"  (1/RTF = {1/total_rtf:.1f}x realtime)")
    print("-" * 70)
    print(f"Median {unit[:3]} WER:    {median_wer * 100:.2f}%")
    print(f"P90 {unit[:3]} WER:       {p90_wer * 100:.2f}%")
    print(f"P95 {unit[:3]} WER:       {p95_wer * 100:.2f}%")
    print(f"Zero-WER {unit[:3]}:      {zero_wer_count}/{n_samples} ({zero_wer_count/n_samples*100:.1f}%)")
    print("-" * 70)
    print(f"Alignment heads:     {len(asr.alignment_heads)} heads, {len(asr.heads_by_layer)} layers")
    print(f"Heads file:          {heads_path or 'NONE (default heuristic)'}")
    print(f"Model loaded in:     {t_load_end - t_load_start:.2f}s")
    print("=" * 70)

    # H100 reference comparison
    print("\nH100 PyTorch SimulStream+KV reference (chapter-grouped, bf=0.25):")
    print("  0.6B: WER 6.44%, RTF 0.109 (91 chapters, 602s)")
    print("  1.7B: WER 8.09%, RTF 0.117 (91 chapters, 602s)")

    # Worst samples
    worst = sorted(per_sample_results, key=lambda r: r["wer"], reverse=True)[:10]
    print(f"\nTop 10 worst {unit}:")
    for r in worst:
        print(f"  {r['id']}: WER={r['wer']*100:.1f}% dur={r['duration_s']:.1f}s rtf={r['rtf']:.3f}")
        if r['wer'] > 0.5:
            print(f"    REF: {r['ref_norm'][:80]}")
            print(f"    HYP: {r['hyp_norm'][:80]}")

    # Save JSON results
    if args.output_json:
        output = {
            "model": f"Qwen3-ASR-{args.model_size.upper()}",
            "backend": "mlx-simul-streaming",
            "mode": mode,
            "platform": "Apple M5 (32GB)",
            "config": {
                "border_fraction": args.border_fraction,
                "chunk_seconds": args.chunk_seconds,
                "chapter_grouped": args.chapter_grouped,
            },
            "n_samples": n_samples,
            "total_audio_s": round(total_audio_duration, 2),
            "total_inference_s": round(total_inference_time, 2),
            "wer": round(total_wer, 6),
            "cer": round(total_cer, 6),
            "rtf": round(total_rtf, 6),
            "median_wer": round(median_wer, 6),
            "p90_wer": round(p90_wer, 6),
            "p95_wer": round(p95_wer, 6),
            "alignment_heads_count": len(asr.alignment_heads),
            "alignment_heads_file": heads_path,
            "per_sample": per_sample_results,
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Results saved to %s", args.output_json)


if __name__ == "__main__":
    main()
