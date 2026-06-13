#!/usr/bin/env python3
"""Two WERs for the causal streaming backend, to separate transcript quality
from live streaming behaviour.

  WER-A "final transcript": finalize('latest'). The causal encoder never sees
  future audio, but the scored transcript lets a word be revised by later
  audio WITHIN its ~12-16 s segment (segments are frozen across rollovers).
  Comparable to an offline transcript; this is the headline 18.1.

  WER-B "live, no retraction": the monotonic LocalAgreement committed stream
  (the text actually emitted to the user, holding back the unstable frontier
  and never rewriting an emitted word), plus an end-of-stream flush of the
  held-back tail. This is what a live caption shows.

The gap between A and B is exactly how much "rewriting the past" the final
score benefits from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def word_lcp_tail(committed: str, hypothesis: str) -> str:
    """Words of `hypothesis` beyond its longest common word-prefix with
    `committed` (the frontier held back live, flushed at end of stream)."""
    cw, hw = committed.split(), hypothesis.split()
    i = 0
    while i < len(cw) and i < len(hw) and cw[i] == hw[i]:
        i += 1
    return " ".join(hw[i:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-jsonl", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--tower-checkpoint", required=True)
    ap.add_argument("--language", default="en")
    ap.add_argument("--chunk-frames", type=int, default=192)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--experiments-dir", type=Path,
                    default=Path(__file__).resolve().parent)
    args = ap.parse_args()

    sys.path.insert(0, str(args.experiments_dir))
    import soundfile as sf
    from qwen3_streaming.metrics import word_error_rate

    from whisperlivekit.qwen3_streaming.asr import Qwen3StreamingASR

    asr = Qwen3StreamingASR(
        lan=args.language, model_size="Qwen/Qwen3-ASR-0.6B",
        qwen3_streaming_audio_backend="causal",
        qwen3_streaming_tower_checkpoint=args.tower_checkpoint,
        qwen3_streaming_dtype=args.dtype,
    )
    rows = [json.loads(l) for l in args.manifest_jsonl.read_text().splitlines() if l.strip()]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    a_list, b_list = [], []
    with args.output_jsonl.open("w") as out:
        for i, row in enumerate(rows):
            path = row.get("audio") or row.get("wav")
            audio, sr = sf.read(path, dtype="float32")
            feats = asr.feature_extractor(
                audio, sampling_rate=sr, padding=True, truncation=False,
                return_attention_mask=True, return_tensors="pt",
            )["input_features"][0].T.to(asr.device)

            streamer = asr.build_streamer(args.language)
            for s in range(0, feats.shape[0], args.chunk_frames):
                streamer.append_mel_chunk(feats[s:s + args.chunk_frames, :].unsqueeze(0))
            streamer.flush_pending_audio()

            # capture the live emit stream BEFORE the revising finalize
            committed_live = streamer.last_global_committed_text
            hypothesis = streamer.last_global_hypothesis_text
            streaming_text = (committed_live + " " + word_lcp_tail(committed_live, hypothesis)).strip()

            text_latest = streamer.finalize(finalize_mode="latest").final_text

            ref = row.get("teacher_text") or row.get("text") or row.get("reference") or ""
            wer_a = word_error_rate(ref, text_latest) if ref else None
            wer_b = word_error_rate(ref, streaming_text) if ref else None
            if wer_a is not None: a_list.append(wer_a)
            if wer_b is not None: b_list.append(wer_b)
            out.write(json.dumps({
                "audio": str(path), "reference": ref,
                "final_text": text_latest,              # WER-A
                "streaming_text": streaming_text,        # WER-B
                "committed_live": committed_live,
                "wer_final": wer_a, "wer_streaming": wer_b,
            }) + "\n")
            out.flush()
            print(f"[{i+1}/{len(rows)}] A={None if wer_a is None else round(wer_a,4)} "
                  f"B={None if wer_b is None else round(wer_b,4)} {Path(path).name}", flush=True)

    summary = {
        "count": len(rows),
        "wer_final_latest_mean": sum(a_list)/len(a_list) if a_list else None,
        "wer_streaming_norevise_mean": sum(b_list)/len(b_list) if b_list else None,
    }
    args.output_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
