#!/usr/bin/env python3
"""D1: distill the block-bidirectional causal tower toward offline embeddings.

Student = Qwen3-ASR audio tower executed under the streaming mask
(bidirectional within blocks, causal across blocks, bounded left window),
trained with the parallel forward from ``qwen3_streaming.tower_distill``
(proven step-equal to the served inference). Teacher = the frozen original
tower run offline on the same audio, computed on the fly — no labels needed.

Gate: frozen-decoder streaming WER on held-out WLK chunks, evaluated
periodically; the best-gate tower state_dict is kept.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from qwen3_streaming.gate import gate_eval
from qwen3_streaming.native_realtime_model import (
    Qwen3ASRRealtimeQwenAudioCausalModel,
    _register_qwen3_asr_transformers,
)
from qwen3_streaming.realtime_config import RealtimeAudioConfig
from qwen3_streaming.tower_distill import (
    block_bidirectional_forward,
    distill_loss,
    teacher_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument(
        "--block-frames",
        type=int,
        default=96,
        help="Bidirectional block size in mel frames; must be a multiple of 8 (conv block). 96 = 0.96s.",
    )
    parser.add_argument(
        "--mix-block-frames",
        default=None,
        help="Comma list (e.g. 96,192): sample the block size per step from these. "
        "Batches are padded to the largest value; overrides --block-frames.",
    )
    parser.add_argument(
        "--mix-block-probs",
        default=None,
        help="Comma list of sampling probabilities matching --mix-block-frames (default uniform).",
    )
    parser.add_argument("--left-context-sec", type=float, default=15.0)
    parser.add_argument("--max-audio-sec", type=float, default=16.0)
    parser.add_argument("--cosine-weight", type=float, default=0.5)
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--dataset-config", default="clean")
    parser.add_argument(
        "--dataset-split",
        default="train.100",
        help="Split name, or comma-separated splits to interleave (e.g. "
        "train.clean.100,train.clean.360,train.other.500 with --dataset-config all).",
    )
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument(
        "--concat-min-sec",
        type=float,
        default=0.0,
        help="If > 0, concatenate streamed utterances into long samples whose "
        "target length is sampled log-uniformly in [min, max] seconds — trains "
        "long block chains (the long-form drift fix).",
    )
    parser.add_argument("--concat-max-sec", type=float, default=96.0)
    parser.add_argument(
        "--batch-frame-budget",
        type=int,
        default=13000,
        help="With concatenation: pack samples into a batch until this many mel "
        "frames (keeps step compute roughly constant across lengths).",
    )
    parser.add_argument(
        "--long-gate-manifest",
        type=Path,
        default=None,
        help="Optional full-file manifest for a chain-robustness gate (segmented, no reset).",
    )
    parser.add_argument("--long-gate-limit", type=int, default=3)
    parser.add_argument("--long-gate-every", type=int, default=10000)
    parser.add_argument(
        "--position-offset-max",
        type=int,
        default=0,
        help="If > 0, with probability --position-offset-prob each step trains the "
        "student at a global position offset sampled log-uniformly in [1, max] "
        "(teacher stays at 0) — position invariance for long sessions.",
    )
    parser.add_argument("--position-offset-prob", type=float, default=0.5)
    parser.add_argument(
        "--gate-position-offset",
        type=int,
        default=0,
        help="If > 0, also gate at this audio position offset (long-session probe).",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=2,
        help="Offset added to the per-epoch shuffle seed (avoid replaying a previous run's order).",
    )
    parser.add_argument("--dataset2", default=None, help="Second streaming corpus to interleave.")
    parser.add_argument("--dataset2-config", default=None)
    parser.add_argument("--dataset2-split", default="train")
    parser.add_argument(
        "--dataset2-prob",
        type=float,
        default=0.35,
        help="Sampling probability of --dataset2 in the interleave.",
    )
    parser.add_argument(
        "--lr-end-ratio",
        type=float,
        default=1.0,
        help="Cosine-decay the LR to this fraction of --lr by --steps (1.0 = constant).",
    )
    parser.add_argument(
        "--resume-tower",
        type=Path,
        default=None,
        help="Load a tower_best.pt checkpoint into the student tower before training.",
    )
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--gate-limit", type=int, default=10)
    parser.add_argument("--gate-every", type=int, default=500)
    parser.add_argument(
        "--gate-chunk-ms",
        default="960",
        help="Comma list of gate streaming chunk sizes in ms (e.g. 960,1920). "
        "A best checkpoint is kept per size, plus tower_last.pt.",
    )
    parser.add_argument("--language", required=True, help="e.g. English (gate prompts)")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_block_sizes(args) -> list[int]:
    if args.mix_block_frames:
        sizes = [int(s) for s in str(args.mix_block_frames).split(",") if s.strip()]
    else:
        sizes = [int(args.block_frames)]
    for size in sizes:
        if size % 8 != 0 or size <= 0:
            raise SystemExit(f"block size {size} must be a positive multiple of 8")
    return sizes


def parse_block_probs(args, n: int) -> list[float]:
    if not args.mix_block_probs:
        return [1.0 / n] * n
    probs = [float(s) for s in str(args.mix_block_probs).split(",") if s.strip()]
    if len(probs) != n or abs(sum(probs) - 1.0) > 1e-6:
        raise SystemExit("--mix-block-probs must match --mix-block-frames and sum to 1")
    return probs


def assert_dataset2_yields_audio(args) -> None:
    """Fail loudly if the second corpus streams nothing usable (sr filter etc.)."""
    if not args.dataset2:
        return
    import io

    import soundfile as sf

    import datasets.config as dsconfig

    dsconfig.TORCHCODEC_AVAILABLE = False
    from datasets import Audio, load_dataset

    ds = load_dataset(
        args.dataset2, args.dataset2_config, split=args.dataset2_split, streaming=True
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    usable = 0
    for idx, item in enumerate(ds):
        if idx >= 20:
            break
        try:
            audio, sr = sf.read(io.BytesIO(item["audio"]["bytes"]), dtype="float32")
        except Exception:
            continue
        if sr == 16_000 and audio.shape[0] >= 16_000:
            usable += 1
    if usable < 5:
        raise SystemExit(
            f"--dataset2 {args.dataset2} yielded {usable}/20 usable 16kHz samples; "
            "refusing to train silently on the primary corpus only"
        )
    print(json.dumps({"dataset2_probe_usable": usable}))


def build_model(args) -> tuple[Qwen3ASRRealtimeQwenAudioCausalModel, object, object]:
    _register_qwen3_asr_transformers()
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)
    hf_config = AutoConfig.from_pretrained(args.model_id)
    config = RealtimeAudioConfig(
        d_model=int(hf_config.thinker_config.text_config.hidden_size),
        qwen_audio_left_context_sec=args.left_context_sec,
        qwen_audio_block_bidirectional=True,
    )
    model = Qwen3ASRRealtimeQwenAudioCausalModel.from_qwen_pretrained(
        args.model_id,
        config=config,
        bos_token_id=(
            int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else 0
        ),
        wait_token_id=None,
        dtype=torch.float32,
        device_map="cpu",
    ).to(args.device)
    return model, tokenizer, processor


def data_batches(args, feature_extractor):
    """Stream (mels [B, T, 128], lengths) batches from the audio dataset."""
    import io

    import numpy as np
    import soundfile as sf

    import datasets.config as dsconfig

    dsconfig.TORCHCODEC_AVAILABLE = False
    from datasets import Audio, load_dataset

    pad_to = max(parse_block_sizes(args))
    cap_sec = (
        args.concat_max_sec if args.concat_min_sec > 0 else args.max_audio_sec
    )
    max_frames = int(cap_sec * 100)
    max_frames -= max_frames % pad_to

    splits = [s.strip() for s in args.dataset_split.split(",") if s.strip()]
    epoch = 0
    while True:  # loop epochs indefinitely; --steps bounds training
        from datasets import interleave_datasets

        parts = [
            load_dataset(args.dataset, args.dataset_config, split=split, streaming=True)
            for split in splits
        ]
        if len(parts) > 1:
            ds = interleave_datasets(parts, stopping_strategy="all_exhausted")
        else:
            ds = parts[0]
        if args.dataset2:
            ds2 = load_dataset(
                args.dataset2, args.dataset2_config, split=args.dataset2_split, streaming=True
            )
            ds = interleave_datasets(
                [ds.select_columns(["audio"]), ds2.select_columns(["audio"])],
                probabilities=[1.0 - args.dataset2_prob, args.dataset2_prob],
                seed=epoch + args.seed_base,
                stopping_strategy="all_exhausted",
            )
        if args.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=args.shuffle_buffer, seed=epoch + args.seed_base)
        epoch += 1
        ds = ds.cast_column("audio", Audio(decode=False))
        batch, lengths = [], []
        import math as _math
        import random as _random

        rng = _random.Random(epoch + args.seed_base)

        def draw_target_frames() -> int:
            lo, hi = args.concat_min_sec, args.concat_max_sec
            sec = _math.exp(rng.uniform(_math.log(lo), _math.log(hi)))
            frames = int(sec * 100)
            return max(8, frames - frames % 8)

        concat_parts: list = []
        concat_frames = 0
        concat_target = draw_target_frames() if args.concat_min_sec > 0 else 0

        def flush_batch():
            nonlocal batch, lengths
            if not batch:
                return None
            longest = max(lengths)
            longest += (-longest) % pad_to
            mels = np.zeros((len(batch), longest, 128), dtype=np.float32)
            for i, feat in enumerate(batch):
                mels[i, : feat.shape[0], :] = feat
            out = torch.from_numpy(mels), torch.tensor(lengths)
            batch, lengths = [], []
            return out

        def push_sample(features):
            frames = int(features.shape[0])
            if args.concat_min_sec > 0:
                budget_hit = (
                    sum(lengths) + frames > args.batch_frame_budget and batch
                )
            else:
                budget_hit = len(batch) >= args.batch_size
            out = flush_batch() if budget_hit else None
            batch.append(features)
            lengths.append(frames)
            if args.concat_min_sec <= 0 and len(batch) == args.batch_size:
                out = out or flush_batch()
            return out

        iterator = iter(ds)
        while True:
            try:
                item = next(iterator)
            except StopIteration:
                break
            except Exception as exc:  # transient hub/CDN errors: restart the epoch stream
                print(f"data stream error ({type(exc).__name__}); restarting epoch", flush=True)
                break
            audio, sr = sf.read(io.BytesIO(item["audio"]["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16_000:
                continue
            features = feature_extractor(
                audio,
                sampling_rate=sr,
                padding=True,
                truncation=False,
                return_attention_mask=True,
                return_tensors="np",
            )["input_features"][0].T  # [T, 128]
            frames = min(features.shape[0], max_frames)
            frames -= frames % 8  # conv block granularity
            if frames < 100:
                continue
            if args.concat_min_sec > 0:
                concat_parts.append(features[:frames])
                concat_frames += frames
                if concat_frames >= concat_target:
                    sample = np.concatenate(concat_parts, axis=0)
                    concat_parts, concat_frames = [], 0
                    concat_target = draw_target_frames()
                    out = push_sample(sample)
                    if out is not None:
                        yield out
            else:
                out = push_sample(features[:frames])
                if out is not None:
                    yield out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, tokenizer, processor = build_model(args)
    student_tower = model.audio_encoder.audio_tower
    teacher_tower = copy.deepcopy(student_tower).eval().requires_grad_(False)
    if args.resume_tower is not None:
        payload = torch.load(args.resume_tower, map_location="cpu", weights_only=True)
        student_tower.load_state_dict(payload["tower_state_dict"])
        print(json.dumps({
            "resumed_from": str(args.resume_tower),
            "resumed_step": payload.get("step"),
            "resumed_gate_wer": payload.get("gate_wer"),
        }))
    for module in (model.text_model, model.lm_head, model.adapter):
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)
    student_tower.train()

    trainable = [p for p in student_tower.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    import math

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        if args.lr_end_ratio >= 1.0:
            return args.lr
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        floor = args.lr * args.lr_end_ratio
        return floor + (args.lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    import random

    random.seed(args.seed_base)
    block_sizes = parse_block_sizes(args)
    block_probs = parse_block_probs(args, len(block_sizes))
    gate_chunks = [float(s) for s in str(args.gate_chunk_ms).split(",") if s.strip()]
    assert_dataset2_yields_audio(args)

    left_context_steps = model.audio_encoder.left_context_steps
    print(json.dumps({
        "trainable_params": n_trainable,
        "left_context_steps": left_context_steps,
        "block_sizes": block_sizes,
        "block_probs": block_probs,
        "gate_chunks": gate_chunks,
    }))

    def run_gates() -> dict[str, float]:
        wers: dict[str, float] = {}
        for chunk_ms in gate_chunks:
            wers[str(int(chunk_ms))] = gate_eval(
                model,
                tokenizer,
                processor,
                manifest=args.gate_manifest,
                limit=args.gate_limit,
                chunk_ms=chunk_ms,
                language=args.language,
                device=device,
            )
        if args.gate_position_offset > 0:
            wers[f"{int(gate_chunks[0])}@off{args.gate_position_offset}"] = gate_eval(
                model,
                tokenizer,
                processor,
                manifest=args.gate_manifest,
                limit=args.gate_limit,
                chunk_ms=gate_chunks[0],
                language=args.language,
                device=device,
                position_offset=args.gate_position_offset,
            )
        student_tower.train()
        return wers

    def run_long_gate() -> float:
        wer = gate_eval(
            model,
            tokenizer,
            processor,
            manifest=args.long_gate_manifest,
            limit=args.long_gate_limit,
            chunk_ms=gate_chunks[0],
            language=args.language,
            device=device,
            segment_max_cached_steps=200,
        )
        student_tower.train()
        return wer

    def save_tower(name: str, step: int, wers: dict[str, float]) -> None:
        torch.save(
            {
                "tower_state_dict": student_tower.state_dict(),
                "step": step,
                "gate_wer": min(wers.values()),
                "gate_wers": dict(wers),
                "block_frames": block_sizes,
                "left_context_sec": args.left_context_sec,
                "model_id": args.model_id,
            },
            args.output_dir / name,
        )

    gates0 = run_gates()
    best_by_chunk = dict(gates0)
    history = [{"step": 0, "gate_wers": dict(gates0)}]
    print(
        "step 0 "
        + " ".join(f"gate@{c}={w:.4f}" for c, w in gates0.items())
        + " (untrained)",
        flush=True,
    )

    batches = data_batches(args, processor.feature_extractor)
    started = time.time()
    for step in range(1, args.steps + 1):
        mels, lengths = next(batches)
        mels = mels.to(device)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)

        step_block = random.choices(block_sizes, weights=block_probs, k=1)[0]
        step_offset = 0
        if args.position_offset_max > 0 and random.random() < args.position_offset_prob:
            step_offset = int(
                math.exp(random.uniform(0.0, math.log(args.position_offset_max)))
            )
        student = block_bidirectional_forward(
            student_tower,
            mels,
            block_frames=step_block,
            left_context_steps=left_context_steps,
            lengths=lengths,
            position_offset=step_offset,
        )
        teacher = teacher_forward(teacher_tower, mels, lengths)
        lengths_steps = torch.tensor(
            [
                int(model.audio_encoder.output_steps_for_mel_frames(int(l)))
                for l in lengths
            ]
        )
        loss, stats = distill_loss(
            student, teacher, lengths_steps=lengths_steps, cosine_weight=args.cosine_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            speed = step / (time.time() - started)
            print(
                f"step {step} loss={float(loss):.5f} mse={stats['mse']:.5f} "
                f"cos_d={stats['cosine_distance']:.5f} steps/s={speed:.2f}",
                flush=True,
            )

        if step % args.gate_every == 0 or step == args.steps:
            wers = run_gates()
            if args.long_gate_manifest is not None and (
                step % args.long_gate_every == 0 or step == args.steps
            ):
                wers["long"] = run_long_gate()
            history.append(
                {"step": step, "gate_wers": dict(wers), "loss": float(loss)}
            )
            print(
                f"step {step} "
                + " ".join(f"gate@{c}={w:.4f}" for c, w in wers.items())
                + " (best "
                + " ".join(f"@{c}={w:.4f}" for c, w in best_by_chunk.items())
                + ")",
                flush=True,
            )
            for key, wer in wers.items():
                if key not in best_by_chunk:
                    best_by_chunk[key] = wer
                if wer < best_by_chunk[key]:
                    best_by_chunk[key] = wer
                    save_tower(f"tower_best_{key.replace('@', '_')}.pt", step, wers)
            save_tower("tower_last.pt", step, wers)
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

    (args.output_dir / "final_metrics.json").write_text(
        json.dumps(
            {
                "steps": args.steps,
                "gate_wers_untrained": dict(gates0),
                "gate_wers_best": dict(best_by_chunk),
                "trainable_params": n_trainable,
                "history": history,
            },
            indent=2,
        )
    )
    print(json.dumps({
        "gate_wers_untrained": dict(gates0),
        "gate_wers_best": dict(best_by_chunk),
    }))


if __name__ == "__main__":
    main()
