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

from qwen3_streaming.cached_full_hypothesis import (
    CachedFullHypothesisConfig,
    CachedFullHypothesisStreamer,
    added_token_id,
    qwen_asr_prompt_text,
)
from qwen3_streaming.metrics import word_error_rate
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
        type=float,
        default=960.0,
        help="Gate streaming chunk; keep = block-frames * 10 for train/serve parity.",
    )
    parser.add_argument("--language", required=True, help="e.g. English (gate prompts)")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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

    max_frames = int(args.max_audio_sec * 100)
    max_frames -= max_frames % args.block_frames

    splits = [s.strip() for s in args.dataset_split.split(",") if s.strip()]
    epoch = 0
    while True:  # loop epochs indefinitely; --steps bounds training
        parts = [
            load_dataset(args.dataset, args.dataset_config, split=split, streaming=True)
            for split in splits
        ]
        if len(parts) > 1:
            from datasets import interleave_datasets

            ds = interleave_datasets(parts, stopping_strategy="all_exhausted")
        else:
            ds = parts[0]
        if args.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=args.shuffle_buffer, seed=epoch)
        epoch += 1
        ds = ds.cast_column("audio", Audio(decode=False))
        batch, lengths = [], []
        for item in ds:
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
            batch.append(features[:frames])
            lengths.append(frames)
            if len(batch) == args.batch_size:
                longest = max(lengths)
                longest += (-longest) % args.block_frames
                mels = np.zeros((len(batch), longest, 128), dtype=np.float32)
                for i, feat in enumerate(batch):
                    mels[i, : feat.shape[0], :] = feat
                yield torch.from_numpy(mels), torch.tensor(lengths)
                batch, lengths = [], []


@torch.no_grad()
def gate_eval(model, tokenizer, processor, args) -> float:
    """Frozen-decoder streaming WER on held-out WLK chunks."""
    import soundfile as sf

    model.eval()
    wait_id = added_token_id(tokenizer, "[P]")
    word_id = added_token_id(tokenizer, "[W]")
    suppress = [wait_id, word_id]
    for token in ("<|audio_start|>", "<|audio_pad|>", "<|audio_end|>", "<|im_start|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0:
            suppress.append(tid)
    prompt_template = tokenizer.encode(
        qwen_asr_prompt_text(context="", language=args.language),
        add_special_tokens=False,
    )
    config = CachedFullHypothesisConfig(
        wait_token_id=wait_id,
        word_start_token_id=word_id,
        eos_token_id=int(tokenizer.eos_token_id),
        max_new_tokens=256,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        suppress_token_ids=tuple(suppress),
        prompt_prefix_template=prompt_template,
        audio_placeholder_token_id=int(tokenizer.convert_tokens_to_ids("<|audio_pad|>")),
    )

    rows = [
        json.loads(line)
        for line in args.gate_manifest.read_text().splitlines()
        if line.strip()
    ][: args.gate_limit]
    chunk_frames = int(round(args.gate_chunk_ms / 10.0))
    wers = []
    for row in rows:
        audio, sr = sf.read(row["audio"], dtype="float32")
        features = processor.feature_extractor(
            audio,
            sampling_rate=sr,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )["input_features"][0].T.to(args.device)
        streamer = CachedFullHypothesisStreamer(model, tokenizer, config)
        for start in range(0, features.shape[0], chunk_frames):
            streamer.append_mel_chunk(
                features[start : start + chunk_frames, :].unsqueeze(0)
            )
        final = streamer.finalize(finalize_mode="latest")
        wer = word_error_rate(row.get("teacher_text") or row.get("text") or "", final.final_text)
        if wer is not None:
            wers.append(wer)
    model.train()
    return sum(wers) / len(wers) if wers else float("nan")


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

    left_context_steps = model.audio_encoder.left_context_steps
    print(json.dumps({
        "trainable_params": n_trainable,
        "left_context_steps": left_context_steps,
        "block_frames": args.block_frames,
    }))

    gate0 = gate_eval(model, tokenizer, processor, args)
    best_wer = gate0
    history = [{"step": 0, "gate_wer": gate0}]
    print(f"step 0 gate_wer={gate0:.4f} (untrained)")

    batches = data_batches(args, processor.feature_extractor)
    started = time.time()
    for step in range(1, args.steps + 1):
        mels, lengths = next(batches)
        mels = mels.to(device)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)

        student = block_bidirectional_forward(
            student_tower,
            mels,
            block_frames=args.block_frames,
            left_context_steps=left_context_steps,
            lengths=lengths,
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
            wer = gate_eval(model, tokenizer, processor, args)
            history.append({"step": step, "gate_wer": wer, "loss": float(loss)})
            print(f"step {step} gate_wer={wer:.4f} (best {best_wer:.4f})", flush=True)
            if wer < best_wer:
                best_wer = wer
                torch.save(
                    {
                        "tower_state_dict": student_tower.state_dict(),
                        "step": step,
                        "gate_wer": wer,
                        "block_frames": args.block_frames,
                        "left_context_sec": args.left_context_sec,
                        "model_id": args.model_id,
                    },
                    args.output_dir / "tower_best.pt",
                )
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

    (args.output_dir / "final_metrics.json").write_text(
        json.dumps(
            {
                "steps": args.steps,
                "gate_wer_untrained": gate0,
                "gate_wer_best": best_wer,
                "trainable_params": n_trainable,
                "history": history,
            },
            indent=2,
        )
    )
    print(json.dumps({"gate_wer_untrained": gate0, "gate_wer_best": best_wer}))


if __name__ == "__main__":
    main()
