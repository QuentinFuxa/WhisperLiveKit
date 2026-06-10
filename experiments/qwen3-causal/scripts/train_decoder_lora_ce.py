#!/usr/bin/env python3
"""D2a: decoder LoRA co-adaptation over the frozen distilled causal tower.

The tower comes from D1 (``--resume-tower``, frozen). LoRA wraps the Qwen
text decoder; teacher-forced CE on teacher transcripts is the only loss.
Because ``lora_b`` is zero-initialized, step 0 reproduces the D1 model
exactly — the step-0 gate doubles as a resume sanity check.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from qwen3_streaming.decoder_ce import build_ce_inputs, ce_forward
from qwen3_streaming.gate import gate_eval
from qwen3_streaming.lora import (
    DECODER_LORA_TARGETS,
    add_lora_to_linear_modules,
    lora_parameters,
    lora_state_dict,
)
from qwen3_streaming.native_realtime_model import (
    Qwen3ASRRealtimeQwenAudioCausalModel,
    _register_qwen3_asr_transformers,
)
from qwen3_streaming.realtime_config import RealtimeAudioConfig
from qwen3_streaming.tower_distill import block_bidirectional_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-tower", type=Path, required=True)
    parser.add_argument(
        "--train-manifests",
        required=True,
        help="Comma-separated JSONL manifests with audio + teacher_text (+language).",
    )
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr-end-ratio", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--block-frames", type=int, default=96)
    parser.add_argument("--left-context-sec", type=float, default=15.0)
    parser.add_argument("--max-audio-sec", type=float, default=16.0)
    parser.add_argument("--max-target-tokens", type=int, default=384)
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--gate-limit", type=int, default=10)
    parser.add_argument("--gate-every", type=int, default=500)
    parser.add_argument("--gate-chunk-ms", type=float, default=960.0)
    parser.add_argument("--language", required=True, help="Gate prompt language, e.g. English")
    parser.add_argument("--default-train-language", default="English")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_rows(manifests: str, *, max_audio_sec: float) -> list[dict]:
    rows: list[dict] = []
    for path in manifests.split(","):
        path = path.strip()
        if not path:
            continue
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("teacher_text") or row.get("text")
            if not text or not row.get("audio"):
                continue
            duration = float(row.get("duration_sec") or 0.0)
            if duration and duration > max_audio_sec:
                continue
            rows.append(
                {
                    "audio": row["audio"],
                    "text": str(text),
                    "language": str(row.get("language") or ""),
                }
            )
    if not rows:
        raise SystemExit("no usable rows in the train manifests")
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)

    _register_qwen3_asr_transformers()
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    import soundfile as sf

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
    ).to(device)

    payload = torch.load(args.resume_tower, map_location="cpu", weights_only=True)
    model.audio_encoder.audio_tower.load_state_dict(payload["tower_state_dict"])
    print(json.dumps({
        "resumed_tower": str(args.resume_tower),
        "tower_step": payload.get("step"),
        "tower_gate_wer": payload.get("gate_wer"),
    }))

    # Freeze everything, then add LoRA to the decoder.
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    wrapped = add_lora_to_linear_modules(
        model.text_model,
        target_names=DECODER_LORA_TARGETS,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model.to(device)
    trainable = lora_parameters(model.text_model)
    for param in trainable:
        param.requires_grad_(True)
    n_trainable = sum(p.numel() for p in trainable)
    print(json.dumps({"lora_modules": len(wrapped), "trainable_params": n_trainable}))

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        floor = args.lr * args.lr_end_ratio
        return floor + (args.lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    audio_placeholder = int(tokenizer.convert_tokens_to_ids("<|audio_pad|>"))
    rows = load_rows(args.train_manifests, max_audio_sec=args.max_audio_sec)
    print(json.dumps({"train_rows": len(rows)}))

    def run_gate() -> float:
        return gate_eval(
            model,
            tokenizer,
            processor,
            manifest=args.gate_manifest,
            limit=args.gate_limit,
            chunk_ms=args.gate_chunk_ms,
            language=args.language,
            device=device,
        )

    gate0 = run_gate()
    best_wer = gate0
    history = [{"step": 0, "gate_wer": gate0}]
    print(f"step 0 gate_wer={gate0:.4f} (LoRA no-op sanity)", flush=True)

    def sample_forward(row: dict) -> tuple[torch.Tensor, dict]:
        audio, sr = sf.read(row["audio"], dtype="float32")
        features = processor.feature_extractor(
            audio,
            sampling_rate=sr,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )["input_features"][0].T
        frames = min(features.shape[0], int(args.max_audio_sec * 100))
        frames -= frames % args.block_frames
        if frames < args.block_frames:
            frames = (features.shape[0] // 8) * 8  # short clip: single partial block
        mels = features[:frames].unsqueeze(0).to(device)
        with torch.no_grad():
            tower_out = block_bidirectional_forward(
                model.audio_encoder.audio_tower,
                mels,
                block_frames=min(args.block_frames, int(mels.shape[1])),
                left_context_steps=model.audio_encoder.left_context_steps,
            )
            frame_hidden = model.adapter._project(tower_out)
        prompt_ids, target_ids, _ = build_ce_inputs(
            tokenizer,
            audio_steps=int(frame_hidden.shape[1]),
            language=row["language"] or args.default_train_language,
            target_text=row["text"],
            audio_placeholder_token_id=audio_placeholder,
            max_target_tokens=args.max_target_tokens,
        )
        return ce_forward(
            model,
            frame_hidden,
            prompt_ids=prompt_ids,
            target_ids=target_ids,
            audio_placeholder_token_id=audio_placeholder,
        )

    order = list(range(len(rows)))
    random.shuffle(order)
    cursor = 0
    started = time.time()
    for step in range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        losses, accs = [], []
        for _ in range(args.grad_accum):
            if cursor >= len(order):
                random.shuffle(order)
                cursor = 0
            row = rows[order[cursor]]
            cursor += 1
            try:
                loss, stats = sample_forward(row)
            except Exception as exc:  # noqa: BLE001 - skip unreadable rows
                print(f"skip row ({exc})", flush=True)
                continue
            (loss / args.grad_accum).backward()
            losses.append(float(loss.detach()))
            accs.append(stats["token_accuracy"])
        if not losses:
            continue
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            speed = step / (time.time() - started)
            print(
                f"step {step} loss={sum(losses)/len(losses):.4f} "
                f"acc={sum(accs)/len(accs):.4f} steps/s={speed:.2f}",
                flush=True,
            )

        if step % args.gate_every == 0 or step == args.steps:
            wer = run_gate()
            history.append({"step": step, "gate_wer": wer})
            print(f"step {step} gate_wer={wer:.4f} (best {best_wer:.4f})", flush=True)
            if wer < best_wer:
                best_wer = wer
                torch.save(
                    {
                        "lora_state_dict": lora_state_dict(model.text_model),
                        "step": step,
                        "gate_wer": wer,
                        "lora_rank": args.lora_rank,
                        "lora_alpha": args.lora_alpha,
                        "lora_targets": list(DECODER_LORA_TARGETS),
                        "tower_checkpoint": str(args.resume_tower),
                        "model_id": args.model_id,
                    },
                    args.output_dir / "lora_best.pt",
                )
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

    (args.output_dir / "final_metrics.json").write_text(
        json.dumps(
            {
                "steps": args.steps,
                "gate_wer_start": gate0,
                "gate_wer_best": best_wer,
                "trainable_params": n_trainable,
                "history": history,
            },
            indent=2,
        )
    )
    print(json.dumps({"gate_wer_start": gate0, "gate_wer_best": best_wer}))


if __name__ == "__main__":
    main()
