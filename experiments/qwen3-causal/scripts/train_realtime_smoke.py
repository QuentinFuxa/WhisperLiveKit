#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from qwen3_streaming.native_realtime_model import Qwen3ASRRealtimeNativeModel
from qwen3_streaming.realtime_config import RealtimeAudioConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiny H100 smoke training for the native realtime ASR scaffold."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/realtime_smoke"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mel-frames", type=int, default=96)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)
    device = torch.device(args.device)

    config = RealtimeAudioConfig(
        d_model=128,
        audio_num_layers=2,
        audio_num_heads=4,
        audio_ffn_multiplier=2,
        conv_kernel_size=5,
        audio_window_sec=15.0,
    )
    model = Qwen3ASRRealtimeNativeModel(
        config,
        vocab_size=args.vocab_size,
        bos_token_id=1,
        decoder_num_layers=2,
        decoder_num_heads=4,
        decoder_ffn_multiplier=2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n_steps = args.mel_frames // config.frames_per_decoder_step
    target_token = 2
    losses: list[float] = []

    for step in range(args.steps):
        mels = torch.randn(args.batch_size, args.mel_frames, config.n_mels, device=device)
        labels = torch.full(
            (args.batch_size, n_steps),
            target_token,
            dtype=torch.long,
            device=device,
        )
        previous = torch.cat(
            [
                torch.full(
                    (args.batch_size, 1),
                    model.bos_token_id,
                    dtype=torch.long,
                    device=device,
                ),
                labels[:, :-1],
            ],
            dim=1,
        )

        logits = model(mels, previous)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), labels.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % 10 == 0:
            print(json.dumps({"step": step + 1, "loss": losses[-1]}))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    (args.output_dir / "smoke_metrics.json").write_text(
        json.dumps(
            {
                "steps": args.steps,
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "loss_decreased": losses[-1] < losses[0],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "loss_decreased": losses[-1] < losses[0],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
