#!/usr/bin/env python3
"""
Convert a Hugging Face style Whisper checkpoint into a WhisperLiveKit .pt file.

Optionally shrink the supported audio chunk length (in seconds) by trimming the
encoder positional embeddings and updating the stored model dimensions.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch

from whisperlivekit.whisper import _convert_hf_state_dict
from whisperlivekit.whisper.audio import HOP_LENGTH, SAMPLE_RATE
from whisperlivekit.whisper.model import ModelDimensions
from whisperlivekit.whisper.utils import exact_div


def _load_state_dict(repo_path: Path) -> Dict[str, torch.Tensor]:
    safetensor_path = repo_path / "model.safetensors"
    bin_path = repo_path / "pytorch_model.bin"

    if safetensor_path.is_file():
        try:
            from safetensors.torch import load_file  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "Install safetensors to load model.safetensors "
                "(pip install safetensors)"
            ) from exc
        return load_file(str(safetensor_path))

    if bin_path.is_file():
        return torch.load(bin_path, map_location="cpu")

    raise FileNotFoundError(
        f"Could not find model.safetensors or pytorch_model.bin under {repo_path}"
    )


def _load_config(repo_path: Path) -> Dict:
    config_path = repo_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Hugging Face checkpoint at {repo_path} is missing config.json"
        )
    with open(config_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _derive_audio_ctx(chunk_length: float) -> Tuple[int, int]:
    n_samples = int(round(chunk_length * SAMPLE_RATE))
    expected_samples = chunk_length * SAMPLE_RATE
    if abs(n_samples - expected_samples) > 1e-6:
        raise ValueError(
            "chunk_length must align with sample rate so that "
            "chunk_length * SAMPLE_RATE is an integer"
        )
    n_frames = exact_div(n_samples, HOP_LENGTH)
    n_audio_ctx = exact_div(n_frames, 2)
    return n_frames, n_audio_ctx


def _build_dims(config: Dict, chunk_length: float) -> Dict:
    base_dims = ModelDimensions(
        n_mels=config["num_mel_bins"],
        n_audio_ctx=config["max_source_positions"],
        n_audio_state=config["d_model"],
        n_audio_head=config["encoder_attention_heads"],
        n_audio_layer=config.get("encoder_layers") or config["num_hidden_layers"],
        n_vocab=config["vocab_size"],
        n_text_ctx=config["max_target_positions"],
        n_text_state=config["d_model"],
        n_text_head=config["decoder_attention_heads"],
        n_text_layer=config["decoder_layers"],
    ).__dict__.copy()

    _, n_audio_ctx = _derive_audio_ctx(chunk_length)
    base_dims["n_audio_ctx"] = n_audio_ctx
    base_dims["chunk_length"] = chunk_length
    return base_dims


def _trim_positional_embedding(
    state_dict: Dict[str, torch.Tensor], target_ctx: int
) -> None:
    key = "encoder.positional_embedding"
    if key not in state_dict:
        raise KeyError(f"{key} missing from converted state dict")

    tensor = state_dict[key]
    if tensor.shape[0] < target_ctx:
        raise ValueError(
            f"Cannot increase encoder ctx from {tensor.shape[0]} to {target_ctx}"
        )
    if tensor.shape[0] == target_ctx:
        return
    state_dict[key] = tensor[:target_ctx].contiguous()


def convert_checkpoint(hf_path: Path, output_path: Path, chunk_length: float) -> None:
    state_dict = _load_state_dict(hf_path)
    converted = _convert_hf_state_dict(state_dict)

    config = _load_config(hf_path)
    dims = _build_dims(config, chunk_length)

    _trim_positional_embedding(converted, dims["n_audio_ctx"])

    package = {"dims": dims, "model_state_dict": converted}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face Whisper checkpoint to WhisperLiveKit format."
    )
    parser.add_argument(
        "hf_path",
        type=str,
        help="Path to the cloned Hugging Face repository (e.g. whisper-tiny.en)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="converted-whisper.pt",
        help="Destination path for the .pt file",
    )
    parser.add_argument(
        "--chunk-length",
        type=float,
        default=30.0,
        help="Audio chunk length in seconds to support (default: 30)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hf_path = Path(os.path.expanduser(args.hf_path)).resolve()
    output_path = Path(os.path.expanduser(args.output)).resolve()

    convert_checkpoint(hf_path, output_path, args.chunk_length)
    print(f"Saved converted checkpoint to {output_path}")


if __name__ == "__main__":
    main()
