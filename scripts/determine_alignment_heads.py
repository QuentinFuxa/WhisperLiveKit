"""Determine alignment heads for a variants, such as distilled model"""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import math
import pathlib
import sys
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from datasets import Audio as DatasetAudio
from datasets import load_dataset

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WHISPER_ROOT = REPO_ROOT / "whisper"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(WHISPER_ROOT))

from whisper import load_model
from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim
from whisper.tokenizer import get_tokenizer

AudioInput = Union[str, pathlib.Path, np.ndarray, torch.Tensor]


def load_dataset_clips(name, config, split, limit):
    ds = load_dataset(name, config, split=split)
    ds = ds.cast_column("audio", DatasetAudio(decode=False))
    clips = []
    for idx, row in enumerate(ds):
        if limit is not None and idx >= limit:
            break
        audio_field = row["audio"]
        transcript = row["text"]

        waveform_np, _ = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.mean(axis=1)
        waveform = waveform_np
        transcript = str(transcript)

        clips.append((waveform, transcript))
    return clips


def load_clips(args):
    return load_dataset_clips(
        args.dataset,
        args.dataset_config,
        args.dataset_split,
        args.dataset_num_samples,
    )


def _waveform_from_source(source: AudioInput) -> torch.Tensor:
    waveform = torch.from_numpy(source.astype(np.float32, copy=False))
    return waveform


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="pytorch_model.bin",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run on",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech_asr"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="clean" 
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="validation[:1%]",
    )
    parser.add_argument(
        "--dataset-num-samples",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Z score threshold for a head to be selected",
    )
    parser.add_argument(
        "--votes",
        type=float,
        default=0.75,
        help="percentage of clips that must vote for a head",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="alignment_heads.b85",
    )
    parser.add_argument(
        "--visualize-top-k",
        type=int,
        default=32,
    )
    return parser.parse_args()


def collect_heads(
    model,
    tokenizer,
    clips: Sequence[Tuple[AudioInput, str]],
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = model.device
    votes = torch.zeros(model.dims.n_text_layer, model.dims.n_text_head, device=device)
    strengths = torch.zeros_like(votes)

    for audio_source, transcript in clips:
        waveform = pad_or_trim(_waveform_from_source(audio_source))
        mel = log_mel_spectrogram(waveform, device=device)

        tokens = torch.tensor(
            [
                *tokenizer.sot_sequence,
                tokenizer.no_timestamps,
                *tokenizer.encode(transcript),
                tokenizer.eot,
            ],
            device=device,
        )

        qks = [None] * model.dims.n_text_layer
        hooks = [
            block.cross_attn.register_forward_hook(
                lambda _, __, outputs, index=i: qks.__setitem__(index, outputs[-1][0])
            )
            for i, block in enumerate(model.decoder.blocks)
        ]

        with torch.no_grad():
            model(mel.unsqueeze(0), tokens.unsqueeze(0))

        for hook in hooks:
            hook.remove()

        for layer_idx, tensor in enumerate(qks):
            if tensor is None:
                continue
            tensor = tensor[:, :, : mel.shape[-1] // 2]
            tensor = tensor.softmax(dim=-1)
            peak = tensor.max(dim=-1).values  # [heads, tokens]
            strengths[layer_idx] += peak.mean(dim=-1)
            zscore = (peak - peak.mean(dim=-1, keepdim=True)) / (
                peak.std(dim=-1, keepdim=True, unbiased=False) + 1e-6
            )
            mask = (zscore > 3).any(dim=-1)
            votes[layer_idx] += mask.float()

    votes /= len(clips)
    strengths /= len(clips)
    return votes, strengths


def _select_heads_for_visualization(selection, strengths, top_k):
    selected = torch.nonzero(selection, as_tuple=False)
    if selected.numel() == 0:
        return []

    entries = [
        (int(layer.item()), int(head.item()), float(strengths[layer, head].item()))
        for layer, head in selected
    ]
    entries.sort(key=lambda item: item[2], reverse=True)
    return entries[:top_k]

def _extract_heatmaps(
    model,
    tokenizer,
    clip: Tuple[AudioInput, str],
    heads: Sequence[Tuple[int, int, float]],
) -> dict:
    if not heads:
        return {}

    target_map = {}
    for layer, head, _ in heads:
        target_map.setdefault(layer, set()).add(head)

    waveform = pad_or_trim(_waveform_from_source(clip[0]))
    mel = log_mel_spectrogram(waveform, device=model.device)
    transcript = clip[1]
    tokens = torch.tensor(
        [
            *tokenizer.sot_sequence,
            tokenizer.no_timestamps,
            *tokenizer.encode(transcript),
            tokenizer.eot,
        ],
        device=model.device,
    )

    QKs = [None] * model.dims.n_text_layer
    hooks = [
        block.cross_attn.register_forward_hook(
            lambda _, __, outputs, index=i: QKs.__setitem__(index, outputs[-1][0])
        )
        for i, block in enumerate(model.decoder.blocks)
    ]

    with torch.no_grad():
        model(mel.unsqueeze(0), tokens.unsqueeze(0))

    for hook in hooks:
        hook.remove()

    heatmaps = {}
    for layer_idx, tensor in enumerate(QKs):
        if tensor is None or layer_idx not in target_map:
            continue
        tensor = tensor[:, :, : mel.shape[-1] // 2]
        tensor = tensor.softmax(dim=-1).cpu()
        for head_idx in target_map[layer_idx]:
            heatmaps[(layer_idx, head_idx)] = tensor[head_idx]

    return heatmaps


def _plot_heatmaps(
    heads, heatmaps, output_path):
    cols = min(3, len(heads))
    rows = math.ceil(len(heads) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)

    for idx, (layer, head, score) in enumerate(heads):
        ax = axes[idx // cols][idx % cols]
        mat = heatmaps.get((layer, head))
        if mat is None:
            ax.axis("off")
            continue
        im = ax.imshow(mat.to(torch.float32).numpy(), aspect="auto", origin="lower")
        ax.set_title(f"L{layer} H{head} · score {score:.2f}")
        ax.set_xlabel("time")
        ax.set_ylabel("tokens")

    for j in range(len(heads), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _dump_mask(mask: torch.Tensor, output_path: str):
    payload = mask.numpy().astype(np.bool_)
    blob = base64.b85encode(gzip.compress(payload.tobytes()))
    with open(output_path, "wb") as f:
        f.write(blob)


def main():
    args = _parse_args()
    model = load_model(args.model, device=args.device)
    model.eval()
    tokenizer = get_tokenizer(multilingual=model.is_multilingual)
    clips = load_clips(args)

    votes, strengths = collect_heads(model, tokenizer, clips, args.threshold)
    # selection = votes > 0.5
    selection = strengths > 0.05
    _dump_mask(selection.cpu(), args.output)

    viz_heads = _select_heads_for_visualization(selection, strengths, args.visualize_top_k)
    heatmaps = _extract_heatmaps(model, tokenizer, clips[0], viz_heads)
    _plot_heatmaps(viz_heads, heatmaps, "alignment_heads.png")

if __name__ == "__main__":
    main()
