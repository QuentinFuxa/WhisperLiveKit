#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from qwen3_streaming.audio_io import load_audio_mono
from qwen3_streaming.ctc import (
    CompactCTCVocab,
    build_compact_ctc_vocab,
    build_ctc_token_targets,
    ctc_greedy_decode,
    ctc_loss,
)
from qwen3_streaming.cached_full_hypothesis import (
    expand_audio_prompt_placeholders,
    qwen_asr_prompt_text,
)
from qwen3_streaming.metrics import (
    merge_token_repetition_stats,
    token_repetition_stats,
)
from qwen3_streaming.rnnt import (
    aligned_window_ce_loss,
    rnnt_aligned_token_margin_loss,
    rnnt_forward_backward_loss,
    rnnt_nonblank_rate_loss,
    rnnt_prefix_targets,
)
from qwen3_streaming.native_realtime_model import (
    Qwen3ASRRealtimeNativeModel,
    Qwen3ASRRealtimeQwenAudioCausalModel,
    Qwen3ASRRealtimeQwenAudioSurgeryModel,
    Qwen3ASRRealtimeQwenDecoderModel,
    configure_compact_ctc_head,
    configure_rnnt_lite_head,
    load_realtime_model,
    qwen3_asr_text_hidden_size,
)
from qwen3_streaming.realtime_config import RealtimeAudioConfig
from qwen3_streaming.realtime_features import (
    clean_decoded_text,
    decode_realtime_token_ids,
    log_mel_spectrogram,
)
from qwen3_streaming.realtime_targets import (
    ScheduledEmission,
    WordAlignment,
    build_frame_targets,
    heuristic_word_alignments,
)


SOURCE_SPECS = {
    "fleurs_en": {
        "dataset": "google/fleurs",
        "config": "en_us",
        "split": "train",
        "text_columns": ("transcription", "raw_transcription", "text"),
        "language": "English",
    },
    "fleurs_fr": {
        "dataset": "google/fleurs",
        "config": "fr_fr",
        "split": "train",
        "text_columns": ("transcription", "raw_transcription", "text"),
        "language": "French",
    },
}


@dataclass
class TinyASRExample:
    source: str
    text: str
    language: str
    duration_sec: float
    mel: torch.Tensor
    labels: torch.Tensor
    previous: torch.Tensor
    ctc_targets: torch.Tensor
    rnnt_token_frames: torch.Tensor


class TinyASRDataset(Dataset[TinyASRExample]):
    def __init__(self, examples: list[TinyASRExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TinyASRExample:
        return self.examples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny native realtime ASR checkpoint on real FLEURS audio."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/realtime_fleurs_tiny_v0"))
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help=(
            "Resume model weights from a previous realtime checkpoint directory. "
            "Optimizer state is not restored; pass the desired LR explicitly."
        ),
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--sources", nargs="+", default=["fleurs_en", "fleurs_fr"], choices=sorted(SOURCE_SPECS))
    parser.add_argument("--train-manifest-jsonl", type=Path)
    parser.add_argument("--eval-manifest-jsonl", type=Path)
    parser.add_argument("--max-train-per-source", type=int, default=48)
    parser.add_argument("--max-eval-per-source", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-acc", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-delay-sec", type=float, default=0.8)
    parser.add_argument("--wait-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--loss-mode",
        choices=["weighted", "balanced"],
        default="weighted",
        help=(
            "weighted applies per-token weights. balanced averages wait and "
            "text losses separately, then combines text + wait_loss_weight * wait."
        ),
    )
    parser.add_argument(
        "--emit-gate-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Use a separate binary emit/wait head. Text CE is applied only on "
            "text frames, then this weight scales the emit/wait loss."
        ),
    )
    parser.add_argument(
        "--emit-gate-wait-weight",
        type=float,
        default=1.0,
        help="Weight for wait frames inside the separate emit/wait gate loss.",
    )
    parser.add_argument(
        "--emit-rate-loss-weight",
        type=float,
        default=0.0,
        help=(
            "MSE regularizer between the batch mean emit probability and the "
            "true text-frame rate. Requires the emit/wait gate path."
        ),
    )
    parser.add_argument(
        "--text-label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for text-token CE only; wait frames stay unsmoothed.",
    )
    parser.add_argument(
        "--emit-threshold",
        type=float,
        default=0.5,
        help="Inference/eval threshold for the separate emit/wait gate.",
    )
    parser.add_argument(
        "--alignment-loss",
        choices=[
            "auto",
            "frame_ce",
            "emit_gate",
            "ctc",
            "compact_ctc",
            "qwen_ar_ce",
            "qwen_ar_context_distill",
            "aligned_window_ce",
            "aligned_window_sampled_ce",
            "rnnt_lite",
            "rnnt_fb",
        ],
        default="auto",
        help=(
            "Training objective. auto preserves the previous behavior: emit_gate "
            "when an emit loss is enabled, otherwise frame_ce. ctc trains the "
            "full-vocab audio-frame CTC head; compact_ctc trains a compact "
            "manifest-token CTC projection; qwen_ar_ce trains the audio path "
            "through frozen Qwen ASR teacher-forced transcript CE; "
            "qwen_ar_context_distill additionally distills a long-context "
            "Qwen audio path into a shorter-context student; "
            "aligned_window_ce trains the compact "
            "projection with word-aligned token windows and weak blank loss; "
            "aligned_window_sampled_ce uses the same target with hard-negative "
            "sampled denominators; rnnt_lite trains a compact aligned joint "
            "blank/token head with a prediction branch; rnnt_fb uses the same "
            "compact joint head with exact RNNT forward-backward loss."
        ),
    )
    parser.add_argument(
        "--qwen-ar-language",
        default="English",
        help="Language string inserted before <asr_text> for --alignment-loss qwen_ar_ce.",
    )
    parser.add_argument(
        "--qwen-ar-context",
        default="",
        help="System prompt context for --alignment-loss qwen_ar_ce.",
    )
    parser.add_argument(
        "--qwen-ar-max-target-tokens",
        type=int,
        default=384,
        help="Maximum transcript tokens, including EOS when enabled, for qwen_ar_ce.",
    )
    parser.add_argument(
        "--qwen-ar-add-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append tokenizer.eos_token_id to qwen_ar_ce transcript targets.",
    )
    parser.add_argument(
        "--qwen-ar-audio-preserve-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional MSE regularizer that keeps qwen_ar_ce frame embeddings close "
            "to a frozen audio reference."
        ),
    )
    parser.add_argument(
        "--qwen-ar-audio-preserve-reference",
        choices=["projection", "adapter"],
        default="projection",
        help=(
            "Frozen reference used by --qwen-ar-audio-preserve-loss-weight. "
            "projection uses the initial Qwen audio projector only; adapter uses "
            "the full initial adapter stack."
        ),
    )
    parser.add_argument(
        "--qwen-ar-streaming-audio",
        action="store_true",
        help=(
            "For --alignment-loss qwen_ar_ce, build audio embeddings through the "
            "same chunked Qwen audio surgery path used at inference instead of "
            "offline forward_full."
        ),
    )
    parser.add_argument(
        "--qwen-ar-streaming-chunk-frames",
        type=int,
        default=100,
        help="Mel frames per training chunk when --qwen-ar-streaming-audio is enabled.",
    )
    parser.add_argument(
        "--qwen-context-distill-teacher-left-context-sec",
        type=float,
        default=12.0,
        help="Teacher Qwen audio left context for --alignment-loss qwen_ar_context_distill.",
    )
    parser.add_argument(
        "--qwen-context-distill-student-left-context-sec",
        default="",
        help=(
            "Comma-separated student left contexts sampled per batch for "
            "qwen_ar_context_distill. Defaults to --qwen-audio-left-context-sec."
        ),
    )
    parser.add_argument("--qwen-context-distill-ce-weight", type=float, default=1.0)
    parser.add_argument("--qwen-context-distill-kl-weight", type=float, default=0.5)
    parser.add_argument("--qwen-context-distill-kl-temperature", type=float, default=1.0)
    parser.add_argument("--qwen-context-distill-frame-mse-weight", type=float, default=2.0)
    parser.add_argument("--qwen-context-distill-frame-cosine-weight", type=float, default=0.0)
    parser.add_argument("--qwen-context-distill-z-loss-weight", type=float, default=1e-4)
    parser.add_argument(
        "--qwen-context-distill-left-padding-frames",
        type=int,
        default=32,
        help=(
            "Number of zero audio embedding frames prepended before the transcript "
            "prompt for qwen_ar_context_distill."
        ),
    )
    parser.add_argument(
        "--aligned-window-frames",
        type=int,
        default=3,
        help="Frame radius around each aligned token for aligned_window_* losses.",
    )
    parser.add_argument(
        "--aligned-window-blank-loss-weight",
        type=float,
        default=0.05,
        help="Weak blank CE weight on frames outside aligned token windows.",
    )
    parser.add_argument(
        "--aligned-window-sampled-hard-negatives",
        type=int,
        default=64,
        help=(
            "Hard-negative count for aligned_window_sampled_ce. Ignored by "
            "aligned_window_ce."
        ),
    )
    parser.add_argument(
        "--aligned-window-token-weighting",
        choices=["none", "inverse_sqrt", "inverse"],
        default="none",
        help="Class-balanced target weighting for aligned_window_* losses.",
    )
    parser.add_argument(
        "--aligned-window-min-token-weight",
        type=float,
        default=0.25,
        help="Minimum nonblank class weight after normalization.",
    )
    parser.add_argument(
        "--aligned-window-max-token-weight",
        type=float,
        default=4.0,
        help="Maximum nonblank class weight after normalization.",
    )
    parser.add_argument(
        "--aligned-window-frequent-negative-count",
        type=int,
        default=0,
        help=(
            "Always include the N most frequent compact nonblank classes in "
            "aligned_window_sampled_ce denominators."
        ),
    )
    parser.add_argument(
        "--compact-ctc-max-tokens",
        type=int,
        default=8192,
        help=(
            "Maximum compact CTC classes including blank. <=0 keeps every token "
            "seen in the train/eval manifests."
        ),
    )
    parser.add_argument(
        "--compact-ctc-blank-bias",
        type=float,
        default=0.0,
        help="Initial blank logit bias for --alignment-loss compact_ctc.",
    )
    parser.add_argument(
        "--rnnt-lite-max-tokens",
        type=int,
        default=32768,
        help=(
            "Maximum RNNT-lite compact classes including blank. <=0 keeps every "
            "token seen in the train/eval manifests."
        ),
    )
    parser.add_argument(
        "--rnnt-lite-blank-bias",
        type=float,
        default=0.0,
        help="Initial blank logit bias for --alignment-loss rnnt_lite.",
    )
    parser.add_argument(
        "--rnnt-lite-pred-dim",
        type=int,
        default=0,
        help="Prediction embedding dim for rnnt_lite; 0 uses d_model.",
    )
    parser.add_argument(
        "--rnnt-lite-joint-dim",
        type=int,
        default=0,
        help="Joint network dim for rnnt_lite; 0 uses d_model.",
    )
    parser.add_argument(
        "--rnnt-fb-normalize-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize each RNNT forward-backward sample loss by T + U.",
    )
    parser.add_argument(
        "--rnnt-duration-prior-weight",
        type=float,
        default=0.0,
        help="Soft temporal prior weight for RNNT-FB label emissions.",
    )
    parser.add_argument(
        "--rnnt-duration-prior-sigma-frames",
        type=float,
        default=6.0,
        help="Frame sigma for the RNNT-FB temporal emission prior.",
    )
    parser.add_argument(
        "--rnnt-duration-prior-max-penalty",
        type=float,
        default=8.0,
        help="Maximum squared-distance penalty for the RNNT-FB temporal prior.",
    )
    parser.add_argument(
        "--rnnt-nonblank-rate-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional RNNT-FB MSE regularizer between mean nonblank probability "
            "and target_tokens / (frames + target_tokens)."
        ),
    )
    parser.add_argument(
        "--rnnt-target-blank-margin-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional RNNT-FB aligned margin regularizer forcing each target "
            "token logit to beat blank near its aligned frame."
        ),
    )
    parser.add_argument(
        "--rnnt-target-other-margin-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional RNNT-FB aligned margin regularizer forcing each target "
            "token logit to beat the best competing nonblank token near its "
            "aligned frame."
        ),
    )
    parser.add_argument(
        "--rnnt-target-margin-window-frames",
        type=int,
        default=2,
        help="Frame radius around each aligned target token for RNNT margin losses.",
    )
    parser.add_argument(
        "--rnnt-target-blank-margin",
        type=float,
        default=1.0,
        help="Desired target-token minus blank-logit margin near aligned frames.",
    )
    parser.add_argument(
        "--rnnt-target-other-margin",
        type=float,
        default=0.0,
        help="Desired target-token minus best-other-token logit margin.",
    )
    parser.add_argument(
        "--no-word-start-token",
        action="store_true",
        help="Do not insert [W] marker targets before each word.",
    )
    parser.add_argument(
        "--decoder-backend",
        choices=["native", "qwen", "qwen_audio_surgery", "qwen_audio_causal_kv"],
        default="native",
    )
    parser.add_argument("--qwen-decoder-model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--qwen-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--qwen-audio-right-context-ms", type=int, default=640)
    parser.add_argument("--qwen-audio-left-context-sec", type=float, default=2.0)
    parser.add_argument(
        "--qwen-audio-strict-causal",
        action="store_true",
        help="Use zero right context for the Qwen audio surgery backend.",
    )
    parser.add_argument(
        "--qwen-audio-adapter-hidden-dim",
        type=int,
        default=0,
        help=(
            "Hidden size for optional residual MLP blocks after the Qwen audio "
            "projector. Keep 0 with --qwen-audio-adapter-layers 0 for the "
            "identity-initialized Stage A baseline."
        ),
    )
    parser.add_argument(
        "--qwen-audio-adapter-layers",
        type=int,
        default=0,
        help="Number of residual MLP adapter blocks after the Qwen audio projector.",
    )
    parser.add_argument(
        "--qwen-audio-adapter-dropout",
        type=float,
        default=0.0,
        help="Dropout inside the optional Qwen audio residual adapter blocks.",
    )
    parser.add_argument(
        "--qwen-audio-adapter-residual-scale",
        type=float,
        default=0.1,
        help="Initial residual scale for optional Qwen audio adapter blocks.",
    )
    parser.add_argument(
        "--qwen-audio-adapter-zero-init",
        action="store_true",
        help=(
            "Initialize optional Qwen audio adapter residual blocks as exact identity "
            "by zeroing their output projection."
        ),
    )
    parser.add_argument(
        "--freeze-qwen-audio",
        action="store_true",
        help="Freeze the pretrained Qwen audio tower; projector/gate stay trainable.",
    )
    parser.add_argument(
        "--train-qwen-audio-last-n-layers",
        type=int,
        default=0,
        help="With the surgery backend, unfreeze only the last N Qwen audio layers.",
    )
    parser.add_argument(
        "--qwen-audio-lora-rank",
        type=int,
        default=0,
        help=(
            "Attach LoRA adapters to Qwen audio tower linear projections. "
            "When > 0, the base Qwen audio tower is frozen."
        ),
    )
    parser.add_argument(
        "--qwen-audio-lora-alpha",
        type=float,
        default=None,
        help="Audio LoRA alpha. Defaults to 2 * rank when --qwen-audio-lora-rank is set.",
    )
    parser.add_argument("--qwen-audio-lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--qwen-audio-lora-targets",
        default="q_proj,k_proj,v_proj,out_proj,fc1,fc2",
        help="Comma-separated Qwen audio tower linear module leaf names to wrap.",
    )
    parser.add_argument(
        "--freeze-qwen-layers",
        action="store_true",
        help="Freeze Qwen transformer layers/norm but keep embeddings and lm_head trainable.",
    )
    parser.add_argument(
        "--freeze-qwen-all",
        action="store_true",
        help="Freeze all Qwen text parameters. Mostly useful for audio-adapter diagnostics.",
    )
    parser.add_argument(
        "--train-qwen-last-n-layers",
        type=int,
        default=0,
        help=(
            "When freezing Qwen layers, keep the last N transformer layers trainable "
            "along with embeddings and lm_head."
        ),
    )
    parser.add_argument(
        "--qwen-lora-rank",
        type=int,
        default=0,
        help=(
            "Attach LoRA adapters to Qwen text decoder linear projections. "
            "When > 0, the base Qwen text model and lm_head are frozen."
        ),
    )
    parser.add_argument(
        "--qwen-lora-alpha",
        type=float,
        default=None,
        help="LoRA alpha. Defaults to 2 * rank when --qwen-lora-rank is set.",
    )
    parser.add_argument("--qwen-lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--qwen-lora-targets",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated Qwen decoder linear module leaf names to wrap.",
    )
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--audio-layers", type=int, default=3)
    parser.add_argument("--audio-heads", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--decoder-heads", type=int, default=6)
    parser.add_argument("--max-audio-sec", type=float, default=16.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def parse_csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("Expected at least one comma-separated item")
    return items


def parse_float_csv(value: str) -> tuple[float, ...]:
    items = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("Expected at least one comma-separated float")
    return items


QWEN_AR_ALIGNMENT_LOSSES = {"qwen_ar_ce", "qwen_ar_context_distill"}
QWEN_AUDIO_BACKENDS = {"qwen_audio_surgery", "qwen_audio_causal_kv"}
QWEN_DECODER_BACKENDS = {"qwen", *QWEN_AUDIO_BACKENDS}
QWEN_AUDIO_MODEL_TYPES = (
    Qwen3ASRRealtimeQwenAudioSurgeryModel,
    Qwen3ASRRealtimeQwenAudioCausalModel,
)


def resolve_alignment_loss(args: argparse.Namespace) -> str:
    if args.alignment_loss != "auto":
        return str(args.alignment_loss)
    if args.emit_gate_loss_weight > 0.0 or args.emit_rate_loss_weight > 0.0:
        return "emit_gate"
    return "frame_ce"


def load_checkpoint_config(checkpoint_dir: Path) -> RealtimeAudioConfig:
    config_path = checkpoint_dir / "realtime_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    return RealtimeAudioConfig(**json.loads(config_path.read_text(encoding="utf-8")))


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def zero_init_qwen_audio_adapter_blocks(model: torch.nn.Module) -> int:
    adapter = getattr(model, "adapter", None)
    blocks = getattr(adapter, "blocks", None)
    if blocks is None:
        return 0
    count = 0
    for block in blocks:
        mlp = getattr(block, "mlp", None)
        down = getattr(mlp, "down", None)
        if isinstance(down, torch.nn.Linear):
            torch.nn.init.zeros_(down.weight)
            if down.bias is not None:
                torch.nn.init.zeros_(down.bias)
            count += 1
    return count


def freeze_auxiliary_prediction_heads(model: torch.nn.Module) -> None:
    for name in (
        "emit_head",
        "ctc_head",
        "compact_ctc_head",
        "rnnt_lite_predictor",
        "rnnt_lite_audio_proj",
        "rnnt_lite_pred_proj",
        "rnnt_lite_norm",
        "rnnt_lite_head",
    ):
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Module):
            for param in module.parameters():
                param.requires_grad = False


def pick_text(row: dict, columns: tuple[str, ...]) -> str:
    for column in columns:
        value = row.get(column)
        if value:
            return clean_decoded_text(str(value))
    return ""


def row_audio(row: dict) -> tuple[np.ndarray, int]:
    audio = row["audio"]
    array = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio["sampling_rate"])
    if array.ndim == 2:
        array = array.mean(axis=1)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:
        array = array / peak
    return array.astype(np.float32, copy=False), sr


def load_examples(
    *,
    source_name: str,
    split: str,
    limit: int,
    tokenizer,
    wait_token_id: int,
    word_start_token_id: int,
    bos_token_id: int | None,
    config: RealtimeAudioConfig,
    target_delay_sec: float,
    max_audio_sec: float,
    include_word_start: bool,
    rng: random.Random,
) -> list[TinyASRExample]:
    spec = SOURCE_SPECS[source_name]
    dataset = load_dataset(spec["dataset"], spec["config"], split=split)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    examples: list[TinyASRExample] = []

    for idx in tqdm(indices, desc=f"{source_name}:{split}"):
        if len(examples) >= limit:
            break
        row = dataset[int(idx)]
        text = pick_text(row, spec["text_columns"])
        if not text:
            continue
        audio, sr = row_audio(row)
        duration_sec = float(len(audio)) / float(sr) if sr else 0.0
        if duration_sec <= 0.5:
            continue
        if duration_sec > max_audio_sec:
            audio = audio[: int(round(max_audio_sec * sr))]
            duration_sec = max_audio_sec

        mel = log_mel_spectrogram(audio, sr, config)
        duration_for_targets = mel.shape[0] * config.mel_hop_ms / 1000.0
        words = heuristic_word_alignments(text, duration_for_targets)
        if not words:
            continue
        targets = build_frame_targets(
            words=words,
            tokenizer=tokenizer,
            duration_sec=duration_for_targets,
            wait_token_id=wait_token_id,
            word_start_token_id=word_start_token_id,
            bos_token_id=bos_token_id,
            frame_sec=config.decoder_frame_sec,
            delay_sec=target_delay_sec,
            include_word_start=include_word_start,
        )
        max_steps = mel.shape[0] // config.frames_per_decoder_step
        if max_steps <= 0:
            continue
        ctc_target_ids = build_ctc_token_targets(
            words=words,
            tokenizer=tokenizer,
            blank_token_id=wait_token_id,
            ignored_token_ids={word_start_token_id, -100},
        )
        if not ctc_target_ids or len(ctc_target_ids) > max_steps:
            continue
        rnnt_token_frames = rnnt_token_frames_from_emissions(
            targets.emissions,
            ctc_target_ids,
            ignored_token_ids={wait_token_id, word_start_token_id, -100},
            max_frame_index=max_steps - 1,
        )
        ctc_targets = torch.tensor(ctc_target_ids, dtype=torch.long)
        rnnt_frames = torch.tensor(rnnt_token_frames, dtype=torch.long)
        labels = torch.tensor(targets.labels[:max_steps], dtype=torch.long)
        previous = torch.tensor(targets.previous_input_ids[:max_steps], dtype=torch.long)
        if labels.numel() < max_steps:
            pad = max_steps - labels.numel()
            labels = F.pad(labels, (0, pad), value=wait_token_id)
            previous = F.pad(previous, (0, pad), value=wait_token_id)

        examples.append(
            TinyASRExample(
                source=source_name,
                text=text,
                language=str(spec.get("language", "")),
                duration_sec=duration_sec,
                mel=mel,
                labels=labels,
                previous=previous,
                ctc_targets=ctc_targets,
                rnnt_token_frames=rnnt_frames,
            )
        )
    return examples


def parse_manifest_words(record: dict) -> list[WordAlignment]:
    words = []
    for item in record.get("word_alignments", []):
        text = clean_decoded_text(str(item.get("text", "")))
        if not text:
            continue
        start = float(item.get("start_sec", item.get("start_time", item.get("start", 0.0))))
        end = float(item.get("end_sec", item.get("end_time", item.get("end", 0.0))))
        if end > start:
            words.append(WordAlignment(text=text, start_sec=start, end_sec=end))
    return words


def rnnt_token_frames_from_emissions(
    emissions: list[ScheduledEmission],
    ctc_token_ids: list[int],
    *,
    ignored_token_ids: set[int] | tuple[int, ...] | list[int],
    max_frame_index: int | None = None,
) -> list[int]:
    ignored = {int(token_id) for token_id in ignored_token_ids}
    frames: list[int] = []
    ctc_pos = 0
    for emission in emissions:
        token_id = int(emission.token_id)
        if token_id in ignored:
            continue
        if ctc_pos >= len(ctc_token_ids):
            raise ValueError("RNNT token-frame mapping has extra emitted tokens")
        expected = int(ctc_token_ids[ctc_pos])
        if token_id != expected:
            raise ValueError(
                "RNNT token-frame mapping diverged from CTC target order: "
                f"got {token_id}, expected {expected} at position {ctc_pos}"
            )
        frame = max(0, int(emission.frame_index))
        if max_frame_index is not None:
            frame = min(frame, int(max_frame_index))
        frames.append(frame)
        ctc_pos += 1
    if ctc_pos != len(ctc_token_ids):
        raise ValueError(
            "RNNT token-frame mapping missed CTC targets: "
            f"mapped {ctc_pos}, expected {len(ctc_token_ids)}"
        )
    return frames


def build_rnnt_token_frame_targets(
    *,
    words: list[WordAlignment],
    tokenizer,
    duration_sec: float,
    wait_token_id: int,
    word_start_token_id: int,
    bos_token_id: int | None,
    frame_sec: float,
    delay_sec: float,
    include_word_start: bool,
    max_frame_index: int | None = None,
) -> tuple[list[int], list[int]]:
    targets = build_frame_targets(
        words=words,
        tokenizer=tokenizer,
        duration_sec=duration_sec,
        wait_token_id=wait_token_id,
        word_start_token_id=word_start_token_id,
        bos_token_id=bos_token_id,
        frame_sec=frame_sec,
        delay_sec=delay_sec,
        include_word_start=include_word_start,
    )
    ctc_token_ids = build_ctc_token_targets(
        words=words,
        tokenizer=tokenizer,
        blank_token_id=wait_token_id,
        ignored_token_ids={word_start_token_id, -100},
    )
    token_frames = rnnt_token_frames_from_emissions(
        targets.emissions,
        ctc_token_ids,
        ignored_token_ids={wait_token_id, word_start_token_id, -100},
        max_frame_index=max_frame_index,
    )
    return ctc_token_ids, token_frames


def load_manifest_examples(
    *,
    manifest_jsonl: Path,
    tokenizer,
    wait_token_id: int,
    word_start_token_id: int,
    bos_token_id: int | None,
    config: RealtimeAudioConfig,
    target_delay_sec: float,
    max_audio_sec: float,
    include_word_start: bool,
    require_word_alignments: bool = False,
) -> list[TinyASRExample]:
    examples: list[TinyASRExample] = []
    with manifest_jsonl.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    for record in tqdm(records, desc=f"{manifest_jsonl.name}:load"):
        text = clean_decoded_text(str(record.get("text", "")))
        if not text:
            continue
        words = parse_manifest_words(record)
        if not words:
            if require_word_alignments:
                raise ValueError(
                    f"Missing word_alignments for {record.get('audio', '<unknown>')}"
                )
            continue
        audio, sr = load_audio_mono(record["audio"], target_sr=config.sample_rate)
        duration_sec = float(len(audio)) / float(sr) if sr else 0.0
        if duration_sec <= 0.5:
            continue
        if max_audio_sec > 0.0 and duration_sec > max_audio_sec:
            audio = audio[: int(round(max_audio_sec * sr))]
            duration_sec = max_audio_sec
            words = [
                WordAlignment(
                    text=word.text,
                    start_sec=word.start_sec,
                    end_sec=min(word.end_sec, duration_sec),
                )
                for word in words
                if word.start_sec < duration_sec
            ]
        if not words:
            continue

        mel = log_mel_spectrogram(audio, sr, config)
        duration_for_targets = mel.shape[0] * config.mel_hop_ms / 1000.0
        targets = build_frame_targets(
            words=words,
            tokenizer=tokenizer,
            duration_sec=duration_for_targets,
            wait_token_id=wait_token_id,
            word_start_token_id=word_start_token_id,
            bos_token_id=bos_token_id,
            frame_sec=config.decoder_frame_sec,
            delay_sec=target_delay_sec,
            include_word_start=include_word_start,
        )
        max_steps = mel.shape[0] // config.frames_per_decoder_step
        if max_steps <= 0:
            continue
        ctc_target_ids = build_ctc_token_targets(
            words=words,
            tokenizer=tokenizer,
            blank_token_id=wait_token_id,
            ignored_token_ids={word_start_token_id, -100},
        )
        if not ctc_target_ids or len(ctc_target_ids) > max_steps:
            continue
        rnnt_token_frames = rnnt_token_frames_from_emissions(
            targets.emissions,
            ctc_target_ids,
            ignored_token_ids={wait_token_id, word_start_token_id, -100},
            max_frame_index=max_steps - 1,
        )
        ctc_targets = torch.tensor(ctc_target_ids, dtype=torch.long)
        rnnt_frames = torch.tensor(rnnt_token_frames, dtype=torch.long)
        labels = torch.tensor(targets.labels[:max_steps], dtype=torch.long)
        previous = torch.tensor(targets.previous_input_ids[:max_steps], dtype=torch.long)
        if labels.numel() < max_steps:
            pad = max_steps - labels.numel()
            labels = F.pad(labels, (0, pad), value=wait_token_id)
            previous = F.pad(previous, (0, pad), value=wait_token_id)
        examples.append(
            TinyASRExample(
                source=str(record.get("source", "")),
                text=text,
                language=str(record.get("language", "")),
                duration_sec=duration_sec,
                mel=mel,
                labels=labels,
                previous=previous,
                ctc_targets=ctc_targets,
                rnnt_token_frames=rnnt_frames,
            )
        )
    return examples


def compact_frame_label_tensors(
    labels: torch.Tensor,
    vocab: CompactCTCVocab,
    *,
    wait_token_id: int,
    word_start_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mapping = vocab.token_to_index
    blank = int(vocab.blank_index)
    compact_labels: list[int] = []
    previous_labels: list[int] = []
    last_nonblank = blank
    for raw_label in labels.tolist():
        label = int(raw_label)
        previous_labels.append(last_nonblank)
        if label == -100:
            compact_labels.append(-100)
            continue
        if label in {int(wait_token_id), int(word_start_token_id)}:
            compact_labels.append(blank)
            continue
        if label not in mapping:
            raise KeyError(f"RNNT-lite compact vocab missing frame token id: {label}")
        compact_label = int(mapping[label])
        compact_labels.append(compact_label)
        if compact_label != blank:
            last_nonblank = compact_label
    return (
        torch.tensor(compact_labels, dtype=torch.long),
        torch.tensor(previous_labels, dtype=torch.long),
    )


def collate_batch(
    batch: list[TinyASRExample],
    wait_token_id: int,
    *,
    word_start_token_id: int,
    compact_ctc_vocab: CompactCTCVocab | None = None,
) -> dict[str, torch.Tensor | list[str]]:
    max_mel = max(example.mel.shape[0] for example in batch)
    max_steps = max(example.labels.shape[0] for example in batch)
    max_ctc_steps = max(example.ctc_targets.shape[0] for example in batch)
    n_mels = batch[0].mel.shape[1]

    mels = torch.zeros(len(batch), max_mel, n_mels, dtype=torch.float32)
    labels = torch.full((len(batch), max_steps), -100, dtype=torch.long)
    previous = torch.full((len(batch), max_steps), wait_token_id, dtype=torch.long)
    ctc_targets = torch.full((len(batch), max_ctc_steps), wait_token_id, dtype=torch.long)
    rnnt_token_frames = torch.full((len(batch), max_ctc_steps), -1, dtype=torch.long)
    compact_ctc_targets = torch.full((len(batch), max_ctc_steps), 0, dtype=torch.long)
    compact_frame_labels = torch.full((len(batch), max_steps), -100, dtype=torch.long)
    compact_previous_labels = torch.zeros((len(batch), max_steps), dtype=torch.long)
    decoder_lengths = torch.zeros(len(batch), dtype=torch.long)
    mel_lengths = torch.zeros(len(batch), dtype=torch.long)
    ctc_target_lengths = torch.zeros(len(batch), dtype=torch.long)
    texts: list[str] = []
    languages: list[str] = []
    sources: list[str] = []
    durations: list[float] = []

    for idx, example in enumerate(batch):
        mels[idx, : example.mel.shape[0], :] = example.mel
        mel_lengths[idx] = example.mel.shape[0]
        labels[idx, : example.labels.shape[0]] = example.labels
        previous[idx, : example.previous.shape[0]] = example.previous
        ctc_targets[idx, : example.ctc_targets.shape[0]] = example.ctc_targets
        if example.rnnt_token_frames.shape[0] != example.ctc_targets.shape[0]:
            raise ValueError("rnnt_token_frames length must match ctc_targets length")
        rnnt_token_frames[idx, : example.rnnt_token_frames.shape[0]] = (
            example.rnnt_token_frames
        )
        if compact_ctc_vocab is not None:
            compact_ids = torch.tensor(
                compact_ctc_vocab.encode(example.ctc_targets.tolist()),
                dtype=torch.long,
            )
            compact_ctc_targets[idx, : compact_ids.shape[0]] = compact_ids
            frame_labels, previous_labels = compact_frame_label_tensors(
                example.labels,
                compact_ctc_vocab,
                wait_token_id=wait_token_id,
                word_start_token_id=word_start_token_id,
            )
            compact_frame_labels[idx, : frame_labels.shape[0]] = frame_labels
            compact_previous_labels[idx, : previous_labels.shape[0]] = previous_labels
        decoder_lengths[idx] = example.labels.shape[0]
        ctc_target_lengths[idx] = example.ctc_targets.shape[0]
        texts.append(example.text)
        languages.append(example.language)
        sources.append(example.source)
        durations.append(example.duration_sec)

    return {
        "mels": mels,
        "labels": labels,
        "previous": previous,
        "ctc_targets": ctc_targets,
        "rnnt_token_frames": rnnt_token_frames,
        "compact_ctc_targets": compact_ctc_targets,
        "compact_frame_labels": compact_frame_labels,
        "compact_previous_labels": compact_previous_labels,
        "decoder_lengths": decoder_lengths,
        "mel_lengths": mel_lengths,
        "ctc_target_lengths": ctc_target_lengths,
        "texts": texts,
        "languages": languages,
        "sources": sources,
        "durations": torch.tensor(durations, dtype=torch.float32),
    }


def frame_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_size: int,
    wait_token_id: int,
    wait_loss_weight: float,
    loss_mode: str,
    text_label_smoothing: float = 0.0,
) -> torch.Tensor:
    flat_labels = labels.reshape(-1)
    token_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        flat_labels,
        ignore_index=-100,
        reduction="none",
    )
    valid = flat_labels != -100
    if text_label_smoothing > 0.0:
        smooth_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            flat_labels,
            ignore_index=-100,
            reduction="none",
            label_smoothing=float(text_label_smoothing),
        )
        text = (flat_labels != wait_token_id) & valid
        token_loss = torch.where(text, smooth_loss, token_loss)
    if loss_mode == "balanced":
        wait = (flat_labels == wait_token_id) & valid
        text = (flat_labels != wait_token_id) & valid
        pieces = []
        if text.any():
            pieces.append(token_loss[text].mean())
        if wait.any() and wait_loss_weight > 0.0:
            pieces.append(wait_loss_weight * token_loss[wait].mean())
        if not pieces:
            return token_loss.new_zeros(())
        return sum(pieces)
    if loss_mode != "weighted":
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    weights = torch.ones_like(token_loss)
    weights = torch.where(
        flat_labels == wait_token_id,
        torch.full_like(weights, wait_loss_weight),
        weights,
    )
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    return (token_loss * weights).sum() / weights.sum().clamp_min(1.0)


def emit_gate_cross_entropy(
    logits: torch.Tensor,
    emit_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_size: int,
    wait_token_id: int,
    gate_loss_weight: float,
    gate_wait_weight: float,
    emit_rate_loss_weight: float = 0.0,
    text_label_smoothing: float = 0.0,
) -> torch.Tensor:
    token_labels = labels.clone()
    token_labels[token_labels == wait_token_id] = -100
    flat_token_labels = token_labels.reshape(-1)
    token_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        flat_token_labels,
        ignore_index=-100,
        reduction="none",
        label_smoothing=float(text_label_smoothing),
    )
    text = flat_token_labels != -100
    if text.any():
        loss = token_loss[text].mean()
    else:
        loss = token_loss.new_zeros(())

    flat_labels = labels.reshape(-1)
    valid = flat_labels != -100
    emit = (flat_labels != wait_token_id) & valid
    wait = (flat_labels == wait_token_id) & valid
    gate_targets = emit.to(dtype=emit_logits.dtype)
    gate_loss = F.binary_cross_entropy_with_logits(
        emit_logits.reshape(-1),
        gate_targets,
        reduction="none",
    )
    gate_pieces = []
    if emit.any():
        gate_pieces.append(gate_loss[emit].mean())
    if wait.any() and gate_wait_weight > 0.0:
        gate_pieces.append(gate_wait_weight * gate_loss[wait].mean())
    if gate_pieces and gate_loss_weight > 0.0:
        loss = loss + gate_loss_weight * sum(gate_pieces)
    if valid.any() and emit_rate_loss_weight > 0.0:
        pred_emit_rate = emit_logits.reshape(-1)[valid].sigmoid().mean()
        target_emit_rate = emit.to(dtype=pred_emit_rate.dtype).mean()
        loss = loss + emit_rate_loss_weight * (pred_emit_rate - target_emit_rate).pow(2)
    return loss


def label_stats(labels: torch.Tensor, wait_token_id: int) -> dict[str, int | float]:
    valid = labels != -100
    valid_count = int(valid.sum().item())
    wait_count = int(((labels == wait_token_id) & valid).sum().item())
    text_count = valid_count - wait_count
    return {
        "valid_labels": valid_count,
        "wait_labels": wait_count,
        "text_or_word_labels": text_count,
        "wait_ratio": float(wait_count / valid_count) if valid_count else 0.0,
    }


def prediction_stats(
    pred_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    wait_token_id: int,
    word_start_token_id: int,
) -> dict[str, int | float]:
    valid = labels != -100
    valid_count = int(valid.sum().item())
    pred_wait = int(((pred_ids == wait_token_id) & valid).sum().item())
    pred_word = int(((pred_ids == word_start_token_id) & valid).sum().item())
    pred_text = int(
        (
            (pred_ids != wait_token_id)
            & (pred_ids != word_start_token_id)
            & valid
        ).sum().item()
    )
    label_wait = int(((labels == wait_token_id) & valid).sum().item())
    label_word = int(((labels == word_start_token_id) & valid).sum().item())
    label_text = int(
        (
            (labels != wait_token_id)
            & (labels != word_start_token_id)
            & valid
        ).sum().item()
    )
    return {
        "valid": valid_count,
        "pred_wait": pred_wait,
        "pred_word": pred_word,
        "pred_text": pred_text,
        "pred_wait_ratio": float(pred_wait / valid_count) if valid_count else 0.0,
        "pred_text_ratio": float(pred_text / valid_count) if valid_count else 0.0,
        "label_wait": label_wait,
        "label_word": label_word,
        "label_text": label_text,
        "label_wait_ratio": float(label_wait / valid_count) if valid_count else 0.0,
        "label_text_ratio": float(label_text / valid_count) if valid_count else 0.0,
    }


def merge_prediction_stats(stats: list[dict[str, int | float]]) -> dict[str, int | float]:
    merged = {
        "valid": 0,
        "pred_wait": 0,
        "pred_word": 0,
        "pred_text": 0,
        "label_wait": 0,
        "label_word": 0,
        "label_text": 0,
    }
    for item in stats:
        for key in merged:
            merged[key] += int(item[key])
    valid = int(merged["valid"])
    merged["pred_wait_ratio"] = float(int(merged["pred_wait"]) / valid) if valid else 0.0
    merged["pred_text_ratio"] = float(int(merged["pred_text"]) / valid) if valid else 0.0
    merged["label_wait_ratio"] = float(int(merged["label_wait"]) / valid) if valid else 0.0
    merged["label_text_ratio"] = float(int(merged["label_text"]) / valid) if valid else 0.0
    for key in (
        "rnnt_nonblank_rate_loss",
        "rnnt_pred_nonblank_rate",
        "rnnt_target_nonblank_rate",
        "rnnt_target_blank_margin_loss",
        "rnnt_target_other_margin_loss",
        "rnnt_target_blank_margin",
        "rnnt_target_other_margin",
        "aligned_window_loss",
        "aligned_window_token_loss",
        "aligned_window_blank_loss",
        "aligned_window_target_blank_margin",
        "aligned_window_target_other_margin",
        "qwen_ar_token_accuracy",
        "qwen_ar_target_tokens",
    ):
        values = [float(item[key]) for item in stats if key in item]
        if values:
            merged[key] = float(np.mean(values))
    return merged


def ctc_label_stats(
    examples: list[TinyASRExample],
    *,
    wait_token_id: int,
) -> dict[str, int | float]:
    target_tokens = sum(int(example.ctc_targets.numel()) for example in examples)
    frame_steps = sum(int(example.labels.numel()) for example in examples)
    return {
        "valid": frame_steps,
        "target_tokens": target_tokens,
        "label_wait": max(0, frame_steps - target_tokens),
        "label_text": target_tokens,
        "label_wait_ratio": float(max(0, frame_steps - target_tokens) / frame_steps)
        if frame_steps
        else 0.0,
        "label_text_ratio": float(target_tokens / frame_steps) if frame_steps else 0.0,
    }


def filter_examples_for_compact_ctc_vocab(
    examples: list[TinyASRExample],
    vocab: CompactCTCVocab,
) -> list[TinyASRExample]:
    token_to_index = vocab.token_to_index
    return [
        example
        for example in examples
        if all(int(token_id) in token_to_index for token_id in example.ctc_targets.tolist())
    ]


def compact_ctc_vocab_stats(vocab: CompactCTCVocab) -> dict[str, int]:
    return {
        "size": len(vocab.token_ids),
        "blank_index": int(vocab.blank_index),
        "blank_token_id": int(vocab.token_ids[vocab.blank_index]),
    }


def compact_token_counts(
    examples: list[TinyASRExample],
    vocab: CompactCTCVocab,
) -> Counter[int]:
    mapping = vocab.token_to_index
    counts: Counter[int] = Counter()
    for example in examples:
        for token_id in example.ctc_targets.tolist():
            counts[int(mapping[int(token_id)])] += 1
    return counts


def compact_frequent_negative_indices(
    counts: Counter[int],
    *,
    blank_index: int,
    limit: int,
) -> list[int]:
    if limit <= 0:
        return []
    return [
        int(index)
        for index, _ in counts.most_common()
        if int(index) != int(blank_index)
    ][: int(limit)]


def compact_token_class_weights(
    counts: Counter[int],
    *,
    vocab_size: int,
    blank_index: int,
    mode: str,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode not in {"inverse_sqrt", "inverse"}:
        raise ValueError(f"Unknown aligned-window token weighting: {mode}")
    if min_weight < 0.0:
        raise ValueError("min_weight must be >= 0")
    if max_weight <= 0.0:
        raise ValueError("max_weight must be > 0")
    if min_weight > max_weight:
        raise ValueError("min_weight must be <= max_weight")

    count_tensor = torch.zeros(int(vocab_size), dtype=torch.float32)
    for index, count in counts.items():
        if 0 <= int(index) < int(vocab_size):
            count_tensor[int(index)] = float(count)
    valid = count_tensor > 0
    valid[int(blank_index)] = False
    weights = torch.ones(int(vocab_size), dtype=torch.float32)
    if not bool(valid.any()):
        return weights
    mean_count = count_tensor[valid].mean().clamp_min(1.0)
    ratio = mean_count / count_tensor[valid].clamp_min(1.0)
    raw = ratio.sqrt() if mode == "inverse_sqrt" else ratio
    raw = raw.clamp(min=float(min_weight), max=float(max_weight))
    occurrence_mean = (
        raw * count_tensor[valid]
    ).sum() / count_tensor[valid].sum().clamp_min(1.0)
    raw = (raw / occurrence_mean.clamp_min(1e-6)).clamp(
        min=float(min_weight),
        max=float(max_weight),
    )
    weights[valid] = raw
    weights[int(blank_index)] = 1.0
    return weights


def compact_weight_stats(
    weights: torch.Tensor | None,
    counts: Counter[int],
    vocab: CompactCTCVocab,
) -> dict[str, object] | None:
    if weights is None:
        return None
    nonblank = [
        int(index)
        for index, count in counts.items()
        if int(index) != int(vocab.blank_index) and int(count) > 0
    ]
    if not nonblank:
        return {
            "min": 1.0,
            "max": 1.0,
            "mean": 1.0,
            "top_frequent": [],
        }
    selected = weights[torch.tensor(nonblank, dtype=torch.long)]
    top_frequent = []
    for index, count in counts.most_common(10):
        index = int(index)
        if index == int(vocab.blank_index):
            continue
        top_frequent.append(
            {
                "compact_index": index,
                "token_id": int(vocab.token_ids[index]),
                "count": int(count),
                "weight": float(weights[index].item()),
            }
        )
        if len(top_frequent) >= 5:
            break
    return {
        "min": float(selected.min().item()),
        "max": float(selected.max().item()),
        "mean": float(selected.mean().item()),
        "top_frequent": top_frequent,
    }


def update_top1_counter(
    counter: Counter[int],
    raw_pred_ids: torch.Tensor,
    input_lengths: torch.Tensor,
) -> None:
    for idx in range(raw_pred_ids.shape[0]):
        length = int(input_lengths[idx].detach().cpu().item())
        counter.update(int(token_id) for token_id in raw_pred_ids[idx, :length].detach().cpu().tolist())


def top1_counter_stats(
    counter: Counter[int],
    vocab: CompactCTCVocab,
    *,
    top_k: int = 10,
) -> dict[str, object]:
    total = sum(counter.values())
    if total <= 0:
        return {
            "top1_total": 0,
            "top1_unique": 0,
            "top1_entropy": 0.0,
            "top1_top": [],
        }
    probs = np.asarray([count / total for count in counter.values()], dtype=np.float64)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    top = []
    for index, count in counter.most_common(top_k):
        index = int(index)
        token_id = (
            int(vocab.token_ids[index])
            if 0 <= index < len(vocab.token_ids)
            else None
        )
        top.append(
            {
                "compact_index": index,
                "token_id": token_id,
                "count": int(count),
                "ratio": float(count / total),
            }
        )
    return {
        "top1_total": int(total),
        "top1_unique": int(len(counter)),
        "top1_entropy": entropy,
        "top1_top": top,
    }


def ctc_prediction_stats(
    raw_pred_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    wait_token_id: int,
    word_start_token_id: int,
) -> tuple[dict[str, int | float], list[dict[str, int | float]], list[list[int]]]:
    valid_count = int(input_lengths.sum().item())
    pred_wait = 0
    pred_text = 0
    label_text = int(target_lengths.sum().item())
    repetition_stats: list[dict[str, int | float]] = []
    collapsed_ids: list[list[int]] = []
    for idx in range(raw_pred_ids.shape[0]):
        length = int(input_lengths[idx].item())
        ids = raw_pred_ids[idx, :length].detach().cpu().tolist()
        decoded = ctc_greedy_decode(
            ids,
            blank_token_id=wait_token_id,
            ignored_token_ids={word_start_token_id, -100},
        )
        pred_wait += int(decoded.blank_count)
        pred_text += int(decoded.raw_text_token_count)
        collapsed_ids.append(decoded.token_ids)
        repetition_stats.append(
            token_repetition_stats(
                decoded.token_ids,
                ignored_token_ids={wait_token_id, word_start_token_id, -100},
            )
        )
    stats = {
        "valid": valid_count,
        "pred_wait": pred_wait,
        "pred_word": 0,
        "pred_text": pred_text,
        "pred_wait_ratio": float(pred_wait / valid_count) if valid_count else 0.0,
        "pred_text_ratio": float(pred_text / valid_count) if valid_count else 0.0,
        "label_wait": max(0, valid_count - label_text),
        "label_word": 0,
        "label_text": label_text,
        "label_wait_ratio": float(max(0, valid_count - label_text) / valid_count)
        if valid_count
        else 0.0,
        "label_text_ratio": float(label_text / valid_count) if valid_count else 0.0,
    }
    return stats, repetition_stats, collapsed_ids


def compact_ctc_prediction_stats(
    raw_pred_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    vocab: CompactCTCVocab,
    *,
    wait_token_id: int,
    word_start_token_id: int,
) -> tuple[dict[str, int | float], list[dict[str, int | float]], list[list[int]]]:
    valid_count = int(input_lengths.sum().item())
    pred_wait = 0
    pred_text = 0
    label_text = int(target_lengths.sum().item())
    repetition_stats: list[dict[str, int | float]] = []
    collapsed_ids: list[list[int]] = []
    for idx in range(raw_pred_ids.shape[0]):
        length = int(input_lengths[idx].item())
        ids = raw_pred_ids[idx, :length].detach().cpu().tolist()
        decoded = ctc_greedy_decode(
            ids,
            blank_token_id=vocab.blank_index,
            ignored_token_ids=set(),
        )
        pred_wait += int(decoded.blank_count)
        pred_text += int(decoded.raw_text_token_count)
        full_token_ids = vocab.decode(decoded.token_ids)
        collapsed_ids.append(full_token_ids)
        repetition_stats.append(
            token_repetition_stats(
                full_token_ids,
                ignored_token_ids={wait_token_id, word_start_token_id, -100},
            )
        )
    stats = {
        "valid": valid_count,
        "pred_wait": pred_wait,
        "pred_word": 0,
        "pred_text": pred_text,
        "pred_wait_ratio": float(pred_wait / valid_count) if valid_count else 0.0,
        "pred_text_ratio": float(pred_text / valid_count) if valid_count else 0.0,
        "label_wait": max(0, valid_count - label_text),
        "label_word": 0,
        "label_text": label_text,
        "label_wait_ratio": float(max(0, valid_count - label_text) / valid_count)
        if valid_count
        else 0.0,
        "label_text_ratio": float(label_text / valid_count) if valid_count else 0.0,
    }
    return stats, repetition_stats, collapsed_ids


def compact_frame_prediction_stats(
    raw_pred_ids: torch.Tensor,
    labels: torch.Tensor,
    vocab: CompactCTCVocab,
    *,
    wait_token_id: int,
    word_start_token_id: int,
) -> tuple[dict[str, int | float], list[dict[str, int | float]], list[list[int]]]:
    valid = labels != -100
    valid_count = int(valid.sum().item())
    pred_wait = int(((raw_pred_ids == vocab.blank_index) & valid).sum().item())
    pred_text = int(((raw_pred_ids != vocab.blank_index) & valid).sum().item())
    label_wait = int(((labels == vocab.blank_index) & valid).sum().item())
    label_text = int(((labels != vocab.blank_index) & valid).sum().item())
    repetition_stats: list[dict[str, int | float]] = []
    decoded_ids: list[list[int]] = []
    for idx in range(raw_pred_ids.shape[0]):
        length = int(valid[idx].sum().item())
        compact_ids = [
            int(token_id)
            for token_id, is_valid in zip(
                raw_pred_ids[idx].detach().cpu().tolist(),
                valid[idx].detach().cpu().tolist(),
                strict=False,
            )
            if is_valid and int(token_id) != int(vocab.blank_index)
        ]
        full_token_ids = vocab.decode(compact_ids)
        decoded_ids.append(full_token_ids)
        repetition_stats.append(
            token_repetition_stats(
                full_token_ids,
                ignored_token_ids={wait_token_id, word_start_token_id, -100},
            )
        )
    stats = {
        "valid": valid_count,
        "pred_wait": pred_wait,
        "pred_word": 0,
        "pred_text": pred_text,
        "pred_wait_ratio": float(pred_wait / valid_count) if valid_count else 0.0,
        "pred_text_ratio": float(pred_text / valid_count) if valid_count else 0.0,
        "label_wait": label_wait,
        "label_word": 0,
        "label_text": label_text,
        "label_wait_ratio": float(label_wait / valid_count) if valid_count else 0.0,
        "label_text_ratio": float(label_text / valid_count) if valid_count else 0.0,
    }
    return stats, repetition_stats, decoded_ids


def forward_for_training(
    model: Qwen3ASRRealtimeNativeModel,
    mels: torch.Tensor,
    previous: torch.Tensor,
    *,
    use_emit_gate: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if use_emit_gate:
        if not hasattr(model, "forward_hidden") or not hasattr(model, "emit_head"):
            raise ValueError("--emit-gate-loss-weight requires a model with emit_head")
        hidden = model.forward_hidden(mels, previous)
        logits = model.lm_head(hidden)
        emit_logits = model.emit_head(
            hidden.to(dtype=model.emit_head.weight.dtype)
        ).squeeze(-1)
        return logits, emit_logits
    return model(mels, previous), None


def forward_ctc_for_training(
    model: Qwen3ASRRealtimeNativeModel,
    mels: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(model, "forward_ctc_logits"):
        raise ValueError("--alignment-loss ctc requires a model with forward_ctc_logits")
    return model.forward_ctc_logits(mels)


def forward_compact_ctc_for_training(
    model: Qwen3ASRRealtimeNativeModel,
    mels: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(model, "forward_compact_ctc_logits"):
        raise ValueError(
            "--alignment-loss compact_ctc requires a model with forward_compact_ctc_logits"
        )
    return model.forward_compact_ctc_logits(mels)


def forward_rnnt_lite_for_training(
    model: Qwen3ASRRealtimeNativeModel,
    mels: torch.Tensor,
    previous_compact_ids: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(model, "forward_rnnt_lite_logits"):
        raise ValueError(
            "--alignment-loss rnnt_lite requires a model with forward_rnnt_lite_logits"
        )
    return model.forward_rnnt_lite_logits(mels, previous_compact_ids)


def forward_rnnt_fb_for_training(
    model: Qwen3ASRRealtimeNativeModel,
    mels: torch.Tensor,
    compact_targets: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    blank_index: int,
) -> torch.Tensor:
    if not hasattr(model, "forward_rnnt_lite_joint_logits"):
        raise ValueError(
            "--alignment-loss rnnt_fb requires a model with "
            "forward_rnnt_lite_joint_logits"
        )
    previous_compact_ids = rnnt_prefix_targets(
        compact_targets,
        target_lengths,
        blank_index=blank_index,
    )
    return model.forward_rnnt_lite_joint_logits(mels, previous_compact_ids)


def build_qwen_audio_preserve_reference(
    model: Qwen3ASRRealtimeNativeModel,
    *,
    mode: str,
) -> torch.nn.Module:
    if not hasattr(model, "adapter"):
        raise ValueError("qwen_ar audio preservation requires a model adapter")
    adapter = model.adapter
    if mode == "projection":
        proj = getattr(adapter, "proj", None)
        if not isinstance(proj, torch.nn.Linear):
            raise ValueError("projection preservation requires adapter.proj")
        reference: torch.nn.Module = copy.deepcopy(proj)
    elif mode == "adapter":
        reference = copy.deepcopy(adapter)
    else:
        raise ValueError(f"Unknown audio preservation reference: {mode}")
    reference.eval()
    for param in reference.parameters():
        param.requires_grad = False
    return reference


def qwen_audio_preserve_reference_frames(
    reference: torch.nn.Module,
    audio_hidden: torch.Tensor,
) -> torch.Tensor:
    if hasattr(reference, "forward_full"):
        return reference.forward_full(audio_hidden)
    if isinstance(reference, torch.nn.Linear):
        out = reference(audio_hidden.to(dtype=reference.weight.dtype))
        return out.to(device=audio_hidden.device)
    return reference(audio_hidden)


def qwen_audio_preserve_loss(
    frame_hidden: torch.Tensor,
    reference_hidden: torch.Tensor,
    decoder_lengths: torch.Tensor,
) -> torch.Tensor:
    steps = min(int(frame_hidden.shape[1]), int(reference_hidden.shape[1]))
    if steps <= 0:
        return frame_hidden.new_zeros(())
    frame_hidden = frame_hidden[:, :steps, :]
    reference_hidden = reference_hidden[:, :steps, :].to(
        device=frame_hidden.device,
        dtype=frame_hidden.dtype,
    )
    lengths = decoder_lengths.to(device=frame_hidden.device, dtype=torch.long).clamp(
        min=0,
        max=steps,
    )
    mask = torch.arange(steps, device=frame_hidden.device)[None, :] < lengths[:, None]
    if not bool(mask.any()):
        return frame_hidden.new_zeros(())
    per_frame = (frame_hidden - reference_hidden).pow(2).mean(dim=-1)
    return per_frame[mask].mean()


def qwen_ar_streaming_audio_frames_for_training(
    model: Qwen3ASRRealtimeNativeModel,
    mels: torch.Tensor,
    mel_lengths: torch.Tensor,
    *,
    chunk_frames: int,
) -> torch.Tensor:
    if chunk_frames <= 0:
        raise ValueError("--qwen-ar-streaming-chunk-frames must be > 0")
    if not hasattr(model, "audio_encoder") or not hasattr(model, "adapter"):
        raise ValueError("qwen_ar streaming audio requires audio_encoder and adapter")
    if not hasattr(model.audio_encoder, "forward_chunk"):
        raise ValueError("qwen_ar streaming audio requires an audio encoder with forward_chunk")
    if not hasattr(model.adapter, "forward_chunk"):
        raise ValueError("qwen_ar streaming audio requires an adapter with forward_chunk")

    lengths = mel_lengths.to(device=mels.device, dtype=torch.long).clamp(
        min=0,
        max=int(mels.shape[1]),
    )
    sample_outputs: list[torch.Tensor] = []
    max_steps = 0
    hidden_dim = int(getattr(model.config, "d_model", 0))

    for idx in range(int(mels.shape[0])):
        mel_len = int(lengths[idx].detach().cpu().item())
        if mel_len <= 0:
            sample_hidden = mels.new_zeros(1, 0, hidden_dim)
            sample_outputs.append(sample_hidden)
            continue

        audio_state = model.audio_encoder.init_state()
        adapter_state = model.adapter.init_state()
        chunks: list[torch.Tensor] = []
        sample = mels[idx : idx + 1, :mel_len, :]

        def append_chunk(chunk: torch.Tensor) -> None:
            nonlocal audio_state, adapter_state
            audio_delta, audio_state = model.audio_encoder.forward_chunk(chunk, audio_state)
            if int(audio_delta.shape[1]) <= 0:
                return
            frame_delta, adapter_state = model.adapter.forward_chunk(audio_delta, adapter_state)
            if int(frame_delta.shape[1]) > 0:
                chunks.append(frame_delta)

        for start in range(0, mel_len, int(chunk_frames)):
            append_chunk(sample[:, start : start + int(chunk_frames), :])

        right_context_frames = int(getattr(model.audio_encoder, "right_context_frames", 0))
        if right_context_frames > 0:
            append_chunk(sample.new_zeros(1, right_context_frames, sample.shape[-1]))
        flush_pending = getattr(model.audio_encoder, "flush_pending", None)
        if callable(flush_pending):
            audio_delta, audio_state = flush_pending(audio_state)
            if int(audio_delta.shape[1]) > 0:
                frame_delta, adapter_state = model.adapter.forward_chunk(
                    audio_delta,
                    adapter_state,
                )
                if int(frame_delta.shape[1]) > 0:
                    chunks.append(frame_delta)

        if chunks:
            sample_hidden = torch.cat(chunks, dim=1)
            hidden_dim = int(sample_hidden.shape[-1])
        else:
            sample_hidden = sample.new_zeros(1, 0, hidden_dim)
        sample_outputs.append(sample_hidden)
        max_steps = max(max_steps, int(sample_hidden.shape[1]))

    if not sample_outputs:
        return mels.new_zeros(0, 0, hidden_dim)
    output = sample_outputs[0].new_zeros(len(sample_outputs), max_steps, hidden_dim)
    for idx, sample_hidden in enumerate(sample_outputs):
        steps = int(sample_hidden.shape[1])
        if steps > 0:
            output[idx, :steps, :] = sample_hidden[0]
    return output


def set_qwen_audio_left_context_sec(
    model: Qwen3ASRRealtimeNativeModel,
    left_context_sec: float,
) -> int:
    if left_context_sec <= 0.0:
        raise ValueError("Qwen audio left context must be > 0")
    audio_encoder = getattr(model, "audio_encoder", None)
    if audio_encoder is None or not hasattr(audio_encoder, "left_context_frames"):
        raise ValueError("Qwen context distillation requires a Qwen audio backend")
    config = getattr(model, "config", None)
    hop_ms = float(getattr(config, "mel_hop_ms", 10.0))
    left_frames = int(round(float(left_context_sec) * 1000.0 / hop_ms))
    if left_frames <= 0:
        raise ValueError("Qwen audio left context resolves to zero frames")
    audio_encoder.left_context_frames = int(left_frames)
    if config is not None and hasattr(config, "qwen_audio_left_context_sec"):
        object.__setattr__(
            config,
            "qwen_audio_left_context_sec",
            float(left_context_sec),
        )
    return int(left_frames)


def prepend_qwen_ar_audio_padding(
    frame_hidden: torch.Tensor,
    decoder_lengths: torch.Tensor,
    *,
    padding_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if padding_frames <= 0:
        return frame_hidden, decoder_lengths
    padding = frame_hidden.new_zeros(
        frame_hidden.shape[0],
        int(padding_frames),
        frame_hidden.shape[-1],
    )
    padded = torch.cat([padding, frame_hidden], dim=1)
    lengths = decoder_lengths.to(device=frame_hidden.device, dtype=torch.long) + int(
        padding_frames
    )
    return padded, lengths


def qwen_ar_audio_frames_for_training(
    model: Qwen3ASRRealtimeNativeModel,
    mels: torch.Tensor,
    decoder_lengths: torch.Tensor,
    *,
    mel_lengths: torch.Tensor | None = None,
    audio_preserve_reference: torch.nn.Module | None = None,
    streaming_audio: bool = False,
    streaming_chunk_frames: int = 100,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    preserve_loss: torch.Tensor | None = None
    if streaming_audio:
        if mel_lengths is None:
            raise ValueError("qwen_ar streaming audio requires mel_lengths")
        frame_hidden = qwen_ar_streaming_audio_frames_for_training(
            model,
            mels,
            mel_lengths,
            chunk_frames=int(streaming_chunk_frames),
        )
    elif audio_preserve_reference is None:
        frame_hidden = model.forward_audio_frames(mels)
    else:
        if not hasattr(model, "audio_encoder") or not hasattr(model, "adapter"):
            raise ValueError("qwen_ar audio preservation requires audio_encoder and adapter")
        audio_hidden = model.audio_encoder.forward_full(mels)
        frame_hidden = model.adapter.forward_full(audio_hidden)
    if audio_preserve_reference is not None:
        if not hasattr(model, "audio_encoder"):
            raise ValueError("qwen_ar audio preservation requires audio_encoder")
        with torch.no_grad():
            reference_audio_hidden = model.audio_encoder.forward_full(mels)
            reference_hidden = qwen_audio_preserve_reference_frames(
                audio_preserve_reference,
                reference_audio_hidden.detach(),
            )
        preserve_loss = qwen_audio_preserve_loss(frame_hidden, reference_hidden, decoder_lengths)
    return frame_hidden, preserve_loss


def qwen_ar_logits_from_frame_hidden(
    model: Qwen3ASRRealtimeNativeModel,
    frame_hidden: torch.Tensor,
    target_token_ids: torch.Tensor,
    decoder_lengths: torch.Tensor,
    *,
    prompt_template_token_ids: list[int] | None = None,
    prompt_template_token_ids_by_sample: list[list[int]] | None = None,
    audio_placeholder_token_id: int,
) -> torch.Tensor:
    if not hasattr(model, "forward_qwen_ar_logits_from_cached_audio"):
        raise ValueError(
            "--alignment-loss qwen_ar_ce requires a Qwen audio backend"
        )
    logits: list[torch.Tensor] = []
    for idx in range(frame_hidden.shape[0]):
        if prompt_template_token_ids_by_sample is not None:
            template_token_ids = prompt_template_token_ids_by_sample[idx]
        elif prompt_template_token_ids is not None:
            template_token_ids = prompt_template_token_ids
        else:
            raise ValueError("qwen_ar_ce missing prompt template token ids")
        audio_steps = min(
            int(decoder_lengths[idx].detach().cpu().item()),
            int(frame_hidden.shape[1]),
        )
        if audio_steps <= 0:
            raise ValueError("qwen_ar_ce received an example with zero audio steps")
        prefix_token_ids = expand_audio_prompt_placeholders(
            template_token_ids,
            audio_placeholder_token_id=int(audio_placeholder_token_id),
            audio_steps=audio_steps,
        )
        logits.append(
            model.forward_qwen_ar_logits_from_cached_audio(
                frame_hidden[idx : idx + 1, :audio_steps, :],
                prefix_token_ids=prefix_token_ids,
                audio_placeholder_token_id=int(audio_placeholder_token_id),
                target_token_ids=target_token_ids[idx : idx + 1],
            )
        )
    return torch.cat(logits, dim=0)


def build_qwen_ar_target_batch(
    texts: list[str],
    tokenizer,
    *,
    eos_token_id: int | None,
    max_target_tokens: int,
    add_eos: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_target_tokens <= 0:
        raise ValueError("--qwen-ar-max-target-tokens must be > 0")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0
    rows: list[list[int]] = []
    for text in texts:
        token_ids = [
            int(token_id)
            for token_id in tokenizer.encode(
                clean_decoded_text(text),
                add_special_tokens=False,
            )
        ]
        if add_eos and eos_token_id is not None:
            token_ids.append(int(eos_token_id))
        token_ids = token_ids[: int(max_target_tokens)]
        if not token_ids:
            continue_token = int(eos_token_id) if eos_token_id is not None else int(pad_token_id)
            token_ids = [continue_token]
        rows.append(token_ids)

    max_len = max(len(row) for row in rows)
    input_ids = torch.full(
        (len(rows), max_len),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    labels = torch.full(
        (len(rows), max_len),
        -100,
        dtype=torch.long,
        device=device,
    )
    for idx, token_ids in enumerate(rows):
        row = torch.tensor(token_ids, dtype=torch.long, device=device)
        input_ids[idx, : row.numel()] = row
        labels[idx, : row.numel()] = row
    return input_ids, labels


def build_qwen_ar_prompt_token_ids(
    languages: list[str],
    tokenizer,
    *,
    context: str,
    default_language: str,
    audio_placeholder_token_id: int,
) -> list[list[int]]:
    prompts: list[list[int]] = []
    for language in languages:
        prompt_language = str(language or default_language or "")
        token_ids = [
            int(token_id)
            for token_id in tokenizer.encode(
                qwen_asr_prompt_text(context=context, language=prompt_language),
                add_special_tokens=False,
            )
        ]
        if int(audio_placeholder_token_id) not in token_ids:
            raise ValueError("Qwen AR prompt does not contain <|audio_pad|>")
        prompts.append(token_ids)
    return prompts


def qwen_ar_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    flat_labels = labels.reshape(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        flat_labels,
        ignore_index=-100,
    )
    valid = flat_labels != -100
    if bool(valid.any()):
        pred = logits.reshape(-1, logits.shape[-1]).argmax(dim=-1)
        accuracy = (pred[valid] == flat_labels[valid]).to(dtype=torch.float32).mean()
        target_tokens = int(valid.sum().item())
    else:
        accuracy = logits.new_zeros(())
        target_tokens = 0
    return loss, {
        "qwen_ar_token_accuracy": float(accuracy.detach().cpu()),
        "qwen_ar_target_tokens": float(target_tokens),
    }


def qwen_ar_kl_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("KL distillation temperature must be > 0")
    steps = min(
        int(student_logits.shape[1]),
        int(teacher_logits.shape[1]),
        int(labels.shape[1]),
    )
    if steps <= 0:
        return student_logits.new_zeros(())
    valid = labels[:, :steps] != -100
    if not bool(valid.any()):
        return student_logits.new_zeros(())
    temp = float(temperature)
    student = student_logits[:, :steps, :].to(dtype=torch.float32) / temp
    teacher = teacher_logits[:, :steps, :].to(
        device=student_logits.device,
        dtype=torch.float32,
    ) / temp
    per_token = F.kl_div(
        F.log_softmax(student, dim=-1),
        F.softmax(teacher, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return per_token[valid].mean().to(dtype=student_logits.dtype) * (temp * temp)


def qwen_ar_frame_distill_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    decoder_lengths: torch.Tensor,
    *,
    cosine_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    steps = min(int(student_hidden.shape[1]), int(teacher_hidden.shape[1]))
    if steps <= 0:
        zero = student_hidden.new_zeros(())
        return zero, {
            "qwen_context_frame_mse": 0.0,
            "qwen_context_frame_cosine_loss": 0.0,
        }
    student = student_hidden[:, :steps, :].to(dtype=torch.float32)
    teacher = teacher_hidden[:, :steps, :].to(
        device=student_hidden.device,
        dtype=torch.float32,
    )
    lengths = decoder_lengths.to(device=student_hidden.device, dtype=torch.long).clamp(
        min=0,
        max=steps,
    )
    mask = torch.arange(steps, device=student_hidden.device)[None, :] < lengths[:, None]
    if not bool(mask.any()):
        zero = student_hidden.new_zeros(())
        return zero, {
            "qwen_context_frame_mse": 0.0,
            "qwen_context_frame_cosine_loss": 0.0,
        }
    mse = (student - teacher).pow(2).mean(dim=-1)[mask].mean()
    cosine_loss = 1.0 - F.cosine_similarity(student, teacher, dim=-1)
    cosine = cosine_loss[mask].mean()
    loss = mse.to(dtype=student_hidden.dtype)
    if cosine_weight > 0.0:
        loss = loss + float(cosine_weight) * cosine.to(dtype=student_hidden.dtype)
    return loss, {
        "qwen_context_frame_mse": float(mse.detach().cpu()),
        "qwen_context_frame_cosine_loss": float(cosine.detach().cpu()),
    }


def qwen_ar_z_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    steps = min(int(logits.shape[1]), int(labels.shape[1]))
    if steps <= 0:
        return logits.new_zeros(())
    valid = labels[:, :steps] != -100
    if not bool(valid.any()):
        return logits.new_zeros(())
    z = torch.logsumexp(logits[:, :steps, :].to(dtype=torch.float32), dim=-1)
    return z[valid].pow(2).mean().to(dtype=logits.dtype)


def forward_qwen_ar_for_training(
    model: Qwen3ASRRealtimeNativeModel,
    mels: torch.Tensor,
    target_token_ids: torch.Tensor,
    decoder_lengths: torch.Tensor,
    *,
    mel_lengths: torch.Tensor | None = None,
    prompt_template_token_ids: list[int] | None = None,
    prompt_template_token_ids_by_sample: list[list[int]] | None = None,
    audio_placeholder_token_id: int,
    audio_preserve_reference: torch.nn.Module | None = None,
    streaming_audio: bool = False,
    streaming_chunk_frames: int = 100,
    left_padding_frames: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not hasattr(model, "forward_audio_frames") or not hasattr(
        model,
        "forward_qwen_ar_logits_from_cached_audio",
    ):
        raise ValueError(
            "--alignment-loss qwen_ar_ce requires a Qwen audio backend"
        )
    frame_hidden, preserve_loss = qwen_ar_audio_frames_for_training(
        model,
        mels,
        decoder_lengths,
        mel_lengths=mel_lengths,
        audio_preserve_reference=audio_preserve_reference,
        streaming_audio=streaming_audio,
        streaming_chunk_frames=streaming_chunk_frames,
    )
    frame_hidden, effective_lengths = prepend_qwen_ar_audio_padding(
        frame_hidden,
        decoder_lengths,
        padding_frames=int(left_padding_frames),
    )
    logits = qwen_ar_logits_from_frame_hidden(
        model,
        frame_hidden,
        target_token_ids,
        effective_lengths,
        prompt_template_token_ids=prompt_template_token_ids,
        prompt_template_token_ids_by_sample=prompt_template_token_ids_by_sample,
        audio_placeholder_token_id=int(audio_placeholder_token_id),
    )
    return logits, preserve_loss


def qwen_ar_prediction_stats(
    pred_ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[dict[str, int | float], list[dict[str, int | float]], list[list[int]]]:
    valid = labels != -100
    valid_count = int(valid.sum().item())
    token_matches = int(((pred_ids == labels) & valid).sum().item())
    decoded_ids: list[list[int]] = []
    repetition_stats: list[dict[str, int | float]] = []
    for idx in range(pred_ids.shape[0]):
        ids = [
            int(token_id)
            for token_id, is_valid in zip(
                pred_ids[idx].detach().cpu().tolist(),
                valid[idx].detach().cpu().tolist(),
                strict=False,
            )
            if is_valid
        ]
        decoded_ids.append(ids)
        repetition_stats.append(token_repetition_stats(ids, ignored_token_ids={-100}))
    stats = {
        "valid": valid_count,
        "pred_wait": 0,
        "pred_word": 0,
        "pred_text": valid_count,
        "pred_wait_ratio": 0.0,
        "pred_text_ratio": 1.0 if valid_count else 0.0,
        "label_wait": 0,
        "label_word": 0,
        "label_text": valid_count,
        "label_wait_ratio": 0.0,
        "label_text_ratio": 1.0 if valid_count else 0.0,
        "qwen_ar_token_accuracy": (
            float(token_matches / valid_count) if valid_count else 0.0
        ),
    }
    return stats, repetition_stats, decoded_ids


def greedy_predictions(
    logits: torch.Tensor,
    emit_logits: torch.Tensor | None,
    *,
    wait_token_id: int,
    emit_threshold: float,
) -> torch.Tensor:
    text_ids = logits.argmax(dim=-1)
    if emit_logits is None:
        return text_ids
    emit = emit_logits.sigmoid() >= emit_threshold
    wait_ids = torch.full_like(text_ids, wait_token_id)
    return torch.where(emit, text_ids, wait_ids)


def evaluate(
    model: Qwen3ASRRealtimeNativeModel,
    loader: DataLoader,
    tokenizer,
    wait_token_id: int,
    word_start_token_id: int,
    device: torch.device,
    vocab_size: int,
    wait_loss_weight: float,
    loss_mode: str,
    emit_gate_loss_weight: float,
    emit_gate_wait_weight: float,
    emit_rate_loss_weight: float,
    text_label_smoothing: float,
    emit_threshold: float,
    alignment_loss: str,
    compact_ctc_vocab: CompactCTCVocab | None = None,
    rnnt_fb_normalize_by_length: bool = True,
    rnnt_duration_prior_weight: float = 0.0,
    rnnt_duration_prior_sigma_frames: float = 6.0,
    rnnt_duration_prior_max_penalty: float = 8.0,
    rnnt_nonblank_rate_loss_weight: float = 0.0,
    rnnt_target_blank_margin_loss_weight: float = 0.0,
    rnnt_target_other_margin_loss_weight: float = 0.0,
    rnnt_target_margin_window_frames: int = 2,
    rnnt_target_blank_margin: float = 1.0,
    rnnt_target_other_margin: float = 0.0,
    aligned_window_frames: int = 3,
    aligned_window_blank_loss_weight: float = 0.05,
    aligned_window_sampled_hard_negatives: int = 64,
    aligned_window_token_class_weights: torch.Tensor | None = None,
    aligned_window_frequent_negative_indices: list[int] | None = None,
    qwen_ar_prompt_token_ids: list[int] | None = None,
    qwen_ar_audio_placeholder_token_id: int | None = None,
    qwen_ar_max_target_tokens: int = 384,
    qwen_ar_add_eos: bool = True,
    qwen_ar_context: str = "",
    qwen_ar_default_language: str = "",
    qwen_ar_streaming_audio: bool = False,
    qwen_ar_streaming_chunk_frames: int = 100,
    qwen_ar_left_padding_frames: int = 0,
) -> tuple[float, list[dict[str, object]], dict[str, int | float]]:
    model.eval()
    losses: list[float] = []
    predictions: list[dict[str, object]] = []
    pred_stats: list[dict[str, int | float]] = []
    repetition_stats: list[dict[str, int | float]] = []
    top1_counter: Counter[int] = Counter()
    with torch.no_grad():
        for batch in loader:
            mels = batch["mels"].to(device)
            previous = batch["previous"].to(device)
            labels = batch["labels"].to(device)
            if alignment_loss in QWEN_AR_ALIGNMENT_LOSSES:
                if qwen_ar_prompt_token_ids is None:
                    raise ValueError(f"{alignment_loss} evaluation missing Qwen prompt token ids")
                if qwen_ar_audio_placeholder_token_id is None:
                    raise ValueError(
                        f"{alignment_loss} evaluation missing Qwen audio placeholder token id"
                    )
                target_token_ids, target_labels = build_qwen_ar_target_batch(
                    batch["texts"],
                    tokenizer,
                    eos_token_id=tokenizer.eos_token_id,
                    max_target_tokens=qwen_ar_max_target_tokens,
                    add_eos=qwen_ar_add_eos,
                    device=device,
                )
                batch_prompt_token_ids = build_qwen_ar_prompt_token_ids(
                    batch["languages"],
                    tokenizer,
                    context=qwen_ar_context,
                    default_language=qwen_ar_default_language,
                    audio_placeholder_token_id=qwen_ar_audio_placeholder_token_id,
                )
                if qwen_ar_prompt_token_ids is not None:
                    batch_prompt_token_ids = [
                        prompt_ids if prompt_ids else qwen_ar_prompt_token_ids
                        for prompt_ids in batch_prompt_token_ids
                    ]
                logits, _ = forward_qwen_ar_for_training(
                    model,
                    mels,
                    target_token_ids,
                    batch["decoder_lengths"].to(device),
                    mel_lengths=batch["mel_lengths"].to(device),
                    prompt_template_token_ids_by_sample=batch_prompt_token_ids,
                    audio_placeholder_token_id=qwen_ar_audio_placeholder_token_id,
                    streaming_audio=qwen_ar_streaming_audio,
                    streaming_chunk_frames=qwen_ar_streaming_chunk_frames,
                    left_padding_frames=(
                        qwen_ar_left_padding_frames
                        if alignment_loss == "qwen_ar_context_distill"
                        else 0
                    ),
                )
                loss, qwen_stats = qwen_ar_cross_entropy(logits, target_labels)
                losses.append(float(loss.detach().cpu()))
                pred_ids = logits.argmax(dim=-1)
                stats, batch_repetitions, decoded_ids = qwen_ar_prediction_stats(
                    pred_ids,
                    target_labels,
                )
                stats = dict(stats)
                stats.update(qwen_stats)
                pred_stats.append(stats)
                repetition_stats.extend(batch_repetitions)
                for idx, ids in enumerate(decoded_ids):
                    hypothesis = clean_decoded_text(
                        tokenizer.decode(ids, skip_special_tokens=True)
                    )
                    predictions.append(
                        {
                            "source": batch["sources"][idx],
                            "reference": batch["texts"][idx],
                            "hypothesis": hypothesis,
                            "hypothesis_char_length": len(hypothesis),
                            "hypothesis_word_count": len(hypothesis.split()),
                            "repetition": batch_repetitions[idx],
                            "decode_mode": alignment_loss,
                            "teacher_forced_argmax": True,
                        }
                    )
                continue
            if alignment_loss in {"aligned_window_ce", "aligned_window_sampled_ce"}:
                if compact_ctc_vocab is None:
                    raise ValueError(f"{alignment_loss} evaluation requires compact vocab")
                logits = forward_compact_ctc_for_training(model, mels)
                input_lengths = torch.minimum(
                    batch["decoder_lengths"].to(device),
                    torch.full_like(batch["decoder_lengths"].to(device), logits.shape[1]),
                )
                target_lengths = batch["ctc_target_lengths"].to(device)
                targets = batch["compact_ctc_targets"].to(device)
                label_frame_targets = batch["rnnt_token_frames"].to(device)
                sampled_negatives = (
                    aligned_window_sampled_hard_negatives
                    if alignment_loss == "aligned_window_sampled_ce"
                    else 0
                )
                (
                    loss,
                    token_loss,
                    blank_loss,
                    target_blank_margin,
                    target_other_margin,
                ) = aligned_window_ce_loss(
                    logits,
                    targets,
                    input_lengths,
                    target_lengths,
                    label_frame_targets,
                    blank_index=compact_ctc_vocab.blank_index,
                    window_frames=aligned_window_frames,
                    blank_loss_weight=aligned_window_blank_loss_weight,
                    sampled_negative_count=sampled_negatives,
                    token_class_weights=aligned_window_token_class_weights,
                    frequent_negative_indices=aligned_window_frequent_negative_indices,
                )
                losses.append(float(loss.detach().cpu()))
                raw_pred_ids = logits.argmax(dim=-1)
                update_top1_counter(top1_counter, raw_pred_ids, input_lengths)
                stats, batch_repetitions, collapsed_ids = compact_ctc_prediction_stats(
                    raw_pred_ids,
                    input_lengths,
                    target_lengths,
                    compact_ctc_vocab,
                    wait_token_id=wait_token_id,
                    word_start_token_id=word_start_token_id,
                )
                stats = dict(stats)
                stats.update(
                    {
                        "aligned_window_loss": float(loss.detach().cpu()),
                        "aligned_window_token_loss": float(token_loss.detach().cpu()),
                        "aligned_window_blank_loss": float(blank_loss.detach().cpu()),
                        "aligned_window_target_blank_margin": float(
                            target_blank_margin.detach().cpu()
                        ),
                        "aligned_window_target_other_margin": float(
                            target_other_margin.detach().cpu()
                        ),
                    }
                )
                pred_stats.append(stats)
                repetition_stats.extend(batch_repetitions)
                for idx, ids in enumerate(collapsed_ids):
                    hypothesis = decode_realtime_token_ids(
                        tokenizer,
                        ids,
                        wait_token_id=wait_token_id,
                        word_start_token_id=word_start_token_id,
                    )
                    predictions.append(
                        {
                            "source": batch["sources"][idx],
                            "reference": batch["texts"][idx],
                            "hypothesis": hypothesis,
                            "hypothesis_char_length": len(hypothesis),
                            "hypothesis_word_count": len(hypothesis.split()),
                            "repetition": batch_repetitions[idx],
                            "decode_mode": alignment_loss,
                        }
                    )
                continue
            if alignment_loss == "rnnt_fb":
                if compact_ctc_vocab is None:
                    raise ValueError("rnnt_fb evaluation requires compact vocab")
                targets = batch["compact_ctc_targets"].to(device)
                target_lengths = batch["ctc_target_lengths"].to(device)
                label_frame_targets = batch["rnnt_token_frames"].to(device)
                logits = forward_rnnt_fb_for_training(
                    model,
                    mels,
                    targets,
                    target_lengths,
                    blank_index=compact_ctc_vocab.blank_index,
                )
                input_lengths = torch.minimum(
                    batch["decoder_lengths"].to(device),
                    torch.full_like(batch["decoder_lengths"].to(device), logits.shape[1]),
                )
                loss = rnnt_forward_backward_loss(
                    logits,
                    targets,
                    input_lengths,
                    target_lengths,
                    blank_index=compact_ctc_vocab.blank_index,
                    label_frame_targets=label_frame_targets,
                    duration_prior_weight=rnnt_duration_prior_weight,
                    duration_prior_sigma_frames=rnnt_duration_prior_sigma_frames,
                    duration_prior_max_penalty=rnnt_duration_prior_max_penalty,
                    normalize_by_length=rnnt_fb_normalize_by_length,
                )
                rate_loss, pred_rate, target_rate = rnnt_nonblank_rate_loss(
                    logits,
                    input_lengths,
                    target_lengths,
                    blank_index=compact_ctc_vocab.blank_index,
                )
                if rnnt_nonblank_rate_loss_weight > 0.0:
                    loss = loss + float(rnnt_nonblank_rate_loss_weight) * rate_loss
                (
                    target_blank_loss,
                    target_other_loss,
                    target_blank_margin,
                    target_other_margin,
                ) = rnnt_aligned_token_margin_loss(
                    logits,
                    targets,
                    input_lengths,
                    target_lengths,
                    label_frame_targets,
                    blank_index=compact_ctc_vocab.blank_index,
                    window_frames=rnnt_target_margin_window_frames,
                    blank_margin=rnnt_target_blank_margin,
                    other_margin=rnnt_target_other_margin,
                )
                if rnnt_target_blank_margin_loss_weight > 0.0:
                    loss = (
                        loss
                        + float(rnnt_target_blank_margin_loss_weight)
                        * target_blank_loss
                    )
                if rnnt_target_other_margin_loss_weight > 0.0:
                    loss = (
                        loss
                        + float(rnnt_target_other_margin_loss_weight)
                        * target_other_loss
                    )
                losses.append(float(loss.detach().cpu()))
                raw_pred_ids = logits[:, :, 0, :].argmax(dim=-1)
                stats, batch_repetitions, collapsed_ids = compact_ctc_prediction_stats(
                    raw_pred_ids,
                    input_lengths,
                    target_lengths,
                    compact_ctc_vocab,
                    wait_token_id=wait_token_id,
                    word_start_token_id=word_start_token_id,
                )
                stats = dict(stats)
                stats.update(
                    {
                        "rnnt_nonblank_rate_loss": float(rate_loss.detach().cpu()),
                        "rnnt_pred_nonblank_rate": float(pred_rate.detach().cpu()),
                        "rnnt_target_nonblank_rate": float(target_rate.detach().cpu()),
                        "rnnt_target_blank_margin_loss": float(
                            target_blank_loss.detach().cpu()
                        ),
                        "rnnt_target_other_margin_loss": float(
                            target_other_loss.detach().cpu()
                        ),
                        "rnnt_target_blank_margin": float(
                            target_blank_margin.detach().cpu()
                        ),
                        "rnnt_target_other_margin": float(
                            target_other_margin.detach().cpu()
                        ),
                    }
                )
                pred_stats.append(stats)
                repetition_stats.extend(batch_repetitions)
                for idx, ids in enumerate(collapsed_ids):
                    hypothesis = decode_realtime_token_ids(
                        tokenizer,
                        ids,
                        wait_token_id=wait_token_id,
                        word_start_token_id=word_start_token_id,
                    )
                    predictions.append(
                        {
                            "source": batch["sources"][idx],
                            "reference": batch["texts"][idx],
                            "hypothesis": hypothesis,
                            "hypothesis_char_length": len(hypothesis),
                            "hypothesis_word_count": len(hypothesis.split()),
                            "repetition": batch_repetitions[idx],
                            "decode_mode": alignment_loss,
                        }
                    )
                continue
            if alignment_loss == "rnnt_lite":
                if compact_ctc_vocab is None:
                    raise ValueError("rnnt_lite evaluation requires compact vocab")
                compact_labels = batch["compact_frame_labels"].to(device)
                compact_previous = batch["compact_previous_labels"].to(device)
                logits = forward_rnnt_lite_for_training(
                    model,
                    mels,
                    compact_previous,
                )
                steps = min(logits.shape[1], compact_labels.shape[1])
                loss = frame_cross_entropy(
                    logits[:, :steps, :],
                    compact_labels[:, :steps],
                    vocab_size=len(compact_ctc_vocab.token_ids),
                    wait_token_id=compact_ctc_vocab.blank_index,
                    wait_loss_weight=wait_loss_weight,
                    loss_mode=loss_mode,
                    text_label_smoothing=text_label_smoothing,
                )
                losses.append(float(loss.detach().cpu()))
                raw_pred_ids = logits[:, :steps, :].argmax(dim=-1)
                stats, batch_repetitions, decoded_ids = compact_frame_prediction_stats(
                    raw_pred_ids,
                    compact_labels[:, :steps],
                    compact_ctc_vocab,
                    wait_token_id=wait_token_id,
                    word_start_token_id=word_start_token_id,
                )
                pred_stats.append(stats)
                repetition_stats.extend(batch_repetitions)
                for idx, ids in enumerate(decoded_ids):
                    hypothesis = decode_realtime_token_ids(
                        tokenizer,
                        ids,
                        wait_token_id=wait_token_id,
                        word_start_token_id=word_start_token_id,
                    )
                    predictions.append(
                        {
                            "source": batch["sources"][idx],
                            "reference": batch["texts"][idx],
                            "hypothesis": hypothesis,
                            "hypothesis_char_length": len(hypothesis),
                            "hypothesis_word_count": len(hypothesis.split()),
                            "repetition": batch_repetitions[idx],
                            "decode_mode": alignment_loss,
                        }
                    )
                continue
            if alignment_loss in {"ctc", "compact_ctc"}:
                if alignment_loss == "compact_ctc":
                    if compact_ctc_vocab is None:
                        raise ValueError("compact_ctc evaluation requires compact_ctc_vocab")
                    logits = forward_compact_ctc_for_training(model, mels)
                    targets = batch["compact_ctc_targets"].to(device)
                    blank_token_id_for_loss = compact_ctc_vocab.blank_index
                else:
                    logits = forward_ctc_for_training(model, mels)
                    targets = batch["ctc_targets"].to(device)
                    blank_token_id_for_loss = wait_token_id
                input_lengths = torch.minimum(
                    batch["decoder_lengths"].to(device),
                    torch.full_like(batch["decoder_lengths"].to(device), logits.shape[1]),
                )
                target_lengths = batch["ctc_target_lengths"].to(device)
                loss = ctc_loss(
                    logits,
                    targets,
                    input_lengths,
                    target_lengths,
                    blank_token_id=blank_token_id_for_loss,
                )
                losses.append(float(loss.detach().cpu()))
                raw_pred_ids = logits.argmax(dim=-1)
                if alignment_loss == "compact_ctc":
                    stats, batch_repetitions, collapsed_ids = compact_ctc_prediction_stats(
                        raw_pred_ids,
                        input_lengths,
                        target_lengths,
                        compact_ctc_vocab,
                        wait_token_id=wait_token_id,
                        word_start_token_id=word_start_token_id,
                    )
                else:
                    stats, batch_repetitions, collapsed_ids = ctc_prediction_stats(
                        raw_pred_ids,
                        input_lengths,
                        target_lengths,
                        wait_token_id=wait_token_id,
                        word_start_token_id=word_start_token_id,
                    )
                pred_stats.append(stats)
                repetition_stats.extend(batch_repetitions)
                for idx, ids in enumerate(collapsed_ids):
                    hypothesis = decode_realtime_token_ids(
                        tokenizer,
                        ids,
                        wait_token_id=wait_token_id,
                        word_start_token_id=word_start_token_id,
                    )
                    predictions.append(
                        {
                            "source": batch["sources"][idx],
                            "reference": batch["texts"][idx],
                            "hypothesis": hypothesis,
                            "hypothesis_char_length": len(hypothesis),
                            "hypothesis_word_count": len(hypothesis.split()),
                            "repetition": batch_repetitions[idx],
                            "decode_mode": alignment_loss,
                        }
                    )
                continue

            use_emit_gate = alignment_loss == "emit_gate"
            logits, emit_logits = forward_for_training(
                model,
                mels,
                previous,
                use_emit_gate=use_emit_gate,
            )
            steps = min(logits.shape[1], labels.shape[1])
            if use_emit_gate:
                loss = emit_gate_cross_entropy(
                    logits[:, :steps, :],
                    emit_logits[:, :steps],
                    labels[:, :steps],
                    vocab_size=vocab_size,
                    wait_token_id=wait_token_id,
                    gate_loss_weight=emit_gate_loss_weight,
                    gate_wait_weight=emit_gate_wait_weight,
                    emit_rate_loss_weight=emit_rate_loss_weight,
                    text_label_smoothing=text_label_smoothing,
                )
            else:
                loss = frame_cross_entropy(
                    logits[:, :steps, :],
                    labels[:, :steps],
                    vocab_size=vocab_size,
                    wait_token_id=wait_token_id,
                    wait_loss_weight=wait_loss_weight,
                    loss_mode=loss_mode,
                    text_label_smoothing=text_label_smoothing,
                )
            losses.append(float(loss.detach().cpu()))
            pred_ids = greedy_predictions(
                logits[:, :steps, :],
                None if emit_logits is None else emit_logits[:, :steps],
                wait_token_id=wait_token_id,
                emit_threshold=emit_threshold,
            )
            pred_stats.append(
                prediction_stats(
                    pred_ids,
                    labels[:, :steps],
                    wait_token_id=wait_token_id,
                    word_start_token_id=word_start_token_id,
                )
            )
            pred_ids_list = pred_ids.detach().cpu().tolist()
            for idx, ids in enumerate(pred_ids_list):
                repetition = token_repetition_stats(
                    ids,
                    ignored_token_ids={wait_token_id, word_start_token_id, -100},
                )
                repetition_stats.append(repetition)
                hypothesis = decode_realtime_token_ids(
                    tokenizer,
                    ids,
                    wait_token_id=wait_token_id,
                    word_start_token_id=word_start_token_id,
                )
                predictions.append(
                    {
                        "source": batch["sources"][idx],
                        "reference": batch["texts"][idx],
                        "hypothesis": hypothesis,
                        "hypothesis_char_length": len(hypothesis),
                        "hypothesis_word_count": len(hypothesis.split()),
                        "repetition": repetition,
                    }
                )
    merged_stats = merge_prediction_stats(pred_stats)
    merged_stats.update(
        {
            f"repetition_{key}": value
            for key, value in merge_token_repetition_stats(repetition_stats).items()
        }
    )
    if compact_ctc_vocab is not None and top1_counter:
        merged_stats.update(top1_counter_stats(top1_counter, compact_ctc_vocab))
    return (
        float(np.mean(losses)) if losses else float("nan"),
        predictions,
        merged_stats,
    )


def main() -> None:
    args = parse_args()
    if args.aligned_window_frames < 0:
        raise ValueError("--aligned-window-frames must be >= 0")
    if args.aligned_window_blank_loss_weight < 0.0:
        raise ValueError("--aligned-window-blank-loss-weight must be >= 0")
    if args.aligned_window_sampled_hard_negatives < 0:
        raise ValueError("--aligned-window-sampled-hard-negatives must be >= 0")
    if args.aligned_window_frequent_negative_count < 0:
        raise ValueError("--aligned-window-frequent-negative-count must be >= 0")
    if args.aligned_window_min_token_weight < 0.0:
        raise ValueError("--aligned-window-min-token-weight must be >= 0")
    if args.aligned_window_max_token_weight <= 0.0:
        raise ValueError("--aligned-window-max-token-weight must be > 0")
    if args.aligned_window_min_token_weight > args.aligned_window_max_token_weight:
        raise ValueError("--aligned-window-min-token-weight must be <= max weight")
    if args.emit_rate_loss_weight < 0.0:
        raise ValueError("--emit-rate-loss-weight must be >= 0")
    if args.rnnt_duration_prior_weight < 0.0:
        raise ValueError("--rnnt-duration-prior-weight must be >= 0")
    if args.rnnt_duration_prior_sigma_frames <= 0.0:
        raise ValueError("--rnnt-duration-prior-sigma-frames must be > 0")
    if args.rnnt_duration_prior_max_penalty < 0.0:
        raise ValueError("--rnnt-duration-prior-max-penalty must be >= 0")
    if args.rnnt_nonblank_rate_loss_weight < 0.0:
        raise ValueError("--rnnt-nonblank-rate-loss-weight must be >= 0")
    if args.rnnt_target_blank_margin_loss_weight < 0.0:
        raise ValueError("--rnnt-target-blank-margin-loss-weight must be >= 0")
    if args.rnnt_target_other_margin_loss_weight < 0.0:
        raise ValueError("--rnnt-target-other-margin-loss-weight must be >= 0")
    if args.rnnt_target_margin_window_frames < 0:
        raise ValueError("--rnnt-target-margin-window-frames must be >= 0")
    if args.rnnt_target_blank_margin < 0.0:
        raise ValueError("--rnnt-target-blank-margin must be >= 0")
    if args.rnnt_target_other_margin < 0.0:
        raise ValueError("--rnnt-target-other-margin must be >= 0")
    if not 0.0 <= args.text_label_smoothing < 1.0:
        raise ValueError("--text-label-smoothing must be in [0, 1)")
    if args.qwen_ar_max_target_tokens <= 0:
        raise ValueError("--qwen-ar-max-target-tokens must be > 0")
    if args.qwen_ar_audio_preserve_loss_weight < 0.0:
        raise ValueError("--qwen-ar-audio-preserve-loss-weight must be >= 0")
    if args.qwen_ar_streaming_chunk_frames <= 0:
        raise ValueError("--qwen-ar-streaming-chunk-frames must be > 0")
    if args.qwen_audio_lora_rank < 0:
        raise ValueError("--qwen-audio-lora-rank must be >= 0")
    if args.qwen_audio_lora_rank > 0 and args.qwen_audio_lora_dropout < 0.0:
        raise ValueError("--qwen-audio-lora-dropout must be >= 0")
    if args.qwen_audio_adapter_zero_init and args.resume_from_checkpoint:
        raise ValueError("--qwen-audio-adapter-zero-init cannot be used with resume checkpoints")
    alignment_loss = resolve_alignment_loss(args)
    if alignment_loss in QWEN_AR_ALIGNMENT_LOSSES:
        if args.decoder_backend not in QWEN_AUDIO_BACKENDS and not args.resume_from_checkpoint:
            raise ValueError(
                f"--alignment-loss {alignment_loss} requires a Qwen audio backend"
            )
        if args.qwen_lora_rank > 0:
            raise ValueError(
                f"{alignment_loss} freezes the Qwen text decoder; do not use --qwen-lora-rank"
            )
    if args.qwen_context_distill_teacher_left_context_sec <= 0.0:
        raise ValueError("--qwen-context-distill-teacher-left-context-sec must be > 0")
    if args.qwen_context_distill_kl_temperature <= 0.0:
        raise ValueError("--qwen-context-distill-kl-temperature must be > 0")
    if args.qwen_context_distill_left_padding_frames < 0:
        raise ValueError("--qwen-context-distill-left-padding-frames must be >= 0")
    for name in (
        "qwen_context_distill_ce_weight",
        "qwen_context_distill_kl_weight",
        "qwen_context_distill_frame_mse_weight",
        "qwen_context_distill_frame_cosine_weight",
        "qwen_context_distill_z_loss_weight",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    qwen_context_distill_student_left_contexts: tuple[float, ...] = ()
    if alignment_loss == "qwen_ar_context_distill":
        if not args.qwen_ar_streaming_audio:
            raise ValueError(
                "qwen_ar_context_distill requires --qwen-ar-streaming-audio "
                "so teacher and student actually use bounded context."
            )
        qwen_context_distill_student_left_contexts = (
            parse_float_csv(args.qwen_context_distill_student_left_context_sec)
            if args.qwen_context_distill_student_left_context_sec.strip()
            else (float(args.qwen_audio_left_context_sec),)
        )
        if any(value <= 0.0 for value in qwen_context_distill_student_left_contexts):
            raise ValueError("student left contexts must all be > 0")
    if args.rnnt_duration_prior_weight > 0.0 and alignment_loss != "rnnt_fb":
        raise ValueError("--rnnt-duration-prior-weight requires --alignment-loss rnnt_fb")
    if args.rnnt_nonblank_rate_loss_weight > 0.0 and alignment_loss != "rnnt_fb":
        raise ValueError("--rnnt-nonblank-rate-loss-weight requires --alignment-loss rnnt_fb")
    margin_loss_enabled = (
        args.rnnt_target_blank_margin_loss_weight > 0.0
        or args.rnnt_target_other_margin_loss_weight > 0.0
    )
    if margin_loss_enabled and alignment_loss != "rnnt_fb":
        raise ValueError("RNNT target margin losses require --alignment-loss rnnt_fb")
    aligned_window_loss_enabled = alignment_loss in {
        "aligned_window_ce",
        "aligned_window_sampled_ce",
    }
    if args.rnnt_duration_prior_weight > 0.0 and not (
        args.train_manifest_jsonl and args.eval_manifest_jsonl
    ):
        raise ValueError("--rnnt-duration-prior-weight requires aligned manifest JSONL inputs")
    if margin_loss_enabled and not (
        args.train_manifest_jsonl and args.eval_manifest_jsonl
    ):
        raise ValueError("RNNT target margin losses require aligned manifest JSONL inputs")
    if aligned_window_loss_enabled and not (
        args.train_manifest_jsonl and args.eval_manifest_jsonl
    ):
        raise ValueError(f"{alignment_loss} requires aligned manifest JSONL inputs")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    tokenizer_source = (
        args.resume_from_checkpoint / "tokenizer"
        if args.resume_from_checkpoint
        and (args.resume_from_checkpoint / "tokenizer").exists()
        else args.tokenizer
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[P]", "[W]"]})
    wait_token_id = int(tokenizer.convert_tokens_to_ids("[P]"))
    word_start_token_id = int(tokenizer.convert_tokens_to_ids("[W]"))
    include_word_start = not args.no_word_start_token
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else wait_token_id
    vocab_size = len(tokenizer)
    qwen_ar_prompt_token_ids: list[int] | None = None
    qwen_ar_audio_placeholder_token_id: int | None = None
    if alignment_loss in QWEN_AR_ALIGNMENT_LOSSES:
        audio_placeholder = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        if not isinstance(audio_placeholder, int) or int(audio_placeholder) < 0:
            raise ValueError("Qwen tokenizer is missing <|audio_pad|>")
        qwen_ar_audio_placeholder_token_id = int(audio_placeholder)
        qwen_ar_prompt_token_ids = [
            int(token_id)
            for token_id in tokenizer.encode(
                qwen_asr_prompt_text(
                    context=args.qwen_ar_context,
                    language=args.qwen_ar_language,
                ),
                add_special_tokens=False,
            )
        ]
        if qwen_ar_audio_placeholder_token_id not in qwen_ar_prompt_token_ids:
            raise ValueError("Qwen AR prompt does not contain <|audio_pad|>")

    if args.resume_from_checkpoint:
        config = load_checkpoint_config(args.resume_from_checkpoint)
    else:
        d_model = (
            qwen3_asr_text_hidden_size(args.qwen_decoder_model)
            if args.decoder_backend in QWEN_DECODER_BACKENDS
            else args.d_model
        )
        config = RealtimeAudioConfig(
            d_model=d_model,
            audio_num_layers=args.audio_layers,
            audio_num_heads=args.audio_heads,
            audio_ffn_multiplier=2,
            conv_kernel_size=5,
            audio_window_sec=15.0,
            qwen_audio_right_context_ms=(
                0 if args.qwen_audio_strict_causal else args.qwen_audio_right_context_ms
            ),
            qwen_audio_left_context_sec=args.qwen_audio_left_context_sec,
            qwen_audio_strict_causal=args.qwen_audio_strict_causal,
            qwen_audio_adapter_hidden_dim=args.qwen_audio_adapter_hidden_dim,
            qwen_audio_adapter_layers=args.qwen_audio_adapter_layers,
            qwen_audio_adapter_dropout=args.qwen_audio_adapter_dropout,
            qwen_audio_adapter_residual_scale=args.qwen_audio_adapter_residual_scale,
        )

    if bool(args.train_manifest_jsonl) != bool(args.eval_manifest_jsonl):
        raise ValueError("Pass both --train-manifest-jsonl and --eval-manifest-jsonl, or neither.")
    require_rnnt_word_alignments = (
        args.rnnt_duration_prior_weight > 0.0
        or margin_loss_enabled
        or aligned_window_loss_enabled
    )

    if args.train_manifest_jsonl and args.eval_manifest_jsonl:
        train_examples = load_manifest_examples(
            manifest_jsonl=args.train_manifest_jsonl,
            tokenizer=tokenizer,
            wait_token_id=wait_token_id,
            word_start_token_id=word_start_token_id,
            bos_token_id=bos_token_id,
            config=config,
            target_delay_sec=args.target_delay_sec,
            max_audio_sec=args.max_audio_sec,
            include_word_start=include_word_start,
            require_word_alignments=require_rnnt_word_alignments,
        )
        eval_examples = load_manifest_examples(
            manifest_jsonl=args.eval_manifest_jsonl,
            tokenizer=tokenizer,
            wait_token_id=wait_token_id,
            word_start_token_id=word_start_token_id,
            bos_token_id=bos_token_id,
            config=config,
            target_delay_sec=args.target_delay_sec,
            max_audio_sec=args.max_audio_sec,
            include_word_start=include_word_start,
            require_word_alignments=require_rnnt_word_alignments,
        )
    else:
        train_examples = []
        eval_examples = []
        for source in args.sources:
            train_examples.extend(
                load_examples(
                    source_name=source,
                    split=SOURCE_SPECS[source]["split"],
                    limit=args.max_train_per_source,
                    tokenizer=tokenizer,
                    wait_token_id=wait_token_id,
                    word_start_token_id=word_start_token_id,
                    bos_token_id=bos_token_id,
                    config=config,
                    target_delay_sec=args.target_delay_sec,
                    max_audio_sec=args.max_audio_sec,
                    include_word_start=include_word_start,
                    rng=rng,
                )
            )
            eval_examples.extend(
                load_examples(
                    source_name=source,
                    split="validation",
                    limit=args.max_eval_per_source,
                    tokenizer=tokenizer,
                    wait_token_id=wait_token_id,
                    word_start_token_id=word_start_token_id,
                    bos_token_id=bos_token_id,
                    config=config,
                    target_delay_sec=args.target_delay_sec,
                    max_audio_sec=args.max_audio_sec,
                    include_word_start=include_word_start,
                    rng=rng,
                )
            )
    compact_ctc_vocab: CompactCTCVocab | None = None
    if alignment_loss in {
        "compact_ctc",
        "aligned_window_ce",
        "aligned_window_sampled_ce",
        "rnnt_lite",
        "rnnt_fb",
    }:
        max_compact_tokens = (
            args.compact_ctc_max_tokens
            if alignment_loss == "compact_ctc"
            else args.rnnt_lite_max_tokens
        )
        compact_ctc_vocab = build_compact_ctc_vocab(
            [example.ctc_targets.tolist() for example in train_examples + eval_examples],
            blank_token_id=wait_token_id,
            max_tokens=max_compact_tokens,
        )
        before_train = len(train_examples)
        before_eval = len(eval_examples)
        train_examples = filter_examples_for_compact_ctc_vocab(
            train_examples,
            compact_ctc_vocab,
        )
        eval_examples = filter_examples_for_compact_ctc_vocab(
            eval_examples,
            compact_ctc_vocab,
        )
        dropped_train = before_train - len(train_examples)
        dropped_eval = before_eval - len(eval_examples)
        if dropped_train or dropped_eval:
            print(
                json.dumps(
                    {
                        "compact_ctc_dropped_train": dropped_train,
                        "compact_ctc_dropped_eval": dropped_eval,
                    }
                )
            )
    compact_counts: Counter[int] = Counter()
    aligned_window_token_weights: torch.Tensor | None = None
    aligned_window_frequent_negatives: list[int] = []
    aligned_window_weight_stats: dict[str, object] | None = None
    if compact_ctc_vocab is not None:
        compact_counts = compact_token_counts(train_examples, compact_ctc_vocab)
        aligned_window_token_weights = compact_token_class_weights(
            compact_counts,
            vocab_size=len(compact_ctc_vocab.token_ids),
            blank_index=compact_ctc_vocab.blank_index,
            mode=args.aligned_window_token_weighting,
            min_weight=args.aligned_window_min_token_weight,
            max_weight=args.aligned_window_max_token_weight,
        )
        aligned_window_frequent_negatives = compact_frequent_negative_indices(
            compact_counts,
            blank_index=compact_ctc_vocab.blank_index,
            limit=args.aligned_window_frequent_negative_count,
        )
        aligned_window_weight_stats = compact_weight_stats(
            aligned_window_token_weights,
            compact_counts,
            compact_ctc_vocab,
        )
    rng.shuffle(train_examples)
    rng.shuffle(eval_examples)
    if not train_examples:
        raise RuntimeError("No training examples loaded")
    if not eval_examples:
        raise RuntimeError("No eval examples loaded")

    collate = lambda batch: collate_batch(  # noqa: E731
        batch,
        wait_token_id,
        word_start_token_id=word_start_token_id,
        compact_ctc_vocab=compact_ctc_vocab,
    )
    train_loader = DataLoader(
        TinyASRDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    eval_loader = DataLoader(
        TinyASRDataset(eval_examples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    train_label_stats = label_stats(
        torch.cat([example.labels for example in train_examples]),
        wait_token_id,
    )
    train_ctc_label_stats = ctc_label_stats(
        train_examples,
        wait_token_id=wait_token_id,
    )

    qwen_lora_modules: list[str] = []
    qwen_audio_lora_modules: list[str] = []
    if args.resume_from_checkpoint:
        model = load_realtime_model(args.resume_from_checkpoint, map_location="cpu")
        if not hasattr(model, "vocab_size"):
            raise TypeError(
                f"Checkpoint model does not expose vocab_size: {type(model).__name__}"
            )
        model.emit_threshold = args.emit_threshold
        qwen_lora_modules = list(getattr(model, "qwen_lora_modules", []))
        qwen_audio_lora_modules = list(
            getattr(model, "qwen_audio_lora_modules", [])
        )
        if alignment_loss in QWEN_AR_ALIGNMENT_LOSSES and not isinstance(
            model,
            QWEN_AUDIO_MODEL_TYPES,
        ):
            raise ValueError(f"{alignment_loss} resume checkpoint must use a Qwen audio backend")
        if isinstance(model, QWEN_AUDIO_MODEL_TYPES):
            if args.freeze_qwen_audio and args.train_qwen_audio_last_n_layers:
                raise ValueError(
                    "--freeze-qwen-audio and --train-qwen-audio-last-n-layers "
                    "are mutually exclusive."
                )
            if args.qwen_audio_lora_rank > 0 and args.train_qwen_audio_last_n_layers:
                raise ValueError(
                    "--qwen-audio-lora-rank should not be combined with "
                    "--train-qwen-audio-last-n-layers."
                )
            if args.freeze_qwen_audio:
                model.freeze_qwen_audio_all()
            elif args.train_qwen_audio_last_n_layers:
                model.freeze_qwen_audio_layers(
                    train_last_n_layers=args.train_qwen_audio_last_n_layers
                )
            if args.qwen_audio_lora_rank > 0:
                existing_audio_lora = getattr(model, "qwen_audio_lora_config", None)
                if existing_audio_lora:
                    qwen_audio_lora_modules = list(
                        getattr(model, "qwen_audio_lora_modules", [])
                    )
                else:
                    model.freeze_qwen_audio_all()
                    qwen_audio_lora_modules = model.add_qwen_audio_lora(
                        rank=args.qwen_audio_lora_rank,
                        alpha=(
                            float(args.qwen_audio_lora_alpha)
                            if args.qwen_audio_lora_alpha is not None
                            else float(args.qwen_audio_lora_rank * 2)
                        ),
                        dropout=args.qwen_audio_lora_dropout,
                        target_names=parse_csv(args.qwen_audio_lora_targets),
                    )
        elif args.qwen_audio_lora_rank > 0:
            raise ValueError("--qwen-audio-lora-rank requires a Qwen audio backend")
        if isinstance(
            model,
            (
                *QWEN_AUDIO_MODEL_TYPES,
                Qwen3ASRRealtimeQwenDecoderModel,
            ),
        ):
            if args.qwen_lora_rank > 0:
                raise ValueError(
                    "--qwen-lora-rank is not supported with "
                    "--resume-from-checkpoint in this script."
                )
            if args.freeze_qwen_all:
                model.freeze_qwen_all()
            elif args.freeze_qwen_layers or args.train_qwen_last_n_layers:
                model.freeze_qwen_layers(
                    train_last_n_layers=args.train_qwen_last_n_layers
                )
            if alignment_loss in QWEN_AR_ALIGNMENT_LOSSES:
                model.freeze_qwen_all()
    elif args.decoder_backend in QWEN_DECODER_BACKENDS:
        if args.decoder_backend == "qwen_audio_surgery":
            model_cls = Qwen3ASRRealtimeQwenAudioSurgeryModel
        elif args.decoder_backend == "qwen_audio_causal_kv":
            model_cls = Qwen3ASRRealtimeQwenAudioCausalModel
        else:
            model_cls = Qwen3ASRRealtimeQwenDecoderModel
        model = model_cls.from_qwen_pretrained(
            args.qwen_decoder_model,
            config=config,
            bos_token_id=int(bos_token_id),
            wait_token_id=wait_token_id,
            dtype=torch_dtype(args.qwen_dtype),
            device_map="cpu",
        )
        model.emit_threshold = args.emit_threshold
        if args.decoder_backend in QWEN_AUDIO_BACKENDS:
            if args.qwen_audio_adapter_zero_init:
                zeroed_blocks = zero_init_qwen_audio_adapter_blocks(model)
                if zeroed_blocks <= 0:
                    raise ValueError(
                        "--qwen-audio-adapter-zero-init requires "
                        "--qwen-audio-adapter-layers > 0"
                    )
            if args.freeze_qwen_audio and args.train_qwen_audio_last_n_layers:
                raise ValueError(
                    "--freeze-qwen-audio and --train-qwen-audio-last-n-layers "
                    "are mutually exclusive."
                )
            if args.qwen_audio_lora_rank > 0 and args.train_qwen_audio_last_n_layers:
                raise ValueError(
                    "--qwen-audio-lora-rank should not be combined with "
                    "--train-qwen-audio-last-n-layers."
                )
            if args.freeze_qwen_audio:
                model.freeze_qwen_audio_all()
            elif args.train_qwen_audio_last_n_layers:
                model.freeze_qwen_audio_layers(
                    train_last_n_layers=args.train_qwen_audio_last_n_layers
                )
            if args.qwen_audio_lora_rank > 0:
                model.freeze_qwen_audio_all()
                qwen_audio_lora_modules = model.add_qwen_audio_lora(
                    rank=args.qwen_audio_lora_rank,
                    alpha=(
                        float(args.qwen_audio_lora_alpha)
                        if args.qwen_audio_lora_alpha is not None
                        else float(args.qwen_audio_lora_rank * 2)
                    ),
                    dropout=args.qwen_audio_lora_dropout,
                    target_names=parse_csv(args.qwen_audio_lora_targets),
                )
        elif args.qwen_audio_lora_rank > 0:
            raise ValueError("--qwen-audio-lora-rank requires a Qwen audio backend")
        if args.qwen_lora_rank > 0:
            if args.train_qwen_last_n_layers:
                raise ValueError(
                    "--qwen-lora-rank should not be combined with "
                    "--train-qwen-last-n-layers in this script."
                )
            model.freeze_qwen_all()
            qwen_lora_modules = model.add_qwen_lora(
                rank=args.qwen_lora_rank,
                alpha=(
                    float(args.qwen_lora_alpha)
                    if args.qwen_lora_alpha is not None
                    else float(args.qwen_lora_rank * 2)
                ),
                dropout=args.qwen_lora_dropout,
                target_names=parse_csv(args.qwen_lora_targets),
            )
        elif args.freeze_qwen_all:
            model.freeze_qwen_all()
        elif args.freeze_qwen_layers or args.train_qwen_last_n_layers:
            model.freeze_qwen_layers(
                train_last_n_layers=args.train_qwen_last_n_layers
            )
        if alignment_loss in QWEN_AR_ALIGNMENT_LOSSES:
            model.freeze_qwen_all()
    else:
        model = Qwen3ASRRealtimeNativeModel(
            config,
            vocab_size=vocab_size,
            bos_token_id=int(bos_token_id),
            decoder_num_layers=args.decoder_layers,
            decoder_num_heads=args.decoder_heads,
            decoder_ffn_multiplier=2,
        )
    if alignment_loss in {
        "compact_ctc",
        "aligned_window_ce",
        "aligned_window_sampled_ce",
    }:
        if compact_ctc_vocab is None:
            raise ValueError(f"{alignment_loss} requested but no compact CTC vocab was built")
        existing_compact_ids = getattr(model, "compact_ctc_token_ids", None)
        if [int(token_id) for token_id in (existing_compact_ids or [])] != compact_ctc_vocab.token_ids:
            configure_compact_ctc_head(
                model,
                compact_ctc_vocab.token_ids,
                blank_index=compact_ctc_vocab.blank_index,
                blank_logit_bias=args.compact_ctc_blank_bias,
            )
    if alignment_loss in {"rnnt_lite", "rnnt_fb"}:
        if compact_ctc_vocab is None:
            raise ValueError(f"{alignment_loss} requested but no compact vocab was built")
        existing_rnnt_ids = getattr(model, "rnnt_lite_token_ids", None)
        if [int(token_id) for token_id in (existing_rnnt_ids or [])] != compact_ctc_vocab.token_ids:
            configure_rnnt_lite_head(
                model,
                compact_ctc_vocab.token_ids,
                blank_index=compact_ctc_vocab.blank_index,
                pred_dim=args.rnnt_lite_pred_dim,
                joint_dim=args.rnnt_lite_joint_dim,
                blank_logit_bias=args.rnnt_lite_blank_bias,
            )
    if alignment_loss in QWEN_AR_ALIGNMENT_LOSSES:
        if not isinstance(model, QWEN_AUDIO_MODEL_TYPES):
            raise ValueError(f"{alignment_loss} requires Qwen audio model weights")
        model.freeze_qwen_all()
        freeze_auxiliary_prediction_heads(model)
    model = model.to(device)
    qwen_context_teacher: Qwen3ASRRealtimeQwenAudioSurgeryModel | None = None
    if alignment_loss == "qwen_ar_context_distill":
        teacher_config = replace(
            config,
            qwen_audio_left_context_sec=float(
                args.qwen_context_distill_teacher_left_context_sec
            ),
        )
        teacher_model = Qwen3ASRRealtimeQwenAudioSurgeryModel.from_qwen_pretrained(
            args.qwen_decoder_model,
            config=teacher_config,
            bos_token_id=int(bos_token_id),
            wait_token_id=wait_token_id,
            dtype=torch_dtype(args.qwen_dtype),
            device_map="cpu",
        )
        if args.qwen_audio_adapter_zero_init:
            zero_init_qwen_audio_adapter_blocks(teacher_model)
        teacher_model.freeze_qwen_audio_all()
        teacher_model.freeze_qwen_all()
        freeze_auxiliary_prediction_heads(teacher_model)
        for param in teacher_model.parameters():
            param.requires_grad = False
        set_qwen_audio_left_context_sec(
            teacher_model,
            float(args.qwen_context_distill_teacher_left_context_sec),
        )
        qwen_context_teacher = teacher_model.to(device)
        qwen_context_teacher.eval()
    qwen_ar_audio_preserve_reference: torch.nn.Module | None = None
    if (
        alignment_loss in QWEN_AR_ALIGNMENT_LOSSES
        and args.qwen_ar_audio_preserve_loss_weight > 0.0
    ):
        qwen_ar_audio_preserve_reference = build_qwen_audio_preserve_reference(
            model,
            mode=args.qwen_ar_audio_preserve_reference,
        ).to(device)
    aligned_window_token_weights_device = (
        None
        if aligned_window_token_weights is None
        else aligned_window_token_weights.to(device)
    )
    model_vocab_size = int(model.vocab_size)
    total_params, trainable_params = count_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": jsonable_args(args),
                "realtime_config": asdict(config),
                "vocab_size": vocab_size,
                "model_vocab_size": model_vocab_size,
                "wait_token_id": wait_token_id,
                "word_start_token_id": word_start_token_id,
                "bos_token_id": int(bos_token_id),
                "train_examples": len(train_examples),
                "eval_examples": len(eval_examples),
                "train_label_stats": train_label_stats,
                "train_ctc_label_stats": train_ctc_label_stats,
                "compact_ctc_vocab_stats": (
                    compact_ctc_vocab_stats(compact_ctc_vocab)
                    if compact_ctc_vocab is not None
                    else None
                ),
                "total_params": total_params,
                "trainable_params": trainable_params,
                "alignment_loss": alignment_loss,
                "loss_mode": args.loss_mode,
                "emit_gate_loss_weight": args.emit_gate_loss_weight,
                "emit_gate_wait_weight": args.emit_gate_wait_weight,
                "emit_rate_loss_weight": args.emit_rate_loss_weight,
                "text_label_smoothing": args.text_label_smoothing,
                "aligned_window_frames": args.aligned_window_frames,
                "aligned_window_blank_loss_weight": args.aligned_window_blank_loss_weight,
                "aligned_window_sampled_hard_negatives": (
                    args.aligned_window_sampled_hard_negatives
                ),
                "aligned_window_token_weighting": args.aligned_window_token_weighting,
                "aligned_window_min_token_weight": args.aligned_window_min_token_weight,
                "aligned_window_max_token_weight": args.aligned_window_max_token_weight,
                "aligned_window_frequent_negative_count": (
                    args.aligned_window_frequent_negative_count
                ),
                "aligned_window_token_weight_stats": aligned_window_weight_stats,
                "aligned_window_frequent_negative_indices": (
                    aligned_window_frequent_negatives
                ),
                "rnnt_duration_prior_weight": args.rnnt_duration_prior_weight,
                "rnnt_duration_prior_sigma_frames": args.rnnt_duration_prior_sigma_frames,
                "rnnt_duration_prior_max_penalty": args.rnnt_duration_prior_max_penalty,
                "rnnt_nonblank_rate_loss_weight": args.rnnt_nonblank_rate_loss_weight,
                "rnnt_target_blank_margin_loss_weight": args.rnnt_target_blank_margin_loss_weight,
                "rnnt_target_other_margin_loss_weight": args.rnnt_target_other_margin_loss_weight,
                "rnnt_target_margin_window_frames": args.rnnt_target_margin_window_frames,
                "rnnt_target_blank_margin": args.rnnt_target_blank_margin,
                "rnnt_target_other_margin": args.rnnt_target_other_margin,
                "emit_threshold": args.emit_threshold,
                "qwen_ar_language": args.qwen_ar_language,
                "qwen_ar_context": args.qwen_ar_context,
                "qwen_ar_max_target_tokens": args.qwen_ar_max_target_tokens,
                "qwen_ar_add_eos": args.qwen_ar_add_eos,
                "qwen_ar_audio_preserve_loss_weight": (
                    args.qwen_ar_audio_preserve_loss_weight
                ),
                "qwen_ar_audio_preserve_reference": (
                    args.qwen_ar_audio_preserve_reference
                ),
                "qwen_ar_streaming_audio": args.qwen_ar_streaming_audio,
                "qwen_ar_streaming_chunk_frames": args.qwen_ar_streaming_chunk_frames,
                "qwen_context_distill_teacher_left_context_sec": (
                    args.qwen_context_distill_teacher_left_context_sec
                ),
                "qwen_context_distill_student_left_context_sec": (
                    args.qwen_context_distill_student_left_context_sec
                ),
                "qwen_context_distill_student_left_contexts": (
                    list(qwen_context_distill_student_left_contexts)
                ),
                "qwen_context_distill_ce_weight": args.qwen_context_distill_ce_weight,
                "qwen_context_distill_kl_weight": args.qwen_context_distill_kl_weight,
                "qwen_context_distill_kl_temperature": (
                    args.qwen_context_distill_kl_temperature
                ),
                "qwen_context_distill_frame_mse_weight": (
                    args.qwen_context_distill_frame_mse_weight
                ),
                "qwen_context_distill_frame_cosine_weight": (
                    args.qwen_context_distill_frame_cosine_weight
                ),
                "qwen_context_distill_z_loss_weight": (
                    args.qwen_context_distill_z_loss_weight
                ),
                "qwen_context_distill_left_padding_frames": (
                    args.qwen_context_distill_left_padding_frames
                ),
                "qwen_audio_adapter_zero_init": args.qwen_audio_adapter_zero_init,
                "qwen_lora_modules": qwen_lora_modules,
                "qwen_audio_lora_rank": args.qwen_audio_lora_rank,
                "qwen_audio_lora_alpha": args.qwen_audio_lora_alpha,
                "qwen_audio_lora_dropout": args.qwen_audio_lora_dropout,
                "qwen_audio_lora_modules": qwen_audio_lora_modules,
                "alignment": (
                    "manifest_word_alignments"
                    if args.train_manifest_jsonl and args.eval_manifest_jsonl
                    else "heuristic_linear_by_word_length"
                ),
                "include_word_start": include_word_start,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model.train()
    step = 0
    train_losses: list[float] = []
    train_rnnt_rate_losses: list[float] = []
    train_rnnt_pred_rates: list[float] = []
    train_rnnt_target_rates: list[float] = []
    train_aux_metric_values: dict[str, list[float]] = {}
    epoch = 0
    progress = tqdm(total=args.steps, desc="train")
    while step < args.steps:
        epoch += 1
        for batch_idx, batch in enumerate(train_loader):
            mels = batch["mels"].to(device)
            previous = batch["previous"].to(device)
            labels = batch["labels"].to(device)
            aux_metrics: dict[str, float] = {}
            if alignment_loss in QWEN_AR_ALIGNMENT_LOSSES:
                if qwen_ar_prompt_token_ids is None:
                    raise ValueError(f"{alignment_loss} training missing Qwen prompt token ids")
                if qwen_ar_audio_placeholder_token_id is None:
                    raise ValueError(
                        f"{alignment_loss} training missing Qwen audio placeholder token id"
                    )
                target_token_ids, target_labels = build_qwen_ar_target_batch(
                    batch["texts"],
                    tokenizer,
                    eos_token_id=tokenizer.eos_token_id,
                    max_target_tokens=args.qwen_ar_max_target_tokens,
                    add_eos=args.qwen_ar_add_eos,
                    device=device,
                )
                batch_prompt_token_ids = build_qwen_ar_prompt_token_ids(
                    batch["languages"],
                    tokenizer,
                    context=args.qwen_ar_context,
                    default_language=args.qwen_ar_language,
                    audio_placeholder_token_id=qwen_ar_audio_placeholder_token_id,
                )
                decoder_lengths = batch["decoder_lengths"].to(device)
                mel_lengths = batch["mel_lengths"].to(device)
                if alignment_loss == "qwen_ar_context_distill":
                    if qwen_context_teacher is None:
                        raise ValueError("qwen_ar_context_distill missing teacher model")
                    student_left_context_sec = float(
                        rng.choice(qwen_context_distill_student_left_contexts)
                    )
                    student_left_context_frames = set_qwen_audio_left_context_sec(
                        model,
                        student_left_context_sec,
                    )
                    set_qwen_audio_left_context_sec(
                        qwen_context_teacher,
                        float(args.qwen_context_distill_teacher_left_context_sec),
                    )
                    student_frame_hidden, preserve_loss = qwen_ar_audio_frames_for_training(
                        model,
                        mels,
                        decoder_lengths,
                        mel_lengths=mel_lengths,
                        audio_preserve_reference=qwen_ar_audio_preserve_reference,
                        streaming_audio=args.qwen_ar_streaming_audio,
                        streaming_chunk_frames=args.qwen_ar_streaming_chunk_frames,
                    )
                    with torch.no_grad():
                        teacher_frame_hidden, _ = qwen_ar_audio_frames_for_training(
                            qwen_context_teacher,
                            mels,
                            decoder_lengths,
                            mel_lengths=mel_lengths,
                            audio_preserve_reference=None,
                            streaming_audio=args.qwen_ar_streaming_audio,
                            streaming_chunk_frames=args.qwen_ar_streaming_chunk_frames,
                        )
                        teacher_padded_hidden, teacher_effective_lengths = (
                            prepend_qwen_ar_audio_padding(
                                teacher_frame_hidden,
                                decoder_lengths,
                                padding_frames=args.qwen_context_distill_left_padding_frames,
                            )
                        )
                        teacher_logits = qwen_ar_logits_from_frame_hidden(
                            qwen_context_teacher,
                            teacher_padded_hidden,
                            target_token_ids,
                            teacher_effective_lengths,
                            prompt_template_token_ids_by_sample=batch_prompt_token_ids,
                            audio_placeholder_token_id=qwen_ar_audio_placeholder_token_id,
                        )
                    student_padded_hidden, student_effective_lengths = (
                        prepend_qwen_ar_audio_padding(
                            student_frame_hidden,
                            decoder_lengths,
                            padding_frames=args.qwen_context_distill_left_padding_frames,
                        )
                    )
                    logits = qwen_ar_logits_from_frame_hidden(
                        model,
                        student_padded_hidden,
                        target_token_ids,
                        student_effective_lengths,
                        prompt_template_token_ids_by_sample=batch_prompt_token_ids,
                        audio_placeholder_token_id=qwen_ar_audio_placeholder_token_id,
                    )
                    ce_loss, aux_metrics = qwen_ar_cross_entropy(logits, target_labels)
                    kl_loss = qwen_ar_kl_distill_loss(
                        logits,
                        teacher_logits,
                        target_labels,
                        temperature=args.qwen_context_distill_kl_temperature,
                    )
                    frame_loss, frame_stats = qwen_ar_frame_distill_loss(
                        student_frame_hidden,
                        teacher_frame_hidden,
                        decoder_lengths,
                        cosine_weight=args.qwen_context_distill_frame_cosine_weight,
                    )
                    z_loss = qwen_ar_z_loss(logits, target_labels)
                    loss = (
                        float(args.qwen_context_distill_ce_weight) * ce_loss
                        + float(args.qwen_context_distill_kl_weight) * kl_loss
                        + float(args.qwen_context_distill_frame_mse_weight) * frame_loss
                        + float(args.qwen_context_distill_z_loss_weight) * z_loss
                    )
                    aux_metrics = dict(aux_metrics)
                    aux_metrics.update(frame_stats)
                    aux_metrics.update(
                        {
                            "qwen_context_student_left_context_sec": student_left_context_sec,
                            "qwen_context_student_left_context_frames": float(
                                student_left_context_frames
                            ),
                            "qwen_context_teacher_left_context_sec": float(
                                args.qwen_context_distill_teacher_left_context_sec
                            ),
                            "qwen_context_ce_loss": float(ce_loss.detach().cpu()),
                            "qwen_context_kl_loss": float(kl_loss.detach().cpu()),
                            "qwen_context_frame_loss": float(frame_loss.detach().cpu()),
                            "qwen_context_z_loss": float(z_loss.detach().cpu()),
                            "qwen_context_weighted_ce_loss": float(
                                (
                                    float(args.qwen_context_distill_ce_weight)
                                    * ce_loss.detach()
                                ).cpu()
                            ),
                            "qwen_context_weighted_kl_loss": float(
                                (
                                    float(args.qwen_context_distill_kl_weight)
                                    * kl_loss.detach()
                                ).cpu()
                            ),
                            "qwen_context_weighted_frame_loss": float(
                                (
                                    float(args.qwen_context_distill_frame_mse_weight)
                                    * frame_loss.detach()
                                ).cpu()
                            ),
                            "qwen_context_weighted_z_loss": float(
                                (
                                    float(args.qwen_context_distill_z_loss_weight)
                                    * z_loss.detach()
                                ).cpu()
                            ),
                        }
                    )
                else:
                    logits, preserve_loss = forward_qwen_ar_for_training(
                        model,
                        mels,
                        target_token_ids,
                        decoder_lengths,
                        mel_lengths=mel_lengths,
                        prompt_template_token_ids_by_sample=batch_prompt_token_ids,
                        audio_placeholder_token_id=qwen_ar_audio_placeholder_token_id,
                        audio_preserve_reference=qwen_ar_audio_preserve_reference,
                        streaming_audio=args.qwen_ar_streaming_audio,
                        streaming_chunk_frames=args.qwen_ar_streaming_chunk_frames,
                    )
                    loss, aux_metrics = qwen_ar_cross_entropy(logits, target_labels)
                if (
                    preserve_loss is not None
                    and args.qwen_ar_audio_preserve_loss_weight > 0.0
                ):
                    loss = (
                        loss
                        + float(args.qwen_ar_audio_preserve_loss_weight)
                        * preserve_loss
                    )
                    aux_metrics = dict(aux_metrics)
                    aux_metrics["qwen_ar_audio_preserve_loss"] = float(
                        preserve_loss.detach().cpu()
                    )
                    aux_metrics["qwen_ar_audio_preserve_weighted_loss"] = float(
                        (
                            float(args.qwen_ar_audio_preserve_loss_weight)
                            * preserve_loss.detach()
                        ).cpu()
                    )
            elif alignment_loss in {"aligned_window_ce", "aligned_window_sampled_ce"}:
                if compact_ctc_vocab is None:
                    raise ValueError(f"{alignment_loss} training requires compact vocab")
                logits = forward_compact_ctc_for_training(model, mels)
                input_lengths = torch.minimum(
                    batch["decoder_lengths"].to(device),
                    torch.full_like(batch["decoder_lengths"].to(device), logits.shape[1]),
                )
                target_lengths = batch["ctc_target_lengths"].to(device)
                sampled_negatives = (
                    args.aligned_window_sampled_hard_negatives
                    if alignment_loss == "aligned_window_sampled_ce"
                    else 0
                )
                (
                    loss,
                    token_loss,
                    blank_loss,
                    target_blank_margin,
                    target_other_margin,
                ) = aligned_window_ce_loss(
                    logits,
                    batch["compact_ctc_targets"].to(device),
                    input_lengths,
                    target_lengths,
                    batch["rnnt_token_frames"].to(device),
                    blank_index=compact_ctc_vocab.blank_index,
                    window_frames=args.aligned_window_frames,
                    blank_loss_weight=args.aligned_window_blank_loss_weight,
                    sampled_negative_count=sampled_negatives,
                    token_class_weights=aligned_window_token_weights_device,
                    frequent_negative_indices=aligned_window_frequent_negatives,
                )
                aux_metrics = {
                    "aligned_window_loss": float(loss.detach().cpu()),
                    "aligned_window_token_loss": float(token_loss.detach().cpu()),
                    "aligned_window_blank_loss": float(blank_loss.detach().cpu()),
                    "aligned_window_target_blank_margin": float(
                        target_blank_margin.detach().cpu()
                    ),
                    "aligned_window_target_other_margin": float(
                        target_other_margin.detach().cpu()
                    ),
                }
            elif alignment_loss == "rnnt_fb":
                if compact_ctc_vocab is None:
                    raise ValueError("rnnt_fb training requires compact vocab")
                targets = batch["compact_ctc_targets"].to(device)
                target_lengths = batch["ctc_target_lengths"].to(device)
                logits = forward_rnnt_fb_for_training(
                    model,
                    mels,
                    targets,
                    target_lengths,
                    blank_index=compact_ctc_vocab.blank_index,
                )
                input_lengths = torch.minimum(
                    batch["decoder_lengths"].to(device),
                    torch.full_like(batch["decoder_lengths"].to(device), logits.shape[1]),
                )
                label_frame_targets = batch["rnnt_token_frames"].to(device)
                loss = rnnt_forward_backward_loss(
                    logits,
                    targets,
                    input_lengths,
                    target_lengths,
                    blank_index=compact_ctc_vocab.blank_index,
                    label_frame_targets=label_frame_targets,
                    duration_prior_weight=args.rnnt_duration_prior_weight,
                    duration_prior_sigma_frames=args.rnnt_duration_prior_sigma_frames,
                    duration_prior_max_penalty=args.rnnt_duration_prior_max_penalty,
                    normalize_by_length=args.rnnt_fb_normalize_by_length,
                )
                rate_loss, pred_rate, target_rate = rnnt_nonblank_rate_loss(
                    logits,
                    input_lengths,
                    target_lengths,
                    blank_index=compact_ctc_vocab.blank_index,
                )
                if args.rnnt_nonblank_rate_loss_weight > 0.0:
                    loss = loss + float(args.rnnt_nonblank_rate_loss_weight) * rate_loss
                (
                    target_blank_loss,
                    target_other_loss,
                    target_blank_margin,
                    target_other_margin,
                ) = rnnt_aligned_token_margin_loss(
                    logits,
                    targets,
                    input_lengths,
                    target_lengths,
                    label_frame_targets,
                    blank_index=compact_ctc_vocab.blank_index,
                    window_frames=args.rnnt_target_margin_window_frames,
                    blank_margin=args.rnnt_target_blank_margin,
                    other_margin=args.rnnt_target_other_margin,
                )
                if args.rnnt_target_blank_margin_loss_weight > 0.0:
                    loss = (
                        loss
                        + float(args.rnnt_target_blank_margin_loss_weight)
                        * target_blank_loss
                    )
                if args.rnnt_target_other_margin_loss_weight > 0.0:
                    loss = (
                        loss
                        + float(args.rnnt_target_other_margin_loss_weight)
                        * target_other_loss
                    )
                aux_metrics = {
                    "rnnt_nonblank_rate_loss": float(rate_loss.detach().cpu()),
                    "rnnt_pred_nonblank_rate": float(pred_rate.detach().cpu()),
                    "rnnt_target_nonblank_rate": float(target_rate.detach().cpu()),
                    "rnnt_target_blank_margin_loss": float(
                        target_blank_loss.detach().cpu()
                    ),
                    "rnnt_target_other_margin_loss": float(
                        target_other_loss.detach().cpu()
                    ),
                    "rnnt_target_blank_margin": float(
                        target_blank_margin.detach().cpu()
                    ),
                    "rnnt_target_other_margin": float(
                        target_other_margin.detach().cpu()
                    ),
                }
            elif alignment_loss == "rnnt_lite":
                if compact_ctc_vocab is None:
                    raise ValueError("rnnt_lite training requires compact vocab")
                compact_labels = batch["compact_frame_labels"].to(device)
                compact_previous = batch["compact_previous_labels"].to(device)
                logits = forward_rnnt_lite_for_training(
                    model,
                    mels,
                    compact_previous,
                )
                steps = min(logits.shape[1], compact_labels.shape[1])
                loss = frame_cross_entropy(
                    logits[:, :steps, :],
                    compact_labels[:, :steps],
                    vocab_size=len(compact_ctc_vocab.token_ids),
                    wait_token_id=compact_ctc_vocab.blank_index,
                    wait_loss_weight=args.wait_loss_weight,
                    loss_mode=args.loss_mode,
                    text_label_smoothing=args.text_label_smoothing,
                )
            elif alignment_loss in {"ctc", "compact_ctc"}:
                if alignment_loss == "compact_ctc":
                    if compact_ctc_vocab is None:
                        raise ValueError("compact_ctc training requires compact_ctc_vocab")
                    logits = forward_compact_ctc_for_training(model, mels)
                    targets = batch["compact_ctc_targets"].to(device)
                    blank_token_id_for_loss = compact_ctc_vocab.blank_index
                else:
                    logits = forward_ctc_for_training(model, mels)
                    targets = batch["ctc_targets"].to(device)
                    blank_token_id_for_loss = wait_token_id
                input_lengths = torch.minimum(
                    batch["decoder_lengths"].to(device),
                    torch.full_like(batch["decoder_lengths"].to(device), logits.shape[1]),
                )
                loss = ctc_loss(
                    logits,
                    targets,
                    input_lengths,
                    batch["ctc_target_lengths"].to(device),
                    blank_token_id=blank_token_id_for_loss,
                )
            else:
                use_emit_gate = alignment_loss == "emit_gate"
                logits, emit_logits = forward_for_training(
                    model,
                    mels,
                    previous,
                    use_emit_gate=use_emit_gate,
                )
                steps = min(logits.shape[1], labels.shape[1])
                if use_emit_gate:
                    loss = emit_gate_cross_entropy(
                        logits[:, :steps, :],
                        emit_logits[:, :steps],
                        labels[:, :steps],
                        vocab_size=model_vocab_size,
                        wait_token_id=wait_token_id,
                        gate_loss_weight=args.emit_gate_loss_weight,
                        gate_wait_weight=args.emit_gate_wait_weight,
                        emit_rate_loss_weight=args.emit_rate_loss_weight,
                        text_label_smoothing=args.text_label_smoothing,
                    )
                else:
                    loss = frame_cross_entropy(
                        logits[:, :steps, :],
                        labels[:, :steps],
                        vocab_size=model_vocab_size,
                        wait_token_id=wait_token_id,
                        wait_loss_weight=args.wait_loss_weight,
                        loss_mode=args.loss_mode,
                        text_label_smoothing=args.text_label_smoothing,
                    )
            (loss / args.grad_acc).backward()
            if (batch_idx + 1) % args.grad_acc == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                loss_value = float(loss.detach().cpu())
                train_losses.append(loss_value)
                if aux_metrics:
                    if "rnnt_nonblank_rate_loss" in aux_metrics:
                        train_rnnt_rate_losses.append(
                            aux_metrics["rnnt_nonblank_rate_loss"]
                        )
                    if "rnnt_pred_nonblank_rate" in aux_metrics:
                        train_rnnt_pred_rates.append(
                            aux_metrics["rnnt_pred_nonblank_rate"]
                        )
                    if "rnnt_target_nonblank_rate" in aux_metrics:
                        train_rnnt_target_rates.append(
                            aux_metrics["rnnt_target_nonblank_rate"]
                        )
                    for key, value in aux_metrics.items():
                        train_aux_metric_values.setdefault(key, []).append(value)
                progress.update(1)
                if step == 1 or (args.log_every > 0 and step % args.log_every == 0):
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "epoch": epoch,
                                "loss": loss_value,
                                **aux_metrics,
                            }
                        )
                    )
                if step >= args.steps:
                    break
    progress.close()

    if alignment_loss == "qwen_ar_context_distill":
        set_qwen_audio_left_context_sec(
            model,
            min(qwen_context_distill_student_left_contexts),
        )

    eval_loss, predictions, eval_prediction_stats = evaluate(
        model,
        eval_loader,
        tokenizer,
        wait_token_id,
        word_start_token_id,
        device,
        model_vocab_size,
        args.wait_loss_weight,
        args.loss_mode,
        args.emit_gate_loss_weight,
        args.emit_gate_wait_weight,
        args.emit_rate_loss_weight,
        args.text_label_smoothing,
        args.emit_threshold,
        alignment_loss,
        compact_ctc_vocab,
        args.rnnt_fb_normalize_by_length,
        args.rnnt_duration_prior_weight,
        args.rnnt_duration_prior_sigma_frames,
        args.rnnt_duration_prior_max_penalty,
        args.rnnt_nonblank_rate_loss_weight,
        args.rnnt_target_blank_margin_loss_weight,
        args.rnnt_target_other_margin_loss_weight,
        args.rnnt_target_margin_window_frames,
        args.rnnt_target_blank_margin,
        args.rnnt_target_other_margin,
        args.aligned_window_frames,
        args.aligned_window_blank_loss_weight,
        args.aligned_window_sampled_hard_negatives,
        aligned_window_token_weights_device,
        aligned_window_frequent_negatives,
        qwen_ar_prompt_token_ids,
        qwen_ar_audio_placeholder_token_id,
        args.qwen_ar_max_target_tokens,
        args.qwen_ar_add_eos,
        args.qwen_ar_context,
        args.qwen_ar_language,
        args.qwen_ar_streaming_audio,
        args.qwen_ar_streaming_chunk_frames,
        args.qwen_context_distill_left_padding_frames,
    )
    model.save_pretrained(args.output_dir)
    with (args.output_dir / "eval_predictions.jsonl").open("w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    metrics = {
        "train_steps": step,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "train_label_stats": train_label_stats,
        "train_ctc_label_stats": train_ctc_label_stats,
        "compact_ctc_vocab_stats": (
            compact_ctc_vocab_stats(compact_ctc_vocab)
            if compact_ctc_vocab is not None
            else None
        ),
        "alignment_loss": alignment_loss,
        "wait_loss_weight": args.wait_loss_weight,
        "loss_mode": args.loss_mode,
        "emit_gate_loss_weight": args.emit_gate_loss_weight,
        "emit_gate_wait_weight": args.emit_gate_wait_weight,
        "emit_rate_loss_weight": args.emit_rate_loss_weight,
        "text_label_smoothing": args.text_label_smoothing,
        "aligned_window_frames": args.aligned_window_frames,
        "aligned_window_blank_loss_weight": args.aligned_window_blank_loss_weight,
        "aligned_window_sampled_hard_negatives": args.aligned_window_sampled_hard_negatives,
        "aligned_window_token_weighting": args.aligned_window_token_weighting,
        "aligned_window_min_token_weight": args.aligned_window_min_token_weight,
        "aligned_window_max_token_weight": args.aligned_window_max_token_weight,
        "aligned_window_frequent_negative_count": args.aligned_window_frequent_negative_count,
        "aligned_window_token_weight_stats": aligned_window_weight_stats,
        "aligned_window_frequent_negative_indices": aligned_window_frequent_negatives,
        "rnnt_fb_normalize_by_length": args.rnnt_fb_normalize_by_length,
        "rnnt_duration_prior_weight": args.rnnt_duration_prior_weight,
        "rnnt_duration_prior_sigma_frames": args.rnnt_duration_prior_sigma_frames,
        "rnnt_duration_prior_max_penalty": args.rnnt_duration_prior_max_penalty,
        "rnnt_nonblank_rate_loss_weight": args.rnnt_nonblank_rate_loss_weight,
        "rnnt_target_blank_margin_loss_weight": args.rnnt_target_blank_margin_loss_weight,
        "rnnt_target_other_margin_loss_weight": args.rnnt_target_other_margin_loss_weight,
        "rnnt_target_margin_window_frames": args.rnnt_target_margin_window_frames,
        "rnnt_target_blank_margin": args.rnnt_target_blank_margin,
        "rnnt_target_other_margin": args.rnnt_target_other_margin,
        "emit_threshold": args.emit_threshold,
        "decoder_backend": args.decoder_backend,
        "qwen_ar_language": args.qwen_ar_language,
        "qwen_ar_context": args.qwen_ar_context,
        "qwen_ar_max_target_tokens": args.qwen_ar_max_target_tokens,
        "qwen_ar_add_eos": args.qwen_ar_add_eos,
        "qwen_ar_audio_preserve_loss_weight": args.qwen_ar_audio_preserve_loss_weight,
        "qwen_ar_audio_preserve_reference": args.qwen_ar_audio_preserve_reference,
        "qwen_ar_streaming_audio": args.qwen_ar_streaming_audio,
        "qwen_ar_streaming_chunk_frames": args.qwen_ar_streaming_chunk_frames,
        "qwen_context_distill_teacher_left_context_sec": (
            args.qwen_context_distill_teacher_left_context_sec
        ),
        "qwen_context_distill_student_left_context_sec": (
            args.qwen_context_distill_student_left_context_sec
        ),
        "qwen_context_distill_student_left_contexts": (
            list(qwen_context_distill_student_left_contexts)
        ),
        "qwen_context_distill_ce_weight": args.qwen_context_distill_ce_weight,
        "qwen_context_distill_kl_weight": args.qwen_context_distill_kl_weight,
        "qwen_context_distill_kl_temperature": args.qwen_context_distill_kl_temperature,
        "qwen_context_distill_frame_mse_weight": (
            args.qwen_context_distill_frame_mse_weight
        ),
        "qwen_context_distill_frame_cosine_weight": (
            args.qwen_context_distill_frame_cosine_weight
        ),
        "qwen_context_distill_z_loss_weight": args.qwen_context_distill_z_loss_weight,
        "qwen_context_distill_left_padding_frames": (
            args.qwen_context_distill_left_padding_frames
        ),
        "qwen_audio_adapter_zero_init": args.qwen_audio_adapter_zero_init,
        "qwen_lora_rank": args.qwen_lora_rank,
        "qwen_lora_alpha": args.qwen_lora_alpha,
        "qwen_lora_dropout": args.qwen_lora_dropout,
        "qwen_lora_modules": qwen_lora_modules,
        "qwen_audio_lora_rank": args.qwen_audio_lora_rank,
        "qwen_audio_lora_alpha": args.qwen_audio_lora_alpha,
        "qwen_audio_lora_dropout": args.qwen_audio_lora_dropout,
        "qwen_audio_lora_modules": qwen_audio_lora_modules,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "first_train_loss": train_losses[0] if train_losses else None,
        "last_train_loss": train_losses[-1] if train_losses else None,
        "train_rnnt_nonblank_rate_loss": (
            float(np.mean(train_rnnt_rate_losses))
            if train_rnnt_rate_losses
            else None
        ),
        "train_rnnt_pred_nonblank_rate": (
            float(np.mean(train_rnnt_pred_rates))
            if train_rnnt_pred_rates
            else None
        ),
        "train_rnnt_target_nonblank_rate": (
            float(np.mean(train_rnnt_target_rates))
            if train_rnnt_target_rates
            else None
        ),
        "eval_loss": eval_loss,
        "eval_prediction_stats": eval_prediction_stats,
        "loss_decreased": bool(train_losses and train_losses[-1] < train_losses[0]),
    }
    for key, values in train_aux_metric_values.items():
        metrics[f"train_{key}"] = float(np.mean(values)) if values else None
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()
