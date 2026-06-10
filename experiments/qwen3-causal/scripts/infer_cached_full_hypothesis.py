#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer

from qwen3_streaming.audio_io import load_audio_mono
from qwen3_streaming.cached_full_hypothesis import (
    CachedFullHypothesisConfig,
    CachedFullHypothesisStreamer,
    SegmentedCachedFullHypothesisStreamer,
    added_token_id,
    qwen_asr_prompt_text,
)
from qwen3_streaming.metrics import token_repetition_stats
from qwen3_streaming.native_realtime_model import (
    Qwen3ASRRealtimeQwenAudioCausalModel,
    Qwen3ASRRealtimeQwenAudioSurgeryModel,
    _register_qwen3_asr_transformers,
)
from qwen3_streaming.realtime_config import RealtimeAudioConfig
from qwen3_streaming.realtime_features import (
    log_mel_spectrogram,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Streaming inference that caches finalized Qwen audio embeddings, "
            "reruns a full greedy text hypothesis on each update, and commits "
            "only stable token prefixes."
        )
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--audio-backend",
        choices=("qwen_audio_surgery", "qwen_audio_causal_kv"),
        default="qwen_audio_surgery",
        help="Audio backend used when loading --model-id directly.",
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-ms", type=float, default=320.0)
    parser.add_argument("--qwen-audio-left-context-sec", type=float, default=None)
    parser.add_argument("--qwen-audio-right-context-ms", type=int, default=None)
    parser.add_argument(
        "--feature-mode",
        choices=("repo_mel", "qwen_processor"),
        default="repo_mel",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--max-consecutive-text-tokens", type=int, default=0)
    parser.add_argument("--hold-back-tokens", type=int, default=4)
    parser.add_argument("--hold-back-words", type=int, default=6)
    parser.add_argument("--stable-iterations", type=int, default=2)
    parser.add_argument("--min-commit-audio-sec", type=float, default=0.0)
    parser.add_argument("--normalize-commit-match", action="store_true")
    parser.add_argument(
        "--commit-mode",
        choices=("word", "token"),
        default="word",
    )
    parser.add_argument(
        "--finalize-mode",
        choices=("latest", "stable"),
        default="latest",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("qwen_asr", "bos", "none", "raw"),
        default="qwen_asr",
    )
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--context", default="")
    parser.add_argument(
        "--language",
        required=True,
        help=(
            "Explicit language for the Qwen prompt, e.g. 'English'. Required: "
            "auto language detection flips accented audio to the wrong "
            "language (see RUNS.md 2026-06-10)."
        ),
    )
    parser.add_argument("--no-default-bos", action="store_true")
    parser.add_argument("--no-flush-right-context", action="store_true")
    parser.add_argument("--allow-realtime-specials", action="store_true")
    parser.add_argument("--segment-max-cached-steps", type=int, default=0)
    parser.add_argument("--segment-keep-tail-steps", type=int, default=0)
    parser.add_argument(
        "--segment-finalize-mode",
        choices=("latest", "stable"),
        default="latest",
    )
    parser.add_argument("--segment-prompt-context-words", type=int, default=0)
    parser.add_argument(
        "--segment-prompt-context-prefix",
        default="Previous transcript context:",
    )
    parser.add_argument("--events-jsonl", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _override_audio_context(model, args: argparse.Namespace) -> None:
    encoder = getattr(model, "audio_encoder", None)
    if encoder is None:
        return
    if args.qwen_audio_left_context_sec is not None:
        encoder.left_context_frames = int(
            round(args.qwen_audio_left_context_sec * 1000.0 / model.config.mel_hop_ms)
        )
    if args.qwen_audio_right_context_ms is not None:
        desired = int(round(args.qwen_audio_right_context_ms / model.config.mel_hop_ms))
        try:
            encoder.right_context_frames = desired
        except AttributeError:
            if int(getattr(encoder, "right_context_frames", 0)) != desired:
                raise ValueError(
                    "selected audio backend does not support non-zero right context"
                ) from None
    if getattr(encoder, "left_context_frames", 1) <= 0:
        raise ValueError("qwen audio left context must be > 0 frames")
    if getattr(encoder, "right_context_frames", 0) < 0:
        raise ValueError("qwen audio right context must be >= 0 frames")


def _model_config_from_context_args(
    *,
    args: argparse.Namespace,
    d_model: int,
) -> RealtimeAudioConfig | None:
    if args.qwen_audio_left_context_sec is None and args.qwen_audio_right_context_ms is None:
        return None
    config_kwargs = {"d_model": int(d_model), "audio_window_sec": 15.0}
    if args.qwen_audio_left_context_sec is not None:
        config_kwargs["qwen_audio_left_context_sec"] = float(
            args.qwen_audio_left_context_sec
        )
    if args.qwen_audio_right_context_ms is not None:
        config_kwargs["qwen_audio_right_context_ms"] = int(
            args.qwen_audio_right_context_ms
        )
    return RealtimeAudioConfig(**config_kwargs)


def main() -> None:
    args = parse_args()
    if args.chunk_ms <= 0.0:
        raise ValueError("--chunk-ms must be > 0")
    if (
        args.qwen_audio_left_context_sec is not None
        and args.qwen_audio_left_context_sec <= 0.0
    ):
        raise ValueError("--qwen-audio-left-context-sec must be > 0")
    if (
        args.qwen_audio_right_context_ms is not None
        and args.qwen_audio_right_context_ms < 0
    ):
        raise ValueError("--qwen-audio-right-context-ms must be >= 0")
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be >= 0")
    if args.repetition_penalty <= 0.0:
        raise ValueError("--repetition-penalty must be > 0")
    if args.no_repeat_ngram_size < 0:
        raise ValueError("--no-repeat-ngram-size must be >= 0")
    if args.max_consecutive_text_tokens < 0:
        raise ValueError("--max-consecutive-text-tokens must be >= 0")
    if args.hold_back_tokens < 0:
        raise ValueError("--hold-back-tokens must be >= 0")
    if args.hold_back_words < 0:
        raise ValueError("--hold-back-words must be >= 0")
    if args.stable_iterations <= 0:
        raise ValueError("--stable-iterations must be > 0")
    if args.min_commit_audio_sec < 0.0:
        raise ValueError("--min-commit-audio-sec must be >= 0")
    if args.segment_max_cached_steps < 0:
        raise ValueError("--segment-max-cached-steps must be >= 0")
    if args.segment_keep_tail_steps < 0:
        raise ValueError("--segment-keep-tail-steps must be >= 0")
    if (
        args.segment_max_cached_steps > 0
        and args.segment_keep_tail_steps >= args.segment_max_cached_steps
    ):
        raise ValueError(
            "--segment-keep-tail-steps must be smaller than --segment-max-cached-steps"
        )
    if args.segment_prompt_context_words < 0:
        raise ValueError("--segment-prompt-context-words must be >= 0")

    if args.model_id is None:
        raise ValueError("pass --model-id")

    device = torch.device(args.device)
    _register_qwen3_asr_transformers()
    from transformers import AutoConfig

    tokenizer_source = str(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    hf_config = AutoConfig.from_pretrained(tokenizer_source)
    text_config = hf_config.thinker_config.text_config
    model_cls = (
        Qwen3ASRRealtimeQwenAudioCausalModel
        if args.audio_backend == "qwen_audio_causal_kv"
        else Qwen3ASRRealtimeQwenAudioSurgeryModel
    )
    model = model_cls.from_qwen_pretrained(
        str(args.model_id),
        config=_model_config_from_context_args(
            args=args,
            d_model=int(text_config.hidden_size),
        ),
        bos_token_id=(
            int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else 0
        ),
        wait_token_id=None,
        dtype=torch.bfloat16,
        device_map="cpu",
    ).to(device)
    _override_audio_context(model, args)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    wait_token_id = added_token_id(tokenizer, "[P]")
    word_start_token_id = added_token_id(tokenizer, "[W]")
    eos_token_id = None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)

    prompt_tokens: torch.Tensor | list[int] | None = None
    prompt_prefix_template: list[int] | None = None
    audio_placeholder_token_id: int | None = None
    if args.prompt_mode == "qwen_asr":
        audio_placeholder_token_id = int(tokenizer.convert_tokens_to_ids("<|audio_pad|>"))
        prompt_prefix_template = tokenizer.encode(
            qwen_asr_prompt_text(context=args.context, language=args.language),
            add_special_tokens=False,
        )
    elif args.prompt_mode == "raw":
        if args.prompt_text is None:
            raise ValueError("--prompt-text is required when --prompt-mode=raw")
        prompt_tokens = tokenizer.encode(args.prompt_text, add_special_tokens=False)
    elif args.prompt_mode == "none" or args.no_default_bos:
        prompt_tokens = torch.empty(1, 0, dtype=torch.long, device=device)
    else:
        prompt_tokens = None

    suppress_token_ids: list[int] = []
    if not args.allow_realtime_specials:
        suppress_token_ids.extend([wait_token_id, word_start_token_id])
        for token in ("<|audio_start|>", "<|audio_pad|>", "<|audio_end|>", "<|im_start|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                suppress_token_ids.append(int(token_id))

    audio, sr = load_audio_mono(args.audio, target_sr=model.config.sample_rate)
    if args.feature_mode == "qwen_processor":
        _register_qwen3_asr_transformers()
        processor = AutoProcessor.from_pretrained(model.qwen_model_id)
        features = processor.feature_extractor(
            audio,
            sampling_rate=sr,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )["input_features"][0]
        mel = features.transpose(0, 1).contiguous().to(device)
    else:
        mel = log_mel_spectrogram(audio, sr, model.config).to(device)
    chunk_frames = max(1, int(round(args.chunk_ms / model.config.mel_hop_ms)))

    config = CachedFullHypothesisConfig(
        wait_token_id=wait_token_id,
        word_start_token_id=word_start_token_id,
        eos_token_id=eos_token_id,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_consecutive_text_tokens=args.max_consecutive_text_tokens,
        hold_back_tokens=args.hold_back_tokens,
        hold_back_words=args.hold_back_words,
        stable_iterations=args.stable_iterations,
        min_commit_audio_sec=args.min_commit_audio_sec,
        commit_mode=args.commit_mode,
        normalize_commit_match=args.normalize_commit_match,
        suppress_token_ids=tuple(suppress_token_ids),
        prompt_token_ids=prompt_tokens,
        prompt_prefix_template=prompt_prefix_template,
        audio_placeholder_token_id=audio_placeholder_token_id,
    )
    if args.segment_max_cached_steps > 0:
        streamer = SegmentedCachedFullHypothesisStreamer(
            model,
            tokenizer,
            config,
            segment_max_cached_steps=args.segment_max_cached_steps,
            segment_keep_tail_steps=args.segment_keep_tail_steps,
            segment_finalize_mode=args.segment_finalize_mode,
            segment_prompt_context_words=args.segment_prompt_context_words,
            segment_prompt_base_context=args.context,
            segment_prompt_language=args.language,
            segment_prompt_context_prefix=args.segment_prompt_context_prefix,
        )
    else:
        streamer = CachedFullHypothesisStreamer(model, tokenizer, config)

    with torch.no_grad():
        for start in range(0, mel.shape[0], chunk_frames):
            streamer.append_mel_chunk(
                mel[start : start + chunk_frames, :].unsqueeze(0),
                is_flush=False,
            )

        right_context_frames = int(getattr(model.audio_encoder, "right_context_frames", 0))
        if right_context_frames > 0 and not args.no_flush_right_context:
            flush = mel.new_zeros(1, right_context_frames, model.config.n_mels)
            streamer.append_mel_chunk(flush, is_flush=True)

    final = streamer.finalize(finalize_mode=args.finalize_mode)
    final_tokens = final.final_tokens
    repetition = token_repetition_stats(
        final_tokens,
        ignored_token_ids={wait_token_id, word_start_token_id, -100},
    )
    payload = {
        "model_id": args.model_id,
        "audio": str(args.audio),
        "final_text": final.final_text,
        "final_display_text": final.final_display_text,
        "stable_committed_text": final.stable_committed_text,
        "last_hypothesis_text": final.last_hypothesis_text,
        "final_tokens": len(final_tokens),
        "last_hypothesis_tokens": len(streamer.last_hypothesis_tokens),
        "final_committed_units": final.final_committed_units,
        "events": len(streamer.events),
        "chunk_ms": args.chunk_ms,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "max_consecutive_text_tokens": args.max_consecutive_text_tokens,
        "hold_back_tokens": args.hold_back_tokens,
        "hold_back_words": args.hold_back_words,
        "stable_iterations": args.stable_iterations,
        "commit_mode": args.commit_mode,
        "finalize_mode": args.finalize_mode,
        "segment_max_cached_steps": args.segment_max_cached_steps,
        "segment_keep_tail_steps": args.segment_keep_tail_steps,
        "segment_finalize_mode": args.segment_finalize_mode,
        "segment_prompt_context_words": args.segment_prompt_context_words,
        "segment_prompt_context_prefix": args.segment_prompt_context_prefix,
        "segments_finalized": int(getattr(streamer, "segments_finalized", 0)),
        "dropped_cached_steps_total": int(
            getattr(streamer, "dropped_cached_steps_total", 0)
        ),
        "prompt_mode": args.prompt_mode,
        "feature_mode": args.feature_mode,
        "language": args.language,
        "qwen_audio_left_context_frames": int(
            getattr(getattr(model, "audio_encoder", None), "left_context_frames", 0)
        ),
        "qwen_audio_right_context_frames": int(
            getattr(getattr(model, "audio_encoder", None), "right_context_frames", 0)
        ),
        "cached_steps": int(
            0
            if streamer.state.frame_hidden is None
            else streamer.state.frame_hidden.shape[1]
        ),
        "audio_frames_seen": int(getattr(streamer.state.audio, "frames_seen", 0)),
        "last_recomputed_frames": int(
            getattr(streamer.state.audio, "last_recomputed_frames", 0)
        ),
        "max_recomputed_context_frames": max(
            (
                int(event.get("last_recomputed_context_frames", 0))
                for event in streamer.events
            ),
            default=0,
        ),
        "repetition": repetition,
    }

    if args.events_jsonl is not None:
        args.events_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.events_jsonl.open("w", encoding="utf-8") as handle:
            for event in streamer.events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(final.final_text)


if __name__ == "__main__":
    main()
