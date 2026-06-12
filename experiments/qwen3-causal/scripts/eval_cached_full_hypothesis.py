#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

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
from qwen3_streaming.metrics import (
    merge_token_repetition_stats,
    streaming_text_event_stats,
    token_repetition_stats,
    word_error_rate,
)
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
        description="Batch eval cached-audio full-hypothesis Qwen ASR streaming."
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--audio-backend",
        choices=("qwen_audio_surgery", "qwen_audio_causal_kv"),
        default="qwen_audio_surgery",
        help="Audio backend used when loading --model-id directly.",
    )
    parser.add_argument("--manifest-jsonl", type=Path, default=None)
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--glob", default="*.wav")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-ms", type=float, default=1000.0)
    parser.add_argument("--qwen-audio-left-context-sec", type=float, default=None)
    parser.add_argument("--qwen-audio-right-context-ms", type=int, default=None)
    parser.add_argument(
        "--qwen-audio-block-bidirectional",
        action="store_true",
        help=(
            "Causal-KV backend: attend bidirectionally within each processed "
            "block (block size = --chunk-ms), causally to the frozen KV "
            "prefix. Append-only; latency = block size."
        ),
    )
    parser.add_argument(
        "--qwen-audio-mutable-tail-sec",
        type=float,
        default=None,
        help=(
            "Bounded mutable tail for --audio-backend qwen_audio_causal_kv: "
            "steps younger than this stay re-computable each chunk; older "
            "steps freeze into the per-layer KV cache. 0 = strict append-only."
        ),
    )
    parser.add_argument(
        "--feature-mode",
        choices=("repo_mel", "qwen_processor"),
        default="qwen_processor",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--max-consecutive-text-tokens", type=int, default=0)
    parser.add_argument("--hold-back-tokens", type=int, default=4)
    parser.add_argument("--hold-back-words", type=int, default=6)
    parser.add_argument("--stable-iterations", type=int, default=2)
    parser.add_argument("--min-commit-audio-sec", type=float, default=0.0)
    parser.add_argument("--normalize-commit-match", action="store_true")
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
    parser.add_argument("--commit-mode", choices=("word", "token"), default="word")
    parser.add_argument("--finalize-mode", choices=("latest", "stable"), default="latest")
    parser.add_argument(
        "--language",
        required=True,
        help=(
            "Explicit language for the Qwen prompt, e.g. 'English'. Required: "
            "auto language detection flips accented audio to the wrong language "
            "and silently corrupts WER comparability (see RUNS.md 2026-06-10)."
        ),
    )
    parser.add_argument("--context", default="")
    parser.add_argument(
        "--reference-field",
        default=None,
        help=(
            "Manifest field to use as the reference (e.g. human_text, "
            "teacher_text). Default keeps the historical fallback chain "
            "teacher_text -> raw_text -> text."
        ),
    )
    parser.add_argument(
        "--audio-position-offset",
        type=int,
        default=0,
        help=(
            "Debug: initialize the causal encoder's global step counter at this "
            "offset, shifting all sinusoidal positions/KV bookkeeping. Probes "
            "position-extrapolation sensitivity on long sessions."
        ),
    )
    parser.add_argument(
        "--tower-state-dict",
        type=Path,
        default=None,
        help="Load a D1 tower_best.pt checkpoint into the audio tower.",
    )
    parser.add_argument(
        "--lora-state-dict",
        type=Path,
        default=None,
        help="Wrap the text decoder with LoRA and load a D2 lora_best.pt checkpoint.",
    )
    parser.add_argument(
        "--encoder-reset-on-rollover",
        action="store_true",
        help="Reset encoder positions/KV at each segment rollover (bounds the "
        "causal chain length to one segment; counters long-form drift).",
    )
    parser.add_argument(
        "--decoder-cache",
        choices=("on", "off"),
        default="on",
        help="Decoder KV-cached incremental decode (on, default) vs legacy full re-forward (off).",
    )
    parser.add_argument(
        "--decoder-rolling-kv",
        choices=("on", "off"),
        default=None,
        help=(
            "Persist the decoder KV over [prompt head + audio] across chunks; "
            "only the new audio and the template tail are re-forwarded per "
            "generation. Default: on (auto-off when --decoder-cache off)."
        ),
    )
    parser.add_argument(
        "--speculative-draft",
        choices=("on", "off"),
        default=None,
        help=(
            "Verify the previous chunk's hypothesis as a draft in the rolling "
            "pass and resume sequential decode from the first divergence. "
            "Default: follows --decoder-rolling-kv; requires it on."
        ),
    )
    parser.add_argument(
        "--decoder-parity-check",
        action="store_true",
        help=(
            "Run every file through BOTH decoder paths (full re-prefill vs "
            "rolling KV + speculative draft) and record byte-parity of every "
            "per-chunk hypothesis and the final text. Doubles compute."
        ),
    )
    parser.add_argument(
        "--segment-punct-rollover",
        action="store_true",
        help=(
            "Roll segments at sentence-final punctuation (IWSLT-style) once "
            "the segment has at least --segment-punct-min-steps cached steps; "
            "--segment-max-cached-steps stays the hard fallback."
        ),
    )
    parser.add_argument("--segment-punct-min-steps", type=int, default=100)
    parser.add_argument(
        "--segment-roll-before-generate",
        action="store_true",
        help=(
            "Roll BEFORE decoding when the incoming chunk would exceed the "
            "segment cap, instead of running the segment's largest generation "
            "on a cache that is dropped right after."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--events-dir", type=Path, default=None)
    return parser.parse_args()


def _load_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.manifest_jsonl is not None:
        items: list[dict[str, Any]] = []
        with args.manifest_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if args.reference_field:
                        reference = record.get(args.reference_field)
                    else:
                        reference = (
                            record.get("teacher_text")
                            or record.get("raw_text")
                            or record.get("text")
                        )
                    items.append(
                        {
                            "id": str(record.get("id") or Path(record["audio"]).stem),
                            "audio": str(record["audio"]),
                            "reference": reference,
                            "source": record.get("source", ""),
                        }
                    )
        return items[: args.limit] if args.limit else items
    if args.audio_dir is None:
        raise ValueError("pass --manifest-jsonl or --audio-dir")
    items = [
        {
            "id": path.stem,
            "audio": str(path),
            "reference": None,
            "source": "",
        }
        for path in sorted(args.audio_dir.glob(args.glob))
    ]
    return items[: args.limit] if args.limit else items


def _mean(values: list[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None]
    return statistics.mean(kept) if kept else None


def _decoder_event_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-chunk decoder cost stats emitted by the streamer."""
    generate_ms = sorted(
        float(event["generate_ms"])
        for event in events
        if event.get("generate_ms") is not None
    )
    draft_tokens = sum(int(event.get("draft_tokens") or 0) for event in events)
    draft_accepted = sum(int(event.get("draft_accepted") or 0) for event in events)
    path_counts: dict[str, int] = {}
    for event in events:
        path = event.get("decoder_path")
        if path:
            path_counts[path] = path_counts.get(path, 0) + 1
    return {
        "generate_ms_mean": statistics.mean(generate_ms) if generate_ms else None,
        "generate_ms_p50": (
            generate_ms[len(generate_ms) // 2] if generate_ms else None
        ),
        "generate_ms_p95": (
            generate_ms[min(len(generate_ms) - 1, int(len(generate_ms) * 0.95))]
            if generate_ms
            else None
        ),
        "draft_tokens_total": draft_tokens,
        "draft_accepted_total": draft_accepted,
        "draft_acceptance_rate": (
            draft_accepted / draft_tokens if draft_tokens > 0 else None
        ),
        "decode_steps_total": sum(
            int(event.get("decode_steps") or 0) for event in events
        ),
        "prefill_positions_total": sum(
            int(event.get("prefill_positions") or 0) for event in events
        ),
        "decoder_path_counts": path_counts,
    }


def _override_audio_context(model, args: argparse.Namespace) -> None:
    encoder = getattr(model, "audio_encoder", None)
    if encoder is None:
        return
    if args.qwen_audio_left_context_sec is not None:
        encoder.left_context_frames = int(
            round(args.qwen_audio_left_context_sec * 1000.0 / model.config.mel_hop_ms)
        )
        if hasattr(encoder, "left_context_steps"):
            # The causal-KV encoder derives its step window once at init; keep
            # it in sync so this override is never a silent no-op.
            encoder.left_context_steps = max(
                1,
                encoder.output_steps_for_mel_frames(encoder.left_context_frames),
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
    if (
        args.qwen_audio_left_context_sec is None
        and args.qwen_audio_right_context_ms is None
        and args.qwen_audio_mutable_tail_sec is None
        and not args.qwen_audio_block_bidirectional
    ):
        return None
    config_kwargs: dict[str, Any] = {"d_model": int(d_model), "audio_window_sec": 15.0}
    if args.qwen_audio_left_context_sec is not None:
        config_kwargs["qwen_audio_left_context_sec"] = float(
            args.qwen_audio_left_context_sec
        )
    if args.qwen_audio_right_context_ms is not None:
        config_kwargs["qwen_audio_right_context_ms"] = int(
            args.qwen_audio_right_context_ms
        )
    if args.qwen_audio_mutable_tail_sec is not None:
        config_kwargs["qwen_audio_mutable_tail_sec"] = float(
            args.qwen_audio_mutable_tail_sec
        )
    if args.qwen_audio_block_bidirectional:
        config_kwargs["qwen_audio_block_bidirectional"] = True
    return RealtimeAudioConfig(**config_kwargs)


def _load_model_and_tokenizer(args: argparse.Namespace, device: torch.device):
    if args.model_id is None:
        raise ValueError("pass --model-id")
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
    if args.tower_state_dict is not None:
        payload = torch.load(args.tower_state_dict, map_location="cpu", weights_only=True)
        model.audio_encoder.audio_tower.load_state_dict(payload["tower_state_dict"])
        print(f"loaded tower checkpoint: {args.tower_state_dict} "
              f"(step {payload.get('step')}, gate {payload.get('gate_wer')})")
    if args.lora_state_dict is not None:
        from qwen3_streaming.lora import add_lora_to_linear_modules

        payload = torch.load(args.lora_state_dict, map_location="cpu", weights_only=True)
        add_lora_to_linear_modules(
            model.text_model,
            target_names=tuple(payload.get("lora_targets") or ()),
            rank=int(payload["lora_rank"]),
            alpha=float(payload["lora_alpha"]),
        )
        missing, unexpected = model.text_model.load_state_dict(
            payload["lora_state_dict"], strict=False
        )
        if unexpected:
            raise RuntimeError(f"unexpected LoRA keys: {unexpected[:5]}")
        model.to(device)
        print(f"loaded LoRA checkpoint: {args.lora_state_dict} "
              f"(step {payload.get('step')}, gate {payload.get('gate_wer')})")
    _override_audio_context(model, args)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    processor = None
    if args.feature_mode == "qwen_processor":
        _register_qwen3_asr_transformers()
        processor = AutoProcessor.from_pretrained(model.qwen_model_id)
    return model, tokenizer, processor


def _run_one(
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    processor,
    item: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    wait_token_id = added_token_id(tokenizer, "[P]")
    word_start_token_id = added_token_id(tokenizer, "[W]")
    eos_token_id = None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)
    audio_placeholder_token_id = int(tokenizer.convert_tokens_to_ids("<|audio_pad|>"))
    prompt_prefix_template = tokenizer.encode(
        qwen_asr_prompt_text(context=args.context, language=args.language),
        add_special_tokens=False,
    )
    suppress_token_ids = [wait_token_id, word_start_token_id]
    for token in ("<|audio_start|>", "<|audio_pad|>", "<|audio_end|>", "<|im_start|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            suppress_token_ids.append(int(token_id))

    audio, sr = load_audio_mono(Path(item["audio"]), target_sr=model.config.sample_rate)
    audio_duration_sec = float(audio.shape[0] / sr) if sr > 0 else None
    if args.feature_mode == "qwen_processor":
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

    def build_streamer(*, rolling: bool, speculative: bool):
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
            prompt_prefix_template=prompt_prefix_template,
            audio_placeholder_token_id=audio_placeholder_token_id,
            use_decoder_kv_cache=args.decoder_cache == "on",
            decoder_rolling_kv=rolling,
            speculative_draft=speculative,
        )
        if args.segment_max_cached_steps > 0 or args.segment_punct_rollover:
            return SegmentedCachedFullHypothesisStreamer(
                model,
                tokenizer,
                config,
                segment_max_cached_steps=args.segment_max_cached_steps,
                segment_keep_tail_steps=args.segment_keep_tail_steps,
                segment_finalize_mode=args.segment_finalize_mode,
                segment_prompt_context_words=args.segment_prompt_context_words,
                segment_prompt_base_context=args.context,
                segment_prompt_language=args.language,
                reset_encoder_on_rollover=args.encoder_reset_on_rollover,
                segment_prompt_context_prefix=args.segment_prompt_context_prefix,
                segment_punct_rollover=args.segment_punct_rollover,
                segment_punct_min_steps=args.segment_punct_min_steps,
                segment_roll_before_generate=args.segment_roll_before_generate,
            )
        return CachedFullHypothesisStreamer(model, tokenizer, config)

    def run_stream(streamer):
        if args.audio_position_offset:
            streamer.state.audio.emitted_steps = int(args.audio_position_offset)
        with torch.no_grad():
            for start in range(0, mel.shape[0], chunk_frames):
                streamer.append_mel_chunk(
                    mel[start : start + chunk_frames, :].unsqueeze(0),
                    is_flush=False,
                )
            right_context = int(
                getattr(model.audio_encoder, "right_context_frames", 0)
            )
            if right_context > 0:
                streamer.append_mel_chunk(
                    mel.new_zeros(1, right_context, model.config.n_mels),
                    is_flush=True,
                )
        return streamer.finalize(finalize_mode=args.finalize_mode)

    parity_fields: dict[str, Any] = {}
    if args.decoder_parity_check:
        baseline = build_streamer(rolling=False, speculative=False)
        baseline_final = run_stream(baseline)
        streamer = build_streamer(rolling=True, speculative=True)
        final = run_stream(streamer)
        mismatch_chunks: list[int] = []
        if len(baseline.events) != len(streamer.events):
            mismatch_chunks.append(min(len(baseline.events), len(streamer.events)))
        else:
            mismatch_chunks = [
                index
                for index, (base_event, cand_event) in enumerate(
                    zip(baseline.events, streamer.events)
                )
                if base_event["hypothesis"] != cand_event["hypothesis"]
            ]
        reference = item.get("reference")
        parity_fields = {
            # Strict byte parity is not attainable in bf16: the incremental
            # prefill computes K/V in different matmul shapes than the
            # monolithic re-prefill, so near-tie argmaxes can flip on rare
            # chunks (self-correcting, since text is regenerated per chunk).
            # Gate on flip rate + WER delta instead.
            "parity_ok": not mismatch_chunks
            and baseline_final.final_text == final.final_text,
            "parity_first_mismatch_chunk": (
                mismatch_chunks[0] if mismatch_chunks else None
            ),
            "parity_mismatch_chunks": len(mismatch_chunks),
            "parity_chunks_total": len(streamer.events),
            "parity_final_text_equal": baseline_final.final_text == final.final_text,
            "baseline_final_text": baseline_final.final_text,
            "baseline_wer_final": (
                word_error_rate(reference, baseline_final.final_text)
                if reference
                else None
            ),
            "baseline_decoder_stats": _decoder_event_stats(baseline.events),
        }
    else:
        streamer = build_streamer(
            rolling=args.decoder_rolling_kv == "on",
            speculative=args.speculative_draft == "on",
        )
        final = run_stream(streamer)

    final_tokens = final.final_tokens
    events = streamer.events

    repetition = token_repetition_stats(
        final_tokens,
        ignored_token_ids={wait_token_id, word_start_token_id, -100},
    )
    if args.events_dir is not None:
        event_path = args.events_dir / f"{item['id']}.jsonl"
        with event_path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    reference = item.get("reference")
    streaming_stats = streaming_text_event_stats(
        events,
        final_text=final.final_text,
        stable_text=final.stable_committed_text,
    )
    max_recomputed = max(
        (int(event["last_recomputed_frames"]) for event in events),
        default=0,
    )
    max_recomputed_context = max(
        (int(event.get("last_recomputed_context_frames", 0)) for event in events),
        default=0,
    )
    cache_bound = int(getattr(model.audio_encoder, "max_recompute_frames", 0))
    context_bound = max(
        0,
        cache_bound
        - min((int(event.get("input_frames", 0)) for event in events), default=0),
    )
    return {
        "id": item["id"],
        "audio": item["audio"],
        "audio_duration_sec": audio_duration_sec,
        "source": item.get("source", ""),
        "reference": reference,
        "final_text": final.final_text,
        "final_display_text": final.final_display_text,
        "stable_committed_text": final.stable_committed_text,
        "last_hypothesis_text": final.last_hypothesis_text,
        "wer_final": word_error_rate(reference, final.final_text) if reference else None,
        "wer_latest": word_error_rate(reference, final.last_hypothesis_text) if reference else None,
        "wer_stable": word_error_rate(reference, final.stable_committed_text) if reference else None,
        "final_tokens": len(final_tokens),
        "last_hypothesis_tokens": len(streamer.last_hypothesis_tokens),
        "final_committed_units": final.final_committed_units,
        "events": len(events),
        "cached_steps": int(
            0
            if streamer.state.frame_hidden is None
            else streamer.state.frame_hidden.shape[1]
        ),
        "segments_finalized": int(getattr(streamer, "segments_finalized", 0)),
        "dropped_cached_steps_total": int(
            getattr(streamer, "dropped_cached_steps_total", 0)
        ),
        "segment_prompt_context_words": int(
            getattr(streamer, "segment_prompt_context_words", 0)
        ),
        "audio_frames_seen": int(getattr(streamer.state.audio, "frames_seen", 0)),
        "max_last_recomputed_frames": max_recomputed,
        "max_recomputed_context_frames": max_recomputed_context,
        "cache_bound_frames": cache_bound,
        "context_recompute_bound_frames": context_bound,
        "cache_bound_ok": bool(cache_bound <= 0 or max_recomputed <= cache_bound),
        "repetition": repetition,
        "streaming": streaming_stats,
        "decoder_stats": _decoder_event_stats(events),
        "segment_rollover_reasons": {
            reason: sum(
                1
                for event in events
                if event.get("segment_rollover_reason") == reason
            )
            for reason in ("cap", "punctuation")
        },
        **parity_fields,
    }


def main() -> None:
    args = parse_args()
    if args.chunk_ms <= 0:
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
    if args.hold_back_tokens < 0 or args.hold_back_words < 0:
        raise ValueError("hold-back values must be >= 0")
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
    if args.decoder_rolling_kv == "on" and args.decoder_cache != "on":
        raise ValueError("--decoder-rolling-kv on requires --decoder-cache on")
    if args.decoder_rolling_kv is None:
        args.decoder_rolling_kv = "on" if args.decoder_cache == "on" else "off"
    if args.speculative_draft == "on" and args.decoder_rolling_kv != "on":
        raise ValueError("--speculative-draft on requires --decoder-rolling-kv on")
    if args.speculative_draft is None:
        args.speculative_draft = args.decoder_rolling_kv
    if args.decoder_parity_check and args.decoder_cache != "on":
        raise ValueError("--decoder-parity-check requires --decoder-cache on")
    if args.segment_punct_min_steps < 1:
        raise ValueError("--segment-punct-min-steps must be >= 1")
    if args.segment_roll_before_generate and args.segment_max_cached_steps <= 0:
        raise ValueError(
            "--segment-roll-before-generate requires --segment-max-cached-steps > 0"
        )

    device = torch.device(args.device)
    items = _load_items(args)
    model, tokenizer, processor = _load_model_and_tokenizer(args, device)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.events_dir is not None:
        args.events_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for item in items:
            start = time.perf_counter()
            error = None
            row: dict[str, Any]
            try:
                row = _run_one(
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    item=item,
                    device=device,
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                row = {
                    "id": item["id"],
                    "audio": item["audio"],
                    "source": item.get("source", ""),
                    "reference": item.get("reference"),
                }
            row["latency_sec"] = time.perf_counter() - start
            duration = row.get("audio_duration_sec")
            row["realtime_factor"] = (
                row["latency_sec"] / float(duration)
                if duration is not None and float(duration) > 0.0
                else None
            )
            row["error"] = error
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    ok_rows = [row for row in rows if row.get("error") is None]
    draft_accepted_total = sum(
        int((row.get("decoder_stats") or {}).get("draft_accepted_total") or 0)
        for row in ok_rows
    )
    draft_tokens_total = sum(
        int((row.get("decoder_stats") or {}).get("draft_tokens_total") or 0)
        for row in ok_rows
    )
    summary = {
        "count": len(rows),
        "ok": len(ok_rows),
        "total_latency_sec": time.perf_counter() - t0,
        "latency_mean_sec": _mean([row.get("latency_sec") for row in ok_rows]),
        "audio_duration_total_sec": sum(
            float(row.get("audio_duration_sec") or 0.0) for row in ok_rows
        ),
        "audio_duration_mean_sec": _mean(
            [row.get("audio_duration_sec") for row in ok_rows]
        ),
        "realtime_factor_mean": _mean([row.get("realtime_factor") for row in ok_rows]),
        "realtime_factor_total": (
            sum(float(row.get("latency_sec") or 0.0) for row in ok_rows)
            / sum(float(row.get("audio_duration_sec") or 0.0) for row in ok_rows)
            if sum(float(row.get("audio_duration_sec") or 0.0) for row in ok_rows) > 0.0
            else None
        ),
        "wer_final_mean": _mean([row.get("wer_final") for row in ok_rows]),
        "wer_latest_mean": _mean([row.get("wer_latest") for row in ok_rows]),
        "wer_stable_mean": _mean([row.get("wer_stable") for row in ok_rows]),
        "cache_bound_violations": sum(
            1 for row in ok_rows if not bool(row.get("cache_bound_ok", False))
        ),
        "max_last_recomputed_frames": max(
            (int(row.get("max_last_recomputed_frames", 0)) for row in ok_rows),
            default=0,
        ),
        "max_recomputed_context_frames": max(
            (int(row.get("max_recomputed_context_frames", 0)) for row in ok_rows),
            default=0,
        ),
        "cache_bound_frames": max(
            (int(row.get("cache_bound_frames", 0)) for row in ok_rows),
            default=0,
        ),
        "final_tokens_mean": _mean([row.get("final_tokens") for row in ok_rows]),
        "latest_tokens_mean": _mean([row.get("last_hypothesis_tokens") for row in ok_rows]),
        "committed_units_mean": _mean([row.get("final_committed_units") for row in ok_rows]),
        "segments_finalized_mean": _mean(
            [row.get("segments_finalized") for row in ok_rows]
        ),
        "dropped_cached_steps_total": sum(
            int(row.get("dropped_cached_steps_total", 0)) for row in ok_rows
        ),
        "generate_ms_mean": _mean(
            [
                (row.get("decoder_stats") or {}).get("generate_ms_mean")
                for row in ok_rows
            ]
        ),
        "draft_acceptance_rate": (
            draft_accepted_total / draft_tokens_total
            if draft_tokens_total > 0
            else None
        ),
        "decode_steps_total": sum(
            int((row.get("decoder_stats") or {}).get("decode_steps_total") or 0)
            for row in ok_rows
        ),
        "prefill_positions_total": sum(
            int((row.get("decoder_stats") or {}).get("prefill_positions_total") or 0)
            for row in ok_rows
        ),
        "parity_ok_all": (
            all(bool(row.get("parity_ok")) for row in ok_rows)
            if any("parity_ok" in row for row in ok_rows)
            else None
        ),
        "parity_mismatch_chunks_total": sum(
            int(row.get("parity_mismatch_chunks") or 0) for row in ok_rows
        ),
        "parity_chunks_total": sum(
            int(row.get("parity_chunks_total") or 0) for row in ok_rows
        ),
        "baseline_wer_final_mean": _mean(
            [row.get("baseline_wer_final") for row in ok_rows]
        ),
        "segment_rollovers_punctuation": sum(
            int((row.get("segment_rollover_reasons") or {}).get("punctuation", 0))
            for row in ok_rows
        ),
        "segment_rollovers_cap": sum(
            int((row.get("segment_rollover_reasons") or {}).get("cap", 0))
            for row in ok_rows
        ),
        "first_display_sec_mean": _mean(
            [
                row.get("streaming", {}).get("first_display_sec")
                for row in ok_rows
            ]
        ),
        "first_commit_sec_mean": _mean(
            [
                row.get("streaming", {}).get("first_commit_sec")
                for row in ok_rows
            ]
        ),
        "stable_coverage_ratio_mean": _mean(
            [
                row.get("streaming", {}).get("stable_coverage_ratio")
                for row in ok_rows
            ]
        ),
        "display_revision_events_mean": _mean(
            [
                row.get("streaming", {}).get("display_revision_events")
                for row in ok_rows
            ]
        ),
        "display_revision_words_mean": _mean(
            [
                row.get("streaming", {}).get("display_revision_words")
                for row in ok_rows
            ]
        ),
        "committed_revision_events_total": sum(
            int(row.get("streaming", {}).get("committed_revision_events", 0))
            for row in ok_rows
        ),
        "repetition": merge_token_repetition_stats(
            [row["repetition"] for row in ok_rows if "repetition" in row]
        ),
        "chunk_ms": args.chunk_ms,
        "feature_mode": args.feature_mode,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "max_consecutive_text_tokens": args.max_consecutive_text_tokens,
        "commit_mode": args.commit_mode,
        "finalize_mode": args.finalize_mode,
        "hold_back_words": args.hold_back_words,
        "stable_iterations": args.stable_iterations,
        "min_commit_audio_sec": args.min_commit_audio_sec,
        "normalize_commit_match": args.normalize_commit_match,
        "segment_max_cached_steps": args.segment_max_cached_steps,
        "segment_keep_tail_steps": args.segment_keep_tail_steps,
        "segment_finalize_mode": args.segment_finalize_mode,
        "segment_prompt_context_words": args.segment_prompt_context_words,
        "segment_prompt_context_prefix": args.segment_prompt_context_prefix,
        "qwen_audio_left_context_frames": int(
            getattr(getattr(model, "audio_encoder", None), "left_context_frames", 0)
        ),
        "qwen_audio_right_context_frames": int(
            getattr(getattr(model, "audio_encoder", None), "right_context_frames", 0)
        ),
    }
    summary_path = args.output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
