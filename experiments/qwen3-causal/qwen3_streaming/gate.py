"""Shared frozen-pipeline streaming WER gate for the D-series trainers.

Runs the real cached-full-hypothesis streamer over held-out WLK chunks with
the model under training and scores against the manifest references. Both
``train_tower_distill.py`` and ``train_decoder_lora_ce.py`` gate on this.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .cached_full_hypothesis import (
    CachedFullHypothesisConfig,
    CachedFullHypothesisStreamer,
    added_token_id,
    qwen_asr_prompt_text,
)
from .metrics import word_error_rate


def build_streamer_config(tokenizer, *, language: str) -> CachedFullHypothesisConfig:
    wait_id = added_token_id(tokenizer, "[P]")
    word_id = added_token_id(tokenizer, "[W]")
    suppress = [wait_id, word_id]
    for token in ("<|audio_start|>", "<|audio_pad|>", "<|audio_end|>", "<|im_start|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0:
            suppress.append(tid)
    prompt_template = tokenizer.encode(
        qwen_asr_prompt_text(context="", language=language),
        add_special_tokens=False,
    )
    return CachedFullHypothesisConfig(
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


@torch.no_grad()
def gate_eval(
    model,
    tokenizer,
    processor,
    *,
    manifest: Path,
    limit: int,
    chunk_ms: float,
    language: str,
    device,
    position_offset: int = 0,
) -> float:
    """Streaming WER over the first ``limit`` manifest rows."""
    import soundfile as sf

    was_training = model.training
    model.eval()
    config = build_streamer_config(tokenizer, language=language)
    rows = [
        json.loads(line)
        for line in Path(manifest).read_text().splitlines()
        if line.strip()
    ][:limit]
    chunk_frames = int(round(chunk_ms / 10.0))
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
        )["input_features"][0].T.to(device)
        streamer = CachedFullHypothesisStreamer(model, tokenizer, config)
        if position_offset:
            streamer.state.audio.emitted_steps = int(position_offset)
        for start in range(0, features.shape[0], chunk_frames):
            streamer.append_mel_chunk(
                features[start : start + chunk_frames, :].unsqueeze(0)
            )
        final = streamer.finalize(finalize_mode="latest")
        wer = word_error_rate(
            row.get("teacher_text") or row.get("text") or "", final.final_text
        )
        if wer is not None:
            wers.append(wer)
    if was_training:
        model.train()
    return sum(wers) / len(wers) if wers else float("nan")
