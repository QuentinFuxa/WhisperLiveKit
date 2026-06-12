"""Opt-in GPU parity: production causal backend vs the experiments harness.

The production port must produce the same transcripts as the validated
experiments stack (same checkpoint, same mels) even though production feeds
variable-size chunks. Gated because it needs the real model, the fine-tuned
tower and local MCIF audio:

    WLK_RUN_QWEN3_CAUSAL_PARITY=1 \
    QWEN3_TOWER_CKPT=~/Downloads/qwen3_checkpoints/tower_ws2_step60k_*.pt \
    QWEN3_PARITY_MANIFEST=path/to/manifest.jsonl \
    pytest tests/test_qwen3_streaming_causal_parity.py -v

The manifest is JSONL with an ``audio`` field per row (first 3 rows used).
In float32 the texts must match exactly; in bf16 rare near-tie argmax flips
are tolerated up to a small WER delta (documented in RUNS.md 2026-06-12).
"""

import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("WLK_RUN_QWEN3_CAUSAL_PARITY") != "1",
    reason="set WLK_RUN_QWEN3_CAUSAL_PARITY=1 (plus QWEN3_TOWER_CKPT and "
    "QWEN3_PARITY_MANIFEST) to run the causal parity check",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNK_FRAMES = 192
VARIABLE_CHUNKS = [200, 430, 96, 64, 333]


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    dist = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        previous = dist[0]
        dist[0] = i
        for j, hyp_word in enumerate(hyp, start=1):
            current = dist[j]
            dist[j] = min(
                dist[j] + 1,
                dist[j - 1] + 1,
                previous + (ref_word != hyp_word),
            )
            previous = current
    return dist[-1] / len(ref)


def test_prod_causal_matches_experiments_harness():
    torch = pytest.importorskip("torch")
    soundfile = pytest.importorskip("soundfile")

    tower_ckpt = os.environ.get("QWEN3_TOWER_CKPT")
    manifest_path = os.environ.get("QWEN3_PARITY_MANIFEST")
    if not tower_ckpt or not manifest_path:
        pytest.skip("QWEN3_TOWER_CKPT and QWEN3_PARITY_MANIFEST are required")

    sys.path.insert(0, str(REPO_ROOT / "experiments" / "qwen3-causal"))
    from qwen3_streaming.cached_full_hypothesis import (
        CachedFullHypothesisConfig as ExpConfig,
        SegmentedCachedFullHypothesisStreamer as ExpStreamer,
        qwen_asr_prompt_text as exp_prompt_text,
    )
    from qwen3_streaming.native_realtime_model import (
        Qwen3ASRRealtimeQwenAudioCausalModel as ExpModel,
    )
    from qwen3_streaming.realtime_config import (
        RealtimeAudioConfig as ExpAudioConfig,
    )

    from whisperlivekit.qwen3_streaming.asr import Qwen3StreamingASR
    from whisperlivekit.qwen3_streaming.causal import load_tower_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if device == "cuda" else "float32"
    is_exact = dtype == "float32"

    # --- production stack (variable-size chunk feeding) ---
    asr = Qwen3StreamingASR(
        lan="en",
        model_size="Qwen/Qwen3-ASR-0.6B",
        qwen3_streaming_audio_backend="causal",
        qwen3_streaming_tower_checkpoint=tower_ckpt,
        qwen3_streaming_device=device,
        qwen3_streaming_dtype=dtype,
    )

    # --- experiments stack (exact 192-frame feeding, eval-harness style) ---
    exp_config = ExpAudioConfig(
        d_model=asr.audio_config.d_model,
        qwen_audio_left_context_sec=15.0,
        qwen_audio_block_bidirectional=True,
    )
    exp_model = (
        ExpModel.from_qwen_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            config=exp_config,
            bos_token_id=asr.model.bos_token_id,
            wait_token_id=None,
            dtype=getattr(torch, dtype),
            device_map="cpu",
        )
        .to(device)
        .eval()
    )
    load_tower_checkpoint(exp_model, Path(tower_ckpt).expanduser())

    rows = [
        json.loads(line)
        for line in Path(manifest_path).expanduser().read_text().splitlines()
        if line.strip()
    ][:3]
    assert rows, "empty parity manifest"

    def run_prod(features, chunk_sizes):
        streamer = asr.build_streamer("en")
        cursor = 0
        i = 0
        while cursor < features.shape[0]:
            size = chunk_sizes[i % len(chunk_sizes)]
            streamer.append_mel_chunk(
                features[cursor : cursor + size, :].unsqueeze(0)
            )
            cursor += size
            i += 1
        streamer.flush_pending_audio()
        return streamer.finalize(finalize_mode="latest").final_text

    for row in rows:
        audio, sample_rate = soundfile.read(row["audio"], dtype="float32")
        features = asr.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )["input_features"][0].T.to(device)

        prod_exact = run_prod(features, [CHUNK_FRAMES])
        prod_paced = run_prod(features, VARIABLE_CHUNKS)

        # Experiments streamer, exact 192-frame chunks (run-D configuration).
        prompt_template = asr.qwen_tokenizer.encode(
            exp_prompt_text(context="", language="English"),
            add_special_tokens=False,
        )
        exp_streamer = ExpStreamer(
            exp_model,
            asr.qwen_tokenizer,
            ExpConfig(
                wait_token_id=asr.wait_token_id,
                word_start_token_id=asr.word_start_token_id,
                eos_token_id=asr.eos_token_id,
                max_new_tokens=asr.max_new_tokens,
                hold_back_words=asr.hold_back_words,
                stable_iterations=asr.stable_iterations,
                commit_mode="word",
                suppress_token_ids=asr.suppress_token_ids,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                prompt_prefix_template=prompt_template,
                audio_placeholder_token_id=asr.audio_placeholder_token_id,
            ),
            segment_max_cached_steps=asr.segment_max_steps,
            segment_finalize_mode="latest",
            segment_prompt_language="English",
            segment_punct_rollover=True,
            segment_punct_min_steps=150,
            segment_roll_before_generate=True,
            reset_encoder_on_rollover=True,
        )
        for start in range(0, features.shape[0], CHUNK_FRAMES):
            exp_streamer.append_mel_chunk(
                features[start : start + CHUNK_FRAMES, :].unsqueeze(0)
            )
        exp_text = exp_streamer.finalize(finalize_mode="latest").final_text

        # Gate 1 — port parity: identical feeding must yield (near-)identical
        # transcripts. fp32: exact; bf16: rare near-tie argmax flips only.
        if is_exact:
            assert prod_exact == exp_text, (
                f"fp32 parity broken on {row['audio']}:\n"
                f"prod: {prod_exact!r}\nexp:  {exp_text!r}"
            )
        else:
            delta = _word_error_rate(exp_text, prod_exact)
            assert delta < 0.02, (
                f"bf16 divergence {delta:.4f} on {row['audio']}:\n"
                f"prod: {prod_exact!r}\nexp:  {exp_text!r}"
            )

        # Gate 2 — pacing robustness: variable chunking moves decode points
        # (hence punctuation-rollover boundaries), so transcripts legitimately
        # differ in form; QUALITY vs the reference must not move.
        reference = row.get("teacher_text") or row.get("text") or ""
        if reference:
            wer_exact = _word_error_rate(reference, prod_exact)
            wer_paced = _word_error_rate(reference, prod_paced)
            # One-sided: variable pacing must not DEGRADE quality (being
            # better is fine — segmentation boundaries shift either way).
            assert wer_paced <= wer_exact + 0.025, (
                f"pacing degraded quality on {row['audio']}: "
                f"WER {wer_exact:.4f} (exact blocks) vs {wer_paced:.4f} (paced)"
            )
        else:
            assert _word_error_rate(prod_exact, prod_paced) < 0.45
