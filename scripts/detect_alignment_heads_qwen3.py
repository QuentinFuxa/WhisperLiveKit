#!/usr/bin/env python3
"""
Detect alignment heads in Qwen3-ASR for SimulStreaming-style inference.

Qwen3-ASR is a decoder-only multimodal model: audio is encoded by an audio
encoder and the resulting embeddings are injected into the text sequence
(replacing <|audio_pad|> placeholder tokens).  The text decoder then attends
over the full sequence -- both audio-derived tokens and text tokens -- via
causal self-attention.  There is **no** cross-attention.

For AlignAtt-style streaming, we need to find which (layer, head) pairs in
the text decoder's self-attention best track the monotonic alignment between
generated text tokens and their corresponding audio positions.

Algorithm
---------
For each audio sample with a known transcript:
  1. Run Qwen3-ASR with output_attentions=True
  2. Use the ForcedAligner to get ground-truth word->timestamp alignments
  3. Convert timestamps to audio token positions in the input sequence
  4. For each generated text token, check whether the argmax of each
     attention head (over the audio-token region) points to the correct
     audio position (as determined by the forced aligner)
  5. Accumulate scores per (layer, head)

The heads whose attention argmax matches the ground-truth alignment most
often are the "alignment heads" usable for SimulStreaming.

Reference: Adapted from scripts/determine_alignment_heads.py (Whisper) and
           iwslt26-sst/SimulMT_tests/heads/detect_translation_heads_qwen3.py
"""

import argparse
import io
import json
import logging
import re
import time
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Compatibility patches for qwen_asr 0.0.6 + transformers >= 5.3 ────
def _apply_transformers_compat_patches():
    """Apply all necessary patches to make qwen_asr work with transformers >= 5.3."""
    # 1. check_model_inputs was removed
    try:
        import transformers.utils.generic as _g
        if not hasattr(_g, "check_model_inputs"):
            def check_model_inputs(*args, **kwargs):
                def decorator(fn):
                    return fn
                return decorator
            _g.check_model_inputs = check_model_inputs
    except ImportError:
        pass

    # 2. 'default' rope type was removed from ROPE_INIT_FUNCTIONS
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if "default" not in ROPE_INIT_FUNCTIONS:
            def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
                if hasattr(config, "head_dim"):
                    head_dim = config.head_dim
                else:
                    head_dim = config.hidden_size // config.num_attention_heads
                partial = getattr(config, "partial_rotary_factor", 1.0)
                dim = int(head_dim * partial)
                base = config.rope_theta
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
                return inv_freq, 1.0
            ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    except ImportError:
        pass

    # 3. pad_token_id missing on thinker config
    try:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
            Qwen3ASRThinkerConfig,
        )
        if not hasattr(Qwen3ASRThinkerConfig, "pad_token_id"):
            Qwen3ASRThinkerConfig.pad_token_id = None
    except ImportError:
        pass

    # 4. fix_mistral_regex is now handled internally by transformers 5.3;
    #    qwen_asr passes it explicitly, causing a duplicate-kwarg error.
    try:
        from transformers.models.auto import processing_auto
        _orig_ap_from_pretrained = processing_auto.AutoProcessor.from_pretrained.__func__

        @classmethod
        def _patched_ap_from_pretrained(cls, *args, **kwargs):
            kwargs.pop("fix_mistral_regex", None)
            return _orig_ap_from_pretrained(cls, *args, **kwargs)

        processing_auto.AutoProcessor.from_pretrained = _patched_ap_from_pretrained
    except Exception:
        pass

    # 5. _finalize_model_loading calls initialize_weights which expects
    #    compute_default_rope_parameters on RotaryEmbedding modules.
    try:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRThinkerTextRotaryEmbedding,
        )
        if not hasattr(Qwen3ASRThinkerTextRotaryEmbedding, "compute_default_rope_parameters"):
            @staticmethod
            def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
                if hasattr(config, "head_dim"):
                    head_dim = config.head_dim
                else:
                    head_dim = config.hidden_size // config.num_attention_heads
                partial = getattr(config, "partial_rotary_factor", 1.0)
                dim = int(head_dim * partial)
                base = config.rope_theta
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
                return inv_freq, 1.0
            Qwen3ASRThinkerTextRotaryEmbedding.compute_default_rope_parameters = _compute_default_rope_parameters
    except ImportError:
        pass

_apply_transformers_compat_patches()

# ── Constants ────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
TS_THRESHOLD = 0.1  # Minimum Translation Score to qualify as alignment head
MIN_TEXT_SIMILARITY = 0.3  # Skip clips where generated text is too different from ground truth


def text_similarity(generated: str, reference: str) -> float:
    """Compute text similarity between generated and reference transcriptions.

    Normalizes both strings (lowercase, remove punctuation, collapse whitespace)
    then returns SequenceMatcher ratio.
    """
    def normalize(s):
        s = s.lower()
        s = re.sub(r'[^\w\s]', '', s)
        return re.sub(r'\s+', ' ', s).strip()

    gen_norm = normalize(generated)
    ref_norm = normalize(reference)
    if not gen_norm or not ref_norm:
        return 0.0
    return SequenceMatcher(None, gen_norm, ref_norm).ratio()


def load_dataset_clips(name, config, split, limit):
    """Load audio clips from a HuggingFace dataset."""
    from datasets import Audio as DatasetAudio
    from datasets import load_dataset

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

        clips.append((waveform_np, str(transcript)))
    return clips


def get_device():
    """Select the best available device."""
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA (%s)", torch.cuda.get_device_name())
        return torch.device("cuda")
    else:
        logger.info("Using CPU (will be slow)")
        return torch.device("cpu")


def load_qwen3_asr(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load Qwen3-ASR model, processor, and forced aligner."""
    from qwen_asr.core.transformers_backend import (
        Qwen3ASRConfig,
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRProcessor,
    )
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner
    from transformers import AutoConfig, AutoModel, AutoProcessor

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

    logger.info("Loading model: %s (dtype=%s, device=%s)", model_id, dtype, device)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="eager",
        device_map=str(device),
    )
    model.eval()

    # Force eager attention on all sub-modules (attn_implementation="eager" doesn't
    # propagate through nested model configs in qwen_asr's custom architecture)
    for name, module in model.named_modules():
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            module.config._attn_implementation = "eager"
            module.config._attn_implementation_internal = "eager"

    try:
        processor = AutoProcessor.from_pretrained(model_id, fix_mistral_regex=True)
    except TypeError:
        processor = AutoProcessor.from_pretrained(model_id)

    logger.info("Loading forced aligner: Qwen/Qwen3-ForcedAligner-0.6B")
    forced_aligner = Qwen3ForcedAligner.from_pretrained(
        "Qwen/Qwen3-ForcedAligner-0.6B",
        dtype=dtype,
        device_map=str(device),
    )

    return model, processor, forced_aligner


def find_audio_token_range(input_ids: torch.Tensor, audio_token_id: int) -> Tuple[int, int]:
    """Find the start and end positions of audio tokens in the input sequence."""
    mask = (input_ids == audio_token_id)
    positions = mask.nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        return 0, 0
    return positions[0].item(), positions[-1].item() + 1


def timestamp_to_audio_token_position(
    timestamp_sec: float,
    audio_duration_sec: float,
    audio_token_start: int,
    audio_token_end: int,
) -> int:
    """Convert a timestamp in seconds to the corresponding audio token position.

    Audio tokens span [audio_token_start, audio_token_end) in the input sequence.
    We linearly interpolate within that range based on the timestamp fraction.
    """
    n_audio_tokens = audio_token_end - audio_token_start
    if n_audio_tokens <= 0 or audio_duration_sec <= 0:
        return audio_token_start

    fraction = min(timestamp_sec / audio_duration_sec, 1.0)
    pos = audio_token_start + int(fraction * (n_audio_tokens - 1))
    return max(audio_token_start, min(pos, audio_token_end - 1))


def run_detection(
    model,
    processor,
    forced_aligner,
    clips: List[Tuple[np.ndarray, str]],
    language: Optional[str],
    device: torch.device,
) -> Tuple[np.ndarray, int]:
    """Run alignment head detection on a set of audio clips.

    Uses PyTorch forward hooks on each self_attn module to capture attention
    weights that the decoder layer discards (``hidden_states, _ = self.self_attn(...)``).
    With eager attention, ``self_attn`` always returns ``(attn_output, attn_weights)``
    so the hook can read the weights from the return value.

    Returns:
        g: array of shape (total_heads,) with alignment hit counts
        m: total number of alignment checks performed
    """
    thinker = model.thinker
    text_config = thinker.config.text_config
    num_layers = text_config.num_hidden_layers
    num_heads = text_config.num_attention_heads
    total_heads = num_layers * num_heads

    audio_token_id = thinker.config.audio_token_id

    logger.info(
        "Text decoder: %d layers x %d heads = %d total heads",
        num_layers, num_heads, total_heads,
    )
    logger.info(
        "KV heads: %d (GQA ratio: %d)",
        text_config.num_key_value_heads,
        num_heads // text_config.num_key_value_heads,
    )

    # Build prompt helper (same as Qwen3ASRModel._build_text_prompt)
    from qwen_asr.inference.utils import normalize_language_name

    def build_messages(audio_payload):
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
        ]

    def build_text_prompt(force_language=None):
        msgs = build_messages("")
        base = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if force_language:
            base = base + f"language {force_language}<asr_text>"
        return base

    force_lang = None
    if language:
        force_lang = normalize_language_name(language)

    # Stop token IDs
    eos_ids = {151645, 151643}  # <|im_end|>, <|endoftext|>
    if processor.tokenizer.eos_token_id is not None:
        eos_ids.add(processor.tokenizer.eos_token_id)

    # Decoder layers: model.thinker.model.layers[i].self_attn
    decoder_layers = thinker.model.layers

    g = np.zeros(total_heads, dtype=np.int64)
    m = 0
    t0 = time.time()

    for clip_idx, (waveform, transcript) in enumerate(clips):
        if not transcript.strip():
            continue

        audio_duration = len(waveform) / SAMPLE_RATE

        # 1. Get forced alignment timestamps
        try:
            align_results = forced_aligner.align(
                audio=[(waveform, SAMPLE_RATE)],
                text=[transcript],
                language=[force_lang or "English"],
            )
            align_result = align_results[0]
        except Exception as e:
            logger.warning("Forced alignment failed for clip %d: %s", clip_idx, e)
            continue

        if not align_result.items:
            continue

        # Build word -> (start_time, end_time) mapping
        word_timestamps = []
        for item in align_result.items:
            word_timestamps.append((item.text, item.start_time, item.end_time))

        # 2. Prepare inputs
        text_prompt = build_text_prompt(force_language=force_lang)
        inputs = processor(
            text=[text_prompt],
            audio=[waveform],
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(model.device).to(model.dtype)
        prompt_len = inputs.input_ids.shape[1]

        # Find audio token range
        audio_start, audio_end = find_audio_token_range(
            inputs.input_ids[0], audio_token_id,
        )
        n_audio_tokens = audio_end - audio_start

        if n_audio_tokens == 0:
            logger.warning("No audio tokens found in clip %d", clip_idx)
            continue

        # 3. Register forward hooks on self_attn to capture attention weights.
        #    The decoder layer discards them: hidden_states, _ = self.self_attn(...)
        #    but eager_attention_forward always computes and returns attn_weights.
        #    We capture just the argmax over the audio region (memory-efficient).
        #    captured_argmax[layer_idx] = list of (num_heads,) tensors, one per decode step.
        captured_argmax = {i: [] for i in range(num_layers)}

        def _make_hook(store, a_start, a_end):
            def hook_fn(module, args, output):
                # output = (attn_output, attn_weights)
                attn_weights = output[1]
                if attn_weights is None:
                    return
                # attn_weights shape: (batch, num_heads, q_len, kv_len)
                # Only capture decode steps (q_len == 1), skip prefill
                if attn_weights.shape[2] != 1:
                    return
                kv_len = attn_weights.shape[-1]
                if a_end > kv_len:
                    return
                # Attention from the new token over audio region
                audio_attn = attn_weights[0, :, 0, a_start:a_end]  # (num_heads, n_audio)
                store.append(audio_attn.argmax(dim=-1).cpu())  # (num_heads,)
            return hook_fn

        hooks = []
        for layer_idx in range(num_layers):
            h = decoder_layers[layer_idx].self_attn.register_forward_hook(
                _make_hook(captured_argmax[layer_idx], audio_start, audio_end)
            )
            hooks.append(h)

        # 4. Run generation
        try:
            with torch.inference_mode():
                outputs = thinker.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
        except Exception as e:
            for h in hooks:
                h.remove()
            logger.warning("Generation failed for clip %d: %s", clip_idx, e)
            continue
        finally:
            for h in hooks:
                h.remove()

        # outputs is (batch, seq_len) tensor
        all_generated = outputs[0, prompt_len:]
        num_gen = len(all_generated)
        for i, tid in enumerate(all_generated):
            if tid.item() in eos_ids:
                num_gen = i
                break
        generated_ids = all_generated[:num_gen]

        if num_gen == 0:
            del outputs, captured_argmax
            continue

        generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Filter out hallucinated clips (e.g. "!!!" patterns)
        sim = text_similarity(generated_text, transcript)
        if sim < MIN_TEXT_SIMILARITY:
            logger.info(
                "[%d/%d] SKIP (sim=%.2f) | %s...",
                clip_idx + 1, len(clips), sim, generated_text[:60],
            )
            del outputs, captured_argmax
            continue

        # Verify hooks captured data
        n_captured = len(captured_argmax[0])
        if n_captured == 0:
            logger.warning(
                "No attention weights captured for clip %d (hooks may not have fired)", clip_idx
            )
            del outputs, captured_argmax
            continue

        # 5. Map generated tokens to word timestamps
        gen_token_strings = [
            processor.tokenizer.decode([tid.item()]) for tid in generated_ids
        ]

        # Map each generated token index -> forced-aligner word index
        accumulated_text = ""
        word_idx = 0
        token_to_word = {}
        for tok_idx, tok_str in enumerate(gen_token_strings):
            accumulated_text += tok_str
            # Advance word index when accumulated text covers the current word
            while (
                word_idx < len(word_timestamps)
                and len(accumulated_text.strip()) >= sum(
                    len(w[0]) + 1 for w in word_timestamps[:word_idx + 1]
                )
            ):
                word_idx += 1
            actual_word_idx = min(word_idx, len(word_timestamps) - 1)
            token_to_word[tok_idx] = actual_word_idx

        # 6. Score each head using captured argmax data
        for gen_step in range(num_gen):
            word_idx = token_to_word.get(gen_step, None)
            if word_idx is None or word_idx >= len(word_timestamps):
                continue

            _, word_start, word_end = word_timestamps[word_idx]
            word_mid = (word_start + word_end) / 2.0

            # Expected audio token position for this word
            expected_pos = timestamp_to_audio_token_position(
                word_mid, audio_duration, audio_start, audio_end,
            )

            # Tolerance: +/- a few audio tokens (proportional to word duration)
            word_dur_tokens = max(1, int(
                (word_end - word_start) / audio_duration * n_audio_tokens / 2
            ))
            tolerance = max(3, word_dur_tokens)

            m += 1

            for layer_idx in range(num_layers):
                if gen_step >= len(captured_argmax[layer_idx]):
                    continue
                argmaxes = captured_argmax[layer_idx][gen_step].numpy()  # (num_heads,)

                for head_idx in range(num_heads):
                    attended_pos = argmaxes[head_idx]  # relative to audio_start
                    attended_abs = audio_start + attended_pos
                    if abs(attended_abs - expected_pos) <= tolerance:
                        g[layer_idx * num_heads + head_idx] += 1

        del outputs, captured_argmax
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        avg = elapsed / (clip_idx + 1)
        eta = avg * (len(clips) - clip_idx - 1)
        logger.info(
            "[%d/%d] m=%d | %s... | %.1fs/clip | ETA: %.0fs",
            clip_idx + 1, len(clips), m,
            generated_text[:60], avg, eta,
        )

    return g, m


def main():
    parser = argparse.ArgumentParser(
        description="Detect alignment heads in Qwen3-ASR for SimulStreaming"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-ASR-1.7B",
        help="Qwen3-ASR model name or path",
    )
    parser.add_argument(
        "--dataset", type=str, default="librispeech_asr",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset-config", type=str, default="clean",
        help="Dataset config/subset",
    )
    parser.add_argument(
        "--dataset-split", type=str, default="validation",
        help="Dataset split",
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, default=50,
        help="Number of audio samples to process",
    )
    parser.add_argument(
        "--language", type=str, default="English",
        help="Language for forced alignment",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16",
        choices=["float32", "bf16", "float16"],
        help="Model dtype",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="alignment_heads_qwen3_asr.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--heatmap", type=str, default="alignment_heads_qwen3_asr.png",
        help="Output heatmap image",
    )
    parser.add_argument(
        "--threshold", type=float, default=TS_THRESHOLD,
        help="Minimum alignment score threshold",
    )
    args = parser.parse_args()

    device = get_device()

    dtype_map = {
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    # Load model
    model, processor, forced_aligner = load_qwen3_asr(args.model, device, dtype)

    # Load data
    logger.info("Loading dataset: %s/%s [%s]", args.dataset, args.dataset_config, args.dataset_split)
    clips = load_dataset_clips(
        args.dataset, args.dataset_config, args.dataset_split, args.num_samples,
    )
    logger.info("Loaded %d clips", len(clips))

    # Run detection
    g, m = run_detection(model, processor, forced_aligner, clips, args.language, device)

    # Compute alignment scores
    thinker = model.thinker
    text_config = thinker.config.text_config
    num_layers = text_config.num_hidden_layers
    num_heads = text_config.num_attention_heads

    ts = g / max(m, 1)
    ts_matrix = ts.reshape(num_layers, num_heads)

    # Identify alignment heads
    tah = []
    for l in range(num_layers):
        for h in range(num_heads):
            score = ts_matrix[l, h]
            if score > args.threshold:
                tah.append({"layer": l, "head": h, "ts": round(float(score), 4)})

    tah.sort(key=lambda x: x["ts"], reverse=True)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"ALIGNMENT HEADS (TS > {args.threshold}): {len(tah)} / {num_layers * num_heads}")
    print(f"{'=' * 60}")
    for entry in tah:
        bar = "#" * int(entry["ts"] * 50)
        print(f"  L{entry['layer']:2d} H{entry['head']:2d} : TS={entry['ts']:.4f}  {bar}")

    n_active = sum(1 for s in ts if s > args.threshold)
    n_low = sum(1 for s in ts if 0 < s <= args.threshold)
    n_zero = sum(1 for s in ts if s == 0)
    total_heads = num_layers * num_heads
    print(f"\nDistribution:")
    print(f"  TS > {args.threshold} (alignment heads): {n_active} ({100 * n_active / total_heads:.1f}%)")
    print(f"  0 < TS <= {args.threshold} (low activity): {n_low} ({100 * n_low / total_heads:.1f}%)")
    print(f"  TS = 0 (inactive):               {n_zero} ({100 * n_zero / total_heads:.1f}%)")
    print(f"\nTotal alignable tokens checked: m={m}")

    # Save JSON
    output = {
        "model": args.model,
        "language": args.language,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": text_config.num_key_value_heads,
        "num_samples": len(clips),
        "total_alignable_tokens": int(m),
        "ts_threshold": args.threshold,
        "ts_matrix": ts_matrix.tolist(),
        "alignment_heads": tah,
        # WhisperLiveKit-compatible format: list of [layer, head] pairs
        "alignment_heads_compact": [[e["layer"], e["head"]] for e in tah],
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", args.output)

    # Generate heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(max(10, num_heads * 0.6), max(8, num_layers * 0.35)),
        )
        im = ax.imshow(
            ts_matrix,
            aspect="auto",
            cmap="RdYlBu_r",
            vmin=0,
            vmax=max(0.4, ts_matrix.max()),
            interpolation="nearest",
        )
        ax.set_xlabel("Head ID", fontsize=12)
        ax.set_ylabel("Layer", fontsize=12)
        ax.set_title(
            f"Alignment Scores - {args.model}\n"
            f"{len(tah)} alignment heads (TS > {args.threshold}), n={len(clips)}",
            fontsize=13,
        )
        ax.set_xticks(range(num_heads))
        ax.set_yticks(range(num_layers))
        plt.colorbar(im, ax=ax, label="Alignment Score", shrink=0.8)

        for entry in tah:
            ax.add_patch(plt.Rectangle(
                (entry["head"] - 0.5, entry["layer"] - 0.5),
                1, 1, fill=False, edgecolor="red", linewidth=1.5,
            ))

        plt.tight_layout()
        plt.savefig(args.heatmap, dpi=150)
        logger.info("Heatmap saved to %s", args.heatmap)
    except Exception as e:
        logger.warning("Could not generate heatmap: %s", e)


if __name__ == "__main__":
    main()
