"""
Qwen3-ASR SimulStreaming (AlignAtt) on MLX for Apple Silicon.

Uses the ``mlx_qwen3_asr`` library for model loading, audio encoding, and
tokenization.  Implements the AlignAtt border-distance policy by monkey-
patching ``TextAttention.__call__`` on alignment layers to capture Q (with
RoPE) during autoregressive decode steps, then computing ``Q @ K_audio^T``
from the KV cache to find the most-attended audio frame.

This is the MLX equivalent of ``qwen3_simul.py`` (PyTorch) which uses
``register_forward_hook`` for the same purpose.
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from whisperlivekit.timed_objects import ASRToken, Transcript

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

# Model size aliases (same as qwen3_mlx_asr.py)
QWEN3_MLX_MODEL_MAPPING = {
    "base": "Qwen/Qwen3-ASR-0.6B",
    "tiny": "Qwen/Qwen3-ASR-0.6B",
    "small": "Qwen/Qwen3-ASR-0.6B",
    "large": "Qwen/Qwen3-ASR-1.7B",
    "medium": "Qwen/Qwen3-ASR-1.7B",
    "large-v3": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "1.7b": "Qwen/Qwen3-ASR-1.7B",
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
}

# Whisper language codes -> Qwen3 canonical language names
WHISPER_TO_QWEN3_LANGUAGE = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese",
    "ar": "Arabic", "de": "German", "fr": "French", "es": "Spanish",
    "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
    "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
    "ja": "Japanese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay",
    "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish",
    "pl": "Polish", "cs": "Czech", "fa": "Persian",
    "el": "Greek", "hu": "Hungarian", "mk": "Macedonian", "ro": "Romanian",
}

QWEN3_TO_WHISPER_LANGUAGE = {v: k for k, v in WHISPER_TO_QWEN3_LANGUAGE.items()}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Qwen3MLXSimulConfig:
    language: str = "auto"
    alignment_heads_path: Optional[str] = None
    border_fraction: float = 0.15
    rewind_fraction: float = 0.12
    audio_min_len: float = 0.5
    audio_max_len: float = 15.0
    max_context_tokens: int = 30
    max_alignment_heads: int = 20


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------


@dataclass
class _SessionState:
    audio_buffer: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    cumulative_time_offset: float = 0.0
    global_time_offset: float = 0.0
    speaker: int = -1

    last_attend_frame: int = -15
    committed_word_count: int = 0
    committed_token_ids: List[int] = field(default_factory=list)
    detected_language: Optional[str] = None
    last_infer_samples: int = 0


# ---------------------------------------------------------------------------
# Shared model holder
# ---------------------------------------------------------------------------


class Qwen3MLXSimulStreamingASR:
    """Loads the Qwen3-ASR model via ``mlx_qwen3_asr`` once and keeps it
    alive for the lifetime of the server.  Shared across sessions."""

    sep = ""
    SAMPLING_RATE = SAMPLE_RATE

    def __init__(
        self,
        model_size: str = None,
        model_dir: str = None,
        model_path: str = None,
        lan: str = "auto",
        alignment_heads_path: Optional[str] = None,
        border_fraction: float = 0.15,
        warmup_file: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        lora_path: Optional[str] = None,
        min_chunk_size: float = 0.1,
        direct_english_translation: bool = False,
        **kwargs,
    ):
        import mlx.core as mx
        import mlx_qwen3_asr

        self.transcribe_kargs = {}
        self.original_language = None if lan == "auto" else lan
        self.warmup_file = warmup_file

        self.cfg = Qwen3MLXSimulConfig(
            language=lan,
            alignment_heads_path=alignment_heads_path,
            border_fraction=border_fraction,
        )

        # Resolve model path
        resolved = model_dir or model_path
        if not resolved:
            size = (model_size or "base").lower()
            if "/" in size or size.startswith("."):
                resolved = size
            else:
                resolved = QWEN3_MLX_MODEL_MAPPING.get(size, "Qwen/Qwen3-ASR-0.6B")

        t0 = time.time()
        logger.info("Loading Qwen3-ASR MLX model '%s' for SimulStreaming ...", resolved)
        self.model, self._config = mlx_qwen3_asr.load_model(resolved, dtype=mx.float16)
        logger.info("Model loaded in %.2fs", time.time() - t0)

        # Tokenizer
        tok_path = getattr(self.model, "_resolved_model_path", None) or resolved
        self.tokenizer = mlx_qwen3_asr.tokenizer.Tokenizer(str(tok_path))

        # Architecture info
        text_cfg = self._config.text_config
        self.num_layers = text_cfg.num_hidden_layers
        self.num_heads = text_cfg.num_attention_heads
        self.num_kv_heads = text_cfg.num_key_value_heads
        self.head_dim = text_cfg.head_dim
        self.gqa_ratio = self.num_heads // self.num_kv_heads
        self.audio_token_id = self._config.audio_token_id

        logger.info(
            "Qwen3-ASR arch: %d layers x %d heads (%d kv), head_dim=%d, GQA=%d",
            self.num_layers, self.num_heads, self.num_kv_heads,
            self.head_dim, self.gqa_ratio,
        )

        # Alignment heads
        self.alignment_heads = self._load_alignment_heads(alignment_heads_path)
        self.heads_by_layer = {}
        for layer_idx, head_idx in self.alignment_heads:
            self.heads_by_layer.setdefault(layer_idx, []).append(head_idx)

        self.backend_choice = "qwen3-mlx-simul"

        # Warmup
        if warmup_file:
            from whisperlivekit.warmup import load_file
            audio = load_file(warmup_file)
            if audio is not None:
                self._warmup(audio)

    def _load_alignment_heads(
        self, path: Optional[str],
    ) -> List[Tuple[int, int]]:
        max_heads = self.cfg.max_alignment_heads

        if path and Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            all_heads = [tuple(h) for h in data["alignment_heads_compact"]]
            heads = all_heads[:max_heads]
            logger.info(
                "Loaded top %d alignment heads from %s (of %d total)",
                len(heads), path, len(all_heads),
            )
            return heads

        # Default heuristic: last quarter of layers, all heads
        default_heads = []
        start_layer = self.num_layers * 3 // 4
        for layer in range(start_layer, self.num_layers):
            for head in range(self.num_heads):
                default_heads.append((layer, head))
        logger.warning(
            "No alignment heads file. Using default heuristic: "
            "%d heads from layers %d-%d.",
            len(default_heads), start_layer, self.num_layers - 1,
        )
        return default_heads[:max_heads]

    def _warmup(self, audio: np.ndarray):
        import mlx.core as mx
        try:
            from mlx_qwen3_asr.audio import compute_features
            audio = audio[:SAMPLE_RATE * 2]
            mel, feat_lens = compute_features(audio)
            mel = mel.astype(mx.float16)
            audio_features, _ = self.model.audio_tower(mel, feat_lens)
            n_audio = int(audio_features.shape[1])
            prompt = self.tokenizer.build_prompt_tokens(n_audio, language="English")
            input_ids = mx.array([prompt])
            positions = mx.arange(input_ids.shape[1])[None, :]
            position_ids = mx.stack([positions, positions, positions], axis=1)
            cache = self.model.create_cache()
            logits = self.model.prefill(input_ids, audio_features, position_ids, cache)
            mx.eval(logits)
            logger.info("Qwen3 MLX SimulStreaming warmup complete")
        except Exception as e:
            logger.warning("Warmup failed: %s", e)

    def transcribe(self, audio):
        pass  # all work in the online processor


# ---------------------------------------------------------------------------
# Attention capture via wrapper replacement
# ---------------------------------------------------------------------------


class _AttnCaptureWrapper:
    """Wraps a TextAttention module to capture alignment scores during decode.

    Replaces ``layer.self_attn`` with this wrapper.  On decode steps (L=1),
    recomputes Q with RoPE, reads cached K from the audio region, computes
    ``Q @ K_audio^T`` for alignment heads, and stores the argmax frame in
    ``capture["step_frames"]``.

    Python dunder resolution (``__call__``) goes through the *class*, not the
    instance, so monkey-patching ``attn.__call__`` on an ``nn.Module`` does
    not work.  This wrapper class defines its own ``__call__`` and delegates
    everything else to the wrapped module via ``__getattr__``.
    """

    def __init__(self, original, layer_idx, head_indices, gqa_ratio,
                 audio_start, audio_end, capture):
        # Store in __dict__ directly to avoid triggering __getattr__
        self.__dict__["_original"] = original
        self.__dict__["_layer_idx"] = layer_idx
        self.__dict__["_head_indices"] = head_indices
        self.__dict__["_gqa_ratio"] = gqa_ratio
        self.__dict__["_audio_start"] = audio_start
        self.__dict__["_audio_end"] = audio_end
        self.__dict__["_capture"] = capture

    def __call__(self, x, cos, sin, mask=None, cache=None, layer_idx=0):
        import mlx.core as mx
        from mlx_qwen3_asr.mrope import apply_rotary_pos_emb

        orig = self.__dict__["_original"]
        B, L, _ = x.shape

        if L == 1 and cache is not None:
            li = self.__dict__["_layer_idx"]
            h_indices = self.__dict__["_head_indices"]
            gqa = self.__dict__["_gqa_ratio"]
            a_start = self.__dict__["_audio_start"]
            a_end = self.__dict__["_audio_end"]
            cap = self.__dict__["_capture"]

            # Recompute Q with RoPE (cheap: single token)
            q = orig.q_proj(x)
            q = q.reshape(B, L, orig.num_heads, orig.head_dim)
            q = orig.q_norm(q)
            q = q.transpose(0, 2, 1, 3)  # (B, H, 1, D)
            q_rope, _ = apply_rotary_pos_emb(q, q, cos, sin)

            # K from cache (already has RoPE baked in from cache.update)
            k_cached = cache.keys[li]
            if k_cached is not None and a_end <= k_cached.shape[2]:
                for h_idx in h_indices:
                    kv_h = h_idx // gqa
                    q_h = q_rope[0, h_idx, 0]           # (head_dim,)
                    k_audio = k_cached[0, kv_h, a_start:a_end]  # (n_audio, D)
                    scores = k_audio @ q_h               # (n_audio,)
                    frame = int(mx.argmax(scores).item())
                    cap["step_frames"].append(frame)

        return orig(x, cos, sin, mask=mask, cache=cache, layer_idx=layer_idx)

    def __getattr__(self, name):
        return getattr(self.__dict__["_original"], name)


def _install_alignment_hooks(model, heads_by_layer, gqa_ratio, audio_start, audio_end, capture):
    """Replace ``self_attn`` on alignment layers with capture wrappers.

    Returns a list of ``(layer_idx, original_attn)`` for later restoration.
    """
    originals = []
    for layer_idx, head_indices in heads_by_layer.items():
        if layer_idx >= len(model.model.layers):
            continue
        layer = model.model.layers[layer_idx]
        orig_attn = layer.self_attn
        wrapper = _AttnCaptureWrapper(
            orig_attn, layer_idx, head_indices, gqa_ratio,
            audio_start, audio_end, capture,
        )
        layer.self_attn = wrapper
        originals.append((layer_idx, orig_attn))
    return originals


def _remove_alignment_hooks(model, originals):
    """Restore original self_attn modules."""
    for layer_idx, orig_attn in originals:
        model.model.layers[layer_idx].self_attn = orig_attn


# ---------------------------------------------------------------------------
# Per-session online processor
# ---------------------------------------------------------------------------


class Qwen3MLXSimulStreamingOnlineProcessor:
    """Per-session processor implementing AlignAtt on MLX.

    Same interface as other online processors:
    insert_audio_chunk / process_iter / get_buffer / start_silence /
    end_silence / finish / warmup / new_speaker.
    """

    SAMPLING_RATE = SAMPLE_RATE
    MIN_DURATION_REAL_SILENCE = 5

    def __init__(self, asr: Qwen3MLXSimulStreamingASR, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.end = 0.0
        self.buffer: List[ASRToken] = []
        self.state = _SessionState()

    # -- properties expected by AudioProcessor --

    @property
    def speaker(self):
        return self.state.speaker

    @speaker.setter
    def speaker(self, value):
        self.state.speaker = value

    @property
    def global_time_offset(self):
        return self.state.global_time_offset

    @global_time_offset.setter
    def global_time_offset(self, value):
        self.state.global_time_offset = value

    # -- audio ingestion --

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        self.end = audio_stream_end_time
        self.state.audio_buffer = np.append(self.state.audio_buffer, audio)

        # Trim if too long
        max_samples = int(self.asr.cfg.audio_max_len * self.SAMPLING_RATE)
        if len(self.state.audio_buffer) > max_samples:
            trim = len(self.state.audio_buffer) - max_samples
            self.state.audio_buffer = self.state.audio_buffer[trim:]
            self.state.cumulative_time_offset += trim / self.SAMPLING_RATE
            self.state.last_infer_samples = max(0, self.state.last_infer_samples - trim)

    # -- main processing --

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        audio_duration = len(self.state.audio_buffer) / self.SAMPLING_RATE
        if audio_duration < self.asr.cfg.audio_min_len:
            return [], self.end

        # Throttle: at least 1s of new audio
        new_samples = len(self.state.audio_buffer) - self.state.last_infer_samples
        if not is_last and new_samples < int(1.0 * self.SAMPLING_RATE):
            return [], self.end

        self.state.last_infer_samples = len(self.state.audio_buffer)

        try:
            words = self._infer(is_last)
        except Exception as e:
            logger.exception("Qwen3 MLX SimulStreaming inference error: %s", e)
            return [], self.end

        if not words:
            return [], self.end

        self.buffer = []
        return words, self.end

    def _infer(self, is_last: bool) -> List[ASRToken]:
        """Run one inference cycle with alignment-head-based stopping."""
        import mlx.core as mx
        from mlx_qwen3_asr.audio import compute_features
        from mlx_qwen3_asr.generate import _detect_repetition

        asr = self.asr
        state = self.state
        model = asr.model

        # 1. Encode audio
        mel, feat_lens = compute_features(state.audio_buffer)
        mel = mel.astype(mx.float16)
        audio_features, _ = model.audio_tower(mel, feat_lens)
        n_audio_tokens = int(audio_features.shape[1])
        mx.eval(audio_features)

        if n_audio_tokens == 0:
            return []

        audio_duration = len(state.audio_buffer) / self.SAMPLING_RATE

        # 2. Build prompt tokens
        lan = asr.cfg.language
        language = None
        if lan and lan != "auto":
            language = WHISPER_TO_QWEN3_LANGUAGE.get(lan, lan)

        prompt_tokens = asr.tokenizer.build_prompt_tokens(
            n_audio_tokens=n_audio_tokens,
            language=language,
        )

        # Append committed context tokens
        if state.committed_token_ids:
            ctx = state.committed_token_ids[-asr.cfg.max_context_tokens:]
            prompt_tokens.extend(ctx)

        input_ids = mx.array([prompt_tokens])
        seq_len = input_ids.shape[1]

        # 3. Find audio token range
        audio_positions = [
            i for i, t in enumerate(prompt_tokens) if t == asr.audio_token_id
        ]
        if not audio_positions:
            return []
        audio_start = audio_positions[0]
        audio_end = audio_positions[-1] + 1

        # 4. MRoPE position IDs
        positions = mx.arange(seq_len, dtype=mx.int32)[None, :]
        position_ids = mx.stack([positions, positions, positions], axis=1)

        # 5. Prefill
        cache = model.create_cache(max_seq_len=seq_len + 120)
        logits = model.prefill(input_ids, audio_features, position_ids, cache)
        mx.eval(logits)

        # 6. Install alignment hooks
        capture = {"step_frames": []}
        originals = _install_alignment_hooks(
            model, asr.heads_by_layer, asr.gqa_ratio,
            audio_start, audio_end, capture,
        )

        # 7. Decode loop with border-distance policy
        eos_ids = set(asr.tokenizer.EOS_TOKEN_IDS)
        per_step_frames: List[List[int]] = []
        last_attend_frame = state.last_attend_frame
        border_stop_step: Optional[int] = None

        border_threshold = max(2, int(n_audio_tokens * asr.cfg.border_fraction))
        rewind_threshold = max(2, int(n_audio_tokens * asr.cfg.rewind_fraction))

        # Max tokens: ~6 tokens/sec of speech + margin
        new_audio_secs = (len(state.audio_buffer) - state.last_infer_samples) / self.SAMPLING_RATE
        if is_last:
            max_tokens = min(int(audio_duration * 6) + 10, 120)
        else:
            max_tokens = min(int(max(new_audio_secs, 1.0) * 6) + 5, 40)

        token = int(mx.argmax(logits.reshape(-1)).item())
        generated = [token]

        try:
            for step in range(1, max_tokens):
                if token in eos_ids:
                    break
                if _detect_repetition(generated):
                    break

                next_ids = mx.array([[token]])
                pos_val = seq_len + step - 1
                next_pos = mx.array([[[pos_val], [pos_val], [pos_val]]], dtype=mx.int32)
                logits = model.step(next_ids, next_pos, cache, validate_input_ids=False)
                mx.eval(logits)

                token = int(mx.argmax(logits.reshape(-1)).item())
                generated.append(token)

                # Collect frames from this step
                if capture["step_frames"]:
                    per_step_frames.append(capture["step_frames"])
                    capture["step_frames"] = []

                    # Border-distance check (skip first 3 steps)
                    if (not is_last
                            and border_stop_step is None
                            and len(per_step_frames) >= 3):
                        latest = per_step_frames[-1]
                        if latest:
                            frames_sorted = sorted(latest)
                            attended = frames_sorted[len(frames_sorted) // 2]

                            # Rewind check
                            if last_attend_frame - attended > rewind_threshold:
                                border_stop_step = max(0, len(per_step_frames) - 2)
                                break

                            last_attend_frame = attended

                            # Border check
                            if (n_audio_tokens - attended) <= border_threshold:
                                border_stop_step = len(per_step_frames) - 1
                                break

                # Periodic eval to prevent graph buildup
                if step % 8 == 0:
                    mx.eval(cache.keys[-1])
        finally:
            _remove_alignment_hooks(model, originals)
            # Flush remaining frames
            if capture["step_frames"]:
                per_step_frames.append(capture["step_frames"])

        state.last_attend_frame = last_attend_frame

        # 8. Process generated tokens
        # Remove trailing EOS
        while generated and generated[-1] in eos_ids:
            generated.pop()

        num_gen = len(generated)
        if num_gen == 0:
            return []

        raw_text = asr.tokenizer.decode(generated)
        logger.info(
            "SimulStreaming raw: %d tokens (border_stop=%s), text=%r",
            num_gen, border_stop_step, raw_text[:100],
        )

        # 9. Strip metadata prefix ("language English<asr_text>...")
        from mlx_qwen3_asr.tokenizer import parse_asr_output
        detected_lang, clean_text = parse_asr_output(
            raw_text,
            user_language=language,
        )

        # Find how many tokens to skip for metadata
        metadata_offset = 0
        asr_text_tokens = asr.tokenizer.encode("<asr_text>")
        asr_text_id = asr_text_tokens[0] if asr_text_tokens else None
        if asr_text_id is not None:
            for i in range(min(num_gen, 10)):
                if generated[i] == asr_text_id:
                    metadata_offset = i + 1
                    break

        if metadata_offset > 0:
            generated = generated[metadata_offset:]
            num_gen -= metadata_offset
            per_step_frames = per_step_frames[metadata_offset:]

        if num_gen <= 0:
            return []

        # Detect language
        if state.detected_language is None and detected_lang and detected_lang != "unknown":
            state.detected_language = QWEN3_TO_WHISPER_LANGUAGE.get(
                detected_lang, detected_lang.lower(),
            )
            logger.info("Auto-detected language: %s", state.detected_language)

        # 10. Determine how many tokens to emit
        step_frames = [f for f in per_step_frames if f]
        if border_stop_step is not None:
            emit_up_to = min(border_stop_step, num_gen)
        else:
            emit_up_to = num_gen

        if emit_up_to <= 0:
            return []

        emitted_ids = generated[:emit_up_to]

        # 11. Build timestamped words
        words = self._build_timestamped_words(
            emitted_ids, step_frames, emit_up_to,
            n_audio_tokens, audio_duration,
        )

        # Update state
        state.committed_word_count += len(words)
        state.committed_token_ids.extend(emitted_ids)

        return words

    def _build_timestamped_words(
        self,
        generated_ids: List[int],
        step_frames: List[List[int]],
        emit_up_to: int,
        n_audio_tokens: int,
        audio_duration: float,
    ) -> List[ASRToken]:
        """Build timestamped ASRToken list from generated tokens and
        alignment-head captured frames."""
        state = self.state
        asr = self.asr

        # Per-token attended frame (median of head votes)
        per_token_frame: List[Optional[int]] = []
        for step_idx in range(emit_up_to):
            if step_idx < len(step_frames) and step_frames[step_idx]:
                frames = sorted(step_frames[step_idx])
                per_token_frame.append(frames[len(frames) // 2])
            else:
                per_token_frame.append(None)

        # Decode full text, split into words
        full_text = asr.tokenizer.decode(generated_ids[:emit_up_to])
        text_words = full_text.split()

        # Map words to frames proportionally
        all_frames = [f for f in per_token_frame if f is not None]
        word_frame_pairs = []
        for wi, word in enumerate(text_words):
            if all_frames:
                frac = wi / max(len(text_words), 1)
                frame_idx = min(int(frac * len(all_frames)), len(all_frames) - 1)
                frame = all_frames[frame_idx]
            else:
                frame = None
            word_frame_pairs.append((word, frame))

        # Convert to ASRToken
        tokens = []
        for i, (text, frame) in enumerate(word_frame_pairs):
            text = text.strip()
            if not text:
                continue

            if frame is not None and n_audio_tokens > 0:
                timestamp = (
                    frame / n_audio_tokens * audio_duration
                    + state.cumulative_time_offset
                )
            else:
                timestamp = (
                    (i / max(len(word_frame_pairs), 1)) * audio_duration
                    + state.cumulative_time_offset
                )

            is_very_first_word = (i == 0 and state.committed_word_count == 0)
            display_text = text if is_very_first_word else " " + text

            token = ASRToken(
                start=round(timestamp, 2),
                end=round(timestamp + 0.1, 2),
                text=display_text,
                speaker=state.speaker,
                detected_language=state.detected_language,
            ).with_offset(state.global_time_offset)
            tokens.append(token)

        return tokens

    # -- silence / speaker / lifecycle --

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        all_tokens = []
        for _ in range(5):
            tokens, _ = self.process_iter(is_last=True)
            if not tokens:
                break
            all_tokens.extend(tokens)
        return all_tokens, self.end

    def end_silence(self, silence_duration: float, offset: float):
        self.end += silence_duration
        long_silence = silence_duration >= self.MIN_DURATION_REAL_SILENCE
        if not long_silence:
            gap_len = int(self.SAMPLING_RATE * silence_duration)
            if gap_len > 0:
                gap_silence = np.zeros(gap_len, dtype=np.float32)
                self.state.audio_buffer = np.append(
                    self.state.audio_buffer, gap_silence,
                )
        else:
            self.state = _SessionState()
            self.state.global_time_offset = silence_duration + offset

    def new_speaker(self, change_speaker):
        self.process_iter(is_last=True)
        self.state = _SessionState()
        self.state.speaker = change_speaker.speaker
        self.state.global_time_offset = change_speaker.start

    def get_buffer(self) -> Transcript:
        return Transcript.from_tokens(tokens=self.buffer, sep='')

    def warmup(self, audio: np.ndarray, init_prompt: str = ""):
        try:
            self.state.audio_buffer = audio[:SAMPLE_RATE]
            self.process_iter(is_last=True)
            self.state = _SessionState()
            logger.info("Qwen3 MLX SimulStreaming processor warmed up")
        except Exception as e:
            logger.warning("Warmup failed: %s", e)
            self.state = _SessionState()

    def finish(self) -> Tuple[List[ASRToken], float]:
        all_tokens = []
        for _ in range(5):
            tokens, _ = self.process_iter(is_last=True)
            if not tokens:
                break
            all_tokens.extend(tokens)
        return all_tokens, self.end
