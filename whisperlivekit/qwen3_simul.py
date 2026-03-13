"""
SimulStreaming-style online processor for Qwen3-ASR.

Architecture overview
---------------------
Qwen3-ASR is a decoder-only multimodal model.  Audio is encoded by an audio
encoder (Whisper-style) into a sequence of embeddings that replace <|audio_pad|>
placeholder tokens in the input sequence.  The text decoder then uses causal
self-attention over the combined audio + text tokens.

Unlike Whisper (which has explicit cross-attention between decoder and encoder),
Qwen3-ASR uses self-attention where generated text tokens attend to earlier
audio tokens and previously generated text.  This means "alignment heads" here
are self-attention heads whose attention over the *audio-token region* tracks
the monotonic audio-to-text alignment.

The border-distance policy works as follows:
  - After each generated token, extract the attention weights from the
    selected alignment heads, restricted to the audio-token region
  - Find which audio frame each head attends to most strongly (argmax)
  - If the most-attended audio frame is approaching the end of the available
    audio, pause generation and wait for more audio
  - If the most-attended frame jumps backward (rewind), discard recent tokens

This module loads the Qwen3-ASR model *directly* via transformers (not through
the qwen_asr package's Qwen3ASRModel wrapper), giving us full control over
forward passes, KV caches, and attention extraction.

Requires:
  - A pre-computed alignment heads JSON file (from detect_alignment_heads_qwen3.py)
  - OR will fall back to all heads in a configurable set of layers
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from whisperlivekit.timed_objects import ASRToken, ChangeSpeaker, Transcript

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class Qwen3SimulConfig:
    """Configuration for Qwen3 SimulStreaming."""
    model_id: str = "Qwen/Qwen3-ASR-1.7B"
    alignment_heads_path: Optional[str] = None
    language: str = "auto"
    # Border/rewind thresholds as fraction of audio tokens (not absolute frames).
    # Qwen3 has ~13 audio tokens/sec vs Whisper's ~50, so absolute thresholds
    # don't transfer. 0.15 = pause when attention is within last 15% of audio.
    border_fraction: float = 0.15  # Fraction of audio tokens from end to trigger pause
    rewind_fraction: float = 0.12  # Max backward jump as fraction of audio tokens
    audio_min_len: float = 0.5  # Minimum audio length before starting decode
    audio_max_len: float = 15.0  # Maximum audio buffer length in seconds
    max_context_tokens: int = 30  # Max committed tokens to include as context
    init_prompt: Optional[str] = None
    max_alignment_heads: int = 20  # Use only top N alignment heads


@dataclass
class Qwen3SimulState:
    """Per-session mutable state for Qwen3 SimulStreaming."""
    # Audio
    audio_buffer: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    cumulative_time_offset: float = 0.0
    global_time_offset: float = 0.0
    speaker: int = -1

    # Decode state
    last_attend_frame: int = -15
    generated_tokens: List[int] = field(default_factory=list)
    committed_text: str = ""
    committed_word_count: int = 0  # How many words already emitted
    committed_token_ids: List[int] = field(default_factory=list)  # token IDs for prompt context

    # Tracking
    first_timestamp: Optional[float] = None
    detected_language: Optional[str] = None
    last_infer_samples: int = 0  # audio_buffer length at last inference


class Qwen3SimulStreamingASR:
    """
    Shared backend for Qwen3-ASR SimulStreaming.

    Loads the model once and is shared across sessions.  Each session gets
    its own Qwen3SimulStreamingOnlineProcessor with independent state.
    """

    sep = ""

    def __init__(
        self,
        model_size: str = None,
        model_dir: str = None,
        lan: str = "auto",
        alignment_heads_path: Optional[str] = None,
        border_fraction: float = 0.15,
        min_chunk_size: float = 0.1,
        warmup_file: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        lora_path: Optional[str] = None,
        direct_english_translation: bool = False,
        **kwargs,
    ):
        self.transcribe_kargs = {}
        self.original_language = None if lan == "auto" else lan
        self.warmup_file = warmup_file

        self.cfg = Qwen3SimulConfig(
            language=lan,
            alignment_heads_path=alignment_heads_path,
            border_fraction=border_fraction,
        )

        # Load model directly via transformers
        self._load_model(model_size, model_dir, model_cache_dir, model_path)

        # Load alignment heads
        self.alignment_heads = self._load_alignment_heads(alignment_heads_path)

        # Warmup
        if warmup_file:
            from whisperlivekit.warmup import load_file
            audio = load_file(warmup_file)
            if audio is not None:
                logger.info("Warming up Qwen3 SimulStreaming model")
                # Simple warmup: just encode a short audio
                self._warmup(audio)

    def _load_model(self, model_size, model_dir, model_cache_dir, model_path):
        """Load Qwen3-ASR via transformers (SDPA attention for speed)."""
        from whisperlivekit.qwen3_asr import (
            QWEN3_MODEL_MAPPING,
            _patch_transformers_compat,
        )
        _patch_transformers_compat()

        from qwen_asr.core.transformers_backend import (
            Qwen3ASRConfig,
            Qwen3ASRForConditionalGeneration,
            Qwen3ASRProcessor,
        )
        from transformers import AutoConfig, AutoModel, AutoProcessor

        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
        AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

        if model_dir:
            model_id = model_dir
        elif model_path:
            model_id = model_path
        elif model_size:
            model_id = QWEN3_MODEL_MAPPING.get(model_size.lower(), model_size)
        else:
            model_id = "Qwen/Qwen3-ASR-1.7B"

        if torch.cuda.is_available():
            dtype, device = torch.bfloat16, "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dtype, device = torch.float32, "mps"
        else:
            dtype, device = torch.float32, "cpu"

        logger.info("Loading Qwen3-ASR for SimulStreaming: %s (sdpa attention)", model_id)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id, fix_mistral_regex=True)

        # Cache model properties
        thinker = self.model.thinker
        text_config = thinker.config.text_config
        self.num_layers = text_config.num_hidden_layers
        self.num_heads = text_config.num_attention_heads
        self.num_kv_heads = text_config.num_key_value_heads
        self.audio_token_id = thinker.config.audio_token_id
        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype

        # Cache special token IDs for metadata stripping
        self.asr_text_token_id = self.processor.tokenizer.convert_tokens_to_ids("<asr_text>")

        logger.info(
            "Qwen3-ASR loaded: %d layers x %d heads, device=%s, <asr_text> id=%d",
            self.num_layers, self.num_heads, self.device, self.asr_text_token_id,
        )

    def _load_alignment_heads(
        self, path: Optional[str],
    ) -> List[Tuple[int, int]]:
        """Load alignment heads from JSON or use defaults.

        Only loads the top N heads (sorted by TS score) for efficiency.
        The Qwen3-ASR model has alignment info spread across most heads
        (decoder-only, no cross-attention), so we pick the strongest ones.
        """
        max_heads = self.cfg.max_alignment_heads

        if path and Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            # alignment_heads_compact is pre-sorted by TS score (descending)
            all_heads = [tuple(h) for h in data["alignment_heads_compact"]]
            heads = all_heads[:max_heads]
            logger.info(
                "Loaded top %d alignment heads from %s (of %d total)",
                len(heads), path, len(all_heads),
            )
            return heads

        # Default: use heads from the last quarter of layers
        default_heads = []
        start_layer = self.num_layers * 3 // 4
        for layer in range(start_layer, self.num_layers):
            for head in range(self.num_heads):
                default_heads.append((layer, head))
        logger.warning(
            "No alignment heads file found. Using default heuristic: "
            "%d heads from layers %d-%d. Run detect_alignment_heads_qwen3.py "
            "to find optimal heads.",
            len(default_heads), start_layer, self.num_layers - 1,
        )
        return default_heads[:max_heads]

    def _warmup(self, audio: np.ndarray):
        """Run a short inference to warmup the model."""
        try:
            audio = audio[:SAMPLE_RATE * 2]  # Max 2 seconds
            msgs = [
                {"role": "system", "content": ""},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]},
            ]
            text_prompt = self.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
            )
            inputs = self.processor(
                text=[text_prompt],
                audio=[audio],
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self.device).to(self.dtype)

            with torch.inference_mode():
                self.model.thinker.generate(
                    **inputs, max_new_tokens=5, do_sample=False,
                )
            logger.info("Qwen3 SimulStreaming warmup complete")
        except Exception as e:
            logger.warning("Warmup failed: %s", e)

    def transcribe(self, audio):
        """No-op -- SimulStreaming uses the online processor directly."""
        pass


class Qwen3SimulStreamingOnlineProcessor:
    """
    Per-session online processor for Qwen3-ASR SimulStreaming.

    Implements the same interface as SimulStreamingOnlineProcessor:
    - insert_audio_chunk(audio, time)
    - process_iter(is_last=False) -> (List[ASRToken], float)
    - get_buffer() -> Transcript
    - start_silence() -> (List[ASRToken], float)
    - end_silence(duration, offset)
    - finish() -> (List[ASRToken], float)
    """

    SAMPLING_RATE = 16000
    MIN_DURATION_REAL_SILENCE = 5

    def __init__(self, asr: Qwen3SimulStreamingASR, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.end = 0.0
        self.buffer: List[ASRToken] = []

        # Per-session state
        self.state = Qwen3SimulState()

        # Build the prompt template once
        self._build_prompt_template()

    def _build_prompt_template(self):
        """Build the base text prompt for Qwen3-ASR."""
        from whisperlivekit.qwen3_asr import WHISPER_TO_QWEN3_LANGUAGE

        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        self._base_prompt = self.asr.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
        )

        # Add language forcing if configured
        lan = self.asr.cfg.language
        if lan and lan != "auto":
            lang_name = WHISPER_TO_QWEN3_LANGUAGE.get(lan, lan)
            self._base_prompt += f"language {lang_name}<asr_text>"

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

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        """Append an audio chunk to be processed."""
        self.end = audio_stream_end_time
        self.state.audio_buffer = np.append(self.state.audio_buffer, audio)

        # Trim audio if too long
        max_samples = int(self.asr.cfg.audio_max_len * self.SAMPLING_RATE)
        if len(self.state.audio_buffer) > max_samples:
            trim = len(self.state.audio_buffer) - max_samples
            self.state.audio_buffer = self.state.audio_buffer[trim:]
            self.state.cumulative_time_offset += trim / self.SAMPLING_RATE
            # Adjust throttle counter so it tracks position within the trimmed buffer
            self.state.last_infer_samples = max(0, self.state.last_infer_samples - trim)

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        """Handle start of silence -- flush all pending tokens.

        Loops inference until the model produces no new tokens, since a
        single is_last call may not exhaust all text for the buffered audio.
        """
        all_tokens = []
        for _ in range(5):  # safety limit
            tokens, processed_upto = self.process_iter(is_last=True)
            if not tokens:
                break
            all_tokens.extend(tokens)
        return all_tokens, self.end

    def end_silence(self, silence_duration: float, offset: float):
        """Handle silence period."""
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
            # Long silence: reset
            self.state = Qwen3SimulState()
            self.state.global_time_offset = silence_duration + offset

    def new_speaker(self, change_speaker: ChangeSpeaker):
        """Handle speaker change event."""
        self.process_iter(is_last=True)
        self.state = Qwen3SimulState()
        self.state.speaker = change_speaker.speaker
        self.state.global_time_offset = change_speaker.start

    def get_buffer(self) -> Transcript:
        """Get the current unvalidated buffer."""
        return Transcript.from_tokens(tokens=self.buffer, sep='')

    @torch.inference_mode()
    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        """
        Process accumulated audio using SimulStreaming with alignment heads.

        This performs a full forward pass (encode audio + greedy decode with
        attention extraction), applying the border-distance policy to decide
        when to stop generating.

        Returns:
            Tuple of (committed ASRToken list, audio processed up to time).
        """
        audio_duration = len(self.state.audio_buffer) / self.SAMPLING_RATE
        if audio_duration < self.asr.cfg.audio_min_len:
            return [], self.end

        # Throttle: skip inference if less than 1s of new audio since last run.
        # Each inference re-encodes the full buffer, so calling too often wastes
        # GPU/CPU time and causes lag to spiral.
        new_samples = len(self.state.audio_buffer) - self.state.last_infer_samples
        min_new_seconds = 1.0
        if not is_last and new_samples < int(min_new_seconds * self.SAMPLING_RATE):
            return [], self.end

        logger.info("Running SimulStreaming inference on %.2fs of audio (%.2fs new)", audio_duration, new_samples / self.SAMPLING_RATE)
        self.state.last_infer_samples = len(self.state.audio_buffer)

        try:
            timestamped_words = self._infer(is_last)
        except Exception as e:
            logger.exception("Qwen3 SimulStreaming inference error: %s", e)
            return [], self.end

        logger.info("SimulStreaming produced %d words", len(timestamped_words))
        if not timestamped_words:
            return [], self.end

        self.buffer = []
        return timestamped_words, self.end

    def _infer(self, is_last: bool) -> List[ASRToken]:
        """Run one inference cycle with alignment-head-based stopping.

        Uses forward hooks on self_attn modules to capture attention weights
        during generation. The Qwen3-ASR decoder layer discards attention
        weights (hidden_states, _ = self.self_attn(...)), so output_attentions
        via generate() would return None. Hooks capture them before discard.
        """
        asr = self.asr
        state = self.state

        # Prepare inputs
        inputs = asr.processor(
            text=[self._base_prompt],
            audio=[state.audio_buffer],
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(asr.device).to(asr.dtype)

        # Append committed token IDs as context so generate() continues from
        # where it left off. Cap at max_context_tokens to prevent prompt growth.
        if state.committed_token_ids:
            ctx = state.committed_token_ids[-asr.cfg.max_context_tokens:]
            ctx_ids = torch.tensor(
                [ctx], dtype=inputs.input_ids.dtype,
                device=inputs.input_ids.device,
            )
            inputs["input_ids"] = torch.cat([inputs.input_ids, ctx_ids], dim=1)
            if "attention_mask" in inputs:
                ctx_mask = torch.ones_like(ctx_ids)
                inputs["attention_mask"] = torch.cat(
                    [inputs.attention_mask, ctx_mask], dim=1,
                )

        prompt_len = inputs.input_ids.shape[1]

        # Find audio token range
        input_ids = inputs.input_ids[0]
        audio_mask = (input_ids == asr.audio_token_id)
        audio_positions = audio_mask.nonzero(as_tuple=True)[0]
        if len(audio_positions) == 0:
            return []
        audio_start = audio_positions[0].item()
        audio_end = audio_positions[-1].item() + 1
        n_audio_tokens = audio_end - audio_start

        audio_duration = len(state.audio_buffer) / self.SAMPLING_RATE

        # Install forward hooks to capture alignment attention from Q and K.
        # With SDPA attention (fast), attn_weights are not returned. Instead,
        # we hook self_attn to compute Q*K^T attention ONLY for alignment heads
        # during autoregressive steps (q_len == 1). This is cheap because we
        # only compute dot products for ~20 heads, not full attention for all.
        #
        # Key detail: self_attn is called with ALL keyword arguments from the
        # decoder layer, so hidden_states/position_embeddings/past_key_values
        # are all in kwargs, not args.
        per_step_frames: List[List[int]] = []
        current_step_frames: List[int] = []

        heads_by_layer: dict = {}
        for layer_idx, head_idx in asr.alignment_heads:
            heads_by_layer.setdefault(layer_idx, []).append(head_idx)

        decoder_layers = asr.model.thinker.model.layers
        num_kv_heads = asr.num_kv_heads
        num_heads = asr.num_heads
        gqa_ratio = num_heads // num_kv_heads  # GQA group size

        # Import RoPE function used by this model's attention
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            apply_rotary_pos_emb,
        )

        hooks = []

        def _make_attn_hook(layer_idx):
            """Forward hook on self_attn that computes Q*K^T for alignment heads.

            After the forward pass, we recompute Q (with RoPE) for the current
            token and dot it against the cached K (which already has RoPE) in
            the audio region. This gives us per-head alignment frames.
            """
            head_indices = heads_by_layer[layer_idx]

            def hook_fn(module, args, kwargs, output):
                # All arguments are keyword-passed from the decoder layer
                hidden_states = kwargs.get('hidden_states')
                if hidden_states is None:
                    hidden_states = args[0] if args else None
                if hidden_states is None or hidden_states.shape[1] != 1:
                    return  # Skip prefill (seq_len > 1)

                position_embeddings = kwargs.get('position_embeddings')
                if position_embeddings is None and len(args) > 1:
                    position_embeddings = args[1]
                past_kv = kwargs.get('past_key_values')
                if position_embeddings is None or past_kv is None:
                    return

                # Recompute Q with RoPE (cheap: single token through q_proj + RoPE)
                hidden_shape = (*hidden_states.shape[:-1], -1, module.head_dim)
                q = module.q_norm(
                    module.q_proj(hidden_states).view(hidden_shape)
                ).transpose(1, 2)
                cos, sin = position_embeddings
                q, _ = apply_rotary_pos_emb(q, q, cos, sin)

                # K from cache already has RoPE applied
                cache_layer = past_kv.layers[module.layer_idx]
                k = cache_layer.keys  # (batch, n_kv_heads, kv_len, head_dim)
                if k is None or audio_end > k.shape[2]:
                    return

                # Compute attention scores for alignment heads only
                for h_idx in head_indices:
                    if h_idx >= q.shape[1]:
                        continue
                    kv_h_idx = h_idx // gqa_ratio
                    q_h = q[0, h_idx, 0]           # (head_dim,)
                    k_audio = k[0, kv_h_idx, audio_start:audio_end]  # (n_audio, head_dim)
                    scores = torch.matmul(k_audio, q_h)  # (n_audio,)
                    frame = scores.argmax().item()
                    current_step_frames.append(frame)

            return hook_fn

        for layer_idx in heads_by_layer:
            if layer_idx < len(decoder_layers):
                h = decoder_layers[layer_idx].self_attn.register_forward_hook(
                    _make_attn_hook(layer_idx),
                    with_kwargs=True,
                )
                hooks.append(h)

        # Step boundary hook on lm_head to separate per-step frames
        # and check border-distance stopping criteria in real-time.
        # This is CRITICAL for performance: instead of generating 200 tokens
        # then truncating, we stop as soon as attention hits the audio border.
        # On MPS, each token costs ~50ms, so stopping at 10 tokens vs 200
        # means ~0.5s vs ~10s inference.
        last_attend_frame = state.last_attend_frame
        border_stop_step: Optional[int] = None

        # Compute absolute thresholds from fractional config
        border_threshold = max(2, int(n_audio_tokens * asr.cfg.border_fraction))
        rewind_threshold = max(2, int(n_audio_tokens * asr.cfg.rewind_fraction))

        def _step_boundary_hook(module, args, output):
            nonlocal current_step_frames, last_attend_frame, border_stop_step
            if current_step_frames:
                per_step_frames.append(current_step_frames)
                current_step_frames = []

                # Check border distance on each step.
                # Allow at least 3 steps before checking, so short buffers
                # can still produce some tokens during streaming.
                if not is_last and border_stop_step is None and len(per_step_frames) >= 3:
                    latest = per_step_frames[-1]
                    if latest:
                        frames_sorted = sorted(latest)
                        attended = frames_sorted[len(frames_sorted) // 2]

                        # Rewind check
                        if last_attend_frame - attended > rewind_threshold:
                            border_stop_step = max(0, len(per_step_frames) - 2)
                            return

                        last_attend_frame = attended

                        # Border check
                        if (n_audio_tokens - attended) <= border_threshold:
                            border_stop_step = len(per_step_frames) - 1
                            return

        lm_head = asr.model.thinker.lm_head
        step_hook = lm_head.register_forward_hook(_step_boundary_hook)
        hooks.append(step_hook)

        # StoppingCriteria that stops generation when border distance is hit
        from transformers import StoppingCriteria, StoppingCriteriaList

        class BorderStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return border_stop_step is not None

        stopping = StoppingCriteriaList([BorderStop()])

        # Limit max tokens to what's reasonable for the audio duration.
        # On MPS, each token costs ~50-100ms, so tight limits are critical.
        # Speech produces ~4-6 tokens/sec; +5 for metadata prefix tokens.
        # With is_last, allow slightly more for flushing remaining text.
        new_audio_secs = (len(state.audio_buffer) - state.last_infer_samples) / self.SAMPLING_RATE
        tokens_per_sec = 6
        if is_last:
            max_tokens = min(int(audio_duration * tokens_per_sec) + 10, 120)
        else:
            max_tokens = min(int(max(new_audio_secs, 1.0) * tokens_per_sec) + 5, 40)

        try:
            outputs = asr.model.thinker.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                stopping_criteria=stopping,
            )
        finally:
            for h in hooks:
                h.remove()
            # Flush any remaining frames
            if current_step_frames:
                per_step_frames.append(current_step_frames)

        state.last_attend_frame = last_attend_frame

        # Extract generated tokens
        all_generated = outputs[0, prompt_len:]
        eos_ids = {151645, 151643}
        if asr.processor.tokenizer.eos_token_id is not None:
            eos_ids.add(asr.processor.tokenizer.eos_token_id)

        num_gen = len(all_generated)
        for i, tid in enumerate(all_generated):
            if tid.item() in eos_ids:
                num_gen = i
                break

        raw_text = asr.processor.tokenizer.decode(all_generated[:num_gen], skip_special_tokens=True)
        logger.info(
            "SimulStreaming raw output: %d tokens (stopped at step %s), text=%r",
            num_gen, border_stop_step, raw_text[:100],
        )

        if num_gen == 0:
            return []

        # Strip metadata prefix: when language is "auto", the model generates
        # "language <Name><asr_text>..." before actual transcription text.
        # Find <asr_text> token and skip everything before it (including itself).
        asr_text_id = asr.asr_text_token_id
        metadata_offset = 0
        for i in range(min(num_gen, 10)):  # metadata is at most ~3-4 tokens
            if all_generated[i].item() == asr_text_id:
                # Detect language from the metadata prefix before stripping
                if state.detected_language is None and i > 0:
                    from whisperlivekit.qwen3_asr import QWEN3_TO_WHISPER_LANGUAGE
                    prefix_text = asr.processor.tokenizer.decode(
                        all_generated[:i].tolist(), skip_special_tokens=True,
                    ).strip()
                    parts = prefix_text.split()
                    if len(parts) >= 2:
                        lang_name = parts[-1]
                        if lang_name.lower() != "none":
                            state.detected_language = QWEN3_TO_WHISPER_LANGUAGE.get(
                                lang_name, lang_name.lower(),
                            )
                            logger.info("Auto-detected language: %s", state.detected_language)
                metadata_offset = i + 1
                break

        if metadata_offset > 0:
            logger.info(
                "Stripping %d metadata prefix tokens (before <asr_text>)",
                metadata_offset,
            )
            all_generated = all_generated[metadata_offset:]
            num_gen -= metadata_offset
            per_step_frames = per_step_frames[metadata_offset:]

        if num_gen <= 0:
            return []

        # Determine how many tokens to emit based on border stopping
        step_frames = [f for f in per_step_frames if f]
        if border_stop_step is not None:
            emit_up_to = min(border_stop_step, num_gen)
        else:
            emit_up_to = num_gen

        # Build timestamped words from the emitted tokens
        generated_ids = all_generated[:emit_up_to]
        if len(generated_ids) == 0:
            return []

        all_words = self._build_timestamped_words(
            generated_ids, step_frames, emit_up_to,
            n_audio_tokens, audio_duration,
        )

        new_words = all_words

        # Update committed word count for space-prefix logic in next batch
        state.committed_word_count += len(new_words)

        # Append newly emitted token IDs to committed context for next call
        new_emitted = outputs[0, prompt_len:prompt_len + emit_up_to + metadata_offset]
        state.committed_token_ids.extend(new_emitted.tolist())

        return new_words

    def _build_timestamped_words(
        self,
        generated_ids: torch.Tensor,
        step_frames: List[List[int]],
        emit_up_to: int,
        n_audio_tokens: int,
        audio_duration: float,
    ) -> List[ASRToken]:
        """Build timestamped ASRToken list from generated tokens and hook-captured frames."""
        asr = self.asr
        state = self.state

        # Get per-token attended audio frame (median of alignment head votes)
        per_token_frame: List[Optional[int]] = []
        for step in range(emit_up_to):
            if step < len(step_frames) and step_frames[step]:
                frames = sorted(step_frames[step])
                per_token_frame.append(frames[len(frames) // 2])
            else:
                per_token_frame.append(None)

        # Decode the full generated sequence at once, then split into words.
        # This is more robust than per-token Ġ detection, which can fail when
        # committed context causes the model to generate sub-word continuations.
        tokenizer = asr.processor.tokenizer
        full_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
        text_words = full_text.split()

        # Map each text word to an approximate frame using token-level alignment.
        # Distribute frames evenly across words (since exact token→word mapping
        # is imprecise with BPE sub-words anyway).
        all_frames = [f for f in per_token_frame if f is not None]
        words = []
        for wi, word in enumerate(text_words):
            if all_frames:
                # Proportionally assign frames to words
                frac = wi / max(len(text_words), 1)
                frame_idx = int(frac * len(all_frames))
                frame_idx = min(frame_idx, len(all_frames) - 1)
                frame = all_frames[frame_idx]
            else:
                frame = None
            words.append((word, frame))

        # Convert to ASRToken with timestamps
        tokens = []
        for i, (text, frame) in enumerate(words):
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
                    (i / max(len(words), 1)) * audio_duration
                    + state.cumulative_time_offset
                )

            # Prefix space: first word of the very first batch has no space;
            # all subsequent words (same batch or later batches) get a space.
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

    @staticmethod
    def _median_frame(frames: List[int]) -> Optional[int]:
        """Return median of frame list, or None if empty."""
        if not frames:
            return None
        frames_sorted = sorted(frames)
        return frames_sorted[len(frames_sorted) // 2]

    def warmup(self, audio: np.ndarray, init_prompt: str = ""):
        """Warmup the model with a short audio clip."""
        try:
            self.state.audio_buffer = audio[:SAMPLE_RATE]
            self.process_iter(is_last=True)
            self.state = Qwen3SimulState()
            logger.info("Qwen3 SimulStreaming online processor warmed up")
        except Exception as e:
            logger.warning("Warmup failed: %s", e)
            self.state = Qwen3SimulState()

    def finish(self) -> Tuple[List[ASRToken], float]:
        """Flush remaining audio at end of stream."""
        all_tokens = []
        for _ in range(5):  # safety limit
            tokens, _ = self.process_iter(is_last=True)
            if not tokens:
                break
            all_tokens.extend(tokens)
        return all_tokens, self.end
