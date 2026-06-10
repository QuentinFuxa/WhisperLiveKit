"""Inference model for the Qwen3 streaming backend.

Wraps the pretrained Qwen3-ASR model (loaded through HF Transformers via the
``qwen_asr`` registration shim) with a streaming contract:

- ``QwenAudioSurgeryEncoder`` re-runs the offline audio tower over a bounded
  local window (left context + right context) and emits only newly finalized
  encoder steps, so cached audio embeddings are append-only.
- ``Qwen3ASRRealtimeQwenDecoderModel.append_audio_to_cache`` accumulates those
  finalized embeddings per session.
- ``generate_full_hypothesis_from_cached_audio`` reruns a greedy full-text
  decode over the cached embeddings (bounded by the segmented streamer).

This file is the promoted inference subset of
``experiments/qwen3-causal/qwen3_streaming/native_realtime_model.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .model_config import RealtimeAudioConfig

@dataclass
class QwenAudioSurgeryState:
    mel_buffer: torch.Tensor | None = None
    window_start_frame: int = 0
    frames_seen: int = 0
    emitted_steps: int = 0
    last_input_frames: int = 0
    last_recomputed_frames: int = 0
    last_recomputed_context_frames: int = 0


@dataclass
class FrameAdapterState:
    pending: torch.Tensor | None = None
    audio_frames_seen: int = 0
    decoder_steps_seen: int = 0


@dataclass
class CachedAudioDecodeState:
    audio: QwenAudioSurgeryState
    adapter: FrameAdapterState
    frame_hidden: torch.Tensor | None = None

    @property
    def decoder_steps_seen(self) -> int:
        return int(self.adapter.decoder_steps_seen)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _batch_token_ids(
    token_ids: torch.Tensor | Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(token_ids, torch.Tensor):
        out = token_ids.to(device=device, dtype=torch.long)
        if out.ndim == 1:
            out = out.unsqueeze(0).expand(batch_size, -1).contiguous()
    else:
        out = torch.tensor(
            list(token_ids),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1).contiguous()
    if out.ndim != 2 or out.shape[0] != batch_size:
        raise ValueError(
            "token ids must have shape [batch, steps] or [steps], "
            f"got {tuple(out.shape)} for batch={batch_size}"
        )
    return out


def _cached_audio_prefix_embeds(
    model: nn.Module,
    frame_hidden: torch.Tensor,
    *,
    prefix_token_ids: torch.Tensor | Sequence[int],
    audio_placeholder_token_id: int,
) -> torch.Tensor:
    batch_size = int(frame_hidden.shape[0])
    prefix = _batch_token_ids(
        prefix_token_ids,
        batch_size=batch_size,
        device=frame_hidden.device,
    )
    audio_mask = prefix == int(audio_placeholder_token_id)
    audio_steps = int(frame_hidden.shape[1])
    counts = audio_mask.sum(dim=1)
    if int(counts.min().item()) != audio_steps or int(counts.max().item()) != audio_steps:
        raise ValueError(
            "prefix_token_ids must contain exactly one audio placeholder per "
            f"cached audio step; got {counts.tolist()} placeholders for "
            f"{audio_steps} cached steps"
        )
    embeds = model.embed_tokens(prefix)
    audio_values = frame_hidden.to(device=embeds.device, dtype=embeds.dtype)
    return embeds.masked_scatter(audio_mask.unsqueeze(-1), audio_values.reshape(-1))


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    for param in module.parameters(recurse=True):
        return param.device, param.dtype
    for buffer in module.buffers(recurse=True):
        return buffer.device, buffer.dtype
    return torch.device("cpu"), torch.float32


def _qwen_audio_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Match Qwen3-ASR's audio conv length helper without importing internals."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2
        + 1
        + (input_lengths // 100) * 13
    )
    return output_lengths.clamp_min(0)


def _banned_ngram_tokens(history: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 0 or len(history) < ngram_size - 1:
        return set()
    if ngram_size == 1:
        return set(history)
    prefix = tuple(history[-(ngram_size - 1) :])
    banned: set[int] = set()
    for start in range(0, len(history) - ngram_size + 1):
        ngram = tuple(history[start : start + ngram_size])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return banned


def _apply_repetition_controls_to_logits(
    logits: torch.Tensor,
    *,
    token_history: list[list[int]],
    consecutive_text_tokens: torch.Tensor | None,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    max_consecutive_text_tokens: int,
    wait_token_id: int | None,
) -> torch.Tensor:
    if (
        repetition_penalty == 1.0
        and no_repeat_ngram_size <= 0
        and max_consecutive_text_tokens <= 0
    ):
        return logits

    controlled = logits.clone()
    vocab_size = controlled.shape[-1]
    for batch_idx, history in enumerate(token_history):
        if repetition_penalty != 1.0 and repetition_penalty > 0.0:
            for token_id in set(history):
                if 0 <= token_id < vocab_size:
                    score = controlled[batch_idx, token_id]
                    controlled[batch_idx, token_id] = torch.where(
                        score < 0,
                        score * repetition_penalty,
                        score / repetition_penalty,
                    )

        for token_id in _banned_ngram_tokens(history, no_repeat_ngram_size):
            if 0 <= token_id < vocab_size:
                controlled[batch_idx, token_id] = torch.finfo(controlled.dtype).min

        if (
            max_consecutive_text_tokens > 0
            and wait_token_id is not None
            and 0 <= int(wait_token_id) < vocab_size
            and consecutive_text_tokens is not None
            and int(consecutive_text_tokens[batch_idx].item())
            >= max_consecutive_text_tokens
        ):
            wait_score = controlled[batch_idx, int(wait_token_id)].clone()
            controlled[batch_idx, :] = torch.finfo(controlled.dtype).min
            controlled[batch_idx, int(wait_token_id)] = wait_score
    return controlled


class QwenAudioSurgeryEncoder(nn.Module):
    """Streaming wrapper around Qwen3-ASR's pretrained audio tower.

    This first surgery pass keeps Qwen's convolution, projection and encoder
    weights intact, but replaces full-history recomputation with bounded-window
    recomputation. It is intentionally conservative: outputs are only emitted
    once the configured right context is available.
    """

    def __init__(
        self,
        audio_tower: nn.Module,
        config: RealtimeAudioConfig,
        *,
        left_context_frames: int | None = None,
        right_context_frames: int | None = None,
    ) -> None:
        super().__init__()
        self.audio_tower = audio_tower
        self.config = config
        self.left_context_frames = (
            config.qwen_audio_left_context_frames
            if left_context_frames is None
            else int(left_context_frames)
        )
        self.right_context_frames = (
            config.qwen_audio_right_context_frames
            if right_context_frames is None
            else int(right_context_frames)
        )
        if self.left_context_frames <= 0:
            raise ValueError("left_context_frames must be > 0")
        if self.right_context_frames < 0:
            raise ValueError("right_context_frames must be >= 0")

    @property
    def max_recompute_frames(self) -> int:
        return self.left_context_frames + self.right_context_frames

    def init_state(self) -> QwenAudioSurgeryState:
        return QwenAudioSurgeryState()

    def _call_audio_tower(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        try:
            outputs = self.audio_tower(
                input_features=input_features,
                feature_lens=feature_lens,
            )
        except TypeError:
            try:
                outputs = self.audio_tower(input_features, feature_lens=feature_lens)
            except TypeError:
                outputs = self.audio_tower(input_features, feature_lens)

        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            hidden = outputs[0]
        else:
            hidden = outputs
        if hidden.ndim == 3 and hidden.shape[0] == 1:
            hidden = hidden[0]
        if hidden.ndim != 2:
            raise ValueError(
                "Qwen audio tower must return [steps, hidden] or [1, steps, hidden], "
                f"got {tuple(hidden.shape)}"
            )
        return hidden

    def output_steps_for_mel_frames(self, mel_frames: int) -> int:
        if mel_frames <= 0:
            return 0
        fn = getattr(self.audio_tower, "_get_feat_extract_output_lengths", None)
        device, _ = _module_device_dtype(self.audio_tower)
        lengths = torch.tensor([mel_frames], device=device, dtype=torch.long)
        with torch.no_grad():
            output_lengths = (
                fn(lengths) if fn is not None else _qwen_audio_output_lengths(lengths)
            )
        return int(output_lengths.reshape(-1)[0].detach().cpu())

    def _encode_window(self, mels: torch.Tensor) -> torch.Tensor:
        if mels.ndim != 3:
            raise ValueError("mels must have shape [batch, frames, n_mels]")
        if mels.shape[-1] != self.config.n_mels:
            raise ValueError(
                f"expected {self.config.n_mels} mel bins, got {mels.shape[-1]}"
            )

        device, dtype = _module_device_dtype(self.audio_tower)
        hidden_by_sample: list[torch.Tensor] = []
        max_steps = 0
        for sample in mels:
            feature_lens = torch.tensor(
                [sample.shape[0]],
                device=device,
                dtype=torch.long,
            )
            input_features = sample.to(device=device, dtype=dtype).transpose(0, 1)
            hidden = self._call_audio_tower(input_features, feature_lens)
            hidden_by_sample.append(hidden)
            max_steps = max(max_steps, hidden.shape[0])

        if not hidden_by_sample:
            return mels.new_zeros(0, 0, self.config.d_model)
        hidden_size = hidden_by_sample[0].shape[-1]
        output = hidden_by_sample[0].new_zeros(len(hidden_by_sample), max_steps, hidden_size)
        for idx, hidden in enumerate(hidden_by_sample):
            output[idx, : hidden.shape[0], :] = hidden
        return output.to(device=mels.device)

    def forward_full(self, mels: torch.Tensor) -> torch.Tensor:
        return self._encode_window(mels)

    def forward_chunk(
        self,
        mels: torch.Tensor,
        state: QwenAudioSurgeryState | None = None,
    ) -> tuple[torch.Tensor, QwenAudioSurgeryState]:
        if state is None:
            state = self.init_state()
        if mels.ndim != 3:
            raise ValueError("mels must have shape [batch, frames, n_mels]")
        if mels.shape[1] == 0:
            empty = mels.new_zeros(mels.shape[0], 0, self.config.d_model)
            return empty, state

        new_mels = mels.detach()
        if state.mel_buffer is None:
            state.mel_buffer = new_mels
        else:
            if state.mel_buffer.shape[0] != new_mels.shape[0]:
                raise ValueError("batch size changed inside qwen audio stream")
            state.mel_buffer = torch.cat([state.mel_buffer, new_mels], dim=1)
        state.frames_seen += int(new_mels.shape[1])
        state.last_input_frames = int(new_mels.shape[1])

        max_frames = self.max_recompute_frames
        if state.mel_buffer.shape[1] > max_frames:
            drop = state.mel_buffer.shape[1] - max_frames
            state.mel_buffer = state.mel_buffer[:, drop:, :]
            state.window_start_frame += int(drop)
        state.last_recomputed_frames = int(state.mel_buffer.shape[1])
        state.last_recomputed_context_frames = max(
            0,
            state.last_recomputed_frames - state.last_input_frames,
        )

        hidden = self._encode_window(state.mel_buffer)
        window_start_steps = self.output_steps_for_mel_frames(state.window_start_frame)
        final_mel_frame = max(0, state.frames_seen - self.right_context_frames)
        final_global_steps = self.output_steps_for_mel_frames(final_mel_frame)

        local_start = max(state.emitted_steps, window_start_steps) - window_start_steps
        local_end = min(final_global_steps - window_start_steps, hidden.shape[1])
        local_start = max(0, min(local_start, hidden.shape[1]))
        local_end = max(local_start, min(local_end, hidden.shape[1]))
        if local_end <= local_start:
            empty = hidden.new_zeros(hidden.shape[0], 0, hidden.shape[-1])
            return empty, state

        output = hidden[:, local_start:local_end, :]
        state.emitted_steps = window_start_steps + local_end
        return output, state


class QwenAudioSurgeryAdapterBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.mlp(self.norm(x))
        return x + self.dropout(update) * self.residual_scale


class QwenAudioSurgeryFrameAdapter(nn.Module):
    """Project Qwen audio hidden states into the Qwen text hidden space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        adapter_hidden_dim: int = 0,
        adapter_layers: int = 0,
        adapter_dropout: float = 0.0,
        adapter_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.adapter_layers = int(adapter_layers)
        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        if self.input_dim == self.output_dim:
            nn.init.eye_(self.proj.weight)
        else:
            nn.init.xavier_uniform_(self.proj.weight)
        if self.adapter_layers < 0:
            raise ValueError("adapter_layers must be >= 0")
        if self.adapter_layers and self.adapter_hidden_dim <= 0:
            raise ValueError("adapter_hidden_dim must be > 0 when adapter_layers is enabled")
        self.blocks = nn.ModuleList(
            [
                QwenAudioSurgeryAdapterBlock(
                    dim=self.output_dim,
                    hidden_dim=self.adapter_hidden_dim,
                    dropout=adapter_dropout,
                    residual_scale=adapter_residual_scale,
                )
                for _ in range(self.adapter_layers)
            ]
        )

    def init_state(self) -> FrameAdapterState:
        return FrameAdapterState()

    def _project(self, audio_hidden: torch.Tensor) -> torch.Tensor:
        projected = self.proj(audio_hidden.to(dtype=self.proj.weight.dtype))
        for block in self.blocks:
            projected = block(projected)
        return projected.to(device=audio_hidden.device)

    def forward_chunk(
        self,
        audio_hidden: torch.Tensor,
        state: FrameAdapterState | None = None,
    ) -> tuple[torch.Tensor, FrameAdapterState]:
        if state is None:
            state = self.init_state()
        state.audio_frames_seen += int(audio_hidden.shape[1])
        state.decoder_steps_seen += int(audio_hidden.shape[1])
        return self._project(audio_hidden), state

    def forward_full(self, audio_hidden: torch.Tensor) -> torch.Tensor:
        return self._project(audio_hidden)


def _register_qwen3_asr_transformers() -> None:
    from qwen_asr.core.transformers_backend import (
        Qwen3ASRConfig,
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRProcessor,
    )
    from transformers import AutoConfig, AutoModel, AutoProcessor

    try:
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    except ValueError:
        pass
    try:
        AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    except ValueError:
        pass
    try:
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
    except ValueError:
        pass


class Qwen3ASRRealtimeQwenDecoderModel(nn.Module):
    """Realtime ASR scaffold using Qwen3-ASR's pretrained text decoder."""

    def __init__(
        self,
        config: RealtimeAudioConfig,
        *,
        qwen_model_id: str,
        text_model: nn.Module,
        lm_head: nn.Module,
        bos_token_id: int,
        wait_token_id: int | None = None,
        audio_encoder: nn.Module,
        adapter: nn.Module,
        audio_backend: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_model_id = qwen_model_id
        self.bos_token_id = bos_token_id
        self.wait_token_id = wait_token_id
        self.audio_backend = audio_backend
        self.audio_encoder = audio_encoder
        self.adapter = adapter
        self.text_model = text_model
        self.lm_head = lm_head
        self.embed_tokens = self.text_model.embed_tokens
        self.vocab_size = int(self.lm_head.weight.shape[0])


    def init_cached_audio_decode_state(self) -> CachedAudioDecodeState:
        return CachedAudioDecodeState(
            audio=self.audio_encoder.init_state(),
            adapter=self.adapter.init_state(),
        )

    @torch.no_grad()
    def append_audio_to_cache(
        self,
        mels: torch.Tensor,
        state: CachedAudioDecodeState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, CachedAudioDecodeState]:
        if state is None:
            state = self.init_cached_audio_decode_state()
        audio_hidden, state.audio = self.audio_encoder.forward_chunk(mels, state.audio)
        frame_delta, state.adapter = self.adapter.forward_chunk(
            audio_hidden,
            state.adapter,
        )

        if state.frame_hidden is None:
            state.frame_hidden = frame_delta.detach()
        elif frame_delta.shape[1] > 0:
            if state.frame_hidden.shape[0] != frame_delta.shape[0]:
                raise ValueError("batch size changed inside cached audio decode state")
            state.frame_hidden = torch.cat(
                [state.frame_hidden.to(frame_delta.device), frame_delta.detach()],
                dim=1,
            )

        cached = state.frame_hidden
        if cached is None:
            cached = mels.new_zeros(mels.shape[0], 0, self.config.d_model)
            state.frame_hidden = cached
        return cached, frame_delta, state

    @torch.no_grad()
    def generate_full_hypothesis_from_cached_audio(
        self,
        frame_hidden: torch.Tensor,
        *,
        prefix_token_ids: torch.Tensor | Sequence[int] | None = None,
        audio_placeholder_token_id: int | None = None,
        prompt_token_ids: torch.Tensor | Sequence[int] | None = None,
        max_new_tokens: int = 128,
        eos_token_id: int | None = None,
        stop_token_ids: Sequence[int] | None = None,
        suppress_token_ids: Sequence[int] | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        max_consecutive_text_tokens: int = 0,
        use_decoder_kv_cache: bool = True,
    ) -> torch.Tensor:
        """Greedy full-hypothesis decode over cached finalized audio embeddings.

        With ``use_decoder_kv_cache`` (default) the prefix is forwarded once and
        tokens decode incrementally over a KV cache — greedy-parity with the
        legacy per-token full re-forward at O(P + T) instead of O(T * (P + T))
        forwarded positions. Falls back to the legacy loop when the text model
        does not return a cache.
        """
        if frame_hidden.ndim != 3:
            raise ValueError("frame_hidden must have shape [batch, steps, hidden]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        batch_size = int(frame_hidden.shape[0])
        device = frame_hidden.device
        if max_new_tokens == 0:
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)

        if prefix_token_ids is not None:
            if audio_placeholder_token_id is None:
                raise ValueError(
                    "audio_placeholder_token_id is required when prefix_token_ids is set"
                )
            fixed_prefix = _cached_audio_prefix_embeds(
                self,
                frame_hidden,
                prefix_token_ids=prefix_token_ids,
                audio_placeholder_token_id=int(audio_placeholder_token_id),
            )
        else:
            fixed_prefix = None

        if prefix_token_ids is not None and prompt_token_ids is None:
            prompt = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        elif prompt_token_ids is None:
            prompt = torch.full(
                (batch_size, 1),
                int(self.bos_token_id),
                dtype=torch.long,
                device=device,
            )
        elif isinstance(prompt_token_ids, torch.Tensor):
            prompt = prompt_token_ids.to(device=device, dtype=torch.long)
            if prompt.ndim == 1:
                prompt = prompt.unsqueeze(0).expand(batch_size, -1).contiguous()
        else:
            prompt = torch.tensor(
                list(prompt_token_ids),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0).expand(batch_size, -1).contiguous()
        if prompt.ndim != 2 or prompt.shape[0] != batch_size:
            raise ValueError(
                "prompt_token_ids must have shape [batch, prompt_steps] or "
                f"[prompt_steps], got {tuple(prompt.shape)} for batch={batch_size}"
            )

        generated = prompt
        prompt_steps = int(prompt.shape[1])
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        stop_ids = set(int(token_id) for token_id in (stop_token_ids or ()))
        if eos_token_id is not None:
            stop_ids.add(int(eos_token_id))
        suppress_ids = [
            int(token_id)
            for token_id in (suppress_token_ids or ())
            if 0 <= int(token_id) < self.vocab_size
        ]
        control_wait_token_id = (
            int(self.wait_token_id)
            if self.wait_token_id is not None
            and int(self.wait_token_id) not in set(suppress_ids)
            else None
        )

        parts: list[torch.Tensor] = []
        if fixed_prefix is not None:
            parts.append(fixed_prefix)
        elif frame_hidden.shape[1] > 0:
            parts.append(frame_hidden.to(dtype=self.embed_tokens.weight.dtype))
        if generated.shape[1] > 0:
            parts.append(self.embed_tokens(generated))
        if not parts:
            return generated[:, prompt_steps:]

        if not use_decoder_kv_cache:
            return self._generate_uncached_full_hypothesis(
                fixed_prefix=fixed_prefix,
                frame_hidden=frame_hidden,
                generated=generated,
                prompt_steps=prompt_steps,
                finished=finished,
                stop_ids=stop_ids,
                suppress_ids=suppress_ids,
                control_wait_token_id=control_wait_token_id,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_consecutive_text_tokens=max_consecutive_text_tokens,
            )

        outputs = self.text_model(inputs_embeds=torch.cat(parts, dim=1), use_cache=True)
        past = getattr(outputs, "past_key_values", None)
        if past is None:
            return self._generate_uncached_full_hypothesis(
                fixed_prefix=fixed_prefix,
                frame_hidden=frame_hidden,
                generated=generated,
                prompt_steps=prompt_steps,
                finished=finished,
                stop_ids=stop_ids,
                suppress_ids=suppress_ids,
                control_wait_token_id=control_wait_token_id,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_consecutive_text_tokens=max_consecutive_text_tokens,
            )
        logits = self.lm_head(outputs.last_hidden_state[:, -1, :])
        return self._greedy_decode_with_cache(
            past=past,
            logits=logits,
            generated=generated,
            prompt_steps=prompt_steps,
            finished=finished,
            stop_ids=stop_ids,
            suppress_ids=suppress_ids,
            control_wait_token_id=control_wait_token_id,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_consecutive_text_tokens=max_consecutive_text_tokens,
        )

    def _control_logits_and_pick(
        self,
        logits: torch.Tensor,
        *,
        generated: torch.Tensor,
        prompt_steps: int,
        finished: torch.Tensor,
        stop_ids: set[int],
        suppress_ids: list[int],
        control_wait_token_id: int | None,
        eos_token_id: int | None,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        max_consecutive_text_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared per-step logits controls + argmax + stop handling."""
        device = logits.device
        if suppress_ids:
            logits = logits.clone()
            logits[:, suppress_ids] = -torch.inf
        token_history = [
            [int(token_id) for token_id in row]
            for row in generated[:, prompt_steps:].detach().cpu().tolist()
        ]
        consecutive_text_tokens = (
            torch.tensor(
                [len(row) for row in token_history],
                dtype=torch.long,
                device=device,
            )
            if max_consecutive_text_tokens > 0
            else None
        )
        logits = _apply_repetition_controls_to_logits(
            logits,
            token_history=token_history,
            consecutive_text_tokens=consecutive_text_tokens,
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            max_consecutive_text_tokens=int(max_consecutive_text_tokens),
            wait_token_id=control_wait_token_id,
        )
        next_token = logits.argmax(dim=-1)
        if stop_ids:
            if bool(finished.any().item()):
                stop_fill_id = (
                    int(eos_token_id) if eos_token_id is not None else min(stop_ids)
                )
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, stop_fill_id),
                    next_token,
                )
            finished_next = torch.tensor(
                [int(token_id) in stop_ids for token_id in next_token.tolist()],
                dtype=torch.bool,
                device=device,
            )
            finished = finished | finished_next
        return next_token, finished

    def _greedy_decode_with_cache(
        self,
        *,
        past,
        logits: torch.Tensor,
        generated: torch.Tensor,
        prompt_steps: int,
        finished: torch.Tensor,
        stop_ids: set[int],
        suppress_ids: list[int],
        control_wait_token_id: int | None,
        max_new_tokens: int,
        eos_token_id: int | None,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        max_consecutive_text_tokens: int,
    ) -> torch.Tensor:
        """Incremental greedy loop over a prefilled KV cache."""
        for step in range(max_new_tokens):
            next_token, finished = self._control_logits_and_pick(
                logits,
                generated=generated,
                prompt_steps=prompt_steps,
                finished=finished,
                stop_ids=stop_ids,
                suppress_ids=suppress_ids,
                control_wait_token_id=control_wait_token_id,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_consecutive_text_tokens=max_consecutive_text_tokens,
            )
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            if bool(finished.all().item()) or step == max_new_tokens - 1:
                break
            outputs = self.text_model(
                inputs_embeds=self.embed_tokens(next_token[:, None]),
                past_key_values=past,
                use_cache=True,
            )
            past = getattr(outputs, "past_key_values", past)
            logits = self.lm_head(outputs.last_hidden_state[:, -1, :])
        return generated[:, prompt_steps:]

    def _generate_uncached_full_hypothesis(
        self,
        *,
        fixed_prefix: torch.Tensor | None,
        frame_hidden: torch.Tensor,
        generated: torch.Tensor,
        prompt_steps: int,
        finished: torch.Tensor,
        stop_ids: set[int],
        suppress_ids: list[int],
        control_wait_token_id: int | None,
        max_new_tokens: int,
        eos_token_id: int | None,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        max_consecutive_text_tokens: int,
    ) -> torch.Tensor:
        """Legacy O(T*(P+T)) loop: full re-forward per token (parity fallback)."""
        for _ in range(max_new_tokens):
            parts: list[torch.Tensor] = []
            if fixed_prefix is not None:
                parts.append(fixed_prefix)
            elif frame_hidden.shape[1] > 0:
                parts.append(frame_hidden.to(dtype=self.embed_tokens.weight.dtype))
            if generated.shape[1] > 0:
                parts.append(self.embed_tokens(generated))
            if not parts:
                break

            decoder_inputs = torch.cat(parts, dim=1)
            outputs = self.text_model(inputs_embeds=decoder_inputs, use_cache=False)
            logits = self.lm_head(outputs.last_hidden_state[:, -1, :])
            next_token, finished = self._control_logits_and_pick(
                logits,
                generated=generated,
                prompt_steps=prompt_steps,
                finished=finished,
                stop_ids=stop_ids,
                suppress_ids=suppress_ids,
                control_wait_token_id=control_wait_token_id,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_consecutive_text_tokens=max_consecutive_text_tokens,
            )
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            if bool(finished.all().item()):
                break

        return generated[:, prompt_steps:]


class Qwen3ASRRealtimeQwenAudioSurgeryModel(Qwen3ASRRealtimeQwenDecoderModel):
    """Realtime scaffold that reuses Qwen3-ASR's pretrained audio tower."""

    def __init__(
        self,
        config: RealtimeAudioConfig,
        *,
        qwen_model_id: str,
        audio_tower: nn.Module,
        text_model: nn.Module,
        lm_head: nn.Module,
        bos_token_id: int,
        wait_token_id: int | None = None,
        audio_output_dim: int | None = None,
    ) -> None:
        audio_encoder = QwenAudioSurgeryEncoder(audio_tower, config)
        adapter = QwenAudioSurgeryFrameAdapter(
            input_dim=int(audio_output_dim or config.d_model),
            output_dim=config.d_model,
            adapter_hidden_dim=config.qwen_audio_adapter_hidden_dim,
            adapter_layers=config.qwen_audio_adapter_layers,
            adapter_dropout=config.qwen_audio_adapter_dropout,
            adapter_residual_scale=config.qwen_audio_adapter_residual_scale,
        )
        super().__init__(
            config,
            qwen_model_id=qwen_model_id,
            text_model=text_model,
            lm_head=lm_head,
            bos_token_id=bos_token_id,
            wait_token_id=wait_token_id,
            audio_encoder=audio_encoder,
            adapter=adapter,
            audio_backend="qwen_audio_surgery",
        )

    @classmethod
    def from_qwen_pretrained(
        cls,
        qwen_model_id: str,
        *,
        config: RealtimeAudioConfig | None = None,
        bos_token_id: int,
        wait_token_id: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict[str, Any] | None = "cpu",
    ) -> "Qwen3ASRRealtimeQwenAudioSurgeryModel":
        _register_qwen3_asr_transformers()
        from transformers import AutoConfig, AutoModel

        qwen_config = AutoConfig.from_pretrained(qwen_model_id)
        text_config = qwen_config.thinker_config.text_config
        audio_config = qwen_config.thinker_config.audio_config
        if config is None:
            config = RealtimeAudioConfig(
                d_model=int(text_config.hidden_size),
                audio_window_sec=15.0,
            )
        if config.d_model != int(text_config.hidden_size):
            raise ValueError(
                f"Realtime d_model={config.d_model} must match Qwen hidden_size="
                f"{int(text_config.hidden_size)}"
            )
        if config.n_mels != int(getattr(audio_config, "num_mel_bins", config.n_mels)):
            raise ValueError(
                f"Realtime n_mels={config.n_mels} must match Qwen num_mel_bins="
                f"{int(audio_config.num_mel_bins)}"
            )

        qwen_model = AutoModel.from_pretrained(
            qwen_model_id,
            dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        thinker = qwen_model.thinker
        return cls(
            config,
            qwen_model_id=qwen_model_id,
            audio_tower=thinker.audio_tower,
            text_model=thinker.model,
            lm_head=thinker.lm_head,
            bos_token_id=bos_token_id,
            wait_token_id=wait_token_id,
            audio_output_dim=int(getattr(audio_config, "output_dim", config.d_model)),
        )
