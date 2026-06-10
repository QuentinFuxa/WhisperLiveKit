from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RealtimeAudioConfig:
    """Configuration for the native frame-synchronous Qwen3-ASR experiment.

    The defaults describe the intended production shape. Tests and smoke runs
    override the dimensions downward so they can run quickly.
    """

    sample_rate: int = 16_000
    n_mels: int = 128
    mel_hop_ms: int = 10
    decoder_step_ms: int = 80
    audio_window_sec: float = 15.0
    qwen_audio_right_context_ms: int = 640
    qwen_audio_left_context_sec: float = 2.0
    qwen_audio_strict_causal: bool = False
    # Bounded mutable tail for the causal-KV backend: encoder steps younger
    # than this stay re-computable on every chunk; older steps freeze into the
    # per-layer KV cache. 0 = strict append-only (no recompute at all).
    qwen_audio_mutable_tail_sec: float = 0.0
    # Intra-block bidirectional attention for the causal-KV backend: steps in
    # the currently processed block attend to every step of that block (plus
    # the causal KV prefix). Still append-only; latency = block size. This is
    # the standard streaming-encoder attention pattern (chunked attention).
    qwen_audio_block_bidirectional: bool = False
    qwen_audio_adapter_hidden_dim: int = 0
    qwen_audio_adapter_layers: int = 0
    qwen_audio_adapter_dropout: float = 0.0
    qwen_audio_adapter_residual_scale: float = 0.1

    d_model: int = 1024
    audio_num_layers: int = 8
    audio_num_heads: int = 8
    audio_ffn_multiplier: int = 4
    conv_kernel_size: int = 5
    rope_theta: float = 10_000.0
    dropout: float = 0.0

    wait_token: str = "[P]"
    word_start_token: str = "[W]"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be > 0")
        if self.mel_hop_ms <= 0:
            raise ValueError("mel_hop_ms must be > 0")
        if self.decoder_step_ms <= 0:
            raise ValueError("decoder_step_ms must be > 0")
        if self.decoder_step_ms % self.mel_hop_ms != 0:
            raise ValueError("decoder_step_ms must be a multiple of mel_hop_ms")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.audio_num_layers <= 0:
            raise ValueError("audio_num_layers must be > 0")
        if self.audio_num_heads <= 0:
            raise ValueError("audio_num_heads must be > 0")
        if self.d_model % self.audio_num_heads != 0:
            raise ValueError("d_model must be divisible by audio_num_heads")
        if self.conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be > 0")
        if self.audio_window_sec <= 0.0:
            raise ValueError("audio_window_sec must be > 0")
        if self.qwen_audio_right_context_ms < 0:
            raise ValueError("qwen_audio_right_context_ms must be >= 0")
        if self.qwen_audio_left_context_sec <= 0.0:
            raise ValueError("qwen_audio_left_context_sec must be > 0")
        if self.qwen_audio_adapter_hidden_dim < 0:
            raise ValueError("qwen_audio_adapter_hidden_dim must be >= 0")
        if self.qwen_audio_adapter_layers < 0:
            raise ValueError("qwen_audio_adapter_layers must be >= 0")
        if self.qwen_audio_adapter_layers and self.qwen_audio_adapter_hidden_dim <= 0:
            raise ValueError(
                "qwen_audio_adapter_hidden_dim must be > 0 when "
                "qwen_audio_adapter_layers is enabled"
            )
        if not 0.0 <= self.qwen_audio_adapter_dropout < 1.0:
            raise ValueError("qwen_audio_adapter_dropout must be in [0, 1)")
        if self.qwen_audio_adapter_residual_scale < 0.0:
            raise ValueError("qwen_audio_adapter_residual_scale must be >= 0")
        if self.qwen_audio_mutable_tail_sec < 0.0:
            raise ValueError("qwen_audio_mutable_tail_sec must be >= 0")

    @property
    def frames_per_decoder_step(self) -> int:
        return self.decoder_step_ms // self.mel_hop_ms

    @property
    def audio_window_frames(self) -> int:
        return int(round(self.audio_window_sec * 1000.0 / self.mel_hop_ms))

    @property
    def qwen_audio_right_context_frames(self) -> int:
        if self.qwen_audio_strict_causal:
            return 0
        return int(round(self.qwen_audio_right_context_ms / self.mel_hop_ms))

    @property
    def qwen_audio_left_context_frames(self) -> int:
        return int(round(self.qwen_audio_left_context_sec * 1000.0 / self.mel_hop_ms))

    @property
    def decoder_frame_sec(self) -> float:
        return self.decoder_step_ms / 1000.0
