"""Compatibility imports for the standalone Qwen3-ASR causal package."""

from ._shim import ensure_qwen3_asr_causal

ensure_qwen3_asr_causal()

from qwen3_asr_causal import Qwen3StreamingASR, Qwen3StreamingOnlineProcessor

__all__ = ["Qwen3StreamingASR", "Qwen3StreamingOnlineProcessor"]
