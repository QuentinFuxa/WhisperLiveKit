"""Compatibility shim for the standalone Qwen3-ASR vLLM backend."""

from __future__ import annotations

from whisperlivekit.qwen3_streaming._shim import ensure_qwen3_asr_causal, reexport

ensure_qwen3_asr_causal()

from qwen3_asr_causal.vllm import *  # noqa: E402,F401,F403
from qwen3_asr_causal.vllm import Qwen3VLLMASR as _StandaloneQwen3VLLMASR  # noqa: E402

reexport("qwen3_asr_causal.vllm", globals())

DEFAULT_QWEN3_VLLM_MODEL = "Qwen/Qwen3-ASR-1.7B"

QWEN3_VLLM_MODEL_MAPPING = {
    "base": "Qwen/Qwen3-ASR-0.6B",
    "tiny": "Qwen/Qwen3-ASR-0.6B",
    "small": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
    "medium": DEFAULT_QWEN3_VLLM_MODEL,
    "large": DEFAULT_QWEN3_VLLM_MODEL,
    "large-v3": DEFAULT_QWEN3_VLLM_MODEL,
    "qwen3-asr-1.7b": DEFAULT_QWEN3_VLLM_MODEL,
    "qwen3-1.7b": DEFAULT_QWEN3_VLLM_MODEL,
    "1.7b": DEFAULT_QWEN3_VLLM_MODEL,
}


def _resolve_model_path(kwargs: dict) -> str:
    model_path = kwargs.get("vllm_model") or kwargs.get("model_dir") or kwargs.get("model_path")
    if model_path:
        return model_path

    model_size = (kwargs.get("model_size") or "").strip()
    if not model_size:
        return DEFAULT_QWEN3_VLLM_MODEL
    lowered = model_size.lower()
    if "/" in model_size or model_size.startswith((".", "/")):
        return model_size
    return QWEN3_VLLM_MODEL_MAPPING.get(lowered, model_size)


class Qwen3VLLMASR(_StandaloneQwen3VLLMASR):
    """WhisperLiveKit-compatible default wrapper around qwen3_asr_causal."""

    def __init__(self, *args, **kwargs):
        if not any(kwargs.get(key) for key in ("vllm_model", "model_dir", "model_path", "model_size")):
            kwargs["model_size"] = DEFAULT_QWEN3_VLLM_MODEL
        super().__init__(*args, **kwargs)
