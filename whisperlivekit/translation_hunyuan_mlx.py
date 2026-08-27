"""Re-exports the MLX translation backend class under its original name.

Backward-compat shim: the backend was renamed to the generic
``MlxLlmTranslation`` so it can host any decoder-LLM MT model, but
existing code that imports ``HunyuanMlxTranslation`` keeps working.
"""
from __future__ import annotations

from whisperlivekit.translation_mlx_llm_mt import MlxLlmTranslation

HunyuanMlxTranslation = MlxLlmTranslation

__all__ = ["HunyuanMlxTranslation", "MlxLlmTranslation"]
