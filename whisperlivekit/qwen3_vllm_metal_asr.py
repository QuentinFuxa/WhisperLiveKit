"""Compatibility shim for the standalone Qwen3-ASR vLLM Metal backend."""

from whisperlivekit.qwen3_streaming._shim import reexport

reexport("qwen3_asr_causal.metal", globals())

