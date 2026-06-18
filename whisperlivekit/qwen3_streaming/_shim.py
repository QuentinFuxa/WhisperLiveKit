"""Compatibility helpers for the old WhisperLiveKit Qwen import paths."""

from importlib import import_module
from pathlib import Path
import sys


def ensure_qwen3_asr_causal() -> None:
    try:
        import_module("qwen3_asr_causal")
        return
    except ModuleNotFoundError:
        pass

    src = Path(__file__).resolve().parents[2] / "third_party" / "qwen3-asr-causal" / "src"
    if src.is_dir():
        sys.path.insert(0, str(src))
        import_module("qwen3_asr_causal")


def reexport(module_name: str, namespace: dict) -> None:
    ensure_qwen3_asr_causal()
    module = import_module(module_name)
    for name in dir(module):
        if name in {"__builtins__", "__cached__", "__file__", "__loader__", "__name__", "__package__", "__spec__"}:
            continue
        namespace[name] = getattr(module, name)
