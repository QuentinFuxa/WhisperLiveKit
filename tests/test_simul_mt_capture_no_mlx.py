"""Regression test: simul_mt_capture must be importable and test-collectable
without MLX installed (MLX is macOS-arm64-only; Linux CI must not fail at
collection time with ModuleNotFoundError).
"""
import importlib
import sys


def test_simul_mt_capture_importable_without_mlx():
    """Importing the module must not raise even if mlx is unavailable."""
    # Remove any cached mlx modules to simulate a non-MLX environment.
    mlx_modules = {k: v for k, v in sys.modules.items() if k.startswith("mlx")}
    for k in list(mlx_modules):
        del sys.modules[k]
    # Also remove the target module so it re-imports fresh.
    sys.modules.pop("whisperlivekit.simul_mt_capture", None)
    try:
        importlib.import_module("whisperlivekit.simul_mt_capture")
    finally:
        # Restore cached modules.
        sys.modules.update(mlx_modules)


def test_captured_attention_class_exists_without_mlx():
    """CapturedAttention class must be defined at module level without MLX."""
    sys.modules.pop("whisperlivekit.simul_mt_capture", None)
    mlx_modules = {k: v for k, v in sys.modules.items() if k.startswith("mlx")}
    for k in list(mlx_modules):
        del sys.modules[k]
    try:
        mod = importlib.import_module("whisperlivekit.simul_mt_capture")
        assert hasattr(mod, "CapturedAttention")
        assert hasattr(mod, "ALIGNMENT_HEADS")
        assert hasattr(mod, "apply_commit_policy")
    finally:
        sys.modules.update(mlx_modules)
