"""Symbioza-DayMind STT core package."""

from typing import TYPE_CHECKING, Any

from .config import STTConfig

if TYPE_CHECKING:  # pragma: no cover - avoid heavy import at runtime
    from .livekit_runner import run_realtime_stt as _run_realtime_stt


def run_realtime_stt(*args: Any, **kwargs: Any):
    """Lazy-load the runner to avoid heavyweight imports during testing."""

    from .livekit_runner import run_realtime_stt as _runner

    return _runner(*args, **kwargs)


__all__ = ["STTConfig", "run_realtime_stt"]
