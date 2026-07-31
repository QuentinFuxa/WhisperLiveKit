"""ASR call coalescing: the deferral gate and its opt-in defaults."""

from whisperlivekit.audio_processor import (
    resolve_coalesce_window,
    should_defer_inference,
)
from whisperlivekit.config import WhisperLiveKitConfig

DISABLED = (0.0, 0.0)


def test_disabled_by_default():
    config = WhisperLiveKitConfig()
    assert resolve_coalesce_window(
        config.asr_coalesce_min_s, config.asr_coalesce_max_s
    ) == DISABLED


def test_window_requires_a_ceiling_above_the_minimum():
    assert resolve_coalesce_window(0.75, 2.0) == (0.75, 2.0)
    assert resolve_coalesce_window(0.75, 0.75) == DISABLED
    assert resolve_coalesce_window(0.75, 0.5) == DISABLED


def test_non_positive_and_missing_values_disable():
    assert resolve_coalesce_window(0.0, 1.0) == DISABLED
    assert resolve_coalesce_window(-1.0, 1.0) == DISABLED
    assert resolve_coalesce_window(None, None) == DISABLED


def test_disabled_window_never_defers():
    assert should_defer_inference(0.0, 0.04, *DISABLED) is False


def test_defers_until_the_minimum_is_reached():
    min_s, max_s = 0.75, 2.0
    assert should_defer_inference(0.0, 0.5, min_s, max_s) is True
    assert should_defer_inference(0.5, 0.5, min_s, max_s) is False


def test_ceiling_forces_a_flush_before_the_minimum():
    # Larger than the ceiling, so not deferred despite a higher minimum.
    assert should_defer_inference(0.0, 3.0, 5.0, 2.0) is False
