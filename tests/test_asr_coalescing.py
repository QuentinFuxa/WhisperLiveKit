"""ASR call coalescing: the deferral gate and its opt-in defaults."""

import math

from whisperlivekit.audio_processor import (
    resolve_coalesce_min_s,
    should_defer_inference,
)
from whisperlivekit.config import WhisperLiveKitConfig


def test_disabled_or_invalid_windows_never_defer_audio(caplog):
    windows = [WhisperLiveKitConfig().asr_coalesce_min_s, None, 0, -1,
               math.nan, math.inf, -math.inf]
    for window in windows:
        caplog.clear()
        with caplog.at_level("WARNING"):
            threshold = resolve_coalesce_min_s(window)
        assert not should_defer_inference(0, .5, threshold), window
        assert not should_defer_inference(1000, .5, threshold), window
        if window is not None and (window < 0 or not math.isfinite(window)):
            assert "coalescing disabled" in caplog.text, window
        else:
            assert not caplog.records, window


def test_deferral_gate_honors_the_configured_window():
    threshold = resolve_coalesce_min_s(0.75)
    assert should_defer_inference(0.0, 0.5, threshold) is True
    assert should_defer_inference(0.5, 0.5, threshold) is False
    assert should_defer_inference(0.0, 0.75, threshold) is False
    assert should_defer_inference(0.0, 3.0, threshold) is False
    assert should_defer_inference(0.0, 0.04, 0.0) is False
