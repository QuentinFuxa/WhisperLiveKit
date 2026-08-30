"""ASR call coalescing: the deferral gate and its opt-in defaults."""

import math

import pytest

from whisperlivekit.audio_processor import (
    resolve_coalesce_min_s,
    should_defer_inference,
)
from whisperlivekit.config import WhisperLiveKitConfig


def test_disabled_by_default():
    assert resolve_coalesce_min_s(WhisperLiveKitConfig().asr_coalesce_min_s) == 0.0


def test_non_positive_and_missing_values_disable():
    assert resolve_coalesce_min_s(0.0) == 0.0
    assert resolve_coalesce_min_s(None) == 0.0


def test_negative_value_warns(caplog):
    with caplog.at_level("WARNING"):
        assert resolve_coalesce_min_s(-1.0) == 0.0
    assert "coalescing disabled" in caplog.text


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_warn_and_disable(value, caplog):
    with caplog.at_level("WARNING"):
        resolved = resolve_coalesce_min_s(value)

    assert resolved == 0.0
    assert "non-finite" in caplog.text
    assert should_defer_inference(1000.0, 0.5, resolved) is False


def test_deferral_gate_honors_the_configured_window():
    threshold = resolve_coalesce_min_s(0.75)
    assert should_defer_inference(0.0, 0.5, threshold) is True
    assert should_defer_inference(0.5, 0.5, threshold) is False
    assert should_defer_inference(0.0, 0.75, threshold) is False
    assert should_defer_inference(0.0, 3.0, threshold) is False
    assert should_defer_inference(0.0, 0.04, 0.0) is False
