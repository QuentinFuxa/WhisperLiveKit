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
    assert resolve_coalesce_min_s(-1.0) == 0.0
    assert resolve_coalesce_min_s(None) == 0.0


def test_negative_value_warns(caplog):
    with caplog.at_level("WARNING"):
        resolve_coalesce_min_s(-1.0)
    assert "coalescing disabled" in caplog.text


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_warn_and_disable(value, caplog):
    with caplog.at_level("WARNING"):
        resolved = resolve_coalesce_min_s(value)

    assert resolved == 0.0
    assert "non-finite" in caplog.text
    assert should_defer_inference(1000.0, 0.5, resolved) is False


def test_positive_value_passes_through():
    assert resolve_coalesce_min_s(0.75) == 0.75


def test_disabled_window_never_defers():
    assert should_defer_inference(0.0, 0.04, 0.0) is False


def test_defers_until_the_minimum_is_reached():
    assert should_defer_inference(0.0, 0.5, 0.75) is True
    assert should_defer_inference(0.5, 0.5, 0.75) is False


def test_a_chunk_at_or_over_the_threshold_is_never_deferred():
    assert should_defer_inference(0.0, 0.75, 0.75) is False
    assert should_defer_inference(0.0, 3.0, 0.75) is False
