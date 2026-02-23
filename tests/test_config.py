"""Tests for WhisperLiveKitConfig."""

import logging
from types import SimpleNamespace

import pytest

from whisperlivekit.config import WhisperLiveKitConfig


class TestDefaults:
    def test_default_backend(self):
        c = WhisperLiveKitConfig()
        assert c.backend == "auto"

    def test_default_policy(self):
        c = WhisperLiveKitConfig()
        assert c.backend_policy == "simulstreaming"

    def test_default_language(self):
        c = WhisperLiveKitConfig()
        assert c.lan == "auto"

    def test_default_vac(self):
        c = WhisperLiveKitConfig()
        assert c.vac is True

    def test_default_model_size(self):
        c = WhisperLiveKitConfig()
        assert c.model_size == "base"

    def test_default_transcription(self):
        c = WhisperLiveKitConfig()
        assert c.transcription is True
        assert c.diarization is False


class TestPostInit:
    def test_en_model_forces_english(self):
        c = WhisperLiveKitConfig(model_size="tiny.en")
        assert c.lan == "en"

    def test_en_suffix_with_auto_language(self):
        c = WhisperLiveKitConfig(model_size="base.en", lan="auto")
        assert c.lan == "en"

    def test_non_en_model_keeps_language(self):
        c = WhisperLiveKitConfig(model_size="base", lan="fr")
        assert c.lan == "fr"

    def test_policy_alias_1(self):
        c = WhisperLiveKitConfig(backend_policy="1")
        assert c.backend_policy == "simulstreaming"

    def test_policy_alias_2(self):
        c = WhisperLiveKitConfig(backend_policy="2")
        assert c.backend_policy == "localagreement"

    def test_policy_no_alias(self):
        c = WhisperLiveKitConfig(backend_policy="localagreement")
        assert c.backend_policy == "localagreement"


class TestFromNamespace:
    def test_known_keys(self):
        ns = SimpleNamespace(backend="faster-whisper", lan="en", model_size="large-v3")
        c = WhisperLiveKitConfig.from_namespace(ns)
        assert c.backend == "faster-whisper"
        assert c.lan == "en"
        assert c.model_size == "large-v3"

    def test_ignores_unknown_keys(self):
        ns = SimpleNamespace(backend="auto", unknown_key="value", another="x")
        c = WhisperLiveKitConfig.from_namespace(ns)
        assert c.backend == "auto"
        assert not hasattr(c, "unknown_key")

    def test_preserves_defaults_for_missing(self):
        ns = SimpleNamespace(backend="voxtral-mlx")
        c = WhisperLiveKitConfig.from_namespace(ns)
        assert c.lan == "auto"
        assert c.vac is True


class TestFromKwargs:
    def test_known_keys(self):
        c = WhisperLiveKitConfig.from_kwargs(backend="mlx-whisper", lan="fr")
        assert c.backend == "mlx-whisper"
        assert c.lan == "fr"

    def test_warns_on_unknown_keys(self, caplog):
        with caplog.at_level(logging.WARNING, logger="whisperlivekit.config"):
            c = WhisperLiveKitConfig.from_kwargs(backend="auto", bogus="value")
        assert c.backend == "auto"
        assert "bogus" in caplog.text

    def test_post_init_runs(self):
        c = WhisperLiveKitConfig.from_kwargs(model_size="small.en")
        assert c.lan == "en"
