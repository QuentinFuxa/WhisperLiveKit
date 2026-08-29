"""Per-session decoder-context regressions for issue #421."""

import threading
from types import SimpleNamespace

import pytest

from whisperlivekit.session_asr_proxy import (
    MAX_SESSION_CONTEXT_CHARS,
    SessionASRProxy,
    normalize_session_context,
    session_context_capability,
)


class _RecordingASR:
    sep = " "
    backend_choice = "faster-whisper"
    confidence_validation = False
    tokenizer = None
    buffer_trimming = "segment"
    buffer_trimming_sec = 15.0

    def __init__(self, language="en"):
        self.original_language = language
        self.calls = []

    def transcribe(self, audio, init_prompt=""):
        self.calls.append((self.original_language, init_prompt, audio))
        return []


def test_context_is_normalized_and_bounded():
    assert normalize_session_context(None) is None
    assert normalize_session_context("  WhisperLiveKit, Qwen3-ASR  ") == (
        "WhisperLiveKit, Qwen3-ASR"
    )
    assert normalize_session_context("   ") is None

    with pytest.raises(ValueError, match="at most 1000"):
        normalize_session_context("x" * (MAX_SESSION_CONTEXT_CHARS + 1))
    with pytest.raises(ValueError, match="NUL"):
        normalize_session_context("valid\x00invalid")


def test_localagreement_context_is_stable_and_session_isolated():
    shared = _RecordingASR(language="en")
    french = SessionASRProxy(
        shared,
        "fr",
        context="WhisperLiveKit, CTranslate2",
    )
    auto = SessionASRProxy(shared, "auto", context="Qwen3-ASR")

    french.transcribe("audio-one", init_prompt="rolling history")
    auto.transcribe("audio-two", init_prompt="other history")

    assert shared.calls == [
        ("fr", "WhisperLiveKit, CTranslate2\nrolling history", "audio-one"),
        (None, "Qwen3-ASR\nother history", "audio-two"),
    ]
    assert shared.original_language == "en"


def test_context_only_proxy_keeps_server_language():
    shared = _RecordingASR(language="de")
    proxy = SessionASRProxy(shared, context="Fachbegriff")

    proxy.transcribe("audio")

    assert shared.calls == [("de", "Fachbegriff", "audio")]
    assert shared.original_language == "de"


def test_simulstreaming_context_uses_an_isolated_static_config(monkeypatch):
    import whisperlivekit.simul_whisper as simul_module
    from whisperlivekit.core import online_factory

    class FakeProcessor:
        def __init__(self, asr):
            self.asr = asr

    monkeypatch.setattr(simul_module, "SimulStreamingOnlineProcessor", FakeProcessor)
    shared = _RecordingASR(language="en")
    shared.cfg = SimpleNamespace(
        language="en",
        static_init_prompt="global terminology",
    )
    shared.use_full_mlx = False
    args = SimpleNamespace(
        backend="faster-whisper",
        backend_policy="simulstreaming",
    )

    processor = online_factory(
        args,
        shared,
        language="fr",
        context="session terminology",
    )

    assert processor.asr.cfg.language == "fr"
    assert processor.asr.cfg.static_init_prompt == (
        "global terminology\nsession terminology"
    )
    assert shared.cfg.language == "en"
    assert shared.cfg.static_init_prompt == "global terminology"


@pytest.mark.parametrize("backend", ["funasr", "canary", "qwen3-streaming", "voxtral"])
def test_unsupported_backends_reject_context_before_audio_processing(backend):
    from whisperlivekit.core import online_factory

    asr = _RecordingASR()
    asr.backend_choice = backend
    # Explicit backends bypass the generic SimulStreaming route even when the
    # server-wide policy retains its default value.
    args = SimpleNamespace(backend=backend, backend_policy="simulstreaming")

    with pytest.raises(ValueError, match=f"not supported by backend {backend!r}"):
        online_factory(args, asr, context="must not be ignored")

    capability = session_context_capability(args, asr)
    assert capability == {
        "supported": False,
        "maxCharacters": 1000,
        "backend": backend,
    }


def test_localagreement_factory_exposes_context_capability_and_proxy():
    from whisperlivekit.core import online_factory
    from whisperlivekit.local_agreement.online_asr import OnlineASRProcessor

    asr = _RecordingASR()
    args = SimpleNamespace(
        backend="faster-whisper",
        backend_policy="localagreement",
    )

    processor = online_factory(args, asr, context="domain phrase")

    assert isinstance(processor, OnlineASRProcessor)
    assert isinstance(processor.asr, SessionASRProxy)
    processor.asr.transcribe("audio", init_prompt="history")
    assert asr.calls[-1][1] == "domain phrase\nhistory"
    assert session_context_capability(args, asr)["supported"] is True

    plain_processor = online_factory(args, asr)
    assert isinstance(plain_processor.asr, SessionASRProxy)
    assert plain_processor.asr._lock is processor.asr._lock


def test_plain_session_cannot_observe_another_sessions_language_override():
    from whisperlivekit.core import online_factory

    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    class BlockingASR(_RecordingASR):
        def transcribe(self, audio, init_prompt=""):
            observed_language = self.original_language
            if audio == "first":
                first_entered.set()
                assert release_first.wait(timeout=2)
            else:
                second_entered.set()
            self.calls.append((audio, observed_language, init_prompt))
            return []

    shared = BlockingASR(language="en")
    args = SimpleNamespace(
        backend="faster-whisper",
        backend_policy="localagreement",
    )
    french = online_factory(args, shared, language="fr", context="termes")
    plain = online_factory(args, shared)

    first = threading.Thread(
        target=french.asr.transcribe,
        args=("first",),
        kwargs={"init_prompt": "historique"},
    )
    second = threading.Thread(target=plain.asr.transcribe, args=("second",))
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()

    # The plain session participates in the shared lock and cannot enter while
    # the French override is installed on the shared backend.
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert shared.calls == [
        ("first", "fr", "termes\nhistorique"),
        ("second", "en", ""),
    ]
