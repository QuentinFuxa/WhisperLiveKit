"""Regression tests for coalescing drains at stream boundaries."""

import asyncio
from time import time
from types import SimpleNamespace

import numpy as np
import pytest

from whisperlivekit.audio_processor import (
    SENTINEL,
    AudioProcessor,
    get_all_from_queue,
)
from whisperlivekit.metrics_collector import SessionMetrics
from whisperlivekit.timed_objects import (
    ASRToken,
    ChangeSpeaker,
    Silence,
    State,
    Transcript,
)


class _SpeakerBoundaryBackend:
    """Two-pass backend matching LocalAgreement at a speaker boundary."""

    SAMPLING_RATE = 16_000

    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.end = 0.0
        self.n_passes = 0

    def insert_audio_chunk(self, audio, audio_stream_end_time):
        self.audio_buffer = np.append(self.audio_buffer, audio)
        self.end = audio_stream_end_time

    def process_iter(self):
        self.n_passes += 1
        if self.n_passes == 1:
            return [], self.end
        return [ASRToken(0.0, self.end, "speaker tail")], self.end

    def new_speaker(self, change_speaker):
        tokens, processed_upto = self.process_iter()
        self.audio_buffer = np.array([], dtype=np.float32)
        return tokens, processed_upto

    def start_silence(self):
        return self.process_iter()

    def finish(self):
        return [], self.end

    def get_buffer(self):
        return Transcript()


class _FinalDrainBackend:
    SAMPLING_RATE = 16_000

    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.end = 0.0

    def insert_audio_chunk(self, audio, audio_stream_end_time):
        self.audio_buffer = np.append(self.audio_buffer, audio)
        self.end = audio_stream_end_time

    def process_iter(self):
        return [ASRToken(0.0, self.end, "final tail")], self.end

    def finish(self):
        return [ASRToken(self.end, self.end, "finish tail")], self.end

    def get_buffer(self):
        return Transcript()


class _FinalStartSilenceBackend:
    """Backend without finish(), matching SimulStreaming's EOF fallback."""

    SAMPLING_RATE = 16_000

    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.end = 0.4

    def start_silence(self):
        return [ASRToken(0.0, self.end, "fallback tail")], self.end

    def get_buffer(self):
        return Transcript()


def _processor_for(backend, *, translation=False):
    processor = object.__new__(AudioProcessor)
    processor.args = SimpleNamespace(asr_coalesce_min_s=0.75)
    processor.transcription = backend
    processor.transcription_queue = asyncio.Queue()
    processor.translation_queue = asyncio.Queue() if translation else None
    processor.translation = None
    processor.translate_on_complete = False
    processor._pending_translation_tokens = []
    processor.diarization_queue = None
    processor.metrics = SessionMetrics()
    processor.state = State()
    processor.lock = asyncio.Lock()
    processor.sample_rate = 16_000
    processor.beg_loop = time()
    processor.sep = " "
    processor.is_stopping = False
    processor._any_asr_output = False
    processor._silent_backend_warned = False
    processor.tokens_alignment = SimpleNamespace(_retention_seconds=300.0)
    return processor


def _assert_asr_metrics(processor, *, calls, tokens):
    """Every successful backend invocation has one latency and unique tokens."""
    assert processor.metrics.n_transcription_calls == calls
    assert len(processor.metrics.transcription_durations) == calls
    assert all(duration >= 0.0 for duration in processor.metrics.transcription_durations)
    assert processor.metrics.n_tokens_produced == tokens


@pytest.mark.asyncio
async def test_change_speaker_is_a_queue_boundary():
    queue = asyncio.Queue()
    before = np.ones(8, dtype=np.float32)
    boundary = ChangeSpeaker(speaker=2, start=1)
    after = np.zeros(8, dtype=np.float32)
    for item in (before, boundary, after):
        await queue.put(item)

    np.testing.assert_array_equal(await get_all_from_queue(queue), before)
    assert await get_all_from_queue(queue) is boundary
    np.testing.assert_array_equal(await get_all_from_queue(queue), after)


@pytest.mark.asyncio
async def test_change_speaker_emits_second_pass_tokens_and_processed_position():
    backend = _SpeakerBoundaryBackend()
    processor = _processor_for(backend, translation=True)
    await processor.transcription_queue.put(np.zeros(3_200, dtype=np.float32))
    await processor.transcription_queue.put(ChangeSpeaker(speaker=2, start=1))
    await processor.transcription_queue.put(SENTINEL)

    await processor.transcription_processor()

    assert backend.n_passes == 2
    assert [token.text for token in processor.state.tokens] == ["speaker tail"]
    assert processor.state.end_transcription_processed == pytest.approx(0.2)
    # Deferred process_iter + new_speaker + terminal finish.
    _assert_asr_metrics(processor, calls=3, tokens=1)
    translated = await processor.translation_queue.get()
    assert translated.text == "speaker tail"
    assert processor.translation_queue.empty()


@pytest.mark.asyncio
async def test_silence_boundary_records_all_call_metrics_once():
    backend = _SpeakerBoundaryBackend()
    processor = _processor_for(backend)
    await processor.transcription_queue.put(np.zeros(3_200, dtype=np.float32))
    await processor.transcription_queue.put(Silence(start=0.2, is_starting=True))
    await processor.transcription_queue.put(SENTINEL)

    await processor.transcription_processor()

    assert backend.n_passes == 2
    assert [token.text for token in processor.state.tokens] == ["speaker tail"]
    assert processor.state.end_transcription_processed == pytest.approx(0.2)
    # Deferred process_iter + start_silence + terminal finish.
    _assert_asr_metrics(processor, calls=3, tokens=1)


@pytest.mark.asyncio
async def test_final_drain_and_finish_tokens_are_each_counted_once():
    processor = _processor_for(_FinalDrainBackend())
    await processor.transcription_queue.put(np.zeros(3_200, dtype=np.float32))
    await processor.transcription_queue.put(SENTINEL)

    await processor.transcription_processor()

    assert [token.text for token in processor.state.tokens] == [
        "final tail",
        "finish tail",
    ]
    # The pending drain and finish are distinct calls. Aggregating the already
    # counted drain token into final output must not increment it a second time.
    _assert_asr_metrics(processor, calls=2, tokens=2)


@pytest.mark.asyncio
async def test_eof_start_silence_fallback_records_one_call():
    processor = _processor_for(_FinalStartSilenceBackend())
    await processor.transcription_queue.put(SENTINEL)

    await processor.transcription_processor()

    assert [token.text for token in processor.state.tokens] == ["fallback tail"]
    _assert_asr_metrics(processor, calls=1, tokens=1)


def test_local_agreement_new_speaker_returns_tokens_and_position():
    from whisperlivekit.local_agreement.online_asr import OnlineASRProcessor

    class FakeASR:
        sep = " "
        tokenizer = None
        confidence_validation = False
        buffer_trimming = "segment"
        buffer_trimming_sec = 15.0

        def transcribe(self, audio, init_prompt=""):
            return object()

        def ts_words(self, result):
            return [ASRToken(0.0, 0.2, "tail")]

        def segments_end_ts(self, result):
            return []

    processor = OnlineASRProcessor(FakeASR())
    processor.insert_audio_chunk(np.zeros(3_200, dtype=np.float32))
    first_pass, _ = processor.process_iter()
    tokens, processed_upto = processor.new_speaker(
        ChangeSpeaker(speaker=2, start=1)
    )

    assert first_pass == []
    assert [token.text for token in tokens] == ["tail"]
    assert processed_upto == pytest.approx(0.2)
    assert processor.get_buffer().text == ""


@pytest.mark.asyncio
async def test_local_agreement_eof_consumes_published_buffer():
    from whisperlivekit.local_agreement.online_asr import OnlineASRProcessor

    class FakeASR:
        sep = " "
        tokenizer = None
        confidence_validation = False
        buffer_trimming = "segment"
        buffer_trimming_sec = 15.0

    backend = OnlineASRProcessor(FakeASR())
    pending = ASRToken(0.0, 0.2, "tail")
    backend.transcript_buffer.buffer = [pending]
    backend.audio_buffer = np.zeros(3_200, dtype=np.float32)
    processor = _processor_for(backend)
    processor.state.buffer_transcription = backend.get_buffer()

    await processor._finish_transcription()

    assert processor.state.tokens == [pending]
    assert processor.state.end_transcription_processed == pytest.approx(0.2)
    assert processor.state.buffer_transcription.text == ""
    _assert_asr_metrics(processor, calls=1, tokens=1)


def test_simulstreaming_new_speaker_returns_tokens_and_position():
    from whisperlivekit.simul_whisper.backend import SimulStreamingOnlineProcessor

    class FakeModel:
        speaker = -1
        global_time_offset = 0.0

        def refresh_segment(self, complete=False):
            self.refreshed = complete

    token = ASRToken(0.0, 2.5, "tail")
    processor = object.__new__(SimulStreamingOnlineProcessor)
    processor.asr = SimpleNamespace(use_full_mlx=True)
    processor.process_iter = lambda is_last=False: ([token], 2.5)
    processor.model = FakeModel()
    processor._last_committed_end = 0.0
    processor._recent_words = ["old"]

    tokens, processed_upto = processor.new_speaker(
        ChangeSpeaker(speaker=3, start=3)
    )

    assert tokens == [token]
    assert processed_upto == 2.5
    assert processor.model.refreshed is True
    assert processor.model.speaker == 3
    assert processor.model.global_time_offset == 3
    assert processor._recent_words == []


def test_qwen_normalizer_preserves_speaker_boundary_tokens():
    from whisperlivekit.core import _ASRTokenNormalizer

    class FakeQwenProcessor:
        def __init__(self):
            self.n_silence_calls = 0

        def start_silence(self):
            self.n_silence_calls += 1
            return [ASRToken(0.0, 1.0, "tail")], 1.0

    inner = FakeQwenProcessor()
    processor = _ASRTokenNormalizer(inner)

    tokens, processed_upto = processor.new_speaker(
        ChangeSpeaker(speaker=2, start=1)
    )

    assert [token.text for token in tokens] == ["tail"]
    assert processed_upto == 1.0
    assert inner.n_silence_calls == 1


def test_qwen_normalizer_converts_real_third_party_token():
    qwen_types = pytest.importorskip("qwen3_asr_causal.types")

    from whisperlivekit.core import _ASRTokenNormalizer

    third_party_token = qwen_types.ASRToken(
        start=0.25,
        end=0.75,
        text=" third-party",
        speaker=4,
        detected_language="en",
        probability=0.9,
    )

    class FakeQwenProcessor:
        def start_silence(self):
            return [third_party_token], 0.75

    tokens, processed_upto = _ASRTokenNormalizer(
        FakeQwenProcessor()
    ).new_speaker(ChangeSpeaker(speaker=2, start=1))

    assert tokens == [
        ASRToken(
            start=0.25,
            end=0.75,
            text=" third-party",
            speaker=4,
            detected_language="en",
            probability=0.9,
        )
    ]
    assert processed_upto == 0.75


def test_voxtral_hf_new_speaker_returns_boundary_result():
    from whisperlivekit.voxtral_hf_streaming import (
        VoxtralHFStreamingOnlineProcessor,
    )

    token = ASRToken(0.0, 1.0, "tail")
    processor = object.__new__(VoxtralHFStreamingOnlineProcessor)
    processor.start_silence = lambda: ([token], 1.0)

    assert processor.new_speaker(ChangeSpeaker(speaker=2, start=1)) == (
        [token],
        1.0,
    )


def test_voxtral_mlx_new_speaker_returns_boundary_result():
    voxtral_mlx = pytest.importorskip("whisperlivekit.voxtral_mlx_asr")

    token = ASRToken(0.0, 1.0, "tail")
    processor = object.__new__(voxtral_mlx.VoxtralMLXOnlineProcessor)
    processor.start_silence = lambda: ([token], 1.0)

    assert processor.new_speaker(ChangeSpeaker(speaker=2, start=1)) == (
        [token],
        1.0,
    )
