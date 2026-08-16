"""Pause-based transcript segmentation regressions for issue #409."""

import asyncio
import math
import sys
from time import time
from types import SimpleNamespace

import numpy as np
import pytest

from whisperlivekit.audio_processor import (
    MIN_DURATION_REAL_SILENCE,
    SENTINEL,
    AudioProcessor,
)
from whisperlivekit.config import WhisperLiveKitConfig
from whisperlivekit.diff_protocol import DiffTracker
from whisperlivekit.metrics_collector import SessionMetrics
from whisperlivekit.test_client import reconstruct_state
from whisperlivekit.timed_objects import (
    ASRToken,
    FrontData,
    Silence,
    SpeakerSegment,
    State,
    Transcript,
)
from whisperlivekit.tokens_alignment import TokensAlignment


def test_pause_segmentation_config_preserves_default():
    config = WhisperLiveKitConfig()

    assert config.pause_segmentation_seconds == 5.0
    assert config.pause_segmentation_seconds == MIN_DURATION_REAL_SILENCE


@pytest.mark.parametrize("value", [-0.1, math.inf, -math.inf, math.nan, "invalid"])
def test_pause_segmentation_config_rejects_invalid_values(value):
    with pytest.raises(
        ValueError,
        match="pause_segmentation_seconds must be finite and non-negative",
    ):
        WhisperLiveKitConfig(pause_segmentation_seconds=value)


def test_parse_args_accepts_pause_segmentation_seconds(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["wlk", "--pause-segmentation-seconds", "1.25"],
    )

    from whisperlivekit.parse_args import parse_args

    assert parse_args().pause_segmentation_seconds == 1.25


def _silence_processor(threshold: float) -> AudioProcessor:
    processor = object.__new__(AudioProcessor)
    processor.pause_segmentation_seconds = threshold
    processor.sample_rate = 100
    processor.total_pcm_samples = 0
    processor.current_silence = Silence(start=1.0, is_starting=True)
    processor.metrics = SessionMetrics()
    processor.state = State()
    processor.args = SimpleNamespace(diarization=False, vac=True)
    processor.transcription_queue = None
    processor.diarization_queue = None
    processor.translation_queue = None
    processor.vac = lambda _pcm: []
    return processor


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("threshold", "duration", "expected_boundaries"),
    [
        (2.0, 1.99, 0),
        (2.0, 2.0, 0),
        (2.0, 2.01, 1),
        (0.0, 10.0, 0),
    ],
)
async def test_completed_pause_threshold_is_deterministic(
    threshold,
    duration,
    expected_boundaries,
):
    processor = _silence_processor(threshold)

    await processor._end_silence(at_sample=round((1.0 + duration) * 100))

    assert len(processor.state.new_tokens) == expected_boundaries
    assert processor.current_silence is None
    assert processor.metrics.n_silence_events == 1


class _BoundaryASR:
    SAMPLING_RATE = 16_000

    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.committed_before_pause = ASRToken(0.0, 1.0, "before")

    def start_silence(self):
        return [self.committed_before_pause], 1.0

    def end_silence(self, duration, offset):
        self.ended_silence = (duration, offset)

    def finish(self):
        return [], 3.5

    def get_buffer(self):
        return Transcript()


def _boundary_processor(
    *,
    translation: bool = False,
    translate_on_complete: bool = False,
) -> AudioProcessor:
    processor = object.__new__(AudioProcessor)
    processor.args = SimpleNamespace(asr_coalesce_min_s=0.0)
    processor.pause_segmentation_seconds = 2.0
    processor.transcription = _BoundaryASR()
    processor.transcription_queue = asyncio.Queue()
    processor.translation_queue = asyncio.Queue() if translation else None
    processor.translation = None
    processor.translate_on_complete = translate_on_complete
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
    processor._prune_state_tokens = lambda: None
    return processor


@pytest.mark.asyncio
async def test_boundary_is_published_after_words_flushed_at_silence_start():
    processor = _boundary_processor()

    pause = Silence(
        start=1.0,
        end=3.5,
        duration=2.5,
        has_ended=True,
    )
    await processor.transcription_queue.put(Silence(start=1.0, is_starting=True))
    await processor.transcription_queue.put(pause)
    await processor.transcription_queue.put(SENTINEL)

    await processor.transcription_processor()

    assert processor.state.new_tokens == [
        processor.transcription.committed_before_pause,
        pause,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("translate_on_complete", [False, True])
async def test_translation_boundary_follows_words_flushed_at_silence_start(
    translate_on_complete,
):
    processor = _boundary_processor(
        translation=True,
        translate_on_complete=translate_on_complete,
    )
    start = Silence(start=1.0, is_starting=True)
    pause = Silence(
        start=1.0,
        end=3.5,
        duration=2.5,
        has_ended=True,
    )
    await processor.transcription_queue.put(start)
    await processor.transcription_queue.put(pause)
    await processor.transcription_queue.put(SENTINEL)

    await processor.transcription_processor()

    queued = []
    while not processor.translation_queue.empty():
        queued.append(processor.translation_queue.get_nowait())
    assert queued == [
        processor.transcription.committed_before_pause,
        start,
        pause,
    ]


@pytest.mark.asyncio
async def test_eof_tail_extends_active_silence_before_committing_boundary():
    processor = _silence_processor(0.65)
    processor.current_silence = Silence(start=0.4, is_starting=True)
    processor.total_pcm_samples = 100
    processor.bytes_per_sample = 2
    processor.pcm_buffer = bytearray(np.zeros(10, dtype=np.int16).tobytes())
    enqueued_audio = []

    async def capture_audio(chunk):
        enqueued_audio.append(chunk)

    processor._enqueue_active_audio = capture_audio

    await processor._flush_remaining_pcm()

    assert processor.total_pcm_samples == 110
    assert processor.current_silence is None
    assert len(processor.state.new_tokens) == 1
    boundary = processor.state.new_tokens[0]
    assert boundary.start == 0.4
    assert boundary.end == 1.1
    assert boundary.duration == pytest.approx(0.7)
    assert enqueued_audio == []


@pytest.mark.asyncio
async def test_eof_tail_honors_vad_speech_start_and_enqueues_active_suffix():
    processor = _silence_processor(0.6)
    processor.current_silence = Silence(start=0.4, is_starting=True)
    processor.total_pcm_samples = 100
    processor.bytes_per_sample = 2
    processor.pcm_buffer = bytearray(np.arange(10, dtype=np.int16).tobytes())
    processor.vac = lambda _pcm: [{"start": 105}]
    enqueued_audio = []

    async def capture_audio(chunk):
        enqueued_audio.append(chunk.copy())

    processor._enqueue_active_audio = capture_audio

    await processor._flush_remaining_pcm()

    assert processor.total_pcm_samples == 110
    assert processor.current_silence is None
    assert len(processor.state.new_tokens) == 1
    boundary = processor.state.new_tokens[0]
    assert boundary.end == 1.05
    assert boundary.duration == pytest.approx(0.65)
    assert [len(chunk) for chunk in enqueued_audio] == [5]


@pytest.mark.asyncio
@pytest.mark.parametrize("pcm_tail", [b"", b"\x00"])
async def test_eof_commits_active_pause_without_complete_pcm_sample(pcm_tail):
    processor = _silence_processor(0.65)
    processor.current_silence = Silence(start=0.3, is_starting=True)
    processor.total_pcm_samples = 100
    processor.bytes_per_sample = 2
    processor.pcm_buffer = bytearray(pcm_tail)
    processor.beg_loop = 1.0
    processor.is_stopping = False
    processor.is_pcm_input = True

    await processor.process_audio(None)

    assert processor.is_stopping is True
    assert processor.current_silence is None
    assert len(processor.state.new_tokens) == 1
    boundary = processor.state.new_tokens[0]
    assert boundary.end == 1.0
    assert boundary.duration == pytest.approx(0.7)


def _segmented_lines(diarization: bool):
    first = ASRToken(0.0, 1.0, "first ")
    pause = Silence(start=1.0, end=3.5, duration=2.5, has_ended=True)
    second = ASRToken(3.5, 4.5, "second")
    state = State(new_tokens=[first, pause, second])
    if diarization:
        state.new_diarization = [
            SpeakerSegment(start=0.0, end=1.1, speaker=0),
            SpeakerSegment(start=3.4, end=5.0, speaker=0),
        ]
    alignment = TokensAlignment(
        state,
        SimpleNamespace(diarization=diarization),
        " ",
    )
    alignment.update()
    first_refresh = alignment.get_lines(diarization=diarization, audio_time=4.5)[0]
    alignment.update()
    second_refresh = alignment.get_lines(diarization=diarization, audio_time=4.5)[0]
    return (first, second), first_refresh, second_refresh


@pytest.mark.parametrize("diarization", [False, True])
def test_pause_creates_one_stable_boundary_without_changing_words(diarization):
    tokens, first_refresh, second_refresh = _segmented_lines(diarization)

    assert sum(line.is_silence() for line in first_refresh) == 1
    assert sum(line.is_silence() for line in second_refresh) == 1
    assert [line.is_silence() for line in second_refresh] == [False, True, False]
    assert [line.to_dict() for line in first_refresh] == [line.to_dict() for line in second_refresh]
    assert [token for line in second_refresh if not line.is_silence() for token in (line.tokens or [])] == list(tokens)
    assert [
        (token.start, token.end, token.text)
        for line in second_refresh
        if not line.is_silence()
        for token in (line.tokens or [])
    ] == [
        (0.0, 1.0, "first "),
        (3.5, 4.5, "second"),
    ]
    if diarization:
        assert [line.speaker for line in second_refresh if not line.is_silence()] == [
            1,
            1,
        ]


def test_in_progress_pause_is_stable_in_full_and_diff_modes():
    state = State(new_tokens=[ASRToken(0.0, 1.0, "speech")])
    alignment = TokensAlignment(state, SimpleNamespace(diarization=False), " ")
    tracker = DiffTracker()
    reconstructed_lines = []
    refreshes = []

    for audio_time in (1.0, 3.1, 4.2):
        alignment.update()
        lines = alignment.get_lines(audio_time=audio_time)[0]
        front_data = FrontData(status="active_transcription", lines=lines)
        message = tracker.to_message(front_data)
        reconstructed = reconstruct_state(message, reconstructed_lines)
        assert reconstructed["lines"] == front_data.to_dict()["lines"]
        refreshes.append((message, reconstructed))

    pause = Silence(start=1.0, end=4.2, duration=3.2, has_ended=True)
    state.new_tokens.append(pause)
    alignment.update()
    lines = alignment.get_lines(audio_time=4.2)[0]
    front_data = FrontData(status="active_transcription", lines=lines)
    message = tracker.to_message(front_data)
    reconstructed = reconstruct_state(message, reconstructed_lines)

    assert [len(item[1]["lines"]) for item in refreshes] == [1, 1, 1]
    assert all("new_lines" not in item[0] for item in refreshes[1:])
    assert reconstructed["lines"] == front_data.to_dict()["lines"]
    assert len(reconstructed_lines) == 2
    assert reconstructed_lines[-1]["speaker"] == -2
    assert len(message["new_lines"]) == 1


@pytest.mark.asyncio
async def test_full_mode_formatter_does_not_publish_in_progress_pause():
    processor = object.__new__(AudioProcessor)
    processor.args = SimpleNamespace(diarization=False)
    processor.pause_segmentation_seconds = 2.0
    processor.sample_rate = 100
    processor.total_pcm_samples = 400
    processor.current_silence = Silence(start=1.0, is_starting=True)
    processor.state = State(new_tokens=[ASRToken(0.0, 1.0, "speech")])
    processor.tokens_alignment = TokensAlignment(processor.state, processor.args, " ")
    processor.translation = None
    processor._ffmpeg_error = None
    processor.last_response_content = FrontData()
    processor.metrics = SessionMetrics()
    processor.is_stopping = False

    async def current_state():
        return processor.state

    processor.get_current_state = current_state
    formatter = processor.results_formatter()

    response = await anext(formatter)
    await formatter.aclose()

    assert [(line["speaker"], line["text"]) for line in response.to_dict()["lines"]] == [(1, "speech")]


def test_diarization_lag_keeps_pause_behind_buffered_speech():
    tokens = [
        ASRToken(0.0, 0.5, "first "),
        ASRToken(0.8, 1.3, "late-before "),
        Silence(start=1.3, end=3.5, duration=2.2, has_ended=True),
        ASRToken(3.5, 4.0, "after"),
    ]
    state = State(
        new_tokens=tokens,
        new_diarization=[SpeakerSegment(start=0.0, end=0.75, speaker=0)],
    )
    alignment = TokensAlignment(state, SimpleNamespace(diarization=True), " ")
    alignment.update()

    lines, buffer_text, _ = alignment.get_lines(diarization=True, audio_time=4.0)

    assert [(line.speaker, line.text) for line in lines] == [(1, "first ")]
    assert buffer_text == "late-before after"
    assert not any(line.is_silence() for line in lines)

    state.new_diarization = [SpeakerSegment(start=0.75, end=5.0, speaker=1)]
    alignment.update()
    lines, buffer_text, _ = alignment.get_lines(diarization=True, audio_time=5.0)

    assert [(line.speaker, line.text) for line in lines] == [
        (1, "first "),
        (2, "late-before "),
        (-2, None),
        (2, "after"),
    ]
    assert buffer_text == ""
    assert [token for line in lines if not line.is_silence() for token in (line.tokens or [])] == [
        tokens[0],
        tokens[1],
        tokens[3],
    ]
