"""Speaker boundary alignment regressions for issue #335."""

import importlib
import sys
import threading
from types import ModuleType, SimpleNamespace

from whisperlivekit.timed_objects import (
    ASRToken,
    FrontData,
    Silence,
    SpeakerSegment,
    State,
    Translation,
)
from whisperlivekit.tokens_alignment import TokensAlignment


def _alignment(
    tokens: list[ASRToken | Silence],
    diarization: list[SpeakerSegment],
) -> tuple[TokensAlignment, State]:
    state = State(
        new_tokens=list(tokens),
        new_diarization=list(diarization),
    )
    alignment = TokensAlignment(state, SimpleNamespace(diarization=True), " ")
    alignment.update()
    return alignment, state


def test_sortformer_final_frame_covers_the_complete_chunk(monkeypatch):
    module_name = "whisperlivekit.diarization.sortformer_backend"
    previous_module = sys.modules.pop(module_name, None)

    nemo = ModuleType("nemo")
    collections = ModuleType("nemo.collections")
    asr = ModuleType("nemo.collections.asr")
    models = ModuleType("nemo.collections.asr.models")
    modules = ModuleType("nemo.collections.asr.modules")
    models.SortformerEncLabelModel = object
    modules.AudioToMelSpectrogramPreprocessor = object
    monkeypatch.setitem(sys.modules, "nemo", nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", asr)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", models)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.modules", modules)

    try:
        sortformer = importlib.import_module(module_name)
        online = object.__new__(sortformer.SortformerDiarizationOnline)

        import torch

        online.total_preds = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]
        )
        online.max_speakers = 2
        online._len_prediction = None
        online.chunk_duration_seconds = 1.0
        online.segment_lock = threading.Lock()
        online._chunk_index = 0
        online.global_time_offset = 0.0

        segments = online._process_predictions()

        assert [(int(segment.speaker), segment.start, segment.end) for segment in segments] == [
            (0, 0.0, 0.5),
            (1, 0.5, 1.0),
        ]
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module


def test_speaker_turn_splits_unpunctuated_transcript_without_losing_tokens():
    tokens = [
        ASRToken(start=0.0, end=0.8, text="Hello ", detected_language="en"),
        ASRToken(start=0.8, end=1.8, text="there ", detected_language="en"),
        ASRToken(start=2.1, end=2.8, text="Goodbye ", detected_language="en"),
        ASRToken(start=2.8, end=3.8, text="now", detected_language="en"),
    ]
    alignment, _ = _alignment(
        tokens,
        [
            SpeakerSegment(start=0.0, end=1.0, speaker=0),
            SpeakerSegment(start=1.0, end=2.0, speaker=0),
            SpeakerSegment(start=2.0, end=4.0, speaker=1),
        ],
    )

    lines, buffer_text, _ = alignment.get_lines(diarization=True, audio_time=4.0)

    assert [(line.speaker, line.start, line.end, line.text) for line in lines] == [
        (1, 0.0, 1.8, "Hello there "),
        (2, 2.1, 3.8, "Goodbye now"),
    ]
    assert [token for line in lines for token in (line.tokens or [])] == tokens
    assert "".join(line.text or "" for line in lines) + buffer_text == "".join(
        token.text for token in tokens
    )

    payload = FrontData(lines=lines, buffer_diarization=buffer_text).to_dict()
    assert [(line["speaker"], line["text"]) for line in payload["lines"]] == [
        (1, "Hello there "),
        (2, "Goodbye now"),
    ]


def test_text_waits_for_diarization_then_moves_to_the_new_speaker_once():
    tokens = [
        ASRToken(start=0.1, end=0.9, text="first "),
        ASRToken(start=1.6, end=2.4, text="second"),
    ]
    alignment, state = _alignment(
        tokens,
        [SpeakerSegment(start=0.0, end=1.5, speaker=0)],
    )

    lines, buffer_text, _ = alignment.get_lines(diarization=True, audio_time=2.5)

    assert [(line.speaker, line.text) for line in lines] == [(1, "first ")]
    assert buffer_text == "second"
    assert "".join(line.text or "" for line in lines) + buffer_text == "first second"

    state.new_diarization = [SpeakerSegment(start=1.5, end=3.0, speaker=1)]
    alignment.update()
    lines, buffer_text, _ = alignment.get_lines(diarization=True, audio_time=3.0)

    assert [(line.speaker, line.text) for line in lines] == [
        (1, "first "),
        (2, "second"),
    ]
    assert buffer_text == ""
    assert "".join(line.text or "" for line in lines) == "first second"
    assert [token for line in lines for token in (line.tokens or [])] == tokens


def test_retrograde_timestamp_after_buffer_start_preserves_text_order():
    tokens = [
        ASRToken(start=1.5, end=2.0, text="late "),
        ASRToken(start=0.2, end=0.8, text="rewound"),
    ]
    alignment, _ = _alignment(
        tokens,
        [SpeakerSegment(start=0.0, end=1.0, speaker=0)],
    )

    lines, buffer_text, _ = alignment.get_lines(diarization=True, audio_time=2.0)

    assert lines == []
    assert buffer_text == "late rewound"
    assert buffer_text == "".join(token.text for token in tokens)


def test_translation_spanning_speaker_turn_is_attached_once_and_stays_stable():
    state = State(
        new_tokens=[
            ASRToken(start=0.0, end=2.0, text="hello "),
            ASRToken(start=2.0, end=4.0, text="goodbye"),
        ],
        new_diarization=[
            SpeakerSegment(start=0.0, end=2.0, speaker=0),
            SpeakerSegment(start=2.0, end=4.0, speaker=1),
        ],
        new_translation=[
            Translation(start=0.0, end=4.0, text="bonjour au revoir"),
        ],
    )
    alignment = TokensAlignment(state, SimpleNamespace(diarization=True), " ")
    alignment.update()

    first, _, _ = alignment.get_lines(
        diarization=True,
        translation=True,
        audio_time=4.0,
    )
    second, _, _ = alignment.get_lines(
        diarization=True,
        translation=True,
        audio_time=4.0,
    )

    expected = ["bonjour au revoir ", ""]
    assert [line.translation for line in first] == expected
    assert [line.translation for line in second] == expected
    assert sum(bool(line.translation) for line in first) == 1


def test_point_translation_at_final_endpoint_is_not_lost():
    state = State(
        new_tokens=[ASRToken(start=0.0, end=4.0, text="source")],
        new_diarization=[SpeakerSegment(start=0.0, end=4.0, speaker=0)],
        new_translation=[Translation(start=4.0, end=4.0, text="point")],
    )
    alignment = TokensAlignment(state, SimpleNamespace(diarization=True), " ")
    alignment.update()

    first, _, _ = alignment.get_lines(
        diarization=True,
        translation=True,
        audio_time=4.0,
    )
    second, _, _ = alignment.get_lines(
        diarization=True,
        translation=True,
        audio_time=4.0,
    )

    assert [line.translation for line in first] == ["point "]
    assert [line.translation for line in second] == ["point "]


def test_zero_duration_token_uses_the_containing_speaker_segment():
    token = ASRToken(start=2.0, end=2.0, text="instant")
    alignment, _ = _alignment(
        [token],
        [SpeakerSegment(start=1.0, end=3.0, speaker=1)],
    )

    lines, buffer_text, _ = alignment.get_lines(diarization=True, audio_time=3.0)

    assert [(line.speaker, line.text) for line in lines] == [(2, "instant")]
    assert buffer_text == ""


def test_zero_duration_punctuation_at_turn_closes_the_previous_speaker():
    tokens = [
        ASRToken(start=0.0, end=1.0, text="Hello"),
        ASRToken(start=1.0, end=2.0, text=" there"),
        ASRToken(start=2.0, end=2.0, text="."),
        ASRToken(start=2.0, end=3.0, text=" Next"),
    ]
    alignment, _ = _alignment(
        tokens,
        [
            SpeakerSegment(start=0.0, end=2.0, speaker=0),
            SpeakerSegment(start=2.0, end=4.0, speaker=1),
        ],
    )

    lines, buffer_text, _ = alignment.get_lines(diarization=True, audio_time=4.0)

    assert [(line.speaker, line.start, line.end, line.text) for line in lines] == [
        (1, 0.0, 2.0, "Hello there."),
        (2, 2.0, 3.0, " Next"),
    ]
    flattened_tokens = [token for line in lines for token in (line.tokens or [])]
    assert len(flattened_tokens) == len(tokens)
    assert all(actual is expected for actual, expected in zip(flattened_tokens, tokens))
    assert "".join(line.text or "" for line in lines) + buffer_text == "".join(
        token.text for token in tokens
    )


def test_missing_diarization_keeps_existing_default_speaker_behavior():
    tokens = [
        ASRToken(start=0.0, end=0.5, text="One. "),
        ASRToken(start=0.5, end=1.0, text="Two"),
    ]
    alignment, _ = _alignment(tokens, [])

    lines, buffer_text, _ = alignment.get_lines(diarization=True, audio_time=1.0)

    assert [(line.speaker, line.start, line.end, line.text) for line in lines] == [
        (-1, 0.0, 1.0, "One. Two"),
    ]
    assert lines[0].tokens == tokens
    assert buffer_text == ""


def test_speaker_resolution_uses_a_chronological_cursor(monkeypatch):
    token_count = 500
    tokens = [
        ASRToken(start=float(index), end=float(index + 1), text=f"w{index} ")
        for index in range(token_count)
    ]
    diarization = [
        SpeakerSegment(
            start=float(index),
            end=float(index + 1),
            speaker=index % 2,
        )
        for index in range(token_count)
    ]
    alignment, _ = _alignment(tokens, diarization)
    intersection_calls = 0
    original_intersection = alignment.intersection_duration

    def count_intersection(first, second):
        nonlocal intersection_calls
        intersection_calls += 1
        return original_intersection(first, second)

    monkeypatch.setattr(alignment, "intersection_duration", count_intersection)

    lines, buffer_text, _ = alignment.get_lines(
        diarization=True,
        audio_time=float(token_count),
    )

    assert len(lines) == token_count
    assert intersection_calls == token_count
    assert buffer_text == ""


def test_speaker_boundaries_preserve_explicit_silence_and_stable_refreshes():
    silence = Silence(start=1.0, end=6.0, duration=5.0, has_ended=True)
    tokens = [
        ASRToken(start=0.0, end=0.8, text="before "),
        silence,
        ASRToken(start=6.1, end=7.0, text="after"),
    ]
    alignment, _ = _alignment(
        tokens,
        [
            SpeakerSegment(start=0.0, end=1.0, speaker=0),
            SpeakerSegment(start=6.0, end=8.0, speaker=1),
        ],
    )

    first, first_buffer, _ = alignment.get_lines(diarization=True, audio_time=8.0)
    second, second_buffer, _ = alignment.get_lines(diarization=True, audio_time=8.0)

    expected = [
        (1, 0.0, 0.8, "before "),
        (-2, 1.0, 6.0, None),
        (2, 6.1, 7.0, "after"),
    ]
    assert [(line.speaker, line.start, line.end, line.text) for line in first] == expected
    assert [(line.speaker, line.start, line.end, line.text) for line in second] == expected
    assert first_buffer == second_buffer == ""
