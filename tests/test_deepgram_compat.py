"""Deepgram protocol regressions for issue #413."""

import asyncio
import unicodedata
from types import SimpleNamespace

import pytest

from whisperlivekit.deepgram_compat import (
    DeepgramAdapter,
    DeepgramOptions,
    handle_deepgram_websocket,
)
from whisperlivekit.timed_objects import (
    ASRToken,
    AudioStreamEvent,
    FrontData,
    Segment,
    Silence,
)


class RecordingWebSocket:
    def __init__(self, query_params=None, incoming=None, headers=None):
        self.query_params = query_params or {}
        self.headers = headers or {}
        self.incoming = asyncio.Queue()
        for message in incoming or []:
            self.incoming.put_nowait(message)
        self.sent = []
        self.accepted = False
        self.closed = None

    async def accept(self):
        self.accepted = True

    async def send_json(self, message):
        self.sent.append(message)

    async def receive(self):
        return await self.incoming.get()

    async def close(self, code=1000, reason=None):
        self.closed = (code, reason)


def _options(**overrides):
    values = {
        "language": None,
        "interim_results": True,
        "vad_events": False,
        "endpointing_ms": 300,
        "utterance_end_ms": None,
        "pcm_input": None,
    }
    values.update(overrides)
    return DeepgramOptions(**values)


def _token(start, end, text):
    return ASRToken(start=start, end=end, text=text)


def _speech_line(*tokens, speaker=-1):
    line = Segment.from_tokens(list(tokens))
    line.speaker = speaker
    return line


def _results(websocket, *, final=None):
    messages = [message for message in websocket.sent if message.get("type") == "Results"]
    if final is None:
        return messages
    return [message for message in messages if message["is_final"] is final]


def _transcript(message):
    return message["channel"]["alternatives"][0]["transcript"]


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, (False, False, 10, None)),
        (
            {
                "interim_results": "true",
                "vad_events": "true",
                "endpointing": "450",
                "utterance_end_ms": "1200",
            },
            (True, True, 450, 1200),
        ),
        ({"endpointing": "false"}, (False, False, None, None)),
        ({"endpointing": "true"}, (False, False, 10, None)),
    ],
)
def test_deepgram_options_parse_supported_values(params, expected):
    options = DeepgramOptions.from_query_params(params, vac_enabled=True)

    assert (
        options.interim_results,
        options.vad_events,
        options.endpointing_ms,
        options.utterance_end_ms,
    ) == expected


def test_linear16_options_select_exact_pcm_contract():
    options = DeepgramOptions.from_query_params(
        {
            "encoding": "linear16",
            "sample_rate": "16000",
            "channels": "1",
        },
        vac_enabled=True,
    )

    assert options.pcm_input is True


def test_context_option_is_preserved_for_the_session_processor():
    options = DeepgramOptions.from_query_params(
        {"context": "WhisperLiveKit, Qwen3-ASR"},
        vac_enabled=True,
    )

    assert options.context == "WhisperLiveKit, Qwen3-ASR"


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"interim_results": "yes"}, "interim_results must be true or false"),
        ({"vad_events": "1"}, "vad_events must be true or false"),
        ({"endpointing": "0"}, "endpointing must be greater than zero"),
        ({"endpointing": "-1"}, "endpointing must be an integer"),
        ({"endpointing": "1.5"}, "endpointing must be an integer"),
        ({"utterance_end_ms": "999", "interim_results": "true"}, "between 1000 and 5000"),
        ({"utterance_end_ms": "5001", "interim_results": "true"}, "between 1000 and 5000"),
        ({"utterance_end_ms": "1000"}, "requires interim_results=true"),
        ({"encoding": "mulaw"}, "encoding must be linear16"),
        ({"encoding": "linear16"}, "sample_rate=16000 is required"),
        (
            {"encoding": "linear16", "sample_rate": "8000"},
            "supports sample_rate=16000 only",
        ),
        ({"channels": "2"}, "supports channels=1 only"),
    ],
)
def test_deepgram_options_reject_invalid_values(params, message):
    with pytest.raises(ValueError, match=message):
        DeepgramOptions.from_query_params(params, vac_enabled=True)


def test_deepgram_options_require_vac_for_endpointing():
    with pytest.raises(ValueError, match="endpointing requires VAC"):
        DeepgramOptions.from_query_params({}, vac_enabled=False)

    options = DeepgramOptions.from_query_params(
        {"endpointing": "false"},
        vac_enabled=False,
    )
    assert options.endpointing_ms is None

    with pytest.raises(ValueError, match="vad_events requires VAC"):
        DeepgramOptions.from_query_params(
            {"endpointing": "false", "vad_events": "true"},
            vac_enabled=False,
        )


@pytest.mark.parametrize(
    ("query", "header", "expected"),
    [
        ({"token": "query-secret"}, "", "query-secret"),
        ({}, "Token sdk-secret", "sdk-secret"),
        ({}, "Bearer bearer-secret", "bearer-secret"),
        ({}, "Basic ignored", None),
    ],
)
def test_websocket_auth_accepts_wlk_and_deepgram_token_styles(
    query,
    header,
    expected,
):
    from whisperlivekit.api_auth import websocket_token

    websocket = RecordingWebSocket(
        query,
        headers={"authorization": header},
    )
    assert websocket_token(websocket) == expected


@pytest.mark.asyncio
async def test_line_growth_and_pause_emit_every_word_once():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options())
    hello = _token(0.1, 0.5, "hello")
    world = _token(0.6, 0.9, "world")

    await adapter.process_update(FrontData(lines=[_speech_line(hello)]))
    await adapter.process_update(FrontData(lines=[_speech_line(hello, world)]))
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 0.9))
    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 0.9)
    )
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 1.199))

    assert [_transcript(message) for message in _results(websocket, final=True)] == ["hello"]
    assert not any(message["speech_final"] for message in _results(websocket, final=True))

    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 1.2))
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 2.0))

    finals = _results(websocket, final=True)
    assert [_transcript(message) for message in finals] == ["hello", "world"]
    assert [message["speech_final"] for message in finals] == [False, True]
    assert " ".join(filter(None, map(_transcript, finals))) == "hello world"


@pytest.mark.asyncio
async def test_mutable_suffix_replacement_never_publishes_retracted_word():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options())
    hello = _token(0.0, 0.4, "hello")
    world = _token(0.4, 0.8, "world")
    there = _token(0.4, 0.85, "there")

    await adapter.process_update(FrontData(lines=[_speech_line(hello, world)]))
    await adapter.process_update(FrontData(lines=[_speech_line(hello, there)]))
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 0.9))
    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 0.9)
    )
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 1.2))

    finals = _results(websocket, final=True)
    assert [_transcript(message) for message in finals] == ["hello", "there"]
    assert all("world" not in _transcript(message) for message in finals)


@pytest.mark.asyncio
async def test_split_and_merge_lines_do_not_duplicate_tokens():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options(endpointing_ms=None))
    one = _token(0.0, 0.3, "go")
    two = _token(0.3, 0.6, "go")
    three = _token(0.6, 0.9, "now")

    await adapter.process_update(FrontData(lines=[_speech_line(one, two)]))
    await adapter.process_update(
        FrontData(lines=[_speech_line(one, speaker=1), _speech_line(two, speaker=2)])
    )
    await adapter.process_update(FrontData(lines=[_speech_line(one, two, three)]))
    await adapter.finish_stream(1.0)

    finals = _results(websocket, final=True)
    assert [_transcript(message) for message in finals] == ["go go", "now"]
    assert " ".join(map(_transcript, finals)) == "go go now"


@pytest.mark.asyncio
async def test_interim_revises_speaker_and_time_without_duplicating_final_text():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options(endpointing_ms=None))
    first = _token(0.0, 0.4, "hello")
    revised = _token(0.05, 0.45, "hello")

    await adapter.process_update(
        FrontData(lines=[_speech_line(first, speaker=1)])
    )
    await adapter.process_update(
        FrontData(lines=[_speech_line(revised, speaker=2)])
    )

    interims = _results(websocket, final=False)
    assert len(interims) == 2
    first_word = interims[0]["channel"]["alternatives"][0]["words"][0]
    revised_word = interims[1]["channel"]["alternatives"][0]["words"][0]
    assert (first_word["speaker"], first_word["start"], first_word["end"]) == (
        0,
        0.0,
        0.4,
    )
    assert (
        revised_word["speaker"],
        revised_word["start"],
        revised_word["end"],
    ) == (1, 0.05, 0.45)

    await adapter.finish_stream(0.5)
    assert [_transcript(message) for message in _results(websocket, final=True)] == [
        "hello"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (["Hello", ",", " world", "!"], "Hello, world!"),
        (["“", "Hello", "”"], "“Hello”"),
        (["你", "好", "，", "世界"], "你好，世界"),
    ],
)
async def test_transcript_preserves_source_spacing_and_punctuation(parts, expected):
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options(endpointing_ms=None))
    tokens = [
        _token(index * 0.1, (index + 1) * 0.1, text)
        for index, text in enumerate(parts)
    ]

    await adapter.process_update(FrontData(lines=[_speech_line(*tokens)]))
    await adapter.finish_stream(len(parts) * 0.1)

    result = _results(websocket, final=True)[0]
    assert _transcript(result) == expected
    words = result["channel"]["alternatives"][0]["words"]
    assert all("_source_text" not in word for word in words)
    assert all(
        not all(
            unicodedata.category(character).startswith("P")
            for character in word["word"]
        )
        for word in words
    )
    if expected == "Hello, world!":
        assert [(word["word"], word["punctuated_word"]) for word in words] == [
            ("Hello", "Hello,"),
            ("world", "world!"),
        ]
    if expected == "“Hello”":
        assert [(word["word"], word["punctuated_word"]) for word in words] == [
            ("Hello", "“Hello”"),
        ]


@pytest.mark.asyncio
async def test_late_terminal_punctuation_is_emitted_once_after_final_word():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options(endpointing_ms=100))
    hello = _token(0.0, 0.4, "hello")
    comma = _token(0.4, 0.45, ",")

    await adapter.process_update(FrontData(lines=[_speech_line(hello)]))
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 0.4))
    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 0.4)
    )
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 0.5))
    await adapter.process_update(FrontData(lines=[_speech_line(hello, comma)]))
    await adapter.process_update(FrontData(lines=[_speech_line(hello, comma)]))

    finals = _results(websocket, final=True)
    assert [_transcript(message) for message in finals] == ["hello", ","]
    assert finals[0]["speech_final"] is True
    assert finals[1]["speech_final"] is False
    assert finals[1]["channel"]["alternatives"][0]["words"] == []


@pytest.mark.asyncio
async def test_snapshot_shrink_cannot_delete_or_crash_below_final_cursor():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options(endpointing_ms=None))
    one = _token(0.0, 0.2, "one")
    two = _token(0.2, 0.4, "two")
    three = _token(0.4, 0.6, "three")

    await adapter.process_update(FrontData(lines=[_speech_line(one, two)]))
    await adapter.process_update(FrontData(lines=[_speech_line(one, two, three)]))
    await adapter.process_update(FrontData(lines=[_speech_line(one)]))
    await adapter.finish_stream(0.6)

    assert [_transcript(message) for message in _results(websocket, final=True)] == [
        "one two"
    ]


@pytest.mark.asyncio
async def test_endpointing_false_never_marks_pause_speech_final():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options(endpointing_ms=None))
    hello = _token(0.0, 0.5, "hello")

    await adapter.process_update(FrontData(lines=[_speech_line(hello)]))
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 0.5))
    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 0.5)
    )
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 5.0))

    assert not _results(websocket, final=True)

    await adapter.finish_stream(5.0)
    finals = _results(websocket, final=True)
    assert [_transcript(message) for message in finals] == ["hello"]
    assert finals[0]["speech_final"] is False


@pytest.mark.asyncio
async def test_interim_option_gates_revisions_and_close_freezes_buffer():
    disabled_socket = RecordingWebSocket()
    disabled = DeepgramAdapter(
        disabled_socket,
        _options(interim_results=False, endpointing_ms=None),
    )
    await disabled.process_update(FrontData(buffer_transcription="hello"))
    assert not _results(disabled_socket)
    await disabled.finish_stream(0.5)
    assert [_transcript(message) for message in _results(disabled_socket)] == ["hello"]
    assert _results(disabled_socket)[0]["is_final"] is True

    enabled_socket = RecordingWebSocket()
    enabled = DeepgramAdapter(
        enabled_socket,
        _options(interim_results=True, endpointing_ms=None),
    )
    await enabled.process_update(FrontData(buffer_transcription="hello"))
    await enabled.process_update(FrontData(buffer_transcription="hello world"))
    await enabled.process_update(FrontData(buffer_transcription="hello world"))
    assert [_transcript(message) for message in _results(enabled_socket)] == [
        "hello",
        "hello world",
    ]
    assert all(not message["is_final"] for message in _results(enabled_socket))


@pytest.mark.asyncio
async def test_diarization_and_asr_buffers_are_visible_and_frozen_at_close():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(
        websocket,
        _options(interim_results=True, endpointing_ms=None),
    )

    await adapter.process_update(
        FrontData(
            buffer_diarization="waiting for speaker",
            buffer_transcription="and ASR",
        )
    )
    assert [_transcript(message) for message in _results(websocket)] == [
        "waiting for speaker and ASR"
    ]

    await adapter.finish_stream(1.0)
    finals = _results(websocket, final=True)
    assert [_transcript(message) for message in finals] == [
        "waiting for speaker and ASR"
    ]
    assert finals[0]["channel"]["alternatives"][0]["words"] == []


@pytest.mark.asyncio
async def test_serialized_diarization_buffer_is_not_dropped():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(
        websocket,
        _options(interim_results=False, endpointing_ms=None),
    )

    await adapter.process_update(
        {
            "lines": [],
            "buffer_diarization": "late diarization",
            "buffer_transcription": "tail",
        }
    )
    await adapter.finish_stream(1.0)

    assert [_transcript(message) for message in _results(websocket, final=True)] == [
        "late diarization tail"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("line_text", "diarization_buffer", "asr_buffer", "expected"),
    [
        ("你", "", "好", "你好"),
        ("Hello", "", ",", "Hello,"),
        ("", "你", "好", "你好"),
        ("", "Hello", "world", "Hello world"),
    ],
)
async def test_buffer_boundaries_preserve_language_appropriate_spacing(
    line_text,
    diarization_buffer,
    asr_buffer,
    expected,
):
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(
        websocket,
        _options(interim_results=True, endpointing_ms=None),
    )
    lines = []
    if line_text:
        lines.append(_speech_line(_token(0.0, 0.2, line_text)))

    await adapter.process_update(
        FrontData(
            lines=lines,
            buffer_diarization=diarization_buffer,
            buffer_transcription=asr_buffer,
        )
    )

    assert [_transcript(message) for message in _results(websocket, final=False)] == [
        expected
    ]


@pytest.mark.asyncio
async def test_final_word_timestamps_are_frozen_across_later_growth():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options(endpointing_ms=None))
    first_hello = _token(0.0, 0.4, "hello")
    revised_hello = _token(0.0, 0.5, "hello")
    world = _token(0.5, 0.8, "world")
    again = _token(0.8, 1.1, "again")

    await adapter.process_update(FrontData(lines=[_speech_line(first_hello)]))
    await adapter.process_update(
        FrontData(lines=[_speech_line(revised_hello, world)])
    )
    first_final = _results(websocket, final=True)[0]
    assert first_final["channel"]["alternatives"][0]["words"][0]["end"] == 0.5

    revised_again = _token(0.0, 0.7, "hello")
    await adapter.process_update(
        FrontData(lines=[_speech_line(revised_again, world, again)])
    )
    assert first_final["channel"]["alternatives"][0]["words"][0]["end"] == 0.5


@pytest.mark.asyncio
async def test_short_pause_is_cancelled_before_endpoint_threshold():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options(endpointing_ms=300))
    hello = _token(0.0, 0.5, "hello")

    await adapter.process_update(FrontData(lines=[_speech_line(hello)]))
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 0.5))
    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 0.5)
    )
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 0.75))
    await adapter.process_stream_event(AudioStreamEvent("speech_started", 0.75))
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 2.0))

    assert not any(message["speech_final"] for message in _results(websocket, final=True))


@pytest.mark.asyncio
async def test_short_or_disabled_endpoint_does_not_delay_vad_speech_started():
    for endpointing_ms in (300, None):
        websocket = RecordingWebSocket()
        adapter = DeepgramAdapter(
            websocket,
            _options(endpointing_ms=endpointing_ms, vad_events=True),
        )
        await adapter.process_stream_event(AudioStreamEvent("silence_started", 0.5))
        await adapter.process_stream_event(AudioStreamEvent("speech_started", 0.7))

        assert [
            message
            for message in websocket.sent
            if message.get("type") == "SpeechStarted"
        ] == [{"type": "SpeechStarted", "channel": [0, 1], "timestamp": 0.7}]


@pytest.mark.asyncio
async def test_long_completed_pause_waits_for_late_transcription_barrier():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(
        websocket,
        _options(endpointing_ms=300, vad_events=True),
    )
    hello = _token(0.0, 0.5, "hello")

    await adapter.process_update(FrontData(lines=[_speech_line(hello)]))
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 0.5))
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 0.9))
    await adapter.process_stream_event(AudioStreamEvent("speech_started", 1.0))
    assert not _results(websocket, final=True)

    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 0.5)
    )

    finals = _results(websocket, final=True)
    assert [_transcript(message) for message in finals] == ["hello"]
    assert finals[0]["speech_final"] is True
    assert [
        message for message in websocket.sent if message.get("type") == "SpeechStarted"
    ] == [{"type": "SpeechStarted", "channel": [0, 1], "timestamp": 1.0}]


@pytest.mark.asyncio
async def test_endpoint_waits_for_final_diarization_speaker():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options(endpointing_ms=300))
    hello = _token(0.0, 0.5, "hello")

    await adapter.process_update(
        FrontData(
            lines=[_speech_line(hello, speaker=1)],
            remaining_time_diarization=0.4,
        )
    )
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 0.5))
    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 0.5)
    )
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 0.8))
    assert not _results(websocket, final=True)

    await adapter.process_update(
        FrontData(
            lines=[_speech_line(hello, speaker=2)],
            remaining_time_diarization=0.0,
        )
    )

    final = _results(websocket, final=True)[0]
    assert _transcript(final) == "hello"
    assert final["speech_final"] is True
    assert final["channel"]["alternatives"][0]["words"][0]["speaker"] == 1


@pytest.mark.asyncio
async def test_line_growth_cannot_freeze_speakers_while_diarization_lags():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options(endpointing_ms=300))
    hello = _token(0.0, 0.5, "hello")
    world = _token(0.5, 0.9, "world")

    await adapter.process_update(
        FrontData(
            lines=[_speech_line(hello, speaker=1)],
            remaining_time_diarization=0.4,
        )
    )
    await adapter.process_update(
        FrontData(
            lines=[_speech_line(hello, world, speaker=1)],
            remaining_time_diarization=0.4,
        )
    )
    assert not _results(websocket, final=True)

    await adapter.process_stream_event(AudioStreamEvent("silence_started", 0.9))
    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 0.9)
    )
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 1.2))
    await adapter.process_update(
        FrontData(
            lines=[
                _speech_line(hello, speaker=2),
                _speech_line(world, speaker=3),
            ],
            remaining_time_diarization=0.0,
        )
    )

    final = _results(websocket, final=True)[0]
    words = final["channel"]["alternatives"][0]["words"]
    assert [_transcript(final)] == ["hello world"]
    assert [word["speaker"] for word in words] == [1, 2]


@pytest.mark.asyncio
async def test_silence_boundary_waits_for_final_diarization_speaker():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options(endpointing_ms=None))
    hello = _token(0.0, 0.5, "hello")
    silence = Silence(start=0.5, end=1.0)

    await adapter.process_update(
        FrontData(
            lines=[_speech_line(hello, speaker=1), silence],
            remaining_time_diarization=0.4,
        )
    )
    assert not _results(websocket, final=True)

    await adapter.process_update(
        FrontData(
            lines=[_speech_line(hello, speaker=2), silence],
            remaining_time_diarization=0.0,
        )
    )

    final = _results(websocket, final=True)[0]
    assert _transcript(final) == "hello"
    assert final["speech_final"] is False
    assert final["channel"]["alternatives"][0]["words"][0]["speaker"] == 1


@pytest.mark.asyncio
async def test_two_delayed_pause_barriers_keep_distinct_endpoints_and_starts():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(
        websocket,
        _options(endpointing_ms=300, vad_events=True),
    )
    one = _token(0.0, 0.5, "one")
    two = _token(1.0, 1.5, "two")

    await adapter.process_update(FrontData(lines=[_speech_line(one)]))
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 0.5))
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 0.9))
    await adapter.process_stream_event(AudioStreamEvent("speech_started", 1.0))
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 1.5))
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 1.9))
    await adapter.process_stream_event(AudioStreamEvent("speech_started", 2.0))

    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 0.5)
    )
    await adapter.process_update(FrontData(lines=[_speech_line(one, two)]))
    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 1.5)
    )

    finals = _results(websocket, final=True)
    assert [_transcript(message) for message in finals] == ["one", "two"]
    assert [message["speech_final"] for message in finals] == [True, True]
    speech_started = [
        message for message in websocket.sent if message.get("type") == "SpeechStarted"
    ]
    assert [message["timestamp"] for message in speech_started] == [1.0, 2.0]


@pytest.mark.asyncio
async def test_diarization_lag_keeps_two_pause_prefixes_distinct():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(
        websocket,
        _options(endpointing_ms=300, vad_events=True),
    )
    one = _token(0.0, 0.5, "one")
    two = _token(1.0, 1.5, "two")

    await adapter.process_update(
        FrontData(
            lines=[_speech_line(one, speaker=1)],
            remaining_time_diarization=0.4,
        )
    )
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 0.5))
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 0.9))
    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 0.5)
    )
    await adapter.process_stream_event(AudioStreamEvent("speech_started", 1.0))

    await adapter.process_update(
        FrontData(
            lines=[_speech_line(one, two, speaker=1)],
            remaining_time_diarization=0.4,
        )
    )
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 1.5))
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 1.9))
    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 1.5)
    )
    await adapter.process_stream_event(AudioStreamEvent("speech_started", 2.0))
    assert not _results(websocket, final=True)

    await adapter.process_update(
        FrontData(
            lines=[
                _speech_line(one, speaker=2),
                _speech_line(two, speaker=3),
            ],
            remaining_time_diarization=0.0,
        )
    )

    finals = _results(websocket, final=True)
    assert [_transcript(message) for message in finals] == ["one", "two"]
    assert [message["speech_final"] for message in finals] == [True, True]
    assert [
        message["channel"]["alternatives"][0]["words"][0]["speaker"]
        for message in finals
    ] == [1, 2]


@pytest.mark.asyncio
async def test_later_ready_pause_cannot_finalize_an_earlier_unready_pause():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options(endpointing_ms=300))
    one = _token(0.0, 0.5, "one")
    two = _token(1.0, 1.5, "two")

    await adapter.process_update(FrontData(lines=[_speech_line(one)]))
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 0.5))
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 0.9))
    await adapter.process_stream_event(AudioStreamEvent("speech_started", 1.0))
    await adapter.process_update(FrontData(lines=[_speech_line(one, two)]))
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 1.5))
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 1.9))
    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 1.5)
    )
    await adapter.process_stream_event(AudioStreamEvent("speech_started", 2.0))
    assert not _results(websocket, final=True)

    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 0.5)
    )

    finals = _results(websocket, final=True)
    assert [_transcript(message) for message in finals] == ["one", "two"]
    assert all(message["speech_final"] is True for message in finals)


@pytest.mark.asyncio
async def test_short_later_pause_cannot_reorder_speech_started_events():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(
        websocket,
        _options(endpointing_ms=300, vad_events=True),
    )
    one = _token(0.0, 0.5, "one")

    await adapter.process_update(FrontData(lines=[_speech_line(one)]))
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 0.5))
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 0.9))
    await adapter.process_stream_event(AudioStreamEvent("speech_started", 1.0))
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 1.5))
    await adapter.process_stream_event(AudioStreamEvent("speech_started", 1.7))
    assert not [
        message
        for message in websocket.sent
        if message.get("type") == "SpeechStarted"
    ]

    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 0.5)
    )

    speech_started = [
        message
        for message in websocket.sent
        if message.get("type") == "SpeechStarted"
    ]
    assert [message["timestamp"] for message in speech_started] == [1.0, 1.7]


@pytest.mark.asyncio
async def test_pending_buffer_cannot_move_endpoint_into_later_words():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options(endpointing_ms=300))
    one = _token(0.0, 0.5, "one")
    two = _token(1.0, 1.5, "two")
    three = _token(1.5, 2.0, "three")

    await adapter.process_update(
        FrontData(lines=[_speech_line(one)], buffer_transcription="pending")
    )
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 0.5))
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 0.9))
    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 0.5)
    )
    await adapter.process_stream_event(AudioStreamEvent("speech_started", 1.0))
    await adapter.process_update(
        FrontData(
            lines=[_speech_line(one, two)],
            buffer_transcription="pending",
        )
    )
    await adapter.process_update(
        FrontData(
            lines=[_speech_line(one, two, three)],
            buffer_transcription="pending",
        )
    )
    assert not _results(websocket, final=True)

    await adapter.process_update(
        FrontData(lines=[_speech_line(one, two, three)])
    )
    await adapter.finish_stream(2.0)

    finals = _results(websocket, final=True)
    assert [_transcript(message) for message in finals] == ["one", "two three"]
    assert [message["speech_final"] for message in finals] == [True, False]


@pytest.mark.asyncio
async def test_utterance_end_has_an_independent_deduplicated_word_gap_timer():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(
        websocket,
        _options(endpointing_ms=None, utterance_end_ms=1000),
    )
    hello = _token(0.0, 0.5, "hello")
    later = _token(2.0, 2.4, "later")

    await adapter.process_update(FrontData(lines=[_speech_line(hello)]))
    await adapter.process_update(FrontData(lines=[_speech_line(hello, later)]))
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 1.499))
    assert not [message for message in websocket.sent if message.get("type") == "UtteranceEnd"]

    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 1.5))
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 2.5))

    events = [message for message in websocket.sent if message.get("type") == "UtteranceEnd"]
    assert events == [
        {
            "type": "UtteranceEnd",
            "channel": [0, 1],
            "last_word_end": 0.5,
        }
    ]


@pytest.mark.asyncio
async def test_late_final_rechecks_utterance_end_against_current_audio_clock():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(
        websocket,
        _options(endpointing_ms=None, utterance_end_ms=1000),
    )
    hello = _token(0.0, 0.5, "hello")
    later = _token(3.1, 3.5, "later")

    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 3.0))
    await adapter.process_update(FrontData(lines=[_speech_line(hello)]))
    await adapter.process_update(FrontData(lines=[_speech_line(hello, later)]))

    final_and_gap = [
        message
        for message in websocket.sent
        if message.get("type") == "UtteranceEnd"
        or (message.get("type") == "Results" and message["is_final"])
    ]
    assert [message["type"] for message in final_and_gap] == [
        "Results",
        "UtteranceEnd",
    ]
    assert final_and_gap[-1]["last_word_end"] == 0.5


@pytest.mark.asyncio
async def test_gap_inside_one_final_result_does_not_emit_utterance_end():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(
        websocket,
        _options(endpointing_ms=None, utterance_end_ms=1000),
    )
    words = [_token(0.0, 0.5, "hello"), _token(2.0, 2.4, "again")]

    await adapter.process_update(FrontData(lines=[_speech_line(*words)]))
    await adapter.finish_stream(3.0)

    assert not [message for message in websocket.sent if message.get("type") == "UtteranceEnd"]


@pytest.mark.asyncio
async def test_speech_started_uses_vad_timestamp_and_deepgram_channel_field():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options(vad_events=True))

    await adapter.process_stream_event(AudioStreamEvent("speech_started", 1.25))
    await adapter.process_stream_event(AudioStreamEvent("speech_started", 1.30))
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 2.0))
    await adapter.process_stream_event(AudioStreamEvent("speech_started", 2.4))
    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 2.0)
    )

    events = [message for message in websocket.sent if message.get("type") == "SpeechStarted"]
    assert events == [
        {"type": "SpeechStarted", "channel": [0, 1], "timestamp": 1.25},
        {"type": "SpeechStarted", "channel": [0, 1], "timestamp": 2.4},
    ]


@pytest.mark.asyncio
async def test_results_never_replace_authoritative_vad_speech_timestamp():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(
        websocket,
        _options(vad_events=True, endpointing_ms=None),
    )

    await adapter.process_update(
        FrontData(lines=[_speech_line(_token(0.15, 0.5, "hello"))])
    )
    assert not [
        message for message in websocket.sent if message.get("type") == "SpeechStarted"
    ]

    await adapter.process_stream_event(AudioStreamEvent("speech_started", 0.08))
    assert [
        message for message in websocket.sent if message.get("type") == "SpeechStarted"
    ] == [{"type": "SpeechStarted", "channel": [0, 1], "timestamp": 0.08}]


@pytest.mark.asyncio
async def test_endpoint_flush_does_not_invent_a_second_speech_started():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(websocket, _options(vad_events=True))
    hello = _token(0.0, 0.5, "hello")

    await adapter.process_stream_event(AudioStreamEvent("speech_started", 0.0))
    await adapter.process_update(FrontData(lines=[_speech_line(hello)]))
    await adapter.process_stream_event(AudioStreamEvent("silence_started", 0.5))
    await adapter.process_stream_event(
        AudioStreamEvent("silence_transcription_ready", 0.5)
    )
    await adapter.process_stream_event(AudioStreamEvent("audio_advanced", 0.8))

    events = [message for message in websocket.sent if message.get("type") == "SpeechStarted"]
    assert events == [
        {"type": "SpeechStarted", "channel": [0, 1], "timestamp": 0.0}
    ]


@pytest.mark.asyncio
async def test_metadata_and_results_include_current_sdk_fields():
    websocket = RecordingWebSocket()
    adapter = DeepgramAdapter(
        websocket,
        _options(endpointing_ms=None),
        SimpleNamespace(backend="faster-whisper"),
    )

    await adapter.send_metadata()
    await adapter.process_update(
        FrontData(lines=[_speech_line(_token(0.0, 0.5, "hello"))])
    )
    await adapter.finish_stream(0.5)
    await adapter.send_summary_metadata(0.5)

    initial, summary = [message for message in websocket.sent if message["type"] == "Metadata"]
    assert initial["transaction_key"] == "deprecated"
    assert summary["duration"] == 0.5
    result = _results(websocket, final=True)[0]
    assert result["metadata"]["request_id"] == adapter.request_id
    assert result["metadata"]["model_info"]["name"] == "faster-whisper"
    assert result["metadata"]["model_uuid"] == adapter.model_uuid


class DelayedFinalProcessor:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.total_pcm_samples = 8000
        self.sample_rate = 16000
        self.stop = asyncio.Event()
        self.cleanup_called = False
        self.create_tasks_calls = 0
        self.processed = []
        type(self).instances.append(self)

    async def create_tasks(self):
        self.create_tasks_calls += 1

        async def results():
            await self.stop.wait()
            await asyncio.sleep(0)
            yield FrontData(
                lines=[_speech_line(_token(0.0, 0.5, "tail"))]
            )

        return results()

    async def process_audio(self, data):
        self.processed.append(data)
        if not data:
            self.stop.set()

    async def cleanup(self):
        self.cleanup_called = True


@pytest.mark.asyncio
async def test_close_stream_drains_results_before_summary_and_close(monkeypatch):
    from whisperlivekit import audio_processor as audio_processor_module

    DelayedFinalProcessor.instances = []
    monkeypatch.setattr(audio_processor_module, "AudioProcessor", DelayedFinalProcessor)
    websocket = RecordingWebSocket(
        {"endpointing": "false", "context": "WhisperLiveKit, Qwen3-ASR"},
        [{"type": "websocket.receive", "text": '{"type":"CloseStream"}'}],
    )

    await handle_deepgram_websocket(
        websocket,
        transcription_engine=object(),
        config=SimpleNamespace(vac=True, backend="fake"),
    )

    processor = DelayedFinalProcessor.instances[0]
    assert processor.kwargs["context"] == "WhisperLiveKit, Qwen3-ASR"
    assert processor.create_tasks_calls == 1
    assert processor.processed == [b""]
    assert processor.cleanup_called
    assert websocket.closed[0] == 1000
    assert [message["type"] for message in websocket.sent] == [
        "Metadata",
        "Results",
        "Metadata",
    ]
    assert _transcript(websocket.sent[1]) == "tail"
    assert websocket.sent[-1]["duration"] == 0.5


@pytest.mark.asyncio
async def test_frontdata_error_is_forwarded_and_closes_immediately(monkeypatch):
    from whisperlivekit import audio_processor as audio_processor_module

    class ErrorProcessor(DelayedFinalProcessor):
        async def create_tasks(self):
            self.create_tasks_calls += 1

            async def results():
                yield FrontData(status="error", error="decoder failed")

            return results()

    ErrorProcessor.instances = []
    monkeypatch.setattr(audio_processor_module, "AudioProcessor", ErrorProcessor)
    websocket = RecordingWebSocket({"endpointing": "false"})

    await handle_deepgram_websocket(
        websocket,
        transcription_engine=object(),
        config=SimpleNamespace(vac=True, backend="fake"),
    )

    processor = ErrorProcessor.instances[0]
    assert processor.cleanup_called
    assert websocket.closed[0] == 1011
    assert websocket.sent[-1]["type"] == "Error"
    assert websocket.sent[-1]["description"] == "decoder failed"


@pytest.mark.asyncio
async def test_create_tasks_failure_closes_and_cleans_up(monkeypatch):
    from whisperlivekit import audio_processor as audio_processor_module

    class SetupFailureProcessor(DelayedFinalProcessor):
        async def create_tasks(self):
            self.create_tasks_calls += 1
            raise RuntimeError("setup exploded")

    SetupFailureProcessor.instances = []
    monkeypatch.setattr(
        audio_processor_module,
        "AudioProcessor",
        SetupFailureProcessor,
    )
    websocket = RecordingWebSocket({"endpointing": "false"})

    await handle_deepgram_websocket(
        websocket,
        transcription_engine=object(),
        config=SimpleNamespace(vac=True, backend="fake"),
    )

    processor = SetupFailureProcessor.instances[0]
    assert processor.create_tasks_calls == 1
    assert processor.cleanup_called
    assert websocket.closed[0] == 1011
    assert [message["type"] for message in websocket.sent] == [
        "Metadata",
        "Error",
    ]


@pytest.mark.asyncio
async def test_handshake_disconnect_still_cleans_up_processor(monkeypatch):
    from whisperlivekit import audio_processor as audio_processor_module

    class DisconnectOnAccept(RecordingWebSocket):
        async def accept(self):
            raise RuntimeError("client left during handshake")

    DelayedFinalProcessor.instances = []
    monkeypatch.setattr(
        audio_processor_module,
        "AudioProcessor",
        DelayedFinalProcessor,
    )
    websocket = DisconnectOnAccept({"endpointing": "false"})

    await handle_deepgram_websocket(
        websocket,
        transcription_engine=object(),
        config=SimpleNamespace(vac=True, backend="fake"),
    )

    assert DelayedFinalProcessor.instances[0].cleanup_called
    assert websocket.closed[0] == 1011


@pytest.mark.asyncio
async def test_result_generator_failure_closes_without_waiting_for_timeout(monkeypatch):
    from whisperlivekit import audio_processor as audio_processor_module

    class GeneratorFailureProcessor(DelayedFinalProcessor):
        async def create_tasks(self):
            self.create_tasks_calls += 1

            async def results():
                if False:
                    yield None
                raise RuntimeError("generator exploded")

            return results()

    GeneratorFailureProcessor.instances = []
    monkeypatch.setattr(
        audio_processor_module,
        "AudioProcessor",
        GeneratorFailureProcessor,
    )
    websocket = RecordingWebSocket({"endpointing": "false"})

    await asyncio.wait_for(
        handle_deepgram_websocket(
            websocket,
            transcription_engine=object(),
            config=SimpleNamespace(vac=True, backend="fake"),
        ),
        timeout=1,
    )

    processor = GeneratorFailureProcessor.instances[0]
    assert processor.cleanup_called
    assert websocket.closed[0] == 1011
    assert websocket.sent[-1]["description"] == "Transcription result stream failed."


@pytest.mark.asyncio
async def test_event_consumer_failure_closes_without_waiting_for_timeout(monkeypatch):
    from whisperlivekit import audio_processor as audio_processor_module

    class MalformedResultProcessor(DelayedFinalProcessor):
        async def create_tasks(self):
            self.create_tasks_calls += 1

            async def results():
                yield FrontData(lines=[object()])

            return results()

    MalformedResultProcessor.instances = []
    monkeypatch.setattr(
        audio_processor_module,
        "AudioProcessor",
        MalformedResultProcessor,
    )
    websocket = RecordingWebSocket({"endpointing": "false"})

    await asyncio.wait_for(
        handle_deepgram_websocket(
            websocket,
            transcription_engine=object(),
            config=SimpleNamespace(vac=True, backend="fake"),
        ),
        timeout=1,
    )

    assert MalformedResultProcessor.instances[0].cleanup_called
    assert websocket.closed[0] == 1011
    assert websocket.sent[-1]["description"] == "Transcription event stream failed."


@pytest.mark.asyncio
async def test_ordered_results_are_sent_before_later_stream_error(monkeypatch):
    from whisperlivekit import audio_processor as audio_processor_module

    class SlowRecordingWebSocket(RecordingWebSocket):
        async def send_json(self, message):
            if message.get("type") == "Results":
                await asyncio.sleep(0.01)
            await super().send_json(message)

    class ResultsThenErrorProcessor(DelayedFinalProcessor):
        async def create_tasks(self):
            self.create_tasks_calls += 1

            async def results():
                one = _token(0.0, 0.2, "one")
                two = _token(0.2, 0.4, "two")
                three = _token(0.4, 0.6, "three")
                yield FrontData(lines=[_speech_line(one)])
                yield FrontData(lines=[_speech_line(one, two)])
                yield FrontData(lines=[_speech_line(one, two, three)])
                yield FrontData(status="error", error="late failure")

            return results()

    ResultsThenErrorProcessor.instances = []
    monkeypatch.setattr(
        audio_processor_module,
        "AudioProcessor",
        ResultsThenErrorProcessor,
    )
    websocket = SlowRecordingWebSocket(
        {"endpointing": "false", "interim_results": "true"}
    )

    await asyncio.wait_for(
        handle_deepgram_websocket(
            websocket,
            transcription_engine=object(),
            config=SimpleNamespace(vac=True, backend="fake"),
        ),
        timeout=1,
    )

    assert [_transcript(message) for message in _results(websocket, final=True)] == [
        "one",
        "two",
    ]
    assert websocket.sent[-1]["type"] == "Error"
    assert websocket.sent[-1]["description"] == "late failure"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("incoming", "error_fragment"),
    [
        (
            {"type": "websocket.receive", "bytes": b""},
            "Empty binary frames",
        ),
        (
            {"type": "websocket.receive", "text": '{"type":"Finalize"}'},
            "Finalize is not supported",
        ),
    ],
)
async def test_unsupported_terminal_inputs_fail_explicitly(
    monkeypatch,
    incoming,
    error_fragment,
):
    from whisperlivekit import audio_processor as audio_processor_module

    DelayedFinalProcessor.instances = []
    monkeypatch.setattr(audio_processor_module, "AudioProcessor", DelayedFinalProcessor)
    websocket = RecordingWebSocket(
        {"endpointing": "false"},
        [incoming],
    )

    await handle_deepgram_websocket(
        websocket,
        transcription_engine=object(),
        config=SimpleNamespace(vac=True, backend="fake"),
    )

    processor = DelayedFinalProcessor.instances[0]
    assert processor.processed == []
    assert processor.create_tasks_calls == 1
    assert processor.cleanup_called
    assert websocket.closed[0] == 4400
    assert error_fragment in websocket.sent[-1]["description"]


@pytest.mark.asyncio
async def test_invalid_query_is_rejected_before_processor_construction(monkeypatch):
    from whisperlivekit import audio_processor as audio_processor_module

    def fail_if_constructed(**kwargs):
        raise AssertionError(f"processor should not be constructed: {kwargs}")

    monkeypatch.setattr(audio_processor_module, "AudioProcessor", fail_if_constructed)
    websocket = RecordingWebSocket({"utterance_end_ms": "1000"})

    await handle_deepgram_websocket(
        websocket,
        transcription_engine=object(),
        config=SimpleNamespace(vac=True),
    )

    assert websocket.accepted
    assert websocket.closed[0] == 4400
    assert "requires interim_results=true" in websocket.sent[0]["description"]


@pytest.mark.asyncio
async def test_audio_processor_emits_sample_clock_silence_events():
    from whisperlivekit.audio_processor import AudioProcessor
    from whisperlivekit.metrics_collector import SessionMetrics
    from whisperlivekit.timed_objects import Silence, State

    processor = object.__new__(AudioProcessor)
    processor.stream_event_queue = asyncio.Queue()
    processor.sample_rate = 100
    processor.total_pcm_samples = 100
    processor.current_silence = None
    processor.transcription_queue = None
    processor.diarization_queue = None
    processor.translation_queue = None
    processor.args = SimpleNamespace(diarization=False)
    processor.metrics = SessionMetrics()
    processor.state = State()
    processor.pause_segmentation_seconds = 5.0

    await processor._begin_silence(at_sample=125)
    assert isinstance(processor.current_silence, Silence)
    await processor._end_silence(at_sample=150, speech_resumed=True)

    events = []
    while not processor.stream_event_queue.empty():
        events.append(processor.stream_event_queue.get_nowait())
    assert events == [
        AudioStreamEvent("silence_started", 1.25),
        AudioStreamEvent("silence_transcription_ready", 1.25),
        AudioStreamEvent("speech_started", 1.5),
    ]


@pytest.mark.asyncio
async def test_transcription_ready_event_waits_for_matching_snapshot():
    from whisperlivekit.audio_processor import AudioProcessor
    from whisperlivekit.metrics_collector import SessionMetrics
    from whisperlivekit.timed_objects import State

    processor = object.__new__(AudioProcessor)
    processor.stream_event_queue = asyncio.Queue()
    processor._stream_events_after_snapshot = asyncio.Queue()
    processor._ffmpeg_error = None
    processor.tokens_alignment = SimpleNamespace(
        update=lambda: None,
        get_lines=lambda **kwargs: ([], "", ""),
    )
    processor.args = SimpleNamespace(diarization=False)
    processor.translation = None
    processor.total_pcm_samples = 0
    processor.sample_rate = 16000
    processor.last_response_content = FrontData(status="no_audio_detected")
    processor.metrics = SessionMetrics()
    processor.is_stopping = False

    async def get_current_state():
        return State()

    processor.get_current_state = get_current_state
    barrier = asyncio.create_task(
        processor._emit_stream_event_after_snapshot(
            "silence_transcription_ready",
            1.25,
        )
    )
    await asyncio.sleep(0)
    assert not barrier.done()
    assert processor.stream_event_queue.empty()

    generator = processor.results_formatter()
    response = await anext(generator)
    assert response.status == "no_audio_detected"
    assert processor.stream_event_queue.empty()

    resume = asyncio.create_task(anext(generator))
    for _ in range(20):
        if not processor.stream_event_queue.empty():
            break
        await asyncio.sleep(0)
    assert processor.stream_event_queue.get_nowait() == AudioStreamEvent(
        "silence_transcription_ready",
        1.25,
    )
    await asyncio.wait_for(barrier, timeout=1)
    resume.cancel()
    await asyncio.gather(resume, return_exceptions=True)
    await generator.aclose()


@pytest.mark.asyncio
async def test_current_deepgram_sdk_receives_typed_exactly_once_close_stream(
    monkeypatch,
):
    import socket

    import uvicorn
    from deepgram import AsyncDeepgramClient
    from deepgram.environment import DeepgramClientEnvironment
    from deepgram.listen.v1.types.listen_v1metadata import ListenV1Metadata
    from deepgram.listen.v1.types.listen_v1results import ListenV1Results
    from fastapi import FastAPI, WebSocket

    from whisperlivekit import audio_processor as audio_processor_module

    class SdkProcessor(DelayedFinalProcessor):
        async def create_tasks(self):
            self.create_tasks_calls += 1

            async def results():
                await self.stop.wait()
                hello = _token(0.0, 0.25, "hello")
                world = _token(0.25, 0.5, "world")
                yield FrontData(lines=[_speech_line(hello)])
                yield FrontData(lines=[_speech_line(hello, world)])

            return results()

    monkeypatch.setattr(audio_processor_module, "AudioProcessor", SdkProcessor)
    app = FastAPI()
    config = SimpleNamespace(vac=True, backend="fake")

    @app.websocket("/v1/listen")
    async def listen(websocket: WebSocket):
        await handle_deepgram_websocket(websocket, object(), config)

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", lifespan="off")
    )
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.01)
    assert server.started

    origin = f"127.0.0.1:{port}"
    environment = DeepgramClientEnvironment(
        base=f"http://{origin}",
        production=f"ws://{origin}",
        agent=f"ws://{origin}",
        agent_rest=f"http://{origin}",
    )
    client = AsyncDeepgramClient(api_key="unused", environment=environment)
    received = []
    try:
        async with client.listen.v1.connect(
            model="nova-3",
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            interim_results=True,
            endpointing=False,
        ) as connection:
            await connection.send_media(b"\x00\x00")
            initial = await asyncio.wait_for(connection.recv(), timeout=2)
            assert isinstance(initial, ListenV1Metadata)
            assert initial.transaction_key == "deprecated"

            await connection.send_close_stream()
            while True:
                message = await asyncio.wait_for(connection.recv(), timeout=2)
                received.append(message)
                if isinstance(message, ListenV1Metadata):
                    break
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=5)
        listener.close()

    finals = [
        message
        for message in received
        if isinstance(message, ListenV1Results) and message.is_final
    ]
    assert [message.channel.alternatives[0].transcript for message in finals] == [
        "hello",
        "world",
    ]
    assert all(message.metadata.request_id for message in finals)
    assert all(message.metadata.model_uuid for message in finals)
    assert received[-1].duration == 0.5
    processor = SdkProcessor.instances[-1]
    assert processor.kwargs["pcm_input"] is True
    assert processor.processed == [b"\x00\x00", b""]
