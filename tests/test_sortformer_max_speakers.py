"""Regression tests for the Sortformer speaker-channel cap."""

import importlib
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
import torch

from whisperlivekit.config import WhisperLiveKitConfig
from whisperlivekit.timed_objects import ASRToken, State
from whisperlivekit.tokens_alignment import TokensAlignment

_MISSING = object()


@pytest.fixture
def sortformer_module():
    module_names = (
        "nemo",
        "nemo.collections",
        "nemo.collections.asr",
        "nemo.collections.asr.models",
        "nemo.collections.asr.modules",
        "whisperlivekit.diarization.sortformer_backend",
    )
    previous = {name: sys.modules.get(name, _MISSING) for name in module_names}

    nemo = ModuleType("nemo")
    collections = ModuleType("nemo.collections")
    asr = ModuleType("nemo.collections.asr")
    models = ModuleType("nemo.collections.asr.models")
    modules = ModuleType("nemo.collections.asr.modules")
    models.SortformerEncLabelModel = object
    modules.AudioToMelSpectrogramPreprocessor = object
    sys.modules.update(
        {
            "nemo": nemo,
            "nemo.collections": collections,
            "nemo.collections.asr": asr,
            "nemo.collections.asr.models": models,
            "nemo.collections.asr.modules": modules,
        }
    )
    sys.modules.pop("whisperlivekit.diarization.sortformer_backend", None)

    try:
        yield importlib.import_module(
            "whisperlivekit.diarization.sortformer_backend"
        )
    finally:
        for name, old_module in previous.items():
            if old_module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def _prediction_processor(sortformer_module, predictions, max_speakers):
    online = object.__new__(sortformer_module.SortformerDiarizationOnline)
    online.total_preds = torch.tensor([predictions], dtype=torch.float32)
    online.max_speakers = max_speakers
    online._len_prediction = None
    online.chunk_duration_seconds = 1.0
    online.segment_lock = threading.Lock()
    online._chunk_index = 0
    online.global_time_offset = 0.0
    return online


def _segment_tuples(segments):
    return [
        (int(segment.speaker), segment.start, segment.end)
        for segment in segments
    ]


def test_two_speaker_cap_keeps_first_arrival_ordered_channels(sortformer_module):
    predictions = [
        [0.90, 0.10, 0.20, 0.05],
        [0.10, 0.80, 0.99, 0.05],
        [0.85, 0.10, 0.99, 0.05],
        [0.80, 0.10, 0.95, 0.05],
    ]
    online = _prediction_processor(sortformer_module, predictions, max_speakers=2)

    assert _segment_tuples(online._process_predictions()) == [
        (0, 0.0, 0.25),
        (1, 0.25, 0.5),
        (0, 0.5, 1.0),
    ]


def test_default_matches_legacy_argmax_across_all_checkpoint_channels(
    sortformer_module,
):
    predictions = [
        [0.90, 0.10, 0.20, 0.05],
        [0.10, 0.80, 0.99, 0.05],
        [0.85, 0.10, 0.99, 0.05],
        [0.80, 0.10, 0.95, 0.05],
    ]
    resolved_default = sortformer_module._resolve_max_speakers(None, 4)
    online = _prediction_processor(
        sortformer_module,
        predictions,
        max_speakers=resolved_default,
    )

    assert resolved_default == 4
    assert _segment_tuples(online._process_predictions()) == [
        (0, 0.0, 0.25),
        (2, 0.25, 1.0),
    ]


def test_cap_preserves_asr_text_and_word_timestamps(sortformer_module):
    predictions = [
        [0.90, 0.10, 0.20, 0.05],
        [0.80, 0.20, 0.30, 0.05],
        [0.10, 0.90, 0.20, 0.05],
        [0.20, 0.80, 0.30, 0.05],
    ]
    default_segments = _prediction_processor(
        sortformer_module,
        predictions,
        max_speakers=4,
    )._process_predictions()
    capped_segments = _prediction_processor(
        sortformer_module,
        predictions,
        max_speakers=2,
    )._process_predictions()

    expected_words = [
        ("Hello ", 0.0, 0.25),
        ("world ", 0.25, 0.5),
        ("again ", 0.5, 0.75),
        ("today", 0.75, 1.0),
    ]

    def render(diarization_segments):
        tokens = [
            ASRToken(start=start, end=end, text=text)
            for text, start, end in expected_words
        ]
        state = State(
            new_tokens=tokens,
            new_diarization=diarization_segments,
        )
        alignment = TokensAlignment(
            state,
            SimpleNamespace(diarization=True),
            " ",
        )
        alignment.update()
        lines, buffer_text, _ = alignment.get_lines(
            diarization=True,
            audio_time=1.0,
        )
        rendered_words = [
            (token.text, token.start, token.end)
            for line in lines
            for token in (line.tokens or [])
        ]
        return lines, buffer_text, rendered_words

    default_lines, default_buffer, default_words = render(default_segments)
    capped_lines, capped_buffer, capped_words = render(capped_segments)

    assert _segment_tuples(default_segments) == _segment_tuples(capped_segments)
    assert default_buffer == capped_buffer == ""
    assert "".join(line.text for line in default_lines) == "Hello world again today"
    assert "".join(line.text for line in capped_lines) == "Hello world again today"
    assert default_words == capped_words == expected_words


def test_cap_does_not_remap_retained_channel_at_chunk_boundary(sortformer_module):
    online = _prediction_processor(
        sortformer_module,
        [
            [0.90, 0.10, 0.05, 0.05],
            [0.80, 0.20, 0.05, 0.05],
            [0.10, 0.90, 0.05, 0.05],
            [0.20, 0.80, 0.05, 0.05],
        ],
        max_speakers=2,
    )
    first = online._process_predictions()

    online.total_preds = torch.tensor(
        [[
            [0.10, 0.90, 0.99, 0.05],
            [0.20, 0.80, 0.99, 0.05],
            [0.90, 0.10, 0.99, 0.05],
            [0.80, 0.20, 0.99, 0.05],
        ]],
        dtype=torch.float32,
    )
    online._chunk_index = 1
    second = online._process_predictions()

    assert _segment_tuples(first) == [
        (0, 0.0, 0.5),
        (1, 0.5, 1.0),
    ]
    assert _segment_tuples(second) == [
        (1, 1.0, 1.5),
        (0, 1.5, 2.0),
    ]


@pytest.mark.parametrize("value", [0, 5, -1, 1.5, True])
def test_config_rejects_unsupported_caps(value):
    with pytest.raises(ValueError, match="integer between 1 and 4"):
        WhisperLiveKitConfig(sortformer_max_speakers=value)


def test_config_rejects_sortformer_option_with_diart():
    with pytest.raises(ValueError, match="requires diarization_backend=sortformer"):
        WhisperLiveKitConfig(
            diarization_backend="diart",
            sortformer_max_speakers=2,
        )


def test_sortformer_cap_preserves_pause_segmentation_config():
    config = WhisperLiveKitConfig(
        pause_segmentation_seconds=1.25,
        sortformer_max_speakers=2,
    )

    assert config.pause_segmentation_seconds == 1.25
    assert config.sortformer_max_speakers == 2


def test_checkpoint_limit_is_validated_at_session_creation(sortformer_module):
    resolve = sortformer_module._resolve_max_speakers

    assert resolve(None, 4) == 4
    assert resolve(2, 4) == 2
    with pytest.raises(ValueError, match="between 1 and 2"):
        resolve(3, 2)


@pytest.mark.parametrize("configured_cap", [None, 2])
def test_online_factory_passes_cap_to_each_sortformer_session(
    monkeypatch,
    sortformer_module,
    configured_cap,
):
    from whisperlivekit.core import online_diarization_factory

    received = {}

    class FakeOnline:
        def __init__(self, **kwargs):
            received.update(kwargs)

    monkeypatch.setattr(sortformer_module, "SortformerDiarizationOnline", FakeOnline)
    shared_model = object()
    args = SimpleNamespace(diarization_backend="sortformer")
    if configured_cap is not None:
        args.sortformer_max_speakers = configured_cap

    online = online_diarization_factory(args, shared_model)

    assert isinstance(online, FakeOnline)
    assert received == {
        "shared_model": shared_model,
        "max_speakers": configured_cap,
    }


def test_parse_args_accepts_two_speaker_cap_with_pause_segmentation(monkeypatch):
    from whisperlivekit.parse_args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wlk",
            "--diarization",
            "--sortformer-max-speakers",
            "2",
            "--pause-segmentation-seconds",
            "1.25",
        ],
    )

    config = parse_args()

    assert config.sortformer_max_speakers == 2
    assert config.pause_segmentation_seconds == 1.25


@pytest.mark.parametrize(
    ("command_name", "runner_name", "arguments", "needs_sounddevice"),
    [
        (
            "cmd_transcribe",
            "_transcribe_files_quiet",
            [
                "--diarization",
                "--sortformer-max-speakers",
                "2",
                "input.wav",
            ],
            False,
        ),
        (
            "cmd_listen",
            "_listen_quiet",
            ["--diarization", "--sortformer-max-speakers", "2"],
            True,
        ),
        (
            "cmd_diagnose",
            "_diagnose_main",
            [
                "input.wav",
                "--diarization",
                "--sortformer-max-speakers",
                "2",
            ],
            False,
        ),
    ],
)
def test_in_process_cli_commands_accept_sortformer_cap(
    monkeypatch,
    command_name,
    runner_name,
    arguments,
    needs_sounddevice,
):
    import whisperlivekit.cli as cli

    captured = {}

    async def fake_runner(parsed):
        captured["parsed"] = parsed

    monkeypatch.setattr(cli, runner_name, fake_runner)
    if needs_sounddevice:
        monkeypatch.setitem(sys.modules, "sounddevice", ModuleType("sounddevice"))

    getattr(cli, command_name)(arguments)

    assert captured["parsed"].diarization is True
    assert captured["parsed"].sortformer_max_speakers == 2


def test_in_process_cli_forwards_only_configured_diarization_options():
    from whisperlivekit.cli import _apply_diarization_cli_kwargs

    kwargs = {"model_size": "base"}
    _apply_diarization_cli_kwargs(
        SimpleNamespace(diarization=True, sortformer_max_speakers=2),
        kwargs,
    )
    assert kwargs == {
        "model_size": "base",
        "diarization": True,
        "sortformer_max_speakers": 2,
    }

    defaults = {"model_size": "base"}
    _apply_diarization_cli_kwargs(
        SimpleNamespace(diarization=False, sortformer_max_speakers=None),
        defaults,
    )
    assert defaults == {"model_size": "base"}
