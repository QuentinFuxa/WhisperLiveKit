import importlib.util
import json
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "tune_stable_commit_from_events.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "tune_stable_commit_from_events",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
tune = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tune)


def test_parse_int_grid_accepts_values_and_ranges():
    assert tune.parse_int_grid("0,2,4:8:2") == [0, 2, 4, 6, 8]


def test_parse_float_grid_accepts_values_and_ranges():
    assert tune.parse_float_grid("0,1.5:2.5:0.5") == [0.0, 1.5, 2.0, 2.5]


def test_replay_text_commit_changes_policy_without_model_outputs():
    events = [
        {"audio_sec": 1.0, "hypothesis": "one two"},
        {"audio_sec": 2.0, "hypothesis": "one two three"},
        {"audio_sec": 3.0, "hypothesis": "one two four"},
    ]

    replayed, committed = tune.replay_text_commit(
        events,
        hold_back_words=0,
        stable_iterations=1,
    )

    assert committed == "one two"
    assert replayed[-1]["committed"] == "one two"
    assert replayed[-1]["committed_units"] == 2


def test_replay_text_commit_respects_min_commit_audio_sec():
    events = [
        {"audio_sec": 1.0, "hypothesis": "one two"},
        {"audio_sec": 2.0, "hypothesis": "one two three"},
        {"audio_sec": 3.0, "hypothesis": "one two four"},
    ]

    replayed, committed = tune.replay_text_commit(
        events,
        hold_back_words=0,
        stable_iterations=1,
        min_commit_audio_sec=3.0,
    )

    assert replayed[1]["committed"] == ""
    assert committed == "one two"


def test_evaluate_policy_reads_events_and_reports_commit_metrics(tmp_path):
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    event_path = events_dir / "sample.jsonl"
    events = [
        {"audio_sec": 1.0, "hypothesis": "hello"},
        {"audio_sec": 2.0, "hypothesis": "hello world"},
        {"audio_sec": 3.0, "hypothesis": "hello world today"},
    ]
    event_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    result = tune.evaluate_policy(
        prediction_rows=[
            {
                "id": "sample",
                "reference": "hello world today",
                "last_hypothesis_text": "hello world today",
                "error": None,
            }
        ],
        events_dir=events_dir,
        hold_back_words=0,
        stable_iterations=1,
    )

    assert result["count"] == 1
    assert result["stable_coverage_ratio_mean"] == 1 / 3
    assert result["first_commit_sec_mean"] == 3.0
    assert result["committed_revision_events_total"] == 0
