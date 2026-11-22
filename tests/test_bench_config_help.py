"""`wlk bench --config` help text and the read path must agree.

The flag names a JSON file and `_run_bench_new` opens it with
`Path(parsed.config).read_text()`. The help text said "JSON object" for long
enough that an inline object was handed to the flag and failed with
`FileNotFoundError`. The first case reads the help string the bench parser
registers; the second drives the same flag through `cmd_bench` and observes how
the value is actually read. Either side drifting away from "a path to a JSON
file" fails one of them.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import pytest

from whisperlivekit import cli


def _bench_config_help(monkeypatch: pytest.MonkeyPatch) -> str:
    """Return the help string `cmd_bench`'s parser registers for `--config`."""
    recorded: Dict[str, Any] = {}
    add_argument = argparse.ArgumentParser.add_argument

    def spy(self, *names, **kwargs):
        if "--config" in names:
            recorded["help"] = kwargs.get("help", "")
        return add_argument(self, *names, **kwargs)

    monkeypatch.setattr(argparse.ArgumentParser, "add_argument", spy)
    monkeypatch.setattr(cli, "_suppress_logging", lambda: None)
    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_bench(["--help"])
    assert exit_info.value.code == 0
    assert "help" in recorded, "cmd_bench never registered --config"
    return recorded["help"]


def _engine_kwargs_via_cli(monkeypatch: pytest.MonkeyPatch, config_value: str) -> Dict[str, Any]:
    """Run `wlk bench` with `--config config_value` and return the engine options it built.

    The benchmark itself is replaced: only the reading of `--config` is under test.
    """
    captured: Dict[str, Any] = {}

    class _Report:
        n_failed = 0
        warmup_results: tuple = ()
        successful_results: tuple = (object(),)
        corpus_sha256 = None

    class _Runner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self):
            return _Report()

    monkeypatch.setattr("whisperlivekit.benchmark.runner.BenchmarkRunner", _Runner)
    monkeypatch.setattr("whisperlivekit.benchmark.report.print_report", lambda report: None)
    monkeypatch.setattr(cli, "_suppress_logging", lambda: None)
    cli.cmd_bench(["--backend", "whisper", "--model", "tiny", "--config", config_value])
    return captured["engine_kwargs"]


def test_bench_config_help_names_a_file_path(monkeypatch):
    help_text = _bench_config_help(monkeypatch).lower()
    assert "path to a json file" in help_text
    assert "json object" not in help_text
    # The engine-options note the flag has always carried stays.
    assert "engine options" in help_text
    assert "pinned model paths" in help_text


def test_bench_config_reads_the_file_path_the_help_names(monkeypatch, tmp_path):
    config_file = tmp_path / "engine.json"
    config_file.write_text('{"pcm_input": true, "vac": false}')
    assert _engine_kwargs_via_cli(monkeypatch, str(config_file)) == {"pcm_input": True, "vac": False}

    # An inline JSON object is not a path, so the flag rejects it exactly the way
    # the old help text invited a caller to trip over.
    with pytest.raises(FileNotFoundError):
        _engine_kwargs_via_cli(monkeypatch, '{"pcm_input": true}')
