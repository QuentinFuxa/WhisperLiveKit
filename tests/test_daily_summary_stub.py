import json
from types import SimpleNamespace

import pytest

from src.gpt_postproc.config import GPTConfig
from src.gpt_postproc.daily_summary import summarize_day
from src.gpt_postproc.ledger_store import LedgerStore


def test_group_by_day(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    entries = [
        {"start": 1731300000.0, "input": "test 1"},
        {"start": 1731386400.0, "input": "test 2"},
    ]
    with open(ledger_path, "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")

    store = LedgerStore(str(ledger_path))
    groups = store.group_by_day()
    assert len(groups.keys()) == 2


class _DummyCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class _DummyClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=_DummyCompletions(content))


@pytest.mark.asyncio
async def test_summarize_day_outputs(tmp_path) -> None:
    entries = [
        {"session_id": 1, "input": "Koupil jsem kávu.", "start": 1731300000.0},
        {"session_id": 1, "input": "Přidej úkol.", "start": 1731300600.0},
    ]
    content = json.dumps([{"expense": "kava"}]) + "---Souhrn dne"
    cfg = GPTConfig(ledger_path=str(tmp_path / "ledger.jsonl"))

    ledger_out, summary_out = await summarize_day(
        "2024-11-11",
        entries,
        cfg=cfg,
        client=_DummyClient(content),
        output_dir=tmp_path,
    )

    with open(ledger_out, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    assert json.loads(lines[0])["expense"] == "kava"

    with open(summary_out, "r", encoding="utf-8") as fh:
        assert "Souhrn dne" in fh.read()
