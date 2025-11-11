import json
from types import SimpleNamespace

import pytest

from src.gpt_postproc.config import GPTConfig
from src.gpt_postproc.ledger_store import LedgerStore
from src.gpt_postproc.processor import process_transcripts


class DummyCompletions:
    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps({"notes": ["done"]}))
                )
            ]
        )


class DummyClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=DummyCompletions())


@pytest.mark.asyncio
async def test_process_transcripts_appends_ledger(tmp_path) -> None:
    transcript_path = tmp_path / "transcripts.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"

    with open(transcript_path, "w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"text": "Koupil jsem kávu za 120 CZK.", "start": 0.0, "end": 2.0}
            )
            + "\n"
        )

    cfg = GPTConfig(
        api_key="test-key",
        model="mock",
        input_path=str(transcript_path),
        ledger_path=str(ledger_path),
        temperature=0.1,
    )

    ledger = LedgerStore(str(ledger_path))

    count = await process_transcripts(
        cfg=cfg,
        client=DummyClient(),
        ledger=ledger,
        max_segments=5,
        sleep_between=0,
    )

    assert count == 1
    with open(ledger_path, "r", encoding="utf-8") as fh:
        data = json.loads(fh.readline())

    assert data["session_id"] == 1
    assert data["gap"] == 0
    assert "Koupil" in data["input"]
    assert "notes" in json.loads(data["gpt_output"])
