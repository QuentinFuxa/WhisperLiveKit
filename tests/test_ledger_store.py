import json

from src.gpt_postproc.ledger_store import LedgerStore


def test_ledger_append(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    store = LedgerStore(str(path))
    store.append({"input": "mock text", "gpt_output": "{'ok': true}"})

    assert path.exists()
    with open(path, "r", encoding="utf-8") as fh:
        data = json.loads(fh.readline())

    assert "input" in data and "gpt_output" in data
    assert "ts" in data
