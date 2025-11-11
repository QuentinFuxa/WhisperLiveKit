import json
from pathlib import Path

from src.finance.config import FinanceConfig
from src.finance.export_beancount import export_beancount


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_export_beancount_basic(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(
        ledger,
        [
            {
                "type": "expense",
                "category": "Food",
                "amount": 120,
                "currency": "CZK",
                "payee": "Bistro",
                "text": "Oběd",
                "start": 1731300000.0,
            },
            {
                "type": "income",
                "category": "Salary",
                "amount": 5000,
                "currency": "CZK",
                "payee": "Acme Corp",
                "text": "Výplata",
                "start": 1731386400.0,
            },
            {
                "type": "expense",
                "amount": 42,
                "text": "Unknown spend",
                "start": 1731472800.0,
            },
        ],
    )
    output = tmp_path / "finance" / "ledger.beancount"
    cfg = FinanceConfig.from_inputs()
    stats = export_beancount(inputs=[str(ledger)], output=str(output), config=cfg)
    data = output.read_text(encoding="utf-8")

    assert stats.written == 3
    assert "Expenses:Food" in data
    assert "Income:Salary" in data
    assert "Expenses:Unknown" in data
    assert "Assets:Cash:DayMind" in data
    assert "option \"operating_currency\" \"CZK\"" in data
    assert "Bistro" in data
    assert "Acme Corp" in data


def test_export_respects_category_map(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(
        ledger,
        [
            {"type": "expense", "category": "Groceries", "amount": 10, "start": 1731300000.0},
        ],
    )
    map_file = tmp_path / "map.yaml"
    map_file.write_text("groceries: Expenses:Household:Groceries\n", encoding="utf-8")
    output = tmp_path / "finance" / "ledger.beancount"
    cfg = FinanceConfig.from_inputs(map_file=str(map_file))
    export_beancount(inputs=[str(ledger)], output=str(output), config=cfg)
    data = output.read_text(encoding="utf-8")
    assert "Expenses:Household:Groceries" in data
