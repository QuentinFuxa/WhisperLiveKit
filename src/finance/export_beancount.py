"""Export DayMind JSONL ledger entries into Beancount format."""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .config import FinanceConfig


@dataclass
class ExportStats:
    total: int = 0
    written: int = 0
    skipped: int = 0


def _load_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _iter_input_paths(input_args: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for arg in input_args:
        expanded = glob.glob(arg)
        if not expanded:
            paths.append(Path(arg))
            continue
        paths.extend(Path(p) for p in expanded)
    # remove duplicates while preserving order
    seen: set[Path] = set()
    ordered: List[Path] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        ordered.append(p)
    return ordered


def _date_from_entry(entry: dict) -> str:
    ts = entry.get("start") or entry.get("ts") or entry.get("timestamp")
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    elif isinstance(ts, str):
        try:
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except ValueError:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    return dt.date().isoformat()


def _amount(entry: dict) -> Optional[Decimal]:
    value = entry.get("amount") or entry.get("value")
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _currency(entry: dict, cfg: FinanceConfig) -> str:
    cur = entry.get("currency") or entry.get("cur")
    if isinstance(cur, str) and cur.strip():
        return cur.strip().upper()
    return cfg.default_currency


def _payee(entry: dict) -> str:
    for key in ("payee", "vendor", "counterparty"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _narration(entry: dict) -> str:
    for key in ("description", "text", "input"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return entry.get("gpt_output", "").strip() or "DayMind Transcript"


def _since_filter(entry: dict, since_date: Optional[str]) -> bool:
    if not since_date:
        return True
    entry_date = _date_from_entry(entry)
    return entry_date >= since_date


def _transaction_lines(entry: dict, cfg: FinanceConfig) -> Optional[tuple[list[str], list[str]]]:
    amount = _amount(entry)
    if amount is None or amount == 0:
        return None
    currency = _currency(entry, cfg)
    date = _date_from_entry(entry)
    payee = _payee(entry)
    narration = _narration(entry)

    entry_type = (entry.get("type") or "").lower()
    counter_account = cfg.cash_account
    primary_account = cfg.account_for(entry.get("category"), entry_type)

    postings: list[str] = []
    used_accounts = {counter_account, primary_account}

    if entry_type in {"income", "earnings", "revenue"}:
        postings.append(f"  {counter_account:<30} {amount} {currency}")
        postings.append(f"  {primary_account}")
    else:
        postings.append(f"  {primary_account:<30} {amount} {currency}")
        postings.append(f"  {counter_account}")

    header = f"{date} * \"{payee}\" \"{narration}\"".rstrip()
    return header + "\n" + "\n".join(postings), sorted(used_accounts)


def export_beancount(
    *,
    inputs: Sequence[str],
    output: str,
    config: FinanceConfig,
    since: str | None = None,
) -> ExportStats:
    stats = ExportStats()
    input_paths = _iter_input_paths(inputs)
    transactions: list[str] = []
    accounts: set[str] = set()

    for path in input_paths:
        for entry in _load_jsonl(path):
            stats.total += 1
            if not _since_filter(entry, since):
                continue
            result = _transaction_lines(entry, config)
            if not result:
                stats.skipped += 1
                continue
            txn_text, used_accounts = result
            transactions.append(txn_text)
            accounts.update(used_accounts)
            stats.written += 1

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "; Generated by DayMind finance exporter",
        f'option "title" "DayMind Finance Ledger"',
        f'option "operating_currency" "{config.default_currency}"',
        "",
    ]

    open_lines = [f"1970-01-01 open {acc}" for acc in sorted(accounts)]

    body = "\n\n".join(transactions)
    out_path.write_text("\n".join(header + open_lines + ["", body, ""]).strip() + "\n", encoding="utf-8")
    return stats


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DayMind JSONL ledgers into Beancount format.")
    parser.add_argument("--input", "-i", action="append", required=True, help="JSONL source (file or glob).")
    parser.add_argument("--out", required=True, help="Output Beancount file path.")
    parser.add_argument("--currency", default=None, help="Override default currency (e.g., CZK).")
    parser.add_argument("--account-cash", default=None, help="Override cash/balancing account.")
    parser.add_argument("--map-file", default=None, help="Optional YAML mapping category->account.")
    parser.add_argument("--since", default=None, help="Filter entries on/after YYYY-MM-DD.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    config = FinanceConfig.from_inputs(
        currency=args.currency,
        cash_account=args.account_cash,
        map_file=args.map_file,
    )
    stats = export_beancount(inputs=args.input, output=args.out, config=config, since=args.since)
    print(
        f"[Finance] Processed {stats.total} entries → {stats.written} transactions "
        f"(skipped: {stats.skipped}). Output: {args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
