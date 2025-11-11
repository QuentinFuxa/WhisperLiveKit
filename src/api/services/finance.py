"""Helpers for aggregating Beancount ledgers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Tuple

from beancount.core import data
from beancount.loader import load_file


@dataclass(slots=True)
class FinanceSummary:
    date: str
    category: str
    amount: Decimal
    currency: str


def summarize_ledger(path: Path, *, date: str | None = None) -> List[FinanceSummary]:
    if not path.exists():
        raise FileNotFoundError(path)

    entries, errors, _ = load_file(str(path))
    # Ignore validation errors (missing open directives, etc.); the exporter already writes them.
    del errors  # quiet linters

    buckets: Dict[Tuple[str, str, str], Decimal] = defaultdict(Decimal)

    for entry in entries:
        if not isinstance(entry, data.Transaction):
            continue
        entry_date = entry.date.isoformat()
        if date and entry_date != date:
            continue
        for posting in entry.postings:
            account = posting.account or ""
            if not account:
                continue
            root = account.split(":")[0]
            if root not in {"Expenses", "Income"}:
                continue
            amt = posting.units
            if amt is None:
                continue
            number = Decimal(amt.number)
            currency = getattr(amt, "currency", None) or ""
            if root == "Income":
                number = -number  # flip sign so income totals read positive
            key = (entry_date, account, currency)
            buckets[key] += number

    summaries = [
        FinanceSummary(date=dt, category=acct, amount=value, currency=curr or "")
        for (dt, acct, curr), value in buckets.items()
        if value != 0
    ]
    summaries.sort(key=lambda item: (item.date, item.category))
    return summaries
