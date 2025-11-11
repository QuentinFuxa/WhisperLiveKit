"""Configuration helpers for the finance exporter."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import yaml


def _load_map(path: Path) -> Dict[str, str]:
    if not path or not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return {}
        # normalize keys for case-insensitive lookup
        return {str(k).strip().lower(): str(v).strip() for k, v in data.items()}
    except Exception:
        return {}


@dataclass(slots=True)
class FinanceConfig:
    """Houses mapping + currency defaults for Beancount export."""

    default_currency: str = field(default_factory=lambda: os.getenv("FINANCE_DEFAULT_CURRENCY", "CZK"))
    cash_account: str = field(default_factory=lambda: os.getenv("FINANCE_CASH_ACCOUNT", "Assets:Cash:DayMind"))
    income_root: str = field(default_factory=lambda: os.getenv("FINANCE_INCOME_ROOT", "Income"))
    expense_root: str = field(default_factory=lambda: os.getenv("FINANCE_EXPENSE_ROOT", "Expenses"))
    liability_root: str = field(default_factory=lambda: os.getenv("FINANCE_LIABILITY_ROOT", "Liabilities"))
    asset_root: str = field(default_factory=lambda: os.getenv("FINANCE_ASSET_ROOT", "Assets"))
    category_map: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_inputs(
        cls,
        *,
        currency: str | None = None,
        cash_account: str | None = None,
        map_file: str | None = None,
    ) -> "FinanceConfig":
        cfg = cls()
        if currency:
            cfg.default_currency = currency
        if cash_account:
            cfg.cash_account = cash_account
        if map_file:
            cfg.category_map = _load_map(Path(map_file))
        else:
            default_map_path = Path(os.getenv("FINANCE_CATEGORY_MAP", ""))
            if default_map_path:
                cfg.category_map = _load_map(default_map_path)
        return cfg

    def account_for(self, category: str | None, entry_type: str | None) -> str:
        key = (category or "").strip().lower()
        if key and key in self.category_map:
            return self.category_map[key]

        cleaned = key.title().replace(" ", "")
        if not cleaned:
            cleaned = "Unknown"

        entry_type = (entry_type or "").lower()
        if entry_type in {"income", "earnings", "revenue"}:
            root = self.income_root
        elif entry_type in {"liability", "debt"}:
            root = self.liability_root
        elif entry_type in {"asset", "transfer"}:
            root = self.asset_root
        else:
            root = self.expense_root

        return f"{root}:{cleaned}"
