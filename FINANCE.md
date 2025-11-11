# DayMind Finance Exporter

US-6.1 introduces a repeatable JSONL → Beancount pipeline so Fava and downstream analytics can consume the GPT ledger artifacts.

## Quick Start

```bash
python -m src.finance.export_beancount \
  --input data/ledger.jsonl \
  --out finance/ledger.beancount
```

- `--currency CZK` – override the default operating currency.
- `--account-cash Assets:Cash:DayMind` – change the balancing account.
- `--map-file finance/category_map.yaml` – provide category → account overrides (YAML, case-insensitive keys).
- `--since 2024-11-01` – only export entries on/after a given date.

The exporter rewrites `finance/ledger.beancount` each run. Generated files include:
- Header + operating currency declaration.
- `open` directives for each account used.
- Dated transactions with payee + narration derived from ledger text fields.

## Category Mapping

1. Create `finance/category_map.yaml`:
   ```yaml
   groceries: Expenses:Household:Groceries
   salary: Income:Primary
   ```
2. Run the exporter with `--map-file finance/category_map.yaml`.
3. Unknown categories fall back to `Expenses:Unknown` (or `Income/Liabilities` depending on the entry `type`).

## Automation Hooks

- Call `python -m src.finance.export_beancount` from cron/GitHub Actions (planned in EPIC-8) to regenerate the Beancount ledger daily.
- `scripts/export_finance.sh` wraps the CLI for local runs or CI steps.
- Next story (US-6.2) wires Fava to serve `finance/ledger.beancount` via `/finance`.

## Fava Dashboard (US-6.2)

Run the dashboard locally:

```bash
python -m src.finance.fava_runner --ledger finance/ledger.beancount --host 0.0.0.0 --port 5000
```

Environment variables:
- `FAVA_LEDGER_PATH` – defaults to `finance/ledger.beancount`
- `FAVA_HOST`, `FAVA_PORT` – network binding for the Flask server

Once running, the FastAPI bridge exposes:
- `/finance` – authenticated redirect to the Fava UI (e.g., `http://localhost:5000/finance/`)
- `/v1/finance` – JSON summary aggregated by date + category; accepts optional `?date=YYYY-MM-DD`

> **Production note:** keep Fava bound to localhost (`FAVA_HOST=127.0.0.1`) and either access it via SSH/VPN or front it with a reverse proxy that re-uses the API key guard. The systemd unit `infra/systemd/daymind-fava.service` mirrors these defaults.

Example response:

```json
{
  "count": 2,
  "items": [
    {"date": "2024-11-10", "category": "Expenses:Food", "total": 120.0, "currency": "CZK"},
    {"date": "2024-11-11", "category": "Income:Salary", "total": 5000.0, "currency": "CZK"}
  ]
}
```
