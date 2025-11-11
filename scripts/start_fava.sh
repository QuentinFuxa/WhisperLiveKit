#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-/opt/daymind}"

LEDGER_DEFAULT="/opt/daymind/runtime/ledger.beancount"
LEDGER_FILE="${LEDGER_FILE:-${FAVA_LEDGER_PATH:-$LEDGER_DEFAULT}}"
FAVA_PORT="${FAVA_PORT:-8010}"
FAVA_HOST="${FAVA_HOST:-127.0.0.1}"

mkdir -p "$(dirname "$LEDGER_FILE")"

if [ ! -s "$LEDGER_FILE" ]; then
  cat <<'LEDGER' > "$LEDGER_FILE"
option "title" "DayMind Ledger"
option "operating_currency" "USD"

1970-01-01 * "Bootstrap" "Ledger initialized"
  equity:opening-balances  0 USD
LEDGER
fi

exec /opt/daymind/venv/bin/fava --host "$FAVA_HOST" --port "$FAVA_PORT" "$LEDGER_FILE"
