#!/usr/bin/env bash
set -euo pipefail

: "${FAVA_PORT:=5000}"
: "${FAVA_LEDGER_PATH:=/opt/daymind/ledger/main.beancount}"

mkdir -p "$(dirname "$FAVA_LEDGER_PATH")"

if [ ! -s "$FAVA_LEDGER_PATH" ]; then
  echo "; DayMind bootstrap ledger" > "$FAVA_LEDGER_PATH"
fi

exec /opt/daymind/venv/bin/fava --host 0.0.0.0 --port "$FAVA_PORT" "$FAVA_LEDGER_PATH"
