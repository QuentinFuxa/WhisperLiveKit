#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:-${ROOT_DIR}/data/ledger.jsonl}"
OUTPUT="${2:-${ROOT_DIR}/finance/ledger.beancount}"

echo "[Finance] Exporting ${INPUT} → ${OUTPUT}"
python -m src.finance.export_beancount --input "${INPUT}" --out "${OUTPUT}"
echo "[Finance] Done."
