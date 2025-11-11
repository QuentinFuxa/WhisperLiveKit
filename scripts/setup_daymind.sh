#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "$ROOT_DIR/.env" ]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  echo "[setup] Copied .env.example to .env - update API keys before running the API."
fi

python3 -m venv "$ROOT_DIR/.venv"
source "$ROOT_DIR/.venv/bin/activate"
pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements.txt"
pip install -e "$ROOT_DIR"

echo "[setup] Dependencies installed. Start the API with:"
echo "[setup]   source .venv/bin/activate && uvicorn src.api.app:app --host 0.0.0.0 --port 8000"
