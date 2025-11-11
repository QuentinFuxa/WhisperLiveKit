#!/usr/bin/env bash
set -euo pipefail

if [ ! -f .env ]; then
  echo "Missing .env file. Copy .env.example and set credentials." >&2
  exit 1
fi

clear
source .env

python -m src.stt_core.livekit_runner &
STT_PID=$!

sleep 3
python -m src.gpt_postproc.daily_summary || true

kill $STT_PID 2>/dev/null || true
