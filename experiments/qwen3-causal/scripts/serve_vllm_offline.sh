#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-ASR-0.6B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

args=(
  vllm serve "${MODEL}"
  --host "${HOST}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)

if [ "${ENFORCE_EAGER}" = "1" ]; then
  args+=(--enforce-eager)
fi

exec "${args[@]}"
