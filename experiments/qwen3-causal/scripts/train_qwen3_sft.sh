#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-ASR-0.6B}"
TRAIN_FILE="${TRAIN_FILE:-data/public_prefix/train.jsonl}"
EVAL_FILE="${EVAL_FILE:-data/public_prefix/eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/qwen3-asr-0.6b-prefix-sft}"
QWEN3_ASR_REPO="${QWEN3_ASR_REPO:-external/Qwen3-ASR}"

BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACC="${GRAD_ACC:-4}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOG_STEPS="${LOG_STEPS:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [ ! -d "${QWEN3_ASR_REPO}/.git" ]; then
  mkdir -p "$(dirname "${QWEN3_ASR_REPO}")"
  git clone --depth 1 https://github.com/QwenLM/Qwen3-ASR.git "${QWEN3_ASR_REPO}"
fi

SCRIPT="${QWEN3_ASR_REPO}/finetuning/qwen3_asr_sft.py"
if [ ! -f "${SCRIPT}" ]; then
  echo "Missing ${SCRIPT}" >&2
  exit 1
fi

if [ ! -f "${TRAIN_FILE}" ]; then
  echo "Missing TRAIN_FILE=${TRAIN_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a CUDA_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC="${NPROC_PER_NODE:-${#CUDA_IDS[@]}}"
else
  NPROC="${NPROC_PER_NODE:-1}"
fi

common_args=(
  "${SCRIPT}"
  --model_path "${MODEL_PATH}"
  --train_file "${TRAIN_FILE}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --grad_acc "${GRAD_ACC}"
  --lr "${LR}"
  --epochs "${EPOCHS}"
  --log_steps "${LOG_STEPS}"
  --save_strategy steps
  --save_steps "${SAVE_STEPS}"
  --save_total_limit 5
  --num_workers "${NUM_WORKERS}"
  --pin_memory 1
  --persistent_workers 1
  --prefetch_factor 2
)

if [ -f "${EVAL_FILE}" ]; then
  common_args+=(--eval_file "${EVAL_FILE}")
fi

if [ "${NPROC}" -gt 1 ]; then
  exec torchrun --nproc_per_node="${NPROC}" "${common_args[@]}"
fi

exec python "${common_args[@]}"
