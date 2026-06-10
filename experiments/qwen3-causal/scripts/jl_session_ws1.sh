#!/bin/bash
# Session A / WS1: clean eval of the D1 checkpoint.
# Gate reproduction (bf16 drift), 20-chunk evals at 960/1920ms, position-offset
# probes, long-file pathology check at limit 5.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

R=runs/jl_ws1
mkdir -p "$R"
CKPT=runs/jl_d1_full_en/tower_best.pt
CHUNKS=data/wlk_chunks16_teacher_aligned_split_v0/eval_manifest.jsonl
FULL=data/wlk_teacher_1p7b_v0/manifest.teacher.jsonl
COMMON="--model-id Qwen/Qwen3-ASR-0.6B --audio-backend qwen_audio_causal_kv \
  --qwen-audio-block-bidirectional --qwen-audio-left-context-sec 15 \
  --tower-state-dict $CKPT --repetition-penalty 1.15 --no-repeat-ngram-size 3 \
  --language English --device cuda"

summarize() {
  python - "$1" "$2" << 'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"{sys.argv[2]:34s} WER={m['wer_final_mean']:.4f}  RTF={m['realtime_factor_mean']:.3f}  n={m['count']}")
PYEOF
}

echo "=== start $(date -u +%H:%M:%S)"

echo "--- 0. gate reproduction (bf16 vs fp32-train drift; expect ~0.249)"
python scripts/eval_cached_full_hypothesis.py $COMMON \
  --manifest-jsonl "$CHUNKS" --limit 10 --chunk-ms 960 \
  --output-jsonl "$R/gate_repro_960.jsonl" > "$R/gate_repro.log" 2>&1
summarize "$R/gate_repro_960.summary.json" gate_repro_960_limit10

echo "--- 1. 20 chunks at 960 and 1920"
for C in 960 1920; do
  python scripts/eval_cached_full_hypothesis.py $COMMON \
    --manifest-jsonl "$CHUNKS" --limit 20 --chunk-ms $C \
    --output-jsonl "$R/chunks20_${C}.jsonl" > "$R/chunks20_${C}.log" 2>&1
  summarize "$R/chunks20_${C}.summary.json" "chunks20_${C}"
done

echo "--- 2. position-offset probes (960ms, offsets 1300/4000)"
for OFF in 1300 4000; do
  python scripts/eval_cached_full_hypothesis.py $COMMON \
    --manifest-jsonl "$CHUNKS" --limit 20 --chunk-ms 960 \
    --audio-position-offset $OFF \
    --output-jsonl "$R/chunks20_960_off${OFF}.jsonl" > "$R/off${OFF}.log" 2>&1
  summarize "$R/chunks20_960_off${OFF}.summary.json" "chunks20_960_off${OFF}"
done

echo "--- 3. long files limit 5 (seg200), 960 and 1920"
for C in 960 1920; do
  python scripts/eval_cached_full_hypothesis.py $COMMON \
    --manifest-jsonl "$FULL" --limit 5 --chunk-ms $C \
    --segment-max-cached-steps 200 \
    --output-jsonl "$R/full5_seg200_${C}.jsonl" > "$R/full5_${C}.log" 2>&1
  summarize "$R/full5_seg200_${C}.summary.json" "full5_seg200_${C}"
done

tar czf /home/ubuntu/jl_ws1_artifacts.tgz "$R"
echo "=== done $(date -u +%H:%M:%S)"
