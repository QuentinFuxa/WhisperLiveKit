#!/bin/bash
# D0 session: untrained block-bidirectional causal-KV eval.
# Protocol matches the mutable-tail sweep (20 held-out WLK 16s chunks,
# decode controls, left 15s, explicit English). Block size = chunk size.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

R=runs/jl_d0_blockbidir
mkdir -p "$R"
echo "=== start $(date -u +%H:%M:%S)"

run_eval() {
  local name="$1"; shift
  python scripts/eval_cached_full_hypothesis.py \
    --model-id Qwen/Qwen3-ASR-0.6B \
    --audio-backend qwen_audio_causal_kv \
    --manifest-jsonl data/wlk_chunks16_teacher_aligned_split_v0/eval_manifest.jsonl \
    --limit 20 \
    --output-jsonl "$R/${name}.jsonl" \
    --device cuda \
    --qwen-audio-left-context-sec 15 \
    --repetition-penalty 1.15 \
    --no-repeat-ngram-size 3 \
    --language English \
    "$@" \
    > "$R/${name}.log" 2>&1
  python - "$R/${name}.summary.json" "$name" << 'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"{sys.argv[2]:32s} WER={m['wer_final_mean']:.4f}  RTF={m['realtime_factor_mean']:.3f}  "
      f"ctx_recompute={m['max_recomputed_context_frames']}")
PYEOF
}

echo "=== block-bidirectional sweep (block = chunk size)"
run_eval bidir_block320ms  --chunk-ms 320  --qwen-audio-block-bidirectional
run_eval bidir_block1000ms --chunk-ms 1000 --qwen-audio-block-bidirectional
run_eval bidir_block2000ms --chunk-ms 2000 --qwen-audio-block-bidirectional

echo "=== block-bidirectional + 1s mutable tail (recompute smooths block seams)"
run_eval bidir_block1000ms_tail1s --chunk-ms 1000 --qwen-audio-block-bidirectional --qwen-audio-mutable-tail-sec 1
run_eval bidir_block2000ms_tail2s --chunk-ms 2000 --qwen-audio-block-bidirectional --qwen-audio-mutable-tail-sec 2

echo "=== control: strict causal at block sizes (mask isolation)"
run_eval strict_block1000ms --chunk-ms 1000
run_eval strict_block2000ms --chunk-ms 2000

tar czf /home/ubuntu/jl_d0_artifacts.tgz "$R"
echo "=== done $(date -u +%H:%M:%S)"
