#!/bin/bash
# Session B: KV-cache GPU parity + official long-form eval with the WS2 tower.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

R=runs/jl_sessionB
mkdir -p "$R"
CKPT=runs/jl_ws2_mix_pos_p2b/tower_last.pt
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
print(f"{sys.argv[2]:38s} WER={m['wer_final_mean']:.4f}  RTF={m['realtime_factor_mean']:.3f}  n={m['count']}")
PYEOF
}

echo "=== start $(date -u +%H:%M:%S)"

echo "--- 1. decoder KV-cache parity, 5 chunks, 960ms"
for MODE in off on; do
  python scripts/eval_cached_full_hypothesis.py $COMMON \
    --manifest-jsonl "$CHUNKS" --limit 5 --chunk-ms 960 \
    --decoder-cache $MODE \
    --output-jsonl "$R/parity_${MODE}.jsonl" > "$R/parity_${MODE}.log" 2>&1
  summarize "$R/parity_${MODE}.summary.json" "parity_cache_${MODE}"
done
python - << 'PYEOF'
import json
texts = {}
for mode in ("off", "on"):
    with open(f"runs/jl_sessionB/parity_{mode}.jsonl") as f:
        texts[mode] = [json.loads(l)["final_text"] for l in f]
same = sum(a == b for a, b in zip(texts["off"], texts["on"]))
print(f"parity: {same}/{len(texts['off'])} identical final texts")
PYEOF

echo "--- 2. official long-form: 21 MCIF files, seg200, cache on"
for C in 960 1920; do
  python scripts/eval_cached_full_hypothesis.py $COMMON \
    --manifest-jsonl "$FULL" --chunk-ms $C \
    --segment-max-cached-steps 200 --decoder-cache on \
    --output-jsonl "$R/full21_seg200_${C}.jsonl" > "$R/full21_${C}.log" 2>&1
  summarize "$R/full21_seg200_${C}.summary.json" "full21_seg200_${C}"
done

echo "--- 3. 20-chunk confirmation at both block sizes"
for C in 960 1920; do
  python scripts/eval_cached_full_hypothesis.py $COMMON \
    --manifest-jsonl "$CHUNKS" --limit 20 --chunk-ms $C --decoder-cache on \
    --output-jsonl "$R/chunks20_${C}.jsonl" > "$R/chunks20_${C}.log" 2>&1
  summarize "$R/chunks20_${C}.summary.json" "chunks20_${C}"
done

tar czf /home/ubuntu/jl_sessionB_artifacts.tgz "$R" runs/jl_ws2_mix_pos_p2b/history.json runs/jl_ws2_mix_pos_p2b/final_metrics.json
echo "=== done $(date -u +%H:%M:%S)"
