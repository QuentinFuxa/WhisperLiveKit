#!/bin/bash
# H100 session 2026-06-10: offline baselines, bounded mutable-tail sweep,
# realistic-latency eval. Run from the remote workspace root:
#   bash scripts/jl_session_20260610.sh
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
pip install -q whisper-normalizer jiwer pyyaml

R=runs/jl_20260610
mkdir -p "$R"
WLK_AUDIO=data/wlk_audio_full
echo "=== start $(date -u +%H:%M:%S)"

echo "=== 0/4 targeted tests"
python -m pytest -q tests/test_mutable_tail.py tests/test_native_realtime_model.py \
  > "$R/tests.log" 2>&1
tail -1 "$R/tests.log"

echo "=== 1/4 offline baselines vs MCIF human refs"
for MODEL in Qwen/Qwen3-ASR-0.6B Qwen/Qwen3-ASR-1.7B; do
  TAG=$(basename "$MODEL" | tr 'A-Z.' 'a-z_')
  python scripts/offline_baseline.py \
    --model-id "$MODEL" \
    --manifest-jsonl data/mcif_refs/manifest.human.jsonl \
    --audio-dir "$WLK_AUDIO" \
    --reference-field human_text \
    --language English \
    --output-jsonl "$R/offline_${TAG}_human.jsonl" \
    > "$R/offline_${TAG}.log" 2>&1
  echo "offline $MODEL: $(tail -4 "$R/offline_${TAG}.log" | tr '\n' ' ')"
done

echo "=== 2/4 mutable-tail sweep (20 held-out 16s chunks)"
for T in 0 0.5 1 2 4 8 12 16; do
  python scripts/eval_cached_full_hypothesis.py \
    --model-id Qwen/Qwen3-ASR-0.6B \
    --audio-backend qwen_audio_causal_kv \
    --manifest-jsonl data/wlk_chunks16_teacher_aligned_split_v0/eval_manifest.jsonl \
    --limit 20 \
    --output-jsonl "$R/tail_${T}s.jsonl" \
    --device cuda \
    --chunk-ms 320 \
    --qwen-audio-left-context-sec 15 \
    --qwen-audio-mutable-tail-sec "$T" \
    --repetition-penalty 1.15 \
    --no-repeat-ngram-size 3 \
    --language English \
    > "$R/tail_${T}s.log" 2>&1
  python - "$R/tail_${T}s.summary.json" "$T" << 'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"tail={sys.argv[2]:>4s}s  WER={m['wer_final_mean']:.4f}  RTF={m['realtime_factor_mean']:.3f}  "
      f"max_ctx_recompute={m['max_recomputed_context_frames']}")
PYEOF
done

echo "=== 3/4 latency eval, surgery left12/seg200, full WLK"
python scripts/eval_cached_full_hypothesis.py \
  --model-id Qwen/Qwen3-ASR-0.6B \
  --manifest-jsonl data/wlk_teacher_1p7b_v0/manifest.teacher.jsonl \
  --output-jsonl "$R/surgery_left12_seg200_chunk2000.jsonl" \
  --device cuda \
  --chunk-ms 2000 \
  --qwen-audio-left-context-sec 12 \
  --qwen-audio-right-context-ms 640 \
  --segment-max-cached-steps 200 \
  --language English \
  > "$R/surgery_chunk2000.log" 2>&1
python scripts/eval_cached_full_hypothesis.py \
  --model-id Qwen/Qwen3-ASR-0.6B \
  --manifest-jsonl data/wlk_teacher_1p7b_v0/manifest.teacher.jsonl \
  --limit 5 \
  --output-jsonl "$R/surgery_left12_seg200_chunk1000_limit5.jsonl" \
  --device cuda \
  --chunk-ms 1000 \
  --qwen-audio-left-context-sec 12 \
  --qwen-audio-right-context-ms 640 \
  --segment-max-cached-steps 200 \
  --language English \
  > "$R/surgery_chunk1000.log" 2>&1

echo "=== 4/4 pack artifacts"
tar czf /home/ubuntu/jl_20260610_artifacts.tgz "$R" \
  runs/jl_418957/segmented_streamer_full_wlk21_left15s_seg200.jsonl \
  runs/jl_418958/segmented_full_wlk21_left15s_seg360_tail0_ctx0.jsonl \
  2>/dev/null
echo "=== done $(date -u +%H:%M:%S)"
