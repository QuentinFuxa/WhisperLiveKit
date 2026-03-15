#!/usr/bin/env python3
"""Standalone Voxtral benchmark — no whisperlivekit imports."""
import json, logging, re, time, wave, queue, threading
import numpy as np
import torch

logging.basicConfig(level=logging.WARNING)
for n in ["transformers","torch","httpx"]:
    logging.getLogger(n).setLevel(logging.ERROR)

from jiwer import wer as compute_wer
from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration, TextIteratorStreamer

def norm(t):
    return re.sub(r' +', ' ', re.sub(r'[^a-z0-9 ]', ' ', t.lower())).strip()

def load_audio(path):
    with wave.open(path, 'r') as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

# Load model
print("Loading Voxtral-Mini-4B...", flush=True)
MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0",
)
print(f"Loaded, GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

def transcribe_batch(audio_np):
    """Simple batch transcription (not streaming)."""
    # Voxtral expects audio as input_features from processor
    inputs = processor(
        audio=audio_np, sampling_rate=16000, return_tensors="pt",
    ).to("cuda:0").to(torch.bfloat16)

    t0 = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=1024)
    t1 = time.perf_counter()

    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return text, t1 - t0

# 1. LibriSpeech test-clean
print("\n=== Voxtral / LibriSpeech test-clean ===", flush=True)
clean = json.load(open("/home/cloud/benchmark_data/metadata.json"))
wers = []; ta = tp = 0
for i, s in enumerate(clean):
    audio = load_audio(s['path'])
    hyp, pt = transcribe_batch(audio)
    w = compute_wer(norm(s['reference']), norm(hyp))
    wers.append(w); ta += s['duration']; tp += pt
    if i < 3 or i % 20 == 0:
        print(f"  [{i}] {s['duration']:.1f}s RTF={pt/s['duration']:.2f} WER={w:.1%} | {hyp[:60]}", flush=True)
clean_wer = np.mean(wers); clean_rtf = tp/ta
print(f"  CLEAN: WER {clean_wer:.2%}, RTF {clean_rtf:.3f} ({len(clean)} samples, {ta:.0f}s)")

# 2. LibriSpeech test-other
print("\n=== Voxtral / LibriSpeech test-other ===", flush=True)
other = json.load(open("/home/cloud/benchmark_data/metadata_other.json"))
wers2 = []; ta2 = tp2 = 0
for i, s in enumerate(other):
    audio = load_audio(s['path'])
    hyp, pt = transcribe_batch(audio)
    w = compute_wer(norm(s['reference']), norm(hyp))
    wers2.append(w); ta2 += s['duration']; tp2 += pt
    if i < 3 or i % 20 == 0:
        print(f"  [{i}] {s['duration']:.1f}s RTF={pt/s['duration']:.2f} WER={w:.1%}", flush=True)
other_wer = np.mean(wers2); other_rtf = tp2/ta2
print(f"  OTHER: WER {other_wer:.2%}, RTF {other_rtf:.3f} ({len(other)} samples, {ta2:.0f}s)")

# 3. ACL6060
print("\n=== Voxtral / ACL6060 ===", flush=True)
acl_results = []
for talk in ["110", "117", "268", "367", "590"]:
    audio = load_audio(f"/home/cloud/acl6060_audio/2022.acl-long.{talk}.wav")
    dur = len(audio) / 16000
    gw = []
    with open(f"/home/cloud/iwslt26-sst/inputs/en/acl6060.ts/gold-jsonl/2022.acl-long.{talk}.jsonl") as f:
        for line in f:
            gw.append(json.loads(line)["text"].strip())
    gold = " ".join(gw)

    # For long audio, process in 30s chunks
    all_hyp = []
    t0 = time.perf_counter()
    chunk_size = 30 * 16000
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start:start + chunk_size]
        if len(chunk) < 1600:  # skip very short tail
            continue
        hyp, _ = transcribe_batch(chunk)
        all_hyp.append(hyp)
    t1 = time.perf_counter()

    full_hyp = " ".join(all_hyp)
    w = compute_wer(norm(gold), norm(full_hyp))
    rtf = (t1 - t0) / dur
    acl_results.append({"talk": talk, "wer": w, "rtf": rtf, "dur": dur})
    print(f"  Talk {talk}: {dur:.0f}s, WER {w:.2%}, RTF {rtf:.3f}", flush=True)

acl_wer = np.mean([r["wer"] for r in acl_results])
acl_rtf = np.mean([r["rtf"] for r in acl_results])
print(f"  ACL6060 AVERAGE: WER {acl_wer:.2%}, RTF {acl_rtf:.3f}")

# Summary
print(f"\n{'='*60}")
print(f"  VOXTRAL BENCHMARK SUMMARY (H100 80GB)")
print(f"{'='*60}")
print(f"  {'Dataset':>25} {'WER':>7} {'RTF':>7}")
print(f"  {'-'*42}")
print(f"  {'LibriSpeech clean':>25} {clean_wer:>6.2%} {clean_rtf:>7.3f}")
print(f"  {'LibriSpeech other':>25} {other_wer:>6.2%} {other_rtf:>7.3f}")
print(f"  {'ACL6060 (5 talks)':>25} {acl_wer:>6.2%} {acl_rtf:>7.3f}")

results = {
    "clean": {"avg_wer": round(float(clean_wer), 4), "rtf": round(float(clean_rtf), 3)},
    "other": {"avg_wer": round(float(other_wer), 4), "rtf": round(float(other_rtf), 3)},
    "acl6060": {"avg_wer": round(float(acl_wer), 4), "avg_rtf": round(float(acl_rtf), 3),
                "talks": [{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in acl_results]},
}
json.dump(results, open("/home/cloud/bench_voxtral_results.json", "w"), indent=2)
print(f"\nSaved to /home/cloud/bench_voxtral_results.json")
