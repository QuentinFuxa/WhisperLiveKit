#!/usr/bin/env python3
"""Benchmark Voxtral via vLLM WebSocket /v1/realtime — proper streaming."""
import asyncio, json, base64, time, wave, re, os
import numpy as np
import websockets
import librosa
from jiwer import wer as compute_wer

MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"
WS_URI = "ws://localhost:8000/v1/realtime"

def norm(t):
    return re.sub(r' +', ' ', re.sub(r'[^a-z0-9 ]', ' ', t.lower())).strip()

async def transcribe(audio_path, max_tokens=4096):
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    pcm16 = (audio * 32767).astype(np.int16).tobytes()
    dur = len(audio) / 16000

    t0 = time.time()
    transcript = ""
    first_token_time = None

    async with websockets.connect(WS_URI, max_size=2**24) as ws:
        await ws.recv()  # session.created
        await ws.send(json.dumps({"type": "session.update", "model": MODEL}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))  # signal ready

        # Send audio in 4KB chunks
        for i in range(0, len(pcm16), 4096):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm16[i:i+4096]).decode(),
            }))

        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        while True:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
                if msg["type"] == "transcription.delta":
                    d = msg.get("delta", "")
                    if d.strip() and first_token_time is None:
                        first_token_time = time.time() - t0
                    transcript += d
                elif msg["type"] == "transcription.done":
                    transcript = msg.get("text", transcript)
                    break
                elif msg["type"] == "error":
                    break
            except asyncio.TimeoutError:
                break

    elapsed = time.time() - t0
    return transcript.strip(), dur, elapsed / dur, first_token_time or elapsed

async def main():
    # Warmup
    print("Warmup...", flush=True)
    await transcribe("/home/cloud/benchmark_data/librispeech_clean_0000.wav")

    # LibriSpeech clean (full 91 samples)
    print("\n=== Voxtral vLLM Realtime / LibriSpeech clean ===", flush=True)
    clean = json.load(open("/home/cloud/benchmark_data/metadata.json"))
    wers = []; ta = tp = 0
    for i, s in enumerate(clean):
        hyp, dur, rtf, fwl = await transcribe(s['path'])
        w = compute_wer(norm(s['reference']), norm(hyp)) if hyp else 1.0
        wers.append(w); ta += dur; tp += dur * rtf
        if i < 3 or i % 20 == 0:
            print(f"  [{i}] {dur:.1f}s RTF={rtf:.3f} FWL={fwl:.2f}s WER={w:.1%} | {hyp[:60]}", flush=True)
    clean_wer = np.mean(wers); clean_rtf = tp / ta
    print(f"  CLEAN ({len(clean)}): WER {clean_wer:.2%}, RTF {clean_rtf:.3f}\n", flush=True)

    # LibriSpeech other (full 133 samples)
    print("=== Voxtral vLLM Realtime / LibriSpeech other ===", flush=True)
    other = json.load(open("/home/cloud/benchmark_data/metadata_other.json"))
    wers2 = []; ta2 = tp2 = 0
    for i, s in enumerate(other):
        hyp, dur, rtf, fwl = await transcribe(s['path'])
        w = compute_wer(norm(s['reference']), norm(hyp)) if hyp else 1.0
        wers2.append(w); ta2 += dur; tp2 += dur * rtf
        if i < 3 or i % 20 == 0:
            print(f"  [{i}] {dur:.1f}s RTF={rtf:.3f} WER={w:.1%}", flush=True)
    other_wer = np.mean(wers2); other_rtf = tp2 / ta2
    print(f"  OTHER ({len(other)}): WER {other_wer:.2%}, RTF {other_rtf:.3f}\n", flush=True)

    # ACL6060 talks
    print("=== Voxtral vLLM Realtime / ACL6060 ===", flush=True)
    acl = []
    for talk in ["110", "117", "268", "367", "590"]:
        gw = []
        with open(f"/home/cloud/iwslt26-sst/inputs/en/acl6060.ts/gold-jsonl/2022.acl-long.{talk}.jsonl") as f:
            for line in f: gw.append(json.loads(line)["text"].strip())
        gold = " ".join(gw)

        hyp, dur, rtf, fwl = await transcribe(f"/home/cloud/acl6060_audio/2022.acl-long.{talk}.wav")
        w = compute_wer(norm(gold), norm(hyp)) if hyp else 1.0
        acl.append({"talk": talk, "wer": round(float(w),4), "rtf": round(float(rtf),3), "dur": round(dur,1)})
        print(f"  Talk {talk}: {dur:.0f}s, WER {w:.2%}, RTF {rtf:.3f}, FWL {fwl:.2f}s", flush=True)

    acl_wer = np.mean([r["wer"] for r in acl])
    acl_rtf = np.mean([r["rtf"] for r in acl])
    print(f"  ACL6060 AVERAGE: WER {acl_wer:.2%}, RTF {acl_rtf:.3f}\n", flush=True)

    # Summary
    print(f"{'='*55}")
    print(f"  VOXTRAL vLLM REALTIME BENCHMARK (H100)")
    print(f"{'='*55}")
    print(f"  LS clean ({len(clean)}): WER {clean_wer:.2%}, RTF {clean_rtf:.3f}")
    print(f"  LS other ({len(other)}): WER {other_wer:.2%}, RTF {other_rtf:.3f}")
    print(f"  ACL6060 (5):     WER {acl_wer:.2%}, RTF {acl_rtf:.3f}")

    results = {
        "clean": {"avg_wer": round(float(clean_wer),4), "rtf": round(float(clean_rtf),3), "n": len(clean)},
        "other": {"avg_wer": round(float(other_wer),4), "rtf": round(float(other_rtf),3), "n": len(other)},
        "acl6060": {"avg_wer": round(float(acl_wer),4), "avg_rtf": round(float(acl_rtf),3), "talks": acl},
    }
    json.dump(results, open("/home/cloud/bench_voxtral_realtime_results.json", "w"), indent=2)
    print(f"\n  Saved to /home/cloud/bench_voxtral_realtime_results.json")

asyncio.run(main())
