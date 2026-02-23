# WhisperLiveKit Benchmark Report

Benchmark comparing all supported ASR backends, streaming policies, and model sizes on Apple Silicon.
All tests run through the full AudioProcessor pipeline (same code path as production WebSocket).

## Test Environment

| Property | Value |
|----------|-------|
| Hardware | Apple M4, 32 GB RAM |
| OS | macOS 25.3.0 (arm64) |
| Python | 3.13 |
| faster-whisper | 1.2.1 |
| mlx-whisper | installed (via mlx) |
| Voxtral MLX | native MLX backend |
| Voxtral (HF) | transformers-based |
| VAC (Silero VAD) | enabled unless noted |
| Chunk size | 100 ms |
| Pacing | no-realtime (as fast as possible) |

## Audio Test Files

| File | Duration | Language | Speakers | Description |
|------|----------|----------|----------|-------------|
| `00_00_07_english_1_speaker.wav` | 7.2 s | English | 1 | Short dictation with pauses |
| `00_00_16_french_1_speaker.wav` | 16.3 s | French | 1 | French speech with intentional silence gaps |
| `00_00_30_english_3_speakers.wav` | 30.0 s | English | 3 | Multi-speaker conversation |

Ground truth transcripts (`.transcript.json`) with per-word timestamps are hand-verified.

---

## Results

### English -- Short (7.2 s, 1 speaker)

| Backend | Policy | Model | RTF | WER | Timestamp MAE |
|---------|--------|-------|-----|-----|---------------|
| faster-whisper | LocalAgreement | base | 0.20x | 21.1% | 0.080 s |
| faster-whisper | SimulStreaming | base | 0.14x | 0.0% | 0.239 s |
| faster-whisper | LocalAgreement | small | 0.59x | 21.1% | 0.089 s |
| faster-whisper | SimulStreaming | small | 0.39x | 0.0% | 0.221 s |
| mlx-whisper | LocalAgreement | base | 0.05x | 21.1% | 0.080 s |
| mlx-whisper | SimulStreaming | base | 0.14x | 10.5% | 0.245 s |
| mlx-whisper | LocalAgreement | small | 0.16x | 21.1% | 0.089 s |
| mlx-whisper | SimulStreaming | small | 0.20x | 10.5% | 0.226 s |
| voxtral-mlx | voxtral | 4B | 0.32x | 0.0% | 0.254 s |
| voxtral (HF) | voxtral | 4B | 1.29x | 0.0% | 1.876 s |

### English -- Multi-speaker (30.0 s, 3 speakers)

| Backend | Policy | Model | RTF | WER | Timestamp MAE |
|---------|--------|-------|-----|-----|---------------|
| faster-whisper | LocalAgreement | base | 0.24x | 44.7% | 0.235 s |
| faster-whisper | SimulStreaming | base | 0.10x | 5.3% | 0.398 s |
| faster-whisper | LocalAgreement | small | 0.59x | 25.0% | 0.226 s |
| faster-whisper | SimulStreaming | small | 0.26x | 5.3% | 0.387 s |
| mlx-whisper | LocalAgreement | base | 0.06x | 23.7% | 0.237 s |
| mlx-whisper | SimulStreaming | base | 0.11x | 5.3% | 0.395 s |
| mlx-whisper | LocalAgreement | small | 0.13x | 25.0% | 0.226 s |
| mlx-whisper | SimulStreaming | small | 0.20x | 5.3% | 0.394 s |
| voxtral-mlx | voxtral | 4B | 0.31x | 9.2% | 0.176 s |
| voxtral (HF) | voxtral | 4B | 1.00x | 32.9% | 1.034 s |

<p align="center">
<img src="benchmark_chart.png" alt="Benchmark comparison on 30s English" width="800">
</p>

<p align="center">
<img src="benchmark_scatter.png" alt="Speed vs Accuracy tradeoff" width="700">
</p>

### French (16.3 s, 1 speaker, `--language fr`)

| Backend | Policy | Model | RTF | WER | Timestamp MAE |
|---------|--------|-------|-----|-----|---------------|
| faster-whisper | LocalAgreement | base | 0.22x | 25.7% | 3.460 s |
| faster-whisper | SimulStreaming | base | 0.10x | 31.4% | 3.660 s |
| faster-whisper | LocalAgreement | small | 0.76x | 42.9% | 0.051 s |
| faster-whisper | SimulStreaming | small | 0.29x | 25.7% | 0.219 s |
| mlx-whisper | LocalAgreement | base | 0.09x | ~45%* | ~5.0 s* |
| mlx-whisper | SimulStreaming | base | 0.09x | 40.0% | 3.540 s |
| mlx-whisper | LocalAgreement | small | 0.14x | 25.7% | 0.083 s |
| mlx-whisper | SimulStreaming | small | 0.17x | 31.4% | 0.203 s |
| voxtral-mlx | voxtral | 4B | 0.18x | 37.1% | 3.422 s |
| voxtral (HF) | voxtral | 4B | 0.63x | 28.6% | 4.040 s |

\* mlx-whisper + LocalAgreement + base is unstable on this French file (WER fluctuates 34-1037% across runs due to hallucination loops). The `small` model does not have this problem.

**Timestamp note:** The base model produces very high timestamp MAE (3.4-3.7s) on this French file because it misaligns words around the silence gaps. The small model handles this much better (0.05-0.22s MAE). Voxtral also drifts on the silence gaps.

---

## Model Size Comparison (base vs small)

| | base | small | Observation |
|--|------|-------|-------------|
| **RTF** | 0.05-0.24x | 0.13-0.76x | small is 2-3x slower |
| **English WER (SS)** | 0-5.3% | 0-5.3% | No improvement: SimulStreaming already saturates on base |
| **English WER (LA)** | 21-44.7% | 21-25% | small reduces LA errors on longer audio |
| **French WER** | 25-40% | 25-43% | Mixed: depends on backend/policy combo |
| **French timestamps** | 3.4-5.0s MAE | 0.05-0.22s MAE | small is dramatically better for French timestamps |

In short: **base + SimulStreaming** gives the best speed/accuracy tradeoff for English. The small model only helps if you need LocalAgreement (for subtitle-grade timestamps) or non-English languages.

---

## Key Findings

### Speed (RTF = processing time / audio duration, lower is better)

1. **mlx-whisper + LocalAgreement + base** is the fastest combo on Apple Silicon: 0.05-0.06x RTF on English. 30 seconds of audio in under 2 seconds.
2. For **faster-whisper**, SimulStreaming is faster than LocalAgreement. For **mlx-whisper**, it is the opposite: LocalAgreement (0.05-0.06x) outperforms SimulStreaming (0.11-0.14x) on speed.
3. **voxtral-mlx** runs at 0.18-0.32x RTF -- 3-5x slower than mlx-whisper base, but well within real-time.
4. **voxtral (HF transformers)** hits 1.0-1.3x RTF. At the real-time boundary on Apple Silicon. Use the MLX variant instead.
5. The **small** model is 2-3x slower than base across all backends.

### Accuracy (WER = Word Error Rate, lower is better)

1. **SimulStreaming** gives dramatically lower WER than LocalAgreement on the whisper backends. On the 30s English file: 5.3% vs 23-44%.
2. **voxtral-mlx** hits 0% on short English and 9.2% on multi-speaker. It auto-detects language natively. Whisper also supports `--language auto`, but tends to bias towards English on short segments.
3. **LocalAgreement** tends to repeat the last sentence at end-of-stream (a known LCP artifact), inflating WER. This is visible in the 21% WER on the 7s file -- the same 4 extra words appear in every LA run.
4. On **French** with the correct `--language fr`, whisper base achieves 25-40% WER -- comparable to Voxtral's 28-37%. The small model does not consistently improve French WER.

### Timestamps (MAE = Mean Absolute Error on word start times)

1. **LocalAgreement** gives the best timestamps on English (0.08-0.09s MAE).
2. **SimulStreaming** is less precise (0.22-0.40s MAE) but good enough for most applications.
3. On French with silence gaps, **base model timestamps are unreliable** (3.4-5s MAE). The **small model fixes this** (0.05-0.22s MAE). This is the strongest argument for using `small` over `base`.
4. **voxtral-mlx** has good timestamps on English (0.18-0.25s MAE) but drifts on audio with long silence gaps (3.4s MAE on the French file).

### VAC (Voice Activity Classification) Impact

| Backend | Policy | VAC | 7s English WER | 30s English WER |
|---------|--------|-----|----------------|-----------------|
| faster-whisper | LocalAgreement | on | 21.1% | 44.7% |
| faster-whisper | LocalAgreement | off | 100.0% | 100.0% |
| voxtral-mlx | voxtral | on | 0.0% | 9.2% |
| voxtral-mlx | voxtral | off | 0.0% | 9.2% |

- **Whisper backends need VAC** to work in streaming mode. Without it the buffer logic breaks down and you get empty or garbage output.
- **Voxtral is unaffected by VAC** since it handles its own internal chunking. Identical results with or without. VAC still saves compute on silent segments.

---

## Recommendations

| Use Case | Backend | Policy | Model | Notes |
|----------|---------|--------|-------|-------|
| Fastest English (Apple Silicon) | mlx-whisper | SimulStreaming | base | 0.11x RTF, 5.3% WER |
| Fastest English (Linux/GPU) | faster-whisper | SimulStreaming | base | 0.10x RTF, 5.3% WER |
| Best accuracy, English | faster-whisper | SimulStreaming | small | 0.26x RTF, 5.3% WER, still fast |
| Multilingual / auto-detect | voxtral-mlx | voxtral | 4B | 100+ languages, 0.18-0.32x RTF |
| Best timestamps | any | LocalAgreement | small | 0.05-0.09s MAE, good for subtitles |
| Low memory / embedded | mlx-whisper | SimulStreaming | base | Smallest footprint, fastest response |

---

## Caveats

- **3 test files, ~53 seconds total.** Results give relative rankings between backends but should not be taken as definitive WER numbers. Run on your own data for production decisions.
- **RTF varies between runs** (up to +/-30%) depending on thermal state, background processes, and model caching. The numbers above are single sequential runs on a warm machine.
- **Only base and small tested.** Medium and large-v3 would likely improve WER at the cost of higher RTF. We did not test them here because they are slow on Apple Silicon without GPU.

---

## Reproducing These Benchmarks

```bash
# Install test dependencies
pip install -e ".[test]"

# Single backend test
python test_backend_offline.py --backend faster-whisper --policy simulstreaming --model base --no-realtime

# With a specific language
python test_backend_offline.py --backend mlx-whisper --policy simulstreaming --model small --lan fr --no-realtime

# Multi-backend auto-detect benchmark
python test_backend_offline.py --benchmark --no-realtime

# Export to JSON
python test_backend_offline.py --benchmark --no-realtime --json results.json

# Test with your own audio
python test_backend_offline.py --backend voxtral-mlx --audio your_file.wav --no-realtime
```

The benchmark harness computes WER and timestamp accuracy automatically when ground truth
`.transcript.json` files exist alongside the audio files. See `audio_tests/` for the format.

---

## Help Us Benchmark on More Hardware

These results are from a single Apple M4 machine. We'd love to see numbers from other setups: Linux with CUDA GPUs, older Macs, different CPU architectures, cloud instances, etc.

If you run the benchmark on your hardware, please open an issue or PR with your results and we will add them here. The more data points we have, the better the recommendations get.

What we are especially interested in:
- **NVIDIA GPUs** (RTX 3090, 4090, A100, T4, etc.) with faster-whisper
- **Older Apple Silicon** (M1, M2, M3) with mlx-whisper and voxtral-mlx
- **Medium and large-v3 models** (we only tested base and small so far)
- **Longer audio files** or domain-specific audio (medical, legal, call center)
- **Other languages** beyond English and French
