# WhisperLiveKit Benchmark Report

Benchmark comparing all supported ASR backends and streaming policies on Apple Silicon,
using the full AudioProcessor pipeline (the same path audio takes in production via WebSocket).

## Test Environment

| Property | Value |
|----------|-------|
| Hardware | Apple M4, 32 GB RAM |
| OS | macOS 25.3.0 (arm64) |
| Python | 3.13 |
| faster-whisper | 1.2.1 |
| mlx-whisper | installed (via mlx) |
| Voxtral (HF) | transformers-based |
| Voxtral MLX | native MLX backend |
| Model size | `base` (default for whisper backends) |
| VAC (Silero VAD) | enabled unless noted |
| Chunk size | 100 ms |
| Pacing | no-realtime (as fast as possible) |

## Audio Test Files

| File | Duration | Language | Speakers | Description |
|------|----------|----------|----------|-------------|
| `00_00_07_english_1_speaker.wav` | 7.2 s | English | 1 | Short dictation with pauses |
| `00_00_16_french_1_speaker.wav` | 16.3 s | French | 1 | French speech with intentional silence gaps |
| `00_00_30_english_3_speakers.wav` | 30.0 s | English | 3 | Multi-speaker conversation about transcription |

All files have hand-verified ground truth transcripts (`.transcript.json`) with per-word timestamps.

---

## Results Overview

### English - Short (7.2 s, 1 speaker)

| Backend | Policy | RTF | WER | Timestamp MAE |
|---------|--------|-----|-----|---------------|
| faster-whisper | LocalAgreement | 0.20x | 21.1% | 0.080 s |
| faster-whisper | SimulStreaming | 0.14x | 0.0% | 0.239 s |
| mlx-whisper | LocalAgreement | 0.05x | 21.1% | 0.080 s |
| mlx-whisper | SimulStreaming | 0.14x | 10.5% | 0.245 s |
| voxtral-mlx | voxtral | 0.32x | 0.0% | 0.254 s |
| voxtral (HF) | voxtral | 1.29x | 0.0% | 1.876 s |

### French (16.3 s, 1 speaker)

| Backend | Policy | RTF | WER | Timestamp MAE |
|---------|--------|-----|-----|---------------|
| faster-whisper | LocalAgreement | 0.20x | 120.0% | 0.540 s |
| faster-whisper | SimulStreaming | 0.10x | 100.0% | 0.120 s |
| mlx-whisper | LocalAgreement | 0.31x | 1737.1% | 0.060 s |
| mlx-whisper | SimulStreaming | 0.08x | 94.3% | 0.120 s |
| voxtral-mlx | voxtral | 0.18x | 37.1% | 3.422 s |
| voxtral (HF) | voxtral | 0.63x | 28.6% | 4.040 s |

Note: The whisper-based backends were run with `--lan en`, so they attempted to transcribe French
audio in English. This is expected to produce high WER. For a fair comparison, the whisper backends
should be run with `--lan fr` or `--lan auto`. The Voxtral backends auto-detect language.

### English - Multi-speaker (30.0 s, 3 speakers)

| Backend | Policy | RTF | WER | Timestamp MAE |
|---------|--------|-----|-----|---------------|
| faster-whisper | LocalAgreement | 0.24x | 44.7% | 0.235 s |
| faster-whisper | SimulStreaming | 0.10x | 5.3% | 0.398 s |
| mlx-whisper | LocalAgreement | 0.06x | 23.7% | 0.237 s |
| mlx-whisper | SimulStreaming | 0.11x | 5.3% | 0.395 s |
| voxtral-mlx | voxtral | 0.31x | 9.2% | 0.176 s |
| voxtral (HF) | voxtral | 1.00x | 32.9% | 1.034 s |

---

## Key Findings

### Speed (RTF = processing time / audio duration, lower is better)

1. **mlx-whisper + LocalAgreement** is the fastest combo on Apple Silicon, reaching 0.05-0.06x RTF
   on English audio. 30 seconds of audio processed in under 2 seconds.
2. For **faster-whisper**, SimulStreaming is consistently faster than LocalAgreement.
   For **mlx-whisper**, it is the opposite: LocalAgreement (0.05-0.06x) is faster than SimulStreaming (0.11-0.14x).
3. **voxtral-mlx** runs at 0.18-0.32x RTF, roughly 3-5x slower than mlx-whisper but well within
   real-time requirements.
4. **voxtral (HF transformers)** is the slowest at 1.0-1.3x RTF. On longer audio it risks
   falling behind real-time. On Apple Silicon, the MLX variant is strongly preferred.

### Accuracy (WER = Word Error Rate, lower is better)

1. **SimulStreaming** produces significantly better WER than LocalAgreement for whisper backends.
   On the 30s English file: 5.3% vs 23.7-44.7%.
2. **voxtral-mlx** has good accuracy (0% on short English, 9.2% on multi-speaker).
   Whisper also supports `--language auto`, but Voxtral's language detection is more
   reliable and does not bias towards English the way Whisper's auto mode tends to.
3. **LocalAgreement** tends to duplicate the last sentence, inflating WER. This is a known
   artifact of the LCP (Longest Common Prefix) commit strategy at end-of-stream.
4. **Voxtral** backends handle French natively with 28-37% WER, while whisper backends
   were run with `--lan en` here (not a fair comparison for French).

### Timestamp Accuracy (MAE = Mean Absolute Error on word start times, lower is better)

1. **LocalAgreement** produces the most accurate timestamps (0.08s MAE on English), since it
   processes overlapping audio windows and validates via prefix matching.
2. **SimulStreaming** timestamps are slightly less precise (0.24-0.40s MAE) but still usable
   for most applications.
3. **voxtral-mlx** has good timestamp accuracy on English (0.18-0.25s MAE) but drifts on
   audio with long silence gaps (3.4s MAE on the French file with 4-second pauses).
4. **voxtral (HF)** has the worst timestamp accuracy (1.0-4.0s MAE). This is likely related to
   differences in the transformers-based decoding pipeline rather than model quality.

### VAC (Voice Activity Classification) Impact

| Backend | Policy | VAC | 7s English WER | 30s English WER |
|---------|--------|-----|----------------|-----------------|
| faster-whisper | LocalAgreement | on | 21.1% | 44.7% |
| faster-whisper | LocalAgreement | off | 100.0% | 100.0% |
| voxtral-mlx | voxtral | on | 0.0% | 9.2% |
| voxtral-mlx | voxtral | off | 0.0% | 9.2% |

- **Whisper backends require VAC** to function in streaming mode. Without it, the entire audio
  is buffered as a single chunk and the LocalAgreement/SimulStreaming buffer logic breaks down.
- **Voxtral backends are VAC-independent** because they handle their own internal chunking and
  produce identical results with or without VAC. VAC still reduces wasted compute on silence.

---

## Recommendations

| Use Case | Recommended Backend | Policy | Notes |
|----------|-------------------|--------|-------|
| Fastest English transcription (Apple Silicon) | mlx-whisper | SimulStreaming | 0.08-0.14x RTF, 5-10% WER |
| Fastest English transcription (Linux/GPU) | faster-whisper | SimulStreaming | 0.10-0.14x RTF, 0-5% WER |
| Multilingual / auto-detect (Apple Silicon) | voxtral-mlx | voxtral | Handles 100+ languages, 0.18-0.32x RTF |
| Multilingual / auto-detect (Linux/GPU) | voxtral (HF) | voxtral | Same model, slower on CPU, needs GPU |
| Best timestamp accuracy | faster-whisper | LocalAgreement | 0.08s MAE, good for subtitle alignment |
| Low latency, low memory | mlx-whisper (tiny) | SimulStreaming | Smallest footprint, fastest response |

---

## Reproducing These Benchmarks

```bash
# Install test dependencies
pip install -e ".[test]"

# Single backend test
python test_backend_offline.py --backend faster-whisper --policy simulstreaming --no-realtime

# Multi-backend auto-detect benchmark
python test_backend_offline.py --benchmark --no-realtime

# Export to JSON for programmatic analysis
python test_backend_offline.py --benchmark --no-realtime --json results.json

# Test with custom audio
python test_backend_offline.py --backend voxtral-mlx --audio your_file.wav --no-realtime
```

The benchmark harness computes WER and timestamp accuracy automatically when ground truth
`.transcript.json` files exist alongside the audio files. See `audio_tests/` for the format.
