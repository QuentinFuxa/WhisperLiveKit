# CLAUDE.md -- WhisperLiveKit

## Build & Test

Install for development:

```sh
pip install -e ".[test]"
```

Test with real audio using `TestHarness` (requires models + audio files):

```python
import asyncio
from whisperlivekit import TestHarness

async def main():
    async with TestHarness(model_size="base", lan="en", diarization=True) as h:
        await h.feed("audio.wav", speed=1.0)     # feed at real-time
        await h.drain(2.0)                         # let ASR catch up
        h.print_state()                            # see current output

        await h.silence(7.0, speed=1.0)            # 7s silence
        await h.wait_for_silence()                 # verify detection

        result = await h.finish()
        print(f"WER: {result.wer('expected text'):.2%}")
        print(f"Speakers: {result.speakers}")
        print(f"Text at 3s: {result.text_at(3.0)}")

asyncio.run(main())
```

## Architecture

WhisperLiveKit is a real-time speech transcription system using WebSockets.

- **TranscriptionEngine** (singleton) loads models once at startup and is shared across all sessions.
- **AudioProcessor** is created per WebSocket session. It runs an async producer-consumer pipeline: FFmpeg decodes audio, Silero VAD detects speech, the ASR backend transcribes, and results stream back to the client.
- Two streaming policies:
  - **LocalAgreement** (HypothesisBuffer) -- confirms tokens only when consecutive inferences agree.
  - **SimulStreaming** (AlignAtt attention-based) -- emits tokens as soon as alignment attention is confident.
- 6 ASR backends: WhisperASR, FasterWhisperASR, MLXWhisper, VoxtralMLX, VoxtralHF, Qwen3.
- **SessionASRProxy** wraps the shared ASR with a per-session language override, using a lock to safely swap `original_language` during `transcribe()`.
- **DiffTracker** implements a snapshot-then-diff protocol for bandwidth-efficient incremental WebSocket updates (opt-in via `?mode=diff`).

## Key Files

| File | Purpose |
|---|---|
| `config.py` | `WhisperLiveKitConfig` dataclass -- single source of truth for configuration |
| `core.py` | `TranscriptionEngine` singleton, `online_factory()`, diarization/translation factories |
| `audio_processor.py` | Per-session async pipeline (FFmpeg -> VAD -> ASR -> output) |
| `basic_server.py` | FastAPI server: WebSocket `/asr`, REST `/v1/audio/transcriptions`, CLI `wlk` |
| `timed_objects.py` | `ASRToken`, `Segment`, `FrontData` data structures |
| `diff_protocol.py` | `DiffTracker` -- snapshot-then-diff WebSocket protocol |
| `session_asr_proxy.py` | `SessionASRProxy` -- thread-safe per-session language wrapper |
| `parse_args.py` | CLI argument parser, returns `WhisperLiveKitConfig` |
| `test_client.py` | Headless WebSocket test client (`wlk-test`) |
| `test_harness.py` | In-process testing harness (`TestHarness`) for real E2E testing |
| `local_agreement/online_asr.py` | `OnlineASRProcessor` for LocalAgreement policy |
| `simul_whisper/` | SimulStreaming policy implementation (AlignAtt) |

## Key Patterns

- **TranscriptionEngine** uses double-checked locking for thread-safe singleton initialization. Never create a second instance in production. Use `TranscriptionEngine.reset()` in tests only to switch backends.
- **WhisperLiveKitConfig** dataclass is the single source of truth. Use `from_namespace()` (from argparse) or `from_kwargs()` (programmatic). `parse_args()` returns a `WhisperLiveKitConfig`, not a raw Namespace.
- **online_factory()** in `core.py` routes to the correct online processor class based on backend and policy.
- **FrontData.to_dict()** is the canonical output format for WebSocket messages.
- **SessionASRProxy** uses `__getattr__` delegation -- it forwards everything except `transcribe()` to the wrapped ASR.
- The server exposes `self.args` as a `Namespace` on `TranscriptionEngine` for backward compatibility with `AudioProcessor`.

## Adding a New ASR Backend

1. Create `whisperlivekit/my_backend.py` with a class implementing:
   - `transcribe(audio, init_prompt="")` -- run inference on audio array
   - `ts_words(result)` -- extract timestamped words from result
   - `segments_end_ts(result)` -- extract segment end timestamps
   - `use_vad()` -- whether this backend needs external VAD
2. Set required attributes on the class: `sep`, `original_language`, `backend_choice`, `SAMPLING_RATE`, `confidence_validation`, `tokenizer`, `buffer_trimming`, `buffer_trimming_sec`.
3. Register in `core.py`:
   - Add an `elif` branch in `TranscriptionEngine._do_init()` to instantiate the backend.
   - Add a routing case in `online_factory()` to return the appropriate online processor.
4. Add the backend choice to CLI args in `parse_args.py`.

## Testing with TestHarness

`TestHarness` wraps AudioProcessor in-process for full pipeline testing without a server.

Key methods:
- `feed(path, speed=1.0)` -- feed audio at controlled speed (0 = instant)
- `silence(duration, speed=1.0)` -- inject silence (>5s triggers silence detection)
- `drain(seconds)` -- wait for ASR to catch up without feeding audio
- `finish(timeout)` -- signal end-of-audio, wait for pipeline to drain
- `state` -- current `TestState` with lines, buffers, speakers, timestamps
- `wait_for(predicate)` / `wait_for_text()` / `wait_for_silence()` / `wait_for_speakers(n)`
- `snapshot_at(audio_time)` -- historical state at a given audio position
- `on_update(callback)` -- register callback for each state update

`TestState` provides:
- `text`, `committed_text` -- full or committed-only transcription
- `speakers`, `n_speakers`, `has_silence` -- speaker/silence info
- `line_at(time_s)`, `speaker_at(time_s)`, `text_at(time_s)` -- query by timestamp
- `lines_between(start, end)`, `text_between(start, end)` -- query by time range
- `wer(reference)`, `wer_detailed(reference)` -- evaluation against ground truth
- `speech_lines`, `silence_segments` -- filtered line lists

## OpenAI-Compatible REST API

The server exposes an OpenAI-compatible batch transcription endpoint:

```bash
# Transcribe a file (drop-in replacement for OpenAI)
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F response_format=verbose_json

# Works with the OpenAI Python client
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
result = client.audio.transcriptions.create(model="whisper-1", file=open("audio.mp3", "rb"))
print(result.text)
```

Supported `response_format` values: `json`, `verbose_json`, `text`, `srt`, `vtt`.
The `model` parameter is accepted but ignored (uses the server's configured backend).

## Do NOT

- Do not create a second `TranscriptionEngine` instance. It is a singleton; the constructor returns the existing instance after the first call.
- Do not modify `original_language` on the shared ASR directly. Use `SessionASRProxy` for per-session language overrides.
- Do not assume the frontend handles diff protocol messages. Diff mode is opt-in (`?mode=diff`) and ignored by default.
- Do not write mock-based unit tests. Use `TestHarness` with real audio for pipeline testing.
