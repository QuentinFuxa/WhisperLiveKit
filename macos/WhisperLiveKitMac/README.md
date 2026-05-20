# WhisperLiveKit macOS

Native SwiftUI frontend for the existing WhisperLiveKit WebSocket backend.

## Requirements

- macOS 14 or newer.
- Xcode Command Line Tools with Swift 5.9 or newer.
- Xcode 26 / macOS 26 SDK to compile the Liquid Glass modifiers. Older runtimes fall back to standard macOS materials at runtime.

## Run

Start the backend in PCM mode:

```bash
wlk --model base --language auto --pcm-input
```

Then launch the app:

```bash
cd macos/WhisperLiveKitMac
swift run WhisperLiveKitMac
```

The app connects to `ws://localhost:8000/asr` by default and streams microphone audio as PCM s16le 16 kHz mono, matching the backend's existing `--pcm-input` WebSocket protocol.

The sidebar contains backend launch settings and a copyable local command. Settings are grouped by model family: Whisper exposes Whisper backends, model sizes, and SimulStreaming/LocalAgreement; Qwen3-ASR exposes vLLM backends, Qwen sizes, and its built-in early-cut streaming path; Voxtral exposes its native streaming backends with no size picker. `language` is also sent as a per-session WebSocket query parameter when it is not set to `Server default`. Use the folder button next to Record to pick an audio file and stream it at real-time speed through the same WebSocket path.
