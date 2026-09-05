# Working on WhisperLiveKit

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for setup, validation, and test scope.

## Runtime boundaries

- `TranscriptionEngine` loads models during startup and is shared within one process. `reset()` is for tests and backend comparisons.
- `AudioProcessor` is per session. It owns audio queues, buffers, and processing tasks. Always call `cleanup()` when leaving a session.
- `WhisperLiveKitConfig` describes configuration. `parse_args()` returns this dataclass; `engine.args` is a compatibility namespace.
- `online_factory()` selects the session processor. Use `SessionASRProxy` or the backend's session wrapper for language/context overrides; do not mutate shared ASR state directly.
- `FrontData.to_dict()` defines native WebSocket JSON. The browser uses full snapshots; diff mode is opt-in for custom clients.
- `translation_alignatt.py` connects to the separate AlignAtt4LLM translation server. Protocol checks do not establish model quality.

## Navigation

| Area | Entry points |
|---|---|
| Server and protocol adapters | `basic_server.py`, `deepgram_compat.py`, `docs/API.md` |
| Streaming pipeline | `audio_processor.py`, `tokens_alignment.py`, `timed_objects.py` |
| Backend construction | `core.py`, `simul_whisper/`, `local_agreement/` |
| CLI and configuration | `cli.py`, `parse_args.py`, `config.py` |
| Real-audio testing | `test_harness.py`, `tests/test_pipeline.py`, `tests/test_asr_coalescing_pipeline.py` |
| Backend setup and translation | `docs/backends.md`, `docs/translation-alignatt.md` |
| Benchmark interpretation | `benchmarks/README.md` |

Paths above are relative to `whisperlivekit/` unless they begin with `docs/`, `tests/`, or `benchmarks/`.

Extend meaningful regression scenarios before adding new test scaffolding. For pipeline behavior, also run real audio on the affected backend. Keep historical benchmark figures in the archive; do not present them as measurements of current code.
