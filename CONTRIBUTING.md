# Contributing

Bug fixes, documentation, tests, and focused features are welcome. Participation follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Report a problem

Search existing issues and discussions first. For bugs, include the command or Python configuration, backend and model, OS, Python version, hardware, traceback, and expected behavior. A small reproducible audio sample helps when it can be shared. Use discussions for setup questions and broader ideas. Follow [SECURITY.md](SECURITY.md) for security reports.

## Development setup

```bash
git clone --recurse-submodules https://github.com/QuentinFuxa/WhisperLiveKit.git
cd WhisperLiveKit
uv sync --extra test
```

If needed, initialize the submodule with `git submodule update --init --recursive`. Python 3.11–3.13 are supported; Python 3.12 is the primary development version. Install the extra for the backend you are changing. Some tests skip when an optional backend dependency is absent; inspect the skip reasons.

## Validation

```bash
uv run ruff check .
uv lock --check
uv run pytest -q tests/ --ignore=tests/test_pipeline.py --ignore=tests/test_asr_coalescing_pipeline.py
```

CI also runs a real-audio Whisper test covering end-of-stream, silence, and speaker changes:

```bash
uv run pytest -q tests/test_asr_coalescing_pipeline.py
```

This step downloads Whisper tiny and audio and feeds it at real-time speed. Changes to streaming, buffering, timestamps, model loading, or silence handling should run the relevant real-model scenarios:

```bash
uv run pytest -v tests/test_pipeline.py -k whisper
```

The full pipeline matrix may download large models. Record selected tests, backend, hardware, and results. NeMo and FunASR scenarios have additional dependency/model requirements documented in their test files.

## What deserves a test

Prefer a scenario that fails when a user-visible contract breaks: lost words at a pause, a repeated final event, a leaked session language, malformed subtitles, or a translation that never reaches the client. Extend an existing scenario when it already exercises the affected path.

Keep small deterministic tests for meaningful edge cases that are expensive or unreliable to reproduce with a model. Avoid tests that only repeat a constant, assert private call order, or reconstruct the implementation with mocks. Test count and coverage percentage are not goals on their own. Real WebSocket protocol checks and real-audio model checks answer different questions; neither substitutes for the other.

## Pull requests

Keep changes focused and explain the problem, resulting behavior, compatibility effects, and validation. Update examples when the interface changes. Exclude unrelated formatting, generated files, and generated attribution trailers. Preserve the public API unless a compatibility change is intentional and described.

Performance claims need reproducible evidence. See [the benchmark notes](benchmarks/README.md) for the data and measurements needed before publishing a comparison.
