# Contributing

Thanks for helping improve WhisperLiveKit. Bug fixes, documentation, tests, and focused features are welcome.

Participation in the project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before opening an issue

Search the existing issues and discussions first. Use an issue for reproducible bugs and concrete feature requests. Use a discussion for setup questions, usage help, and broader ideas.

A useful bug report includes:

- the exact command or Python configuration;
- the backend, model, operating system, Python version, and accelerator;
- the expected and observed behavior;
- the complete traceback or relevant logs;
- a minimal audio sample or reproduction when licensing permits it.

Security reports should follow [SECURITY.md](SECURITY.md) and must not be filed as public issues.

## Development setup

Clone the repository with its submodule, then install the development and test dependencies:

```bash
git clone --recurse-submodules https://github.com/QuentinFuxa/WhisperLiveKit.git
cd WhisperLiveKit
uv sync --extra test
```

If the repository was cloned without submodules, initialize them before running `uv sync`:

```bash
git submodule update --init --recursive
```

WhisperLiveKit supports Python 3.11 through 3.13. Python 3.12 is the primary local and CI development version.

## Validation

Run the dependency-light suite and static checks before submitting a pull request:

```bash
uv run ruff check .
uv lock --check
uv run pytest -q tests/ --ignore=tests/test_pipeline.py
```

Changes to streaming, buffering, timestamps, model loading, or silence handling should also run the real-model pipeline tests that cover the affected backend:

```bash
uv run pytest -v tests/test_pipeline.py -k whisper
```

These tests download models and audio, so record the backend, hardware, selected tests, and results in the pull request. Add a focused regression test for every bug fix.

## Pull requests

Keep each pull request focused on one problem. For a large behavior change, open or join an issue first so the approach can be agreed before substantial implementation work.

Pull requests should:

- explain the user-visible impact and any compatibility risk;
- update documentation and examples when behavior changes;
- include the commands and results used for validation;
- keep generated files and unrelated formatting changes out of the diff;
- preserve the existing public API unless the change is explicitly discussed;
- use clear commit messages without generated attribution trailers.

Maintainers may ask for a smaller diff, additional tests, benchmark evidence, or a rebase onto `main` before merging.
