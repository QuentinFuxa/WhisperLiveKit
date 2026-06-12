"""Make the tests directory importable so test modules can share fakes
(qwen3_streaming_fakes) without packaging them."""

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
