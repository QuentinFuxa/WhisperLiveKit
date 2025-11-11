from pathlib import Path
import builtins
import importlib
import sys


def test_whisper_livekit_absent_from_requirements():
    contents = Path("requirements.txt").read_text(encoding="utf-8")
    assert "whisper-livekit" not in contents


def test_livekit_runner_import_graceful_without_livekit(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "whisper_livekit":
            raise ModuleNotFoundError("simulated missing whisper_livekit")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("src.stt_core.livekit_runner", None)
    module = importlib.import_module("src.stt_core.livekit_runner")
    assert getattr(module, "LiveKit", None) is None
