"""Tests for pasting dictated text with xdotool."""

import pytest

from whisperlivekit.dictate import paste as paste_module


@pytest.fixture
def recorded(monkeypatch):
    """Replace clipboard and subprocess calls; return the list of commands run."""
    commands = []
    clipboard = {"value": "old"}

    def fake_run(cmd, **kwargs):
        commands.append(cmd)

    monkeypatch.setattr(paste_module.subprocess, "run", fake_run)
    monkeypatch.setattr(paste_module.pyperclip, "copy", lambda text: clipboard.update(value=text))
    monkeypatch.setattr(paste_module.pyperclip, "paste", lambda: clipboard["value"])
    monkeypatch.setattr(paste_module, "RESTORE_DELAY", 0)
    return commands, clipboard


def test_paste_uses_ctrl_v(recorded, monkeypatch):
    commands, clipboard = recorded
    monkeypatch.setattr(paste_module, "is_terminal", lambda window: False)

    paste_module.paste(text="hello", window=b"42", restore=False)

    assert clipboard["value"] == "hello"
    assert ["xdotool", "windowactivate", "--sync", b"42"] in commands
    assert commands[-1][-1] == "ctrl+v"


def test_paste_uses_ctrl_shift_v_in_terminal(recorded, monkeypatch):
    commands, _ = recorded
    monkeypatch.setattr(paste_module, "is_terminal", lambda window: True)

    paste_module.paste(text="ls", window=b"42", restore=False)

    assert commands[-1][-1] == "ctrl+shift+v"


def test_is_terminal_matches_wm_class(monkeypatch):
    monkeypatch.setattr(
        paste_module.subprocess, "check_output",
        lambda cmd, **kwargs: b'WM_CLASS(STRING) = "gnome-terminal-server", "Gnome-terminal"',
    )

    assert paste_module.is_terminal(b"42")


def test_paste_still_sends_key_when_focus_hangs(recorded, monkeypatch):
    commands, _ = recorded
    monkeypatch.setattr(paste_module, "is_terminal", lambda window: False)

    def run(cmd, **kwargs):
        assert kwargs.get("timeout"), cmd
        if "windowactivate" in cmd:
            raise paste_module.subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])
        commands.append(cmd)

    monkeypatch.setattr(paste_module.subprocess, "run", run)

    paste_module.paste(text="hello", window=b"42", restore=False)

    assert commands[-1][-1] == "ctrl+v"
