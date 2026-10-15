"""Paste text into a window on Linux X11 with the clipboard and xdotool."""

import subprocess
import threading
import time

import pyperclip

TERMINALS = [
    "gnome-terminal", "xterm", "konsole", "alacritty", "kitty",
    "terminator", "tilix", "xfce4-terminal", "urxvt", "st",
    "sakura", "guake", "tilda", "hyper", "wezterm",
]
RESTORE_DELAY = 1.0  # Seconds before the old clipboard is put back
COMMAND_TIMEOUT = 2.0  # xdotool can block forever if the window manager refuses focus


def active_window() -> bytes | None:
    """Return the focused window's id, or None."""
    try:
        return subprocess.check_output(
            ["xdotool", "getactivewindow"], stderr=subprocess.DEVNULL, timeout=COMMAND_TIMEOUT,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def is_terminal(window: bytes | None) -> bool:
    """True if the window's WM_CLASS names a known terminal."""
    if not window:
        return False
    try:
        wm_class = subprocess.check_output(
            ["xprop", "-id", window, "WM_CLASS"], stderr=subprocess.DEVNULL, timeout=COMMAND_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    wm_class = wm_class.decode().lower()
    return any(name in wm_class for name in TERMINALS)


def paste(text: str, window: bytes | None, restore: bool = True):
    """Focus the window and paste the text, then restore the old clipboard."""
    try:
        old_clipboard = pyperclip.paste()
    except pyperclip.PyperclipException:
        old_clipboard = None

    pyperclip.copy(text)
    time.sleep(0.05)
    if window:
        try:
            subprocess.run(
                ["xdotool", "windowactivate", "--sync", window], stderr=subprocess.DEVNULL, timeout=COMMAND_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            pass  # Paste into whatever has focus
        time.sleep(0.05)

    key = "ctrl+shift+v" if is_terminal(window) else "ctrl+v"
    subprocess.run(
        ["xdotool", "key", "--clearmodifiers", "--delay", "50", key], stderr=subprocess.DEVNULL, timeout=COMMAND_TIMEOUT,
    )

    if restore and old_clipboard:
        threading.Thread(target=_restore_clipboard, args=(old_clipboard, len(text)), daemon=True).start()


def _restore_clipboard(old_clipboard: str, text_length: int):
    # Longer text can take the target app longer to read from the clipboard
    time.sleep(min(3.0, RESTORE_DELAY + text_length * 0.0001))
    try:
        pyperclip.copy(old_clipboard)
    except pyperclip.PyperclipException:
        pass
