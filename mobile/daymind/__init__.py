"""DayMind Android client package."""

__all__ = ["DayMindApp"]


def __getattr__(name):
    if name == "DayMindApp":
        from .app import DayMindApp

        return DayMindApp
    raise AttributeError(name)
