from typing import Dict, Optional


class SpeakerNameManager:
    """Manages mapping between numeric speaker IDs and custom names"""

    def __init__(self):
        self._names: Dict[int, str] = {}

    def set_name(self, speaker_id: int, name: str) -> None:
        """Assign a custom name to a speaker ID."""
        self._names[speaker_id] = name

    def get_name(self, speaker_id: int) -> str:
        """Get the display name for a speaker (customized name or default numeric value)"""
        return self._names.get(speaker_id, str(speaker_id))

    def remove_name(self, speaker_id: int) -> None:
        """Remove customized name, revert back to numeric display"""
        self._names.pop(speaker_id, None)

    def get_all_mappings(self) -> Dict[int, str]:
        """Return all current speaker name maps."""
        return self._names.copy()

    def clear(self) -> None:
        """Clear all customized speaker names."""
        self._names.clear()
