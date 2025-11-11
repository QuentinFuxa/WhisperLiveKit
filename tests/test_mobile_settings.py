from pathlib import Path

from mobile.daymind.store.settings_store import SettingsStore


def test_settings_store_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    assert store.get().server_url == ""

    store.update(server_url="https://example.com", api_key="abc")
    data = path.read_text()
    assert "example.com" in data

    store2 = SettingsStore(path)
    assert store2.get().api_key == "abc"
