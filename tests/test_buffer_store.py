import json

from src.stt_core.buffer_store import BufferStore


def test_buffer_store_append_and_truncate(tmp_path) -> None:
    path = tmp_path / "buffers" / "transcripts.jsonl"
    store = BufferStore(str(path), max_mb=1)
    store.max_bytes = 120  # shrink for predictable truncation in tests

    for idx in range(5):
        store.append({"text": f"line-{idx}", "blob": "x" * 80})

    assert path.exists()

    with open(path, "r", encoding="utf-8") as fh:
        entries = [json.loads(line) for line in fh]

    texts = [entry["text"] for entry in entries]
    assert "line-0" not in texts  # oldest entries trimmed
    assert texts[-1] == "line-4"
    assert all("ts" in entry for entry in entries)
