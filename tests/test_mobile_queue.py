from pathlib import Path

from mobile.daymind.store.queue_store import ChunkQueue


def test_chunk_queue_enqueue_and_retry(tmp_path):
    queue = ChunkQueue(tmp_path / "queue.json")
    chunk_path = tmp_path / "chunk.wav"
    chunk_path.write_bytes(b"data")

    entry_id = queue.enqueue(str(chunk_path))
    assert len(queue) == 1
    entry = queue.peek()
    assert entry["id"] == entry_id

    queue.mark_failed(entry_id, "timeout")
    import time

    time.sleep(2)
    entry2 = queue.peek()
    assert entry2 is not None
    queue.mark_sent(entry_id)
    assert len(queue) == 0
