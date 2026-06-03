import pytest

from scripts.slice_audio_manifest import chunk_spans


def test_chunk_spans_respects_limit_and_min_duration():
    spans = chunk_spans(
        52.0,
        chunk_sec=20.0,
        stride_sec=20.0,
        min_chunk_sec=5.0,
        max_chunks=2,
    )

    assert spans == [(0.0, 20.0), (20.0, 40.0)]


def test_chunk_spans_keeps_tail_when_long_enough():
    spans = chunk_spans(
        44.0,
        chunk_sec=20.0,
        stride_sec=20.0,
        min_chunk_sec=3.0,
    )

    assert spans == [(0.0, 20.0), (20.0, 40.0), (40.0, 44.0)]


def test_chunk_spans_rejects_invalid_values():
    with pytest.raises(ValueError):
        chunk_spans(10.0, chunk_sec=0.0, stride_sec=1.0, min_chunk_sec=1.0)
