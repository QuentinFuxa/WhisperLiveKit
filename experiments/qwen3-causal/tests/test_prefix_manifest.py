from qwen3_streaming.metrics import stable_prefix_stats, word_error_rate
from qwen3_streaming.prefix_manifest import prefix_points, stable_prefix_text
from qwen3_streaming.realtime import is_silent_pcm16, post_process_realtime_text


def test_prefix_points_includes_full_duration():
    assert prefix_points(3.2, min_prefix_sec=1.0, stride_sec=1.0) == [
        1.0,
        2.0,
        3.0,
        3.2,
    ]


def test_stable_prefix_text_holds_back_right_context():
    text = "one two three four five"
    got = stable_prefix_text(
        text,
        prefix_end_sec=6.0,
        duration_sec=10.0,
        right_context_sec=2.0,
    )
    assert got == "one two"


def test_stable_prefix_text_returns_full_at_full_duration_without_holdback_override():
    text = "one two three four five"
    got = stable_prefix_text(
        text,
        prefix_end_sec=10.0,
        duration_sec=10.0,
        right_context_sec=0.0,
    )
    assert got == text


def test_word_error_rate_and_stable_prefix_stats():
    assert word_error_rate("hello world", "hello there") == 0.5
    stats = stable_prefix_stats("hello world again", "hello world nope")
    assert stats.common_prefix_words == 2
    assert stats.revision_words == 1


def test_realtime_post_process_strips_qwen_markers():
    raw = (
        "language English<asr_text>Hello world.\n"
        "language English\n"
        "language English<asr_text>Second segment."
    )

    assert post_process_realtime_text(raw) == "Hello world. Second segment."


def test_silent_pcm16_detection():
    assert is_silent_pcm16(b"\x00\x00\x00\x00")
    assert not is_silent_pcm16(b"\x00\x00\x01\x00")
    assert not is_silent_pcm16(b"")
