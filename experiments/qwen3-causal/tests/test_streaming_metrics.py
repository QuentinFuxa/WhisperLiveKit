from qwen3_streaming.metrics import streaming_text_event_stats, text_revision_stats


def test_text_revision_stats_counts_replaced_tail_words():
    stats = text_revision_stats(
        [
            "hello brave world",
            "hello bright world",
            "hello bright world today",
        ]
    )

    assert stats["revision_events"] == 1
    assert stats["revision_words"] == 2
    assert stats["max_revision_words"] == 2


def test_text_revision_stats_does_not_count_extensions():
    stats = text_revision_stats(["hello", "hello world", "hello world today"])

    assert stats["revision_events"] == 0
    assert stats["revision_words"] == 0


def test_streaming_text_event_stats_reports_latency_coverage_and_revisions():
    events = [
        {
            "audio_sec": 1.0,
            "display": "",
            "committed": "",
            "hypothesis": "",
        },
        {
            "audio_sec": 2.0,
            "display": "hello brave",
            "committed": "",
            "hypothesis": "hello brave",
        },
        {
            "audio_sec": 3.0,
            "display": "hello bright world",
            "committed": "hello",
            "hypothesis": "hello bright world",
        },
    ]

    stats = streaming_text_event_stats(
        events,
        final_text="hello bright world today",
        stable_text="hello",
    )

    assert stats["first_display_sec"] == 2.0
    assert stats["first_commit_sec"] == 3.0
    assert stats["stable_word_count"] == 1
    assert stats["final_word_count"] == 4
    assert stats["stable_coverage_ratio"] == 0.25
    assert stats["display_revision_events"] == 1
    assert stats["display_revision_words"] == 1
    assert stats["committed_revision_events"] == 0
