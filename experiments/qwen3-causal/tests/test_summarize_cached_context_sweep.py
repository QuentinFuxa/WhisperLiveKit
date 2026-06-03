import importlib.util
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_cached_context_sweep.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "summarize_cached_context_sweep",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
summarize = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summarize)


def test_summarize_context_sweep_orders_by_left_context_and_adds_delta():
    rows = summarize.summarize_context_sweep(
        [
            {
                "summary_path": "left15.summary.json",
                "count": 21,
                "ok": 21,
                "qwen_audio_left_context_frames": 1500,
                "qwen_audio_right_context_frames": 64,
                "cache_bound_frames": 1564,
                "max_last_recomputed_frames": 1564,
                "max_recomputed_context_frames": 1500,
                "wer_final_mean": 0.16,
            },
            {
                "summary_path": "left2.summary.json",
                "count": 21,
                "ok": 21,
                "qwen_audio_left_context_frames": 200,
                "qwen_audio_right_context_frames": 64,
                "cache_bound_frames": 264,
                "max_last_recomputed_frames": 264,
                "max_recomputed_context_frames": 200,
                "wer_final_mean": 0.18,
            },
        ],
        mel_hop_ms=10,
    )

    assert [row["left_context_sec"] for row in rows] == [2.0, 15.0]
    assert rows[0]["right_context_ms"] == 640
    assert rows[0]["wer_final_delta_vs_max_left"] == 0.01999999999999999
    assert rows[1]["wer_final_delta_vs_max_left"] == 0.0


def test_markdown_table_contains_core_metrics():
    table = summarize.markdown_table(
        [
            {
                "left_context_sec": 2.0,
                "right_context_ms": 640,
                "max_last_recomputed_frames": 264,
                "max_recomputed_context_frames": 200,
                "wer_final_mean": 0.18,
                "wer_final_delta_vs_max_left": 0.02,
                "first_display_sec_mean": 1.0,
                "first_commit_sec_mean": 16.0,
                "stable_coverage_ratio_mean": 0.5,
            }
        ]
    )

    assert "left_s" in table
    assert "264" in table
    assert "+0.0200" in table
