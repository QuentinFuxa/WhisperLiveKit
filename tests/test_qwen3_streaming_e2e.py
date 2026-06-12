"""Opt-in E2E smoke for the qwen3-streaming backend.

Downloads Qwen/Qwen3-ASR-0.6B (~2.5 GB) and runs real inference, so it is
gated behind an environment variable:

    WLK_RUN_QWEN3_STREAMING_E2E=1 pytest tests/test_qwen3_streaming_e2e.py -v

Runs on CUDA, MPS (Apple Silicon) or CPU (slow).
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("WLK_RUN_QWEN3_STREAMING_E2E") != "1",
    reason="set WLK_RUN_QWEN3_STREAMING_E2E=1 to run the qwen3-streaming E2E smoke",
)


def _parse_time(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


@pytest.mark.asyncio
async def test_qwen3_streaming_transcribes_librispeech():
    from whisperlivekit import TestHarness
    from whisperlivekit.test_data import get_sample

    sample = get_sample("librispeech_short")

    async with TestHarness(
        backend="qwen3-streaming",
        model_size="Qwen/Qwen3-ASR-0.6B",
        lan="en",
        vac=True,
    ) as harness:
        await harness.feed(sample.path, speed=0)
        result = await harness.finish(timeout=180.0)

    wer = result.wer(sample.reference)
    assert wer < 0.35, f"WER {wer:.2%} too high; got: {result.text!r}"

    # Every committed line must carry usable, monotonic timestamps.
    lines = result.speech_lines
    assert lines, "no transcript lines produced"
    previous_end = 0.0
    for line in lines:
        start = _parse_time(line.get("start", 0))
        end = _parse_time(line.get("end", 0))
        assert start >= 0.0
        assert start <= end
        assert start >= previous_end - 1.0  # allow mild overlap at edges
        previous_end = end


@pytest.mark.asyncio
async def test_qwen3_streaming_causal_transcribes_librispeech():
    """Causal backend E2E: additionally requires QWEN3_TOWER_CKPT pointing at
    the fine-tuned tower (local file/dir or HF repo id)."""
    tower = os.environ.get("QWEN3_TOWER_CKPT")
    if not tower:
        pytest.skip("set QWEN3_TOWER_CKPT to run the causal E2E smoke")

    from whisperlivekit import TestHarness
    from whisperlivekit.test_data import get_sample

    sample = get_sample("librispeech_short")

    async with TestHarness(
        backend="qwen3-streaming",
        model_size="Qwen/Qwen3-ASR-0.6B",
        lan="en",
        vac=True,
        qwen3_streaming_audio_backend="causal",
        qwen3_streaming_tower_checkpoint=tower,
    ) as harness:
        await harness.feed(sample.path, speed=0)
        result = await harness.finish(timeout=180.0)

    wer = result.wer(sample.reference)
    assert wer < 0.35, f"WER {wer:.2%} too high; got: {result.text!r}"

    lines = result.speech_lines
    assert lines, "no transcript lines produced"
    previous_end = 0.0
    for line in lines:
        start = _parse_time(line.get("start", 0))
        end = _parse_time(line.get("end", 0))
        assert start >= 0.0
        assert start <= end
        assert start >= previous_end - 1.0
        previous_end = end
