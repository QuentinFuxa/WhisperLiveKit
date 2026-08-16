"""Coalescing must not drop audio at a boundary.

These cases exercise LocalAgreement. Its finish() runs no inference, and
HypothesisBuffer commits a token only after two agreeing passes, so deferred
audio reaching a boundary without a prior pass is lost silently. Speaker changes
also reset the hypothesis buffer, so their second pass must be returned first.

Audio must be fed at speed=1.0: at speed=0 the whole file arrives as a single
chunk, nothing is ever deferred, and these tests would pass vacuously. The cut
point matters too; it has to leave a sub-threshold remainder outstanding.
"""

import logging

import pytest

from whisperlivekit.timed_objects import ChangeSpeaker

logger = logging.getLogger(__name__)

COALESCE_S = 0.75
# Chosen so the feed stops mid-speech leaving ~0.2s deferred; see module docstring.
CUT_S = 9.7

BASE = {"model_size": "tiny", "lan": "en", "backend_policy": "localagreement"}


@pytest.fixture(scope="session")
def sample():
    from whisperlivekit.test_data import get_samples
    return {s.name: s for s in get_samples()}["librispeech_1"]


def test_sample_decodes_without_ffmpeg(sample, monkeypatch):
    """CI runners have no ffmpeg; the standard samples must not require it."""
    from whisperlivekit import test_harness

    def no_ffmpeg(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr(test_harness.subprocess, "run", no_ffmpeg)
    pcm = test_harness.load_audio_pcm(sample.path)
    assert len(pcm) > 16000 * 2, "expected more than one second of s16le audio"


async def _transcribe(sample, coalesce_s, silence_s=None):
    from whisperlivekit.test_harness import TestHarness

    async with TestHarness(**BASE, asr_coalesce_min_s=coalesce_s) as h:
        player = h.load_audio(sample)
        await player.play_until(CUT_S, speed=1.0, chunk_duration=0.5)
        if silence_s:
            await h.pause(silence_s, speed=0)
            await h.drain(3.0)
        result = await h.finish(timeout=90)
        return (
            result.text.split(),
            h.metrics.n_transcription_calls,
            len(h.metrics.transcription_durations),
        )


async def _transcribe_across_speaker_boundary(sample, coalesce_s):
    from whisperlivekit.test_harness import TestHarness

    async with TestHarness(**BASE, asr_coalesce_min_s=coalesce_s) as h:
        player = h.load_audio(sample)
        await player.play_until(CUT_S, speed=1.0, chunk_duration=0.5)
        await h._processor.transcription_queue.put(
            ChangeSpeaker(speaker=2, start=CUT_S)
        )
        await h.drain(3.0)
        result = await h.finish(timeout=90)
        return (
            result.text.split(),
            h.metrics.n_transcription_calls,
            len(h.metrics.transcription_durations),
        )


@pytest.mark.asyncio
async def test_end_of_stream_transcribes_deferred_audio(sample):
    """Audio still deferred when the stream ends must be inferred by finish()."""
    baseline, baseline_calls, baseline_durations = await _transcribe(
        sample,
        coalesce_s=0.0,
    )
    coalesced, coalesced_calls, coalesced_durations = await _transcribe(
        sample,
        coalesce_s=COALESCE_S,
    )

    assert baseline, "baseline produced no text"
    logger.info("baseline=%s", " ".join(baseline[-6:]))
    logger.info("coalesced=%s", " ".join(coalesced[-6:]))
    logger.info(
        "EOF ASR calls: baseline=%d coalesced=%d",
        baseline_calls,
        coalesced_calls,
    )

    assert coalesced[-3:] == baseline[-3:], (
        f"tail diverged: baseline ended {baseline[-3:]}, coalesced ended {coalesced[-3:]}"
    )
    assert baseline_durations == baseline_calls
    assert coalesced_durations == coalesced_calls
    assert coalesced_calls < baseline_calls, "test did not exercise coalescing"


@pytest.mark.asyncio
async def test_deferred_audio_survives_a_long_silence(sample):
    """A >5s silence calls init(), discarding anything not yet committed."""
    baseline, baseline_calls, baseline_durations = await _transcribe(
        sample,
        coalesce_s=0.0,
        silence_s=7.0,
    )
    coalesced, coalesced_calls, coalesced_durations = await _transcribe(
        sample,
        coalesce_s=COALESCE_S,
        silence_s=7.0,
    )

    assert baseline, "baseline produced no text"
    logger.info("baseline=%s", " ".join(baseline[-6:]))
    logger.info("coalesced=%s", " ".join(coalesced[-6:]))
    logger.info(
        "silence ASR calls: baseline=%d coalesced=%d",
        baseline_calls,
        coalesced_calls,
    )

    assert coalesced[-3:] == baseline[-3:], (
        f"tail diverged: baseline ended {baseline[-3:]}, coalesced ended {coalesced[-3:]}"
    )
    assert baseline_durations == baseline_calls
    assert coalesced_durations == coalesced_calls
    assert coalesced_calls < baseline_calls, "test did not exercise coalescing"


@pytest.mark.asyncio
async def test_deferred_audio_survives_a_speaker_change(sample):
    """The second boundary pass must be emitted before new_speaker() resets."""
    baseline, baseline_calls, baseline_durations = (
        await _transcribe_across_speaker_boundary(
            sample,
            coalesce_s=0.0,
        )
    )
    coalesced, coalesced_calls, coalesced_durations = (
        await _transcribe_across_speaker_boundary(
            sample,
            coalesce_s=COALESCE_S,
        )
    )

    logger.info(
        "speaker ASR calls: baseline=%d coalesced=%d",
        baseline_calls,
        coalesced_calls,
    )

    assert baseline, "baseline produced no text"
    assert coalesced[-3:] == baseline[-3:], (
        f"tail diverged: baseline ended {baseline[-3:]}, coalesced ended {coalesced[-3:]}"
    )
    assert baseline_durations == baseline_calls
    assert coalesced_durations == coalesced_calls
    # Both totals include the speaker-boundary pass and terminal finish pass.
    # The strict difference therefore still measures arrival-pass coalescing.
    assert coalesced_calls < baseline_calls, "test did not exercise coalescing"
