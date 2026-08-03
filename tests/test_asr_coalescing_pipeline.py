"""Coalescing must not drop audio at a boundary.

Both cases are specific to LocalAgreement. Its finish() runs no inference, and
HypothesisBuffer commits a token only after two agreeing passes, so deferred
audio reaching a boundary without a prior pass is lost silently.

Audio must be fed at speed=1.0: at speed=0 the whole file arrives as a single
chunk, nothing is ever deferred, and these tests would pass vacuously. The cut
point matters too — it has to leave a sub-threshold remainder outstanding.
"""

import logging

import pytest

logger = logging.getLogger(__name__)

COALESCE_S = 0.75
# Chosen so the feed stops mid-speech leaving ~0.2s deferred; see module docstring.
CUT_S = 9.7

BASE = {"model_size": "tiny", "lan": "en", "backend_policy": "localagreement"}


@pytest.fixture(scope="session")
def sample():
    from whisperlivekit.test_data import get_samples
    return {s.name: s for s in get_samples()}["librispeech_1"]


async def _transcribe(sample, coalesce_s, silence_s=None):
    from whisperlivekit.test_harness import TestHarness

    async with TestHarness(**BASE, asr_coalesce_min_s=coalesce_s) as h:
        player = h.load_audio(sample)
        await player.play_until(CUT_S, speed=1.0, chunk_duration=0.5)
        if silence_s:
            await h.pause(silence_s, speed=0)
            await h.drain(3.0)
        result = await h.finish(timeout=90)
        return result.text.split()


@pytest.mark.asyncio
async def test_end_of_stream_transcribes_deferred_audio(sample):
    """Audio still deferred when the stream ends must be inferred by finish()."""
    baseline = await _transcribe(sample, coalesce_s=0.0)
    coalesced = await _transcribe(sample, coalesce_s=COALESCE_S)

    assert baseline, "baseline produced no text"
    logger.info("baseline=%s", " ".join(baseline[-6:]))
    logger.info("coalesced=%s", " ".join(coalesced[-6:]))

    assert coalesced[-3:] == baseline[-3:], (
        f"tail diverged: baseline ended {baseline[-3:]}, coalesced ended {coalesced[-3:]}"
    )


@pytest.mark.asyncio
async def test_deferred_audio_survives_a_long_silence(sample):
    """A >5s silence calls init(), discarding anything not yet committed."""
    baseline = await _transcribe(sample, coalesce_s=0.0, silence_s=7.0)
    coalesced = await _transcribe(sample, coalesce_s=COALESCE_S, silence_s=7.0)

    assert baseline, "baseline produced no text"
    logger.info("baseline=%s", " ".join(baseline[-6:]))
    logger.info("coalesced=%s", " ".join(coalesced[-6:]))

    assert coalesced[-3:] == baseline[-3:], (
        f"tail diverged: baseline ended {baseline[-3:]}, coalesced ended {coalesced[-3:]}"
    )
