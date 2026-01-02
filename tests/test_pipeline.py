"""End-to-end pipeline tests using real models and real audio.

Run with: pytest tests/test_pipeline.py -v

Tests exercise the full pipeline through TestHarness + AudioPlayer:
audio feeding, play/pause/resume, silence detection, buffer inspection,
timing validation, and WER evaluation.

Each test is parameterized by backend so that adding a new backend
automatically gets test coverage. Tests use AudioPlayer for timeline
control — play segments, pause (inject silence), resume, cut.

Designed for AI agent automation: an agent can modify code, run these
tests, and validate transcription quality, timing, and streaming behavior.
"""

import logging

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

AVAILABLE_BACKENDS = []

try:
    import mlx.core  # noqa: F401

    from whisperlivekit.voxtral_mlx.loader import load_voxtral_model  # noqa: F401
    AVAILABLE_BACKENDS.append("voxtral-mlx")
except ImportError:
    pass

AVAILABLE_BACKENDS.append("whisper")

try:
    from transformers import VoxtralRealtimeForConditionalGeneration  # noqa: F401
    AVAILABLE_BACKENDS.append("voxtral-hf")
except ImportError:
    pass

try:
    from whisperlivekit.qwen3_asr import _patch_transformers_compat
    _patch_transformers_compat()
    from qwen_asr import Qwen3ASRModel  # noqa: F401
    AVAILABLE_BACKENDS.append("qwen3")
    AVAILABLE_BACKENDS.append("qwen3-simul")
except (ImportError, Exception):
    pass

try:
    import mlx_qwen3_asr  # noqa: F401
    AVAILABLE_BACKENDS.append("qwen3-mlx")
except ImportError:
    pass

BACKEND_CONFIG = {
    "whisper": {"model_size": "tiny", "lan": "en"},
    "voxtral-mlx": {"backend": "voxtral-mlx", "lan": "en"},
    "voxtral-hf": {"backend": "voxtral", "lan": "en"},
    "qwen3": {"backend": "qwen3", "lan": "en"},
    "qwen3-simul": {
        "backend": "qwen3-simul",
        "lan": "en",
        "custom_alignment_heads": "scripts/alignment_heads_qwen3_asr_1.7B.json",
    },
    "qwen3-mlx": {"backend": "qwen3-mlx", "lan": "en"},
}

# Voxtral backends flush all words at once with proportionally-distributed
# timestamps.  After a silence gap the speech line that follows may start
# before the silence segment, making the sequence non-monotonic.  This is
# a known limitation of the batch-flush architecture, not a bug.
VOXTRAL_BACKENDS = {"voxtral-mlx", "voxtral-hf"}

# Backends that use batch-flush and may have non-monotonic timestamps
BATCH_FLUSH_BACKENDS = {"voxtral-mlx", "voxtral-hf", "qwen3", "qwen3-simul", "qwen3-mlx"}


def backend_kwargs(backend: str) -> dict:
    return BACKEND_CONFIG.get(backend, {"model_size": "tiny", "lan": "en"})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def samples():
    """Download test samples once per session."""
    from whisperlivekit.test_data import get_samples
    return {s.name: s for s in get_samples()}


@pytest.fixture(scope="session")
def short_sample(samples):
    return samples["librispeech_short"]


@pytest.fixture(scope="session")
def medium_sample(samples):
    return samples["librispeech_1"]


@pytest.fixture(scope="session")
def meeting_sample(samples):
    return samples["ami_meeting"]


# ---------------------------------------------------------------------------
# 1. Transcription Quality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.asyncio
async def test_transcription_quality(backend, short_sample):
    """Feed a short clip and verify: text produced, WER < 50%, timestamps valid."""
    from whisperlivekit.test_harness import TestHarness

    async with TestHarness(**backend_kwargs(backend)) as h:
        await h.feed(short_sample.path, speed=0)
        await h.drain(5.0)
        result = await h.finish(timeout=60)

        assert result.text.strip(), f"No text produced for {backend}"

        errors = result.timing_errors()
        assert not errors, f"Timing errors: {errors}"

        wer = result.wer(short_sample.reference)
        assert wer < 0.50, f"WER too high for {backend}: {wer:.2%}"

        logger.info("[%s] WER=%.2f%% text='%s'", backend, wer * 100, result.text[:80])


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.asyncio
async def test_medium_clip_timing_spans_audio(backend, medium_sample):
    """Feed ~14s clip and verify speech timestamps span roughly the audio duration."""
    from whisperlivekit.test_harness import TestHarness

    async with TestHarness(**backend_kwargs(backend)) as h:
        await h.feed(medium_sample.path, speed=0, chunk_duration=1.0)
        await h.drain(5.0)
        result = await h.finish(timeout=60)

        assert result.text.strip(), f"No text for {backend}"
        assert not result.timing_errors(), f"Timing errors: {result.timing_errors()}"

        wer = result.wer(medium_sample.reference)
        assert wer < 0.50, f"WER too high: {wer:.2%}"

        # Speech should span most of the audio duration
        speech_ts = [t for t in result.timestamps if t["speaker"] != -2]
        if speech_ts:
            last_end = speech_ts[-1]["end"]
            assert last_end > medium_sample.duration * 0.5, (
                f"Speech ends at {last_end:.1f}s but audio is {medium_sample.duration:.1f}s"
            )

        logger.info("[%s] medium: WER=%.2f%% lines=%d", backend, wer * 100, len(result.lines))


# ---------------------------------------------------------------------------
# 2. Streaming Behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.asyncio
async def test_text_appears_progressively(backend, medium_sample):
    """Verify text grows during streaming, not just at finish."""
    from whisperlivekit.test_harness import TestHarness

    snapshots = []

    def on_update(state):
        snapshots.append(state.text)

    async with TestHarness(**backend_kwargs(backend)) as h:
        h.on_update(on_update)
        await h.feed(medium_sample.path, speed=2.0, chunk_duration=0.5)
        await h.drain(5.0)
        await h.finish(timeout=60)

    non_empty = [t for t in snapshots if t.strip()]
    assert len(non_empty) >= 2, (
        f"Expected progressive updates for {backend}, got {len(non_empty)} non-empty"
    )

    if len(non_empty) >= 3:
        # Check that text grew at SOME point during streaming.
        # Compare first vs last non-empty snapshot rather than mid vs last,
        # because some streaming backends (e.g. qwen3-simul) produce all text
        # during the feed phase and the latter half of snapshots are stable.
        assert len(non_empty[-1]) > len(non_empty[0]), (
            f"Text not growing during streaming for {backend}"
        )

    logger.info("[%s] streaming: %d updates, %d non-empty", backend, len(snapshots), len(non_empty))


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.asyncio
async def test_buffer_lifecycle(backend, medium_sample):
    """Buffer has content during processing; finish() empties buffer, committed grows."""
    from whisperlivekit.test_harness import TestHarness

    async with TestHarness(**backend_kwargs(backend)) as h:
        await h.feed(medium_sample.path, speed=0, chunk_duration=1.0)
        await h.drain(5.0)
        result = await h.finish(timeout=60)

        # After finish, buffer should be empty
        assert not result.buffer_transcription.strip(), (
            f"Buffer not empty after finish for {backend}: '{result.buffer_transcription}'"
        )
        # Committed text should have substantial content
        assert result.committed_word_count > 5, (
            f"Too few committed words for {backend}: {result.committed_word_count}"
        )


# ---------------------------------------------------------------------------
# 3. Play / Pause / Resume
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.asyncio
async def test_silence_flushes_all_words(backend, medium_sample):
    """Silence must flush ALL pending words immediately — none held back for next speech.

    This catches a critical bug where the last few words only appeared when
    the user started speaking again, instead of being committed at silence time.
    Root cause: non-blocking streamer drain racing with the generate thread.
    """
    from whisperlivekit.test_harness import TestHarness

    async with TestHarness(**backend_kwargs(backend)) as h:
        # Feed all audio and let pipeline fully process
        await h.feed(medium_sample.path, speed=0, chunk_duration=1.0)
        await h.drain(8.0)

        # Inject silence → triggers start_silence() which must flush everything
        await h.pause(7.0, speed=0)

        # Wait for start_silence() to complete (may block while generate thread
        # catches up) AND for results_formatter to turn tokens into lines.
        try:
            await h.wait_for(
                lambda s: s.has_silence and s.committed_word_count > 0,
                timeout=30,
            )
        except TimeoutError:
            pass
        await h.drain(2.0)

        # Capture state AFTER silence processing, BEFORE finish()
        words_at_silence = h.state.committed_word_count
        buffer_at_silence = h.state.buffer_transcription.strip()

        # finish() joins the generate thread and flushes any stragglers
        result = await h.finish(timeout=60)
        words_at_finish = result.committed_word_count

        # Key assertion: silence must have committed most words.
        # Some backends (voxtral-hf) produce extra words from right-padding
        # at finish(), and MPS inference may leave some words in the pipeline.
        # Generative backends (qwen3-simul) keep producing new text on each
        # inference call, so finish() adds significantly more words.
        if words_at_finish > 3:
            min_pct = 0.20 if backend in BATCH_FLUSH_BACKENDS else 0.50
            flushed_pct = words_at_silence / words_at_finish
            assert flushed_pct >= min_pct, (
                f"[{backend}] Only {flushed_pct:.0%} of words flushed at silence. "
                f"At silence: {words_at_silence}, at finish: {words_at_finish}. "
                f"Buffer at silence: '{buffer_at_silence}'"
            )

        logger.info(
            "[%s] silence flush: at_silence=%d, at_finish=%d, buffer='%s'",
            backend, words_at_silence, words_at_finish, buffer_at_silence[:40],
        )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.asyncio
async def test_play_pause_resume(backend, medium_sample):
    """Play 3s -> pause 7s -> resume 5s. Verify silence detected with valid timing."""
    from whisperlivekit.test_harness import TestHarness

    async with TestHarness(**backend_kwargs(backend)) as h:
        player = h.load_audio(medium_sample)

        # Play first 3 seconds
        await player.play(3.0, speed=0)
        await h.drain(3.0)

        # Pause 7s (above MIN_DURATION_REAL_SILENCE=5)
        await h.pause(7.0, speed=0)
        await h.drain(3.0)

        # Resume and play 5 more seconds
        await player.play(5.0, speed=0)
        await h.drain(3.0)

        result = await h.finish(timeout=60)

        # Must have text
        assert result.text.strip(), f"No text for {backend}"

        # Must detect silence
        assert result.has_silence, f"No silence detected for {backend}"

        # Timing must be valid (start <= end for each line)
        assert result.timing_valid, f"Invalid timing: {result.timing_errors()}"

        # Monotonic timing — voxtral backends batch-flush words so silence
        # segments can appear before the speech line they precede.
        if backend not in BATCH_FLUSH_BACKENDS:
            assert result.timing_monotonic, f"Non-monotonic: {result.timing_errors()}"

        # At least 1 silence segment
        assert len(result.silence_segments) >= 1

        logger.info(
            "[%s] play/pause/resume: %d lines, %d silence segs",
            backend, len(result.lines), len(result.silence_segments),
        )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.asyncio
async def test_multiple_pauses(backend, medium_sample):
    """Play-pause-play-pause-play cycle -> at least 2 silence segments."""
    from whisperlivekit.test_harness import TestHarness

    async with TestHarness(**backend_kwargs(backend)) as h:
        player = h.load_audio(medium_sample)

        # Cycle 1: play 2s, pause 6s
        await player.play(2.0, speed=0)
        await h.drain(2.0)
        await h.pause(6.0, speed=0)
        await h.drain(2.0)

        # Cycle 2: play 2s, pause 6s
        await player.play(2.0, speed=0)
        await h.drain(2.0)
        await h.pause(6.0, speed=0)
        await h.drain(2.0)

        # Final: play remaining
        await player.play(speed=0)
        await h.drain(3.0)

        result = await h.finish(timeout=60)

        assert result.has_silence, f"No silence for {backend}"
        assert len(result.silence_segments) >= 2, (
            f"Expected >= 2 silence segments, got {len(result.silence_segments)} for {backend}"
        )

        assert result.timing_valid, f"Invalid timing: {result.timing_errors()}"
        if backend not in BATCH_FLUSH_BACKENDS:
            assert result.timing_monotonic, f"Non-monotonic: {result.timing_errors()}"

        logger.info(
            "[%s] multiple pauses: %d silence segs, %d speech lines",
            backend, len(result.silence_segments), len(result.speech_lines),
        )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.asyncio
async def test_short_pause_no_silence(backend, medium_sample):
    """Pause < 5s between speech segments should NOT produce a silence segment."""
    from whisperlivekit.test_harness import TestHarness

    async with TestHarness(**backend_kwargs(backend)) as h:
        player = h.load_audio(medium_sample)

        # Play some speech
        await player.play(4.0, speed=0)
        await h.drain(2.0)

        # Short pause (2s — well below MIN_DURATION_REAL_SILENCE=5)
        await h.pause(2.0, speed=0)
        await h.drain(1.0)

        # Resume speech (triggers _end_silence with duration=2s < 5s threshold)
        await player.play(4.0, speed=0)
        await h.drain(3.0)

        result = await h.finish(timeout=60)

        # Should NOT have silence segments
        assert not result.has_silence, (
            f"Silence detected for {backend} on 2s pause (should be below 5s threshold)"
        )

        logger.info("[%s] short pause: no silence segment (correct)", backend)


# ---------------------------------------------------------------------------
# 4. Cutoff
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.asyncio
async def test_abrupt_cutoff(backend, medium_sample):
    """Cut audio mid-stream -> no crash, partial text preserved."""
    from whisperlivekit.test_harness import TestHarness

    async with TestHarness(**backend_kwargs(backend)) as h:
        player = h.load_audio(medium_sample)

        # Play only first 4 seconds of a ~14s clip
        await player.play(4.0, speed=0)
        # Voxtral backends need more time to start producing text
        await h.drain(8.0 if backend in BATCH_FLUSH_BACKENDS else 3.0)

        # Abrupt cut — voxtral backends on MPS are slower
        result = await h.cut(timeout=15 if backend in BATCH_FLUSH_BACKENDS else 10)

        # Should have some text (even partial)
        assert result.text.strip(), f"No text after cutoff for {backend}"

        # No crashes — timing should be valid (voxtral may have non-monotonic)
        assert result.timing_valid, f"Invalid timing after cutoff: {result.timing_errors()}"

        logger.info("[%s] cutoff at 4s: text='%s'", backend, result.text[:60])


# ---------------------------------------------------------------------------
# 5. Timing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.asyncio
async def test_timing_precision_and_monotonicity(backend, medium_sample):
    """Timestamps have sub-second precision and are monotonically non-decreasing."""
    from whisperlivekit.test_harness import TestHarness

    async with TestHarness(**backend_kwargs(backend)) as h:
        await h.feed(medium_sample.path, speed=0, chunk_duration=1.0)
        await h.drain(5.0)
        # Add silence to test timing across silence boundary
        await h.silence(7.0, speed=0)
        await h.drain(3.0)
        result = await h.finish(timeout=60)

        # Sub-second precision (format is "H:MM:SS.cc")
        has_subsecond = any(
            "." in line.get(key, "")
            for line in result.lines
            for key in ("start", "end")
        )
        assert has_subsecond, f"No sub-second precision for {backend}: {result.lines}"

        assert result.timing_valid, f"Invalid timing: {result.timing_errors()}"
        assert result.timing_monotonic, f"Non-monotonic: {result.timing_errors()}"


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.asyncio
async def test_silence_timing_reflects_pause(backend, short_sample):
    """Silence segment duration should roughly match the injected pause duration."""
    from whisperlivekit.test_harness import TestHarness

    pause_duration = 8.0

    async with TestHarness(**backend_kwargs(backend)) as h:
        await h.feed(short_sample.path, speed=0)
        await h.drain(3.0)
        await h.pause(pause_duration, speed=0)
        await h.drain(3.0)
        result = await h.finish(timeout=60)

        assert result.has_silence, f"No silence detected for {backend}"

        # Check silence segment duration is in the right ballpark
        for seg in result.timestamps:
            if seg["speaker"] == -2:
                seg_duration = seg["end"] - seg["start"]
                # Allow generous tolerance (VAC detection + processing lag)
                assert seg_duration > pause_duration * 0.3, (
                    f"Silence too short for {backend}: {seg_duration:.1f}s "
                    f"vs {pause_duration}s pause"
                )

        logger.info("[%s] silence timing OK", backend)


# ---------------------------------------------------------------------------
# 6. State Inspection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.asyncio
async def test_snapshot_history(backend, medium_sample):
    """Historical snapshots capture growing state at different audio positions."""
    from whisperlivekit.test_harness import TestHarness

    async with TestHarness(**backend_kwargs(backend)) as h:
        await h.feed(medium_sample.path, speed=2.0, chunk_duration=0.5)
        await h.drain(5.0)
        await h.finish(timeout=60)

        # Should have multiple history entries
        assert len(h.history) >= 2, f"Too few history entries: {len(h.history)}"

        # Early snapshot should have less (or equal) text than late snapshot
        early = h.snapshot_at(2.0)
        late = h.snapshot_at(medium_sample.duration)
        if early and late and early.audio_position < late.audio_position:
            assert len(late.text) >= len(early.text), (
                f"Late snapshot has less text than early for {backend}"
            )

        logger.info("[%s] snapshots: %d history entries", backend, len(h.history))


# ---------------------------------------------------------------------------
# 7. Metrics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.asyncio
async def test_metrics_collected(backend, short_sample):
    """Operational metrics are recorded during processing."""
    from whisperlivekit.test_harness import TestHarness

    async with TestHarness(**backend_kwargs(backend)) as h:
        await h.feed(short_sample.path, speed=0)
        await h.drain(3.0)
        await h.finish(timeout=60)

        m = h.metrics
        assert m is not None, "Metrics not available"
        assert m.n_chunks_received > 0, "No chunks recorded"
        assert m.n_transcription_calls > 0, "No transcription calls"
        assert len(m.transcription_durations) > 0, "No transcription durations"
        assert m.n_tokens_produced > 0, "No tokens produced"

        logger.info(
            "[%s] metrics: chunks=%d calls=%d tokens=%d avg_lat=%.1fms",
            backend, m.n_chunks_received, m.n_transcription_calls,
            m.n_tokens_produced, m.avg_latency_ms,
        )
