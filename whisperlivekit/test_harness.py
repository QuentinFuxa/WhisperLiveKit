"""In-process testing harness for the full WhisperLiveKit pipeline.

Wraps AudioProcessor to provide a controllable, observable interface
for testing transcription, diarization, silence detection, and timing
without needing a running server or WebSocket connection.

Designed for use by AI agents: feed audio with timeline control,
inspect state at any point, pause/resume to test silence detection,
cut to test abrupt termination.

Usage::

    import asyncio
    from whisperlivekit.test_harness import TestHarness

    async def main():
        async with TestHarness(model_size="base", lan="en") as h:
            # Load audio with timeline control
            player = h.load_audio("interview.wav")

            # Play first 5 seconds at real-time speed
            await player.play(5.0, speed=1.0)
            print(h.state.text)  # Check what's transcribed so far

            # Pause for 7 seconds (triggers silence detection)
            await h.pause(7.0, speed=1.0)
            assert h.state.has_silence

            # Resume playback
            await player.play(5.0, speed=1.0)

            # Finish and evaluate
            result = await h.finish()
            print(f"WER: {result.wer('expected transcription'):.2%}")
            print(f"Speakers: {result.speakers}")
            print(f"Silence segments: {len(result.silence_segments)}")

            # Inspect historical state at specific audio position
            snap = h.snapshot_at(3.0)
            print(f"At 3s: '{snap.text}'")

    asyncio.run(main())
"""

import asyncio
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from whisperlivekit.timed_objects import FrontData

logger = logging.getLogger(__name__)

# Engine cache: avoids reloading models when switching backends in tests.
# Key is a frozen config tuple, value is the TranscriptionEngine instance.
_engine_cache: Dict[Tuple, "Any"] = {}

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # s16le


def _parse_time(time_str: str) -> float:
    """Parse 'H:MM:SS.cc' timestamp string to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def load_audio_pcm(audio_path: str, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Load any audio file and convert to PCM s16le mono via ffmpeg."""
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        "-loglevel", "error",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr.decode().strip()}")
    if not proc.stdout:
        raise RuntimeError(f"ffmpeg produced no output for {audio_path}")
    return proc.stdout


# ---------------------------------------------------------------------------
# TestState — observable transcription state
# ---------------------------------------------------------------------------

@dataclass
class TestState:
    """Observable transcription state at a point in time.

    Provides accessors for inspecting lines, buffers, speakers, timestamps,
    silence segments, and computing evaluation metrics like WER.

    All time-based queries accept seconds as floats.
    """

    lines: List[Dict[str, Any]] = field(default_factory=list)
    buffer_transcription: str = ""
    buffer_diarization: str = ""
    buffer_translation: str = ""
    remaining_time_transcription: float = 0.0
    remaining_time_diarization: float = 0.0
    audio_position: float = 0.0
    status: str = ""
    error: str = ""

    @classmethod
    def from_front_data(cls, front_data: FrontData, audio_position: float = 0.0) -> "TestState":
        d = front_data.to_dict()
        return cls(
            lines=d.get("lines", []),
            buffer_transcription=d.get("buffer_transcription", ""),
            buffer_diarization=d.get("buffer_diarization", ""),
            buffer_translation=d.get("buffer_translation", ""),
            remaining_time_transcription=d.get("remaining_time_transcription", 0),
            remaining_time_diarization=d.get("remaining_time_diarization", 0),
            audio_position=audio_position,
            status=d.get("status", ""),
            error=d.get("error", ""),
        )

    # ── Text accessors ──

    @property
    def text(self) -> str:
        """Full transcription: committed lines + buffer."""
        parts = [l["text"] for l in self.lines if l.get("text")]
        if self.buffer_transcription:
            parts.append(self.buffer_transcription)
        return " ".join(parts)

    @property
    def committed_text(self) -> str:
        """Only committed (finalized) lines, no buffer."""
        return " ".join(l["text"] for l in self.lines if l.get("text"))

    @property
    def committed_word_count(self) -> int:
        """Number of words in committed lines."""
        t = self.committed_text
        return len(t.split()) if t.strip() else 0

    @property
    def buffer_word_count(self) -> int:
        """Number of words in the unconfirmed buffer."""
        return len(self.buffer_transcription.split()) if self.buffer_transcription.strip() else 0

    # ── Speaker accessors ──

    @property
    def speakers(self) -> Set[int]:
        """Set of speaker IDs (excluding silence marker -2)."""
        return {l["speaker"] for l in self.lines if l.get("speaker", 0) > 0}

    @property
    def n_speakers(self) -> int:
        return len(self.speakers)

    def speaker_at(self, time_s: float) -> Optional[int]:
        """Speaker ID at the given timestamp, or None if no segment covers it."""
        line = self.line_at(time_s)
        return line["speaker"] if line else None

    def speakers_in(self, start_s: float, end_s: float) -> Set[int]:
        """All speaker IDs active in the time range (excluding silence -2)."""
        return {
            l.get("speaker")
            for l in self.lines_between(start_s, end_s)
            if l.get("speaker", 0) > 0
        }

    @property
    def speaker_timeline(self) -> List[Dict[str, Any]]:
        """Timeline: [{"start": float, "end": float, "speaker": int}] for all lines."""
        return [
            {
                "start": _parse_time(l.get("start", "0:00:00")),
                "end": _parse_time(l.get("end", "0:00:00")),
                "speaker": l.get("speaker", -1),
            }
            for l in self.lines
        ]

    @property
    def n_speaker_changes(self) -> int:
        """Number of speaker transitions (excluding silence segments)."""
        speech = [s for s in self.speaker_timeline if s["speaker"] != -2]
        return sum(
            1 for i in range(1, len(speech))
            if speech[i]["speaker"] != speech[i - 1]["speaker"]
        )

    # ── Silence accessors ──

    @property
    def has_silence(self) -> bool:
        """Whether any silence segment (speaker=-2) exists."""
        return any(l.get("speaker") == -2 for l in self.lines)

    @property
    def silence_segments(self) -> List[Dict[str, Any]]:
        """All silence segments (raw line dicts)."""
        return [l for l in self.lines if l.get("speaker") == -2]

    def silence_at(self, time_s: float) -> bool:
        """True if time_s falls within a silence segment."""
        line = self.line_at(time_s)
        return line is not None and line.get("speaker") == -2

    # ── Line / segment accessors ──

    @property
    def speech_lines(self) -> List[Dict[str, Any]]:
        """Lines excluding silence segments."""
        return [l for l in self.lines if l.get("speaker", 0) != -2 and l.get("text")]

    def line_at(self, time_s: float) -> Optional[Dict[str, Any]]:
        """Find the line covering the given timestamp (seconds)."""
        for line in self.lines:
            start = _parse_time(line.get("start", "0:00:00"))
            end = _parse_time(line.get("end", "0:00:00"))
            if start <= time_s <= end:
                return line
        return None

    def text_at(self, time_s: float) -> Optional[str]:
        """Text of the segment covering the given timestamp."""
        line = self.line_at(time_s)
        return line["text"] if line else None

    def lines_between(self, start_s: float, end_s: float) -> List[Dict[str, Any]]:
        """All lines overlapping the time range [start_s, end_s]."""
        result = []
        for line in self.lines:
            ls = _parse_time(line.get("start", "0:00:00"))
            le = _parse_time(line.get("end", "0:00:00"))
            if le >= start_s and ls <= end_s:
                result.append(line)
        return result

    def text_between(self, start_s: float, end_s: float) -> str:
        """Concatenated text of all lines overlapping the time range."""
        return " ".join(
            l["text"] for l in self.lines_between(start_s, end_s)
            if l.get("text")
        )

    # ── Evaluation ──

    def wer(self, reference: str) -> float:
        """Word Error Rate of committed text against reference.

        Returns:
            WER as a float (0.0 = perfect, 1.0 = 100% error rate).
        """
        from whisperlivekit.metrics import compute_wer
        result = compute_wer(reference, self.committed_text)
        return result["wer"]

    def wer_detailed(self, reference: str) -> Dict:
        """Full WER breakdown: substitutions, insertions, deletions, etc."""
        from whisperlivekit.metrics import compute_wer
        return compute_wer(reference, self.committed_text)

    # ── Timing validation ──

    @property
    def timestamps(self) -> List[Dict[str, Any]]:
        """All line timestamps as [{"start": float, "end": float, "speaker": int, "text": str}]."""
        result = []
        for line in self.lines:
            result.append({
                "start": _parse_time(line.get("start", "0:00:00")),
                "end": _parse_time(line.get("end", "0:00:00")),
                "speaker": line.get("speaker", -1),
                "text": line.get("text", ""),
            })
        return result

    @property
    def timing_valid(self) -> bool:
        """All timestamps have start <= end and no negative values."""
        for ts in self.timestamps:
            if ts["start"] < 0 or ts["end"] < 0:
                return False
            if ts["end"] < ts["start"]:
                return False
        return True

    @property
    def timing_monotonic(self) -> bool:
        """Line start times are non-decreasing."""
        stamps = self.timestamps
        for i in range(1, len(stamps)):
            if stamps[i]["start"] < stamps[i - 1]["start"]:
                return False
        return True

    def timing_errors(self) -> List[str]:
        """Human-readable list of timing issues found."""
        errors = []
        stamps = self.timestamps
        for i, ts in enumerate(stamps):
            if ts["start"] < 0:
                errors.append(f"Line {i}: negative start {ts['start']:.2f}s")
            if ts["end"] < 0:
                errors.append(f"Line {i}: negative end {ts['end']:.2f}s")
            if ts["end"] < ts["start"]:
                errors.append(
                    f"Line {i}: end ({ts['end']:.2f}s) < start ({ts['start']:.2f}s)"
                )
        for i in range(1, len(stamps)):
            if stamps[i]["start"] < stamps[i - 1]["start"]:
                errors.append(
                    f"Line {i}: start ({stamps[i]['start']:.2f}s) < previous start "
                    f"({stamps[i-1]['start']:.2f}s) — non-monotonic"
                )
        return errors


# ---------------------------------------------------------------------------
# AudioPlayer — timeline control for a loaded audio file
# ---------------------------------------------------------------------------

class AudioPlayer:
    """Controls playback of a loaded audio file through the pipeline.

    Tracks position in the audio, enabling play/pause/resume patterns::

        player = h.load_audio("speech.wav")
        await player.play(3.0)           # Play first 3 seconds
        await h.pause(7.0)               # 7s silence (triggers detection)
        await player.play(5.0)           # Play next 5 seconds
        await player.play()              # Play all remaining audio

    Args:
        harness: The TestHarness instance.
        pcm_data: Raw PCM s16le 16kHz mono bytes.
        sample_rate: Audio sample rate (default 16000).
    """

    def __init__(self, harness: "TestHarness", pcm_data: bytes, sample_rate: int = SAMPLE_RATE):
        self._harness = harness
        self._pcm = pcm_data
        self._sr = sample_rate
        self._bps = sample_rate * BYTES_PER_SAMPLE  # bytes per second
        self._pos = 0  # current position in bytes

    @property
    def position(self) -> float:
        """Current playback position in seconds."""
        return self._pos / self._bps

    @property
    def duration(self) -> float:
        """Total audio duration in seconds."""
        return len(self._pcm) / self._bps

    @property
    def remaining(self) -> float:
        """Remaining audio in seconds."""
        return max(0.0, (len(self._pcm) - self._pos) / self._bps)

    @property
    def done(self) -> bool:
        """True if all audio has been played."""
        return self._pos >= len(self._pcm)

    async def play(
        self,
        duration_s: Optional[float] = None,
        speed: float = 1.0,
        chunk_duration: float = 0.5,
    ) -> None:
        """Play audio from the current position.

        Args:
            duration_s: Seconds of audio to play. None = all remaining.
            speed: 1.0 = real-time, 0 = instant, >1 = faster.
            chunk_duration: Size of each chunk fed to the pipeline (seconds).
        """
        if duration_s is None:
            end_pos = len(self._pcm)
        else:
            end_pos = min(self._pos + int(duration_s * self._bps), len(self._pcm))

        # Align to sample boundary
        end_pos = (end_pos // BYTES_PER_SAMPLE) * BYTES_PER_SAMPLE

        if end_pos <= self._pos:
            return

        segment = self._pcm[self._pos:end_pos]
        self._pos = end_pos
        await self._harness.feed_pcm(segment, speed=speed, chunk_duration=chunk_duration)

    async def play_until(
        self,
        time_s: float,
        speed: float = 1.0,
        chunk_duration: float = 0.5,
    ) -> None:
        """Play until reaching time_s in the audio timeline."""
        target = min(int(time_s * self._bps), len(self._pcm))
        target = (target // BYTES_PER_SAMPLE) * BYTES_PER_SAMPLE

        if target <= self._pos:
            return

        segment = self._pcm[self._pos:target]
        self._pos = target
        await self._harness.feed_pcm(segment, speed=speed, chunk_duration=chunk_duration)

    def seek(self, time_s: float) -> None:
        """Move the playback cursor without feeding audio."""
        pos = int(time_s * self._bps)
        pos = (pos // BYTES_PER_SAMPLE) * BYTES_PER_SAMPLE
        self._pos = max(0, min(pos, len(self._pcm)))

    def reset(self) -> None:
        """Reset to the beginning of the audio."""
        self._pos = 0


# ---------------------------------------------------------------------------
# TestHarness — pipeline controller
# ---------------------------------------------------------------------------

class TestHarness:
    """In-process testing harness for the full WhisperLiveKit pipeline.

    Use as an async context manager. Provides methods to feed audio,
    pause/resume, inspect state, and evaluate results.

    Methods:
        load_audio(path)    → AudioPlayer with play/seek controls
        feed(path, speed)   → feed entire audio file (simple mode)
        pause(duration)     → inject silence (triggers detection if > 5s)
        drain(seconds)      → let pipeline catch up
        finish()            → flush and return final state
        cut()               → abrupt stop, return partial state
        wait_for(pred)      → wait for condition on state

    State inspection:
        .state              → current TestState
        .history            → all historical states
        .snapshot_at(t)     → state at audio position t
        .metrics            → SessionMetrics (latency, RTF, etc.)

    Args:
        All keyword arguments passed to AudioProcessor.
        Common: model_size, lan, backend, diarization, vac.
    """

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("pcm_input", True)
        self._engine_kwargs = kwargs
        self._processor = None
        self._results_gen = None
        self._collect_task = None
        self._state = TestState()
        self._audio_position = 0.0
        self._history: List[TestState] = []
        self._on_update: Optional[Callable[[TestState], None]] = None

    async def __aenter__(self) -> "TestHarness":
        from whisperlivekit.audio_processor import AudioProcessor
        from whisperlivekit.core import TranscriptionEngine

        # Cache engines by config to avoid reloading models when switching
        # backends between tests. The singleton is reset only when the
        # requested config doesn't match any cached engine.
        cache_key = tuple(sorted(self._engine_kwargs.items()))

        if cache_key not in _engine_cache:
            TranscriptionEngine.reset()
            _engine_cache[cache_key] = TranscriptionEngine(**self._engine_kwargs)

        engine = _engine_cache[cache_key]

        self._processor = AudioProcessor(transcription_engine=engine)
        self._results_gen = await self._processor.create_tasks()
        self._collect_task = asyncio.create_task(self._collect_results())
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._processor:
            await self._processor.cleanup()
        if self._collect_task and not self._collect_task.done():
            self._collect_task.cancel()
            try:
                await self._collect_task
            except asyncio.CancelledError:
                pass

    async def _collect_results(self) -> None:
        """Background task: consume results from the pipeline."""
        try:
            async for front_data in self._results_gen:
                self._state = TestState.from_front_data(front_data, self._audio_position)
                self._history.append(self._state)
                if self._on_update:
                    self._on_update(self._state)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("Result collector ended: %s", e)

    # ── Properties ──

    @property
    def state(self) -> TestState:
        """Current transcription state (updated live as results arrive)."""
        return self._state

    @property
    def history(self) -> List[TestState]:
        """All states received so far, in order."""
        return self._history

    @property
    def audio_position(self) -> float:
        """How many seconds of audio have been fed so far."""
        return self._audio_position

    @property
    def metrics(self):
        """Pipeline's SessionMetrics (latency, RTF, token counts, etc.)."""
        if self._processor:
            return self._processor.metrics
        return None

    def on_update(self, callback: Callable[[TestState], None]) -> None:
        """Register a callback invoked on each new state update."""
        self._on_update = callback

    # ── Audio loading and feeding ──

    def load_audio(self, source) -> AudioPlayer:
        """Load audio and return a player with timeline control.

        Args:
            source: Path to audio file (str), or a TestSample with .path attribute.

        Returns:
            AudioPlayer with play/play_until/seek/reset methods.
        """
        path = source.path if hasattr(source, "path") else str(source)
        pcm = load_audio_pcm(path)
        return AudioPlayer(self, pcm)

    async def feed(
        self,
        audio_path: str,
        speed: float = 1.0,
        chunk_duration: float = 0.5,
    ) -> None:
        """Feed an entire audio file to the pipeline (simple mode).

        For timeline control (play/pause/resume), use load_audio() instead.

        Args:
            audio_path: Path to any audio file ffmpeg can decode.
            speed: Playback speed (1.0 = real-time, 0 = instant).
            chunk_duration: Size of each PCM chunk in seconds.
        """
        pcm = load_audio_pcm(audio_path)
        await self.feed_pcm(pcm, speed=speed, chunk_duration=chunk_duration)

    async def feed_pcm(
        self,
        pcm_data: bytes,
        speed: float = 1.0,
        chunk_duration: float = 0.5,
    ) -> None:
        """Feed raw PCM s16le 16kHz mono bytes to the pipeline.

        Args:
            pcm_data: Raw PCM bytes.
            speed: Playback speed multiplier.
            chunk_duration: Duration of each chunk sent (seconds).
        """
        chunk_bytes = int(chunk_duration * SAMPLE_RATE * BYTES_PER_SAMPLE)
        offset = 0
        while offset < len(pcm_data):
            end = min(offset + chunk_bytes, len(pcm_data))
            await self._processor.process_audio(pcm_data[offset:end])
            chunk_seconds = (end - offset) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            self._audio_position += chunk_seconds
            offset = end
            if speed > 0:
                await asyncio.sleep(chunk_duration / speed)

    # ── Pause / silence ──

    async def pause(self, duration_s: float, speed: float = 1.0) -> None:
        """Inject silence to simulate a pause in speech.

        Pauses > 5s trigger silence segment detection (MIN_DURATION_REAL_SILENCE).
        Pauses < 5s are treated as brief gaps and produce no silence segment
        (provided speech resumes afterward).

        Args:
            duration_s: Duration of silence in seconds.
            speed: Playback speed (1.0 = real-time, 0 = instant).
        """
        silent_pcm = bytes(int(duration_s * SAMPLE_RATE * BYTES_PER_SAMPLE))
        await self.feed_pcm(silent_pcm, speed=speed)

    async def silence(self, duration_s: float, speed: float = 1.0) -> None:
        """Alias for pause(). Inject silence for the given duration."""
        await self.pause(duration_s, speed=speed)

    # ── Waiting ──

    async def wait_for(
        self,
        predicate: Callable[[TestState], bool],
        timeout: float = 30.0,
        poll_interval: float = 0.1,
    ) -> TestState:
        """Wait until predicate(state) returns True.

        Raises:
            TimeoutError: If the condition is not met within timeout.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if predicate(self._state):
                return self._state
            await asyncio.sleep(poll_interval)
        raise TimeoutError(
            f"Condition not met within {timeout}s. "
            f"Current state: {len(self._state.lines)} lines, "
            f"buffer='{self._state.buffer_transcription[:50]}', "
            f"audio_pos={self._audio_position:.1f}s"
        )

    async def wait_for_text(self, timeout: float = 30.0) -> TestState:
        """Wait until any transcription text appears."""
        return await self.wait_for(lambda s: s.text.strip(), timeout=timeout)

    async def wait_for_lines(self, n: int = 1, timeout: float = 30.0) -> TestState:
        """Wait until at least n committed speech lines exist."""
        return await self.wait_for(lambda s: len(s.speech_lines) >= n, timeout=timeout)

    async def wait_for_silence(self, timeout: float = 30.0) -> TestState:
        """Wait until a silence segment is detected."""
        return await self.wait_for(lambda s: s.has_silence, timeout=timeout)

    async def wait_for_speakers(self, n: int = 2, timeout: float = 30.0) -> TestState:
        """Wait until at least n distinct speakers are detected."""
        return await self.wait_for(lambda s: s.n_speakers >= n, timeout=timeout)

    async def drain(self, seconds: float = 2.0) -> None:
        """Let the pipeline process without feeding audio.

        Useful after feeding audio to allow the ASR backend to catch up.
        """
        await asyncio.sleep(seconds)

    # ── Finishing ──

    async def finish(self, timeout: float = 30.0) -> TestState:
        """Signal end of audio and wait for pipeline to flush all results.

        Returns:
            Final TestState with all committed lines and empty buffer.
        """
        await self._processor.process_audio(b"")
        if self._collect_task:
            try:
                await asyncio.wait_for(self._collect_task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for pipeline to finish after %.0fs", timeout)
            except asyncio.CancelledError:
                pass
        return self._state

    async def cut(self, timeout: float = 5.0) -> TestState:
        """Abrupt audio stop — signal EOF and return current state quickly.

        Simulates user closing the connection mid-speech. Sends EOF but
        uses a short timeout, so partial results are returned even if
        the pipeline hasn't fully flushed.

        Returns:
            TestState with whatever has been processed so far.
        """
        await self._processor.process_audio(b"")
        if self._collect_task:
            try:
                await asyncio.wait_for(self._collect_task, timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        return self._state

    # ── History inspection ──

    def snapshot_at(self, audio_time: float) -> Optional[TestState]:
        """Find the historical state closest to when audio_time was reached.

        Args:
            audio_time: Audio position in seconds.

        Returns:
            The TestState captured at that point, or None if no history.
        """
        if not self._history:
            return None
        best = None
        best_diff = float("inf")
        for s in self._history:
            diff = abs(s.audio_position - audio_time)
            if diff < best_diff:
                best_diff = diff
                best = s
        return best

    # ── Debug ──

    def print_state(self) -> None:
        """Print current state to stdout for debugging."""
        s = self._state
        print(f"--- Audio: {self._audio_position:.1f}s | Status: {s.status} ---")
        for line in s.lines:
            speaker = line.get("speaker", "?")
            text = line.get("text", "")
            start = line.get("start", "")
            end = line.get("end", "")
            tag = "SILENCE" if speaker == -2 else f"Speaker {speaker}"
            print(f"  [{start} -> {end}] {tag}: {text}")
        if s.buffer_transcription:
            print(f"  [buffer] {s.buffer_transcription}")
        if s.buffer_diarization:
            print(f"  [diar buffer] {s.buffer_diarization}")
        print(f"  Speakers: {s.speakers or 'none'} | Silence: {s.has_silence}")
        print()
