"""Cumulative ASR counters and a bounded window of inference-call durations."""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict

logger = logging.getLogger(__name__)


@dataclass
class SessionMetrics:
    """Per-session metrics collected by AudioProcessor."""

    session_start: float = 0.0
    total_audio_duration_s: float = 0.0
    total_processing_time_s: float = 0.0

    # Chunk / call counters. A transcription call is one successful invocation
    # of process_iter, start_silence, new_speaker, or finish, including explicit
    # coalescing drains at boundaries and EOF.
    n_chunks_received: int = 0
    n_transcription_calls: int = 0
    n_tokens_produced: int = 0
    n_responses_sent: int = 0
    backpressure_wait_s: float = 0.0
    peak_queued_audio_s: float = 0.0

    # Keep recent calls for the p95 without retaining an entire long session.
    transcription_durations: Deque[float] = field(default_factory=lambda: deque(maxlen=4096))

    # Silence
    n_silence_events: int = 0
    total_silence_duration_s: float = 0.0

    # --- Computed properties ---

    @property
    def rtf(self) -> float:
        """Real-time factor: processing_time / audio_duration."""
        if self.total_audio_duration_s <= 0:
            return 0.0
        return self.total_processing_time_s / self.total_audio_duration_s

    @property
    def avg_latency_ms(self) -> float:
        """Average per-call ASR latency in milliseconds."""
        if not self.n_transcription_calls:
            return 0.0
        return self.total_processing_time_s / self.n_transcription_calls * 1000

    @property
    def p95_latency_ms(self) -> float:
        """95th percentile of the last 4096 ASR calls, in milliseconds."""
        if not self.transcription_durations:
            return 0.0
        sorted_d = sorted(self.transcription_durations)
        idx = int(len(sorted_d) * 0.95)
        idx = min(idx, len(sorted_d) - 1)
        return sorted_d[idx] * 1000

    def to_dict(self) -> Dict:
        """Serialize to a plain dict (JSON-safe)."""
        return {
            "session_start": self.session_start,
            "total_audio_duration_s": round(self.total_audio_duration_s, 3),
            "total_processing_time_s": round(self.total_processing_time_s, 3),
            "rtf": round(self.rtf, 3),
            "n_chunks_received": self.n_chunks_received,
            "n_transcription_calls": self.n_transcription_calls,
            "n_tokens_produced": self.n_tokens_produced,
            "n_responses_sent": self.n_responses_sent,
            "backpressure_wait_s": round(self.backpressure_wait_s, 3),
            "peak_queued_audio_s": round(self.peak_queued_audio_s, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "duration_window_calls": len(self.transcription_durations),
            "n_silence_events": self.n_silence_events,
            "total_silence_duration_s": round(self.total_silence_duration_s, 3),
        }

    def log_summary(self) -> None:
        """Emit a structured log line summarising the session."""
        d = self.to_dict()
        d["session_elapsed_s"] = round(time.time() - self.session_start, 3) if self.session_start else 0
        logger.info(f"SESSION_METRICS {d}")
