"""Per-session online processor for the qwen3-streaming backend.

State machine per WebSocket session::

    insert_audio_chunk -> pending sample buffer (cheap, no GPU work)
    process_iter       -> when enough audio is pending, featurize incrementally,
                          append mel to the streamer (one bounded full-hypothesis
                          decode), then emit newly committed words as ASRTokens
    get_buffer         -> the unstable hypothesis tail (display-only)
    start_silence/finish -> flush mel tail + right context, finalize, emit rest

Word timestamps are linear interpolations across each newly committed span
(the streamer is text-only). Typical error is on the order of a second —
fine for line/diarization alignment, not for precise word timing; use the
``qwen3-vllm`` backend (ForcedAligner) when exact timestamps matter.

Decode pacing is self-adjusting: each decode must "pay for itself", so the
next one waits for at least ``_PACING x`` the previous decode duration in new
audio. On hardware slower than the audio rate the chunks grow until the
per-audio-second cost amortizes, instead of lag-spiraling.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import List, Tuple

import numpy as np

from whisperlivekit.timed_objects import ASRToken, Transcript

logger = logging.getLogger(__name__)


class Qwen3StreamingOnlineProcessor:
    SAMPLING_RATE = 16_000
    _PACING = 1.2
    # Average distance between the committed frontier and the audio head:
    # right context (0.64s) plus the stable-commit holdback (~6 words).
    _COMMIT_LAG_SECONDS = 2.5
    _MIN_WORD_SECONDS = 0.05

    def __init__(self, asr, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        session_language = getattr(asr, "_session_language", None)
        self._language = session_language or asr.original_language
        self._detected_language = (
            self._language if self._language != "auto" else None
        )

        self.streamer = asr.build_streamer(self._language)
        self.mel = asr.new_mel_extractor()

        self.end = 0.0
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer = []

        self._emitted_words: list[str] = []
        self._any_word_emitted = False
        self._last_commit_time = 0.0
        self._last_event: dict | None = None
        self._last_decode_duration = 0.0

    # ------------------------------------------------------------------
    # audio_processor contract
    # ------------------------------------------------------------------

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        self.end = audio_stream_end_time
        self.audio_buffer = np.append(self.audio_buffer, audio.astype(np.float32))

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        try:
            if is_last:
                return self._flush(), self.end
            pending_sec = len(self.audio_buffer) / self.SAMPLING_RATE
            due_after = max(self.asr.chunk_sec, self._PACING * self._last_decode_duration)
            if pending_sec < due_after:
                return [], self.end
            event = self._decode_pending()
            if event is None:
                return [], self.end
            return self._emit_committed(event["committed"], self.end), self.end
        except Exception as exc:
            logger.warning("[qwen3-streaming] process_iter error: %s", exc, exc_info=True)
            return [], self.end

    def get_buffer(self) -> Transcript:
        unstable = (self._last_event or {}).get("unstable", "")
        if not unstable:
            return Transcript(start=None, end=None, text="")
        return Transcript(start=self._last_commit_time, end=self.end, text=unstable)

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        tokens = self._flush()
        logger.info("[qwen3-streaming] start_silence: flushed %d words", len(tokens))
        self._reset_for_next_utterance()
        return tokens, self.end

    def end_silence(self, silence_duration: float, offset: float):
        self.end += silence_duration
        self._last_commit_time += silence_duration

    def new_speaker(self, change_speaker):
        self.start_silence()

    def warmup(self, audio, init_prompt=""):
        return None

    def finish(self) -> Tuple[List[ASRToken], float]:
        tokens = self._flush()
        logger.info("[qwen3-streaming] finish: flushed %d words", len(tokens))
        return tokens, self.end

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _decode_pending(self) -> dict | None:
        """Featurize pending audio and run one streamer update."""
        audio = self.audio_buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        frames = self.mel.append(audio)
        if frames is None or frames.shape[1] == 0:
            return None
        started = time.perf_counter()
        with self.asr.decode_lock:
            event = self.streamer.append_mel_chunk(frames.to(self.asr.device))
        self._last_decode_duration = time.perf_counter() - started
        self._last_event = event
        return event

    def _flush(self) -> List[ASRToken]:
        """Flush mel tail and right context, finalize the active segment."""
        audio = self.audio_buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        with self.asr.decode_lock:
            frames = self.mel.append(audio)
            if frames is not None and frames.shape[1] > 0:
                self.streamer.append_mel_chunk(frames.to(self.asr.device))
            tail = self.mel.flush()
            if tail is not None and tail.shape[1] > 0:
                self.streamer.append_mel_chunk(tail.to(self.asr.device))
            right_context_frames = self.asr.right_context_frames
            if right_context_frames > 0 and self.streamer.events:
                import torch

                zeros = torch.zeros(
                    1, right_context_frames, self.asr.n_mels, device=self.asr.device
                )
                self.streamer.append_mel_chunk(zeros, is_flush=True)
            final = self.streamer.finalize(finalize_mode="latest")
        self._last_event = None
        return self._emit_committed(final.final_text, self.end, flush=True)

    def _reset_for_next_utterance(self):
        self.streamer = self.asr.build_streamer(self._language)
        self.mel.reset()
        self._emitted_words = []
        self._last_event = None
        self._last_decode_duration = 0.0
        self._last_commit_time = self.end

    def _emit_committed(
        self, committed_text: str, event_time: float, flush: bool = False
    ) -> List[ASRToken]:
        """Diff global committed text against already-emitted words.

        Output is append-only: if a segment-rollover finalization revised
        already-emitted words, the revision is dropped (logged) and the new
        text becomes the diff baseline.
        """
        target = committed_text.split()
        n_emitted = len(self._emitted_words)
        if len(target) < n_emitted:
            logger.debug(
                "[qwen3-streaming] committed text shrank (%d -> %d words); keeping baseline",
                n_emitted,
                len(target),
            )
            return []
        if target[:n_emitted] != self._emitted_words:
            logger.debug(
                "[qwen3-streaming] %d already-emitted words were revised at a segment "
                "boundary; revision dropped from output",
                n_emitted,
            )
        new_words = target[n_emitted:]
        self._emitted_words = target
        if not new_words:
            return []

        t0 = self._last_commit_time
        t1 = event_time if flush else event_time - self._COMMIT_LAG_SECONDS
        t1 = min(max(t1, t0 + self._MIN_WORD_SECONDS * len(new_words)), self.end)
        if t1 <= t0:
            t1 = min(t0 + self._MIN_WORD_SECONDS * len(new_words), self.end)
            t1 = max(t1, t0)

        tokens: list[ASRToken] = []
        span = t1 - t0
        for idx, word in enumerate(new_words):
            start = t0 + span * idx / len(new_words)
            end = t0 + span * (idx + 1) / len(new_words)
            text = word if not self._any_word_emitted else " " + word
            self._any_word_emitted = True
            tokens.append(
                ASRToken(
                    start=start,
                    end=end,
                    text=text,
                    detected_language=self._detected_language,
                )
            )
        self._last_commit_time = t1
        return tokens
