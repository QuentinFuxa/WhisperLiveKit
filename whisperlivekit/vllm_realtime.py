"""
vLLM Realtime WebSocket streaming backend for WhisperLiveKit.

Connects to a vLLM server's ``/v1/realtime`` WebSocket endpoint to stream
audio and receive transcription deltas.  Uses ``websockets.sync.client``
for simplicity since ``process_iter`` runs inside ``asyncio.to_thread``.

Provides ``VLLMRealtimeASR`` (lightweight model holder) and
``VLLMRealtimeOnlineProcessor`` (streaming processor) that plug into
WhisperLiveKit's audio processing pipeline.
"""

import base64
import json
import logging
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from whisperlivekit.timed_objects import ASRToken, Transcript

logger = logging.getLogger(__name__)


class VLLMRealtimeASR:
    """Lightweight model holder — stores connection info for the vLLM server."""

    sep = " "
    SAMPLING_RATE = 16000
    backend_choice = "vllm-realtime"

    def __init__(self, vllm_url="ws://localhost:8000/v1/realtime",
                 model_name="Qwen/Qwen3-ASR-1.7B", lan="auto", **kwargs):
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.original_language = None if lan == "auto" else lan
        self.tokenizer = None

    def transcribe(self, audio):
        pass


class VLLMRealtimeOnlineProcessor:
    """
    Online processor that streams audio to a vLLM Realtime WebSocket.

    Uses a background thread for WebSocket receiving and
    ``websockets.sync.client`` for the sync WebSocket connection.
    """

    SAMPLING_RATE = 16000
    # Minimum audio samples before connecting (0.5s of audio)
    _MIN_CONNECT_SAMPLES = SAMPLING_RATE // 2

    def __init__(self, asr: VLLMRealtimeASR):
        self.asr = asr
        self.end = 0.0
        self.buffer = []
        self.audio_buffer = np.array([], dtype=np.float32)

        self._reset_state()

        logger.info(
            "[vllm-realtime] Initialized. url=%s model=%s",
            asr.vllm_url, asr.model_name,
        )

    def _reset_state(self):
        self._pending_audio = np.zeros(0, dtype=np.float32)
        self._ws = None
        self._recv_thread: Optional[threading.Thread] = None
        self._connected = False
        self._done = False
        self._recv_error: Optional[Exception] = None

        # Text accumulation and word extraction
        self._accumulated_text = ""
        self._n_committed_words = 0
        self._total_audio_duration = 0.0
        self._global_time_offset = 0.0

        # Lock for text state accessed from both recv thread and main thread
        self._text_lock = threading.Lock()

    # ── Interface methods ──

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        self.end = audio_stream_end_time
        self._pending_audio = np.append(self._pending_audio, audio)
        self.audio_buffer = self._pending_audio

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        try:
            return self._process_iter_inner(is_last)
        except Exception as e:
            logger.warning("[vllm-realtime] process_iter exception: %s", e, exc_info=True)
            return [], self.end

    def get_buffer(self) -> Transcript:
        """Return all uncommitted text as buffer."""
        self._drain_deltas()
        with self._text_lock:
            text = self._accumulated_text
        if not text:
            return Transcript(start=None, end=None, text="")

        words = text.split()
        uncommitted = words[self._n_committed_words:]
        if uncommitted:
            return Transcript(start=self.end, end=self.end, text=" ".join(uncommitted))
        return Transcript(start=None, end=None, text="")

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        """Flush all pending words when silence starts.

        Sends commit(final=true) to signal end of utterance, waits for
        transcription.done, flushes all words, then prepares for reconnection
        on the next utterance.
        """
        if not self._connected or self._done:
            words = self._flush_all_pending_words()
            logger.info("[vllm-realtime] start_silence (not connected): flushed %d words", len(words))
            return words, self.end

        # Send any remaining buffered audio
        self._send_pending_audio()

        # Signal end of stream
        self._send_commit(final=True)

        # Wait for transcription.done
        self._wait_for_done(timeout=10.0)

        # Flush all remaining words
        words = self._flush_all_pending_words()

        # Close and reset for next utterance
        self._close_ws()
        old_offset = self._global_time_offset + self._total_audio_duration
        self._reset_state()
        self._global_time_offset = old_offset

        logger.info("[vllm-realtime] start_silence: flushed %d words", len(words))
        return words, self.end

    def end_silence(self, silence_duration: float, offset: float):
        self._global_time_offset += silence_duration
        self.end += silence_duration

    def new_speaker(self, change_speaker):
        self.start_silence()

    def warmup(self, audio, init_prompt=""):
        pass

    def finish(self) -> Tuple[List[ASRToken], float]:
        """Close connection and flush all remaining words."""
        if self._connected and not self._done:
            # Send remaining audio
            self._send_pending_audio()

            # Signal final commit
            self._send_commit(final=True)

            # Wait for transcription.done
            self._wait_for_done(timeout=30.0)

        # Flush all words
        words = self._flush_all_pending_words()

        # Close WebSocket
        self._close_ws()

        logger.info("[vllm-realtime] finish: flushed %d words", len(words))
        return words, self.end

    # ── WebSocket connection management ──

    def _connect(self):
        """Connect to the vLLM realtime WebSocket and start the receive thread."""
        from websockets.sync.client import connect

        url = self.asr.vllm_url
        logger.info("[vllm-realtime] Connecting to %s", url)

        self._ws = connect(url)

        # Send session.update to select model
        self._ws.send(json.dumps({
            "type": "session.update",
            "model": self.asr.model_name,
        }))

        # Send initial commit(final=false) to start generation
        self._send_commit(final=False)

        # Start receive thread
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        self._connected = True
        logger.info("[vllm-realtime] Connected and started receive thread")

    def _close_ws(self):
        """Close the WebSocket connection and join the receive thread."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

        if self._recv_thread is not None:
            self._recv_thread.join(timeout=5.0)
            self._recv_thread = None

    def _recv_loop(self):
        """Background thread: receive messages from the vLLM WebSocket."""
        try:
            while not self._done and self._ws is not None:
                try:
                    raw = self._ws.recv(timeout=0.1)
                except TimeoutError:
                    continue
                except Exception:
                    break

                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "transcription.delta":
                    delta = msg.get("delta", "")
                    if delta:
                        with self._text_lock:
                            self._accumulated_text += delta

                elif msg_type == "transcription.done":
                    done_text = msg.get("text", "")
                    if done_text:
                        with self._text_lock:
                            # Replace accumulated text with final text
                            self._accumulated_text = done_text
                    self._done = True
                    break

        except Exception as e:
            logger.error("[vllm-realtime] recv_loop error: %s", e, exc_info=True)
            self._recv_error = e
            self._done = True

    # ── Protocol messages ──

    def _send_commit(self, final: bool):
        """Send input_audio_buffer.commit message."""
        if self._ws is None:
            return
        try:
            self._ws.send(json.dumps({
                "type": "input_audio_buffer.commit",
                "final": final,
            }))
        except Exception as e:
            logger.warning("[vllm-realtime] Failed to send commit: %s", e)

    def _send_audio(self, audio: np.ndarray):
        """Send audio as a base64-encoded PCM16 append message."""
        if self._ws is None:
            return

        # Convert float32 [-1, 1] to int16 PCM
        pcm16 = (audio * 32767).astype(np.int16)
        audio_bytes = pcm16.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        try:
            self._ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }))
        except Exception as e:
            logger.warning("[vllm-realtime] Failed to send audio: %s", e)

    def _send_pending_audio(self):
        """Send all pending audio to the vLLM server."""
        if len(self._pending_audio) == 0:
            return

        # Track total audio duration for timestamp estimation
        self._total_audio_duration += len(self._pending_audio) / self.SAMPLING_RATE

        # Send in chunks of 0.5s to avoid overwhelming the WebSocket
        chunk_samples = self.SAMPLING_RATE // 2
        while len(self._pending_audio) >= chunk_samples:
            chunk = self._pending_audio[:chunk_samples]
            self._send_audio(chunk)
            self._pending_audio = self._pending_audio[chunk_samples:]

        # Send remaining audio if any
        if len(self._pending_audio) > 0:
            self._send_audio(self._pending_audio)
            self._pending_audio = np.zeros(0, dtype=np.float32)

        self.audio_buffer = self._pending_audio

    # ── Receive helpers ──

    def _drain_deltas(self):
        """No-op since the recv thread accumulates text directly."""
        pass

    def _wait_for_done(self, timeout: float = 10.0):
        """Wait for transcription.done message from the server."""
        deadline = time.time() + timeout
        while not self._done and time.time() < deadline:
            time.sleep(0.05)

        if not self._done:
            logger.warning("[vllm-realtime] Timed out waiting for transcription.done")

    # ── Word extraction (same approach as VoxtralHF) ──

    def _time_for_word(self, word_idx: int, n_words_total: int) -> Tuple[float, float]:
        """Estimate timestamps by linearly distributing words across audio duration."""
        duration = max(self._total_audio_duration, 0.001)
        n_total = max(n_words_total, 1)

        start_time = (word_idx / n_total) * duration + self._global_time_offset
        end_time = ((word_idx + 1) / n_total) * duration + self._global_time_offset

        return start_time, end_time

    def _extract_new_words(self) -> List[ASRToken]:
        """Extract complete words (all but the last, which may still grow)."""
        with self._text_lock:
            text = self._accumulated_text
        if not text:
            return []

        words = text.split()
        new_words: List[ASRToken] = []
        n_words_total = len(words)

        while len(words) > self._n_committed_words + 1:
            word = words[self._n_committed_words]
            start_time, end_time = self._time_for_word(self._n_committed_words, n_words_total)

            text_out = word if self._n_committed_words == 0 else " " + word
            new_words.append(ASRToken(start=start_time, end=end_time, text=text_out))
            self._n_committed_words += 1

        return new_words

    def _flush_all_pending_words(self) -> List[ASRToken]:
        """Flush ALL words including the last partial one."""
        with self._text_lock:
            text = self._accumulated_text
        if not text:
            return []

        words = text.split()
        new_words: List[ASRToken] = []
        n_words_total = max(len(words), 1)

        while self._n_committed_words < len(words):
            word = words[self._n_committed_words]
            start_time, end_time = self._time_for_word(self._n_committed_words, n_words_total)

            text_out = word if self._n_committed_words == 0 else " " + word
            new_words.append(ASRToken(start=start_time, end=end_time, text=text_out))
            self._n_committed_words += 1

        return new_words

    # ── Core processing ──

    def _process_iter_inner(self, is_last: bool) -> Tuple[List[ASRToken], float]:
        # Connect when we have enough audio buffered
        if not self._connected:
            if len(self._pending_audio) >= self._MIN_CONNECT_SAMPLES:
                self._connect()
                self._send_pending_audio()
            else:
                return [], self.end

        # Send any new pending audio
        if self._connected and not self._done:
            self._send_pending_audio()

        # If connection closed unexpectedly but new audio arrived, reconnect
        if self._done and len(self._pending_audio) >= self._MIN_CONNECT_SAMPLES:
            flush_words = self._flush_all_pending_words()
            old_offset = self._global_time_offset + self._total_audio_duration
            self._close_ws()
            self._reset_state()
            self._global_time_offset = old_offset
            self._connect()
            self._send_pending_audio()
            return flush_words, self.end

        # Extract complete words
        new_words = self._extract_new_words()

        if new_words:
            logger.info(
                "[vllm-realtime] returning %d words: %s",
                len(new_words), [w.text for w in new_words],
            )

        self.buffer = []
        return new_words, self.end
