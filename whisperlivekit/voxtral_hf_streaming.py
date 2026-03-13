"""
Voxtral Mini Realtime streaming backend using HuggingFace Transformers.

Uses VoxtralRealtimeForConditionalGeneration with a background generate thread
and queue-based audio feeding for real-time streaming transcription.
Supports CUDA, CPU, and MPS devices.
"""

import logging
import queue
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from whisperlivekit.timed_objects import ASRToken, Transcript

logger = logging.getLogger(__name__)


class VoxtralHFStreamingASR:
    """Voxtral model holder using HuggingFace Transformers."""

    sep = " "

    def __init__(self, logfile=sys.stderr, **kwargs):
        import torch
        from transformers import (
            AutoProcessor,
            VoxtralRealtimeForConditionalGeneration,
        )

        self.logfile = logfile
        self.transcribe_kargs = {}

        lan = kwargs.get("lan", "auto")
        self.original_language = None if lan == "auto" else lan

        DEFAULT_MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"
        model_path = kwargs.get("model_dir") or kwargs.get("model_path")
        if not model_path:
            model_size = kwargs.get("model_size", "")
            if model_size and ("/" in model_size or model_size.startswith(".")):
                model_path = model_size
            else:
                model_path = DEFAULT_MODEL

        t = time.time()
        logger.info(f"Loading Voxtral model '{model_path}' via HF Transformers...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        logger.info(f"Voxtral HF model loaded in {time.time() - t:.2f}s on {self.model.device}")

        self.backend_choice = "voxtral"
        self.tokenizer = None  # sentence tokenizer — not needed for streaming

    def transcribe(self, audio):
        pass


class VoxtralHFStreamingOnlineProcessor:
    """
    Online processor for Voxtral streaming ASR via HuggingFace Transformers.

    Uses a background thread running model.generate() with a queue-based
    input_features_generator and TextIteratorStreamer for real-time output.
    Each decoded token corresponds to ~80ms of audio.
    """

    SAMPLING_RATE = 16000

    def __init__(self, asr: VoxtralHFStreamingASR, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.end = 0.0
        self.buffer = []
        self.audio_buffer = np.array([], dtype=np.float32)

        processor = asr.processor
        self._first_chunk_samples = processor.num_samples_first_audio_chunk
        self._chunk_samples = processor.num_samples_per_audio_chunk
        self._chunk_step = processor.raw_audio_length_per_tok
        # num_right_pad_tokens is a method in some transformers versions, a property in others
        n_right_pad = processor.num_right_pad_tokens
        if callable(n_right_pad):
            n_right_pad = n_right_pad()
        self._right_pad_samples = int(n_right_pad * processor.raw_audio_length_per_tok)
        self._seconds_per_token = processor.raw_audio_length_per_tok / self.SAMPLING_RATE

        self._reset_state()

        logger.info(
            f"[voxtral-hf] Initialized. first_chunk={self._first_chunk_samples} samples, "
            f"chunk={self._chunk_samples}, step={self._chunk_step}, "
            f"right_pad={self._right_pad_samples}"
        )

    def _reset_state(self):
        self._pending_chunks: List[np.ndarray] = []
        self._pending_len = 0
        self._audio_queue: queue.Queue = queue.Queue()
        self._streamer_texts: List[str] = []
        self._generate_thread: Optional[threading.Thread] = None
        self._generate_started = False
        self._generate_finished = False
        self._generate_error: Optional[Exception] = None

        # Text accumulation (list of fragments, joined on demand)
        self._text_fragments: List[str] = []
        self._text_len = 0
        # Fragment position tracking for accurate word timestamps:
        # each entry is (char_offset_in_full_text, audio_tok_pos_consumed)
        self._fragment_positions: List[Tuple[int, int]] = []
        self._n_text_tokens_received = 0
        self._n_audio_tokens_fed = 0
        # Audio tokens actually consumed by the model (tracked inside generator)
        self._n_audio_tokens_consumed = 0
        self._n_committed_words = 0
        self._global_time_offset = 0.0

        # Event signalled by the generate thread when it finishes
        self._generate_done = threading.Event()

        # Lock for text state accessed from both generate thread and main thread
        self._text_lock = threading.Lock()

    # ── Audio / text helpers ──

    def _get_pending_audio(self) -> np.ndarray:
        """Flatten pending audio chunks into a single array."""
        if not self._pending_chunks:
            return np.zeros(0, dtype=np.float32)
        if len(self._pending_chunks) == 1:
            return self._pending_chunks[0]
        flat = np.concatenate(self._pending_chunks)
        self._pending_chunks = [flat]
        return flat

    def _set_pending_audio(self, arr: np.ndarray):
        """Replace pending audio with a single array."""
        if len(arr) == 0:
            self._pending_chunks = []
            self._pending_len = 0
        else:
            self._pending_chunks = [arr]
            self._pending_len = len(arr)

    def _get_accumulated_text(self) -> str:
        """Get the full accumulated text (joins fragments if needed)."""
        if not self._text_fragments:
            return ""
        if len(self._text_fragments) == 1:
            return self._text_fragments[0]
        joined = "".join(self._text_fragments)
        self._text_fragments = [joined]
        return joined

    # ── Interface methods ──

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        self.end = audio_stream_end_time
        self._pending_chunks.append(audio)
        self._pending_len += len(audio)
        self.audio_buffer = audio  # diagnostic only

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        try:
            return self._process_iter_inner(is_last)
        except Exception as e:
            logger.warning(f"[voxtral-hf] process_iter exception: {e}", exc_info=True)
            return [], self.end

    def get_buffer(self) -> Transcript:
        """Return all uncommitted text as buffer.

        Drains the streamer first so late-arriving tokens (common on
        slower devices like MPS) are picked up even between audio chunks.
        """
        self._drain_streamer()
        with self._text_lock:
            text = self._get_accumulated_text()
        if not text:
            return Transcript(start=None, end=None, text="")

        words = text.split()
        uncommitted = words[self._n_committed_words:]
        if uncommitted:
            return Transcript(start=self.end, end=self.end, text=" ".join(uncommitted))
        return Transcript(start=None, end=None, text="")

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        """Flush all uncommitted words when silence starts.

        Feeds right-padding (silence) so the model has enough future context
        to emit the last few tokens, then drains repeatedly until the model
        has finished producing text.  Without right-padding the model holds
        back the last few words because it hasn't seen enough audio yet.
        """
        if not self._generate_started or self._generate_finished:
            self._drain_streamer()
            words = self._flush_all_pending_words()
            logger.info(f"[voxtral-hf] start_silence (no thread): flushed {len(words)} words")
            return words, self.end

        # Feed any remaining real audio
        self._feed_pending_audio()

        # Add right-padding so the model can decode trailing tokens.
        # Don't count these toward _n_audio_tokens_fed — they're not
        # real audio and shouldn't affect word timestamp calculations.
        if self._right_pad_samples > 0:
            right_pad = np.zeros(self._right_pad_samples, dtype=np.float32)
            self._pending_chunks.append(right_pad)
            self._pending_len += len(right_pad)
            saved_count = self._n_audio_tokens_fed
            self._feed_pending_audio()
            self._n_audio_tokens_fed = saved_count

        # Drain in a loop: the model may continue producing text tokens after
        # the audio queue is empty (autoregressive generation). Each iteration
        # uses an event-driven blocking drain with short timeouts.
        all_words: List[ASRToken] = []
        for _ in range(5):
            self._drain_streamer_blocking(timeout=5.0)
            batch = self._flush_all_pending_words()
            all_words.extend(batch)
            if not batch:
                break  # no new text — model has caught up

        logger.info(f"[voxtral-hf] start_silence: flushed {len(all_words)} words")
        return all_words, self.end

    def end_silence(self, silence_duration: float, offset: float):
        self._global_time_offset += silence_duration
        self.end += silence_duration

    def new_speaker(self, change_speaker):
        self.start_silence()

    def warmup(self, audio, init_prompt=""):
        pass

    def finish(self) -> Tuple[List[ASRToken], float]:
        """Flush remaining audio with right-padding and stop the generate thread."""
        # Add right-padding so the model can finish decoding
        if self._right_pad_samples > 0:
            right_pad = np.zeros(self._right_pad_samples, dtype=np.float32)
            self._pending_chunks.append(right_pad)
            self._pending_len += len(right_pad)

        # Feed remaining audio
        if self._generate_started and not self._generate_finished:
            self._feed_pending_audio()
            # Signal end of audio
            self._audio_queue.put(None)
            # Wait for generate to finish
            if self._generate_thread is not None:
                self._generate_thread.join(timeout=30.0)
        elif not self._generate_started and self._pending_len >= self._first_chunk_samples:
            # Never started but have enough audio — start and immediately finish
            self._start_generate_thread()
            self._feed_pending_audio()
            self._audio_queue.put(None)
            if self._generate_thread is not None:
                self._generate_thread.join(timeout=30.0)

        self._drain_streamer()
        words = self._flush_all_pending_words()
        logger.info(f"[voxtral-hf] finish: flushed {len(words)} words")
        return words, self.end

    # ── Generate thread management ──

    def _start_generate_thread(self):
        """Start model.generate() in a background thread with streaming."""
        import torch
        from transformers import TextIteratorStreamer

        processor = self.asr.processor
        model = self.asr.model

        # Extract first chunk
        pending = self._get_pending_audio()
        first_chunk_audio = pending[:self._first_chunk_samples]
        self._set_pending_audio(pending[self._first_chunk_samples:])
        # First chunk covers multiple audio tokens
        self._n_audio_tokens_fed += max(1, self._first_chunk_samples // self._chunk_step)

        first_inputs = processor(
            first_chunk_audio,
            is_streaming=True,
            is_first_audio_chunk=True,
            return_tensors="pt",
        )
        first_inputs = first_inputs.to(model.device, dtype=model.dtype)

        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        self._streamer = streamer

        audio_queue = self._audio_queue

        def input_features_gen():
            # Track audio consumption inside the generator (runs in generate thread)
            self._n_audio_tokens_consumed = max(1, self._first_chunk_samples // self._chunk_step)
            yield first_inputs.input_features
            while True:
                chunk_audio = audio_queue.get()
                if chunk_audio is None:
                    break
                self._n_audio_tokens_consumed += 1
                inputs = processor(
                    chunk_audio,
                    is_streaming=True,
                    is_first_audio_chunk=False,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device, dtype=model.dtype)
                yield inputs.input_features

        def run_generate():
            try:
                with torch.no_grad():
                    # Pass generator as input_features — the model detects GeneratorType
                    # and internally converts it to input_features_generator
                    generate_kwargs = {
                        k: v for k, v in first_inputs.items()
                        if k != "input_features"
                    }
                    model.generate(
                        input_features=input_features_gen(),
                        streamer=streamer,
                        **generate_kwargs,
                    )
            except Exception as e:
                logger.error(f"[voxtral-hf] generate error: {e}", exc_info=True)
                self._generate_error = e
            finally:
                self._generate_finished = True
                self._generate_done.set()

        self._generate_thread = threading.Thread(target=run_generate, daemon=True)
        self._generate_thread.start()
        self._generate_started = True
        logger.info("[voxtral-hf] generate thread started")

    def _feed_pending_audio(self):
        """Convert pending audio into properly-sized chunks for the generator."""
        chunk_size = self._chunk_samples
        step_size = self._chunk_step

        pending = self._get_pending_audio()
        while len(pending) >= chunk_size:
            chunk = pending[:chunk_size]
            self._audio_queue.put(chunk)
            pending = pending[step_size:]
            self._n_audio_tokens_fed += 1

        self._set_pending_audio(pending)
        self.audio_buffer = pending

    def _append_text_fragment(self, text_fragment: str):
        """Append a text fragment with its audio position (must hold _text_lock)."""
        self._fragment_positions.append((self._text_len, self._n_audio_tokens_consumed))
        self._text_fragments.append(text_fragment)
        self._text_len += len(text_fragment)
        self._n_text_tokens_received += 1

    def _drain_streamer(self):
        """Non-blocking drain of all available text from the streamer."""
        if not self._generate_started:
            return

        text_queue = self._streamer.text_queue
        while True:
            try:
                text_fragment = text_queue.get_nowait()
            except queue.Empty:
                break
            if text_fragment is None:
                self._generate_finished = True
                break
            if text_fragment:
                with self._text_lock:
                    self._append_text_fragment(text_fragment)

    def _drain_streamer_blocking(self, timeout=30.0):
        """Blocking drain: wait for the generate thread to finish producing text.

        Uses the _generate_done event to know when the model is truly finished.
        Falls back to text-queue polling with adaptive timeouts.
        """
        if not self._generate_started or self._generate_finished:
            self._drain_streamer()
            return

        text_queue = self._streamer.text_queue
        deadline = time.time() + timeout
        # Count consecutive empty polls to detect when model has caught up
        empty_streak = 0

        while time.time() < deadline:
            remaining = max(deadline - time.time(), 0.01)

            # If generate thread is done, do a final flush and exit
            if self._generate_done.is_set() or self._generate_finished:
                self._drain_streamer()
                return

            # Adaptive wait: short while audio is queued, longer once queue is empty
            if self._audio_queue.empty():
                wait = min(remaining, 0.5)
            else:
                wait = min(remaining, 0.1)

            try:
                text_fragment = text_queue.get(timeout=wait)
            except queue.Empty:
                empty_streak += 1
                # Only exit if audio queue is empty AND we've had enough empty polls
                # This prevents premature exit when the model is slow
                if self._audio_queue.empty() and empty_streak >= 4:
                    break
                continue

            empty_streak = 0
            if text_fragment is None:
                self._generate_finished = True
                break
            if text_fragment:
                with self._text_lock:
                    self._append_text_fragment(text_fragment)

    # ── Word extraction ──

    def _pos_to_time(self, token_position: int) -> float:
        """Convert audio token position to seconds."""
        return token_position * self._seconds_per_token + self._global_time_offset

    def _audio_pos_for_char(self, char_idx: int) -> int:
        """Look up the audio token position for a character index in the text.

        Uses the fragment position index recorded when text arrives from the
        generate thread.  Returns the audio position of the fragment that
        contains ``char_idx``, giving much better word timestamps than the
        old uniform-distribution heuristic.
        """
        if not self._fragment_positions:
            return 0
        # _fragment_positions is sorted by char_offset — find the last entry
        # whose char_offset <= char_idx (the fragment containing this char).
        pos = 0
        for offset, audio_tok in self._fragment_positions:
            if offset > char_idx:
                break
            pos = audio_tok
        return pos

    def _word_timestamps(self, text: str, words: List[str], start_idx: int, end_idx: int) -> List[Tuple[int, int]]:
        """Compute (tok_start, tok_end) for words[start_idx:end_idx] using fragment positions."""
        # Build char offsets for each word
        result = []
        char_pos = 0
        for i, word in enumerate(words):
            if i > 0:
                char_pos += 1  # space separator
            if start_idx <= i < end_idx:
                tok_start = self._audio_pos_for_char(char_pos)
                tok_end = self._audio_pos_for_char(char_pos + len(word))
                result.append((tok_start, tok_end))
            char_pos += len(word)
        return result

    def _extract_new_words(self) -> List[ASRToken]:
        """Extract complete words (all but the last, which may still be growing)."""
        with self._text_lock:
            text = self._get_accumulated_text()
        if not text:
            return []

        words = text.split()
        new_words: List[ASRToken] = []
        n_to_commit = len(words) - 1  # keep last word (may still grow)

        if n_to_commit <= self._n_committed_words:
            return []

        timestamps = self._word_timestamps(text, words, self._n_committed_words, n_to_commit)

        for tok_start, tok_end in timestamps:
            word = words[self._n_committed_words]
            start_time = self._pos_to_time(tok_start)
            end_time = self._pos_to_time(max(tok_end, tok_start + 1))

            text_out = word if self._n_committed_words == 0 else " " + word
            new_words.append(ASRToken(start=start_time, end=end_time, text=text_out))
            self._n_committed_words += 1

        return new_words

    def _flush_all_pending_words(self) -> List[ASRToken]:
        """Flush ALL words including the last partial one."""
        with self._text_lock:
            text = self._get_accumulated_text()
        if not text:
            return []

        words = text.split()
        new_words: List[ASRToken] = []

        if self._n_committed_words >= len(words):
            return []

        timestamps = self._word_timestamps(text, words, self._n_committed_words, len(words))

        for tok_start, tok_end in timestamps:
            word = words[self._n_committed_words]
            start_time = self._pos_to_time(tok_start)
            end_time = self._pos_to_time(max(tok_end, tok_start + 1))

            text_out = word if self._n_committed_words == 0 else " " + word
            new_words.append(ASRToken(start=start_time, end=end_time, text=text_out))
            self._n_committed_words += 1

        return new_words

    # ── Core processing ──

    def _process_iter_inner(self, is_last: bool) -> Tuple[List[ASRToken], float]:
        # Start generate thread when enough audio is buffered
        if not self._generate_started:
            if self._pending_len >= self._first_chunk_samples:
                self._start_generate_thread()
                self._feed_pending_audio()
            else:
                return [], self.end

        # Feed any new pending audio
        if self._generate_started and not self._generate_finished:
            self._feed_pending_audio()

        # If generate finished unexpectedly (EOS) but new audio arrived, restart
        if self._generate_finished and self._pending_len >= self._first_chunk_samples:
            self._drain_streamer()
            flush_words = self._flush_all_pending_words()
            # Reset for new utterance
            old_offset = self._global_time_offset
            self._reset_state()
            self._global_time_offset = old_offset
            self._start_generate_thread()
            self._feed_pending_audio()
            return flush_words, self.end

        # Drain available text from streamer
        self._drain_streamer()

        # Extract complete words
        new_words = self._extract_new_words()

        if new_words:
            logger.info(f"[voxtral-hf] returning {len(new_words)} words: {[w.text for w in new_words]}")

        self.buffer = []
        return new_words, self.end
