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
        self._chunk_step = processor.num_samples_per_audio_chunk_step
        self._right_pad_samples = int(
            processor.num_right_pad_tokens * processor.raw_audio_length_per_tok
        )
        self._seconds_per_token = processor.raw_audio_length_per_tok / self.SAMPLING_RATE

        self._reset_state()

        logger.info(
            f"[voxtral-hf] Initialized. first_chunk={self._first_chunk_samples} samples, "
            f"chunk={self._chunk_samples}, step={self._chunk_step}, "
            f"right_pad={self._right_pad_samples}"
        )

    def _reset_state(self):
        self._pending_audio = np.zeros(0, dtype=np.float32)
        self._audio_queue: queue.Queue = queue.Queue()
        self._streamer_texts: List[str] = []
        self._generate_thread: Optional[threading.Thread] = None
        self._generate_started = False
        self._generate_finished = False
        self._generate_error: Optional[Exception] = None

        # Text accumulation and word extraction
        self._accumulated_text = ""
        self._n_text_tokens_received = 0
        self._n_committed_words = 0
        self._global_time_offset = 0.0

        # Lock for text state accessed from both generate thread and main thread
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
            logger.warning(f"[voxtral-hf] process_iter exception: {e}", exc_info=True)
            return [], self.end

    def get_buffer(self) -> Transcript:
        """Return all uncommitted text as buffer."""
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
        """Flush all uncommitted words when silence starts."""
        self._drain_streamer()
        words = self._flush_all_pending_words()
        logger.info(f"[voxtral-hf] start_silence: flushed {len(words)} words")
        return words, self.end

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
            self._pending_audio = np.append(self._pending_audio, right_pad)

        # Feed remaining audio
        if self._generate_started and not self._generate_finished:
            self._feed_pending_audio()
            # Signal end of audio
            self._audio_queue.put(None)
            # Wait for generate to finish
            if self._generate_thread is not None:
                self._generate_thread.join(timeout=30.0)
        elif not self._generate_started and len(self._pending_audio) >= self._first_chunk_samples:
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
        first_chunk_audio = self._pending_audio[:self._first_chunk_samples]
        self._pending_audio = self._pending_audio[self._first_chunk_samples:]

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
            yield first_inputs.input_features
            while True:
                chunk_audio = audio_queue.get()
                if chunk_audio is None:
                    break
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
                    model.generate(
                        input_features_generator=input_features_gen(),
                        streamer=streamer,
                        **first_inputs,
                    )
            except Exception as e:
                logger.error(f"[voxtral-hf] generate error: {e}", exc_info=True)
                self._generate_error = e
            finally:
                self._generate_finished = True

        self._generate_thread = threading.Thread(target=run_generate, daemon=True)
        self._generate_thread.start()
        self._generate_started = True
        logger.info("[voxtral-hf] generate thread started")

    def _feed_pending_audio(self):
        """Convert pending audio into properly-sized chunks for the generator."""
        chunk_size = self._chunk_samples
        step_size = self._chunk_step

        while len(self._pending_audio) >= chunk_size:
            chunk = self._pending_audio[:chunk_size]
            self._audio_queue.put(chunk)
            self._pending_audio = self._pending_audio[step_size:]

        self.audio_buffer = self._pending_audio

    def _drain_streamer(self):
        """Non-blocking drain of all available text from the streamer."""
        if not self._generate_started:
            return

        streamer = self._streamer
        try:
            for text_fragment in streamer:
                if text_fragment:
                    with self._text_lock:
                        self._accumulated_text += text_fragment
                    self._n_text_tokens_received += 1
                # Check if more is immediately available (non-blocking)
                if streamer.text_queue.empty():
                    break
        except StopIteration:
            pass

    # ── Word extraction ──

    def _pos_to_time(self, token_position: int) -> float:
        """Convert token position to seconds."""
        return token_position * self._seconds_per_token + self._global_time_offset

    def _extract_new_words(self) -> List[ASRToken]:
        """Extract complete words (all but the last, which may still be growing)."""
        with self._text_lock:
            text = self._accumulated_text
        if not text:
            return []

        words = text.split()
        new_words: List[ASRToken] = []
        n_tokens = self._n_text_tokens_received
        n_words_total = len(words)

        while len(words) > self._n_committed_words + 1:
            word = words[self._n_committed_words]
            word_idx = self._n_committed_words

            tok_start = int(word_idx / n_words_total * n_tokens) if n_words_total > 0 else 0
            tok_end = int((word_idx + 1) / n_words_total * n_tokens) if n_words_total > 0 else 0

            start_time = self._pos_to_time(tok_start)
            end_time = self._pos_to_time(tok_end)

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
        n_tokens = max(self._n_text_tokens_received, 1)
        n_words_total = max(len(words), 1)

        while self._n_committed_words < len(words):
            word = words[self._n_committed_words]
            word_idx = self._n_committed_words

            tok_start = int(word_idx / n_words_total * n_tokens)
            tok_end = int((word_idx + 1) / n_words_total * n_tokens)

            start_time = self._pos_to_time(tok_start)
            end_time = self._pos_to_time(tok_end)

            text_out = word if self._n_committed_words == 0 else " " + word
            new_words.append(ASRToken(start=start_time, end=end_time, text=text_out))
            self._n_committed_words += 1

        return new_words

    # ── Core processing ──

    def _process_iter_inner(self, is_last: bool) -> Tuple[List[ASRToken], float]:
        # Start generate thread when enough audio is buffered
        if not self._generate_started:
            if len(self._pending_audio) >= self._first_chunk_samples:
                self._start_generate_thread()
                self._feed_pending_audio()
            else:
                return [], self.end

        # Feed any new pending audio
        if self._generate_started and not self._generate_finished:
            self._feed_pending_audio()

        # If generate finished unexpectedly (EOS) but new audio arrived, restart
        if self._generate_finished and len(self._pending_audio) >= self._first_chunk_samples:
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
