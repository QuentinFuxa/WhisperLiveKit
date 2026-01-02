"""
MLX-accelerated Qwen3-ASR backend for WhisperLiveKit.

Provides ``Qwen3MLXASR`` (model holder) and ``Qwen3MLXOnlineProcessor``
(batch-based processor) that plug into WhisperLiveKit's audio processing
pipeline via ``insert_audio_chunk`` / ``process_iter`` / ``get_buffer`` etc.

Uses the ``mlx-qwen3-asr`` package for fast Qwen3 inference on Apple Silicon.
The batch ``session.transcribe()`` API is called on the full accumulated audio
buffer, and LocalAgreement-style diffing (HypothesisBuffer) commits stable
words across consecutive inferences.
"""

import logging
import sys
import time
from typing import List, Tuple

import numpy as np

from whisperlivekit.timed_objects import ASRToken, Transcript

logger = logging.getLogger(__name__)

# Whisper language codes -> Qwen3 canonical language names
# (duplicated from qwen3_asr.py to avoid importing torch at module level)
WHISPER_TO_QWEN3_LANGUAGE = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese",
    "ar": "Arabic", "de": "German", "fr": "French", "es": "Spanish",
    "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
    "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
    "ja": "Japanese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay",
    "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish",
    "pl": "Polish", "cs": "Czech", "fa": "Persian",
    "el": "Greek", "hu": "Hungarian", "mk": "Macedonian", "ro": "Romanian",
}

# Model size aliases -> HuggingFace model IDs
QWEN3_MLX_MODEL_MAPPING = {
    "base": "Qwen/Qwen3-ASR-0.6B",
    "tiny": "Qwen/Qwen3-ASR-0.6B",
    "small": "Qwen/Qwen3-ASR-0.6B",
    "large": "Qwen/Qwen3-ASR-1.7B",
    "medium": "Qwen/Qwen3-ASR-1.7B",
    "large-v3": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "1.7b": "Qwen/Qwen3-ASR-1.7B",
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
}


# ---------------------------------------------------------------------------
# Model holder
# ---------------------------------------------------------------------------


class Qwen3MLXASR:
    """Lightweight model holder -- loads the mlx-qwen3-asr model once and
    keeps it alive for the lifetime of the server."""

    sep = ""
    SAMPLING_RATE = 16_000

    def __init__(self, logfile=sys.stderr, **kwargs):
        import mlx.core as mx
        import mlx_qwen3_asr

        self.logfile = logfile
        self.transcribe_kargs = {}

        lan = kwargs.get("lan", "auto")
        self.original_language = None if lan == "auto" else lan

        # Resolve model ID from size aliases or explicit path
        model_path = kwargs.get("model_dir") or kwargs.get("model_path")
        if not model_path:
            model_size = kwargs.get("model_size", "")
            if model_size and ("/" in model_size or model_size.startswith(".")):
                model_path = model_size
            else:
                model_path = QWEN3_MLX_MODEL_MAPPING.get(
                    (model_size or "base").lower(), "Qwen/Qwen3-ASR-0.6B"
                )

        t0 = time.time()
        logger.info("Loading Qwen3 MLX model '%s' ...", model_path)
        self.session = mlx_qwen3_asr.Session(model_path, dtype=mx.float16)
        logger.info("Qwen3 MLX model loaded in %.2fs", time.time() - t0)

        self.backend_choice = "qwen3-mlx"
        self.tokenizer = None

    def transcribe(self, audio):
        pass  # all work happens in the online processor


# ---------------------------------------------------------------------------
# Online processor
# ---------------------------------------------------------------------------


class Qwen3MLXOnlineProcessor:
    """Batch-based processor that accumulates audio and periodically calls
    ``session.transcribe()`` on the full buffer.

    Uses LocalAgreement-style diffing (HypothesisBuffer) to commit stable
    words across consecutive inferences, exactly like the PyTorch Qwen3
    backend with ``OnlineASRProcessor``.

    Lifecycle (called by ``AudioProcessor.transcription_processor``):

        insert_audio_chunk(pcm, time)  ->  process_iter()  ->  get_buffer()
                      ... repeat ...
        start_silence() / end_silence()
        finish()
    """

    SAMPLING_RATE = 16_000

    def __init__(self, asr: Qwen3MLXASR, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.end = 0.0

        self._session = asr.session
        lan = asr.original_language
        self._language = WHISPER_TO_QWEN3_LANGUAGE.get(lan, "English") if lan else None

        # Audio accumulation
        self.audio_buffer = np.array([], dtype=np.float32)
        self._buffer_time_offset: float = 0.0  # absolute time of audio_buffer[0]

        # Throttle: minimum new audio (in samples) before re-running inference
        self._min_new_samples: int = int(1.0 * self.SAMPLING_RATE)  # 1 second
        self._samples_since_last_inference: int = 0

        # Buffer trimming — keep buffer short for fast re-transcription.
        # The model produces ~0.2x RTF, so 15s buffer = ~3s per call.
        self._max_buffer_sec: float = 15.0
        self._trim_sec: float = 10.0  # keep this many seconds after trimming

        # HypothesisBuffer for LocalAgreement diffing
        self._committed: List[ASRToken] = []
        self._prev_tokens: List[ASRToken] = []  # previous hypothesis (buffer role)
        self._last_committed_time: float = 0.0

        # Global time tracking
        self._global_time_offset: float = 0.0  # extra offset from silences

    # -- audio ingestion --

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        self.end = audio_stream_end_time
        self.audio_buffer = np.append(self.audio_buffer, audio)
        self._samples_since_last_inference += len(audio)

    # -- batch transcription --

    def _transcribe_buffer(self) -> List[ASRToken]:
        """Run batch transcription on the full audio buffer and return tokens."""
        if len(self.audio_buffer) < 400:  # too short for meaningful transcription
            return []

        t0 = time.time()
        try:
            result = self._session.transcribe(
                self.audio_buffer,
                language=self._language,
                return_timestamps=True,
            )
        except Exception as e:
            logger.warning("[qwen3-mlx] transcribe error: %s", e, exc_info=True)
            return []
        dur = time.time() - t0
        audio_dur = len(self.audio_buffer) / self.SAMPLING_RATE
        logger.debug(
            "[qwen3-mlx] transcribed %.1fs audio in %.2fs (%.2fx RTF)",
            audio_dur, dur, dur / max(audio_dur, 0.01),
        )

        text = (result.text or "").strip()
        if not text:
            return []

        # Build tokens from segments (word-level timestamps)
        tokens: List[ASRToken] = []
        if result.segments:
            for i, seg in enumerate(result.segments):
                word = seg["text"]
                start = self._buffer_time_offset + seg["start"]
                end = self._buffer_time_offset + seg["end"]
                label = word if i == 0 else " " + word
                tokens.append(ASRToken(start=start, end=end, text=label))
        else:
            # Fallback: estimate timestamps from word count
            words = text.split()
            step = audio_dur / max(len(words), 1)
            for i, w in enumerate(words):
                t_start = self._buffer_time_offset + i * step
                t_end = self._buffer_time_offset + (i + 1) * step
                label = w if i == 0 else " " + w
                tokens.append(ASRToken(start=t_start, end=t_end, text=label))

        return tokens

    def _local_agreement(self, new_tokens: List[ASRToken]) -> List[ASRToken]:
        """LocalAgreement diffing: commit the longest common prefix between
        the previous hypothesis (``self._prev_tokens``) and the new tokens.

        Before comparing, strips tokens that correspond to already-committed
        audio (i.e., tokens whose start time is before ``_last_committed_time``).
        Also deduplicates boundary tokens (ngram matching) to avoid re-committing
        the tail of the previous committed output.

        Returns the newly committed tokens.
        """
        # Step 1: Only keep tokens that are roughly "new" (after last committed time)
        fresh_tokens = [
            t for t in new_tokens
            if t.start > self._last_committed_time - 0.1
        ]

        # Step 2: Remove duplicates at the boundary with committed tokens
        # (like HypothesisBuffer.insert's ngram dedup)
        if fresh_tokens and self._committed:
            max_ngram = min(len(self._committed), len(fresh_tokens), 5)
            for n in range(1, max_ngram + 1):
                committed_ngram = " ".join(
                    t.text.strip() for t in self._committed[-n:]
                )
                fresh_ngram = " ".join(
                    t.text.strip() for t in fresh_tokens[:n]
                )
                if committed_ngram == fresh_ngram:
                    fresh_tokens = fresh_tokens[n:]
                    break

        # Step 3: LocalAgreement -- longest common prefix between prev and fresh
        committed: List[ASRToken] = []
        prev = self._prev_tokens
        i = 0
        j = 0

        while i < len(fresh_tokens) and j < len(prev):
            if fresh_tokens[i].text.strip() == prev[j].text.strip():
                # Agreement: commit this token (use the new token's timestamps)
                committed.append(fresh_tokens[i])
                i += 1
                j += 1
            else:
                break

        # The remaining fresh tokens become the new "previous hypothesis"
        self._prev_tokens = fresh_tokens[i:] if i < len(fresh_tokens) else []
        return committed

    def _trim_buffer_if_needed(self):
        """Trim the audio buffer if it exceeds max_buffer_sec.

        Keeps the last ``_trim_sec`` seconds of audio. Also adjusts
        committed token tracking and buffer_time_offset.
        """
        buffer_dur = len(self.audio_buffer) / self.SAMPLING_RATE
        if buffer_dur <= self._max_buffer_sec:
            return

        keep_sec = self._trim_sec
        keep_samples = int(keep_sec * self.SAMPLING_RATE)
        cut_samples = len(self.audio_buffer) - keep_samples
        if cut_samples <= 0:
            return

        cut_sec = cut_samples / self.SAMPLING_RATE
        self.audio_buffer = self.audio_buffer[cut_samples:]
        self._buffer_time_offset += cut_sec

        # Remove committed tokens that are before the new buffer start
        self._committed = [
            t for t in self._committed if t.end > self._buffer_time_offset
        ]

        logger.debug(
            "[qwen3-mlx] trimmed buffer: cut %.1fs, new offset %.1f, buffer %.1fs",
            cut_sec, self._buffer_time_offset, len(self.audio_buffer) / self.SAMPLING_RATE,
        )

    # -- interface methods --

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        """Process the current audio buffer.

        Throttles inference to at least 1s of new audio between calls.
        Returns (newly_committed_tokens, audio_processed_upto_time).
        """
        try:
            # Throttle: skip if not enough new audio since last inference
            if (not is_last
                    and self._samples_since_last_inference < self._min_new_samples):
                return [], self.end

            self._samples_since_last_inference = 0

            # Trim buffer if too long
            self._trim_buffer_if_needed()

            # Run batch transcription
            new_tokens = self._transcribe_buffer()

            # LocalAgreement diffing
            committed = self._local_agreement(new_tokens)

            if committed:
                self._committed.extend(committed)
                self._last_committed_time = committed[-1].end

            return committed, self.end
        except Exception as e:
            logger.warning("[qwen3-mlx] process_iter error: %s", e, exc_info=True)
            return [], self.end

    def get_buffer(self) -> Transcript:
        """Return the unconfirmed text (the tail of the last hypothesis
        that was not committed by LocalAgreement)."""
        if not self._prev_tokens:
            return Transcript(start=None, end=None, text="")

        text = "".join(t.text for t in self._prev_tokens)
        start = self._prev_tokens[0].start
        end = self._prev_tokens[-1].end
        return Transcript(start=start, end=end, text=text)

    def _flush_all(self) -> List[ASRToken]:
        """Force a final transcription and commit all remaining words."""
        # Run one last transcription on the full buffer
        self._samples_since_last_inference = self._min_new_samples  # bypass throttle
        new_tokens = self._transcribe_buffer()

        # Commit everything: first the agreed prefix, then the remainder
        committed = self._local_agreement(new_tokens)

        # Also commit any remaining buffer tokens
        remaining = self._prev_tokens
        self._prev_tokens = []

        all_new = committed + remaining
        if all_new:
            self._committed.extend(all_new)
            self._last_committed_time = all_new[-1].end

        return all_new

    def _reset_for_new_utterance(self):
        """Reset buffers for a new utterance, preserving time continuity."""
        new_offset = self._buffer_time_offset + len(self.audio_buffer) / self.SAMPLING_RATE
        saved_end = self.end

        self.audio_buffer = np.array([], dtype=np.float32)
        self._buffer_time_offset = new_offset
        self._samples_since_last_inference = 0
        self._committed = []
        self._prev_tokens = []

        self.end = saved_end

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        """Flush pending words when silence starts.

        Unlike other backends, does NOT reset the audio buffer — the model
        produces better results re-transcribing the full accumulated audio.
        Buffer trimming at 30s handles memory naturally.
        """
        words = self._flush_all()
        logger.info("[qwen3-mlx] start_silence: flushed %d words", len(words))
        return words, self.end

    def end_silence(self, silence_duration: float, offset: float):
        self._global_time_offset += silence_duration
        self.end += silence_duration

    def new_speaker(self, change_speaker):
        self.start_silence()

    def warmup(self, audio, init_prompt=""):
        pass

    def finish(self) -> Tuple[List[ASRToken], float]:
        words = self._flush_all()
        logger.info("[qwen3-mlx] finish: flushed %d words", len(words))
        return words, self.end
