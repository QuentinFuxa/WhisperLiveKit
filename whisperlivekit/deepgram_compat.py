"""Deepgram Live Transcription compatibility for WhisperLiveKit.

The adapter keeps Deepgram's independent concepts separate:

* interim hypotheses may be replaced;
* final word ranges are immutable and emitted exactly once;
* endpointing is driven by VAD pauses on the submitted audio clock;
* UtteranceEnd is a separate gap detector anchored to a finalized word.
"""

import asyncio
import json
import logging
import re
import time
import unicodedata
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from fastapi import WebSocket, WebSocketDisconnect

from whisperlivekit.timed_objects import AudioStreamEvent, FrontData

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINTING_MS = 10
_MIN_UTTERANCE_END_MS = 1000
_MAX_UTTERANCE_END_MS = 5000
_EVENTS_DONE = object()


def _parse_time_str(value: Any) -> float:
    """Parse a frontend timestamp or pass through a numeric timestamp."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def _parse_bool(params: Mapping[str, str], name: str, default: bool) -> bool:
    raw = params.get(name)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{name} must be true or false.")


def _parse_integer_ms(raw: Any, name: str) -> int:
    value = str(raw).strip()
    if not value or not value.isascii() or not value.isdecimal():
        raise ValueError(f"{name} must be an integer number of milliseconds.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return parsed


def _parse_positive_integer(raw: Any, name: str) -> int:
    value = str(raw).strip()
    if not value or not value.isascii() or not value.isdecimal():
        raise ValueError(f"{name} must be a positive integer.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


@dataclass(frozen=True)
class DeepgramOptions:
    """Supported per-connection Deepgram query options."""

    language: Optional[str]
    interim_results: bool
    vad_events: bool
    endpointing_ms: Optional[int]
    utterance_end_ms: Optional[int]
    pcm_input: Optional[bool] = None
    context: Optional[str] = None

    @classmethod
    def from_query_params(
        cls,
        params: Mapping[str, str],
        *,
        vac_enabled: bool,
    ) -> "DeepgramOptions":
        interim_results = _parse_bool(params, "interim_results", False)
        vad_events = _parse_bool(params, "vad_events", False)

        endpointing_raw = params.get("endpointing")
        if endpointing_raw is None:
            endpointing_ms: Optional[int] = _DEFAULT_ENDPOINTING_MS
        elif str(endpointing_raw).strip().lower() == "false":
            endpointing_ms = None
        elif str(endpointing_raw).strip().lower() == "true":
            endpointing_ms = _DEFAULT_ENDPOINTING_MS
        else:
            endpointing_ms = _parse_integer_ms(endpointing_raw, "endpointing")

        if endpointing_ms is not None and not vac_enabled:
            raise ValueError(
                "endpointing requires VAC. Enable --vac or use endpointing=false."
            )
        if vad_events and not vac_enabled:
            raise ValueError("vad_events requires VAC. Enable --vac or use vad_events=false.")

        utterance_raw = params.get("utterance_end_ms")
        utterance_end_ms = None
        if utterance_raw is not None:
            utterance_end_ms = _parse_integer_ms(
                utterance_raw,
                "utterance_end_ms",
            )
            if not _MIN_UTTERANCE_END_MS <= utterance_end_ms <= _MAX_UTTERANCE_END_MS:
                raise ValueError(
                    "utterance_end_ms must be an integer between 1000 and 5000."
                )
            if not interim_results:
                raise ValueError(
                    "utterance_end_ms requires interim_results=true."
                )

        encoding_raw = params.get("encoding")
        encoding = str(encoding_raw).strip().lower() if encoding_raw else None
        if encoding not in {None, "linear16"}:
            raise ValueError("encoding must be linear16 when specified.")

        sample_rate_raw = params.get("sample_rate")
        if encoding == "linear16" and sample_rate_raw is None:
            raise ValueError("sample_rate=16000 is required with encoding=linear16.")
        if sample_rate_raw is not None:
            sample_rate = _parse_positive_integer(sample_rate_raw, "sample_rate")
            if sample_rate != 16000:
                raise ValueError("WhisperLiveKit supports sample_rate=16000 only.")

        channels_raw = params.get("channels")
        if channels_raw is not None:
            channels = _parse_positive_integer(channels_raw, "channels")
            if channels != 1:
                raise ValueError("WhisperLiveKit supports channels=1 only.")

        return cls(
            language=params.get("language"),
            interim_results=interim_results,
            vad_events=vad_events,
            endpointing_ms=endpointing_ms,
            utterance_end_ms=utterance_end_ms,
            pcm_input=True if encoding == "linear16" else None,
            context=params.get("context"),
        )


@dataclass(frozen=True)
class _SilenceBoundary:
    word_count: int
    start: float
    end: float


@dataclass
class _PauseState:
    start: float
    end: Optional[float] = None
    transcription_ready: bool = False
    endpointed: bool = False
    resume_timestamp: Optional[float] = None


@dataclass(frozen=True)
class _StreamFailure:
    description: str


def _speaker_index(value: Any) -> int:
    try:
        speaker = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, speaker - 1)


def _span_to_words(
    text: str,
    start: float,
    end: float,
    speaker: int,
) -> list[dict[str, Any]]:
    """Split one timestamped source span into Deepgram word objects."""
    source_text = str(text or "")
    matches = list(re.finditer(r"\S+", source_text))
    if not matches:
        return []
    duration = max(0.0, end - start)
    step = duration / len(matches)
    words = []
    previous_end = 0
    for index, match in enumerate(matches):
        piece = match.group(0)
        word_start = start + index * step
        word_end = end if index == len(matches) - 1 else start + (index + 1) * step
        words.append(
            {
                "word": piece,
                "start": round(word_start, 3),
                "end": round(word_end, 3),
                "confidence": 0.0,
                "punctuated_word": piece,
                "speaker": speaker,
                "_source_text": source_text[previous_end:match.end()],
            }
        )
        previous_end = match.end()
    words[-1]["_source_text"] += source_text[previous_end:]
    return words


def _line_to_words(line: Any) -> list[dict[str, Any]]:
    """Convert one Segment or serialized line without losing real token times."""
    if isinstance(line, dict):
        if line.get("speaker") == -2:
            return []
        return _span_to_words(
            line.get("text", ""),
            _parse_time_str(line.get("start", 0.0)),
            _parse_time_str(line.get("end", 0.0)),
            _speaker_index(line.get("speaker", 1)),
        )

    if line.is_silence():
        return []
    speaker = _speaker_index(getattr(line, "speaker", 1))
    tokens = getattr(line, "tokens", None) or []
    if tokens:
        words = []
        for token in tokens:
            words.extend(
                _span_to_words(
                    token.text,
                    float(token.start or 0.0),
                    float(token.end or token.start or 0.0),
                    speaker,
                )
            )
        return words
    return _span_to_words(
        getattr(line, "text", ""),
        float(getattr(line, "start", 0.0) or 0.0),
        float(getattr(line, "end", 0.0) or 0.0),
        speaker,
    )


def _punctuation_only(text: str) -> bool:
    return bool(text) and all(
        unicodedata.category(character).startswith("P")
        for character in text
    )


def _append_word(
    words: list[dict[str, Any]],
    word: dict[str, Any],
    group_start: int,
) -> None:
    """Attach punctuation-only tokens to the preceding Deepgram word."""
    text = str(word.get("punctuated_word") or word.get("word") or "").strip()
    if _punctuation_only(text) and len(words) > group_start:
        previous = words[-1]
        previous["punctuated_word"] = (
            str(previous.get("punctuated_word") or previous.get("word") or "")
            + text
        )
        previous["end"] = max(float(previous["end"]), float(word["end"]))
        previous["_source_text"] = (
            str(previous.get("_source_text") or "")
            + str(word.get("_source_text") or text)
        )
        return
    if (
        not _punctuation_only(text)
        and len(words) > group_start
        and _punctuation_only(
            str(words[-1].get("word") or "").strip()
        )
    ):
        leading = words.pop()
        word["punctuated_word"] = (
            str(leading.get("punctuated_word") or leading.get("word") or "")
            + str(word.get("punctuated_word") or word.get("word") or "")
        )
        word["_source_text"] = (
            str(leading.get("_source_text") or "")
            + str(word.get("_source_text") or "")
        )
    words.append(word)


def _extract_update(
    update: FrontData | dict[str, Any],
) -> tuple[list[dict[str, Any]], str, list[_SilenceBoundary], float]:
    if isinstance(update, FrontData):
        lines = update.lines
        buffer_parts = (
            update.buffer_diarization,
            update.buffer_transcription,
        )
        remaining_time_diarization = update.remaining_time_diarization
    else:
        lines = update.get("lines", [])
        buffer_parts = (
            update.get("buffer_diarization", ""),
            update.get("buffer_transcription", ""),
        )
        remaining_time_diarization = update.get("remaining_time_diarization", 0.0)
    buffer = _join_source_fragments(buffer_parts)

    words: list[dict[str, Any]] = []
    boundaries: list[_SilenceBoundary] = []
    group_start = 0
    for line in lines:
        if isinstance(line, dict):
            is_silence = line.get("speaker") == -2
            start = _parse_time_str(line.get("start", 0.0))
            end = _parse_time_str(line.get("end", start))
        else:
            is_silence = line.is_silence()
            start = float(line.start or 0.0)
            end = float(line.end or start)
        if is_silence:
            boundaries.append(
                _SilenceBoundary(
                    word_count=len(words),
                    start=start,
                    end=end,
                )
            )
            group_start = len(words)
        else:
            for word in _line_to_words(line):
                _append_word(words, word, group_start)
    try:
        diarization_lag = max(0.0, float(remaining_time_diarization or 0.0))
    except (TypeError, ValueError):
        diarization_lag = 0.0
    return words, str(buffer), boundaries, diarization_lag


def _word_identity(word: dict[str, Any]) -> str:
    """Identity in transcript order; speaker and interim times may still move."""
    return str(word.get("punctuated_word") or word.get("word") or "")


def _is_wide_character(character: str) -> bool:
    return unicodedata.east_asian_width(character) in {"W", "F"}


def _join_source_fragments(fragments: list[str] | tuple[Any, ...]) -> str:
    """Preserve explicit spacing and infer it only between bare Latin words."""
    transcript = ""
    for raw_fragment in fragments:
        fragment = str(raw_fragment or "")
        if not fragment:
            continue
        if (
            transcript
            and not fragment[0].isspace()
            and transcript[-1].isalnum()
            and fragment[0].isalnum()
            and not _is_wide_character(transcript[-1])
            and not _is_wide_character(fragment[0])
        ):
            transcript += " "
        transcript += fragment
    return transcript.strip()


def _compose_transcript(words: list[dict[str, Any]]) -> str:
    """Reconstruct source spacing while separating bare Latin word tokens."""
    return _join_source_fragments(
        [
            str(
                word.get("_source_text")
                or word.get("punctuated_word")
                or word.get("word")
                or ""
            )
            for word in words
        ]
    )


def _common_prefix_length(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> int:
    common = 0
    for old, new in zip(left, right):
        if _word_identity(old) != _word_identity(new):
            break
        common += 1
    return common


class DeepgramAdapter:
    """Translate FrontData snapshots and sample-clock events to Deepgram JSON."""

    def __init__(
        self,
        websocket: WebSocket,
        options: Optional[DeepgramOptions] = None,
        config: Any = None,
    ) -> None:
        self.websocket = websocket
        self.options = options or DeepgramOptions(
            language=None,
            interim_results=False,
            vad_events=False,
            endpointing_ms=_DEFAULT_ENDPOINTING_MS,
            utterance_end_ms=None,
        )
        self.request_id = str(uuid.uuid4())
        self.model_uuid = str(uuid.uuid4())
        self.backend = getattr(config, "backend", "whisper") if config else "whisper"
        self._lock = asyncio.Lock()

        self._observed_words: list[dict[str, Any]] = []
        self._emitted_count = 0
        self._latest_buffer = ""
        self._remaining_time_diarization = 0.0
        self._last_interim_signature: Optional[tuple[Any, ...]] = None
        self._seen_boundaries: set[tuple[float, float]] = set()

        self._audio_time = 0.0
        self._active_silence_start: Optional[float] = None
        self._pauses: dict[float, _PauseState] = {}
        self._speech_started_timestamps: set[float] = set()
        self._in_speech = False
        self._utterance_open = False
        self._last_speech_final_count = 0

        self._last_final_word_end: Optional[float] = None
        self._utterance_end_sent_for: Optional[float] = None

    def _result_metadata(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_info": {
                "name": self.backend,
                "version": "whisperlivekit",
                "arch": self.backend,
            },
            "model_uuid": self.model_uuid,
        }

    def _metadata_message(self, duration: float) -> dict[str, Any]:
        return {
            "type": "Metadata",
            "transaction_key": "deprecated",
            "request_id": self.request_id,
            "sha256": "",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration": round(max(0.0, duration), 3),
            "channels": 1,
            "models": [self.backend],
        }

    async def send_metadata(self, duration: float = 0.0) -> None:
        async with self._lock:
            await self.websocket.send_json(self._metadata_message(duration))

    async def send_summary_metadata(self, duration: float) -> None:
        await self.send_metadata(duration)

    async def send_error(self, description: str) -> None:
        async with self._lock:
            await self.websocket.send_json(
                {
                    "type": "Error",
                    "description": description,
                    "request_id": self.request_id,
                }
            )

    def _result_message(
        self,
        words: list[dict[str, Any]],
        *,
        is_final: bool,
        speech_final: bool,
        transcript: Optional[str] = None,
        from_finalize: bool = False,
    ) -> dict[str, Any]:
        if transcript is None:
            transcript = _compose_transcript(words)
        if words:
            start = float(words[0]["start"])
            end = float(words[-1]["end"])
        else:
            start = self._last_final_word_end or self._audio_time
            end = start
        return {
            "type": "Results",
            "channel_index": [0, 1],
            "duration": round(max(0.0, end - start), 3),
            "start": round(start, 3),
            "is_final": is_final,
            "speech_final": speech_final,
            "from_finalize": from_finalize,
            "channel": {
                "alternatives": [
                    {
                        "transcript": transcript,
                        "confidence": 0.0,
                        "words": [
                            {
                                key: value
                                for key, value in word.items()
                                if not key.startswith("_")
                            }
                            for word in words
                        ],
                    }
                ]
            },
            "metadata": self._result_metadata(),
        }

    async def _send_speech_started(self, timestamp: float) -> None:
        timestamp_key = round(float(timestamp), 6)
        if (
            not self.options.vad_events
            or timestamp_key in self._speech_started_timestamps
        ):
            return
        await self.websocket.send_json(
            {
                "type": "SpeechStarted",
                "channel": [0, 1],
                "timestamp": round(timestamp, 3),
            }
        )
        self._speech_started_timestamps.add(timestamp_key)

    async def _send_utterance_end(self) -> None:
        anchor = self._last_final_word_end
        if anchor is None or self._utterance_end_sent_for == anchor:
            return
        await self.websocket.send_json(
            {
                "type": "UtteranceEnd",
                "channel": [0, 1],
                "last_word_end": round(anchor, 3),
            }
        )
        self._utterance_end_sent_for = anchor

    async def _maybe_send_gap_before(self, next_word_start: float) -> None:
        if self.options.utterance_end_ms is None:
            return
        anchor = self._last_final_word_end
        if anchor is None or self._utterance_end_sent_for == anchor:
            return
        threshold = self.options.utterance_end_ms / 1000.0
        if next_word_start - anchor >= threshold:
            await self._send_utterance_end()

    async def _emit_final_until(
        self,
        target_count: int,
        *,
        speech_final: bool,
        from_finalize: bool = False,
        force: bool = False,
    ) -> bool:
        if self._remaining_time_diarization > 0.0 and not force:
            return False
        if not force:
            pause_limit = self._pending_pause_limit()
            if pause_limit is not None:
                if not speech_final:
                    return False
                target_count = min(target_count, pause_limit)
        target_count = min(max(target_count, self._emitted_count), len(self._observed_words))
        words = [dict(word) for word in self._observed_words[self._emitted_count:target_count]]
        if words:
            await self._maybe_send_gap_before(float(words[0]["start"]))
            await self.websocket.send_json(
                self._result_message(
                    words,
                    is_final=True,
                    speech_final=speech_final,
                    from_finalize=from_finalize,
                )
            )
            self._emitted_count = target_count
            self._last_final_word_end = float(words[-1]["end"])
            self._utterance_end_sent_for = None
            self._utterance_open = not speech_final
            if speech_final:
                self._last_speech_final_count = target_count
            self._last_interim_signature = None
            await self._evaluate_utterance_end()
            return True

        if speech_final and target_count > self._last_speech_final_count:
            await self.websocket.send_json(
                self._result_message(
                    [],
                    is_final=True,
                    speech_final=True,
                    from_finalize=from_finalize,
                )
            )
            self._last_speech_final_count = target_count
            self._utterance_open = False
            self._last_interim_signature = None
            return True
        return False

    async def _send_interim(self) -> None:
        if not self.options.interim_results:
            return
        words = [dict(word) for word in self._observed_words[self._emitted_count:]]
        word_text = _compose_transcript(words)
        transcript = _join_source_fragments(
            [word_text, self._latest_buffer]
        )
        signature = (
            tuple(
                (
                    _word_identity(word),
                    word.get("start"),
                    word.get("end"),
                    word.get("speaker"),
                    word.get("_source_text"),
                )
                for word in words
            ),
            self._latest_buffer.strip(),
        )
        if signature == self._last_interim_signature:
            return
        if not transcript and self._last_interim_signature is None:
            return
        if transcript:
            self._utterance_open = True
        await self.websocket.send_json(
            self._result_message(
                words,
                is_final=False,
                speech_final=False,
                transcript=transcript,
            )
        )
        self._last_interim_signature = signature

    async def _evaluate_utterance_end(self) -> None:
        if self.options.utterance_end_ms is None:
            return
        anchor = self._last_final_word_end
        if anchor is None or self._utterance_end_sent_for == anchor:
            return
        threshold = self.options.utterance_end_ms / 1000.0
        if self._audio_time - anchor < threshold:
            return
        if self._latest_buffer.strip():
            return
        pending = self._observed_words[self._emitted_count:]
        if pending and float(pending[0]["start"]) - anchor < threshold:
            return
        await self._send_utterance_end()

    @staticmethod
    def _pause_key(timestamp: float) -> float:
        return round(float(timestamp), 6)

    async def _evaluate_pause(self, pause: _PauseState) -> None:
        endpointing_ms = self.options.endpointing_ms
        if endpointing_ms is None or pause.endpointed or not pause.transcription_ready:
            return
        pause_end = pause.end if pause.end is not None else self._audio_time
        endpoint_due = (
            pause_end - pause.start + 1e-9 >= endpointing_ms / 1000.0
        )
        if (
            not endpoint_due
            or self._oldest_pending_pause() is not pause
            or self._latest_buffer.strip()
            or self._remaining_time_diarization > 0.0
        ):
            return
        target_count = self._pause_target_count(pause)
        if target_count > self._last_speech_final_count:
            await self._emit_final_until(
                target_count,
                speech_final=True,
            )
        pause.endpointed = True

    def _pause_is_due(self, pause: _PauseState) -> bool:
        if self.options.endpointing_ms is None:
            return False
        pause_end = pause.end if pause.end is not None else self._audio_time
        return (
            pause_end - pause.start + 1e-9
            >= self.options.endpointing_ms / 1000.0
        )

    def _pause_target_count(self, pause: _PauseState) -> int:
        target_count = 0
        for index, word in enumerate(self._observed_words):
            if float(word.get("start", 0.0)) >= pause.start:
                break
            target_count = index + 1
        return target_count

    def _oldest_pending_pause(self) -> Optional[_PauseState]:
        for pause in self._pauses.values():
            if not pause.endpointed and self._pause_is_due(pause):
                return pause
        return None

    def _pending_pause_limit(self) -> Optional[int]:
        pause = self._oldest_pending_pause()
        return self._pause_target_count(pause) if pause is not None else None

    async def _complete_pause_if_ready(self, pause: _PauseState) -> None:
        await self._evaluate_pause(pause)
        if pause.end is None:
            return
        pause_due = self._pause_is_due(pause)
        if pause_due and (not pause.transcription_ready or not pause.endpointed):
            return
        if any(
            earlier is not pause and earlier.start < pause.start
            for earlier in self._pauses.values()
        ):
            return
        if pause.resume_timestamp is not None:
            await self._send_speech_started(pause.resume_timestamp)
        self._pauses.pop(self._pause_key(pause.start), None)

    async def _evaluate_timers(self) -> None:
        for pause in list(self._pauses.values()):
            await self._complete_pause_if_ready(pause)
        await self._evaluate_utterance_end()

    async def process_update(self, update: FrontData | dict[str, Any]) -> None:
        """Reconcile one complete WLK snapshot and emit only protocol deltas."""
        words, buffer, boundaries, diarization_lag = _extract_update(update)
        async with self._lock:
            previous = self._observed_words
            common = _common_prefix_length(previous, words)
            if common < self._emitted_count:
                late_punctuation = []
                punctuation_only_revision = len(words) >= self._emitted_count
                if punctuation_only_revision:
                    for index in range(common, self._emitted_count):
                        old_word = previous[index]
                        new_word = words[index]
                        old_plain = str(old_word.get("word") or "")
                        new_plain = str(new_word.get("word") or "")
                        old_punctuated = str(
                            old_word.get("punctuated_word") or old_plain
                        )
                        new_punctuated = str(
                            new_word.get("punctuated_word") or new_plain
                        )
                        delta = (
                            new_punctuated[len(old_punctuated):]
                            if new_punctuated.startswith(old_punctuated)
                            else ""
                        )
                        if (
                            old_plain != new_plain
                            or not delta
                            or not all(
                                character.isspace()
                                or unicodedata.category(character).startswith("P")
                                for character in delta
                            )
                        ):
                            punctuation_only_revision = False
                            break
                        late_punctuation.append(delta)

                if punctuation_only_revision:
                    for delta in late_punctuation:
                        await self.websocket.send_json(
                            self._result_message(
                                [],
                                is_final=True,
                                speech_final=False,
                                transcript=delta.strip(),
                            )
                        )
                    common = self._emitted_count
                else:
                    logger.error(
                        "Deepgram source revised %d finalized words; keeping the immutable prefix.",
                        self._emitted_count - common,
                    )
                    frozen = previous[:self._emitted_count]
                    words = frozen + words[self._emitted_count:]
                    common = self._emitted_count

            changed = [
                _word_identity(word) for word in previous
            ] != [
                _word_identity(word) for word in words
            ]
            self._observed_words = words
            self._latest_buffer = buffer
            self._remaining_time_diarization = diarization_lag
            if previous and changed and common > self._emitted_count:
                await self._emit_final_until(common, speech_final=False)

            for boundary in boundaries:
                key = (round(boundary.start, 3), round(boundary.end, 3))
                if key in self._seen_boundaries:
                    continue
                if diarization_lag > 0.0:
                    continue
                self._seen_boundaries.add(key)
                duration = max(0.0, boundary.end - boundary.start)
                endpoint = (
                    self.options.endpointing_ms is not None
                    and duration >= self.options.endpointing_ms / 1000.0
                )
                if boundary.word_count < self._emitted_count:
                    logger.debug(
                        "Ignoring a late silence boundary behind the final cursor."
                    )
                    continue
                await self._emit_final_until(
                    boundary.word_count,
                    speech_final=endpoint,
                )

            await self._evaluate_timers()
            await self._send_interim()

    async def process_stream_event(self, event: AudioStreamEvent) -> None:
        """Apply a sample-clock event emitted by AudioProcessor."""
        async with self._lock:
            self._audio_time = max(self._audio_time, event.timestamp)
            if event.kind == "silence_started":
                self._active_silence_start = event.timestamp
                self._pauses[self._pause_key(event.timestamp)] = _PauseState(
                    start=event.timestamp,
                )
                self._in_speech = False
            elif event.kind == "silence_transcription_ready":
                pause = self._pauses.get(self._pause_key(event.timestamp))
                if pause is not None:
                    pause.transcription_ready = True
                    await self._complete_pause_if_ready(pause)
            elif event.kind == "speech_started":
                pause = None
                if self._active_silence_start is not None:
                    pause = self._pauses.get(
                        self._pause_key(self._active_silence_start)
                    )
                self._active_silence_start = None
                if pause is None:
                    if not self._in_speech:
                        await self._send_speech_started(event.timestamp)
                else:
                    pause.end = event.timestamp
                    pause.resume_timestamp = event.timestamp
                    await self._complete_pause_if_ready(pause)
                self._in_speech = True
            elif event.kind != "audio_advanced":
                logger.debug("Unknown audio stream event: %s", event.kind)
            await self._evaluate_timers()

    async def finish_stream(self, duration: float) -> None:
        """Freeze any remaining suffix after AudioProcessor has fully drained."""
        async with self._lock:
            self._audio_time = max(self._audio_time, duration)
            await self._emit_final_until(
                len(self._observed_words),
                speech_final=False,
                force=True,
            )
            if self._latest_buffer.strip():
                await self.websocket.send_json(
                    self._result_message(
                        [],
                        is_final=True,
                        speech_final=False,
                        transcript=self._latest_buffer.strip(),
                    )
                )
                self._latest_buffer = ""
            self._last_interim_signature = None


async def _reject_connection(websocket: WebSocket, description: str) -> None:
    await websocket.accept()
    try:
        await websocket.send_json({"type": "Error", "description": description})
    finally:
        await websocket.close(code=4400, reason="invalid Deepgram session")


async def handle_deepgram_websocket(
    websocket: WebSocket,
    transcription_engine: Any,
    config: Any,
) -> None:
    """Handle one Deepgram-compatible WebSocket session."""
    from whisperlivekit.audio_processor import AudioProcessor

    try:
        options = DeepgramOptions.from_query_params(
            websocket.query_params,
            vac_enabled=bool(getattr(config, "vac", True)),
        )
    except ValueError as exc:
        await _reject_connection(websocket, str(exc))
        logger.warning("Deepgram-compatible WebSocket rejected: %s", exc)
        return

    event_queue: asyncio.Queue[Any] = asyncio.Queue()
    try:
        audio_processor = AudioProcessor(
            transcription_engine=transcription_engine,
            language=options.language,
            mode="full",
            stream_event_queue=event_queue,
            pcm_input=options.pcm_input,
            context=options.context,
        )
    except ValueError as exc:
        await _reject_connection(websocket, str(exc))
        logger.warning("Deepgram-compatible WebSocket rejected: %s", exc)
        return

    adapter = DeepgramAdapter(websocket, options, config)
    try:
        await websocket.accept()
        logger.info("Deepgram-compatible WebSocket opened")
        await adapter.send_metadata()
        results_generator = await audio_processor.create_tasks()
    except Exception:
        logger.exception("Deepgram-compatible WebSocket setup failed")
        with suppress(Exception):
            await adapter.send_error("Internal transcription setup error.")
        with suppress(Exception):
            await websocket.close(code=1011, reason="transcription setup failed")
        with suppress(Exception):
            await audio_processor.cleanup()
        return

    delivered_errors: asyncio.Queue[str] = asyncio.Queue()

    async def consume_results() -> None:
        try:
            async for response in results_generator:
                if (
                    getattr(response, "error", "")
                    or getattr(response, "status", "") == "error"
                ):
                    await event_queue.put(
                        _StreamFailure(
                            getattr(response, "error", "")
                            or "Transcription failed."
                        )
                    )
                    return
                await event_queue.put(response)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Deepgram transcription result stream failed")
            await event_queue.put(
                _StreamFailure("Transcription result stream failed.")
            )

    async def consume_events() -> None:
        try:
            while True:
                event = await event_queue.get()
                try:
                    if event is _EVENTS_DONE:
                        return
                    if isinstance(event, _StreamFailure):
                        await delivered_errors.put(event.description)
                        return
                    if isinstance(event, AudioStreamEvent):
                        await adapter.process_stream_event(event)
                    else:
                        await adapter.process_update(event)
                finally:
                    event_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Deepgram ordered event consumer failed")
            await delivered_errors.put("Transcription event stream failed.")

    results_task = asyncio.create_task(consume_results())
    events_task = asyncio.create_task(consume_events())
    server_closed = False
    client_disconnected = False

    async def drain_and_close() -> None:
        nonlocal server_closed
        await audio_processor.process_audio(b"")
        await results_task
        await event_queue.put(_EVENTS_DONE)
        await events_task
        if not delivered_errors.empty():
            description = delivered_errors.get_nowait()
            delivered_errors.task_done()
            await adapter.send_error(description)
            await websocket.close(code=1011, reason="transcription failed")
            server_closed = True
            return
        duration = audio_processor.total_pcm_samples / audio_processor.sample_rate
        await adapter.finish_stream(duration)
        await adapter.send_summary_metadata(duration)
        await websocket.close(code=1000, reason="CloseStream complete")
        server_closed = True

    try:
        while True:
            receive_task = asyncio.create_task(websocket.receive())
            error_task = asyncio.create_task(delivered_errors.get())
            done, pending = await asyncio.wait(
                (receive_task, error_task),
                timeout=30.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                await adapter.send_error(
                    "No audio or KeepAlive message was received for 30 seconds."
                )
                await websocket.close(code=1011, reason="stream timeout")
                server_closed = True
                break

            if error_task in done:
                description = error_task.result()
                delivered_errors.task_done()
                receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)
                await adapter.send_error(description)
                await websocket.close(code=1011, reason="transcription failed")
                server_closed = True
                break

            error_task.cancel()
            await asyncio.gather(error_task, return_exceptions=True)
            message = receive_task.result()

            message_type = message.get("type")
            if message_type == "websocket.disconnect":
                client_disconnected = True
                break

            data = message.get("bytes")
            if data is not None:
                if not data:
                    await adapter.send_error(
                        "Empty binary frames are not an end-of-stream signal. "
                        "Send a CloseStream control message."
                    )
                    await websocket.close(code=4400, reason="empty binary frame")
                    server_closed = True
                    break
                await audio_processor.process_audio(data)
                continue

            text = message.get("text")
            if text is None:
                client_disconnected = True
                break
            try:
                control = json.loads(text)
            except json.JSONDecodeError:
                await adapter.send_error("Control messages must be valid JSON objects.")
                continue

            message_name = control.get("type", "") if isinstance(control, dict) else ""
            if message_name == "KeepAlive":
                continue
            if message_name == "CloseStream":
                await drain_and_close()
                break
            if message_name == "Finalize":
                await adapter.send_error(
                    "Finalize is not supported by WhisperLiveKit because its ASR "
                    "flush is terminal. Send CloseStream to flush and close."
                )
                await websocket.close(code=4400, reason="Finalize is unsupported")
                server_closed = True
                break
            logger.debug("Unknown Deepgram control message: %s", message_name)

    except WebSocketDisconnect:
        client_disconnected = True
        logger.info("Deepgram-compatible WebSocket disconnected")
    except Exception:
        logger.exception("Deepgram-compatible WebSocket failed")
        if not server_closed and not client_disconnected:
            with suppress(Exception):
                await adapter.send_error("Internal transcription error.")
            with suppress(Exception):
                await websocket.close(code=1011, reason="internal error")
            server_closed = True
    finally:
        for task in (results_task, events_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(results_task, events_task, return_exceptions=True)
        await audio_processor.cleanup()
        logger.info("Deepgram-compatible WebSocket cleaned up")
