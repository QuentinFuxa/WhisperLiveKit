import math
import unicodedata
from dataclasses import replace
from time import time
from typing import Any, List, Optional, Tuple, Union

from whisperlivekit.timed_objects import (
    ASRToken,
    PuncSegment,
    Segment,
    Silence,
    SilentSegment,
    SpeakerSegment,
    TimedText,
)

_DEFAULT_RETENTION_SECONDS: float = 300.0


def resolve_retention_seconds(requested: Optional[float], mode: str) -> float:
    """History retention policy for a session.

    An explicit --retention-seconds always wins (0 or negative = unlimited).
    Otherwise clients in "full" mode get unlimited history: they receive the
    whole transcript on every update, so server-side pruning permanently
    deletes their earliest lines (issue #372). Diff-mode clients keep their
    own copy, so the bounded default applies.
    """
    if requested is not None:
        return math.inf if float(requested) <= 0 else float(requested)
    if mode == "diff":
        return _DEFAULT_RETENTION_SECONDS
    return math.inf


class TokensAlignment:

    def __init__(
        self,
        state: Any,
        args: Any,
        sep: Optional[str],
        retention_seconds: Optional[float] = None,
    ) -> None:
        self.state = state
        self.diarization = args.diarization

        self.all_tokens: List[ASRToken] = []
        self.all_diarization_segments: List[SpeakerSegment] = []
        self.all_translation_segments: List[Any] = []

        self.new_tokens: List[ASRToken] = []
        self.new_diarization: List[SpeakerSegment] = []
        self.new_translation: List[Any] = []
        self.new_translation_buffer: Union[TimedText, str] = TimedText()
        self.new_tokens_buffer: List[Any] = []
        self.sep: str = sep if sep is not None else ' '
        self.beg_loop: Optional[float] = None

        self.validated_segments: List[Segment] = []
        self.current_line_tokens: List[ASRToken] = []
        self.diarization_buffer: List[ASRToken] = []

        self.last_punctuation = None
        self.last_uncompleted_punc_segment: PuncSegment = None
        self.unvalidated_tokens: PuncSegment = []

        self._retention_seconds: float = (
            retention_seconds
            if retention_seconds is not None
            else _DEFAULT_RETENTION_SECONDS
        )

    def update(self) -> None:
        """Drain state buffers into the running alignment context."""
        self.new_tokens, self.state.new_tokens = self.state.new_tokens, []
        self.new_diarization, self.state.new_diarization = self.state.new_diarization, []
        self.new_translation, self.state.new_translation = self.state.new_translation, []
        self.new_tokens_buffer, self.state.new_tokens_buffer = self.state.new_tokens_buffer, []

        self.all_tokens.extend(self.new_tokens)
        self.all_diarization_segments.extend(self.new_diarization)
        self.all_translation_segments.extend(self.new_translation)
        self.new_translation_buffer = self.state.new_translation_buffer

    def _prune(self) -> None:
        """Drop tokens/segments older than ``_retention_seconds`` from the latest token."""
        if not self.all_tokens or math.isinf(self._retention_seconds):
            return

        latest = self.all_tokens[-1].end
        cutoff = latest - self._retention_seconds
        if cutoff <= 0:
            return

        def _find_cutoff(items: list) -> int:
            """Return the index of the first item whose end >= cutoff."""
            for i, item in enumerate(items):
                if item.end >= cutoff:
                    return i
            return len(items)

        def _prune_items(items: list) -> list:
            idx = _find_cutoff(items)
            return items[idx:] if idx else items

        self.all_tokens = _prune_items(self.all_tokens)
        self.all_diarization_segments = _prune_items(self.all_diarization_segments)
        self.all_translation_segments = _prune_items(self.all_translation_segments)
        self.validated_segments = _prune_items(self.validated_segments)
        self.current_line_tokens = _prune_items(self.current_line_tokens)
        self.unvalidated_tokens = _prune_items(self.unvalidated_tokens)

    def add_translation(self, segment: Segment) -> None:
        """Append translated text segments that overlap with a segment."""
        if segment.translation is None:
            segment.translation = ''
        for ts in self.all_translation_segments:
            if ts.is_within(segment):
                if ts.text:
                    segment.translation += ts.text + self.sep
            elif segment.translation:
                break

    def add_translations(self, segments: List[Segment]) -> None:
        """Attach each validated translation to exactly one speech line.

        A translated span can cover a source phrase that diarization splits
        across multiple speakers. Assigning only fully contained spans loses
        that translation, while assigning it to every overlapping line
        duplicates it. The largest temporal overlap gives one deterministic
        owner. A span outside every line falls back to the nearest speech line,
        so each validated translation has one owner whenever speech exists.
        """
        speech_segments = [segment for segment in segments if not segment.is_silence()]
        for segment in speech_segments:
            segment.translation = ''

        for translated in self.all_translation_segments:
            if not translated.text:
                continue

            best_segment: Optional[Segment] = None
            best_overlap = 0.0
            if translated.start == translated.end:
                best_segment = next(
                    (
                        segment
                        for segment in speech_segments
                        if segment.start <= translated.start < segment.end
                    ),
                    None,
                )
            else:
                for segment in speech_segments:
                    overlap = self.intersection_duration(translated, segment)
                    if overlap > best_overlap:
                        best_segment = segment
                        best_overlap = overlap

            if best_segment is None and speech_segments:
                def temporal_distance(segment: Segment) -> float:
                    if translated.end <= segment.start:
                        return segment.start - translated.end
                    if segment.end <= translated.start:
                        return translated.start - segment.end
                    return 0.0

                best_segment = min(speech_segments, key=temporal_distance)

            if best_segment is not None:
                best_segment.translation += translated.text + self.sep


    def compute_punctuations_segments(self, tokens: Optional[List[ASRToken]] = None) -> List[PuncSegment]:
        """Group tokens into segments split by punctuation and explicit silence."""
        segments = []
        segment_start_idx = 0
        for i, token in enumerate(self.all_tokens):
            if token.is_silence():
                previous_segment = PuncSegment.from_tokens(
                        tokens=self.all_tokens[segment_start_idx: i],
                    )
                if previous_segment:
                    segments.append(previous_segment)
                segment = PuncSegment.from_tokens(
                    tokens=[token],
                    is_silence=True
                )
                segments.append(segment)
                segment_start_idx = i+1
            else:
                if token.has_punctuation():
                    segment = PuncSegment.from_tokens(
                        tokens=self.all_tokens[segment_start_idx: i+1],
                    )
                    segments.append(segment)
                    segment_start_idx = i+1

        final_segment = PuncSegment.from_tokens(
            tokens=self.all_tokens[segment_start_idx:],
        )
        if final_segment:
            segments.append(final_segment)
        return segments

    def compute_new_punctuations_segments(self) -> List[PuncSegment]:
        new_punc_segments = []
        segment_start_idx = 0
        self.unvalidated_tokens += self.new_tokens
        for i, token in enumerate(self.unvalidated_tokens):
            if token.is_silence():
                previous_segment = PuncSegment.from_tokens(
                        tokens=self.unvalidated_tokens[segment_start_idx: i],
                    )
                if previous_segment:
                    new_punc_segments.append(previous_segment)
                segment = PuncSegment.from_tokens(
                    tokens=[token],
                    is_silence=True
                )
                new_punc_segments.append(segment)
                segment_start_idx = i+1
            else:
                if token.has_punctuation():
                    segment = PuncSegment.from_tokens(
                        tokens=self.unvalidated_tokens[segment_start_idx: i+1],
                    )
                    new_punc_segments.append(segment)
                    segment_start_idx = i+1

        self.unvalidated_tokens = self.unvalidated_tokens[segment_start_idx:]
        return new_punc_segments


    def concatenate_diar_segments(self) -> List[SpeakerSegment]:
        """Merge consecutive diarization slices that share the same speaker.

        Works on copies: extending ``merged[-1].end`` in place would mutate
        the stored ``all_diarization_segments`` entries, and since this runs
        on every ``get_lines`` refresh, the stored spans would grow a little
        more corrupt each time.
        """
        if not self.all_diarization_segments:
            return []
        merged = [replace(self.all_diarization_segments[0])]
        for segment in self.all_diarization_segments[1:]:
            if segment.speaker == merged[-1].speaker:
                merged[-1].end = segment.end
            else:
                merged.append(replace(segment))
        return merged


    @staticmethod
    def intersection_duration(seg1: TimedText, seg2: TimedText) -> float:
        """Return the overlap duration between two timed segments."""
        start = max(seg1.start, seg2.start)
        end = min(seg1.end, seg2.end)

        return max(0, end - start)

    def _speaker_for_token(
        self,
        token: ASRToken,
        diarization_segments: List[SpeakerSegment],
        search_start: int,
    ) -> Tuple[int, int]:
        """Resolve one token to a 1-based speaker using the largest overlap.

        ``search_start`` is a forward-only cursor for chronological input. This
        keeps a full refresh linear in the number of tokens and speaker spans
        instead of comparing every token with every diarization span.
        """
        token_start = token.start
        token_end = max(token.start, token.end)
        segment_index = search_start

        while (
            segment_index < len(diarization_segments)
            and diarization_segments[segment_index].end <= token_start
        ):
            segment_index += 1

        max_overlap = 0.0
        speaker: Optional[int] = None
        candidate_index = segment_index

        if token_end > token_start:
            while (
                candidate_index < len(diarization_segments)
                and diarization_segments[candidate_index].start < token_end
            ):
                diarization_segment = diarization_segments[candidate_index]
                overlap = self.intersection_duration(token, diarization_segment)
                if overlap > max_overlap:
                    max_overlap = overlap
                    speaker = diarization_segment.speaker + 1
                candidate_index += 1
        elif segment_index < len(diarization_segments):
            diarization_segment = diarization_segments[segment_index]
            if diarization_segment.start <= token_start < diarization_segment.end:
                speaker = diarization_segment.speaker + 1

        # Preserve the previous behavior for an internal diarization gap: text
        # inside covered stream time stays visible under the default speaker.
        return (speaker if speaker is not None else 1), segment_index

    @staticmethod
    def _segment_from_token_group(tokens: List[ASRToken], speaker: int) -> Optional[Segment]:
        """Build one output line from consecutive tokens for the same speaker."""
        segment = PuncSegment.from_tokens(tokens)
        if segment is not None:
            segment.speaker = speaker
            segment.start = min(token.start for token in tokens)
            segment.end = max(token.end for token in tokens)
        return segment

    @staticmethod
    def _is_punctuation_only(token: ASRToken) -> bool:
        """Return true for a token made only of Unicode punctuation."""
        text = token.text.strip()
        return bool(text) and all(unicodedata.category(char).startswith('P') for char in text)

    @staticmethod
    def _merge_adjacent_segments(segments: List[Segment]) -> List[Segment]:
        """Merge adjacent lines with the same speaker while retaining tokens."""
        if not segments:
            return []

        merged = [segments[0]]
        for segment in segments[1:]:
            if segment.speaker != merged[-1].speaker:
                merged.append(segment)
                continue
            if merged[-1].text:
                merged[-1].text += segment.text
            merged[-1].end = segment.end
            if segment.tokens:
                if merged[-1].tokens:
                    merged[-1].tokens.extend(segment.tokens)
                else:
                    merged[-1].tokens = list(segment.tokens)
        return merged

    def build_token_speaker_segments(
        self,
        diarization_segments: List[SpeakerSegment],
    ) -> Tuple[List[Segment], str]:
        """Split transcript lines at token-level diarization boundaries."""
        if not diarization_segments:
            punctuation_segments = self.compute_punctuations_segments()
            return self._merge_adjacent_segments(punctuation_segments), ''

        segments: List[Segment] = []
        pending_tokens: List[ASRToken] = []
        pending_speaker: Optional[int] = None
        buffer_parts: List[str] = []
        last_diarization_end = max(segment.end for segment in diarization_segments)
        search_start = 0
        previous_token_start = -math.inf
        buffering_suffix = False

        def flush_pending() -> None:
            nonlocal pending_tokens, pending_speaker
            if pending_tokens and pending_speaker is not None:
                segment = self._segment_from_token_group(pending_tokens, pending_speaker)
                if segment is not None:
                    segments.append(segment)
            pending_tokens = []
            pending_speaker = None

        for token in self.all_tokens:
            if token.is_silence():
                # A boundary cannot overtake earlier speech that is still
                # waiting for diarization. Keep the entire remaining suffix
                # volatile until speaker coverage catches up.
                if buffering_suffix:
                    continue
                flush_pending()
                silence_segment = PuncSegment.from_tokens([token], is_silence=True)
                if silence_segment is not None:
                    segments.append(silence_segment)
                continue

            # Standalone punctuation closes the preceding text. Its timestamp
            # often sits exactly on a diarization boundary, where assigning it
            # to the new speaker would create a punctuation-only output line.
            if pending_speaker is not None and self._is_punctuation_only(token):
                pending_tokens.append(token)
                continue

            if token.start < previous_token_start:
                search_start = 0
            previous_token_start = token.start

            # Lines plus buffer must remain a prefix/suffix partition of the
            # original token order. Once diarization falls behind, keep every
            # later text token buffered for this refresh, even if a backend
            # emits a later token with a retrograde timestamp.
            if buffering_suffix or token.start >= last_diarization_end:
                flush_pending()
                buffer_parts.append(token.text)
                buffering_suffix = True
                continue

            speaker, search_start = self._speaker_for_token(
                token,
                diarization_segments,
                search_start,
            )
            if pending_speaker is not None and speaker != pending_speaker:
                flush_pending()
            pending_speaker = speaker
            pending_tokens.append(token)

        flush_pending()
        return segments, ''.join(buffer_parts)

    def get_lines_diarization(self) -> Tuple[List[Segment], str]:
        """Build lines split at speaker turns and track unattributed text."""
        diarization_segments = self.concatenate_diar_segments()
        return self.build_token_speaker_segments(diarization_segments)


    def get_lines(
            self,
            diarization: bool = False,
            translation: bool = False,
            current_silence: Optional[Silence] = None,
            audio_time: Optional[float] = None,
        ) -> Tuple[List[Segment], str, Union[str, TimedText]]:
        """Return the formatted segments plus buffers, optionally with diarization/translation.

        Args:
            audio_time: Current audio stream position in seconds. Used as fallback
                for ongoing silence end time instead of wall-clock (which breaks
                when audio is fed faster or slower than real-time).
        """
        # Fallback for ongoing silence: prefer audio stream time over wall-clock
        _silence_now = audio_time if audio_time is not None else (time() - self.beg_loop)

        if diarization:
            segments, diarization_buffer = self.get_lines_diarization()
        else:
            diarization_buffer = ''
            for token in self.new_tokens:
                if isinstance(token, Silence):
                    if self.current_line_tokens:
                        self.validated_segments.append(Segment.from_tokens(self.current_line_tokens))
                        self.current_line_tokens = []

                    end_silence = token.end if token.has_ended else _silence_now
                    if self.validated_segments and self.validated_segments[-1].is_silence():
                        self.validated_segments[-1].end = end_silence
                    else:
                        self.validated_segments.append(SilentSegment(
                            start=token.start,
                            end=end_silence
                        ))
                else:
                    self.current_line_tokens.append(token)

            segments = list(self.validated_segments)
            if self.current_line_tokens:
                segments.append(Segment.from_tokens(self.current_line_tokens))

        if current_silence:
            end_silence = current_silence.end if current_silence.has_ended else _silence_now
            if segments and segments[-1].is_silence():
                segments[-1] = SilentSegment(start=segments[-1].start, end=end_silence)
            else:
                segments.append(SilentSegment(
                    start=current_silence.start,
                    end=end_silence
                ))
        if translation:
            self.add_translations(segments)

        self._prune()

        return segments, diarization_buffer, self.new_translation_buffer.text
