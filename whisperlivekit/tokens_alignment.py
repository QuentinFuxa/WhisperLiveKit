from time import time
from typing import Any, List, Optional, Tuple, Union

from whisperlivekit.timed_objects import (ASRToken, Segment, SegmentBuffer, PuncSegment, Silence,
                                          SilentSegment, SpeakerSegment,
                                          TimedText)


class TokensAlignment:
    # Minimum duration (seconds) for a silence to be displayed
    MIN_SILENCE_DISPLAY_DURATION = 2.0

    def __init__(self, state: Any, args: Any, sep: Optional[str]) -> None:
        self.state = state
        self.diarization = args.diarization
        self._tokens_index: int = 0
        self._diarization_index: int = 0
        self._translation_index: int = 0

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
        self.tokens_after_last_punctuation: PuncSegment = []
        self.all_validated_segments: List[Segment] = []
        
        # For token-by-token validation with diarization
        self.pending_tokens: List[ASRToken] = []
        self.last_validated_token_end: float = 0.0
        
        # Segment ID counter for the new API
        self._next_segment_id: int = 1

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

    def add_translation(self, segment: Segment) -> None:
        """Append translated text segments that overlap with a segment."""
        for ts in self.all_translation_segments:
            if ts.is_within(segment):
                segment.translation += ts.text + (self.sep if ts.text else '')
            elif segment.translation:
                break


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
        self.tokens_after_last_punctuation += self.new_tokens
        for i, token in enumerate(self.tokens_after_last_punctuation):
            if token.is_silence():
                previous_segment = PuncSegment.from_tokens(
                        tokens=self.tokens_after_last_punctuation[segment_start_idx: i],
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
                        tokens=self.tokens_after_last_punctuation[segment_start_idx: i+1],
                    )
                    new_punc_segments.append(segment)
                    segment_start_idx = i+1

        self.tokens_after_last_punctuation = self.tokens_after_last_punctuation[segment_start_idx:]
        return new_punc_segments


    def concatenate_diar_segments(self) -> List[SpeakerSegment]:
        """Merge consecutive diarization slices that share the same speaker."""
        if not self.all_diarization_segments:
            return []
        merged = [self.all_diarization_segments[0]]
        for segment in self.all_diarization_segments[1:]:
            if segment.speaker == merged[-1].speaker:
                merged[-1].end = segment.end
            else:
                merged.append(segment)
        return merged


    @staticmethod
    def intersection_duration(seg1: TimedText, seg2: TimedText) -> float:
        """Return the overlap duration between two timed segments."""
        start = max(seg1.start, seg2.start)
        end = min(seg1.end, seg2.end)

        return max(0, end - start)

    def _get_speaker_for_token(self, token: ASRToken, diarization_segments: List[SpeakerSegment]) -> Optional[int]:
        """Get speaker ID for a token based on diarization overlap. Returns None if not covered."""
        if not diarization_segments:
            return None
        
        # Check if token is beyond diarization coverage
        if token.start >= diarization_segments[-1].end:
            return None
        
        # Find speaker with max overlap
        max_overlap = 0.0
        best_speaker = None
        for diar_seg in diarization_segments:
            overlap = self.intersection_duration(token, diar_seg)
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = diar_seg.speaker + 1  # 1-indexed
        
        return best_speaker if max_overlap > 0 else None

    def get_lines_diarization(self) -> Tuple[List[Segment], str]:
        """Build segments with token-by-token validation when diarization covers them."""
        diarization_segments = self.concatenate_diar_segments()
        
        # Add new tokens to pending
        self.pending_tokens.extend(self.new_tokens)
        
        # Process pending tokens - validate those covered by diarization
        still_pending = []
        for token in self.pending_tokens:
            if token.is_silence():
                # Handle silence tokens
                silence_duration = (token.end or 0) - (token.start or 0)
                if silence_duration >= self.MIN_SILENCE_DISPLAY_DURATION:
                    # Significant silence - add as separate segment
                    if self.all_validated_segments and not self.all_validated_segments[-1].is_silence():
                        self.all_validated_segments.append(SilentSegment(
                            start=token.start,
                            end=token.end
                        ))
                    elif self.all_validated_segments and self.all_validated_segments[-1].is_silence():
                        # Extend existing silence
                        self.all_validated_segments[-1].end = token.end
                    else:
                        self.all_validated_segments.append(SilentSegment(
                            start=token.start,
                            end=token.end
                        ))
                # Short silences are ignored (don't go to pending either)
                continue
            
            speaker = self._get_speaker_for_token(token, diarization_segments)
            
            if speaker is not None:
                # Token is covered by diarization - validate it
                if self.all_validated_segments:
                    last_seg = self.all_validated_segments[-1]
                    if not last_seg.is_silence() and last_seg.speaker == speaker:
                        # Same speaker - append to existing segment
                        last_seg.text += token.text
                        last_seg.end = token.end
                    else:
                        # Different speaker or after silence - new segment
                        new_seg = Segment(
                            start=token.start,
                            end=token.end,
                            text=token.text,
                            speaker=speaker,
                            start_speaker=token.start,
                            detected_language=token.detected_language
                        )
                        self.all_validated_segments.append(new_seg)
                else:
                    # First segment
                    new_seg = Segment(
                        start=token.start,
                        end=token.end,
                        text=token.text,
                        speaker=speaker,
                        start_speaker=token.start,
                        detected_language=token.detected_language
                    )
                    self.all_validated_segments.append(new_seg)
                
                self.last_validated_token_end = token.end
            else:
                # Token not yet covered by diarization - keep pending
                still_pending.append(token)
        
        self.pending_tokens = still_pending
        
        # Build diarization buffer from pending tokens
        diarization_buffer = ''.join(t.text for t in self.pending_tokens if not t.is_silence())
        
        return self.all_validated_segments, diarization_buffer


    def _assign_segment_ids(self, segments: List[Segment]) -> None:
        """Assign unique IDs to segments that don't have one yet."""
        for segment in segments:
            if segment.id is None:
                segment.id = self._next_segment_id
                self._next_segment_id += 1

    def _assign_buffers_to_last_segment(
        self,
        segments: List[Segment],
        buffer_transcription: str,
        buffer_diarization: str,
        buffer_translation: str
    ) -> None:
        """Assign buffer content to the last non-silent segment."""
        # First, clear ALL buffers (they're ephemeral and shouldn't persist)
        for segment in segments:
            segment.buffer = SegmentBuffer()
        
        # Find the last non-silent segment and assign buffers to it
        for segment in reversed(segments):
            if not segment.is_silence():
                segment.buffer = SegmentBuffer(
                    transcription=buffer_transcription,
                    diarization=buffer_diarization,
                    translation=buffer_translation
                )
                break

    def _filter_and_merge_segments(self, segments: List[Segment]) -> List[Segment]:
        """Filter parasitic silences and merge consecutive same-speaker segments."""
        if not segments:
            return segments
        
        result = []
        for seg in segments:
            if seg.is_silence():
                # Filter short silences
                duration = (seg.end or 0) - (seg.start or 0)
                if duration < self.MIN_SILENCE_DISPLAY_DURATION:
                    continue
                # Merge consecutive silences
                if result and result[-1].is_silence():
                    result[-1].end = seg.end
                    continue
            else:
                # Merge same speaker segments (across filtered silences)
                if result and not result[-1].is_silence() and result[-1].speaker == seg.speaker:
                    result[-1].text += seg.text
                    result[-1].end = seg.end
                    continue
            
            result.append(seg)
        
        return result

    def get_lines(
            self, 
            diarization: bool = False,
            translation: bool = False,
            current_silence: Optional[Silence] = None,
            buffer_transcription: str = ''
        ) -> List[Segment]:
        """Return the formatted segments with per-segment buffers, optionally with diarization/translation."""
        diarization_buffer = ''
        
        if diarization:
            segments, diarization_buffer = self.get_lines_diarization()
        else:
            for token in self.new_tokens:
                if token.is_silence():
                    # Check silence duration before adding
                    silence_duration = (token.end or 0) - (token.start or 0)
                    if silence_duration >= self.MIN_SILENCE_DISPLAY_DURATION:
                        if self.current_line_tokens:
                            self.validated_segments.append(Segment().from_tokens(self.current_line_tokens))
                            self.current_line_tokens = []
                        
                        end_silence = token.end if token.has_ended else time() - self.beg_loop
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
                segments.append(Segment().from_tokens(self.current_line_tokens))

        # Handle current ongoing silence
        if current_silence:
            silence_duration = (current_silence.end or time() - self.beg_loop) - (current_silence.start or 0)
            if silence_duration >= self.MIN_SILENCE_DISPLAY_DURATION:
                end_silence = current_silence.end if current_silence.has_ended else time() - self.beg_loop
                if segments and segments[-1].is_silence():
                    segments[-1] = SilentSegment(start=segments[-1].start, end=end_silence)
                else:
                    segments.append(SilentSegment(
                        start=current_silence.start,
                        end=end_silence
                    ))
        
        if translation:
            [self.add_translation(segment) for segment in segments if not segment.is_silence()]
        
        # Get translation buffer text
        translation_buffer = self.new_translation_buffer.text if self.new_translation_buffer else ''
        
        # Filter parasitic silences and merge same-speaker segments
        segments = self._filter_and_merge_segments(segments)
        
        # Assign unique IDs to all segments
        self._assign_segment_ids(segments)
        
        # Assign buffers to the last active segment
        self._assign_buffers_to_last_segment(
            segments,
            buffer_transcription=buffer_transcription,
            buffer_diarization=diarization_buffer,
            buffer_translation=translation_buffer
        )
        
        return segments
