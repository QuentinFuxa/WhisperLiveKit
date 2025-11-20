from time import time
from typing import Optional

from whisperlivekit.timed_objects import Line, SilentLine, ASRToken, SpeakerSegment, Silence
from whisperlivekit.timed_objects import PunctuationSegment

ALIGNMENT_TIME_TOLERANCE = 0.2  # seconds


class TokensAlignment:

    def __init__(self, state, args, sep):
        self.state = state
        self.diarization = args.diarization
        self._tokens_index = 0
        self._diarization_index = 0
        self._translation_index = 0

        self.all_tokens : list[ASRToken] = []
        self.all_diarization_segments: list[SpeakerSegment] = []
        self.all_translation_segments = []

        self.new_tokens : list[ASRToken] = []
        self.new_diarization: list[SpeakerSegment] = []
        self.new_translation = []
        self.new_tokens_buffer = []
        self.sep = sep if sep is not None else ' '
        self.beg_loop = None

    def update(self):
        self.new_tokens, self.state.new_tokens = self.state.new_tokens, []
        self.new_diarization, self.state.new_diarization = self.state.new_diarization, []
        self.new_translation, self.state.new_translation = self.state.new_translation, []
        self.new_tokens_buffer, self.state.new_tokens_buffer = self.state.new_tokens_buffer, []

        self.all_tokens.extend(self.new_tokens)
        self.all_diarization_segments.extend(self.new_diarization)
        self.all_translation_segments.extend(self.new_translation)

    def get_lines(self, current_silence):
        """
        In the case without diarization
        """
        lines = []
        current_line_tokens = []
        for token in self.all_tokens:
            if type(token) == Silence:
                if current_line_tokens:
                    lines.append(Line().build_from_tokens(current_line_tokens))
                    current_line_tokens = []
                end_silence = token.end if token.has_ended else time() - self.beg_loop
                if lines and lines[-1].is_silent():
                    lines[-1].end = end_silence
                else:
                    lines.append(SilentLine(
                        start = token.start,
                        end = end_silence
                    ))
            else:
                current_line_tokens.append(token)
        if current_line_tokens:
            lines.append(Line().build_from_tokens(current_line_tokens))
        if current_silence:
            end_silence = current_silence.end if current_silence.has_ended else time() - self.beg_loop
            if lines and lines[-1].is_silent():
                lines[-1].end = end_silence
            else:
                lines.append(SilentLine(
                    start = current_silence.start,
                    end = end_silence
                ))

        return lines 


    def _get_asr_tokens(self) -> list[ASRToken]:
        return [token for token in self.all_tokens if isinstance(token, ASRToken)]

    def _tokens_to_text(self, tokens: list[ASRToken]) -> str:
        return ''.join(token.text for token in tokens)

    def _extract_detected_language(self, tokens: list[ASRToken]):
        for token in tokens:
            if getattr(token, 'detected_language', None):
                return token.detected_language
        return None

    def _speaker_display_id(self, raw_speaker) -> int:
        if isinstance(raw_speaker, int):
            speaker_index = raw_speaker
        else:
            digits = ''.join(ch for ch in str(raw_speaker) if ch.isdigit())
            speaker_index = int(digits) if digits else 0
        return speaker_index + 1 if speaker_index >= 0 else 0

    def _line_from_tokens(self, tokens: list[ASRToken], speaker: int) -> Line:
        line = Line().build_from_tokens(tokens)
        line.speaker = speaker
        detected_language = self._extract_detected_language(tokens)
        if detected_language:
            line.detected_language = detected_language
        return line

    def _find_initial_diar_index(self, diar_segments: list[SpeakerSegment], start_time: float) -> int:
        for idx, segment in enumerate(diar_segments):
            if segment.end + ALIGNMENT_TIME_TOLERANCE >= start_time:
                return idx
        return len(diar_segments)

    def _find_speaker_for_token(self, token: ASRToken, diar_segments: list[SpeakerSegment], diar_idx: int):
        if not diar_segments:
            return None, diar_idx
        idx = min(diar_idx, len(diar_segments) - 1)
        midpoint = (token.start + token.end) / 2 if token.end is not None else token.start

        while idx < len(diar_segments) and diar_segments[idx].end + ALIGNMENT_TIME_TOLERANCE < midpoint:
            idx += 1

        candidate_indices = []
        if idx < len(diar_segments):
            candidate_indices.append(idx)
        if idx > 0:
            candidate_indices.append(idx - 1)

        for candidate_idx in candidate_indices:
            segment = diar_segments[candidate_idx]
            seg_start = (segment.start or 0) - ALIGNMENT_TIME_TOLERANCE
            seg_end = (segment.end or 0) + ALIGNMENT_TIME_TOLERANCE
            if seg_start <= midpoint <= seg_end:
                return segment.speaker, candidate_idx

        return None, idx

    def _build_lines_for_tokens(self, tokens: list[ASRToken], diar_segments: list[SpeakerSegment], diar_idx: int):
        if not tokens:
            return [], diar_idx

        segment_lines: list[Line] = []
        current_tokens: list[ASRToken] = []
        current_speaker = None
        pointer = diar_idx

        for token in tokens:
            speaker_raw, pointer = self._find_speaker_for_token(token, diar_segments, pointer)
            if speaker_raw is None:
                return [], diar_idx
            speaker = self._speaker_display_id(speaker_raw)
            if current_speaker is None or current_speaker != speaker:
                if current_tokens:
                    segment_lines.append(self._line_from_tokens(current_tokens, current_speaker))
                current_tokens = [token]
                current_speaker = speaker
            else:
                current_tokens.append(token)

        if current_tokens:
            segment_lines.append(self._line_from_tokens(current_tokens, current_speaker))

        return segment_lines, pointer

    def compute_punctuations_segments(self, tokens: Optional[list[ASRToken]] = None):
        """Compute segments of text between punctuation marks.
        
        Returns a list of PunctuationSegment objects, each representing
        the text from the start (or previous punctuation) to the current punctuation mark.
        """
        
        tokens = tokens if tokens is not None else self._get_asr_tokens()
        if not tokens:
            return []
        punctuation_indices = [
            i for i, token in enumerate[ASRToken](tokens)
            if token.is_punctuation()
        ]
        if not punctuation_indices:
            return []
        
        segments = []
        for i, punct_idx in enumerate(punctuation_indices):
            start_idx = punctuation_indices[i - 1] + 1 if i > 0 else 0
            end_idx = punct_idx            
            if start_idx <= end_idx:
                segment = PunctuationSegment.from_token_range(
                    tokens=tokens,
                    token_index_start=start_idx,
                    token_index_end=end_idx,
                    punctuation_token_index=punct_idx
                )
                segments.append(segment)
        return segments


    def concatenate_diar_segments(self):
        if not self.all_diarization_segments:
            return []
        merged = [self.all_diarization_segments[0]]
        for segment in self.all_diarization_segments[1:]:
            if segment.speaker == merged[-1].speaker:
                merged[-1].end = segment.end
            else:
                merged.append(segment)
        return merged

    def get_lines(self, diarization=False, translation=False):
        """
        Align diarization speaker segments with punctuation-delimited transcription
        segments (see docs/alignement_principles.md).
        """
        tokens = self._get_asr_tokens()
        if not tokens:
            return [], ''

        punctuation_segments = self.compute_punctuations_segments(tokens=tokens)
        diar_segments = self.concatenate_diar_segments()

        if not punctuation_segments or not diar_segments:
            return [], self._tokens_to_text(tokens)

        max_diar_end = diar_segments[-1].end
        if max_diar_end is None:
            return [], self._tokens_to_text(tokens)

        lines: list[Line] = []
        last_consumed_index = -1
        diar_idx = self._find_initial_diar_index(diar_segments, tokens[0].start or 0)

        for segment in punctuation_segments:
            if segment.end is None or segment.end > max_diar_end:
                break
            slice_tokens = tokens[segment.token_index_start:segment.token_index_end + 1]
            segment_lines, diar_idx = self._build_lines_for_tokens(slice_tokens, diar_segments, diar_idx)
            if not segment_lines:
                break
            lines.extend(segment_lines)
            last_consumed_index = segment.token_index_end

        buffer_tokens = tokens[last_consumed_index + 1:] if last_consumed_index + 1 < len(tokens) else []
        buffer_diarization = self._tokens_to_text(buffer_tokens)

        return lines, buffer_diarization