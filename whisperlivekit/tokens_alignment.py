from time import time
from typing import Optional, List, Tuple, Union, Any

from whisperlivekit.timed_objects import Line, SilentLine, ASRToken, SpeakerSegment, Silence, TimedText, Segment


class TokensAlignment:

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

    def update(self) -> None:
        self.new_tokens, self.state.new_tokens = self.state.new_tokens, []
        self.new_diarization, self.state.new_diarization = self.state.new_diarization, []
        self.new_translation, self.state.new_translation = self.state.new_translation, []
        self.new_tokens_buffer, self.state.new_tokens_buffer = self.state.new_tokens_buffer, []

        self.all_tokens.extend(self.new_tokens)
        self.all_diarization_segments.extend(self.new_diarization)
        self.all_translation_segments.extend(self.new_translation)
        self.new_translation_buffer = self.state.new_translation_buffer

    def add_translation(self, line: Line) -> None:
        for ts in self.all_translation_segments:
            if ts.is_within(line):
                line.translation += ts.text + (self.sep if ts.text else '')
            elif line.translation:
                break


    def compute_punctuations_segments(self, tokens: Optional[List[ASRToken]] = None) -> List[Segment]:
        segments = []
        segment_start_idx = 0
        for i, token in enumerate(self.all_tokens):
            if token.is_silence():
                previous_segment = Segment.from_tokens(
                        tokens=self.all_tokens[segment_start_idx: i],
                    )
                if previous_segment:
                    segments.append(previous_segment)
                segment = Segment.from_tokens(
                    tokens=[token],
                    is_silence=True
                )
                segments.append(segment)
                segment_start_idx = i+1
            else:
                if token.is_punctuation():
                    segment = Segment.from_tokens(
                        tokens=self.all_tokens[segment_start_idx: i+1],
                    )
                    segments.append(segment)
                    segment_start_idx = i+1

        final_segment = Segment.from_tokens(
            tokens=self.all_tokens[segment_start_idx:],
        )
        if final_segment:
            segments.append(final_segment)
        return segments


    def concatenate_diar_segments(self) -> List[SpeakerSegment]:
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
        start = max(seg1.start, seg2.start)
        end = min(seg1.end, seg2.end)

        return max(0, end - start)

    def get_lines_diarization(self) -> Tuple[List[Line], str]:
        """
        use compute_punctuations_segments, concatenate_diar_segments, intersection_duration
        """
        diarization_buffer = ''
        punctuation_segments = self.compute_punctuations_segments()
        diarization_segments = self.concatenate_diar_segments()
        for punctuation_segment in punctuation_segments:
            if not punctuation_segment.is_silence():
                if diarization_segments and punctuation_segment.start >= diarization_segments[-1].end:
                    diarization_buffer += punctuation_segment.text
                else:
                    max_overlap = 0.0
                    max_overlap_speaker = 1
                    for diarization_segment in diarization_segments:
                        intersec = self.intersection_duration(punctuation_segment, diarization_segment)
                        if intersec > max_overlap:
                            max_overlap = intersec
                            max_overlap_speaker = diarization_segment.speaker + 1
                    punctuation_segment.speaker = max_overlap_speaker
        
        lines = []
        if punctuation_segments:
            lines = [Line().build_from_segment(punctuation_segments[0])]
            for segment in punctuation_segments[1:]:
                if segment.speaker == lines[-1].speaker:
                    if lines[-1].text:
                        lines[-1].text += segment.text
                    lines[-1].end = segment.end
                else:
                    lines.append(Line().build_from_segment(segment))

        return lines, diarization_buffer


    def get_lines(
            self, 
            diarization: bool = False,
            translation: bool = False,
            current_silence: Optional[Silence] = None
        ) -> Tuple[List[Line], str, Union[str, TimedText]]:
        """
        In the case without diarization
        """
        if diarization:
            lines, diarization_buffer = self.get_lines_diarization()
        else:
            diarization_buffer = ''
            lines = []
            current_line_tokens = []
            for token in self.all_tokens:
                if token.is_silence():
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
        if translation:
            [self.add_translation(line) for line in lines if not type(line) == Silence]
        return lines, diarization_buffer, self.new_translation_buffer.text
