from time import time
from typing import Optional

from whisperlivekit.timed_objects import Line, SilentLine, ASRToken, SpeakerSegment, Silence, TimedText, Segment


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
        self.new_translation_buffer = TimedText()
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
        # self.all_translation_segments.extend(self.new_translation) #future
        self.all_translation_segments = self.new_translation if self.new_translation != [] else self.all_translation_segments
        self.new_translation_buffer = self.state.new_translation_buffer if self.new_translation else self.new_translation_buffer
        self.new_translation_buffer = self.new_translation_buffer if type(self.new_translation_buffer) == str else self.new_translation_buffer.text

    def add_translation(self, line : Line):

        for ts in self.all_translation_segments:
            if ts.is_within(line):
                line.translation += ts.text + self.sep
            elif line.translation:
                break


    def compute_punctuations_segments(self, tokens: Optional[list[ASRToken]] = None):
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


    @staticmethod
    def intersection_duration(seg1, seg2):
        start = max(seg1.start, seg2.start)
        end = min(seg1.end, seg2.end)

        return max(0, end - start)

    def get_lines_diarization(self):
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
            diarization=False,
            translation=False,
            current_silence=None
        ):
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
        return lines, diarization_buffer, self.new_translation_buffer
