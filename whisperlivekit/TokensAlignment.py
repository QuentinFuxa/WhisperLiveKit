from whisperlivekit.timed_objects import Line, SilentLine, format_time, SpeakerSegment, Silence
from whisperlivekit.timed_objects import PunctuationSegment
from time import time


class TokensAlignment:

    def __init__(self, state, args, sep):
        self.state = state
        self.diarization = args.diarization
        self._tokens_index = 0
        self._diarization_index = 0
        self._translation_index = 0

        self.all_tokens = []
        self.all_diarization_segments = []
        self.all_translation_segments = []

        self.new_tokens = []
        self.new_translation = []
        self.new_diarization = []
        self.new_tokens_buffer = []
        self.sep = ' '

    def update(self):
        self.new_tokens, self.state.new_tokens = self.state.new_tokens, []
        self.new_diarization, self.state.new_diarization = self.state.new_diarization, []
        self.new_translation, self.state.new_translation = self.state.new_translation, []
        self.new_tokens_buffer, self.state.new_tokens_buffer = self.state.new_tokens_buffer, []

        self.all_tokens.extend(self.new_tokens)
        self.all_diarization_segments.extend(self.new_diarization)
        self.all_translation_segments.extend(self.new_translation)

    def create_lines_from_tokens(self, current_silence, beg_loop):
        lines = []
        current_line_tokens = []
        for token in self.all_tokens:
            if type(token) == Silence:
                if current_line_tokens:
                    lines.append(Line().build_from_tokens(current_line_tokens))
                    current_line_tokens = []
                end_silence = token.end if token.has_ended else time() - beg_loop
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
            end_silence = current_silence.end if current_silence.has_ended else time() - beg_loop
            if lines and lines[-1].is_silent():
                lines[-1].end = end_silence
            else:
                lines.append(SilentLine(
                    start = current_silence.start,
                    end = end_silence
                ))

        return lines 

    def align_tokens(self):
        if not self.diarization:
            pass
        # return self.all_tokens

    def compute_punctuations_segments(self):
        """Compute segments of text between punctuation marks.
        
        Returns a list of PunctuationSegment objects, each representing
        the text from the start (or previous punctuation) to the current punctuation mark.
        """
        
        if not self.all_tokens:
            return []
        punctuation_indices = [
            i for i, token in enumerate(self.all_tokens)
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
                    tokens=self.all_tokens,
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