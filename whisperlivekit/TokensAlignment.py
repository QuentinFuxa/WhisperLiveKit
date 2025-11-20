from whisperlivekit.timed_objects import Line, format_time, SpeakerSegment, Silence
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
                if lines and lines[-1].speaker == -2:
                    lines[-1].end = end_silence
                else:
                    lines.append(Line(
                        speaker = -2,
                        text = '',
                        start = token.start,
                        end = end_silence
                    ))
            else:
                current_line_tokens.append(token)
        if current_line_tokens:
            lines.append(Line().build_from_tokens(current_line_tokens))
        if current_silence:
            end_silence = current_silence.end if current_silence.has_ended else time() - beg_loop
            if lines and lines[-1].speaker == -2:
                lines[-1].end = end_silence
            else:
                lines.append(Line(
                    speaker = -2,
                    text = '',
                    start = current_silence.start,
                    end = end_silence
                ))

        return lines 

    def align_tokens(self):
        if not self.diarization:
            pass
        # return self.all_tokens

    def compute_punctuations_segments(self):
        punctuations_breaks = []
        new_tokens = self.state.tokens[self.state.last_validated_token:]
        for i in range(len(new_tokens)):
            token = new_tokens[i]
            if token.is_punctuation():
                punctuations_breaks.append({
                    'token_index': i,
                    'token': token,
                    'start': token.start,
                    'end': token.end,
                })
        punctuations_segments = []
        for i, break_info in enumerate(punctuations_breaks):
            start = punctuations_breaks[i - 1]['end'] if i > 0 else 0.0
            end = break_info['end']
            punctuations_segments.append({
                'start': start,
                'end': end,
                'token_index': break_info['token_index'],
                'token': break_info['token']
            })
        return punctuations_segments

    def concatenate_diar_segments(self):
        diarization_segments = self.state.diarization_segments
