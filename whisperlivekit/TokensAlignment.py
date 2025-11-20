class TokensAlignment:

    def __init__(self, state_light, silence=None, args=None):
        self.state_light = state_light
        self.silence = silence
        self.args = args

        self._tokens_index = 0
        self._diarization_index = 0
        self._translation_index = 0

    def update(self):
        pass


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

if __name__ == "__main__":
    from whisperlivekit.timed_objects import State, ASRToken, SpeakerSegment, Transcript, Silence
    
    # Reconstruct the state from the backup data
    tokens = [
        ASRToken(start=1.38, end=1.48, text=' The'),
        ASRToken(start=1.42, end=1.52, text=' description'),
        ASRToken(start=1.82, end=1.92, text=' technology'),
        ASRToken(start=2.54, end=2.64, text=' has'),
        ASRToken(start=2.7, end=2.8, text=' improved'),
        ASRToken(start=3.24, end=3.34, text=' so'),
        ASRToken(start=3.66, end=3.76, text=' much'),
        ASRToken(start=4.02, end=4.12, text=' in'),
        ASRToken(start=4.08, end=4.18, text=' the'),
        ASRToken(start=4.26, end=4.36, text=' past'),
        ASRToken(start=4.48, end=4.58, text=' few'),
        ASRToken(start=4.76, end=4.86, text=' years'),
        ASRToken(start=5.76, end=5.86, text='.'),
        ASRToken(start=5.72, end=5.82, text=' Have'),
        ASRToken(start=5.92, end=6.02, text=' you'),
        ASRToken(start=6.08, end=6.18, text=' noticed'),
        ASRToken(start=6.52, end=6.62, text=' how'),
        ASRToken(start=6.8, end=6.9, text=' accurate'),
        ASRToken(start=7.46, end=7.56, text=' real'),
        ASRToken(start=7.72, end=7.82, text='-time'),
        ASRToken(start=8.06, end=8.16, text=' speech'),
        ASRToken(start=8.48, end=8.58, text=' to'),
        ASRToken(start=8.68, end=8.78, text=' text'),
        ASRToken(start=9.0, end=9.1, text=' is'),
        ASRToken(start=9.24, end=9.34, text=' now'),
        ASRToken(start=9.82, end=9.92, text='?'),
        ASRToken(start=9.86, end=9.96, text=' Absolutely'),
        ASRToken(start=11.26, end=11.36, text='.'),
        ASRToken(start=11.36, end=11.46, text=' I'),
        ASRToken(start=11.58, end=11.68, text=' use'),
        ASRToken(start=11.78, end=11.88, text=' it'),
        ASRToken(start=11.94, end=12.04, text=' all'),
        ASRToken(start=12.08, end=12.18, text=' the'),
        ASRToken(start=12.32, end=12.42, text=' time'),
        ASRToken(start=12.58, end=12.68, text=' for'),
        ASRToken(start=12.78, end=12.88, text=' taking'),
        ASRToken(start=13.14, end=13.24, text=' notes'),
        ASRToken(start=13.4, end=13.5, text=' during'),
        ASRToken(start=13.78, end=13.88, text=' meetings'),
        ASRToken(start=14.6, end=14.7, text='.'),
        ASRToken(start=14.82, end=14.92, text=' It'),
        ASRToken(start=14.92, end=15.02, text="'s"),
        ASRToken(start=15.04, end=15.14, text=' amazing'),
        ASRToken(start=15.5, end=15.6, text=' how'),
        ASRToken(start=15.66, end=15.76, text=' it'),
        ASRToken(start=15.8, end=15.9, text=' can'),
        ASRToken(start=15.96, end=16.06, text=' recognize'),
        ASRToken(start=16.58, end=16.68, text=' different'),
        ASRToken(start=16.94, end=17.04, text=' speakers'),
        ASRToken(start=17.82, end=17.92, text=' and'),
        ASRToken(start=18.0, end=18.1, text=' even'),
        ASRToken(start=18.42, end=18.52, text=' add'),
        ASRToken(start=18.74, end=18.84, text=' punct'),
        ASRToken(start=19.02, end=19.12, text='uation'),
        ASRToken(start=19.68, end=19.78, text='.'),
        ASRToken(start=20.04, end=20.14, text=' Yeah'),
        ASRToken(start=20.5, end=20.6, text=','),
        ASRToken(start=20.6, end=20.7, text=' but'),
        ASRToken(start=20.76, end=20.86, text=' sometimes'),
        ASRToken(start=21.42, end=21.52, text=' noise'),
        ASRToken(start=21.82, end=21.92, text=' can'),
        ASRToken(start=22.08, end=22.18, text=' still'),
        ASRToken(start=22.38, end=22.48, text=' cause'),
        ASRToken(start=22.72, end=22.82, text=' mistakes'),
        ASRToken(start=23.74, end=23.84, text='.'),
        ASRToken(start=23.96, end=24.06, text=' Does'),
        ASRToken(start=24.16, end=24.26, text=' this'),
        ASRToken(start=24.4, end=24.5, text=' system'),
        ASRToken(start=24.76, end=24.86, text=' handle'),
        ASRToken(start=25.12, end=25.22, text=' that'),
        ASRToken(start=25.38, end=25.48, text=' well'),
        ASRToken(start=25.68, end=25.78, text='?'),
        ASRToken(start=26.4, end=26.5, text=' It'),
        ASRToken(start=26.5, end=26.6, text=' does'),
        ASRToken(start=26.7, end=26.8, text=' a'),
        ASRToken(start=27.08, end=27.18, text=' pretty'),
        ASRToken(start=27.12, end=27.22, text=' good'),
        ASRToken(start=27.34, end=27.44, text=' job'),
        ASRToken(start=27.64, end=27.74, text=' filtering'),
        ASRToken(start=28.1, end=28.2, text=' noise'),
        ASRToken(start=28.64, end=28.74, text=','),
        ASRToken(start=28.78, end=28.88, text=' especially'),
        ASRToken(start=29.3, end=29.4, text=' with'),
        ASRToken(start=29.51, end=29.61, text=' models'),
        ASRToken(start=29.99, end=30.09, text=' that'),
        ASRToken(start=30.21, end=30.31, text=' use'),
        ASRToken(start=30.51, end=30.61, text=' voice'),
        ASRToken(start=30.83, end=30.93, text=' activity'),
    ]
    
    diarization_segments = [
        SpeakerSegment(start=1.3255040645599365, end=4.3255040645599365, speaker=0),
        SpeakerSegment(start=4.806154012680054, end=9.806154012680054, speaker=0),
        SpeakerSegment(start=9.806154012680054, end=10.806154012680054, speaker=1),
        SpeakerSegment(start=11.168735027313232, end=14.168735027313232, speaker=1),
        SpeakerSegment(start=14.41029405593872, end=17.41029405593872, speaker=1),
        SpeakerSegment(start=17.52983808517456, end=19.52983808517456, speaker=1),
        SpeakerSegment(start=19.64953374862671, end=20.066200415293377, speaker=1),
        SpeakerSegment(start=20.066200415293377, end=22.64953374862671, speaker=2),
        SpeakerSegment(start=23.012792587280273, end=25.012792587280273, speaker=2),
        SpeakerSegment(start=25.495875597000122, end=26.41254226366679, speaker=2),
        SpeakerSegment(start=26.41254226366679, end=30.495875597000122, speaker=0),
    ]
    
    state = State(
        tokens=tokens,
        last_validated_token=72,
        last_speaker=-1,
        last_punctuation_index=71,
        translation_validated_segments=[],
        buffer_translation=Transcript(start=0, end=0, speaker=-1),
        buffer_transcription=Transcript(start=None, end=None, speaker=-1),
        diarization_segments=diarization_segments,
        end_buffer=31.21587559700018,
        end_attributed_speaker=30.495875597000122,
        remaining_time_transcription=0.4,
        remaining_time_diarization=0.7,
        beg_loop=1763627603.968919
    )
    
    alignment = TokensAlignment(state)