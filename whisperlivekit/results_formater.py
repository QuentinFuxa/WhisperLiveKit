
import logging
from whisperlivekit.remove_silences import handle_silences
from whisperlivekit.timed_objects import Line, Segment, format_time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CHECK_AROUND = 4
DEBUG = False


def is_punctuation(token):
    if token.is_punctuation():
        return True
    return False

def next_punctuation_change(i, tokens):
    for ind in range(i+1, min(len(tokens), i+CHECK_AROUND+1)):
        if is_punctuation(tokens[ind]):
            return ind        
    return None

def next_speaker_change(i, tokens, speaker):
    for ind in range(i-1, max(0, i-CHECK_AROUND)-1, -1):
        token = tokens[ind]
        if is_punctuation(token):
            break
        if token.speaker != speaker:
            return ind, token.speaker
    return None, speaker
    
def new_line(
    token,
):
    return Line(
        speaker = token.corrected_speaker,
        text = token.text + (f"[{format_time(token.start)} : {format_time(token.end)}]" if DEBUG else ""),
        start = token.start,
        end = token.end,
        detected_language=token.detected_language
    )

def append_token_to_last_line(lines, sep, token):
    if not lines:
        lines.append(new_line(token))
    else:
        if token.text:
            lines[-1].text += sep + token.text + (f"[{format_time(token.start)} : {format_time(token.end)}]" if DEBUG else "")
            lines[-1].end = token.end
        if not lines[-1].detected_language and token.detected_language:
            lines[-1].detected_language = token.detected_language
            

def format_output(state, silence, args, sep):
    diarization = args.diarization
    disable_punctuation_split = args.disable_punctuation_split
    tokens = state.tokens
    translation_validated_segments = state.translation_validated_segments # Here we will attribute the speakers only based on the timestamps of the segments
    translation_buffer = state.translation_buffer
    last_validated_token = state.last_validated_token
    
    previous_speaker = 1
    undiarized_text = []
    tokens = handle_silences(tokens, state.beg_loop, silence)
    last_punctuation = None
    for i, token in enumerate(tokens[last_validated_token:]):
        speaker = int(token.speaker)
        token.corrected_speaker = speaker
        if not diarization:
            if speaker == -1: #Speaker -1 means no attributed by diarization. In the frontend, it should appear under 'Speaker 1'
                token.corrected_speaker = 1
                token.validated_speaker = True
        else:
            if is_punctuation(token):
                last_punctuation = i  
            
            if last_punctuation == i-1:
                if token.speaker != previous_speaker:
                    token.validated_speaker = True
                    # perfect, diarization perfectly aligned
                    last_punctuation = None
                else:
                    speaker_change_pos, new_speaker = next_speaker_change(i, tokens, speaker)
                    if speaker_change_pos:
                        # Corrects delay:
                        # That was the idea. <Okay> haha |SPLIT SPEAKER| that's a good one 
                        # should become:
                        # That was the idea. |SPLIT SPEAKER| <Okay> haha that's a good one 
                        token.corrected_speaker = new_speaker
                        token.validated_speaker = True
            elif speaker != previous_speaker:
                if not (speaker == -2 or previous_speaker == -2):
                    if next_punctuation_change(i, tokens):
                        # Corrects advance:
                        # Are you |SPLIT SPEAKER| <okay>? yeah, sure. Absolutely 
                        # should become:
                        # Are you <okay>? |SPLIT SPEAKER| yeah, sure. Absolutely 
                        token.corrected_speaker = previous_speaker
                        token.validated_speaker = True
                    else: #Problematic, except if the language has no punctuation. We append to previous line, except if disable_punctuation_split is set to True.
                        if not disable_punctuation_split:
                            token.corrected_speaker = previous_speaker
                            token.validated_speaker = False
        if token.validated_speaker:
            state.last_validated_token = i + last_validated_token
        previous_speaker = token.corrected_speaker  

    for token in tokens[last_validated_token+1:state.last_validated_token+1]:
        if not state.segments or int(token.corrected_speaker) != int(state.segments[-1].speaker):
            state.segments.append(
                Segment(
                    speaker=token.corrected_speaker,
                    words=[token]
                )
            )
        else:
            state.segments[-1].words.append(token)
 
    for token in tokens[state.last_validated_token+1:]:
        # if not state.segments or int(token.corrected_speaker) != int(state.segments[-1].speaker):
        #     state.segments.append(
        #         Segment(
        #             speaker=token.corrected_speaker,
        #             buffer_tokens=[token]
        #         )
        #     )
        # else:
        state.segments[-1].buffer_tokens.append(token) 
    
    for segment in state.segments:
        segment.consolidate(sep)
    # lines = []
    # for token in tokens:
    #     if int(token.corrected_speaker) != int(previous_speaker):
    #         lines.append(new_line(token))
    #     else:
    #         append_token_to_last_line(lines, sep, token)

    #     previous_speaker = token.corrected_speaker

    for ts in translation_validated_segments:
        for segment in state.segments[state.last_validated_segment:]:
            if ts.is_within(segment):
                segment.translation += ts.text + sep
                break

    for ts in translation_buffer:
        for segment in state.segments[state.last_validated_segment:]:
            if ts.is_within(segment):
                segment.buffer.translation += ts.text + sep
                break
    
    # if state.buffer_transcription and lines:
    #     lines[-1].end = max(state.buffer_transcription.end, lines[-1].end)
    
    lines = []
    for segment in state.segments:
        lines.append(Line(
            start=segment.start,
            end=segment.end,
            speaker=segment.speaker,
            text=segment.text,
            translation=segment.translation
        ))
    
    return lines, undiarized_text
