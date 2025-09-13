
import logging
from whisperlivekit.remove_silences import handle_silences
from whisperlivekit.timed_objects import Line, format_time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PUNCTUATION_MARKS = {'.', '!', '?', '。', '！', '？'}
CHECK_AROUND = 4

def is_punctuation(token):
    if token.text.strip() in PUNCTUATION_MARKS:
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
    speaker,
    debug_info = ""
):
    return Line(
        speaker = speaker,
        text = token.text + debug_info,
        start = token.start,
        end = token.end,
    )

def append_token_to_last_line(lines, sep, token, debug_info):
    if token.text:
        lines[-1].text += sep + token.text + debug_info
        lines[-1].end = token.end

def format_output(state, silence, current_time, args, debug, sep):
    diarization = args.diarization
    disable_punctuation_split = args.disable_punctuation_split
    tokens = state.tokens
    translated_segments = state.translated_segments # Here we will attribute the speakers only based on the timestamps of the segments
    buffer_transcription = state.buffer_transcription
    buffer_diarization = state.buffer_diarization
    end_attributed_speaker = state.end_attributed_speaker
    
    previous_speaker = -1
    lines = []
    undiarized_text = []
    tokens, buffer_transcription, buffer_diarization = handle_silences(tokens, buffer_transcription, buffer_diarization, current_time, silence)
    last_punctuation = None
    for i, token in enumerate(tokens):
        speaker = token.speaker
        if not diarization and speaker == -1: #Speaker -1 means no attributed by diarization. In the frontend, it should appear under 'Speaker 1'
            speaker = 1
        if diarization and not tokens[-1].speaker == -2:
            if (speaker in [-1, 0]) and token.end >= end_attributed_speaker:
                undiarized_text.append(token.text)
                continue
            elif (speaker in [-1, 0]) and token.end < end_attributed_speaker:
                speaker = previous_speaker
        debug_info = ""
        if debug:
            debug_info = f"[{format_time(token.start)} : {format_time(token.end)}]"
            
        if not lines:
            lines.append(new_line(token, speaker, debug_info = ""))
            continue
        else:
            previous_speaker = lines[-1].speaker
        
        if is_punctuation(token):
            last_punctuation = i
            
        
        if last_punctuation == i-1:
            if speaker != previous_speaker:
                # perfect, diarization perfectly aligned
                lines.append(new_line(token, speaker, debug_info = ""))
                last_punctuation, next_punctuation = None, None
                continue
            
            speaker_change_pos, new_speaker = next_speaker_change(i, tokens, speaker)
            if speaker_change_pos:
                # Corrects delay:
                # That was the idea. Okay haha |SPLIT SPEAKER| that's a good one 
                # should become:
                # That was the idea. |SPLIT SPEAKER| Okay haha that's a good one 
                lines.append(new_line(token, new_speaker, debug_info = ""))
            else:
                # No speaker change to come
                append_token_to_last_line(lines, sep, token, debug_info)
            continue
        

        if speaker != previous_speaker:
            if speaker == -2 or previous_speaker == -2: #silences can happen anytime
                lines.append(new_line(token, speaker, debug_info = ""))
                continue
            elif next_punctuation_change(i, tokens):
                # Corrects advance:
                # Are you |SPLIT SPEAKER| okay? yeah, sure. Absolutely 
                # should become:
                # Are you okay? |SPLIT SPEAKER| yeah, sure. Absolutely 
                append_token_to_last_line(lines, sep, token, debug_info)
                continue
            else: #we create a new speaker, but that's no ideal. We are not sure about the split. We prefer to append to previous line
                if disable_punctuation_split:
                    lines.append(new_line(token, speaker, debug_info = ""))
                    continue
                pass
            
        append_token_to_last_line(lines, sep, token, debug_info)
    if lines and translated_segments:
        cts_idx = 0 # current_translated_segment_idx
        for line in lines:
            while cts_idx < len(translated_segments):
                ts = translated_segments[cts_idx]
                if ts and ts.start and ts.start >= line.start and ts.end <= line.end:
                    line.translation += ts.text + ' '
                    cts_idx += 1
                else:
                    break
    return lines, undiarized_text, buffer_transcription, '' 

