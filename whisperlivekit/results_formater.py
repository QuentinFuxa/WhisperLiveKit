
import logging
import re
from whisperlivekit.remove_silences import handle_silences
from whisperlivekit.timed_objects import Line, format_time, SpeakerSegment
from typing import List

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CHECK_AROUND = 4
DEBUG = False

def next_punctuation_change(i, tokens):
    for ind in range(i+1, min(len(tokens), i+CHECK_AROUND+1)):
        if tokens[ind].is_punctuation():
            return ind        
    return None

def next_speaker_change(i, tokens, speaker):
    for ind in range(i-1, max(0, i-CHECK_AROUND)-1, -1):
        token = tokens[ind]
        if token.is_punctuation():
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

def extract_number(s) -> int:
    """Extract number from speaker string (for diart compatibility)."""
    if isinstance(s, int):
        return s
    m = re.search(r'\d+', str(s))
    return int(m.group()) if m else 0

def concatenate_speakers(segments: List[SpeakerSegment]) -> List[dict]:
    """Concatenate consecutive segments from the same speaker."""
    if not segments:
        return []
    
    # Get speaker number from first segment
    first_speaker = extract_number(segments[0].speaker)
    segments_concatenated = [{"speaker": first_speaker + 1, "begin": segments[0].start, "end": segments[0].end}]
    
    for segment in segments[1:]:
        speaker = extract_number(segment.speaker) + 1
        if segments_concatenated[-1]['speaker'] != speaker:
            segments_concatenated.append({"speaker": speaker, "begin": segment.start, "end": segment.end})
        else:
            segments_concatenated[-1]['end'] = segment.end
    
    return segments_concatenated

def add_speaker_to_tokens_with_punctuation(segments: List[SpeakerSegment], tokens: list) -> list:
    """Assign speakers to tokens with punctuation-aware boundary adjustment."""
    punctuation_marks = {'.', '!', '?'}
    punctuation_tokens = [token for token in tokens if token.text.strip() in punctuation_marks]
    segments_concatenated = concatenate_speakers(segments)
    
    for ind, segment in enumerate(segments_concatenated):
        for i, punctuation_token in enumerate(punctuation_tokens):
            if punctuation_token.start > segment['end']:
                after_length = punctuation_token.start - segment['end']
                before_length = segment['end'] - punctuation_tokens[i - 1].end if i > 0 else float('inf')
                if before_length > after_length:
                    segment['end'] = punctuation_token.start
                    if i < len(punctuation_tokens) - 1 and ind + 1 < len(segments_concatenated):
                        segments_concatenated[ind + 1]['begin'] = punctuation_token.start
                else:
                    segment['end'] = punctuation_tokens[i - 1].end if i > 0 else segment['end']
                    if i < len(punctuation_tokens) - 1 and ind - 1 >= 0:
                        segments_concatenated[ind - 1]['begin'] = punctuation_tokens[i - 1].end
                break
    
    # Ensure non-overlapping tokens
    last_end = 0.0
    for token in tokens:
        start = max(last_end + 0.01, token.start)
        token.start = start
        token.end = max(start, token.end)
        last_end = token.end
    
    # Assign speakers based on adjusted segments
    ind_last_speaker = 0
    for segment in segments_concatenated:
        for i, token in enumerate(tokens[ind_last_speaker:]):
            if token.end <= segment['end']:
                token.speaker = segment['speaker']
                ind_last_speaker = i + 1
            elif token.start > segment['end']:
                break
    
    return tokens

def assign_speakers_to_tokens(tokens: list, segments: List[SpeakerSegment], use_punctuation_split: bool = False) -> list:
    """
    Assign speakers to tokens based on timing overlap with speaker segments.
    
    Args:
        tokens: List of tokens with timing information
        segments: List of speaker segments
        use_punctuation_split: Whether to use punctuation for boundary refinement
        
    Returns:
        List of tokens with speaker assignments
    """
    if not segments or not tokens:
        logger.debug("No segments or tokens available for speaker assignment")
        return tokens
    
    logger.debug(f"Assigning speakers to {len(tokens)} tokens using {len(segments)} segments")
    
    if not use_punctuation_split:
        # Simple overlap-based assignment
        for token in tokens:
            token.speaker = -1  # Default to no speaker
            for segment in segments:
                # Check for timing overlap
                if not (segment.end <= token.start or segment.start >= token.end):
                    speaker_num = extract_number(segment.speaker)
                    token.speaker = speaker_num + 1  # Convert to 1-based indexing
                    break
    else:
        # Use punctuation-aware assignment
        tokens = add_speaker_to_tokens_with_punctuation(segments, tokens)
    
    return tokens

def format_output(state, silence, args, sep):
    diarization = args.diarization
    disable_punctuation_split = args.disable_punctuation_split
    tokens = state.tokens
    translation_validated_segments = state.translation_validated_segments # Here we will attribute the speakers only based on the timestamps of the segments
    last_validated_token = state.last_validated_token
     
    last_speaker = abs(state.last_speaker)
    undiarized_text = []
    tokens = handle_silences(tokens, state.beg_loop, silence)
    
    # Assign speakers to tokens based on segments stored in state
    if diarization and state.speaker_segments:
        use_punctuation_split = args.punctuation_split if hasattr(args, 'punctuation_split') else False
        tokens = assign_speakers_to_tokens(tokens, state.speaker_segments, use_punctuation_split=use_punctuation_split)
    for i in range(last_validated_token, len(tokens)):
        token = tokens[i]
        speaker = int(token.speaker)
        token.corrected_speaker = speaker
        if not diarization:
            if speaker == -1: #Speaker -1 means no attributed by diarization. In the frontend, it should appear under 'Speaker 1'
                token.corrected_speaker = 1
                token.validated_speaker = True
        else:
            if token.speaker == -1:
                undiarized_text.append(token.text)
            elif token.is_punctuation():
                state.last_punctuation_index = i
                token.corrected_speaker = last_speaker
                token.validated_speaker = True
            elif state.last_punctuation_index == i-1:
                if token.speaker != last_speaker:
                    token.corrected_speaker = token.speaker
                    token.validated_speaker = True
                    # perfect, diarization perfectly aligned
                else:
                    speaker_change_pos, new_speaker = next_speaker_change(i, tokens, speaker)
                    if speaker_change_pos:
                        # Corrects delay:
                        # That was the idea. <Okay> haha |SPLIT SPEAKER| that's a good one 
                        # should become:
                        # That was the idea. |SPLIT SPEAKER| <Okay> haha that's a good one 
                        token.corrected_speaker = new_speaker
                        token.validated_speaker = True
            elif speaker != last_speaker:
                if not (speaker == -2 or last_speaker == -2):
                    if next_punctuation_change(i, tokens):
                        # Corrects advance:
                        # Are you |SPLIT SPEAKER| <okay>? yeah, sure. Absolutely 
                        # should become:
                        # Are you <okay>? |SPLIT SPEAKER| yeah, sure. Absolutely 
                        token.corrected_speaker = last_speaker
                        token.validated_speaker = True
                    else: #Problematic, except if the language has no punctuation. We append to previous line, except if disable_punctuation_split is set to True.
                        if not disable_punctuation_split:
                            token.corrected_speaker = last_speaker
                            token.validated_speaker = False
        if token.validated_speaker:
            state.last_validated_token = i
        state.last_speaker = token.corrected_speaker  

    last_speaker = 1
    
    lines = []
    for token in tokens:
        if token.corrected_speaker != -1:
            if int(token.corrected_speaker) != int(last_speaker):
                lines.append(new_line(token))
            else:
                append_token_to_last_line(lines, sep, token)

        last_speaker = token.corrected_speaker        

    if lines:
        unassigned_translated_segments = []
        for ts in translation_validated_segments:
            assigned = False
            for line in lines:
                if ts and ts.overlaps_with(line):
                    if ts.is_within(line):
                        line.translation += ts.text + ' '
                        assigned = True
                        break
                    else:
                        ts0, ts1 = ts.approximate_cut_at(line.end)
                        if ts0 and line.overlaps_with(ts0):
                            line.translation += ts0.text + ' '
                        if ts1:
                            unassigned_translated_segments.append(ts1)
                        assigned = True
                        break
            if not assigned:
                unassigned_translated_segments.append(ts)
        
        if unassigned_translated_segments:
            for line in lines:
                remaining_segments = []
                for ts in unassigned_translated_segments:
                    if ts and ts.overlaps_with(line):
                        line.translation += ts.text + ' '
                    else:
                        remaining_segments.append(ts)
                unassigned_translated_segments = remaining_segments #maybe do smth in the future about that
    
    if state.buffer_transcription and lines:
        lines[-1].end = max(state.buffer_transcription.end, lines[-1].end)
        
    return lines, undiarized_text
