from dataclasses import dataclass, field
from typing import Optional, Any, List
from datetime import timedelta

PUNCTUATION_MARKS = {'.', '!', '?', '。', '！', '？'}

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

@dataclass
class Timed:
    start: Optional[float] = 0
    end: Optional[float] = 0

@dataclass
class TimedText(Timed):
    text: Optional[str] = ''
    speaker: Optional[int] = -1
    detected_language: Optional[str] = None
    
    def is_punctuation(self):
        return self.text.strip() in PUNCTUATION_MARKS
    
    def overlaps_with(self, other: 'TimedText') -> bool:
        return not (self.end <= other.start or other.end <= self.start)
    
    def is_within(self, other: 'TimedText') -> bool:
        return other.contains_timespan(self)

    def duration(self) -> float:
        return self.end - self.start

    def contains_time(self, time: float) -> bool:
        return self.start <= time <= self.end

    def contains_timespan(self, other: 'TimedText') -> bool:
        return self.start <= other.start and self.end >= other.end
    
    def __bool__(self):
        return bool(self.text)


@dataclass()
class ASRToken(TimedText):
    
    corrected_speaker: Optional[int] = -1
    validated_speaker: bool = False
    validated_text: bool = False
    validated_language: bool = False
    
    def with_offset(self, offset: float) -> "ASRToken":
        """Return a new token with the time offset added."""
        return ASRToken(self.start + offset, self.end + offset, self.text, self.speaker, detected_language=self.detected_language)

@dataclass
class Sentence(TimedText):
    pass

@dataclass
class Transcript(TimedText):
    """
    represents a concatenation of several ASRToken
    """

    @classmethod
    def from_tokens(
        cls,
        tokens: List[ASRToken],
        sep: Optional[str] = None,
        offset: float = 0
    ) -> "Transcript":
        sep = sep if sep is not None else ' '
        text = sep.join(token.text for token in tokens)
        if tokens:
            start = offset + tokens[0].start
            end = offset + tokens[-1].end
        else:
            start = None
            end = None
        return cls(start, end, text)


@dataclass
class SpeakerSegment(Timed):
    """Represents a segment of audio attributed to a specific speaker.
    No text nor probability is associated with this segment.
    """
    speaker: Optional[int] = -1
    pass

@dataclass
class Translation(TimedText):
    pass

    def approximate_cut_at(self, cut_time):
        """
        Each word in text is considered to be of duration (end-start)/len(words in text)
        """
        if not self.text or not self.contains_time(cut_time):
            return self, None

        words = self.text.split()
        num_words = len(words)
        if num_words == 0:
            return self, None

        duration_per_word = self.duration() / num_words
        
        cut_word_index = int((cut_time - self.start) / duration_per_word)
        
        if cut_word_index >= num_words:
            cut_word_index = num_words -1
        
        text0 = " ".join(words[:cut_word_index])
        text1 = " ".join(words[cut_word_index:])

        segment0 = Translation(start=self.start, end=cut_time, text=text0)
        segment1 = Translation(start=cut_time, end=self.end, text=text1)

        return segment0, segment1
        

@dataclass
class Silence():
    start: Optional[float] = None
    end: Optional[float] = None
    duration: Optional[float] = None
    is_starting: bool = False
    has_ended: bool = False

    def compute_duration(self) -> float:
        if self.start is None or self.end is None:
            return None
        self.duration = self.end - self.start
    
@dataclass
class Line(TimedText):
    translation: str = ''
    
    def to_dict(self):
        _dict = {
            'speaker': int(self.speaker) if self.speaker != -1 else 1,
            'text': self.text,
            'start': format_time(self.start),
            'end': format_time(self.end),
        }
        if self.translation:
            _dict['translation'] = self.translation
        if self.detected_language:
            _dict['detected_language'] = self.detected_language
        return _dict
    
    def build_from_tokens(self, tokens: List[ASRToken]):
        self.text = ''.join([token.text for token in tokens])
        self.start = tokens[0].start
        self.end = tokens[-1].end
        self.speaker = 1
        return self

    def is_silent(self) -> bool:
        return self.speaker == -2

class SilentLine(Line):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speaker = -2
        self.text = ''


@dataclass  
class FrontData():
    status: str = ''
    error: str = ''
    lines: list[Line] = field(default_factory=list)
    buffer_transcription: str = ''
    buffer_diarization: str = ''
    buffer_translation: str = ''
    remaining_time_transcription: float = 0.
    remaining_time_diarization: float = 0.
    
    def to_dict(self):
        _dict = {
            'status': self.status,
            'lines': [line.to_dict() for line in self.lines if (line.text or line.speaker == -2)],
            'buffer_transcription': self.buffer_transcription,
            'buffer_diarization': self.buffer_diarization,
            'buffer_translation': self.buffer_translation,
            'remaining_time_transcription': self.remaining_time_transcription,
            'remaining_time_diarization': self.remaining_time_diarization,
        }
        if self.error:
            _dict['error'] = self.error
        return _dict

@dataclass
class PunctuationSegment():
    """Represents a segment of text between punctuation marks."""
    start: Optional[float]
    end: Optional[float]
    token_index_start: int
    token_index_end: int
    punctuation_token_index: int
    punctuation_token: ASRToken
    
    @classmethod
    def from_token_range(
        cls,
        tokens: List[ASRToken],
        token_index_start: int,
        token_index_end: int,
        punctuation_token_index: int
    ) -> "PunctuationSegment":
        """Create a PunctuationSegment from a range of tokens ending at a punctuation mark."""
        if not tokens or token_index_start < 0 or token_index_end >= len(tokens):
            raise ValueError("Invalid token indices")
        
        start_token = tokens[token_index_start]
        end_token = tokens[token_index_end]
        punctuation_token = tokens[punctuation_token_index]
        
        # Build text from tokens in the segment
        segment_tokens = tokens[token_index_start:token_index_end + 1]
        text = ''.join(token.text for token in segment_tokens)
        
        return cls(
            start=start_token.start,
            end=end_token.end,
            text=text,
            token_index_start=token_index_start,
            token_index_end=token_index_end,
            punctuation_token_index=punctuation_token_index,
            punctuation_token=punctuation_token
        )


@dataclass  
class ChangeSpeaker:
    speaker: int
    start: int

@dataclass  
class State():
    tokens: list = field(default_factory=list)
    last_validated_token: int = 0
    last_speaker: int = 1
    last_punctuation_index: Optional[int] = None
    translation_validated_segments: list = field(default_factory=list)
    buffer_translation: str = field(default_factory=Transcript)
    buffer_transcription: str = field(default_factory=Transcript)
    diarization_segments: list = field(default_factory=list)
    end_buffer: float = 0.0
    end_attributed_speaker: float = 0.0
    remaining_time_transcription: float = 0.0
    remaining_time_diarization: float = 0.0


@dataclass  
class StateLight():
    new_tokens: list = field(default_factory=list)
    new_translation: list = field(default_factory=list)
    new_diarization: list = field(default_factory=list)
    new_tokens_buffer: list = field(default_factory=list) #only when local agreement