from dataclasses import dataclass, field
from typing import Optional, Any, List
from datetime import timedelta

PUNCTUATION_MARKS = {'.', '!', '?', '。', '！', '？'}

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


@dataclass
class TimedText:
    start: Optional[float] = 0
    end: Optional[float] = 0
    text: Optional[str] = ''
    speaker: Optional[int] = -1
    probability: Optional[float] = None
    is_dummy: Optional[bool] = False
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
        return ASRToken(self.start + offset, self.end + offset, self.text, self.speaker, self.probability, detected_language=self.detected_language)

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
        probability = sum(token.probability for token in tokens if token.probability) / len(tokens) if tokens else None
        if tokens:
            start = offset + tokens[0].start
            end = offset + tokens[-1].end
        else:
            start = None
            end = None
        return cls(start, end, text, probability=probability)


@dataclass
class SpeakerSegment(TimedText):
    """Represents a segment of audio attributed to a specific speaker.
    No text nor probability is associated with this segment.
    """
    pass

@dataclass
class Translation(TimedText):
    is_validated : bool = False
    pass

    # def split(self):
    #     return self.text.split(" ") # should be customized with the sep

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
        
    def cut_position(self, position):
        sep=" "
        words = self.text.split(sep)
        num_words = len(words)
        duration_per_word = self.duration() / num_words
        cut_time=duration_per_word*position
        
        text0 = sep.join(words[:position])
        text1 = sep.join(words[position:])

        segment0 = Translation(start=self.start, end=cut_time, text=text0)
        segment1 = Translation(start=cut_time, end=self.end, text=text1)
        return segment0, segment1

@dataclass
class Silence():
    duration: float
    
    
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
    
@dataclass
class WordValidation:
    """Validation status for word-level data."""
    text: bool = False
    speaker: bool = False
    language: bool = False
    
    def to_dict(self):
        return {
            'text': self.text,
            'speaker': self.speaker,
            'language': self.language
        }


@dataclass
class Word:
    """Word-level object with timing and validation information."""
    text: str = ''
    start: float = 0.0
    end: float = 0.0
    validated: WordValidation = field(default_factory=WordValidation)
    
    def to_dict(self):
        return {
            'text': self.text,
            'start': self.start,
            'end': self.end,
            'validated': self.validated.to_dict()
        }


@dataclass
class SegmentBuffer:
    """Per-segment temporary buffers for ephemeral data."""
    transcription: str = ''
    diarization: str = ''
    translation: str = ''
    
    def to_dict(self):
        return {
            'transcription': self.transcription,
            'diarization': self.diarization,
            'translation': self.translation
        }


@dataclass
class Segment:
    """Represents a segment in the new API structure."""
    id: int = 0
    speaker: int = -1
    text: str = ''
    start_speaker: float = 0.0
    start: float = 0.0
    end: float = 0.0
    language: Optional[str] = None
    translation: str = ''
    words: List[ASRToken] = field(default_factory=list)
    buffer_tokens: List[ASRToken] = field(default_factory=list)
    buffer_translation = ''
    buffer: SegmentBuffer = field(default_factory=SegmentBuffer)
    
    def to_dict(self):
        """Convert segment to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'speaker': self.speaker,
            'text': self.text,
            'start_speaker': self.start_speaker,
            'start': self.start,
            'end': self.end,
            'language': self.language,
            'translation': self.translation,
            'words': [word.to_dict() for word in self.words],
            'buffer': self.buffer.to_dict()
        }

    def consolidate(self, sep):
        self.text = sep.join([word.text for word in self.words])
        if self.words:
            self.start = self.words[0].start
            self.end = self.words[-1].end
        

@dataclass  
class FrontData():
    status: str = ''
    error: str = ''
    lines: list[Line] = field(default_factory=list)
    buffer_transcription: str = ''
    buffer_diarization: str = ''
    remaining_time_transcription: float = 0.
    remaining_time_diarization: float = 0.
    
    def to_dict(self):
        _dict = {
            'status': self.status,
            'lines': [line.to_dict() for line in self.lines if (line.text or line.speaker == -2)],
            'buffer_transcription': self.buffer_transcription,
            'buffer_diarization': self.buffer_diarization,
            'remaining_time_transcription': self.remaining_time_transcription,
            'remaining_time_diarization': self.remaining_time_diarization,
        }
        if self.error:
            _dict['error'] = self.error
        return _dict

@dataclass  
class ChangeSpeaker:
    speaker: int
    start: int

@dataclass  
class State():
    tokens: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    last_validated_token: int = 0
    last_validated_segment: int = 0 # validated means tokens speaker and transcription are validated and terminated
    translation_validated_segments: list = field(default_factory=list)
    translation_buffer: list = field(default_factory=list)
    buffer_transcription: str = field(default_factory=Transcript)
    end_buffer: float = 0.0
    end_attributed_speaker: float = 0.0
    remaining_time_transcription: float = 0.0
    remaining_time_diarization: float = 0.0
    beg_loop: Optional[int] = None

