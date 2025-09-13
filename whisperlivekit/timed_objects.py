from dataclasses import dataclass, field
from typing import Optional
from datetime import timedelta

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

@dataclass
class ASRToken(TimedText):
    def with_offset(self, offset: float) -> "ASRToken":
        """Return a new token with the time offset added."""
        return ASRToken(self.start + offset, self.end + offset, self.text, self.speaker, self.probability)

@dataclass
class Sentence(TimedText):
    pass

@dataclass
class Transcript(TimedText):
    pass

@dataclass
class SpeakerSegment(TimedText):
    """Represents a segment of audio attributed to a specific speaker.
    No text nor probability is associated with this segment.
    """
    pass

@dataclass
class Translation(TimedText):
    pass

@dataclass
class Silence():
    duration: float
    
    
@dataclass
class Line(TimedText):
    translation: str = ''
    
    def to_dict(self):
        return {
            'speaker': int(self.speaker),
            'text': self.text,
            'translation': self.translation,
            'start': format_time(self.start),
            'end': format_time(self.end),
        }
        
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
            'lines': [line.to_dict() for line in self.lines],
            'buffer_transcription': self.buffer_transcription,
            'buffer_diarization': self.buffer_diarization,
            'remaining_time_transcription': self.remaining_time_transcription,
            'remaining_time_diarization': self.remaining_time_diarization,
        }
        if self.error:
            _dict['error'] = self.error
        return _dict
    
@dataclass  
class State():
    tokens: list
    translated_segments: list
    buffer_transcription: str
    buffer_diarization: str
    end_buffer: float
    end_attributed_speaker: float
    remaining_time_transcription: float
    remaining_time_diarization: float