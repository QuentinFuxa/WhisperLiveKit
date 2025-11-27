from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

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
    
    def has_punctuation(self) -> bool:
        return any(char in PUNCTUATION_MARKS for char in self.text.strip())
    
    def is_within(self, other: 'TimedText') -> bool:
        return other.contains_timespan(self)

    def duration(self) -> float:
        return self.end - self.start

    def contains_timespan(self, other: 'TimedText') -> bool:
        return self.start <= other.start and self.end >= other.end
    
    def __bool__(self) -> bool:
        return bool(self.text)
    
    def __str__(self) -> str:
        return str(self.text)

@dataclass()
class ASRToken(TimedText):
    
    def with_offset(self, offset: float) -> "ASRToken":
        """Return a new token with the time offset added."""
        return ASRToken(self.start + offset, self.end + offset, self.text, self.speaker, detected_language=self.detected_language)

    def is_silence(self) -> bool:
        return False


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
        """Collapse multiple ASR tokens into a single transcript span."""
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

@dataclass
class Silence():
    start: Optional[float] = None
    end: Optional[float] = None
    duration: Optional[float] = None
    is_starting: bool = False
    has_ended: bool = False

    def compute_duration(self) -> Optional[float]:
        if self.start is None or self.end is None:
            return None
        self.duration = self.end - self.start
        return self.duration
    
    def is_silence(self) -> bool:
        return True


@dataclass
class Segment(TimedText):
    """Generic contiguous span built from tokens or silence markers."""
    start: Optional[float]
    end: Optional[float]
    text: Optional[str]
    speaker: Optional[str]
    tokens: Optional[ASRToken] = None
    translation: Optional[Translation] = None

    @classmethod
    def from_tokens(
        cls,
        tokens: List[Union[ASRToken, Silence]],
        is_silence: bool = False
    ) -> Optional["Segment"]:
        """Return a normalized segment representing the provided tokens."""
        if not tokens:
            return None
        
        start_token = tokens[0]
        end_token = tokens[-1]        
        if is_silence:
            return cls(
                start=start_token.start,
                end=end_token.end,
                text=None,
                speaker=-2
            )
        else:
            return cls(
                start=start_token.start,
                end=end_token.end,
                text=''.join(token.text for token in tokens),
                speaker=-1,
                detected_language=start_token.detected_language
            )

    def is_silence(self) -> bool:
        """True when this segment represents a silence gap."""
        return self.speaker == -2

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the segment for frontend consumption."""
        _dict: Dict[str, Any] = {
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
class PuncSegment(Segment):
    pass

class SilentSegment(Segment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.speaker = -2
        self.text = ''


@dataclass  
class FrontData():
    status: str = ''
    error: str = ''
    lines: list[Segment] = field(default_factory=list)
    buffer_transcription: str = ''
    buffer_diarization: str = ''
    buffer_translation: str = ''
    remaining_time_transcription: float = 0.
    remaining_time_diarization: float = 0.
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the front-end data payload."""
        _dict: Dict[str, Any] = {
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
class ChangeSpeaker:
    speaker: int
    start: int

@dataclass  
class State():
    """Unified state class for audio processing.
    
    Contains both persistent state (tokens, buffers) and temporary update buffers
    (new_* fields) that are consumed by TokensAlignment.
    """
    # Persistent state
    tokens: List[ASRToken] = field(default_factory=list)
    buffer_transcription: Transcript = field(default_factory=Transcript)
    end_buffer: float = 0.0
    end_attributed_speaker: float = 0.0
    remaining_time_transcription: float = 0.0
    remaining_time_diarization: float = 0.0
    
    # Temporary update buffers (consumed by TokensAlignment.update())
    new_tokens: List[Union[ASRToken, Silence]] = field(default_factory=list)
    new_translation: List[Any] = field(default_factory=list)
    new_diarization: List[Any] = field(default_factory=list)
    new_tokens_buffer: List[Any] = field(default_factory=list)  # only when local agreement
    new_translation_buffer= TimedText()