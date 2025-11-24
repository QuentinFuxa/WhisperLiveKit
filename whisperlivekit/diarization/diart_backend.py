import asyncio
import logging
import re
import threading
import time
from queue import Empty, SimpleQueue
from typing import Any, List, Tuple

import diart.models as m
import numpy as np
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.sources import AudioSource, MicrophoneAudioSource
from pyannote.core import Annotation
from rx.core import Observer

from whisperlivekit.timed_objects import SpeakerSegment

logger = logging.getLogger(__name__)

def extract_number(s: str) -> int:
    m = re.search(r'\d+', s)
    return int(m.group()) if m else None

class DiarizationObserver(Observer):
    """Observer that logs all data emitted by the diarization pipeline and stores speaker segments."""
    
    def __init__(self):
        self.diarization_segments = []
        self.processed_time = 0
        self.segment_lock = threading.Lock()
        self.global_time_offset = 0.0
    
    def on_next(self, value: Tuple[Annotation, Any]):
        annotation, audio = value
        
        logger.debug("\n--- New Diarization Result ---")
        
        duration = audio.extent.end - audio.extent.start
        logger.debug(f"Audio segment: {audio.extent.start:.2f}s - {audio.extent.end:.2f}s (duration: {duration:.2f}s)")
        logger.debug(f"Audio shape: {audio.data.shape}")
        
        with self.segment_lock:
            if audio.extent.end > self.processed_time:
                self.processed_time = audio.extent.end            
            if annotation and len(annotation._labels) > 0:
                logger.debug("\nSpeaker segments:")
                for speaker, label in annotation._labels.items():
                    for start, end in zip(label.segments_boundaries_[:-1], label.segments_boundaries_[1:]):
                        print(f"  {speaker}: {start:.2f}s-{end:.2f}s")
                        self.diarization_segments.append(SpeakerSegment(
                            speaker=speaker,
                            start=start + self.global_time_offset,
                            end=end + self.global_time_offset
                        ))
            else:
                logger.debug("\nNo speakers detected in this segment")
                
    def get_segments(self) -> List[SpeakerSegment]:
        """Get a copy of the current speaker segments."""
        with self.segment_lock:
            return self.diarization_segments.copy()
    
    def clear_old_segments(self, older_than: float = 30.0):
        """Clear segments older than the specified time."""
        with self.segment_lock:
            current_time = self.processed_time
            self.diarization_segments = [
                segment for segment in self.diarization_segments 
                if current_time - segment.end < older_than
            ]
    
    def on_error(self, error):
        """Handle an error in the stream."""
        logger.debug(f"Error in diarization stream: {error}")
        
    def on_completed(self):
        """Handle the completion of the stream."""
        logger.debug("Diarization stream completed")


class WebSocketAudioSource(AudioSource):
    """
    Buffers incoming audio and releases it in fixed-size chunks at regular intervals.
    """
    def __init__(self, uri: str = "websocket", sample_rate: int = 16000, block_duration: float = 0.5):
        super().__init__(uri, sample_rate)
        self.block_duration = block_duration
        self.block_size = int(np.rint(block_duration * sample_rate))
        self._queue = SimpleQueue()
        self._buffer = np.array([], dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._closed = False
        self._close_event = threading.Event()
        self._processing_thread = None
        self._last_chunk_time = time.time()

    def read(self):
        """Start processing buffered audio and emit fixed-size chunks."""
        self._processing_thread = threading.Thread(target=self._process_chunks)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        
        self._close_event.wait()
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)

    def _process_chunks(self):
        """Process audio from queue and emit fixed-size chunks at regular intervals."""
        while not self._closed:
            try:
                audio_chunk = self._queue.get(timeout=0.1)
                
                with self._buffer_lock:
                    self._buffer = np.concatenate([self._buffer, audio_chunk])
                    
                    while len(self._buffer) >= self.block_size:
                        chunk = self._buffer[:self.block_size]
                        self._buffer = self._buffer[self.block_size:]
                        
                        current_time = time.time()
                        time_since_last = current_time - self._last_chunk_time
                        if time_since_last < self.block_duration:
                            time.sleep(self.block_duration - time_since_last)
                        
                        chunk_reshaped = chunk.reshape(1, -1)
                        self.stream.on_next(chunk_reshaped)
                        self._last_chunk_time = time.time()
                        
            except Empty:
                with self._buffer_lock:
                    if len(self._buffer) > 0 and time.time() - self._last_chunk_time > self.block_duration:
                        padded_chunk = np.zeros(self.block_size, dtype=np.float32)
                        padded_chunk[:len(self._buffer)] = self._buffer
                        self._buffer = np.array([], dtype=np.float32)
                        
                        chunk_reshaped = padded_chunk.reshape(1, -1)
                        self.stream.on_next(chunk_reshaped)
                        self._last_chunk_time = time.time()
            except Exception as e:
                logger.error(f"Error in audio processing thread: {e}")
                self.stream.on_error(e)
                break
        
        with self._buffer_lock:
            if len(self._buffer) > 0:
                padded_chunk = np.zeros(self.block_size, dtype=np.float32)
                padded_chunk[:len(self._buffer)] = self._buffer
                chunk_reshaped = padded_chunk.reshape(1, -1)
                self.stream.on_next(chunk_reshaped)
        
        self.stream.on_completed()

    def close(self):
        if not self._closed:
            self._closed = True
            self._close_event.set()

    def push_audio(self, chunk: np.ndarray):
        """Add audio chunk to the processing queue."""
        if not self._closed:
            if chunk.ndim > 1:
                chunk = chunk.flatten()
            self._queue.put(chunk)
            logger.debug(f'Added chunk to queue with {len(chunk)} samples')


class DiartDiarization:
    def __init__(self, sample_rate: int = 16000, config : SpeakerDiarizationConfig = None, use_microphone: bool = False, block_duration: float = 1.5, segmentation_model_name: str = "pyannote/segmentation-3.0", embedding_model_name: str = "pyannote/embedding"):
        segmentation_model = m.SegmentationModel.from_pretrained(segmentation_model_name)
        embedding_model = m.EmbeddingModel.from_pretrained(embedding_model_name)
        
        if config is None:
            config = SpeakerDiarizationConfig(
                segmentation=segmentation_model,
                embedding=embedding_model,
            )
        
        self.pipeline = SpeakerDiarization(config=config)        
        self.observer = DiarizationObserver()
        
        if use_microphone:
            self.source = MicrophoneAudioSource(block_duration=block_duration)
            self.custom_source = None
        else:
            self.custom_source = WebSocketAudioSource(
                uri="websocket_source", 
                sample_rate=sample_rate,
                block_duration=block_duration
            )
            self.source = self.custom_source
            
        self.inference = StreamingInference(
            pipeline=self.pipeline,
            source=self.source,
            do_plot=False,
            show_progress=False,
        )
        self.inference.attach_observers(self.observer)
        asyncio.get_event_loop().run_in_executor(None, self.inference)

    def insert_silence(self, silence_duration):
        self.observer.global_time_offset += silence_duration

    async def diarize(self, pcm_array: np.ndarray):
        """
        Process audio data for diarization.
        Only used when working with WebSocketAudioSource.
        """
        if self.custom_source:
            self.custom_source.push_audio(pcm_array)            
        # self.observer.clear_old_segments()        

    def close(self):
        """Close the audio source."""
        if self.custom_source:
            self.custom_source.close()

        
def concatenate_speakers(segments):
    segments_concatenated = [{"speaker": 1, "begin": 0.0, "end": 0.0}]
    for segment in segments:
        speaker = extract_number(segment.speaker) + 1
        if segments_concatenated[-1]['speaker'] != speaker:
            segments_concatenated.append({"speaker": speaker, "begin": segment.start, "end": segment.end})
        else:
            segments_concatenated[-1]['end'] = segment.end
    # print("Segments concatenated:")
    # for entry in segments_concatenated:
    #     print(f"Speaker {entry['speaker']}: {entry['begin']:.2f}s - {entry['end']:.2f}s")   
    return segments_concatenated


def add_speaker_to_tokens(segments, tokens):
    """
    Assign speakers to tokens based on diarization segments, with punctuation-aware boundary adjustment.
    """
    punctuation_marks = {'.', '!', '?'}
    punctuation_tokens = [token for token in tokens if token.text.strip() in punctuation_marks]
    segments_concatenated = concatenate_speakers(segments)
    for ind, segment in enumerate(segments_concatenated):
            for i, punctuation_token in enumerate(punctuation_tokens):
                if punctuation_token.start > segment['end']:
                    after_length = punctuation_token.start - segment['end']
                    before_length = segment['end'] - punctuation_tokens[i - 1].end
                    if before_length > after_length:
                        segment['end'] = punctuation_token.start
                        if i < len(punctuation_tokens) - 1 and ind + 1 < len(segments_concatenated):
                            segments_concatenated[ind + 1]['begin'] = punctuation_token.start
                    else:
                        segment['end'] = punctuation_tokens[i - 1].end
                        if i < len(punctuation_tokens) - 1 and ind - 1 >= 0:
                            segments_concatenated[ind - 1]['begin'] = punctuation_tokens[i - 1].end
                    break

    last_end = 0.0
    for token in tokens:
        start = max(last_end + 0.01, token.start)
        token.start = start
        token.end = max(start, token.end)
        last_end = token.end

    ind_last_speaker = 0
    for segment in segments_concatenated:
        for i, token in enumerate(tokens[ind_last_speaker:]):
            if token.end <= segment['end']:
                token.speaker = segment['speaker']
                ind_last_speaker = i + 1
                # print(
                #     f"Token '{token.text}' ('begin': {token.start:.2f}, 'end': {token.end:.2f}) "
                #     f"assigned to Speaker {segment['speaker']} ('segment': {segment['begin']:.2f}-{segment['end']:.2f})"
                # )
            elif token.start > segment['end']:
                break
    return tokens


def visualize_tokens(tokens):
    conversation = [{"speaker": -1, "text": ""}]
    for token in tokens:
        speaker = conversation[-1]['speaker']
        if token.speaker != speaker:
            conversation.append({"speaker": token.speaker, "text": token.text})
        else:
            conversation[-1]['text'] += token.text
    print("Conversation:")
    for entry in conversation:
        print(f"Speaker {entry['speaker']}: {entry['text']}")