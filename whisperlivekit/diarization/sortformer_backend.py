import logging
import threading
import time
import wave
from queue import Empty, SimpleQueue
from typing import List, Optional

import numpy as np
import torch

from whisperlivekit.timed_objects import SpeakerSegment

logger = logging.getLogger(__name__)

try:
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
except ImportError:
    raise SystemExit("""Please use `pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"` to use the Sortformer diarization""")


class StreamingSortformerState:
    """
    This class creates a class instance that will be used to store the state of the
    streaming Sortformer model.

    Attributes:
        spkcache (torch.Tensor): Speaker cache to store embeddings from start
        spkcache_lengths (torch.Tensor): Lengths of the speaker cache
        spkcache_preds (torch.Tensor): The speaker predictions for the speaker cache parts
        fifo (torch.Tensor): FIFO queue to save the embedding from the latest chunks
        fifo_lengths (torch.Tensor): Lengths of the FIFO queue
        fifo_preds (torch.Tensor): The speaker predictions for the FIFO queue parts
        spk_perm (torch.Tensor): Speaker permutation information for the speaker cache
        mean_sil_emb (torch.Tensor): Mean silence embedding
        n_sil_frames (torch.Tensor): Number of silence frames
    """

    def __init__(self):
        self.spkcache = None  # Speaker cache to store embeddings from start
        self.spkcache_lengths = None
        self.spkcache_preds = None  # speaker cache predictions
        self.fifo = None  # to save the embedding from the latest chunks
        self.fifo_lengths = None
        self.fifo_preds = None
        self.spk_perm = None
        self.mean_sil_emb = None
        self.n_sil_frames = None


class SortformerDiarization:
    def __init__(self, model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2"):
        """
        Stores the shared streaming Sortformer diarization model. Used when a new online_diarization is initialized.
        """
        self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        """Load and configure the Sortformer model for streaming."""
        try:
            self.diar_model = SortformerEncLabelModel.from_pretrained(model_name)
            self.diar_model.eval()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.diar_model.to(device)
            
            ## to test
            # for name, param in self.diar_model.named_parameters():
            #     if param.device != device:
            #         raise RuntimeError(f"Parameter {name} is on {param.device} but should be on {device}")
            
            logger.info(f"Using {device.type.upper()} for Sortformer model")

            self.diar_model.sortformer_modules.chunk_len = 10
            self.diar_model.sortformer_modules.subsampling_factor = 10
            self.diar_model.sortformer_modules.chunk_right_context = 0
            self.diar_model.sortformer_modules.chunk_left_context = 10
            self.diar_model.sortformer_modules.spkcache_len = 188
            self.diar_model.sortformer_modules.fifo_len = 188
            self.diar_model.sortformer_modules.spkcache_update_period = 144
            self.diar_model.sortformer_modules.log = False
            self.diar_model.sortformer_modules._check_streaming_parameters()
                        
        except Exception as e:
            logger.error(f"Failed to load Sortformer model: {e}")
            raise
 
class SortformerDiarizationOnline:
    def __init__(self, shared_model, sample_rate: int = 16000):
        """
        Initialize the streaming Sortformer diarization system.
        
        Args:
            sample_rate: Audio sample rate (default: 16000)
            model_name: Pre-trained model name (default: "nvidia/diar_streaming_sortformer_4spk-v2")
        """
        self.sample_rate = sample_rate
        self.diarization_segments = []
        self.diar_segments = []
        self.buffer_audio = np.array([], dtype=np.float32)
        self.segment_lock = threading.Lock()
        self.global_time_offset = 0.0
        self.debug = False
                
        self.diar_model = shared_model.diar_model
             
        self.audio2mel = AudioToMelSpectrogramPreprocessor(
            window_size=0.025,
            normalize="NA",
            n_fft=512,
            features=128,
            pad_to=0
        )
        self.audio2mel.to(self.diar_model.device)
        
        self.chunk_duration_seconds = (
            self.diar_model.sortformer_modules.chunk_len * 
            self.diar_model.sortformer_modules.subsampling_factor * 
            self.diar_model.preprocessor._cfg.window_stride
        )
        
        self._init_streaming_state()
        
        self._previous_chunk_features = None
        self._chunk_index = 0
        self._len_prediction = None
        
        # Audio buffer to store PCM chunks for debugging
        self.audio_buffer = []
        
        # Buffer for accumulating audio chunks until reaching chunk_duration_seconds
        self.audio_chunk_buffer = []
        self.accumulated_duration = 0.0
        
        logger.info("SortformerDiarization initialized successfully")


    def _init_streaming_state(self):
        """Initialize the streaming state for the model."""
        batch_size = 1
        device = self.diar_model.device
        
        self.streaming_state = StreamingSortformerState()
        self.streaming_state.spkcache = torch.zeros(
            (batch_size, self.diar_model.sortformer_modules.spkcache_len, self.diar_model.sortformer_modules.fc_d_model), 
            device=device
        )
        self.streaming_state.spkcache_preds = torch.zeros(
            (batch_size, self.diar_model.sortformer_modules.spkcache_len, self.diar_model.sortformer_modules.n_spk), 
            device=device
        )
        self.streaming_state.spkcache_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        self.streaming_state.fifo = torch.zeros(
            (batch_size, self.diar_model.sortformer_modules.fifo_len, self.diar_model.sortformer_modules.fc_d_model), 
            device=device
        )
        self.streaming_state.fifo_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        self.streaming_state.mean_sil_emb = torch.zeros((batch_size, self.diar_model.sortformer_modules.fc_d_model), device=device)
        self.streaming_state.n_sil_frames = torch.zeros((batch_size,), dtype=torch.long, device=device)        
        self.total_preds = torch.zeros((batch_size, 0, self.diar_model.sortformer_modules.n_spk), device=device)

    def insert_silence(self, silence_duration: Optional[float]):
        """
        Insert silence period by adjusting the global time offset.
        
        Args:
            silence_duration: Duration of silence in seconds
        """
        with self.segment_lock:
            self.global_time_offset += silence_duration
        logger.debug(f"Inserted silence of {silence_duration:.2f}s, new offset: {self.global_time_offset:.2f}s")

    def insert_audio_chunk(self, pcm_array: np.ndarray):
        if self.debug:
            self.audio_buffer.append(pcm_array.copy())
        self.buffer_audio = np.concatenate([self.buffer_audio, pcm_array.copy()])
  

    async def diarize(self):
        """
        Process audio data for diarization in streaming fashion.
        
        Args:
            pcm_array: Audio data as numpy array
        """

        threshold = int(self.chunk_duration_seconds * self.sample_rate)
        
        if not len(self.buffer_audio) >= threshold:
            return []
        
        audio = self.buffer_audio[:threshold]
        self.buffer_audio = self.buffer_audio[threshold:]
        
        device = self.diar_model.device
        audio_signal_chunk = torch.tensor(audio, device=device).unsqueeze(0)
        audio_signal_length_chunk = torch.tensor([audio_signal_chunk.shape[1]], device=device)
        
        processed_signal_chunk, processed_signal_length_chunk = self.audio2mel.get_features(
            audio_signal_chunk, audio_signal_length_chunk
        )
        processed_signal_chunk = processed_signal_chunk.to(device)
        processed_signal_length_chunk = processed_signal_length_chunk.to(device)
        
        if self._previous_chunk_features is not None:
            to_add = self._previous_chunk_features[:, :, -99:].to(device)
            total_features = torch.concat([to_add, processed_signal_chunk], dim=2).to(device)
        else:
            total_features = processed_signal_chunk.to(device)
        
        self._previous_chunk_features = processed_signal_chunk.to(device)
        
        chunk_feat_seq_t = torch.transpose(total_features, 1, 2).to(device)
        
        with torch.inference_mode():
            left_offset = 8 if self._chunk_index > 0 else 0
            right_offset = 8
            
            self.streaming_state, self.total_preds = self.diar_model.forward_streaming_step(
                processed_signal=chunk_feat_seq_t,
                processed_signal_length=torch.tensor([chunk_feat_seq_t.shape[1]]).to(device),
                streaming_state=self.streaming_state,
                total_preds=self.total_preds,
                left_offset=left_offset,
                right_offset=right_offset,
            )                
        new_segments = self._process_predictions()
        
        self._chunk_index += 1
        return new_segments

    def _process_predictions(self):
        """Process model predictions and convert to speaker segments."""
        preds_np = self.total_preds[0].cpu().numpy()
        active_speakers = np.argmax(preds_np, axis=1)
        
        if self._len_prediction is None:
            self._len_prediction = len(active_speakers) #12
        
        frame_duration = self.chunk_duration_seconds / self._len_prediction
        current_chunk_preds = active_speakers[-self._len_prediction:]
        
        new_segments = []

        with self.segment_lock:
            base_time = self._chunk_index * self.chunk_duration_seconds + self.global_time_offset
            current_spk = current_chunk_preds[0]
            start_time = round(base_time, 2)
            for idx, spk in enumerate(current_chunk_preds):
                current_time = round(base_time + idx * frame_duration, 2)
                if spk != current_spk:
                    new_segments.append(SpeakerSegment(
                        speaker=current_spk,
                        start=start_time,
                        end=current_time
                    ))
                    start_time = current_time
                    current_spk = spk
            new_segments.append(
                SpeakerSegment(
                        speaker=current_spk,
                        start=start_time,
                        end=current_time
            )
            )
        return new_segments
                
    def get_segments(self) -> List[SpeakerSegment]:
        """Get a copy of the current speaker segments."""
        with self.segment_lock:
            return self.diarization_segments.copy()

    def close(self):
        """Close the diarization system and clean up resources."""
        logger.info("Closing SortformerDiarization")
        with self.segment_lock:
            self.diarization_segments.clear()
        
        if self.debug:
            concatenated_audio = np.concatenate(self.audio_buffer)
            audio_data_int16 = (concatenated_audio * 32767).astype(np.int16)                
            with wave.open("diarization_audio.wav", "wb") as wav_file:
                wav_file.setnchannels(1)  # mono audio
                wav_file.setsampwidth(2)   # 2 bytes per sample (int16)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data_int16.tobytes())
            logger.info(f"Saved {len(concatenated_audio)} samples to diarization_audio.wav")


def extract_number(s: str) -> int:
    """Extract number from speaker string (compatibility function)."""
    import re
    m = re.search(r'\d+', s)
    return int(m.group()) if m else 0


if __name__ == '__main__':
    import asyncio

    import librosa
    
    async def main():
        """TEST ONLY."""
        an4_audio = 'diarization_audio.wav'
        signal, sr = librosa.load(an4_audio, sr=16000)
        signal = signal[:16000*30]

        print("\n" + "=" * 50)
        print("ground truth:")
        print("Speaker 0: 0:00 - 0:09")
        print("Speaker 1: 0:09 - 0:19") 
        print("Speaker 2: 0:19 - 0:25")
        print("Speaker 0: 0:25 - 0:30")
        print("=" * 50)
        
        diarization_backend = SortformerDiarization()
        diarization = SortformerDiarizationOnline(shared_model = diarization_backend)     
        chunk_size = 1600
        
        for i in range(0, len(signal), chunk_size):
            chunk = signal[i:i+chunk_size]
            new_segments = await diarization.diarize(chunk)
            print(f"Processed chunk {i // chunk_size + 1}")
            print(new_segments)
        
        segments = diarization.get_segments()
        print("\nDiarization results:")
        for segment in segments:
            print(f"Speaker {segment.speaker}: {segment.start:.2f}s - {segment.end:.2f}s")
    
    asyncio.run(main())
