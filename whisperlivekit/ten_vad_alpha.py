"""
ALPHA. results are not great yet
To replace `whisperlivekit.silero_vad_iterator import FixedVADIterator`
by `from whisperlivekit.ten_vad_iterator import TenVADIterator`

Use self.vac = TenVADIterator() instead of self.vac = FixedVADIterator(models.vac_model)
"""

import numpy as np
from ten_vad import TenVad


class TenVADIterator:    
    def __init__(self,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30):
        self.vad = TenVad()
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        
        self.min_silence_samples = int(sampling_rate * min_silence_duration_ms / 1000)
        self.speech_pad_samples = int(sampling_rate * speech_pad_ms / 1000)
        
        self.reset_states()
    
    def reset_states(self):
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        self.buffer = np.array([], dtype=np.float32)
    
    def __call__(self, x, return_seconds=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)
        
        self.buffer = np.append(self.buffer, x)
        
        chunk_size = 256
        ret = None
        
        while len(self.buffer) >= chunk_size:
            chunk = self.buffer[:chunk_size].astype(np.int16)
            self.buffer = self.buffer[chunk_size:]
            
            window_size_samples = len(chunk)
            self.current_sample += window_size_samples
            speech_prob, speech_flag = self.vad.process(chunk)
            if (speech_prob >= self.threshold) and self.temp_end:
                self.temp_end = 0
            
            if (speech_prob >= self.threshold) and not self.triggered:
                self.triggered = True
                speech_start = max(0, self.current_sample - self.speech_pad_samples - window_size_samples)
                result = {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, 1)}
                if ret is None:
                    ret = result
                elif "end" in ret:
                    ret = result
                else:
                    ret.update(result)
            
            if (speech_prob < self.threshold - 0.15) and self.triggered:
                if not self.temp_end:
                    self.temp_end = self.current_sample
                if self.current_sample - self.temp_end < self.min_silence_samples:
                    continue
                else:
                    speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                    self.temp_end = 0
                    self.triggered = False
                    result = {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, 1)}
                    if ret is None:
                        ret = result
                    else:
                        ret.update(result)
        
        return ret if ret != {} else None


def test_on_record_wav():
    import os
    from pathlib import Path
            
    audio_file = Path("record.wav")
    if not audio_file.exists():
        return
   
    import soundfile as sf
    audio_data, sample_rate = sf.read(str(audio_file), dtype='float32')

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    vad = TenVADIterator(
        threshold=0.5,
        sampling_rate=sample_rate,
        min_silence_duration_ms=100,
        speech_pad_ms=30
    )
    
    chunk_size = 1024
    speech_segments = []
    current_segment = None
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        
        if chunk.dtype != np.int16:
            chunk_int16 = (chunk * 32767.0).astype(np.int16)
        else:
            chunk_int16 = chunk
        
        result = vad(chunk_int16, return_seconds=True)
        
        if result is not None:
            if 'start' in result:
                current_segment = {'start': result['start'], 'end': None}
                print(f"Speech start detected at {result['start']:.2f}s")
            elif 'end' in result:
                if current_segment:
                    current_segment['end'] = result['end']
                    duration = current_segment['end'] - current_segment['start']
                    speech_segments.append(current_segment)
                    print(f"Speech end detected at {result['end']:.2f}s (duration: {duration:.2f}s)")
                    current_segment = None
                else:
                    print(f"Speech end detected at {result['end']:.2f}s (no corresponding start)")
    
    if current_segment and current_segment['end'] is None:
        current_segment['end'] = len(audio_data) / sample_rate
        speech_segments.append(current_segment)
        print(f"End of file (last segment at {current_segment['start']:.2f}s)")
    
    print("-" * 60)
    print(f"\nSummary:")
    print(f"Number of speech segments detected: {len(speech_segments)}")
    
    if speech_segments:
        total_speech_time = sum(seg['end'] - seg['start'] for seg in speech_segments)
        total_time = len(audio_data) / sample_rate
        speech_ratio = (total_speech_time / total_time) * 100
        
        print(f"Total speech time: {total_speech_time:.2f}s")
        print(f"Total file time: {total_time:.2f}s")
        print(f"Speech ratio: {speech_ratio:.1f}%")
        print(f"\nDetected segments:")
        for i, seg in enumerate(speech_segments, 1):
            print(f"  {i}. {seg['start']:.2f}s - {seg['end']:.2f}s (duration: {seg['end'] - seg['start']:.2f}s)")
    else:
        print("No speech segments detected")
    
    print("\n" + "=" * 60)
    print("Extracting silence segments...")
    
    silence_segments = []
    total_time = len(audio_data) / sample_rate
    
    if not speech_segments:
        silence_segments = [{'start': 0.0, 'end': total_time}]
    else:
        if speech_segments[0]['start'] > 0:
            silence_segments.append({'start': 0.0, 'end': speech_segments[0]['start']})
        
        for i in range(len(speech_segments) - 1):
            silence_start = speech_segments[i]['end']
            silence_end = speech_segments[i + 1]['start']
            if silence_end > silence_start:
                silence_segments.append({'start': silence_start, 'end': silence_end})
        
        if speech_segments[-1]['end'] < total_time:
            silence_segments.append({'start': speech_segments[-1]['end'], 'end': total_time})
    
    silence_audio = np.array([], dtype=audio_data.dtype)
    
    for seg in silence_segments:
        start_sample = int(seg['start'] * sample_rate)
        end_sample = int(seg['end'] * sample_rate)
        start_sample = max(0, min(start_sample, len(audio_data)))
        end_sample = max(0, min(end_sample, len(audio_data)))
        
        if end_sample > start_sample:
            silence_audio = np.concatenate([silence_audio, audio_data[start_sample:end_sample]])
    
    if len(silence_audio) > 0:
        output_file = "record_silence_only.wav"
        try:
            import soundfile as sf
            sf.write(output_file, silence_audio, sample_rate)
            print(f"Silence file saved: {output_file}")
        except ImportError:
            try:
                from scipy.io import wavfile
                if silence_audio.dtype == np.float32:
                    silence_audio_int16 = (silence_audio * 32767.0).astype(np.int16)
                else:
                    silence_audio_int16 = silence_audio.astype(np.int16)
                wavfile.write(output_file, sample_rate, silence_audio_int16)
                print(f"Silence file saved: {output_file}")
            except ImportError:
                print("Unable to save: soundfile or scipy required")
        
        total_silence_time = sum(seg['end'] - seg['start'] for seg in silence_segments)
        silence_ratio = (total_silence_time / total_time) * 100
        print(f"Total silence duration: {total_silence_time:.2f}s")
        print(f"Silence ratio: {silence_ratio:.1f}%")
        print(f"Number of silence segments: {len(silence_segments)}")
        print(f"\nYou can listen to {output_file} to verify that only silences are present.")
    else:
        print("No silence segments found (file entirely speech)")
    
    print("\n" + "=" * 60)
    print("Extracting speech segments...")
    
    if speech_segments:
        speech_audio = np.array([], dtype=audio_data.dtype)
        
        for seg in speech_segments:
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            start_sample = max(0, min(start_sample, len(audio_data)))
            end_sample = max(0, min(end_sample, len(audio_data)))
            
            if end_sample > start_sample:
                speech_audio = np.concatenate([speech_audio, audio_data[start_sample:end_sample]])
        
        if len(speech_audio) > 0:
            output_file = "record_speech_only.wav"
            try:
                import soundfile as sf
                sf.write(output_file, speech_audio, sample_rate)
                print(f"Speech file saved: {output_file}")
            except ImportError:
                try:
                    from scipy.io import wavfile
                    if speech_audio.dtype == np.float32:
                        speech_audio_int16 = (speech_audio * 32767.0).astype(np.int16)
                    else:
                        speech_audio_int16 = speech_audio.astype(np.int16)
                    wavfile.write(output_file, sample_rate, speech_audio_int16)
                    print(f"Speech file saved: {output_file}")
                except ImportError:
                    print("Unable to save: soundfile or scipy required")
            
            total_speech_time = sum(seg['end'] - seg['start'] for seg in speech_segments)
            speech_ratio = (total_speech_time / total_time) * 100
            print(f"Total speech duration: {total_speech_time:.2f}s")
            print(f"Speech ratio: {speech_ratio:.1f}%")
            print(f"Number of speech segments: {len(speech_segments)}")
            print(f"\nYou can listen to {output_file} to verify that only speech segments are present.")
        else:
            print("No speech audio to extract")
    else:
        print("No speech segments found (file entirely silence)")


if __name__ == "__main__":
    test_on_record_wav()
