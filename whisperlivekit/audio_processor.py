import asyncio
import numpy as np
from time import time, sleep
import math
import logging
import traceback
from whisperlivekit.timed_objects import ASRToken, Silence, Line, FrontData, State, Transcript, ChangeSpeaker
from whisperlivekit.core import TranscriptionEngine, online_factory, online_diarization_factory, online_translation_factory
from whisperlivekit.silero_vad_iterator import VADIterator
from whisperlivekit.results_formater import format_output
from whisperlivekit.ffmpeg_manager import FFmpegManager, FFmpegState

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SENTINEL = object() # unique sentinel object for end of stream marker

def cut_at(cumulative_pcm, cut_sec):
    cumulative_len = 0
    cut_sample = int(cut_sec * 16000)
    
    for ind, pcm_array in enumerate(cumulative_pcm):
        if (cumulative_len + len(pcm_array)) >= cut_sample:
            cut_chunk = cut_sample - cumulative_len
            before = np.concatenate(cumulative_pcm[:ind] + [cumulative_pcm[ind][:cut_chunk]])
            after = [cumulative_pcm[ind][cut_chunk:]] + cumulative_pcm[ind+1:]
            return before, after
        cumulative_len += len(pcm_array)
    return np.concatenate(cumulative_pcm), []

async def get_all_from_queue(queue):
    items = []
    try:
        while True:
            item = queue.get_nowait()
            items.append(item)
    except asyncio.QueueEmpty:
        pass
    return items

class AudioProcessor:
    """
    Processes audio streams for transcription and diarization.
    Handles audio processing, state management, and result formatting.
    """
    
    def __init__(self, **kwargs):
        """Initialize the audio processor with configuration, models, and state."""
        
        if 'transcription_engine' in kwargs and isinstance(kwargs['transcription_engine'], TranscriptionEngine):
            models = kwargs['transcription_engine']
        else:
            models = TranscriptionEngine(**kwargs)
        
        # Audio processing settings
        self.args = models.args
        self.sample_rate = 16000
        self.channels = 1
        self.samples_per_sec = int(self.sample_rate * self.args.min_chunk_size)
        self.bytes_per_sample = 2
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz
        self.is_pcm_input = self.args.pcm_input

        # State management
        self.is_stopping = False
        self.silence_duration = 0.0
        self.tokens = []
        self.last_validated_token = 0
        self.translated_segments = []
        self.buffer_transcription = Transcript()
        self.end_buffer = 0
        self.end_attributed_speaker = 0
        self.lock = asyncio.Lock()
        self.beg_loop = 0.0 #to deal with a potential little lag at the websocket initialization, this is now set in process_audio
        self.sep = " "  # Default separator
        self.last_response_content = FrontData()
        self.last_detected_speaker = None
        self.speaker_languages = {}
        self.diarization_before_transcription = False
        
        # VAD frame size (512 samples = 32ms @16kHz)
        self.vad_frame_samples = 512
        self.vad_frame_bytes = self.vad_frame_samples * self.bytes_per_sample  
        
        # Tuning parameters (can be exposed via args or kept as safe defaults)
        self.vad_pre_roll_frames  = int(getattr(self.args, "vad_pre_roll_frames", 5))   # ≈160ms
        self.vad_min_start_frames = int(getattr(self.args, "vad_min_start_frames", 1))  # start hysteresis (frames)
        self.vad_min_end_frames   = int(getattr(self.args, "vad_min_end_frames", 3))    # end hysteresis (frames)
        self.vad_min_hold_frames  = int(getattr(self.args, "vad_min_hold_frames", 5))   # minimum active frames before end
        
        from collections import deque
        self._vad_ring = deque(maxlen=self.vad_pre_roll_frames + max(1, self.vad_min_start_frames))
        self._vad_triggered = False
        self._vad_pending_start = 0
        self._vad_pending_end = 0
        self._vad_hold = 0
        self._vad_silence_frames = 0        
              
        # Formatter compatibility
        self.silence = True
        self.start_silence = None            
        
        # Confirmation-silence correction: after an end is confirmed, record a small gap
        # to subtract from the next measured inter-utterance silence at the next start.
        # Note: even without injecting audio, we adjust by a small constant (default 0.2s).
        self._confirmation_gap_pending = 0.0
        self.confirmation_gap_s = float(getattr(self.args, "confirmation_gap_s", 0.2))
        
        # ASR invocation period (batch processing)
        self.asr_window_s = float(getattr(self.args, "asr_window_s", max(0.5, float(self.args.min_chunk_size))))
        self._asr_since_last_process_s = 0.0      
        
        self._pending_frames = []     # List[np.ndarray], buffer for 32ms frames
        self._pending_samples = 0     # accumulated sample count (optional, for end-time calculation)            

        if self.diarization_before_transcription:
            self.cumulative_pcm = []
            self.last_start = 0.0
            self.last_end = 0.0
        
        # Models and processing
        self.asr = models.asr
        self.vac_model = models.vac_model
        if self.args.vac:
            self.vac = VADIterator(
                models.vac_model,                # TranscriptionEngine passed JIT model
                threshold=float(getattr(self.args, "vad_threshold", 0.5)),
                sampling_rate=self.sample_rate,
                min_silence_duration_ms=int(getattr(self.args, "vad_min_silence_ms", 500)),
                speech_pad_ms=int(getattr(self.args, "vad_speech_pad_ms", 100)),
            )
        else:
            self.vac = None
                         
        self.ffmpeg_manager = None
        self.ffmpeg_reader_task = None
        self._ffmpeg_error = None

        if not self.is_pcm_input:
            self.ffmpeg_manager = FFmpegManager(
                sample_rate=self.sample_rate,
                channels=self.channels
            )
            async def handle_ffmpeg_error(error_type: str):
                logger.error(f"FFmpeg error: {error_type}")
                self._ffmpeg_error = error_type
            self.ffmpeg_manager.on_error_callback = handle_ffmpeg_error
             
        self.transcription_queue = asyncio.Queue() if self.args.transcription else None
        self.diarization_queue = asyncio.Queue() if self.args.diarization else None
        self.translation_queue = asyncio.Queue() if self.args.target_language else None
        self.pcm_buffer = bytearray()

        self.transcription_task = None
        self.diarization_task = None
        self.translation_task = None
        self.watchdog_task = None
        self.all_tasks_for_cleanup = []
        
        self.transcription = None
        self.translation = None
        self.diarization = None

        if self.args.transcription:
            self.transcription = online_factory(self.args, models.asr)        
            self.sep = self.transcription.asr.sep   
        if self.args.diarization:
            self.diarization = online_diarization_factory(self.args, models.diarization_model)
        if models.translation_model:
            self.translation = online_translation_factory(self.args, models.translation_model)
            
    def _flush_pending_audio_to_transcription(self, stream_time_end: float):
        """Insert accumulated 32ms frames at once into the transcription buffer."""
        if not self._pending_frames:
            return
        if len(self._pending_frames) == 1:
            concat = self._pending_frames[0]
        else:
            concat = np.concatenate(self._pending_frames, axis=0)
        self.transcription.insert_audio_chunk(concat, stream_time_end)
        self._pending_frames.clear()
        self._pending_samples = 0            

    def convert_pcm_to_float(self, pcm_buffer):
        """Convert PCM buffer (s16le) to a normalized float32 NumPy array."""
        return np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

    async def add_dummy_token(self):
        """Placeholder token when no transcription is available."""
        async with self.lock:
            current_time = time() - self.beg_loop
            self.tokens.append(ASRToken(
                start=current_time, end=current_time + 1,
                text=".", speaker=-1, is_dummy=True
            ))
            
    async def get_current_state(self):
        """Get current state."""
        async with self.lock:
            current_time = time()
            
            # Calculate remaining times
            remaining_transcription = 0
            if self.end_buffer > 0:
                remaining_transcription = max(0, round(current_time - self.beg_loop - self.end_buffer, 1))
                
            remaining_diarization = 0
            if self.tokens:
                latest_end = max(self.end_buffer, self.tokens[-1].end if self.tokens else 0)
                remaining_diarization = max(0, round(latest_end - self.end_attributed_speaker, 1))
                
            return State(
                tokens=self.tokens.copy(),
                last_validated_token=self.last_validated_token,
                translated_segments=self.translated_segments.copy(),
                buffer_transcription=self.buffer_transcription,
                end_buffer=self.end_buffer,
                end_attributed_speaker=self.end_attributed_speaker,
                remaining_time_transcription=remaining_transcription,
                remaining_time_diarization=remaining_diarization
            )
            
    async def reset(self):
        """Reset all state variables to initial values."""
        async with self.lock:
            self.tokens = []
            self.translated_segments = []
            self.buffer_transcription = Transcript()
            self.end_buffer = self.end_attributed_speaker = 0
            self.beg_loop = time()

    async def ffmpeg_stdout_reader(self):
        """Read audio data from FFmpeg stdout and process it into the PCM pipeline."""
        beg = time()
        while True:
            try:
                if self.is_stopping:
                    logger.info("Stopping ffmpeg_stdout_reader due to stopping flag.")
                    break

                state = await self.ffmpeg_manager.get_state() if self.ffmpeg_manager else FFmpegState.STOPPED
                if state == FFmpegState.FAILED:
                    logger.error("FFmpeg is in FAILED state, cannot read data")
                    break
                elif state == FFmpegState.STOPPED:
                    logger.info("FFmpeg is stopped")
                    break
                elif state != FFmpegState.RUNNING:
                    await asyncio.sleep(0.1)
                    continue

                current_time = time()
                elapsed_time = max(0.0, current_time - beg)
                buffer_size = max(int(32000 * elapsed_time), 4096)  # dynamic read
                beg = current_time

                chunk = await self.ffmpeg_manager.read_data(buffer_size)
                if not chunk:
                    # No data currently available
                    await asyncio.sleep(0.05)
                    continue

                self.pcm_buffer.extend(chunk)
                await self.handle_pcm_data()

            except asyncio.CancelledError:
                logger.info("ffmpeg_stdout_reader cancelled.")
                break
            except Exception as e:
                logger.warning(f"Exception in ffmpeg_stdout_reader: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.2)

        logger.info("FFmpeg stdout processing finished. Signaling downstream processors if needed.")
        if not self.diarization_before_transcription and self.transcription_queue:
            await self.transcription_queue.put(SENTINEL)
        if self.diarization:
            await self.diarization_queue.put(SENTINEL)
        if self.translation:
            await self.translation_queue.put(SENTINEL)

    async def transcription_processor(self):
        """
        Process audio for transcription (frame-level collection, windowed process_iter).
        - accumulate 32ms frames in a pending list
        - when accumulated duration reaches asr_window_s (e.g., 0.5s/1.0s), insert once and run process_iter
        - on Silence/ChangeSpeaker/SENTINEL, flush pending first to keep boundary consistency
        """
        import numpy as np

        cumulative_pcm_duration_stream_time = 0.0
        self._asr_since_last_process_s = 0.0

        # Use class-level self._pending_frames/self._pending_samples for pending frames

        while True:
            try:
                item = await self.transcription_queue.get()

                # Termination signal
                if item is SENTINEL:
                    # Flush any pending frames first
                    if self._asr_since_last_process_s > 0 or self._pending_frames:
                        self._flush_pending_audio_to_transcription(cumulative_pcm_duration_stream_time)
                        new_tokens, current_audio_processed_upto = await asyncio.to_thread(self.transcription.process_iter)

                        _buffer_transcript = self.transcription.get_buffer()
                        buffer_text = _buffer_transcript.text
                        if new_tokens:
                            validated_text = self.sep.join([t.text for t in new_tokens])
                            if buffer_text.startswith(validated_text):
                                _buffer_transcript.text = buffer_text[len(validated_text):].lstrip()

                        candidate_end_times = [self.end_buffer]
                        if new_tokens:
                            candidate_end_times.append(new_tokens[-1].end)
                        if _buffer_transcript.end is not None:
                            candidate_end_times.append(_buffer_transcript.end)
                        candidate_end_times.append(current_audio_processed_upto)

                        async with self.lock:
                            self.tokens.extend(new_tokens)
                            self.buffer_transcription = _buffer_transcript
                            self.end_buffer = max(candidate_end_times)

                        if self.translation_queue:
                            for token in new_tokens:
                                await self.translation_queue.put(token)

                        self._asr_since_last_process_s = 0.0

                    self.transcription_queue.task_done()
                    break

                # Logging: internal buffer duration and estimated lag
                asr_internal_buffer_duration_s = len(getattr(self.transcription, 'audio_buffer', [])) / self.transcription.SAMPLING_RATE
                transcription_lag_s = max(0.0, (time() - self.beg_loop) - self.end_buffer) if self.beg_loop else 0.0
                asr_processing_logs = f"internal_buffer={asr_internal_buffer_duration_s:.2f}s | lag={transcription_lag_s:.2f}s |"

                # Silence event: flush pending → insert silence → run one flush process_iter
                if isinstance(item, Silence):
                    if self.tokens:
                        asr_processing_logs += f" + Silence of = {item.duration:.2f}s | last_end = {self.tokens[-1].end} |"
                    else:
                        asr_processing_logs += f" + Silence of = {item.duration:.2f}s |"
                    logger.info(asr_processing_logs)

                    # 1) flush accumulated frames first
                    self._flush_pending_audio_to_transcription(cumulative_pcm_duration_stream_time)

                    # 2) insert silence
                    self.transcription.insert_silence(item.duration, self.tokens[-1].end if self.tokens else 0)

                    # 3) run a decoding pass after the silence insertion
                    new_tokens, current_audio_processed_upto = await asyncio.to_thread(self.transcription.process_iter)

                    _buffer_transcript = self.transcription.get_buffer()
                    buffer_text = _buffer_transcript.text
                    if new_tokens:
                        validated_text = self.sep.join([t.text for t in new_tokens])
                        if buffer_text.startswith(validated_text):
                            _buffer_transcript.text = buffer_text[len(validated_text):].lstrip()

                    candidate_end_times = [self.end_buffer]
                    if new_tokens:
                        candidate_end_times.append(new_tokens[-1].end)
                    if _buffer_transcript.end is not None:
                        candidate_end_times.append(_buffer_transcript.end)
                    candidate_end_times.append(current_audio_processed_upto)

                    async with self.lock:
                        self.tokens.extend(new_tokens)
                        self.buffer_transcription = _buffer_transcript
                        self.end_buffer = max(candidate_end_times)

                    if self.translation_queue:
                        for token in new_tokens:
                            await self.translation_queue.put(token)

                    # 4) If there are no commits and buffer has text: force-commit to avoid losing the last utterance
                    if (not new_tokens) and _buffer_transcript and _buffer_transcript.text:
                        text_to_commit = _buffer_transcript.text.strip()
                        if text_to_commit:
                            # Prefer buffer start/end timestamps; if missing, fall back to the last commit's end
                            forced_start = _buffer_transcript.start if _buffer_transcript.start is not None else (self.tokens[-1].start if self.tokens else 0.0)
                            forced_end = _buffer_transcript.end if _buffer_transcript.end is not None else (self.tokens[-1].end if self.tokens else self.end_buffer)
                            forced_token = ASRToken(start=forced_start, end=forced_end, text=text_to_commit, probability=0.95)
                            async with self.lock:
                                self.tokens.append(forced_token)
                                self.buffer_transcription = Transcript()  # clear buffer
                                self.end_buffer = max(self.end_buffer, forced_token.end)
                            if self.translation_queue:
                                await self.translation_queue.put(forced_token)
                            # Re-align streaming ASR internal buffer time to current processed time to avoid duplicates
                            try:
                                self.transcription.init(offset=current_audio_processed_upto)
                            except Exception as _:
                                pass

                    # Reset window counter at the silence boundary
                    self._asr_since_last_process_s = 0.0
                    self.transcription_queue.task_done()
                    continue

                # ChangeSpeaker handling (ensure boundary consistency)
                if isinstance(item, ChangeSpeaker):
                    # Flush pending before speaker boundary
                    self._flush_pending_audio_to_transcription(cumulative_pcm_duration_stream_time)
                    self.transcription.new_speaker(item)
                    self.transcription_queue.task_done()
                    continue

                # Regular audio frame (np.ndarray)
                if isinstance(item, np.ndarray):
                    pcm_array = item
                    logger.info(asr_processing_logs)

                    # Accumulate 32ms frames in the pending buffer
                    duration_this_chunk = len(pcm_array) / self.sample_rate
                    cumulative_pcm_duration_stream_time += duration_this_chunk
                    stream_time_end_of_current_pcm = cumulative_pcm_duration_stream_time

                    self._pending_frames.append(pcm_array)
                    self._pending_samples += len(pcm_array)

                    # When asr_window_s is reached: insert pending at once → run process_iter
                    self._asr_since_last_process_s += duration_this_chunk
                    if self._asr_since_last_process_s >= self.asr_window_s:
                        # (1) pending flush
                        self._flush_pending_audio_to_transcription(stream_time_end_of_current_pcm)

                        # (2) run decoding
                        new_tokens, current_audio_processed_upto = await asyncio.to_thread(self.transcription.process_iter)

                        _buffer_transcript = self.transcription.get_buffer()
                        buffer_text = _buffer_transcript.text
                        if new_tokens:
                            validated_text = self.sep.join([t.text for t in new_tokens])
                            if buffer_text.startswith(validated_text):
                                _buffer_transcript.text = buffer_text[len(validated_text):].lstrip()

                        candidate_end_times = [self.end_buffer]
                        if new_tokens:
                            candidate_end_times.append(new_tokens[-1].end)
                        if _buffer_transcript.end is not None:
                            candidate_end_times.append(_buffer_transcript.end)
                        candidate_end_times.append(current_audio_processed_upto)

                        async with self.lock:
                            self.tokens.extend(new_tokens)
                            self.buffer_transcription = _buffer_transcript
                            self.end_buffer = max(candidate_end_times)

                        if self.translation_queue:
                            for token in new_tokens:
                                await self.translation_queue.put(token)

                        self._asr_since_last_process_s = 0.0

                    self.transcription_queue.task_done()
                    continue

                # Fallback handling
                self.transcription_queue.task_done()
                continue

            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if 'item' in locals() and item is not SENTINEL:
                    self.transcription_queue.task_done()

        # Post-termination handling (unchanged)
        if self.is_stopping:
            logger.info("Transcription processor finishing due to stopping flag.")
            if self.diarization_queue:
                await self.diarization_queue.put(SENTINEL)
            if self.translation_queue:
                await self.translation_queue.put(SENTINEL)

        logger.info("Transcription processor task finished.")


    async def diarization_processor(self, diarization_obj):
        """Process audio chunks for speaker diarization."""
        if self.diarization_before_transcription:
            self.current_speaker = 0
            await self.transcription_queue.put(ChangeSpeaker(speaker=self.current_speaker, start=0.0))
        while True:
            try:
                item = await self.diarization_queue.get()
                if item is SENTINEL:
                    logger.debug("Diarization processor received sentinel. Finishing.")
                    self.diarization_queue.task_done()
                    break
                elif type(item) is Silence:
                    diarization_obj.insert_silence(item.duration)
                    continue
                elif isinstance(item, np.ndarray):
                    pcm_array = item
                else:
                    raise Exception('item should be pcm_array') 
                
                
                
                # Process diarization
                await diarization_obj.diarize(pcm_array)
                if self.diarization_before_transcription:
                    segments = diarization_obj.get_segments()
                    self.cumulative_pcm.append(pcm_array)
                    if segments:
                        last_segment = segments[-1]                    
                        if last_segment.speaker != self.current_speaker:
                            cut_sec = last_segment.start - self.last_end
                            to_transcript, self.cumulative_pcm = cut_at(self.cumulative_pcm, cut_sec)
                            await self.transcription_queue.put(to_transcript)
                            
                            self.current_speaker = last_segment.speaker
                            await self.transcription_queue.put(ChangeSpeaker(speaker=self.current_speaker, start=last_segment.start))
                            
                            cut_sec = last_segment.end - last_segment.start
                            to_transcript, self.cumulative_pcm = cut_at(self.cumulative_pcm, cut_sec)
                            await self.transcription_queue.put(to_transcript)                            
                            self.last_start = last_segment.start
                            self.last_end = last_segment.end
                        else:
                            cut_sec = last_segment.end - self.last_end
                            to_transcript, self.cumulative_pcm = cut_at(self.cumulative_pcm, cut_sec)
                            await self.transcription_queue.put(to_transcript)
                            self.last_end = last_segment.end
                elif not self.diarization_before_transcription:           
                    async with self.lock:
                        self.tokens = diarization_obj.assign_speakers_to_tokens(
                            self.tokens,
                            use_punctuation_split=self.args.punctuation_split
                        )
                if len(self.tokens) > 0:
                    self.end_attributed_speaker = max(self.tokens[-1].end, self.end_attributed_speaker)
                self.diarization_queue.task_done()
                
            except Exception as e:
                logger.warning(f"Exception in diarization_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if 'pcm_array' in locals() and pcm_array is not SENTINEL:
                    self.diarization_queue.task_done()
        logger.info("Diarization processor task finished.")

    async def translation_processor(self):
        # the idea is to ignore diarization for the moment. We use only transcription tokens. 
        # And the speaker is attributed given the segments used for the translation
        # in the future we want to have different languages for each speaker etc, so it will be more complex.
        while True:
            try:
                item = await self.translation_queue.get() #block until at least 1 token
                if item is SENTINEL:
                    logger.debug("Translation processor received sentinel. Finishing.")
                    self.translation_queue.task_done()
                    break
                elif type(item) is Silence:
                    self.translation.insert_silence(item.duration)
                    continue
                
                # get all the available tokens for translation. The more words, the more precise
                tokens_to_process = [item]
                additional_tokens = await get_all_from_queue(self.translation_queue)
                
                sentinel_found = False
                for additional_token in additional_tokens:
                    if additional_token is SENTINEL:
                        sentinel_found = True
                        break
                    elif type(additional_token) is Silence:
                        self.translation.insert_silence(additional_token.duration)
                        continue
                    else:
                        tokens_to_process.append(additional_token)                
                if tokens_to_process:
                    self.translation.insert_tokens(tokens_to_process)
                    self.translated_segments = await asyncio.to_thread(self.translation.process)
                self.translation_queue.task_done()
                for _ in additional_tokens:
                    self.translation_queue.task_done()
                
                if sentinel_found:
                    logger.debug("Translation processor received sentinel in batch. Finishing.")
                    break
                
            except Exception as e:
                logger.warning(f"Exception in translation_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if 'token' in locals() and item is not SENTINEL:
                    self.translation_queue.task_done()
                if 'additional_tokens' in locals():
                    for _ in additional_tokens:
                        self.translation_queue.task_done()
        logger.info("Translation processor task finished.")

    async def results_formatter(self):
        """Format processing results for output."""
        while True:
            try:
                if self._ffmpeg_error:
                    yield FrontData(status="error", error=f"FFmpeg error: {self._ffmpeg_error}")
                    self._ffmpeg_error = None
                    await asyncio.sleep(1)
                    continue

                state = await self.get_current_state()
                
                
                lines, undiarized_text = format_output(
                    state,
                    self.silence,
                    current_time = time() - self.beg_loop,
                    args = self.args,
                    sep=self.sep
                )
                if lines and lines[-1].speaker == -2:
                    buffer_transcription = Transcript()
                else:
                    buffer_transcription = state.buffer_transcription

                buffer_diarization = ''
                if undiarized_text:
                    buffer_diarization = self.sep.join(undiarized_text)

                    async with self.lock:
                        self.end_attributed_speaker = state.end_attributed_speaker
                
                response_status = "active_transcription"
                if not state.tokens and not buffer_transcription and not buffer_diarization:
                    response_status = "no_audio_detected"
                    lines = []
                elif not lines:
                    lines = [Line(
                        speaker=1,
                        start=state.end_buffer,
                        end=state.end_buffer
                    )]
                
                response = FrontData(
                    status=response_status,
                    lines=lines,
                    buffer_transcription=buffer_transcription.text.strip(),
                    buffer_diarization=buffer_diarization,
                    remaining_time_transcription=state.remaining_time_transcription,
                    remaining_time_diarization=state.remaining_time_diarization if self.args.diarization else 0
                )
                                
                should_push = (response != self.last_response_content)
                if should_push and (lines or buffer_transcription or buffer_diarization or response_status == "no_audio_detected"):
                    yield response
                    self.last_response_content = response
                
                # Check for termination condition
                if self.is_stopping:
                    all_processors_done = True
                    if self.args.transcription and self.transcription_task and not self.transcription_task.done():
                        all_processors_done = False
                    if self.args.diarization and self.diarization_task and not self.diarization_task.done():
                        all_processors_done = False
                    
                    if all_processors_done:
                        logger.info("Results formatter: All upstream processors are done and in stopping state. Terminating.")
                        return
                
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.warning(f"Exception in results_formatter: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.5)
        
    async def create_tasks(self):
        """Create and start processing tasks."""
        self.all_tasks_for_cleanup = []
        processing_tasks_for_watchdog = []

        # If using FFmpeg (non-PCM input), start it and spawn stdout reader
        if not self.is_pcm_input:
            success = await self.ffmpeg_manager.start()
            if not success:
                logger.error("Failed to start FFmpeg manager")
                async def error_generator():
                    yield FrontData(
                        status="error",
                        error="FFmpeg failed to start. Please check that FFmpeg is installed."
                    )
                return error_generator()
            self.ffmpeg_reader_task = asyncio.create_task(self.ffmpeg_stdout_reader())
            self.all_tasks_for_cleanup.append(self.ffmpeg_reader_task)
            processing_tasks_for_watchdog.append(self.ffmpeg_reader_task)

        if self.transcription:
            self.transcription_task = asyncio.create_task(self.transcription_processor())
            self.all_tasks_for_cleanup.append(self.transcription_task)
            processing_tasks_for_watchdog.append(self.transcription_task)
            
        if self.diarization:
            self.diarization_task = asyncio.create_task(self.diarization_processor(self.diarization))
            self.all_tasks_for_cleanup.append(self.diarization_task)
            processing_tasks_for_watchdog.append(self.diarization_task)
        
        if self.translation:
            self.translation_task = asyncio.create_task(self.translation_processor())
            self.all_tasks_for_cleanup.append(self.translation_task)
            processing_tasks_for_watchdog.append(self.translation_task)
        
        # Monitor overall system health
        self.watchdog_task = asyncio.create_task(self.watchdog(processing_tasks_for_watchdog))
        self.all_tasks_for_cleanup.append(self.watchdog_task)
        
        return self.results_formatter()

    async def watchdog(self, tasks_to_monitor):
        """Monitors the health of critical processing tasks."""
        while True:
            try:
                await asyncio.sleep(10)
                
                for i, task in enumerate(tasks_to_monitor):
                    if task.done():
                        exc = task.exception()
                        task_name = task.get_name() if hasattr(task, 'get_name') else f"Monitored Task {i}"
                        if exc:
                            logger.error(f"{task_name} unexpectedly completed with exception: {exc}")
                        else:
                            logger.info(f"{task_name} completed normally.")
                    
            except asyncio.CancelledError:
                logger.info("Watchdog task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in watchdog task: {e}", exc_info=True)
        
    async def cleanup(self):
        """Clean up resources when processing is complete."""
        logger.info("Starting cleanup of AudioProcessor resources.")
        self.is_stopping = True
        for task in self.all_tasks_for_cleanup:
            if task and not task.done():
                task.cancel()
            
        created_tasks = [t for t in self.all_tasks_for_cleanup if t]
        if created_tasks:
            await asyncio.gather(*created_tasks, return_exceptions=True)
        logger.info("All processing tasks cancelled or finished.")

        if not self.is_pcm_input and self.ffmpeg_manager:
            try:
                await self.ffmpeg_manager.stop()
                logger.info("FFmpeg manager stopped.")
            except Exception as e:
                logger.warning(f"Error stopping FFmpeg manager: {e}")
        if self.diarization:
            self.diarization.close()
        logger.info("AudioProcessor cleanup complete.")


    async def process_audio(self, message):
        """Process incoming audio data."""

        if not self.beg_loop:
            self.beg_loop = time()

        if not message:
            logger.info("Empty audio message received, initiating stop sequence.")
            self.is_stopping = True
             
            if self.transcription_queue:
                await self.transcription_queue.put(SENTINEL)

            if not self.is_pcm_input and self.ffmpeg_manager:
                await self.ffmpeg_manager.stop()

            return

        if self.is_stopping:
            logger.warning("AudioProcessor is stopping. Ignoring incoming audio.")
            return

        if self.is_pcm_input:
            self.pcm_buffer.extend(message)
            await self.handle_pcm_data()
        else:
            if not self.ffmpeg_manager:
                logger.error("FFmpeg manager not initialized for non-PCM input.")
                return
            success = await self.ffmpeg_manager.write_data(message)
            if not success:
                ffmpeg_state = await self.ffmpeg_manager.get_state()
                if ffmpeg_state == FFmpegState.FAILED:
                    logger.error("FFmpeg is in FAILED state, cannot process audio")
                else:
                    logger.warning("Failed to write audio data to FFmpeg")

    async def handle_pcm_data(self):
        """
        Notes:
        - Run VAD in 512-sample (≈32ms @16k) frames
        - Maintain pre-roll + start/end hysteresis
        - Send only voiced frames to queues
        - Between utterances, compute silence duration from frames and enqueue Silence(duration)
        """
        while len(self.pcm_buffer) >= self.vad_frame_bytes:
            frame_bytes = self.pcm_buffer[:self.vad_frame_bytes]
            self.pcm_buffer = self.pcm_buffer[self.vad_frame_bytes:]

            # Convert to float32 [-1..1], fixed 512-sample frame
            frame_f32 = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if frame_f32.size != self.vad_frame_samples:
                continue

            # VAC disabled: bypass the gate and forward frames directly to queues
            if self.vac is None:
                self.silence = False
                if not self.diarization_before_transcription and self.transcription_queue:
                    await self.transcription_queue.put(frame_f32.copy())
                if self.args.diarization and self.diarization_queue:
                    await self.diarization_queue.put(frame_f32.copy())
                continue

            # Run VAD
            speech_event = None
            if self.vac is not None:
                speech_event = self.vac(frame_f32, return_seconds=False)

            event_start = bool(speech_event and ('start' in speech_event) and not self._vad_triggered)
            event_end   = bool(speech_event and ('end'   in speech_event) and self._vad_triggered)

            # Before start is confirmed
            if not self._vad_triggered:
                # Accumulate pre-roll
                self._vad_ring.append(frame_f32)

                # Start hysteresis countdown
                if event_start:
                    self._vad_pending_start = max(self._vad_pending_start, self.vad_min_start_frames)

                if self._vad_pending_start > 0:
                    self._vad_pending_start -= 1
                    if self._vad_pending_start == 0:
                        # (1) Emit silence for previous end→start gap
                        if self._vad_silence_frames > 0:
                            # Subtract pending confirmation-gap from measured silence
                            measured_silence_sec = (self._vad_silence_frames * self.vad_frame_samples) / float(self.sample_rate)
                            adjusted_silence_sec = max(0.0, measured_silence_sec - self._confirmation_gap_pending)
                            # Consume pending confirmation-gap
                            self._confirmation_gap_pending = 0.0
                            sil = Silence(duration=adjusted_silence_sec)
                            if not self.diarization_before_transcription and self.transcription_queue:
                                await self.transcription_queue.put(sil)
                            if self.args.diarization and self.diarization_queue:
                                await self.diarization_queue.put(sil)
                            if self.translation_queue:
                                await self.translation_queue.put(sil)
                            self._vad_silence_frames = 0

                        # (2) Send pre-roll + current frame
                        while self._vad_ring:
                            fr = self._vad_ring.popleft()
                            if not self.diarization_before_transcription and self.transcription_queue:
                                await self.transcription_queue.put(fr.copy())
                            if self.args.diarization and self.diarization_queue:
                                await self.diarization_queue.put(fr.copy())

                        # State transition: start confirmed
                        self._vad_triggered = True
                        self._vad_hold = 0
                        self._vad_pending_end = 0
                        self.silence = False
                        self.start_silence = None
                else:
                    # Still before start → accumulate silent frames
                    self._vad_silence_frames += 1

                continue  # stop here while start is not yet confirmed

            # Active speech: forward frames
            if not self.diarization_before_transcription and self.transcription_queue:
                await self.transcription_queue.put(frame_f32.copy())
            if self.args.diarization and self.diarization_queue:
                await self.diarization_queue.put(frame_f32.copy())
            self._vad_hold += 1

            # End hysteresis
            if event_end:
                if self._vad_hold >= self.vad_min_hold_frames:
                    self._vad_pending_end = max(self._vad_pending_end, self.vad_min_end_frames)
                else:
                    self._vad_pending_end = 0

            if self._vad_pending_end > 0:
                self._vad_pending_end -= 1
                if self._vad_pending_end == 0:
                    # End confirmed
                    self._vad_triggered = False
                    self._vad_hold = 0
                    self.silence = True
                    self._vad_silence_frames = 0
                    self.start_silence = None
                    # Record pending confirmation-gap to subtract at the next start
                    if self.confirmation_gap_s > 0:
                        self._confirmation_gap_pending += self.confirmation_gap_s
