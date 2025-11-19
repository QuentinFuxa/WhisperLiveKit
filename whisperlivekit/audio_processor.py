import asyncio
import numpy as np
from time import time, sleep
import math
import logging
import traceback
from whisperlivekit.timed_objects import ASRToken, Silence, Line, FrontData, State, Transcript, ChangeSpeaker
from whisperlivekit.core import TranscriptionEngine, online_factory, online_diarization_factory, online_translation_factory
from whisperlivekit.silero_vad_iterator import FixedVADIterator
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

    first_item = await queue.get()
    queue.task_done()
    if first_item is SENTINEL:
        return first_item
    if isinstance(first_item, Silence):
        return first_item
    items.append(first_item)
    
    while True:
        if not queue._queue:
            break
        next_item = queue._queue[0]
        if next_item is SENTINEL:
            break
        if isinstance(next_item, Silence):
            break
        items.append(await queue.get())
        queue.task_done()
    if isinstance(items[0], np.ndarray):
        return np.concatenate(items)
    else: #translation
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
        self.silence = True
        self.silence_duration = 0.0
        self.start_silence = None
        self.last_silence_dispatch_time = None
        self.state = State()
        self.lock = asyncio.Lock()
        self.sep = " "  # Default separator
        self.last_response_content = FrontData()
        self.last_detected_speaker = None
        self.speaker_languages = {}
        self.diarization_before_transcription = False

        self.segments = []
        

        if self.diarization_before_transcription:
            self.cumulative_pcm = []
            self.last_start = 0.0
            self.last_end = 0.0
        
        # Models and processing
        self.asr = models.asr
        self.vac_model = models.vac_model
        if self.args.vac:
            self.vac = FixedVADIterator(models.vac_model)
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
        self.total_pcm_samples = 0

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

    async def _push_silence_event(self, silence_buffer: Silence):
        if not self.diarization_before_transcription and self.transcription_queue:
            await self.transcription_queue.put(silence_buffer)
        if self.args.diarization and self.diarization_queue:
            await self.diarization_queue.put(silence_buffer)
        if self.translation_queue:
            await self.translation_queue.put(silence_buffer)

    async def _begin_silence(self):
        if self.silence:
            return
        self.silence = True
        now = time()
        self.start_silence = now
        self.last_silence_dispatch_time = now
        await self._push_silence_event(Silence(is_starting=True))

    async def _end_silence(self):
        if not self.silence:
            return
        now = time()
        duration = now - (self.last_silence_dispatch_time if self.last_silence_dispatch_time else self.state.beg_loop)
        await self._push_silence_event(Silence(duration=duration, has_ended=True))
        self.last_silence_dispatch_time = now
        self.silence = False
        self.start_silence = None
        self.last_silence_dispatch_time = None

    async def _enqueue_active_audio(self, pcm_chunk: np.ndarray):
        if pcm_chunk is None or pcm_chunk.size == 0:
            return
        if not self.diarization_before_transcription and self.transcription_queue:
            await self.transcription_queue.put(pcm_chunk.copy())
        if self.args.diarization and self.diarization_queue:
            await self.diarization_queue.put(pcm_chunk.copy())
        self.silence_duration = 0.0

    def _slice_before_silence(self, pcm_array, chunk_sample_start, silence_sample):
        if silence_sample is None:
            return None
        relative_index = int(silence_sample - chunk_sample_start)
        if relative_index <= 0:
            return None
        split_index = min(relative_index, len(pcm_array))
        if split_index <= 0:
            return None
        return pcm_array[:split_index]

    def convert_pcm_to_float(self, pcm_buffer):
        """Convert PCM buffer in s16le format to normalized NumPy array."""
        return np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

    async def add_dummy_token(self):
        """Placeholder token when no transcription is available."""
        async with self.lock:
            current_time = time() - self.state.beg_loop
            self.state.tokens.append(ASRToken(
                start=current_time, end=current_time + 1,
                text=".", speaker=-1, is_dummy=True
            ))
            
    async def get_current_state(self):
        """Get current state."""
        async with self.lock:
            current_time = time()
            
            remaining_transcription = 0
            if self.state.end_buffer > 0:
                remaining_transcription = max(0, round(current_time - self.state.beg_loop - self.state.end_buffer, 1))
                
            remaining_diarization = 0
            if self.state.tokens:
                latest_end = max(self.state.end_buffer, self.state.tokens[-1].end if self.state.tokens else 0)
                remaining_diarization = max(0, round(latest_end - self.state.end_attributed_speaker, 1))
                
            self.state.remaining_time_transcription = remaining_transcription
            self.state.remaining_time_diarization = remaining_diarization
            
            return self.state

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
        """Process audio chunks for transcription."""
        cumulative_pcm_duration_stream_time = 0.0
        
        while True:
            try:
                # item = await self.transcription_queue.get()
                item = await get_all_from_queue(self.transcription_queue)
                if item is SENTINEL:
                    logger.debug("Transcription processor received sentinel. Finishing.")
                    break

                asr_internal_buffer_duration_s = len(getattr(self.transcription, 'audio_buffer', [])) / self.transcription.SAMPLING_RATE
                transcription_lag_s = max(0.0, time() - self.state.beg_loop - self.state.end_buffer)
                asr_processing_logs = f"internal_buffer={asr_internal_buffer_duration_s:.2f}s | lag={transcription_lag_s:.2f}s |"
                stream_time_end_of_current_pcm = cumulative_pcm_duration_stream_time
                new_tokens = []
                current_audio_processed_upto = self.state.end_buffer

                if isinstance(item, Silence):
                    if item.is_starting:
                        new_tokens, current_audio_processed_upto = await asyncio.to_thread(
                            self.transcription.start_silence
                        )
                        asr_processing_logs += f" + Silence starting"
                    if item.has_ended:
                        asr_processing_logs += f" + Silence of = {item.duration:.2f}s"
                        cumulative_pcm_duration_stream_time += item.duration
                        current_audio_processed_upto = cumulative_pcm_duration_stream_time
                        self.transcription.end_silence(item.duration, self.state.tokens[-1].end if self.state.tokens else 0)
                    if self.state.tokens:
                        asr_processing_logs += f" | last_end = {self.state.tokens[-1].end} |"
                    logger.info(asr_processing_logs)
                    new_tokens = new_tokens or []
                    current_audio_processed_upto = max(current_audio_processed_upto, stream_time_end_of_current_pcm)
                elif isinstance(item, ChangeSpeaker):
                    self.transcription.new_speaker(item)
                    # self.transcription_queue.task_done()
                    continue
                elif isinstance(item, np.ndarray):
                    pcm_array = item
                    logger.info(asr_processing_logs)
                    cumulative_pcm_duration_stream_time += len(pcm_array) / self.sample_rate
                    stream_time_end_of_current_pcm = cumulative_pcm_duration_stream_time
                    self.transcription.insert_audio_chunk(pcm_array, stream_time_end_of_current_pcm)
                    new_tokens, current_audio_processed_upto = await asyncio.to_thread(self.transcription.process_iter)
                    new_tokens = new_tokens or []
                else:
                    continue
                    return

                _buffer_transcript = self.transcription.get_buffer()
                buffer_text = _buffer_transcript.text

                if new_tokens:
                    validated_text = self.sep.join([t.text for t in new_tokens])
                    if buffer_text.startswith(validated_text):
                        _buffer_transcript.text = buffer_text[len(validated_text):].lstrip()

                candidate_end_times = [self.state.end_buffer]

                if new_tokens:
                    candidate_end_times.append(new_tokens[-1].end)
                
                if _buffer_transcript.end is not None:
                    candidate_end_times.append(_buffer_transcript.end)
                
                candidate_end_times.append(current_audio_processed_upto)
                
                async with self.lock:
                    self.state.tokens.extend(new_tokens)
                    self.state.buffer_transcription = _buffer_transcript
                    self.state.end_buffer = max(candidate_end_times)
                
                if self.translation_queue:
                    for token in new_tokens:
                        await self.translation_queue.put(token)                
            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if 'pcm_array' in locals() and pcm_array is not SENTINEL : # Check if pcm_array was assigned from queue
                    self.transcription_queue.task_done()
        
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
                item = await get_all_from_queue(self.diarization_queue)
                if item is SENTINEL:
                    logger.debug("Diarization processor received sentinel. Finishing.")
                    break
                elif type(item) is Silence:
                    if item.has_ended:
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
                    segments = diarization_obj.get_segments()
                    async with self.lock:
                        self.state.speaker_segments = segments.copy()
                        if segments:
                            self.state.end_attributed_speaker = max(seg.end for seg in segments)
                
            except Exception as e:
                logger.warning(f"Exception in diarization_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
        logger.info("Diarization processor task finished.")

    async def translation_processor(self):
        # the idea is to ignore diarization for the moment. We use only transcription tokens. 
        # And the speaker is attributed given the segments used for the translation
        # in the future we want to have different languages for each speaker etc, so it will be more complex.
        while True:
            try:
                tokens_to_process = await get_all_from_queue(self.translation_queue)
                if tokens_to_process is SENTINEL:
                    logger.debug("Translation processor received sentinel. Finishing.")
                    self.translation_queue.task_done()
                    break
                elif type(tokens_to_process) is Silence:
                    if tokens_to_process.has_ended:
                        self.translation.insert_silence(tokens_to_process.duration)
                    continue               
                if tokens_to_process:
                    self.translation.insert_tokens(tokens_to_process)
                    translation_validated_segments, buffer_translation = await asyncio.to_thread(self.translation.process)
                    async with self.lock:
                        self.state.translation_validated_segments = translation_validated_segments
                        self.state.buffer_translation = buffer_translation
            except Exception as e:
                logger.warning(f"Exception in translation_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
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
                        self.state.end_attributed_speaker = state.end_attributed_speaker

                buffer_translation_text = ''
                if state.buffer_translation:
                    raw_buffer_translation = getattr(state.buffer_translation, 'text', state.buffer_translation)
                    if raw_buffer_translation:
                        buffer_translation_text = raw_buffer_translation.strip()
                
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
                    buffer_translation=buffer_translation_text,
                    remaining_time_transcription=state.remaining_time_transcription,
                    remaining_time_diarization=state.remaining_time_diarization if self.args.diarization else 0
                )
                                
                should_push = (response != self.last_response_content)
                if should_push and (lines or buffer_transcription or buffer_diarization or response_status == "no_audio_detected"):
                    yield response
                    self.last_response_content = response
                
                if self.is_stopping and self._processing_tasks_done():
                    logger.info("Results formatter: All upstream processors are done and in stopping state. Terminating.")
                    return
                
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.warning(f"Exception in results_formatter. Traceback: {traceback.format_exc()}")
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
        tasks_remaining = [task for task in tasks_to_monitor if task]
        while True:
            try:
                if not tasks_remaining:
                    logger.info("Watchdog task finishing: all monitored tasks completed.")
                    return

                await asyncio.sleep(10)
                
                for i, task in enumerate(list(tasks_remaining)):
                    if task.done():
                        exc = task.exception()
                        task_name = task.get_name() if hasattr(task, 'get_name') else f"Monitored Task {i}"
                        if exc:
                            logger.error(f"{task_name} unexpectedly completed with exception: {exc}")
                        else:
                            logger.info(f"{task_name} completed normally.")
                        tasks_remaining.remove(task)
                    
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

    def _processing_tasks_done(self):
        """Return True when all active processing tasks have completed."""
        tasks_to_check = [
            self.transcription_task,
            self.diarization_task,
            self.translation_task,
            self.ffmpeg_reader_task,
        ]
        return all(task.done() for task in tasks_to_check if task)


    async def process_audio(self, message):
        """Process incoming audio data."""

        if not self.state.beg_loop:
            self.state.beg_loop = time()

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
        # Process when enough data
        if len(self.pcm_buffer) < self.bytes_per_sec:
            return

        if len(self.pcm_buffer) > self.max_bytes_per_sec:
            logger.warning(
                f"Audio buffer too large: {len(self.pcm_buffer) / self.bytes_per_sec:.2f}s. "
                f"Consider using a smaller model."
            )

        chunk_size = min(len(self.pcm_buffer), self.max_bytes_per_sec)
        aligned_chunk_size = (chunk_size // self.bytes_per_sample) * self.bytes_per_sample
        
        if aligned_chunk_size == 0:
            return
        pcm_array = self.convert_pcm_to_float(self.pcm_buffer[:aligned_chunk_size])
        self.pcm_buffer = self.pcm_buffer[aligned_chunk_size:]

        num_samples = len(pcm_array)
        chunk_sample_start = self.total_pcm_samples
        chunk_sample_end = chunk_sample_start + num_samples

        res = None
        if self.args.vac:
            res = self.vac(pcm_array)

        if res is not None:
            silence_detected = res.get("end", 0) > res.get("start", 0)
            if silence_detected and not self.silence:
                pre_silence_chunk = self._slice_before_silence(
                    pcm_array, chunk_sample_start, res.get("end")
                )
                if pre_silence_chunk is not None and pre_silence_chunk.size > 0:
                    await self._enqueue_active_audio(pre_silence_chunk)
                await self._begin_silence()
            elif self.silence:
                await self._end_silence()

        if not self.silence:
            await self._enqueue_active_audio(pcm_array)

        self.total_pcm_samples = chunk_sample_end

        if not self.args.transcription and not self.args.diarization:
            await asyncio.sleep(0.1)
