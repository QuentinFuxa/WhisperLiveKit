import sys
import numpy as np
import logging
from typing import List, Tuple, Optional
from whisperlivekit.timed_objects import ASRToken, Sentence, Transcript

logger = logging.getLogger(__name__)

# simulStreaming imports - we check if the files are here
try:
    import torch
    from whisperlivekit.simul_whisper.config import AlignAttConfig
    SIMULSTREAMING_AVAILABLE = True
except ImportError:
    logger.warning("SimulStreaming dependencies not available for online processor.")
    SIMULSTREAMING_AVAILABLE = False
    OnlineProcessorInterface = None
    torch = None


class HypothesisBuffer:
    """
    Buffer to store and process ASR hypothesis tokens.

    It holds:
      - committed_in_buffer: tokens that have been confirmed (committed)
      - buffer: the last hypothesis that is not yet committed
      - new: new tokens coming from the recognizer
    """
    def __init__(self, logfile=sys.stderr, confidence_validation=False):
        self.confidence_validation = confidence_validation
        self.committed_in_buffer: List[ASRToken] = []
        self.buffer: List[ASRToken] = []
        self.new: List[ASRToken] = []
        self.last_committed_time = 0.0
        self.last_committed_word: Optional[str] = None
        self.logfile = logfile

    def insert(self, new_tokens: List[ASRToken], offset: float):
        """
        Insert new tokens (after applying a time offset) and compare them with the 
        already committed tokens. Only tokens that extend the committed hypothesis 
        are added.
        """
        # Apply the offset to each token.
        new_tokens = [token.with_offset(offset) for token in new_tokens]
        # Only keep tokens that are roughly “new”
        self.new = [token for token in new_tokens if token.start > self.last_committed_time - 0.1]

        if self.new:
            first_token = self.new[0]
            if abs(first_token.start - self.last_committed_time) < 1:
                if self.committed_in_buffer:
                    committed_len = len(self.committed_in_buffer)
                    new_len = len(self.new)
                    # Try to match 1 to 5 consecutive tokens
                    max_ngram = min(min(committed_len, new_len), 5)
                    for i in range(1, max_ngram + 1):
                        committed_ngram = " ".join(token.text for token in self.committed_in_buffer[-i:])
                        new_ngram = " ".join(token.text for token in self.new[:i])
                        if committed_ngram == new_ngram:
                            removed = []
                            for _ in range(i):
                                removed_token = self.new.pop(0)
                                removed.append(repr(removed_token))
                            logger.debug(f"Removing last {i} words: {' '.join(removed)}")
                            break

    def flush(self) -> List[ASRToken]:
        """
        Returns the committed chunk, defined as the longest common prefix
        between the previous hypothesis and the new tokens.
        """
        committed: List[ASRToken] = []
        while self.new:
            current_new = self.new[0]
            if self.confidence_validation and current_new.probability and current_new.probability > 0.95:
                committed.append(current_new)
                self.last_committed_word = current_new.text
                self.last_committed_time = current_new.end
                self.new.pop(0)
                self.buffer.pop(0) if self.buffer else None
            elif not self.buffer:
                break
            elif current_new.text == self.buffer[0].text:
                committed.append(current_new)
                self.last_committed_word = current_new.text
                self.last_committed_time = current_new.end
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(committed)
        return committed

    def pop_committed(self, time: float):
        """
        Remove tokens (from the beginning) that have ended before `time`.
        """
        while self.committed_in_buffer and self.committed_in_buffer[0].end <= time:
            self.committed_in_buffer.pop(0)



class OnlineASRProcessor:
    """
    Processes incoming audio in a streaming fashion, calling the ASR system
    periodically, and uses a hypothesis buffer to commit and trim recognized text.
    
    The processor supports two types of buffer trimming:
      - "sentence": trims at sentence boundaries (using a sentence tokenizer)
      - "segment": trims at fixed segment durations.
    """
    SAMPLING_RATE = 16000

    def __init__(
        self,
        asr,
        tokenize_method: Optional[callable] = None,
        buffer_trimming: Tuple[str, float] = ("segment", 15),
        confidence_validation = False,
        logfile=sys.stderr,
    ):
        """
        asr: An ASR system object (for example, a WhisperASR instance) that
             provides a `transcribe` method, a `ts_words` method (to extract tokens),
             a `segments_end_ts` method, and a separator attribute `sep`.
        tokenize_method: A function that receives text and returns a list of sentence strings.
        buffer_trimming: A tuple (option, seconds), where option is either "sentence" or "segment".
        """
        self.asr = asr
        self.tokenize = tokenize_method
        self.logfile = logfile
        self.confidence_validation = confidence_validation
        self.init()

        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

        if self.buffer_trimming_way not in ["sentence", "segment"]:
            raise ValueError("buffer_trimming must be either 'sentence' or 'segment'")
        if self.buffer_trimming_sec <= 0:
            raise ValueError("buffer_trimming_sec must be positive")
        elif self.buffer_trimming_sec > 30:
            logger.warning(
                f"buffer_trimming_sec is set to {self.buffer_trimming_sec}, which is very long. It may cause OOM."
            )

    def init(self, offset: Optional[float] = None):
        """Initialize or reset the processing buffers."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile, confidence_validation=self.confidence_validation)
        self.buffer_time_offset = offset if offset is not None else 0.0
        self.transcript_buffer.last_committed_time = self.buffer_time_offset
        self.committed: List[ASRToken] = []
        self.time_of_last_asr_output = 0.0

    def get_audio_buffer_end_time(self) -> float:
        """Returns the absolute end time of the current audio_buffer."""
        return self.buffer_time_offset + (len(self.audio_buffer) / self.SAMPLING_RATE)

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: Optional[float] = None):
        """Append an audio chunk (a numpy array) to the current audio buffer."""
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self) -> Tuple[str, str]:
        """
        Returns a tuple: (prompt, context), where:
          - prompt is a 200-character suffix of committed text that falls 
            outside the current audio buffer.
          - context is the committed text within the current audio buffer.
        """
        k = len(self.committed)
        while k > 0 and self.committed[k - 1].end > self.buffer_time_offset:
            k -= 1

        prompt_tokens = self.committed[:k]
        prompt_words = [token.text for token in prompt_tokens]
        prompt_list = []
        length_count = 0
        # Use the last words until reaching 200 characters.
        while prompt_words and length_count < 200:
            word = prompt_words.pop(-1)
            length_count += len(word) + 1
            prompt_list.append(word)
        non_prompt_tokens = self.committed[k:]
        context_text = self.asr.sep.join(token.text for token in non_prompt_tokens)
        return self.asr.sep.join(prompt_list[::-1]), context_text

    def get_buffer(self):
        """
        Get the unvalidated buffer in string format.
        """
        return self.concatenate_tokens(self.transcript_buffer.buffer)
        

    def process_iter(self) -> Tuple[List[ASRToken], float]:
        """
        Processes the current audio buffer.

        Returns a tuple: (list of committed ASRToken objects, float representing the audio processed up to time).
        """
        current_audio_processed_upto = self.get_audio_buffer_end_time()
        prompt_text, _ = self.prompt()
        logger.debug(
            f"Transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:.2f} seconds from {self.buffer_time_offset:.2f}"
        )
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt_text)
        tokens = self.asr.ts_words(res)
        self.transcript_buffer.insert(tokens, self.buffer_time_offset)
        committed_tokens = self.transcript_buffer.flush()
        self.committed.extend(committed_tokens)

        if committed_tokens:
            self.time_of_last_asr_output = self.committed[-1].end

        completed = self.concatenate_tokens(committed_tokens)
        logger.debug(f">>>> COMPLETE NOW: {completed.text}")
        incomp = self.concatenate_tokens(self.transcript_buffer.buffer)
        logger.debug(f"INCOMPLETE: {incomp.text}")

        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE
        if not committed_tokens and buffer_duration > self.buffer_trimming_sec:
            time_since_last_output = self.get_audio_buffer_end_time() - self.time_of_last_asr_output
            if time_since_last_output > self.buffer_trimming_sec:
                logger.warning(
                    f"No ASR output for {time_since_last_output:.2f}s. "
                    f"Resetting buffer to prevent freezing."
                )
                self.init(offset=self.get_audio_buffer_end_time())
                return [], current_audio_processed_upto

        if committed_tokens and self.buffer_trimming_way == "sentence":
            if len(self.audio_buffer) / self.SAMPLING_RATE > self.buffer_trimming_sec:
                self.chunk_completed_sentence()

        s = self.buffer_trimming_sec if self.buffer_trimming_way == "segment" else 30
        if len(self.audio_buffer) / self.SAMPLING_RATE > s:
            self.chunk_completed_segment(res)
            logger.debug("Chunking segment")
        logger.debug(
            f"Length of audio buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:.2f} seconds"
        )
        return committed_tokens, current_audio_processed_upto

    def chunk_completed_sentence(self):
        """
        If the committed tokens form at least two sentences, chunk the audio
        buffer at the end time of the penultimate sentence.
        Also ensures chunking happens if audio buffer exceeds a time limit.
        """
        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE        
        if not self.committed:
            if buffer_duration > self.buffer_trimming_sec:
                chunk_time = self.buffer_time_offset + (buffer_duration / 2)
                logger.debug(f"--- No speech detected, forced chunking at {chunk_time:.2f}")
                self.chunk_at(chunk_time)
            return
        
        logger.debug("COMPLETED SENTENCE: " + " ".join(token.text for token in self.committed))
        sentences = self.words_to_sentences(self.committed)
        for sentence in sentences:
            logger.debug(f"\tSentence: {sentence.text}")
        
        chunk_done = False
        if len(sentences) >= 2:
            while len(sentences) > 2:
                sentences.pop(0)
            chunk_time = sentences[-2].end
            logger.debug(f"--- Sentence chunked at {chunk_time:.2f}")
            self.chunk_at(chunk_time)
            chunk_done = True
        
        if not chunk_done and buffer_duration > self.buffer_trimming_sec:
            last_committed_time = self.committed[-1].end
            logger.debug(f"--- Not enough sentences, chunking at last committed time {last_committed_time:.2f}")
            self.chunk_at(last_committed_time)

    def chunk_completed_segment(self, res):
        """
        Chunk the audio buffer based on segment-end timestamps reported by the ASR.
        Also ensures chunking happens if audio buffer exceeds a time limit.
        """
        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE        
        if not self.committed:
            if buffer_duration > self.buffer_trimming_sec:
                chunk_time = self.buffer_time_offset + (buffer_duration / 2)
                logger.debug(f"--- No speech detected, forced chunking at {chunk_time:.2f}")
                self.chunk_at(chunk_time)
            return
        
        logger.debug("Processing committed tokens for segmenting")
        ends = self.asr.segments_end_ts(res)
        last_committed_time = self.committed[-1].end        
        chunk_done = False
        if len(ends) > 1:
            logger.debug("Multiple segments available for chunking")
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > last_committed_time:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= last_committed_time:
                logger.debug(f"--- Segment chunked at {e:.2f}")
                self.chunk_at(e)
                chunk_done = True
            else:
                logger.debug("--- Last segment not within committed area")
        else:
            logger.debug("--- Not enough segments to chunk")
        
        if not chunk_done and buffer_duration > self.buffer_trimming_sec:
            logger.debug(f"--- Buffer too large, chunking at last committed time {last_committed_time:.2f}")
            self.chunk_at(last_committed_time)
        
        logger.debug("Segment chunking complete")
        
    def chunk_at(self, time: float):
        """
        Trim both the hypothesis and audio buffer at the given time.
        """
        logger.debug(f"Chunking at {time:.2f}s")
        logger.debug(
            f"Audio buffer length before chunking: {len(self.audio_buffer)/self.SAMPLING_RATE:.2f}s"
        )
        self.transcript_buffer.pop_committed(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.SAMPLING_RATE):]
        self.buffer_time_offset = time
        logger.debug(
            f"Audio buffer length after chunking: {len(self.audio_buffer)/self.SAMPLING_RATE:.2f}s"
        )

    def words_to_sentences(self, tokens: List[ASRToken]) -> List[Sentence]:
        """
        Converts a list of tokens to a list of Sentence objects using the provided
        sentence tokenizer.
        """
        if not tokens:
            return []

        full_text = " ".join(token.text for token in tokens)

        if self.tokenize:
            try:
                sentence_texts = self.tokenize(full_text)
            except Exception as e:
                # Some tokenizers (e.g., MosesSentenceSplitter) expect a list input.
                try:
                    sentence_texts = self.tokenize([full_text])
                except Exception as e2:
                    raise ValueError("Tokenization failed") from e2
        else:
            sentence_texts = [full_text]

        sentences: List[Sentence] = []
        token_index = 0
        for sent_text in sentence_texts:
            sent_text = sent_text.strip()
            if not sent_text:
                continue
            sent_tokens = []
            accumulated = ""
            # Accumulate tokens until roughly matching the length of the sentence text.
            while token_index < len(tokens) and len(accumulated) < len(sent_text):
                token = tokens[token_index]
                accumulated = (accumulated + " " + token.text).strip() if accumulated else token.text
                sent_tokens.append(token)
                token_index += 1
            if sent_tokens:
                sentence = Sentence(
                    start=sent_tokens[0].start,
                    end=sent_tokens[-1].end,
                    text=" ".join(t.text for t in sent_tokens),
                )
                sentences.append(sentence)
        return sentences
    
    def finish(self) -> Tuple[List[ASRToken], float]:
        """
        Flush the remaining transcript when processing ends.
        Returns a tuple: (list of remaining ASRToken objects, float representing the final audio processed up to time).
        """
        remaining_tokens = self.transcript_buffer.buffer
        logger.debug(f"Final non-committed tokens: {remaining_tokens}")
        final_processed_upto = self.buffer_time_offset + (len(self.audio_buffer) / self.SAMPLING_RATE)
        self.buffer_time_offset = final_processed_upto
        return remaining_tokens, final_processed_upto

    def concatenate_tokens(
        self,
        tokens: List[ASRToken],
        sep: Optional[str] = None,
        offset: float = 0
    ) -> Transcript:
        sep = sep if sep is not None else self.asr.sep
        text = sep.join(token.text for token in tokens)
        probability = sum(token.probability for token in tokens if token.probability) / len(tokens) if tokens else None
        if tokens:
            start = offset + tokens[0].start
            end = offset + tokens[-1].end
        else:
            start = None
            end = None
        return Transcript(start, end, text, probability=probability)


class VACOnlineASRProcessor:
    """
    Wraps an OnlineASRProcessor with a Voice Activity Controller (VAC).
    
    It receives small chunks of audio, applies VAD (e.g. with Silero),
    and when the system detects a pause in speech (or end of an utterance)
    it finalizes the utterance immediately.
    """
    SAMPLING_RATE = 16000

    def __init__(self, online_chunk_size: float, *args, **kwargs):
        self.online_chunk_size = online_chunk_size
        self.online = OnlineASRProcessor(*args, **kwargs)
        self.asr = self.online.asr
        
        # Load a VAD model (e.g. Silero VAD)
        import torch
        model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
        from .silero_vad_iterator import FixedVADIterator

        self.vac = FixedVADIterator(model)
        self.logfile = self.online.logfile
        self.last_input_audio_stream_end_time: float = 0.0
        self.init()

    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0
        self.last_input_audio_stream_end_time = self.online.buffer_time_offset
        self.is_currently_final = False
        self.status: Optional[str] = None  # "voice" or "nonvoice"
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0  # in frames

    def get_audio_buffer_end_time(self) -> float:
        """Returns the absolute end time of the audio processed by the underlying OnlineASRProcessor."""
        return self.online.get_audio_buffer_end_time()

    def clear_buffer(self):
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([], dtype=np.float32)

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        """
        Process an incoming small audio chunk:
          - run VAD on the chunk,
          - decide whether to send the audio to the online ASR processor immediately,
          - and/or to mark the current utterance as finished.
        """
        self.last_input_audio_stream_end_time = audio_stream_end_time
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            # VAD returned a result; adjust the frame number
            frame = list(res.values())[0] - self.buffer_offset
            if "start" in res and "end" not in res:
                self.status = "voice"
                send_audio = self.audio_buffer[frame:]
                self.online.init(offset=(frame + self.buffer_offset) / self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.clear_buffer()
            elif "end" in res and "start" not in res:
                self.status = "nonvoice"
                send_audio = self.audio_buffer[:frame]
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
            else:
                beg = res["start"] - self.buffer_offset
                end = res["end"] - self.buffer_offset
                self.status = "nonvoice"
                send_audio = self.audio_buffer[beg:end]
                self.online.init(offset=(beg + self.buffer_offset) / self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
        else:
            if self.status == "voice":
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                # Keep 1 second worth of audio in case VAD later detects voice,
                # but trim to avoid unbounded memory usage.
                self.buffer_offset += max(0, len(self.audio_buffer) - self.SAMPLING_RATE)
                self.audio_buffer = self.audio_buffer[-self.SAMPLING_RATE:]

    def process_iter(self) -> Tuple[List[ASRToken], float]:
        """
        Depending on the VAD status and the amount of accumulated audio,
        process the current audio chunk.
        Returns a tuple: (list of committed ASRToken objects, float representing the audio processed up to time).
        """
        if self.is_currently_final:
            return self.finish()
        elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            return self.online.process_iter()
        else:
            logger.debug("No online update, only VAD")
            return [], self.last_input_audio_stream_end_time

    def finish(self) -> Tuple[List[ASRToken], float]:
        """
        Finish processing by flushing any remaining text.
        Returns a tuple: (list of remaining ASRToken objects, float representing the final audio processed up to time).
        """
        result_tokens, processed_upto = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return result_tokens, processed_upto
    
    def get_buffer(self):
        """
        Get the unvalidated buffer in string format.
        """
        return self.online.concatenate_tokens(self.online.transcript_buffer.buffer)


class SimulStreamingOnlineProcessor:
    SAMPLING_RATE = 16000

    def __init__(
        self,
        asr,
        tokenize_method: Optional[callable] = None,
        buffer_trimming: Tuple[str, float] = ("segment", 15),
        confidence_validation = False,
        logfile=sys.stderr,
    ):
        if not SIMULSTREAMING_AVAILABLE:
            raise ImportError("SimulStreaming dependencies are not available.")
        
        self.asr = asr
        self.tokenize = tokenize_method
        self.logfile = logfile
        self.confidence_validation = confidence_validation
        self.init()
        
        # buffer does not work yet
        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self, offset: Optional[float] = None):
        """Initialize or reset the processing state."""
        self.audio_chunks = []
        self.offset = offset if offset is not None else 0.0
        self.is_last = False
        self.beg = self.offset
        self.end = self.offset
        self.cumulative_audio_duration = 0.0
        self.last_audio_stream_end_time = self.offset
        
        self.committed: List[ASRToken] = []
        self.last_result_tokens: List[ASRToken] = []        
        self.buffer_content = ""
        self.processed_audio_duration = 0.0

    def get_audio_buffer_end_time(self) -> float:
        """Returns the absolute end time of the current audio buffer."""
        return self.end

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: Optional[float] = None):
        """Append an audio chunk to be processed by SimulStreaming."""
        if torch is None:
            raise ImportError("PyTorch is required for SimulStreaming but not available")
            
        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        self.audio_chunks.append(audio_tensor)
        
        # Update timing
        chunk_duration = len(audio) / self.SAMPLING_RATE
        self.cumulative_audio_duration += chunk_duration
        
        if audio_stream_end_time is not None:
            self.last_audio_stream_end_time = audio_stream_end_time
            self.end = audio_stream_end_time
        else:
            self.end = self.offset + self.cumulative_audio_duration

    def prompt(self) -> Tuple[str, str]:
        """
        Returns a tuple: (prompt, context).
        SimulStreaming handles prompting internally, so we return empty strings.
        """
        return "", ""

    def get_buffer(self):
        """
        Get the unvalidated buffer content.
        """
        buffer_end = self.end if hasattr(self, 'end') else None
        return Transcript(
            start=None, 
            end=buffer_end, 
            text=self.buffer_content, 
            probability=None
        )

    def timestamped_text(self, tokens, generation):
        # From the simulstreaming repo. self.model to self.asr.model
        pr = generation["progress"]
        if "result" not in generation:
            split_words, split_tokens = self.asr.model.tokenizer.split_to_word_tokens(tokens)
        else:
            split_words, split_tokens = generation["result"]["split_words"], generation["result"]["split_tokens"]

        frames = [p["most_attended_frames"][0] for p in pr]
        tokens = tokens.copy()
        ret = []
        for sw,st in zip(split_words,split_tokens):
            b = None
            for stt in st:
                t,f = tokens.pop(0), frames.pop(0)
                if t != stt:
                    raise ValueError(f"Token mismatch: {t} != {stt} at frame {f}.")
                if b is None:
                    b = f
            e = f
            out = (b*0.02, e*0.02, sw)
            ret.append(out)
            logger.debug(f"TS-WORD:\t{' '.join(map(str, out))}")
        return ret

    def process_iter(self) -> Tuple[List[ASRToken], float]:
        """
        Process accumulated audio chunks using SimulStreaming.
        
        Returns a tuple: (list of committed ASRToken objects, float representing the audio processed up to time).
        """
        if not self.audio_chunks:
            return [], self.end

        try:
            # concatenate all audio chunks
            if len(self.audio_chunks) == 1:
                audio = self.audio_chunks[0]
            else:
                audio = torch.cat(self.audio_chunks, dim=0)
            
            audio_duration = audio.shape[0] / self.SAMPLING_RATE if audio.shape[0] > 0 else 0
            self.processed_audio_duration += audio_duration
            
            self.audio_chunks = []
            
            logger.debug(f"SimulStreaming processing audio shape: {audio.shape}, duration: {audio_duration:.2f}s")
            logger.debug(f"Current end time: {self.end:.2f}s, last stream time: {self.last_audio_stream_end_time:.2f}s")
            
            self.asr.model.insert_audio(audio)
            tokens, generation_progress = self.asr.model.infer(is_last=self.is_last)
            ts_words = self.timestamped_text(tokens, generation_progress)
            text = self.asr.model.tokenizer.decode(tokens)
            
            new_tokens = []
            for ts_word in ts_words:
                
                start, end, word = ts_word
                token = ASRToken(
                    start=start,
                    end=end,
                    text=word,
                    probability=0.95  # fake prob. Maybe we can extract it from the model?
                )
                new_tokens.append(token)
                self.committed.extend(new_tokens)
            
            return new_tokens, self.end

            
        except Exception as e:
            logger.exception(f"SimulStreaming processing error: {e}")
            return [], self.end

    def finish(self) -> Tuple[List[ASRToken], float]:
        logger.debug("SimulStreaming finish() called")
        self.is_last = True        
        final_tokens, final_time = self.process_iter()
        self.is_last = False
        return final_tokens, final_time

    def concatenate_tokens(
        self,
        tokens: List[ASRToken],
        sep: Optional[str] = None,
        offset: float = 0
    ) -> Transcript:
        """Concatenate tokens into a Transcript object."""
        sep = sep if sep is not None else self.asr.sep
        text = sep.join(token.text for token in tokens)
        probability = sum(token.probability for token in tokens if token.probability) / len(tokens) if tokens else None
        if tokens:
            start = offset + tokens[0].start
            end = offset + tokens[-1].end
        else:
            start = None
            end = None
        return Transcript(start, end, text, probability=probability)

    def chunk_at(self, time: float):
        """
        useless but kept for compatibility
        """
        logger.debug(f"SimulStreaming chunk_at({time:.2f}) - handled internally")
        pass

    def words_to_sentences(self, tokens: List[ASRToken]) -> List[Sentence]:
        """
        Create simple sentences.
        """
        if not tokens:
            return []
        
        full_text = " ".join(token.text for token in tokens)
        sentence = Sentence(
            start=tokens[0].start,
            end=tokens[-1].end,
            text=full_text
        )
        return [sentence]
