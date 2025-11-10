import sys
import logging
import io
import soundfile as sf
import math
from typing import List
import numpy as np
from whisperlivekit.timed_objects import ASRToken
logger = logging.getLogger(__name__)
class ASRBase:
    sep = " "  # join transcribe words with this character (" " for whisper_timestamped,
              # "" for faster-whisper because it emits the spaces when needed)

    def __init__(self, lan, model_size=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan
        self.model = self.load_model(model_size, cache_dir, model_dir)

    def with_offset(self, offset: float) -> ASRToken:
        # This method is kept for compatibility (typically you will use ASRToken.with_offset)
        return ASRToken(self.start + offset, self.end + offset, self.text)

    def __repr__(self):
        return f"ASRToken(start={self.start:.2f}, end={self.end:.2f}, text={self.text!r})"

    def load_model(self, model_size, cache_dir, model_dir):
        raise NotImplementedError("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplementedError("must be implemented in the child class")

    def use_vad(self):
        raise NotImplementedError("must be implemented in the child class")


class WhisperTimestampedASR(ASRBase):
    """Uses whisper_timestamped as the backend."""
    sep = " "

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        import whisper
        import whisper_timestamped
        from whisper_timestamped import transcribe_timestamped

        self.transcribe_timestamped = transcribe_timestamped
        if model_dir is not None:
            logger.debug("ignoring model_dir, not implemented")
        return whisper.load_model(model_size, download_root=cache_dir)

    def transcribe(self, audio, init_prompt=""):
        result = self.transcribe_timestamped(
            self.model,
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            verbose=None,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return result

    def ts_words(self, r) -> List[ASRToken]:
        """
        Converts the whisper_timestamped result to a list of ASRToken objects.
        """
        tokens = []
        for segment in r["segments"]:
            for word in segment["words"]:
                token = ASRToken(word["start"], word["end"], word["text"])
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [segment["end"] for segment in res["segments"]]

    def use_vad(self):
        self.transcribe_kargs["vad"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class FasterWhisperASR(ASRBase):
    """Uses faster-whisper as the backend."""
    sep = ""

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. "
                         f"model_size and cache_dir parameters are not used.")
            model_size_or_path = model_dir
        elif model_size is not None:
            model_size_or_path = model_size
        else:
            raise ValueError("Either model_size or model_dir must be set")
        device = "auto" # Allow CTranslate2 to decide available device
        compute_type = "auto" # Allow CTranslate2 to decide faster compute type
                              

        model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
        )
        return model

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> list:
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return list(segments)

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.no_speech_prob > 0.9:
                continue
            for word in segment.words:
                token = ASRToken(word.start, word.end, word.word, probability=word.probability)
                tokens.append(token)
        return tokens

    def segments_end_ts(self, segments) -> List[float]:
        return [segment.end for segment in segments]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class MLXWhisper(ASRBase):
    """
    Uses MLX Whisper optimized for Apple Silicon.
    """
    sep = ""

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from mlx_whisper.transcribe import ModelHolder, transcribe
        import mlx.core as mx

        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. model_size parameter is not used.")
            model_size_or_path = model_dir
        elif model_size is not None:
            model_size_or_path = self.translate_model_name(model_size)
            logger.debug(f"Loading whisper model {model_size}. You use mlx whisper, so {model_size_or_path} will be used.")
        else:
            raise ValueError("Either model_size or model_dir must be set")

        self.model_size_or_path = model_size_or_path
        dtype = mx.float16
        ModelHolder.get_model(model_size_or_path, dtype)
        return transcribe

    def translate_model_name(self, model_name):
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx",
        }
        mlx_model_path = model_mapping.get(model_name)
        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized or not supported.")

    def transcribe(self, audio, init_prompt=""):
        if self.transcribe_kargs:
            logger.warning("Transcribe kwargs (vad, task) are not compatible with MLX Whisper and will be ignored.")
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
        )
        return segments.get("segments", [])

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.get("no_speech_prob", 0) > 0.9:
                continue
            for word in segment.get("words", []):
                token = ASRToken(word["start"], word["end"], word["word"], probability=word["probability"])
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s["end"] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class OpenaiApiASR(ASRBase):
    """Uses OpenAI's Whisper API for transcription."""
    def __init__(self, lan=None, temperature=0, logfile=sys.stderr):
        self.logfile = logfile
        self.modelname = "whisper-1"
        self.original_language = None if lan == "auto" else lan
        self.response_format = "verbose_json"
        self.temperature = temperature
        self.load_model()
        self.use_vad_opt = False
        self.task = "transcribe"

    def load_model(self, *args, **kwargs):
        from openai import OpenAI
        self.client = OpenAI()
        self.transcribed_seconds = 0

    def ts_words(self, segments) -> List[ASRToken]:
        """
        Converts OpenAI API response words into ASRToken objects while
        optionally skipping words that fall into no-speech segments.
        """
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in segments.segments:
                if segment.no_speech_prob > 0.8:
                    no_speech_segments.append((segment.start, segment.end))
        tokens = []
        for word in segments.words:
            start = word.start
            end = word.end
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                continue
            tokens.append(ASRToken(start, end, word.word))
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s.end for s in res.words]

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        self.transcribed_seconds += math.ceil(len(audio_data) / 16000)
        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"],
        }
        if self.task != "translate" and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt
        proc = self.client.audio.translations if self.task == "translate" else self.client.audio.transcriptions
        transcript = proc.create(**params)
        logger.debug(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds")
        return transcript

    def use_vad(self):
        self.use_vad_opt = True

    def set_translate_task(self):
        self.task = "translate"