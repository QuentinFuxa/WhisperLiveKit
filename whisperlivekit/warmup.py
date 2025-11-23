
import logging

logger = logging.getLogger(__name__)

def load_file(warmup_file=None, timeout=5):
    import os
    import tempfile
    import urllib.request

    import librosa

    if warmup_file == "":
        logger.info(f"Skipping warmup.")
        return None

    # Download JFK sample if not already present
    if warmup_file is None:
        jfk_url = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
        temp_dir = tempfile.gettempdir()
        warmup_file = os.path.join(temp_dir, "whisper_warmup_jfk.wav")
        if not os.path.exists(warmup_file) or os.path.getsize(warmup_file) == 0:
            try:
                logger.debug(f"Downloading warmup file from {jfk_url}")
                with urllib.request.urlopen(jfk_url, timeout=timeout) as r, open(warmup_file, "wb") as f:
                    f.write(r.read())
            except Exception as e:
                logger.warning(f"Warmup file download failed: {e}.")
                return None

    # Validate file and load
    if not os.path.exists(warmup_file) or os.path.getsize(warmup_file) == 0:
        logger.warning(f"Warmup file {warmup_file} is invalid or missing.")
        return None

    try:
        audio, _ = librosa.load(warmup_file, sr=16000)
        return audio
    except Exception as e:
        logger.warning(f"Failed to load warmup file: {e}")
        return None

def warmup_asr(asr, warmup_file=None, timeout=5):
    """
    Warmup the ASR model by transcribing a short audio file.
    """
    audio = load_file(warmup_file=warmup_file, timeout=timeout)
    if audio is None:
        logger.warning("Warmup file unavailable. Skipping ASR warmup.")
        return
    asr.transcribe(audio)
    logger.info("ASR model is warmed up.")