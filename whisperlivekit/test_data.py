"""Standard test audio samples for evaluating the WhisperLiveKit pipeline.

Downloads curated samples from public ASR datasets (LibriSpeech, AMI)
and caches them locally. Each sample includes the audio file path,
ground truth transcript, speaker info, and timing metadata.

Usage::

    from whisperlivekit.test_data import get_samples, get_sample

    # Download all standard test samples (first call downloads, then cached)
    samples = get_samples()

    for s in samples:
        print(f"{s.name}: {s.duration:.1f}s, {s.n_speakers} speaker(s)")
        print(f"  Reference: {s.reference[:60]}...")

    # Use with TestHarness
    from whisperlivekit.test_harness import TestHarness

    async with TestHarness(model_size="base", lan="en") as h:
        sample = get_sample("librispeech_short")
        await h.feed(sample.path, speed=0)
        result = await h.finish()
        print(f"WER: {result.wer(sample.reference):.2%}")

Requires: pip install whisperlivekit[test]  (installs 'datasets' and 'librosa')
"""

import json
import logging
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "whisperlivekit" / "test_data"
METADATA_FILE = "metadata.json"


@dataclass
class TestSample:
    """A test audio sample with ground truth metadata."""

    name: str
    path: str  # absolute path to WAV file
    reference: str  # ground truth transcript
    duration: float  # audio duration in seconds
    sample_rate: int = 16000
    n_speakers: int = 1
    language: str = "en"
    source: str = ""  # dataset name
    # Per-utterance ground truth for multi-speaker: [(start, end, speaker, text), ...]
    utterances: List[Dict] = field(default_factory=list)

    @property
    def has_timestamps(self) -> bool:
        return len(self.utterances) > 0


def _save_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    """Save numpy audio array as 16-bit PCM WAV."""
    # Ensure mono
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    # Normalize to int16 range
    if audio.dtype in (np.float32, np.float64):
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    elif audio.dtype != np.int16:
        audio = audio.astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def _load_metadata() -> Dict:
    """Load cached metadata if it exists."""
    meta_path = CACHE_DIR / METADATA_FILE
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def _save_metadata(meta: Dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / METADATA_FILE).write_text(json.dumps(meta, indent=2))


def _ensure_datasets():
    """Check that the datasets library is available."""
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for test data download. "
            "Install it with: pip install whisperlivekit[test]"
        )


def _decode_audio(audio_bytes: bytes) -> tuple:
    """Decode audio bytes using soundfile (avoids torchcodec dependency).

    Returns:
        (audio_array, sample_rate) — float32 numpy array and int sample rate.
    """
    import io

    import soundfile as sf
    audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    return np.array(audio_array, dtype=np.float32), sr


# ---------------------------------------------------------------------------
# Dataset-specific download functions
# ---------------------------------------------------------------------------

def _download_librispeech_samples(n_samples: int = 3) -> List[Dict]:
    """Download short samples from LibriSpeech test-clean."""
    _ensure_datasets()
    import datasets.config
    datasets.config.TORCHCODEC_AVAILABLE = False
    from datasets import Audio, load_dataset

    logger.info("Downloading LibriSpeech test-clean samples (streaming)...")
    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
        streaming=True,
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    samples = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break

        audio_array, sr = _decode_audio(item["audio"]["bytes"])
        duration = len(audio_array) / sr
        text = item["text"]
        sample_id = item.get("id", f"librispeech_{i}")

        # Save WAV
        wav_name = f"librispeech_{i}.wav"
        wav_path = CACHE_DIR / wav_name
        _save_wav(wav_path, audio_array, sr)

        # Name: first sample is "librispeech_short", rest are numbered
        name = "librispeech_short" if i == 0 else f"librispeech_{i}"

        samples.append({
            "name": name,
            "file": wav_name,
            "reference": text,
            "duration": round(duration, 2),
            "sample_rate": sr,
            "n_speakers": 1,
            "language": "en",
            "source": "openslr/librispeech_asr (test-clean)",
            "source_id": str(sample_id),
            "utterances": [],
        })
        logger.info(
            "  [%d] %.1fs - %s",
            i, duration, text[:60] + ("..." if len(text) > 60 else ""),
        )

    return samples


def _download_ami_sample() -> List[Dict]:
    """Download one AMI meeting segment with multiple speakers."""
    _ensure_datasets()
    import datasets.config
    datasets.config.TORCHCODEC_AVAILABLE = False
    from datasets import Audio, load_dataset

    logger.info("Downloading AMI meeting test sample (streaming)...")

    # Use the edinburghcstr/ami version which has pre-segmented utterances
    # with speaker_id, begin_time, end_time, text
    ds = load_dataset(
        "edinburghcstr/ami",
        "ihm",
        split="test",
        streaming=True,
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    # Collect utterances from one meeting
    meeting_utterances = []
    meeting_id = None
    audio_arrays = []
    sample_rate = None

    for item in ds:
        mid = item.get("meeting_id", "unknown")

        # Take the first meeting only
        if meeting_id is None:
            meeting_id = mid
        elif mid != meeting_id:
            # We've moved to a different meeting, stop
            break

        audio_array, sr = _decode_audio(item["audio"]["bytes"])
        sample_rate = sr

        meeting_utterances.append({
            "start": round(item.get("begin_time", 0.0), 2),
            "end": round(item.get("end_time", 0.0), 2),
            "speaker": item.get("speaker_id", "unknown"),
            "text": item.get("text", ""),
        })
        audio_arrays.append(audio_array)

        # Limit to reasonable size (~60s of utterances)
        total_dur = sum(u["end"] - u["start"] for u in meeting_utterances)
        if total_dur > 60:
            break

    if not audio_arrays:
        logger.warning("No AMI samples found")
        return []

    # Concatenate all utterance audio
    full_audio = np.concatenate(audio_arrays)
    duration = len(full_audio) / sample_rate

    # Build reference text
    speakers = set(u["speaker"] for u in meeting_utterances)
    reference = " ".join(u["text"] for u in meeting_utterances if u["text"])

    wav_name = "ami_meeting.wav"
    wav_path = CACHE_DIR / wav_name
    _save_wav(wav_path, full_audio, sample_rate)

    logger.info(
        "  AMI meeting %s: %.1fs, %d speakers, %d utterances",
        meeting_id, duration, len(speakers), len(meeting_utterances),
    )

    return [{
        "name": "ami_meeting",
        "file": wav_name,
        "reference": reference,
        "duration": round(duration, 2),
        "sample_rate": sample_rate,
        "n_speakers": len(speakers),
        "language": "en",
        "source": f"edinburghcstr/ami (ihm, meeting {meeting_id})",
        "source_id": meeting_id,
        "utterances": meeting_utterances,
    }]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_test_samples(force: bool = False) -> List[TestSample]:
    """Download standard test audio samples.

    Downloads samples from LibriSpeech (clean single-speaker) and
    AMI (multi-speaker meetings) on first call. Subsequent calls
    return cached data.

    Args:
        force: Re-download even if cached.

    Returns:
        List of TestSample objects ready for use with TestHarness.
    """
    meta = _load_metadata()

    if meta.get("samples") and not force:
        # Check all files still exist
        all_exist = all(
            (CACHE_DIR / s["file"]).exists()
            for s in meta["samples"]
        )
        if all_exist:
            return _meta_to_samples(meta["samples"])

    logger.info("Downloading test samples to %s ...", CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_samples = []

    try:
        all_samples.extend(_download_librispeech_samples(n_samples=3))
    except Exception as e:
        logger.warning("Failed to download LibriSpeech samples: %s", e)

    try:
        all_samples.extend(_download_ami_sample())
    except Exception as e:
        logger.warning("Failed to download AMI sample: %s", e)

    if not all_samples:
        raise RuntimeError(
            "Failed to download any test samples. "
            "Check your internet connection and ensure 'datasets' is installed: "
            "pip install whisperlivekit[test]"
        )

    _save_metadata({"samples": all_samples})
    logger.info("Downloaded %d test samples to %s", len(all_samples), CACHE_DIR)

    return _meta_to_samples(all_samples)


def get_samples() -> List[TestSample]:
    """Get standard test samples (downloads on first call)."""
    return download_test_samples()


def get_sample(name: str) -> TestSample:
    """Get a specific test sample by name.

    Available names: 'librispeech_short', 'librispeech_1', 'librispeech_2',
    'ami_meeting'.

    Raises:
        KeyError: If the sample name is not found.
    """
    samples = get_samples()
    for s in samples:
        if s.name == name:
            return s
    available = [s.name for s in samples]
    raise KeyError(f"Sample '{name}' not found. Available: {available}")


def list_sample_names() -> List[str]:
    """List names of available test samples (downloads if needed)."""
    return [s.name for s in get_samples()]


def _meta_to_samples(meta_list: List[Dict]) -> List[TestSample]:
    """Convert metadata dicts to TestSample objects."""
    samples = []
    for m in meta_list:
        samples.append(TestSample(
            name=m["name"],
            path=str(CACHE_DIR / m["file"]),
            reference=m["reference"],
            duration=m["duration"],
            sample_rate=m.get("sample_rate", 16000),
            n_speakers=m.get("n_speakers", 1),
            language=m.get("language", "en"),
            source=m.get("source", ""),
            utterances=m.get("utterances", []),
        ))
    return samples
