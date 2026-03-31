"""Benchmark audio datasets from public HuggingFace repositories.

Downloads curated samples across languages, noise conditions, and speaker
configurations. All datasets are public and freely accessible — no auth
tokens required.

Samples are cached in ~/.cache/whisperlivekit/benchmark_data/ and reused
across benchmark runs.

Datasets used:
    - LibriSpeech test-clean  (English, clean, single speaker)
    - LibriSpeech test-other  (English, noisy/hard, single speaker)
    - Multilingual LibriSpeech (French, Spanish, German, Portuguese, Italian, Polish, Dutch)
    - AMI                      (English, multi-speaker meeting)
"""

import json
import logging
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "whisperlivekit" / "benchmark_data"
METADATA_FILE = "benchmark_metadata.json"


@dataclass
class BenchmarkSample:
    """A benchmark audio sample with metadata and ground truth."""

    name: str
    path: str
    reference: str
    duration: float
    language: str
    category: str  # "clean", "noisy", "multilingual", "meeting"
    sample_rate: int = 16000
    n_speakers: int = 1
    source: str = ""
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "file": Path(self.path).name,
            "reference": self.reference,
            "duration": self.duration,
            "language": self.language,
            "category": self.category,
            "sample_rate": self.sample_rate,
            "n_speakers": self.n_speakers,
            "source": self.source,
            "tags": list(self.tags),
        }


# ---------------------------------------------------------------------------
# Dataset catalog — defines what to download
# ---------------------------------------------------------------------------

BENCHMARK_CATALOG = {
    # English clean (LibriSpeech test-clean)
    "en_clean_short": {
        "dataset": "openslr/librispeech_asr",
        "config": "clean",
        "split": "test",
        "language": "en",
        "category": "clean",
        "n_samples": 1,
        "skip": 0,
        "tags": {"short"},
    },
    "en_clean_medium": {
        "dataset": "openslr/librispeech_asr",
        "config": "clean",
        "split": "test",
        "language": "en",
        "category": "clean",
        "n_samples": 1,
        "skip": 1,
        "tags": {"medium"},
    },
    # English noisy (LibriSpeech test-other)
    "en_noisy_1": {
        "dataset": "openslr/librispeech_asr",
        "config": "other",
        "split": "test",
        "language": "en",
        "category": "noisy",
        "n_samples": 1,
        "skip": 0,
        "tags": {"accented"},
    },
    "en_noisy_2": {
        "dataset": "openslr/librispeech_asr",
        "config": "other",
        "split": "test",
        "language": "en",
        "category": "noisy",
        "n_samples": 1,
        "skip": 1,
        "tags": {"accented"},
    },
    # French (Multilingual LibriSpeech)
    "fr_clean_1": {
        "dataset": "facebook/multilingual_librispeech",
        "config": "french",
        "split": "test",
        "language": "fr",
        "category": "multilingual",
        "n_samples": 1,
        "skip": 0,
        "tags": set(),
    },
    "fr_clean_2": {
        "dataset": "facebook/multilingual_librispeech",
        "config": "french",
        "split": "test",
        "language": "fr",
        "category": "multilingual",
        "n_samples": 1,
        "skip": 1,
        "tags": set(),
    },
    # Spanish (Multilingual LibriSpeech)
    "es_clean_1": {
        "dataset": "facebook/multilingual_librispeech",
        "config": "spanish",
        "split": "test",
        "language": "es",
        "category": "multilingual",
        "n_samples": 1,
        "skip": 0,
        "tags": set(),
    },
    # German (Multilingual LibriSpeech)
    "de_clean_1": {
        "dataset": "facebook/multilingual_librispeech",
        "config": "german",
        "split": "test",
        "language": "de",
        "category": "multilingual",
        "n_samples": 1,
        "skip": 0,
        "tags": set(),
    },
    # Portuguese (Multilingual LibriSpeech)
    "pt_clean_1": {
        "dataset": "facebook/multilingual_librispeech",
        "config": "portuguese",
        "split": "test",
        "language": "pt",
        "category": "multilingual",
        "n_samples": 1,
        "skip": 0,
        "tags": set(),
    },
    # Italian (Multilingual LibriSpeech)
    "it_clean_1": {
        "dataset": "facebook/multilingual_librispeech",
        "config": "italian",
        "split": "test",
        "language": "it",
        "category": "multilingual",
        "n_samples": 1,
        "skip": 0,
        "tags": set(),
    },
    # Polish (Multilingual LibriSpeech)
    "pl_clean_1": {
        "dataset": "facebook/multilingual_librispeech",
        "config": "polish",
        "split": "test",
        "language": "pl",
        "category": "multilingual",
        "n_samples": 1,
        "skip": 0,
        "tags": set(),
    },
    # Dutch (Multilingual LibriSpeech)
    "nl_clean_1": {
        "dataset": "facebook/multilingual_librispeech",
        "config": "dutch",
        "split": "test",
        "language": "nl",
        "category": "multilingual",
        "n_samples": 1,
        "skip": 0,
        "tags": set(),
    },
    # English multi-speaker meeting (AMI)
    "en_meeting": {
        "dataset": "edinburghcstr/ami",
        "config": "ihm",
        "split": "test",
        "language": "en",
        "category": "meeting",
        "n_samples": 1,
        "skip": 0,
        "tags": {"multi_speaker", "long"},
        "max_duration": 60.0,
    },
}

# Quick mode: subset of samples for fast smoke tests
QUICK_SAMPLES = {"en_clean_short", "en_clean_medium", "en_noisy_1", "fr_clean_1"}


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _save_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
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


def _decode_audio(audio_bytes: bytes) -> tuple:
    import io

    import soundfile as sf
    audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    return np.array(audio_array, dtype=np.float32), sr


def _ensure_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for benchmark data. "
            "Install with: pip install whisperlivekit[test]"
        )


# ---------------------------------------------------------------------------
# Download functions per dataset type
# ---------------------------------------------------------------------------

def _download_librispeech(config: str, n_samples: int, skip: int,
                          category: str, language: str,
                          prefix: str) -> List[Dict]:
    """Download from openslr/librispeech_asr (clean or other)."""
    _ensure_datasets()
    import datasets.config
    datasets.config.TORCHCODEC_AVAILABLE = False
    from datasets import Audio, load_dataset

    logger.info("Downloading LibriSpeech %s samples...", config)
    ds = load_dataset(
        "openslr/librispeech_asr", config, split="test", streaming=True,
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    samples = []
    for i, item in enumerate(ds):
        if i < skip:
            continue
        if len(samples) >= n_samples:
            break

        audio_array, sr = _decode_audio(item["audio"]["bytes"])
        duration = len(audio_array) / sr
        text = item["text"]

        wav_name = f"{prefix}_{i}.wav"
        _save_wav(CACHE_DIR / wav_name, audio_array, sr)

        samples.append({
            "file": wav_name,
            "reference": text,
            "duration": round(duration, 2),
            "sample_rate": sr,
            "language": language,
            "category": category,
            "n_speakers": 1,
            "source": f"openslr/librispeech_asr ({config})",
        })
        logger.info("  %.1fs - %s", duration, text[:60])

    return samples


def _download_mls(config: str, n_samples: int, skip: int,
                  language: str, prefix: str) -> List[Dict]:
    """Download from facebook/multilingual_librispeech."""
    _ensure_datasets()
    import datasets.config
    datasets.config.TORCHCODEC_AVAILABLE = False
    from datasets import Audio, load_dataset

    logger.info("Downloading MLS %s samples...", config)
    ds = load_dataset(
        "facebook/multilingual_librispeech", config, split="test", streaming=True,
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    samples = []
    for i, item in enumerate(ds):
        if i < skip:
            continue
        if len(samples) >= n_samples:
            break

        audio_array, sr = _decode_audio(item["audio"]["bytes"])
        duration = len(audio_array) / sr
        text = item.get("text", item.get("transcript", ""))

        wav_name = f"{prefix}_{i}.wav"
        _save_wav(CACHE_DIR / wav_name, audio_array, sr)

        samples.append({
            "file": wav_name,
            "reference": text,
            "duration": round(duration, 2),
            "sample_rate": sr,
            "language": language,
            "category": "multilingual",
            "n_speakers": 1,
            "source": f"facebook/multilingual_librispeech ({config})",
        })
        logger.info("  [%s] %.1fs - %s", language, duration, text[:60])

    return samples


def _download_fleurs(config: str, n_samples: int, skip: int,
                     language: str, prefix: str) -> List[Dict]:
    """Download from google/fleurs."""
    _ensure_datasets()
    import datasets.config
    datasets.config.TORCHCODEC_AVAILABLE = False
    from datasets import Audio, load_dataset

    logger.info("Downloading FLEURS %s samples...", config)
    ds = load_dataset(
        "google/fleurs", config, split="test", streaming=True,
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    samples = []
    for i, item in enumerate(ds):
        if i < skip:
            continue
        if len(samples) >= n_samples:
            break

        audio_array, sr = _decode_audio(item["audio"]["bytes"])
        duration = len(audio_array) / sr
        text = item.get("transcription", item.get("raw_transcription", ""))

        wav_name = f"{prefix}_{i}.wav"
        _save_wav(CACHE_DIR / wav_name, audio_array, sr)

        samples.append({
            "file": wav_name,
            "reference": text,
            "duration": round(duration, 2),
            "sample_rate": sr,
            "language": language,
            "category": "multilingual",
            "n_speakers": 1,
            "source": f"google/fleurs ({config})",
        })
        logger.info("  [%s] %.1fs - %s", language, duration, text[:60])

    return samples


def _download_ami(max_duration: float = 60.0) -> List[Dict]:
    """Download one AMI meeting segment with multiple speakers."""
    _ensure_datasets()
    import datasets.config
    datasets.config.TORCHCODEC_AVAILABLE = False
    from datasets import Audio, load_dataset

    logger.info("Downloading AMI meeting sample...")
    ds = load_dataset("edinburghcstr/ami", "ihm", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    meeting_id = None
    audio_arrays = []
    texts = []
    sample_rate = None

    for item in ds:
        mid = item.get("meeting_id", "unknown")
        if meeting_id is None:
            meeting_id = mid
        elif mid != meeting_id:
            break

        audio_array, sr = _decode_audio(item["audio"]["bytes"])
        sample_rate = sr
        texts.append(item.get("text", ""))
        audio_arrays.append(audio_array)

        total_dur = sum(len(a) / sr for a in audio_arrays)
        if total_dur > max_duration:
            break

    if not audio_arrays:
        return []

    full_audio = np.concatenate(audio_arrays)
    duration = len(full_audio) / sample_rate
    reference = " ".join(t for t in texts if t)

    wav_name = "ami_meeting.wav"
    _save_wav(CACHE_DIR / wav_name, full_audio, sample_rate)

    logger.info("  AMI meeting: %.1fs, %d utterances", duration, len(texts))
    return [{
        "file": wav_name,
        "reference": reference,
        "duration": round(duration, 2),
        "sample_rate": sample_rate,
        "language": "en",
        "category": "meeting",
        "n_speakers": 4,
        "source": f"edinburghcstr/ami (ihm, meeting {meeting_id})",
    }]


# ---------------------------------------------------------------------------
# Dispatcher — routes catalog entries to download functions
# ---------------------------------------------------------------------------

def _download_catalog_entry(name: str, spec: Dict) -> List[Dict]:
    """Download a single catalog entry and return metadata dicts."""
    dataset = spec["dataset"]
    config = spec.get("config", "")
    n_samples = spec.get("n_samples", 1)
    skip = spec.get("skip", 0)
    language = spec["language"]
    category = spec["category"]

    if dataset == "openslr/librispeech_asr":
        return _download_librispeech(
            config=config, n_samples=n_samples, skip=skip,
            category=category, language=language, prefix=name,
        )
    elif dataset == "facebook/multilingual_librispeech":
        return _download_mls(
            config=config, n_samples=n_samples, skip=skip,
            language=language, prefix=name,
        )
    elif dataset == "google/fleurs":
        return _download_fleurs(
            config=config, n_samples=n_samples, skip=skip,
            language=language, prefix=name,
        )
    elif dataset == "edinburghcstr/ami":
        return _download_ami(max_duration=spec.get("max_duration", 60.0))
    else:
        logger.warning("Unknown dataset: %s", dataset)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_benchmark_samples(
    languages: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    quick: bool = False,
    force: bool = False,
) -> List[BenchmarkSample]:
    """Download and return benchmark samples, filtered by language/category.

    Args:
        languages: List of language codes to include (None = all).
        categories: List of categories to include (None = all).
        quick: If True, only download a small subset for smoke tests.
        force: Re-download even if cached.

    Returns:
        List of BenchmarkSample objects ready for benchmarking.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE_DIR / METADATA_FILE

    # Load cached metadata
    cached = {}
    if meta_path.exists() and not force:
        cached = json.loads(meta_path.read_text())

    # Determine which entries to download
    entries = BENCHMARK_CATALOG
    if quick:
        entries = {k: v for k, v in entries.items() if k in QUICK_SAMPLES}

    if languages:
        lang_set = set(languages)
        entries = {k: v for k, v in entries.items() if v["language"] in lang_set}

    if categories:
        cat_set = set(categories)
        entries = {k: v for k, v in entries.items() if v["category"] in cat_set}

    # Download missing entries
    all_meta = cached.get("samples", {})
    for name, spec in entries.items():
        if name in all_meta and not force:
            # Check file exists
            file_path = CACHE_DIR / all_meta[name][0]["file"]
            if file_path.exists():
                continue

        logger.info("Downloading benchmark sample: %s", name)
        try:
            downloaded = _download_catalog_entry(name, spec)
            if downloaded:
                all_meta[name] = downloaded
        except Exception as e:
            logger.warning("Failed to download %s: %s", name, e)

    # Save metadata
    meta_path.write_text(json.dumps({"samples": all_meta}, indent=2))

    # Build BenchmarkSample objects
    samples = []
    for name, spec in entries.items():
        if name not in all_meta:
            continue
        for meta in all_meta[name]:
            file_path = CACHE_DIR / meta["file"]
            if not file_path.exists():
                continue
            catalog_entry = BENCHMARK_CATALOG.get(name, {})
            samples.append(BenchmarkSample(
                name=name,
                path=str(file_path),
                reference=meta["reference"],
                duration=meta["duration"],
                language=meta["language"],
                category=meta["category"],
                sample_rate=meta.get("sample_rate", 16000),
                n_speakers=meta.get("n_speakers", 1),
                source=meta.get("source", ""),
                tags=set(catalog_entry.get("tags", set())),
            ))

    logger.info("Loaded %d benchmark samples", len(samples))
    return samples
