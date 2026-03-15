#!/usr/bin/env python3
"""Create long benchmark samples (5min+) by concatenating utterances from public datasets."""

import io
import json
import logging
import wave
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CACHE = Path.home() / ".cache/whisperlivekit/benchmark_data"
CACHE.mkdir(parents=True, exist_ok=True)
SR = 16000


def save_wav(path, audio, sr=SR):
    audio = np.clip(audio, -1, 1)
    audio_int = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int.tobytes())


def decode_audio(audio_bytes):
    import soundfile as sf
    arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    return np.array(arr, dtype=np.float32), sr


def download_long_librispeech(config, lang_code, target_dur=300):
    """Concatenate LibriSpeech utterances into a ~5min sample."""
    import datasets.config
    datasets.config.TORCHCODEC_AVAILABLE = False
    from datasets import Audio, load_dataset

    logger.info(f"Downloading LibriSpeech {config} for {lang_code} (~{target_dur}s)...")
    ds = load_dataset("openslr/librispeech_asr", config, split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    chunks, texts = [], []
    total = 0
    for item in ds:
        arr, sr = decode_audio(item["audio"]["bytes"])
        chunks.append(arr)
        texts.append(item["text"])
        total += len(arr) / sr
        if total >= target_dur:
            break
        if len(chunks) % 20 == 0:
            logger.info(f"  {total:.0f}s / {target_dur}s ({len(chunks)} utterances)")

    # Insert small silences between utterances for natural transitions
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    interleaved = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            interleaved.append(silence)
        interleaved.append(chunk)
    full = np.concatenate(interleaved)
    total = len(full) / sr
    ref = " ".join(texts)
    name = f"{lang_code}_long_{config}"
    path = CACHE / f"{name}.wav"
    save_wav(path, full)
    logger.info(f"  -> {name}: {total:.1f}s ({len(texts)} utterances)")
    return {"name": name, "path": str(path), "reference": ref,
            "duration": round(total, 2), "language": lang_code.split("_")[0]}


def download_long_mls(config, lang_code, target_dur=300):
    """Concatenate MLS utterances into a ~5min sample."""
    import datasets.config
    datasets.config.TORCHCODEC_AVAILABLE = False
    from datasets import Audio, load_dataset

    logger.info(f"Downloading MLS {config} for {lang_code} (~{target_dur}s)...")
    ds = load_dataset("facebook/multilingual_librispeech", config, split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    chunks, texts = [], []
    total = 0
    for item in ds:
        arr, sr = decode_audio(item["audio"]["bytes"])
        chunks.append(arr)
        texts.append(item.get("text", item.get("transcript", "")))
        total += len(arr) / sr
        if total >= target_dur:
            break
        if len(chunks) % 20 == 0:
            logger.info(f"  {total:.0f}s / {target_dur}s ({len(chunks)} utterances)")

    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    interleaved = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            interleaved.append(silence)
        interleaved.append(chunk)
    full = np.concatenate(interleaved)
    total = len(full) / sr
    ref = " ".join(texts)
    name = f"{lang_code}_long"
    path = CACHE / f"{name}.wav"
    save_wav(path, full)
    logger.info(f"  -> {name}: {total:.1f}s ({len(texts)} utterances)")
    return {"name": name, "path": str(path), "reference": ref,
            "duration": round(total, 2), "language": lang_code}


def main():
    samples = []

    # English clean ~90s
    samples.append(download_long_librispeech("clean", "en", target_dur=90))

    # English noisy ~90s
    samples.append(download_long_librispeech("other", "en_noisy", target_dur=90))

    # French ~90s
    samples.append(download_long_mls("french", "fr", target_dur=90))

    # Save metadata
    meta_path = CACHE / "long_samples.json"
    meta_path.write_text(json.dumps(samples, indent=2))
    logger.info(f"\nSaved metadata to {meta_path}")

    total = sum(s["duration"] for s in samples)
    logger.info(f"Total: {len(samples)} long samples, {total:.0f}s ({total/60:.1f}min)")


if __name__ == "__main__":
    main()
