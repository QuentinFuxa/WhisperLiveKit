#!/usr/bin/env python3
"""Smoke-test the NVIDIA Canary-1b-v2 ASR backend — no GPU required.

Loads the Canary backend (NeMo ``EncDecMultiTaskModel``), transcribes a short
16 kHz mono clip, and prints the text plus per-word timestamps. Runs on CPU
(slower, but fully functional) so the backend can be validated without a GPU.
Exits non-zero if the model produces no output.

Model card: https://huggingface.co/nvidia/canary-1b-v2

Usage
-----
    pip install -e ".[canary]"

    # Validate with your own 16 kHz mono clip:
    python scripts/smoke_canary.py path/to/clip.wav

    # Or with no argument, using a bundled LibriSpeech test sample:
    python scripts/smoke_canary.py

The model (~4 GB) downloads from Hugging Face on first run. For an equivalent
in-pipeline check, see the ``TestHarness`` snippet in tests/test_canary_backend.py
(``test_canary_end_to_end_via_testharness``).
"""

import argparse
import sys

import numpy as np


def load_audio(path: str) -> np.ndarray:
    """Load an audio file as a 16 kHz mono float32 numpy array."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return np.asarray(audio, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "audio",
        nargs="?",
        help="16 kHz mono WAV/FLAC path. If omitted, a bundled LibriSpeech sample is used.",
    )
    parser.add_argument(
        "--model",
        default="nvidia/canary-1b-v2",
        help="Canary model id or local .nemo path (default: nvidia/canary-1b-v2).",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Source language code, e.g. en, fr, de (default: en).",
    )
    args = parser.parse_args()

    reference = None
    if args.audio:
        audio = load_audio(args.audio)
    else:
        from whisperlivekit.test_data import get_sample

        sample = get_sample("librispeech_short")
        audio = load_audio(sample.path)
        reference = getattr(sample, "reference", None)

    from whisperlivekit.canary_backend import CanaryASR

    print(f"Loading Canary backend '{args.model}' (CPU is fine; first run downloads ~4 GB)...")
    asr = CanaryASR(lan=args.language, canary_model=args.model)

    res = asr.transcribe(audio, source_lang=args.language)
    tokens = asr.ts_words(res)
    # Canary emits space-prefixed word tokens (sep=""), so a plain join is correct.
    text = "".join(t.text for t in tokens).strip()

    print("\n--- transcription ---")
    print(text or "(empty!)")
    if reference:
        print(f"\n(reference: {reference})")

    print("\n--- word timestamps (first 12) ---")
    for t in tokens[:12]:
        print(f"  [{t.start:6.2f} -> {t.end:6.2f}] {t.text!r}")

    ok = bool(text)
    print(f"\nSMOKE {'PASS' if ok else 'FAIL'}: {len(tokens)} word token(s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
