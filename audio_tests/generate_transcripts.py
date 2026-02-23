#!/usr/bin/env python3
"""Generate word-level timestamped transcripts using faster-whisper (offline).

Produces one JSON file per audio with: [{word, start, end}, ...]
"""

import json
import os
from faster_whisper import WhisperModel

AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = [
    ("00_00_07_english_1_speaker.wav", "en"),
    ("00_00_16_french_1_speaker.wav", "fr"),
    ("00_00_30_english_3_speakers.wav", "en"),
]

def main():
    print("Loading faster-whisper model (base, cpu, float32)...")
    model = WhisperModel("base", device="cpu", compute_type="float32")

    for filename, lang in FILES:
        audio_path = os.path.join(AUDIO_DIR, filename)
        out_path = os.path.join(
            AUDIO_DIR, filename.rsplit(".", 1)[0] + ".transcript.json"
        )

        print(f"\n{'='*60}")
        print(f"Transcribing: {filename} (language={lang})")
        print(f"{'='*60}")

        segments, info = model.transcribe(
            audio_path, word_timestamps=True, language=lang
        )

        words = []
        for segment in segments:
            if segment.words:
                for w in segment.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                    })
                    print(f"  {w.start:6.2f} - {w.end:6.2f}  {w.word.strip()}")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(words, f, indent=2, ensure_ascii=False)

        print(f"\n  -> {len(words)} words written to {os.path.basename(out_path)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
