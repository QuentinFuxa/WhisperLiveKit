#!/usr/bin/env python3
"""Build a human-reference manifest from the MCIF long-form dataset.

The MCIF layout pairs ``audio-segments.yaml`` (ordered segment list with
``wav``/``offset``/``duration``) with ``ref/<lang>.txt`` (one reference line
per segment, same order). This script groups segments per wav file and emits:

- ``manifest.human.jsonl``: one row per wav with the concatenated full-talk
  reference plus the per-segment breakdown (kept so time-windowed references,
  e.g. "first 20s", can be derived later).

Reference lines carry CSV-style quoting artifacts (outer quotes, doubled
inner quotes); they are unescaped here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def unquote_csv_line(line: str) -> str:
    line = line.strip()
    if len(line) >= 2 and line.startswith('"') and line.endswith('"'):
        line = line[1:-1]
    return line.replace('""', '"').strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mcif-dir",
        type=Path,
        default=Path.home() / "Downloads" / "mcif-long-trans",
        help="Directory containing audio-segments.yaml, ref/, audio/.",
    )
    parser.add_argument("--language", default="en", help="Reference language file stem under ref/.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "mcif_refs",
    )
    args = parser.parse_args()

    segments = yaml.safe_load((args.mcif_dir / "audio-segments.yaml").read_text())
    ref_lines = (args.mcif_dir / "ref" / f"{args.language}.txt").read_text().splitlines()
    if len(segments) != len(ref_lines):
        raise SystemExit(
            f"segment/reference count mismatch: {len(segments)} yaml entries "
            f"vs {len(ref_lines)} reference lines"
        )

    per_wav: dict[str, list[dict]] = {}
    for entry, raw_line in zip(segments, ref_lines):
        text = unquote_csv_line(raw_line)
        per_wav.setdefault(entry["wav"], []).append(
            {
                "offset_sec": float(entry["offset"]),
                "duration_sec": float(entry["duration"]),
                "speaker_id": entry.get("speaker_id"),
                "text": text,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.human.jsonl"
    with manifest_path.open("w") as f:
        for wav, segs in per_wav.items():
            offsets = [s["offset_sec"] for s in segs]
            if offsets != sorted(offsets):
                raise SystemExit(f"non-monotonic segment offsets for {wav}")
            row = {
                "audio_id": Path(wav).stem,
                "wav": wav,
                "language": args.language,
                "human_text": " ".join(s["text"] for s in segs if s["text"]),
                "n_segments": len(segs),
                "segments": segs,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "wavs": len(per_wav),
                "segments": len(segments),
                "language": args.language,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
