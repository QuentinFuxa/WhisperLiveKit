#!/usr/bin/env python3
"""Freeze 30 parallel FLEURS test utterances per language and 10-minute streams."""

import argparse
import csv
import hashlib
import io
import json
import random
import tarfile
import wave
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
CONFIGS = {"en": "en_us", "fr": "fr_fr", "zh": "cmn_hans_cn"}
CACHE = Path.home() / ".cache/whisperlivekit/benchmark_data/fleurs_v1"


def download(filename):
    return Path(hf_hub_download("google/fleurs", filename, repo_type="dataset", revision=REVISION))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/corpora/fleurs-90.json"))
    args = parser.parse_args()
    metadata, sources = {}, {}
    for language, config in CONFIGS.items():
        tsv = download(f"data/{config}/test.tsv")
        rows = list(csv.reader(io.StringIO(tsv.read_text()), delimiter="\t"))
        by_id = {}
        for row in sorted(rows, key=lambda row: row[1]):
            by_id.setdefault(int(row[0]), row)
        metadata[language] = by_id
        sources[language] = {"config": config, "tsv_sha256": hashlib.sha256(tsv.read_bytes()).hexdigest()}
    common = sorted(set.intersection(*(set(rows) for rows in metadata.values())))
    selected = sorted(random.Random(0).sample(common, 30))
    samples, streams = [], []
    for language, config in CONFIGS.items():
        selected_rows = {metadata[language][idx][1]: metadata[language][idx] for idx in selected}
        paths = {name: CACHE/config/name for name in selected_rows}
        if any(not path.exists() for path in paths.values()):
            print(f"Downloading {config} test audio", flush=True)
            archive = download(f"data/{config}/audio/test.tar.gz")
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar:
                    name = Path(member.name).name
                    if member.isfile() and name in paths:
                        target = paths[name]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(tar.extractfile(member).read())
        language_samples = []
        for idx in selected:
            row = metadata[language][idx]
            original = paths[row[1]]
            values, rate = sf.read(original, dtype="float32")
            assert rate == 16000 and values.ndim == 1
            path = original.with_name(original.stem + ".pcm.wav")
            sf.write(path, values, rate, subtype="PCM_16")
            with wave.open(str(path)) as audio:
                assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
                frames = audio.getnframes()
                assert frames == int(row[5]), (path, frames, row[5])
            sample = {"name": f"fleurs_{language}_{idx}", "file": f"{config}/{path.name}",
                      "reference": row[2], "normalized_reference": row[3], "duration": frames / 16000,
                      "language": language, "category": "fleurs", "sample_rate": 16000,
                      "sentence_id": idx, "recording_id": original.stem, "start_sample": 0, "end_sample": frames,
                      "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "original_audio_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                      "conversion": "soundfile float32 to PCM_16 at original 16 kHz mono",
                      "source": f"google/fleurs@{REVISION}/{config}/test"}
            language_samples.append(sample)
        samples.extend(language_samples)
        # Stress audio has exact component boundaries. The truncated final
        # utterance has no aligned reference, so the stream is not WER-scored.
        pcm, components = bytearray(), []
        i, limit = 0, 600 * 16000 * 2
        while len(pcm) < limit:
            sample = language_samples[i % len(language_samples)]
            with wave.open(str(CACHE/sample["file"])) as audio:
                chunk = audio.readframes(audio.getnframes())
            chunk = chunk[:limit-len(pcm)]
            start = len(pcm) // 2
            pcm.extend(chunk)
            components.append({"sample": sample["name"], "start_sample": start,
                               "end_sample": len(pcm)//2, "source_start_sample": 0,
                               "source_end_sample": len(chunk)//2,
                               "reference": sample["reference"], "truncated": len(chunk)//2 < sample["end_sample"]})
            i += 1
        target = CACHE/config/"continuous_600s.wav"
        with wave.open(str(target), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(pcm)
        streams.append({"name": f"fleurs_{language}_continuous_600s", "file": f"{config}/{target.name}",
                        "reference": "", "duration": 600, "language": language, "category": "continuous",
                        "sample_rate": 16000, "audio_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "source": f"google/fleurs@{REVISION}/{config}/test", "components": components})
        print(f"{language}: {len(language_samples)} clips and 600-second stream ready", flush=True)
    manifest = {"manifest_version": 1, "dataset": "google/fleurs", "revision": REVISION,
                "split": "test", "seed": 0, "selection": "30 shared sentence IDs; first recording by filename",
                "cache_subdir": "fleurs_v1", "sources": sources, "samples": samples, "continuous": streams}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
