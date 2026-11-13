#!/usr/bin/env python3
"""Compare sequential and concurrent sessions sharing one candidate ASR model.

Run from the candidate checkout with its installed backend extra. This is a
local model evaluation, not part of the default test suite. Audio and model
paths must already be cached. No latency or memory comparison is made here.
"""

import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from whisperlivekit.benchmark.datasets import load_manifest
from whisperlivekit.benchmark.metrics import get_system_info
from whisperlivekit.config import WhisperLiveKitConfig
from whisperlivekit.core import online_factory
from whisperlivekit.test_harness import load_audio_pcm


def describe(tokens, duration):
    previous = 0.0
    for token in tokens:
        assert math.isfinite(token.start) and math.isfinite(token.end)
        assert 0 <= token.start <= token.end <= duration + 1e-6, token
        assert token.start >= previous, token
        previous = token.start
    return {"text": "".join(token.text for token in tokens),
            "tokens": [{"start": token.start, "end": token.end, "text": token.text}
                       for token in tokens]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["mlx-qwen3-asr", "nemotron-mlx-asr"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    config = WhisperLiveKitConfig(backend=args.backend, lan="en", vac=False,
                                 **json.loads(args.config.read_text()))
    report = {"system": get_system_info(), "config": asdict(config), "sessions": {}, "status": "error"}
    try:
        if args.backend == "mlx-qwen3-asr":
            from whisperlivekit.asr_mlx_qwen3 import MlxQwen3ASR
            shared = MlxQwen3ASR(config)
        else:
            from whisperlivekit.asr_nemotron_mlx import NemotronMLXASR
            shared = NemotronMLXASR(**asdict(config))
        corpus = load_manifest(args.manifest)
        recordings = {}
        for language in ("en", "zh"):
            sample = next(sample for sample in corpus if sample.language == language)
            digest = hashlib.sha256(Path(sample.path).read_bytes()).hexdigest()
            assert digest == sample.expected_sha256, sample.name
            recordings[language] = np.frombuffer(load_audio_pcm(sample.path), dtype="<i2").astype(np.float32) / 32768
            report["sessions"][language] = {"sample": sample.name, "audio_sha256": digest,
                                            "reference": sample.reference}

        chunk_samples = 2397  # Deliberately not aligned with model frame boundaries.
        for language, audio in recordings.items():
            online = online_factory(config, shared, language=language)
            tokens = []
            for offset in range(0, len(audio), chunk_samples):
                chunk = audio[offset:offset + chunk_samples]
                online.insert_audio_chunk(chunk, (offset + len(chunk)) / 16000)
                tokens.extend(online.process_iter()[0])
            tokens.extend(online.finish()[0])
            report["sessions"][language]["sequential"] = describe(tokens, len(audio) / 16000)

        online = {language: online_factory(config, shared, language=language) for language in recordings}
        tokens = {language: [] for language in recordings}
        with ThreadPoolExecutor(max_workers=2) as pool:
            for offset in range(0, max(map(len, recordings.values())), chunk_samples):
                pending = {}
                for language, audio in recordings.items():
                    chunk = audio[offset:offset + chunk_samples]
                    if len(chunk):
                        online[language].insert_audio_chunk(chunk, (offset + len(chunk)) / 16000)
                        pending[language] = pool.submit(online[language].process_iter)
                for language, future in pending.items():
                    tokens[language].extend(future.result()[0])
            pending = {language: pool.submit(session.finish) for language, session in online.items()}
            for language, future in pending.items():
                tokens[language].extend(future.result()[0])
        for language, audio in recordings.items():
            session = report["sessions"][language]
            session["concurrent"] = describe(tokens[language], len(audio) / 16000)
            session["same_text"] = session["concurrent"]["text"] == session["sequential"]["text"]
            assert session["same_text"], f"{language}: shared-model concurrency changed the transcript"
        report["status"] = "ok"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(args.output, report["status"])
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
