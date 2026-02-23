#!/usr/bin/env python3
"""
Offline test harness and benchmark suite for WhisperLiveKit backends.

Simulates a client-server session by feeding audio files as PCM bytes through
the full AudioProcessor pipeline (the same path used by the WebSocket server),
without needing a browser or microphone.

Computes WER (Word Error Rate) and timestamp accuracy when ground truth
transcript files (.transcript.json) are available alongside audio files.

Usage:
    # Test with a single audio file:
    python test_backend_offline.py --backend faster-whisper --audio audio_tests/00_00_07_english_1_speaker.wav

    # Test all files in audio_tests/:
    python test_backend_offline.py --backend faster-whisper --no-realtime

    # Override streaming policy:
    python test_backend_offline.py --backend faster-whisper --policy simulstreaming --no-realtime

    # Multi-backend benchmark (auto-detects all installed backends):
    python test_backend_offline.py --benchmark --no-realtime

    # Export results as JSON:
    python test_backend_offline.py --benchmark --no-realtime --json results.json

    # Insert silence for testing silence handling:
    python test_backend_offline.py --backend faster-whisper --insert-silence 3.0 2.0
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("test_offline")
logger.setLevel(logging.INFO)

SAMPLE_RATE = 16000
JFK_WAV_URL = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
CACHE_DIR = Path(__file__).parent / ".test_cache"
AUDIO_TESTS_DIR = Path(__file__).parent / "audio_tests"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass
class WordTimestamp:
    """Word with its start/end time."""
    word: str
    start: float
    end: float


@dataclass
class TestResult:
    """Structured result from a single test run."""
    audio_file: str
    audio_duration_s: float
    backend: str
    policy: str
    language: str
    chunk_ms: int
    realtime_pacing: bool
    # Timing
    processing_time_s: float
    rtf: float  # real-time factor
    # Transcription output
    transcription: str
    n_lines: int
    n_responses: int
    # WER metrics (None if no ground truth)
    wer: Optional[float] = None
    wer_details: Optional[dict] = None
    # Timestamp accuracy (None if no ground truth)
    timestamp_mae: Optional[float] = None
    timestamp_max_delta: Optional[float] = None
    timestamp_median_delta: Optional[float] = None
    # Word-level timestamps
    word_timestamps: List[WordTimestamp] = field(default_factory=list)
    # Raw last response
    last_response: Optional[dict] = None


def download_sample_audio() -> Path:
    """Download the jfk.wav sample if not cached."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / "jfk.wav"
    if not path.exists():
        logger.info(f"Downloading sample audio to {path} ...")
        urllib.request.urlretrieve(JFK_WAV_URL, path)
        logger.info("Done.")
    return path


def load_audio(path: str) -> np.ndarray:
    """Load audio file as float32 mono 16kHz numpy array.

    Supports WAV, FLAC (via soundfile) and MP3, OGG, M4A (via librosa).
    """
    ext = Path(path).suffix.lower()
    if ext in (".mp3", ".ogg", ".m4a"):
        import librosa
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return audio.astype(np.float32)

    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def insert_silence(audio: np.ndarray, silence_sec: float, position_sec: float) -> np.ndarray:
    """Insert silence into audio at a given position.

    Args:
        audio: Float32 mono audio array at SAMPLE_RATE.
        silence_sec: Duration of silence to insert in seconds.
        position_sec: Position in seconds where silence starts.
    Returns:
        New audio array with silence inserted.
    """
    pos_samples = int(position_sec * SAMPLE_RATE)
    silence_samples = int(silence_sec * SAMPLE_RATE)
    pos_samples = min(pos_samples, len(audio))
    silence = np.zeros(silence_samples, dtype=np.float32)
    return np.concatenate([audio[:pos_samples], silence, audio[pos_samples:]])


def float32_to_s16le_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 audio to s16le PCM bytes (what the browser sends)."""
    return (audio * 32768).clip(-32768, 32767).astype(np.int16).tobytes()


def create_engine(
    backend: str, model_size: str, lan: str,
    diarization: bool = False, vac: bool = True, policy: str = "",
):
    """Create a TranscriptionEngine with the given backend config."""
    import gc
    from whisperlivekit.core import TranscriptionEngine

    # Reset singleton so we get a fresh instance
    TranscriptionEngine._instance = None
    TranscriptionEngine._initialized = False
    gc.collect()

    kwargs = dict(
        backend=backend,
        lan=lan,
        pcm_input=True,
        vac=vac,
        transcription=True,
        diarization=diarization,
    )
    if model_size:
        kwargs["model_size"] = model_size
    if policy:
        kwargs["backend_policy"] = policy

    return TranscriptionEngine(**kwargs)


def _extract_text_from_response(response_dict: dict) -> str:
    """Extract full transcription text from a FrontData dict."""
    segments = response_dict.get("lines", [])
    full_text = " ".join(
        seg.get("text", "").strip()
        for seg in segments
        if seg.get("text", "").strip()
    )
    buf = response_dict.get("buffer_transcription", "").strip()
    if buf:
        full_text = f"{full_text} {buf}".strip() if full_text else buf
    return full_text


async def run_test(
    engine, audio: np.ndarray, chunk_ms: int, realtime: bool,
    audio_file: str = "", backend: str = "", policy: str = "", lan: str = "",
) -> TestResult:
    """
    Simulate a client session through the full AudioProcessor pipeline.

    1. Create AudioProcessor (one per "client session")
    2. Start async pipeline (transcription_processor, results_formatter, etc.)
    3. Feed audio as PCM bytes in timed chunks
    4. Collect and display FrontData responses
    5. Signal EOF and cleanup
    """
    from whisperlivekit.audio_processor import AudioProcessor

    chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
    total_samples = len(audio)
    audio_duration = total_samples / SAMPLE_RATE

    logger.info(
        f"Audio: {audio_duration:.2f}s | "
        f"Chunk: {chunk_ms}ms ({chunk_samples} samples) | "
        f"Steps: {total_samples // chunk_samples + 1} | "
        f"Realtime: {realtime}"
    )

    # --- Server side: create processor and start pipeline ---
    processor = AudioProcessor(transcription_engine=engine)
    results_generator = await processor.create_tasks()

    # Collect results in background (like handle_websocket_results)
    all_responses = []
    response_count = 0
    last_printed_text = ""

    async def collect_results():
        nonlocal response_count, last_printed_text
        async for response in results_generator:
            all_responses.append(response)
            response_count += 1
            d = response.to_dict()

            # Only print when transcription text actually changes
            current_text = _extract_text_from_response(d)
            if current_text and current_text != last_printed_text:
                buf = d.get("buffer_transcription", "").strip()
                committed = current_text
                if buf and committed.endswith(buf):
                    committed = committed[:-len(buf)].strip()

                # Show committed text + buffer separately
                display = committed
                if buf:
                    display = f"{committed} \033[90m{buf}\033[0m" if committed else f"\033[90m{buf}\033[0m"
                print(f"  > {display}", flush=True)
                last_printed_text = current_text

    result_task = asyncio.create_task(collect_results())

    # --- Client side: feed audio as PCM bytes ---
    t_start = time.time()

    for offset in range(0, total_samples, chunk_samples):
        chunk = audio[offset : offset + chunk_samples]
        pcm_bytes = float32_to_s16le_bytes(chunk)
        await processor.process_audio(pcm_bytes)
        if realtime:
            await asyncio.sleep(chunk_ms / 1000)

    feed_elapsed = time.time() - t_start

    logger.info(f"Audio fed in {feed_elapsed:.2f}s. Signaling EOF...")

    # Signal end of audio (like client disconnect / empty message)
    await processor.process_audio(None)

    # Wait for pipeline to drain completely
    try:
        await asyncio.wait_for(result_task, timeout=120.0)
    except asyncio.TimeoutError:
        logger.warning("Timed out waiting for results. Proceeding with cleanup.")
        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    # --- Capture word-level timestamps before cleanup ---
    word_timestamps = []
    try:
        state = await processor.get_current_state()
        for token in state.tokens:
            if hasattr(token, 'start') and hasattr(token, 'text') and token.text:
                word_timestamps.append(WordTimestamp(
                    word=token.text.strip(),
                    start=round(token.start, 3),
                    end=round(token.end, 3),
                ))
    except Exception as e:
        logger.warning(f"Could not capture word timestamps: {e}")

    # Cleanup
    await processor.cleanup()

    total_elapsed = time.time() - t_start

    # --- Build result ---
    transcription = ""
    n_lines = 0
    last_response_dict = None

    if all_responses:
        last = all_responses[-1].to_dict()
        last_response_dict = last
        n_lines = len(last.get("lines", []))
        transcription = _extract_text_from_response(last)

    # --- Compute WER and timestamp accuracy against ground truth ---
    from whisperlivekit.metrics import compute_wer, compute_timestamp_accuracy

    wer_val = None
    wer_details = None
    ts_mae = None
    ts_max_delta = None
    ts_median_delta = None

    gt_path = Path(audio_file).with_suffix(".transcript.json")
    if not gt_path.exists():
        gt_path = AUDIO_TESTS_DIR / gt_path
    gt = None
    if gt_path.exists():
        with open(gt_path) as f:
            gt = json.load(f)

        # WER
        gt_text = " ".join(w["word"] for w in gt)
        wer_result = compute_wer(gt_text, transcription)
        wer_val = round(wer_result["wer"], 4)
        wer_details = wer_result

        # Timestamp accuracy
        if word_timestamps:
            pred_dicts = [{"word": wt.word, "start": wt.start, "end": wt.end} for wt in word_timestamps]
            ts_result = compute_timestamp_accuracy(pred_dicts, gt)
            ts_mae = ts_result["mae_start"]
            ts_max_delta = ts_result["max_delta_start"]
            ts_median_delta = ts_result["median_delta_start"]

    result = TestResult(
        audio_file=audio_file,
        audio_duration_s=round(audio_duration, 2),
        backend=backend,
        policy=policy,
        language=lan,
        chunk_ms=chunk_ms,
        realtime_pacing=realtime,
        processing_time_s=round(total_elapsed, 2),
        rtf=round(total_elapsed / audio_duration, 2),
        transcription=transcription,
        n_lines=n_lines,
        n_responses=response_count,
        wer=wer_val,
        wer_details=wer_details,
        timestamp_mae=round(ts_mae, 3) if ts_mae is not None else None,
        timestamp_max_delta=round(ts_max_delta, 3) if ts_max_delta is not None else None,
        timestamp_median_delta=round(ts_median_delta, 3) if ts_median_delta is not None else None,
        word_timestamps=word_timestamps,
        last_response=last_response_dict,
    )

    # --- Print summary ---
    print(f"\n{'=' * 60}")
    print(f"RESULT: {audio_file}")
    print(f"{'=' * 60}")
    print(f"Transcription: {transcription}")
    print(f"Lines: {n_lines} | Responses: {response_count}")
    print(f"Audio: {audio_duration:.2f}s | Time: {total_elapsed:.2f}s | RTF: {result.rtf:.2f}x")

    if wer_val is not None:
        print(f"WER: {wer_val:.2%} (S={wer_details['substitutions']} I={wer_details['insertions']} D={wer_details['deletions']})")

    # Print word timestamps if available
    if word_timestamps:
        print(f"\nWord timestamps ({len(word_timestamps)} words):")
        for wt in word_timestamps:
            print(f"  [{wt.start:6.2f} - {wt.end:6.2f}] {wt.word}")

        # Detailed comparison with ground truth
        if gt:
            print(f"\n  vs Ground truth ({len(gt)} words):")
            max_words = max(len(word_timestamps), len(gt))
            for i in range(max_words):
                pred = word_timestamps[i] if i < len(word_timestamps) else None
                ref = gt[i] if i < len(gt) else None
                p_str = f"[{pred.start:5.2f}-{pred.end:5.2f}] {pred.word:<15}" if pred else " " * 30
                r_str = f"[{ref['start']:5.2f}-{ref['end']:5.2f}] {ref['word']:<15}" if ref else ""
                delta = ""
                if pred and ref:
                    d = pred.start - ref['start']
                    delta = f"  Δstart={d:+.2f}"
                print(f"  {p_str}  |  {r_str}{delta}")

        if ts_mae is not None:
            print(f"\n  Timestamp stats: MAE={ts_mae:.3f}s  max|Δ|={ts_max_delta:.3f}s  median|Δ|={ts_median_delta:.3f}s")

    print(f"{'=' * 60}")

    return result


def discover_audio_files(directory: str) -> List[Path]:
    """Find all supported audio files in directory."""
    d = Path(directory)
    files = sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    return files


async def run_all_tests(
    engine, audio_files: List[Path], chunk_ms: int, realtime: bool,
    backend: str, policy: str, lan: str, max_duration: float = 60.0,
    silence_insertions: Optional[List[List[float]]] = None,
) -> List[TestResult]:
    """Run tests on multiple audio files sequentially."""
    results = []
    for audio_path in audio_files:
        # Detect language from filename if "french" in name
        file_lan = lan
        if "french" in audio_path.name.lower() and lan == "en":
            file_lan = "fr"
            logger.info(f"Auto-detected language 'fr' from filename")

        audio = load_audio(str(audio_path))

        # Insert silence segments (applied in reverse position order to keep offsets valid)
        if silence_insertions:
            for secs, at_sec in sorted(silence_insertions, key=lambda x: x[1], reverse=True):
                logger.info(f"Inserting {secs:.1f}s silence at {at_sec:.1f}s")
                audio = insert_silence(audio, secs, at_sec)

        duration = len(audio) / SAMPLE_RATE

        if duration > max_duration:
            logger.info(f"Skipping {audio_path.name} ({duration:.0f}s > {max_duration:.0f}s max)")
            continue

        print(f"\n{'#' * 60}")
        print(f"# Testing: {audio_path.name} ({duration:.1f}s)")
        print(f"{'#' * 60}")

        result = await run_test(
            engine, audio, chunk_ms, realtime,
            audio_file=audio_path.name, backend=backend, policy=policy, lan=file_lan,
        )
        results.append(result)

    return results


def print_benchmark_summary(results: List[TestResult]):
    """Print a tabular summary of all test results."""
    print(f"\n{'=' * 110}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 110}")
    print(
        f"{'File':<40} {'Duration':>8} {'Time':>8} {'RTF':>6} "
        f"{'WER':>7} {'MAE(s)':>7} {'Lines':>5}"
    )
    print(f"{'-' * 110}")
    for r in results:
        wer_str = f"{r.wer:.2%}" if r.wer is not None else "  -"
        mae_str = f"{r.timestamp_mae:.3f}" if r.timestamp_mae is not None else "  -"
        print(
            f"{r.audio_file:<40} {r.audio_duration_s:>7.1f}s {r.processing_time_s:>7.1f}s "
            f"{r.rtf:>5.2f}x {wer_str:>7} {mae_str:>7} {r.n_lines:>5}"
        )
    print(f"{'-' * 110}")
    total_audio = sum(r.audio_duration_s for r in results)
    total_time = sum(r.processing_time_s for r in results)
    avg_rtf = total_time / total_audio if total_audio > 0 else 0
    wer_vals = [r.wer for r in results if r.wer is not None]
    avg_wer_str = f"{sum(wer_vals)/len(wer_vals):.2%}" if wer_vals else "  -"
    mae_vals = [r.timestamp_mae for r in results if r.timestamp_mae is not None]
    avg_mae_str = f"{sum(mae_vals)/len(mae_vals):.3f}" if mae_vals else "  -"
    print(
        f"{'TOTAL/AVG':<40} {total_audio:>7.1f}s {total_time:>7.1f}s "
        f"{avg_rtf:>5.2f}x {avg_wer_str:>7} {avg_mae_str:>7}"
    )
    print(f"{'=' * 110}")

    # Print transcription excerpts
    print(f"\nTRANSCRIPTIONS:")
    print(f"{'-' * 110}")
    for r in results:
        excerpt = r.transcription[:120] + "..." if len(r.transcription) > 120 else r.transcription
        print(f"  {r.audio_file}:")
        print(f"    {excerpt}")
    print(f"{'=' * 110}")


def detect_available_backends() -> List[dict]:
    """Probe which backends can be imported and return (backend, policy) combos.

    Returns list of dicts with keys: backend, policy, description.
    """
    combos = []

    # faster-whisper
    try:
        import faster_whisper  # noqa: F401
        combos.append({"backend": "faster-whisper", "policy": "localagreement", "description": "faster-whisper + LocalAgreement"})
        combos.append({"backend": "faster-whisper", "policy": "simulstreaming", "description": "faster-whisper + SimulStreaming"})
    except ImportError:
        pass

    # mlx-whisper (macOS only)
    try:
        import mlx_whisper  # noqa: F401
        combos.append({"backend": "mlx-whisper", "policy": "localagreement", "description": "mlx-whisper + LocalAgreement"})
        combos.append({"backend": "mlx-whisper", "policy": "simulstreaming", "description": "mlx-whisper + SimulStreaming"})
    except ImportError:
        pass

    # openai-whisper
    try:
        import whisper  # noqa: F401
        combos.append({"backend": "whisper", "policy": "localagreement", "description": "openai-whisper + LocalAgreement"})
        combos.append({"backend": "whisper", "policy": "simulstreaming", "description": "openai-whisper + SimulStreaming"})
    except ImportError:
        pass

    # voxtral-mlx
    try:
        from whisperlivekit.voxtral_mlx import VoxtralMLXModel  # noqa: F401
        combos.append({"backend": "voxtral-mlx", "policy": "voxtral", "description": "voxtral-mlx (MLX)"})
    except ImportError:
        pass

    # voxtral (HuggingFace)
    try:
        from transformers import AutoModelForSpeechSeq2Seq  # noqa: F401
        combos.append({"backend": "voxtral", "policy": "voxtral", "description": "voxtral (HuggingFace)"})
    except ImportError:
        pass

    return combos


def print_cross_backend_comparison(all_results: List[TestResult]):
    """Print a comparison table across backends and policies."""
    print(f"\n{'=' * 110}")
    print("CROSS-BACKEND BENCHMARK COMPARISON")
    print(f"{'=' * 110}")
    print(
        f"{'Backend':<18} {'Policy':<16} {'File':<30} "
        f"{'WER':>7} {'RTF':>6} {'MAE(s)':>7} {'MaxΔ(s)':>8}"
    )
    print(f"{'-' * 110}")

    for r in all_results:
        wer_str = f"{r.wer:.2%}" if r.wer is not None else "  -"
        rtf_str = f"{r.rtf:.2f}x"
        mae_str = f"{r.timestamp_mae:.3f}" if r.timestamp_mae is not None else "  -"
        max_str = f"{r.timestamp_max_delta:.3f}" if r.timestamp_max_delta is not None else "  -"
        # Truncate filename for readability
        fname = r.audio_file[:28] + ".." if len(r.audio_file) > 30 else r.audio_file
        print(
            f"{r.backend:<18} {r.policy:<16} {fname:<30} "
            f"{wer_str:>7} {rtf_str:>6} {mae_str:>7} {max_str:>8}"
        )

    print(f"{'-' * 110}")

    # Per-backend averages
    from collections import defaultdict
    by_combo = defaultdict(list)
    for r in all_results:
        by_combo[(r.backend, r.policy)].append(r)

    print(f"\n{'Backend':<18} {'Policy':<16} {'Avg WER':>8} {'Avg RTF':>8} {'Avg MAE':>8} {'Files':>6}")
    print(f"{'-' * 80}")
    for (backend, policy), group in sorted(by_combo.items()):
        wer_vals = [r.wer for r in group if r.wer is not None]
        rtf_vals = [r.rtf for r in group]
        mae_vals = [r.timestamp_mae for r in group if r.timestamp_mae is not None]
        avg_wer = f"{sum(wer_vals)/len(wer_vals):.2%}" if wer_vals else "  -"
        avg_rtf = f"{sum(rtf_vals)/len(rtf_vals):.2f}x"
        avg_mae = f"{sum(mae_vals)/len(mae_vals):.3f}" if mae_vals else "  -"
        print(
            f"{backend:<18} {policy:<16} {avg_wer:>8} {avg_rtf:>8} {avg_mae:>8} {len(group):>6}"
        )
    print(f"{'=' * 110}")


def _quiet_loggers(verbose: bool):
    """Set internal module log levels to reduce noise."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        for mod in (
            "whisperlivekit.audio_processor", "whisperlivekit.simul_whisper",
            "whisperlivekit.tokens_alignment", "whisperlivekit.simul_whisper.align_att_base",
            "whisperlivekit.simul_whisper.simul_whisper",
        ):
            logging.getLogger(mod).setLevel(logging.WARNING)


async def run_benchmark(
    audio_files: List[Path], chunk_ms: int, realtime: bool,
    model_size: str, lan: str, max_duration: float, vac: bool,
    verbose: bool,
) -> List[TestResult]:
    """Run benchmark across all available backend+policy combinations."""
    combos = detect_available_backends()
    if not combos:
        logger.error("No backends available. Install at least one ASR backend.")
        return []

    logger.info(f"Detected {len(combos)} backend+policy combinations:")
    for c in combos:
        logger.info(f"  - {c['description']}")

    all_results = []
    for i, combo in enumerate(combos, 1):
        backend = combo["backend"]
        policy = combo["policy"]
        desc = combo["description"]

        print(f"\n{'*' * 70}")
        print(f"* BENCHMARK {i}/{len(combos)}: {desc}")
        print(f"{'*' * 70}")

        try:
            engine = create_engine(
                backend, model_size, lan, vac=vac, policy=policy,
            )
            _quiet_loggers(verbose)

            results = await run_all_tests(
                engine, audio_files, chunk_ms, realtime,
                backend=backend, policy=policy, lan=lan,
                max_duration=max_duration,
            )
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Failed to run {desc}: {e}")
            import traceback
            traceback.print_exc()

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Offline backend test harness (AudioProcessor-level)"
    )
    parser.add_argument(
        "--backend", default="faster-whisper",
        help="Backend: voxtral, voxtral-mlx, auto, faster-whisper, mlx-whisper, whisper.",
    )
    parser.add_argument(
        "--policy", default="",
        help="Override backend policy: localagreement, simulstreaming, voxtral.",
    )
    parser.add_argument(
        "--audio", default=None,
        help="Path to a single audio file (WAV, MP3, FLAC, etc.).",
    )
    parser.add_argument(
        "--audio-dir", default=None,
        help="Directory of audio files to test. Defaults to audio_tests/ if neither --audio nor --audio-dir given.",
    )
    parser.add_argument(
        "--chunk-ms", type=int, default=100,
        help="Chunk size in milliseconds (simulates real-time interval).",
    )
    parser.add_argument(
        "--model", default="", dest="model_size",
        help="Model size or HF repo ID.",
    )
    parser.add_argument("--lan", default="en", help="Language code.")
    parser.add_argument(
        "--no-realtime", action="store_true",
        help="Skip real-time pacing between chunks (faster but less realistic).",
    )
    parser.add_argument(
        "--no-vac", action="store_true",
        help="Disable Voice Activity Classification (send all audio without silence filtering).",
    )
    parser.add_argument(
        "--diarization", action="store_true",
        help="Enable speaker diarization.",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark across all detected backend+policy combinations.",
    )
    parser.add_argument(
        "--json", default=None, dest="json_output",
        help="Write structured JSON results to this file.",
    )
    parser.add_argument(
        "--max-duration", type=float, default=60.0,
        help="Skip audio files longer than this many seconds (default: 60).",
    )
    parser.add_argument(
        "--insert-silence", nargs=2, type=float, metavar=("SECS", "AT_SEC"),
        action="append", default=[],
        help="Insert SECS of silence at AT_SEC position. Can be repeated. "
             "E.g.: --insert-silence 3.0 2.0 --insert-silence 5.0 7.0",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show debug-level logs from all components.",
    )
    args = parser.parse_args()

    realtime = not args.no_realtime
    vac = not args.no_vac

    # Resolve audio file(s)
    if args.audio:
        audio_files = [Path(args.audio)]
    elif args.audio_dir:
        audio_files = discover_audio_files(args.audio_dir)
    elif AUDIO_TESTS_DIR.is_dir():
        audio_files = discover_audio_files(str(AUDIO_TESTS_DIR))
    else:
        # Fall back to jfk.wav download
        audio_files = [download_sample_audio()]

    if not audio_files:
        logger.error("No audio files found.")
        sys.exit(1)

    logger.info(f"Audio files: {[f.name for f in audio_files]}")

    if args.benchmark:
        # --- Multi-backend benchmark mode ---
        all_results = asyncio.run(
            run_benchmark(
                audio_files, args.chunk_ms, realtime,
                args.model_size, args.lan, args.max_duration, vac,
                args.verbose,
            )
        )
        if all_results:
            print_cross_backend_comparison(all_results)
        results = all_results
    else:
        # --- Single-backend mode ---
        policy = args.policy
        logger.info(f"Creating {args.backend} engine...")
        engine = create_engine(
            args.backend, args.model_size, args.lan,
            diarization=args.diarization, vac=vac, policy=policy,
        )
        logger.info("Engine ready.")

        _quiet_loggers(args.verbose)

        results = asyncio.run(
            run_all_tests(
                engine, audio_files, args.chunk_ms, realtime,
                args.backend, policy, args.lan,
                max_duration=args.max_duration,
                silence_insertions=args.insert_silence or None,
            )
        )

        if len(results) > 1:
            print_benchmark_summary(results)

    # JSON output
    if args.json_output and results:
        json_results = []
        for r in results:
            d = asdict(r)
            d.pop("last_response", None)  # too verbose for summary
            json_results.append(d)
        Path(args.json_output).write_text(
            json.dumps(json_results, indent=2, ensure_ascii=False)
        )
        logger.info(f"Results written to {args.json_output}")


if __name__ == "__main__":
    main()
