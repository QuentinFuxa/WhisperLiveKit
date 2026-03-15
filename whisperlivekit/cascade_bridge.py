"""
Bridge between WhisperLiveKit STT and IWSLT26 MT pipeline.

Converts streaming ASRToken output from SimulStreaming into the JSONL
format expected by the AlignAtt MT agent (iwslt26-sst).

Output format (one JSON per line):
  {"text": "word or phrase", "emission_time": 1.234, "is_final": false, "speech_time": 1.0}

Where:
  - text: the emitted word/phrase
  - emission_time: wall-clock time when the word was emitted (for compute-aware eval)
  - speech_time: timestamp in the audio (for compute-unaware eval)
  - is_final: whether this is the last word of a segment/silence boundary
"""

import json
import time
from typing import List, TextIO

from whisperlivekit.timed_objects import ASRToken


class CascadeBridge:
    """Converts ASRToken stream to JSONL for the MT agent."""

    def __init__(self, output_file: TextIO = None):
        self.output_file = output_file
        self.start_time = time.time()
        self.entries: List[dict] = []

    def emit_tokens(self, tokens: List[ASRToken], is_final: bool = False):
        """Emit a batch of tokens from the STT."""
        wall_clock = time.time() - self.start_time

        for i, token in enumerate(tokens):
            entry = {
                "text": token.text.strip(),
                "emission_time": round(wall_clock, 3),
                "speech_time": round(token.start, 3),
                "is_final": is_final and (i == len(tokens) - 1),
            }
            self.entries.append(entry)
            if self.output_file:
                self.output_file.write(json.dumps(entry) + "\n")
                self.output_file.flush()

    def get_entries(self) -> List[dict]:
        return self.entries

    def get_text(self) -> str:
        """Get the full transcribed text."""
        return " ".join(e["text"] for e in self.entries if e["text"])

    def save(self, path: str):
        """Save all entries to a JSONL file."""
        with open(path, "w") as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + "\n")


def run_stt_to_jsonl(
    audio_path: str,
    output_path: str,
    model_id: str = "Qwen/Qwen3-ASR-0.6B",
    alignment_heads_path: str = None,
    border_fraction: float = 0.20,
    language: str = "en",
    chunk_sec: float = 1.0,
):
    """Run STT on an audio file and save JSONL output for the MT agent.

    This is the main entry point for the cascade: audio file → JSONL.
    """
    import wave
    import numpy as np
    from whisperlivekit.qwen3_simul_kv import Qwen3SimulKVASR, Qwen3SimulKVOnlineProcessor

    # Load audio
    with wave.open(audio_path, 'r') as wf:
        audio = np.frombuffer(
            wf.readframes(wf.getnframes()), dtype=np.int16
        ).astype(np.float32) / 32768.0

    # Initialize STT
    asr = Qwen3SimulKVASR(
        model_dir=model_id,
        lan=language,
        alignment_heads_path=alignment_heads_path,
        border_fraction=border_fraction,
    )
    proc = Qwen3SimulKVOnlineProcessor(asr)
    bridge = CascadeBridge()

    # Stream audio in chunks
    chunk_samples = int(chunk_sec * 16000)
    offset = 0
    stream_time = 0.0

    while offset < len(audio):
        chunk = audio[offset:offset + chunk_samples]
        stream_time += len(chunk) / 16000
        proc.insert_audio_chunk(chunk, stream_time)
        words, _ = proc.process_iter(is_last=False)
        if words:
            bridge.emit_tokens(words, is_final=False)
        offset += chunk_samples

    # Final flush
    final_words, _ = proc.finish()
    if final_words:
        bridge.emit_tokens(final_words, is_final=True)

    # Save
    bridge.save(output_path)
    return bridge
