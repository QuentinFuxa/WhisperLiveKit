from __future__ import annotations


def is_silent_pcm16(chunk: bytes) -> bool:
    return bool(chunk) and all(byte == 0 for byte in chunk)


def post_process_realtime_text(text: str) -> str:
    if not text:
        return ""

    pieces: list[str] = []
    for line in text.replace("\r", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        if "<asr_text>" in line:
            _, line = line.rsplit("<asr_text>", 1)
        elif line.startswith("language "):
            continue
        line = line.strip()
        if line:
            pieces.append(line)

    return " ".join(pieces).strip()
