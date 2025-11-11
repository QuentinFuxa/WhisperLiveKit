"""GPT-4o-mini transcript post-processing pipeline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from .config import GPTConfig
from .ledger_store import LedgerStore

PROMPT_TEMPLATE = (
    "Z textu extrahuj všechny výdaje, poznámky, nebo úkoly ve formátu JSON. Text:\n{body}"
)


async def process_transcripts(
    *,
    max_segments: int = 20,
    cfg: Optional[GPTConfig] = None,
    client: Optional[AsyncOpenAI] = None,
    ledger: Optional[LedgerStore] = None,
    sleep_between: float = 0.5,
) -> int:
    """Process the most recent transcript segments and log GPT outputs.

    Returns the number of segments successfully processed.
    """

    cfg = cfg or GPTConfig()
    client = client or AsyncOpenAI(api_key=cfg.api_key)
    ledger = ledger or LedgerStore(cfg.ledger_path)

    segments = _load_segments(cfg.input_path, max_segments)
    if not segments:
        print(f"[GPT] No transcripts found at {cfg.input_path}")
        return 0

    enriched_segments = _assign_sessions(segments, cfg.session_gap_sec)
    processed = 0
    for data, session_id, gap in enriched_segments:
        prompt_header = (
            f"Session {session_id} (gap {gap:.1f}s): Analyzuj text a vyhledej "
            "příkazy, poznámky, úkoly nebo výdaje.\n"
        )
        prompt = prompt_header + data.get("text", "")

        try:
            response = await client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
            )
            result = response.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover - network edge
            result = f"[Error: {exc}]"

        ledger.append(
            {
                "session_id": session_id,
                "gap": gap,
                "start": data.get("start"),
                "end": data.get("end"),
                "input": data.get("text", ""),
                "gpt_output": result,
            }
        )
        print(f"[GPT] processed segment -> {cfg.ledger_path} (session {session_id})")
        processed += 1
        if sleep_between:
            await asyncio.sleep(sleep_between)

    return processed


def _load_segments(path: str, max_segments: int) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()[-max_segments:]

    segments: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            segments.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return segments


def _assign_sessions(
    segments: List[Dict[str, Any]],
    gap_threshold: float,
) -> List[Tuple[Dict[str, Any], int, float]]:
    session_id = 1
    prev_end: Optional[float] = None
    enriched: List[Tuple[Dict[str, Any], int, float]] = []

    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        gap = 0.0
        if prev_end is not None and isinstance(start, (int, float)):
            gap = float(start) - float(prev_end)
        if gap > gap_threshold:
            session_id += 1
            print(f"[Session] Gap {gap:.1f}s → new session {session_id}")

        enriched.append((segment, session_id, max(gap, 0.0)))

        if isinstance(end, (int, float)):
            prev_end = float(end)

    return enriched


if __name__ == "__main__":
    asyncio.run(process_transcripts())
