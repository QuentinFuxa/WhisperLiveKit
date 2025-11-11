"""Daily summary generator for GPT post-processing outputs."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import AsyncOpenAI

from .config import GPTConfig
from .ledger_store import LedgerStore

SUMMARY_PROMPT = (
    "Z těchto transkriptů vytvoř:\n"
    "1. Strukturovaný JSON (výdaje, úkoly, poznámky) – každý záznam jako objekt.\n"
    "2. Textové shrnutí dne (v češtině).\n"
    "Výstup: nejdříve JSON blok, pak --- a text shrnutí.\n"
)


async def summarize_day(
    day: str,
    entries: List[Dict[str, Any]],
    *,
    cfg: Optional[GPTConfig] = None,
    client: Optional[AsyncOpenAI] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    cfg = cfg or GPTConfig()
    client = client or AsyncOpenAI(api_key=cfg.api_key)
    out_dir = output_dir or Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    content = "\n".join(f"[{e.get('session_id', '?')}] {e.get('input', '')}" for e in entries)
    response = await client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": SUMMARY_PROMPT + content}],
        temperature=0.2,
    )
    text = response.choices[0].message.content or ""
    json_part, separator, summary_part = text.partition("---")

    parsed = safe_json_parse(json_part)
    ledger_out = out_dir / f"ledger_{day}.jsonl"
    summary_out = out_dir / f"summary_{day}.md"

    with open(ledger_out, "w", encoding="utf-8") as json_file:
        for obj in parsed:
            json_file.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(summary_out, "w", encoding="utf-8") as summary_file:
        summary_content = summary_part.strip() if separator else text.strip()
        summary_file.write(summary_content)

    print(f"[Summary] Generated outputs for {day}: {ledger_out}, {summary_out}")
    return ledger_out, summary_out


async def run_daily_summaries(
    *, cfg: Optional[GPTConfig] = None, client: Optional[AsyncOpenAI] = None
) -> None:
    cfg = cfg or GPTConfig()
    store = LedgerStore(cfg.ledger_path)
    groups = store.group_by_day()
    if not groups:
        print("[Summary] No ledger entries to summarize.")
        return

    for day in sorted(groups.keys()):
        await summarize_day(day, groups[day], cfg=cfg, client=client)


def safe_json_parse(payload: str) -> List[Dict[str, Any]]:
    """Parse GPT JSON output while tolerating markdown fences and errors."""

    if not payload:
        return []

    cleaned = re.sub(r"```(?:json)?", "", payload, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(cleaned)
    except Exception as exc:  # pragma: no cover - error path
        logging.warning("[safe_json_parse] JSON parse failed: %s", exc)
        return [{"error": "parse_failed", "raw": cleaned[:200]}]

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        normalized: List[Dict[str, Any]] = []
        for item in parsed:
            if isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append({"value": item})
        return normalized
    return [{"value": parsed}]


if __name__ == "__main__":
    asyncio.run(run_daily_summaries())
