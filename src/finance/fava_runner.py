"""Launch a Fava dashboard for the DayMind ledger."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from fava.application import create_app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Fava dashboard for DayMind.")
    parser.add_argument("--ledger", default=os.getenv("FAVA_LEDGER_PATH", "finance/ledger.beancount"))
    parser.add_argument("--host", default=os.getenv("FAVA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FAVA_PORT", "5000")))
    parser.add_argument("--read-only", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ledger_path = Path(args.ledger)
    if not ledger_path.exists():
        print(f"[Fava] Ledger not found at {ledger_path}. Run the exporter first.", file=sys.stderr)
        return 1

    app = create_app([ledger_path], load=False, read_only=args.read_only)
    print(f"[Fava] Serving {ledger_path} at http://{args.host}:{args.port}/finance/")
    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
