#!/usr/bin/env python3
"""Measure fresh-process import and engine initialization using a cached model.

Run with the selected backend's Python environment from its source checkout.
Includes constructor warmup; excludes subsequent pipeline audio and HTTP startup.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

CHILD = '''
import json, sys
from whisperlivekit.core import TranscriptionEngine
options = json.loads(sys.argv[1])
engine = TranscriptionEngine(**options)
print("__WLK_ENGINE_READY__", flush=True)
from whisperlivekit.benchmark.metrics import get_system_info
print("__WLK_SYSTEM__" + json.dumps(get_system_info()), flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    options = {"backend": args.backend, "model_size": args.model, "lan": "en", "pcm_input": True,
               **json.loads(args.config.read_text())}
    if options.get("api_token"):
        parser.error("Do not pass server credentials to a local startup measurement")
    record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "options": options,
              "process_to_engine_ready_s": None, "status": "error", "log": []}
    started = time.perf_counter()
    with subprocess.Popen([sys.executable, "-c", CHILD, json.dumps(options)],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        def timeout():
            record["error"] = "Engine process exceeded five minutes"
            process.kill()

        timer = threading.Timer(300, timeout)
        timer.start()
        try:
            for line in process.stdout:
                if line.strip() == "__WLK_ENGINE_READY__":
                    record["process_to_engine_ready_s"] = time.perf_counter() - started
                elif line.startswith("__WLK_SYSTEM__"):
                    record["system"] = json.loads(line.removeprefix("__WLK_SYSTEM__"))
                else:
                    record["log"].append(line.rstrip())
            record["exit_code"] = process.wait()
        finally:
            timer.cancel()
    if record["exit_code"] == 0 and record["process_to_engine_ready_s"] is not None:
        record["status"] = "ok"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(args.output, record["status"], record["process_to_engine_ready_s"])
    if record["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
