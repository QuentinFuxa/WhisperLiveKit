#!/usr/bin/env python3
"""Run the sixteen M5 screening reports serially, only on AC power.

The workspace contains one source checkout per measured revision, each with
its own environment, plus the engine configuration files:

    WORKSPACE/whisper/.venv     Whisper and Qwen Metal source (b2baee9)
    WORKSPACE/nemotron/.venv    Nemotron candidate (1842f4c)
    WORKSPACE/qwen/.venv        native Qwen candidate (24082a0)
    WORKSPACE/configs/{whisper,nemotron,qwen,metal}.json

Qwen Metal runs in the separately installed vllm-metal environment, with WLK,
the qwen3-asr-causal submodule and an optional overlay directory on PYTHONPATH.

Order is fixed: Whisper, Nemotron, native Qwen, Qwen Metal; within each backend
English, French, Chinese, then the continuous Chinese stream. Each run is a fresh
process. Power is sampled every 60 s with pmset; a run starts only on AC, and a
run that observed another power source is flagged in runs.jsonl. Existing
reports are skipped, so an interrupted series can resume.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ENTRY = 'import sys; sys.argv[0]="wlk"; from whisperlivekit.cli import main; sys.exit(main())'
BACKENDS = (
    # report key, checkout, backend, model label, configuration
    ("whisper-small-la", "whisper", "mlx-whisper", "small", "whisper"),
    ("nemotron-native", "nemotron", "nemotron-mlx-asr", "0.6B", "nemotron"),
    ("qwen-native", "qwen", "mlx-qwen3-asr", "0.6B", "qwen"),
    ("qwen-metal", "whisper", "qwen3-vllm-metal", "0.6B", "metal"),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def power_sample():
    battery = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    thermal = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout.strip()
    source = re.search(r"Now drawing from '([^']+)'", battery)
    percent = re.search(r"(\d+)%", battery)
    return {"timestamp": now(), "power_source": source.group(1) if source else "unknown",
            "battery_percent": int(percent.group(1)) if percent else None, "thermal_status": thermal}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metal-python", type=Path, default=Path.home() / ".venv-vllm-metal/bin/python")
    parser.add_argument("--metal-overlay", type=Path, help="Optional --target directory added to PYTHONPATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workspace, reports, logs = args.workspace.resolve(), args.output / "reports", args.output / "logs"
    reports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    other_power, stop = threading.Event(), threading.Event()

    def log_power():
        while not stop.is_set():
            sample = power_sample()
            if sample["power_source"] != "AC Power":
                other_power.set()
            with (args.output / "power-log.jsonl").open("a") as handle:
                handle.write(json.dumps(sample) + "\n")
            stop.wait(60)

    threading.Thread(target=log_power, daemon=True).start()
    metal_path = [workspace / "whisper", workspace / "whisper/third_party/qwen3-asr-causal/src"]
    if args.metal_overlay:
        metal_path.append(args.metal_overlay.resolve())

    for key, checkout, backend, model, config in BACKENDS:
        if key == "qwen-metal":
            python, extra_env = args.metal_python, {"PYTHONPATH": os.pathsep.join(map(str, metal_path))}
        else:
            python, extra_env = workspace / checkout / ".venv/bin/python", {}
        for language, continuous in (("en", False), ("fr", False), ("zh", False), ("zh", True)):
            name = f"{key}-{language}" + ("-continuous" if continuous else "")
            report = reports / f"{name}.json"
            if report.exists():
                print(f"{now()} skip {name} (report exists)", flush=True)
                continue
            announced = False
            while power_sample()["power_source"] != "AC Power":
                if not announced:
                    print(f"{now()} paused: not on AC power", flush=True)
                    announced = True
                time.sleep(60)
            other_power.clear()
            command = [str(python), "-c", ENTRY, "bench", "--backend", backend, "--model", model,
                       "--languages", language, "--manifest", str(args.manifest.resolve()),
                       "--config", str(workspace / "configs" / f"{config}.json"),
                       "--speed", "1", "--json", str(report.resolve())]
            command += ["--continuous", "--repeats", "1"] if continuous else ["--warmup", "--repeats", "3"]
            start = now()
            print(f"{start} start {name}", flush=True)
            with (logs / f"{name}.log").open("w") as log:
                try:
                    code = subprocess.run(command, cwd=workspace / checkout, env={**os.environ, **extra_env},
                                          stdout=log, stderr=subprocess.STDOUT, timeout=3600).returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"
            entry = {"name": name, "start": start, "end": now(), "exit": code,
                     "battery_during_run": other_power.is_set() or power_sample()["power_source"] != "AC Power",
                     "report": report.exists(), "command": command[3:]}
            with (args.output / "runs.jsonl").open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
            print(f"{entry['end']} done {name} exit={code} report={entry['report']} "
                  f"other_power={entry['battery_during_run']}", flush=True)
    stop.set()
    print(f"{now()} series complete", flush=True)


if __name__ == "__main__":
    sys.exit(main())
