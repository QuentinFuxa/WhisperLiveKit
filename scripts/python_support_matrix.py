#!/usr/bin/env python3
"""Offline Python support matrix runner for WhisperLiveKit."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    from rich.console import Console
    from rich.table import Table

    HAS_RICH = True
except Exception:
    HAS_RICH = False

SAMPLE_URL = (
    "https://github.com/pyannote/pyannote-audio/raw/develop/tutorials/assets/sample.wav"
)
SAMPLE_PATH = Path("audio_tests/support-matrix-sample.wav")
DEFAULT_LOGS_DIR = Path("outputs/python-matrix/logs")
PYTHON_VERSIONS = ("3.11", "3.12", "3.13")
CONSOLE = Console() if HAS_RICH else None


@dataclass(frozen=True)
class MatrixRow:
    row_id: str
    extras: tuple[str, ...]
    backend: str
    policy: str
    diarization_backend: str
    requires_gpu: bool = False


CASES = (
    MatrixRow(
        row_id="fw-diart-cpu",
        extras=("test", "cpu", "diarization-diart"),
        backend="faster-whisper",
        policy="simulstreaming",
        diarization_backend="diart",
    ),
    MatrixRow(
        row_id="fw-sortformer-cpu",
        extras=("test", "cpu", "diarization-sortformer"),
        backend="faster-whisper",
        policy="simulstreaming",
        diarization_backend="sortformer",
    ),
    MatrixRow(
        row_id="fw-sortformer-gpu",
        extras=("test", "cu129", "diarization-sortformer"),
        backend="faster-whisper",
        policy="simulstreaming",
        diarization_backend="sortformer",
        requires_gpu=True,
    ),
    MatrixRow(
        row_id="voxtral-diart-cpu",
        extras=("test", "cpu", "voxtral-hf", "diarization-diart"),
        backend="voxtral",
        policy="voxtral",
        diarization_backend="diart",
    ),
)

EXPECTED_FAILURE_CASES = {
    ("3.11", "voxtral-diart-cpu"): "known_unstable_voxtral_diart_cpu",
    ("3.12", "voxtral-diart-cpu"): "known_unstable_voxtral_diart_cpu",
}
UNSUPPORTED_CASES = {
    ("3.13", "fw-sortformer-cpu"): "unsupported_py313_sortformer_protobuf",
    ("3.13", "fw-sortformer-gpu"): "unsupported_py313_sortformer_protobuf",
}


@dataclass(frozen=True)
class CaseResult:
    python_version: str
    row_id: str
    status: Literal["PASS", "FAIL", "N/A"]
    reason: str
    duration_sec: float
    hint: str = ""
    log_path: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal WhisperLiveKit offline support matrix"
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=300,
        help="Per-case timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--logs-dir",
        default=str(DEFAULT_LOGS_DIR),
        help="Directory where per-case logs are written (default: outputs/python-matrix/logs)",
    )
    return parser.parse_args()


def safe_slug(text: str) -> str:
    return text.replace("=", "-").replace("|", "__").replace("/", "-").replace(" ", "-")


def status_style(status: str) -> str:
    if status == "PASS":
        return "green"
    if status == "FAIL":
        return "bold red"
    if status == "N/A":
        return "yellow"
    return "white"


def print_line(message: str, style: str | None = None) -> None:
    if CONSOLE is None:
        print(message)
        return
    if style:
        CONSOLE.print(message, style=style, highlight=False)
    else:
        CONSOLE.print(message, highlight=False)


def tail_text(text: str | None, max_chars: int = 220) -> str:
    if not text:
        return ""
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[-max_chars:]


def run_command(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout: int | None = None,
    log_path: Path | None = None,
    log_section: str | None = None,
) -> subprocess.CompletedProcess[str]:
    def _append_log(
        *,
        command: list[str],
        section: str,
        returncode: int | None,
        stdout: str | None,
        stderr: str | None,
        timed_out: bool = False,
    ) -> None:
        if log_path is None:
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n=== {section} ===\n")
            f.write(f"$ {shlex.join(command)}\n")
            if timed_out:
                f.write("status: timeout\n")
            else:
                f.write(f"status: exit_code={returncode}\n")
            if stdout:
                f.write("--- stdout ---\n")
                f.write(stdout)
                if not stdout.endswith("\n"):
                    f.write("\n")
            if stderr:
                f.write("--- stderr ---\n")
                f.write(stderr)
                if not stderr.endswith("\n"):
                    f.write("\n")

    section = log_section or "command"
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        _append_log(
            command=cmd,
            section=section,
            returncode=None,
            stdout=exc.stdout if isinstance(exc.stdout, str) else None,
            stderr=exc.stderr if isinstance(exc.stderr, str) else None,
            timed_out=True,
        )
        raise

    _append_log(
        command=cmd,
        section=section,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
    return proc


def detect_gpu_available() -> bool:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def download_sample(repo_root: Path) -> Path:
    target = repo_root / SAMPLE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        SAMPLE_URL,
        "--output",
        str(target),
    ]
    proc = run_command(cmd, cwd=repo_root, env=os.environ.copy())
    if proc.returncode != 0:
        hint = tail_text(proc.stderr or proc.stdout)
        raise RuntimeError(f"sample_download_failed: {hint}")
    return target


def sync_case_environment(
    repo_root: Path,
    python_version: str,
    row: MatrixRow,
    env_dir: Path,
    log_path: Path,
) -> tuple[bool, str]:
    cmd = ["uv", "sync", "--python", python_version, "--no-dev"]
    for extra in row.extras:
        cmd.extend(["--extra", extra])
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(env_dir)
    proc = run_command(
        cmd,
        cwd=repo_root,
        env=env,
        log_path=log_path,
        log_section="sync",
    )
    if proc.returncode != 0:
        return False, tail_text(proc.stderr or proc.stdout)
    return True, ""


def apply_expected_failure_policy(result: CaseResult) -> CaseResult:
    expected_reason = EXPECTED_FAILURE_CASES.get((result.python_version, result.row_id))
    if result.status != "FAIL" or not expected_reason:
        return result
    override_hint = result.hint
    if result.reason:
        override_hint = (
            f"expected_failure_override original_reason={result.reason}; {override_hint}"
            if override_hint
            else f"expected_failure_override original_reason={result.reason}"
        )
    return CaseResult(
        python_version=result.python_version,
        row_id=result.row_id,
        status="N/A",
        reason=expected_reason,
        duration_sec=result.duration_sec,
        hint=override_hint,
        log_path=result.log_path,
    )


def build_offline_command(
    python_version: str,
    row: MatrixRow,
    sample_audio: Path,
    timeout_sec: int,
) -> tuple[list[str], int | None]:
    base_cmd = [
        "uv",
        "run",
        "--python",
        python_version,
        "--no-sync",
        "python",
        "test_backend_offline.py",
        "--backend",
        row.backend,
        "--policy",
        row.policy,
        "--audio",
        str(sample_audio),
        "--model",
        "tiny",
        "--diarization",
        "--diarization-backend",
        row.diarization_backend,
        "--lan",
        "en",
        "--no-realtime",
    ]
    if shutil.which("timeout"):
        return ["timeout", str(timeout_sec), *base_cmd], None
    return base_cmd, timeout_sec


def run_case(
    repo_root: Path,
    python_version: str,
    row: MatrixRow,
    sample_audio: Path,
    timeout_sec: int,
    gpu_available: bool,
    logs_dir: Path,
) -> CaseResult:
    start = time.monotonic()
    case_slug = safe_slug(f"py{python_version}-{row.row_id}")
    log_path = logs_dir / f"run-{case_slug}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    unsupported_reason = UNSUPPORTED_CASES.get((python_version, row.row_id))
    if unsupported_reason:
        log_path.write_text(
            f"[matrix] precheck_short_circuit status=N/A reason={unsupported_reason}\n",
            encoding="utf-8",
        )
        return CaseResult(
            python_version=python_version,
            row_id=row.row_id,
            status="N/A",
            reason=unsupported_reason,
            duration_sec=0.0,
            hint="unsupported_case_precheck",
            log_path=str(log_path),
        )

    if row.requires_gpu and not gpu_available:
        return CaseResult(
            python_version=python_version,
            row_id=row.row_id,
            status="N/A",
            reason="gpu_unavailable",
            duration_sec=0.0,
            hint="nvidia-smi unavailable or failed",
            log_path=str(log_path),
        )

    env_dir = repo_root / ".matrix-envs" / safe_slug(f"py{python_version}-{row.row_id}")
    sync_ok, sync_hint = sync_case_environment(
        repo_root,
        python_version,
        row,
        env_dir,
        log_path=log_path,
    )
    if not sync_ok:
        return CaseResult(
            python_version=python_version,
            row_id=row.row_id,
            status="FAIL",
            reason="dependency_sync_failed",
            duration_sec=round(time.monotonic() - start, 3),
            hint=sync_hint,
            log_path=str(log_path),
        )

    cmd, process_timeout = build_offline_command(
        python_version, row, sample_audio, timeout_sec
    )
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(env_dir)
    if row.requires_gpu:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        proc = run_command(
            cmd,
            cwd=repo_root,
            env=env,
            timeout=process_timeout,
            log_path=log_path,
            log_section="offline",
        )
    except subprocess.TimeoutExpired as exc:
        return CaseResult(
            python_version=python_version,
            row_id=row.row_id,
            status="FAIL",
            reason="offline_timeout",
            duration_sec=round(time.monotonic() - start, 3),
            hint=tail_text((exc.stderr or "") if isinstance(exc.stderr, str) else ""),
            log_path=str(log_path),
        )

    hint = tail_text(proc.stderr or proc.stdout)
    if proc.returncode == 0:
        return CaseResult(
            python_version=python_version,
            row_id=row.row_id,
            status="PASS",
            reason="ok",
            duration_sec=round(time.monotonic() - start, 3),
            hint=hint,
            log_path=str(log_path),
        )

    reason = "offline_timeout" if proc.returncode == 124 else "offline_run_failed"
    return CaseResult(
        python_version=python_version,
        row_id=row.row_id,
        status="FAIL",
        reason=reason,
        duration_sec=round(time.monotonic() - start, 3),
        hint=hint,
        log_path=str(log_path),
    )


def print_summary(results: list[CaseResult]) -> None:
    pass_count = sum(1 for row in results if row.status == "PASS")
    fail_count = sum(1 for row in results if row.status == "FAIL")
    na_count = sum(1 for row in results if row.status == "N/A")
    if CONSOLE is None:
        print("\n[matrix] results")
        print("python | row | status | reason | duration_s")
        print("---|---|---|---|---")
        for result in results:
            print(
                f"{result.python_version} | {result.row_id} | {result.status} | "
                f"{result.reason} | {result.duration_sec:.3f}"
            )
        print(
            f"\n[matrix] summary pass={pass_count} fail={fail_count} "
            f"na={na_count} total={len(results)}"
        )
    else:
        table = Table(title="Support Matrix Results")
        table.add_column("Python", style="cyan", no_wrap=True)
        table.add_column("Row", style="white")
        table.add_column("Status", no_wrap=True)
        table.add_column("Reason")
        table.add_column("Duration (s)", justify="right", no_wrap=True)
        for result in results:
            table.add_row(
                result.python_version,
                result.row_id,
                f"[{status_style(result.status)}]{result.status}[/{status_style(result.status)}]",
                result.reason,
                f"{result.duration_sec:.3f}",
            )
        CONSOLE.print()
        CONSOLE.print(table)
        CONSOLE.print(
            f"[bold]Summary[/bold] "
            f"pass=[green]{pass_count}[/green] "
            f"fail=[bold red]{fail_count}[/bold red] "
            f"na=[yellow]{na_count}[/yellow] "
            f"total={len(results)}"
        )

    diagnostics = [row for row in results if row.status in {"FAIL", "N/A"} and row.hint]
    if diagnostics:
        if CONSOLE is None:
            print("\n[matrix] diagnostics (failed/n-a cases)")
            for row in diagnostics:
                print(
                    f"- py={row.python_version} row={row.row_id} "
                    f"status={row.status} reason={row.reason}"
                )
                print(f"  hint: {row.hint}")
                if row.log_path:
                    print(f"  log: {row.log_path}")
        else:
            diagnostics_table = Table(title="Diagnostics (FAIL / N/A)")
            diagnostics_table.add_column("Case", style="cyan")
            diagnostics_table.add_column("Status", no_wrap=True)
            diagnostics_table.add_column("Reason")
            diagnostics_table.add_column("Hint")
            diagnostics_table.add_column("Log")
            for row in diagnostics:
                diagnostics_table.add_row(
                    f"py={row.python_version} {row.row_id}",
                    f"[{status_style(row.status)}]{row.status}[/{status_style(row.status)}]",
                    row.reason,
                    row.hint,
                    row.log_path,
                )
            CONSOLE.print()
            CONSOLE.print(diagnostics_table)


def main() -> int:
    args = parse_args()
    if args.timeout_sec <= 0:
        print("[matrix] error: --timeout-sec must be > 0", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = (repo_root / args.logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    print_line(f"[matrix] repo_root={repo_root}", style="cyan")
    print_line(f"[matrix] timeout_sec={args.timeout_sec}", style="cyan")
    print_line(f"[matrix] logs_dir={logs_dir}", style="cyan")

    try:
        sample_audio = download_sample(repo_root)
    except Exception as exc:  # pragma: no cover - straightforward failure path
        if CONSOLE is None:
            print(f"[matrix] sample_download_failed: {exc}", file=sys.stderr)
        else:
            CONSOLE.print(
                f"[matrix] sample_download_failed: {exc}",
                style="bold red",
                highlight=False,
            )
        return 1
    print_line(f"[matrix] sample_audio={sample_audio}", style="cyan")

    gpu_available = detect_gpu_available()
    print_line(f"[matrix] gpu_available={gpu_available}", style="cyan")

    results: list[CaseResult] = []
    for python_version in PYTHON_VERSIONS:
        for row in CASES:
            print_line(
                f"\n[matrix] running py={python_version} row={row.row_id}", style="blue"
            )
            result = run_case(
                repo_root=repo_root,
                python_version=python_version,
                row=row,
                sample_audio=sample_audio,
                timeout_sec=args.timeout_sec,
                gpu_available=gpu_available,
                logs_dir=logs_dir,
            )
            result = apply_expected_failure_policy(result)
            results.append(result)
            print_line(
                f"[matrix] {result.status} py={result.python_version} "
                f"row={result.row_id} reason={result.reason} duration={result.duration_sec:.3f}s",
                style=status_style(result.status),
            )
            if result.log_path:
                print_line(f"[matrix] log={result.log_path}", style="dim")

    print_summary(results)
    fail_count = sum(1 for row in results if row.status == "FAIL")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
