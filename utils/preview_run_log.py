"""Capture a preview worker's Python and native output with Beijing timestamps."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


BEIJING_TIME = ZoneInfo("Asia/Shanghai")
WORKER_ENV_VAR = "ROBODOJO_PREVIEW_RECORD_WORKER"


def run_with_beijing_log(script_path: str, argv: list[str], log_path: str | Path) -> int:
    """Run the Isaac worker and append every stdout/stderr line to one fixed log."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, script_path, *argv]
    worker_env = os.environ.copy()
    worker_env[WORKER_ENV_VAR] = "1"
    worker_env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("ab", buffering=0) as log_file:
        run_id = f"{datetime.now(BEIJING_TIME):%Y%m%d_%H%M%S}_{os.getpid()}"

        def emit(raw_line: bytes) -> None:
            timestamp = datetime.now(BEIJING_TIME).isoformat(timespec="seconds")
            line = f"[{timestamp} 北京时间] [run={run_id}] ".encode("utf-8") + raw_line.rstrip(b"\r\n") + b"\n"
            os.write(log_file.fileno(), line)
            try:
                os.write(1, line)
            except BrokenPipeError:
                pass

        emit(f"[record] start pid={os.getpid()} command={shlex.join(command)}".encode("utf-8"))
        emit(f"[record] log_file={log_path}".encode("utf-8"))
        try:
            worker = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=worker_env, start_new_session=True,
            )
        except OSError as exc:
            emit(f"[record] start_failed={exc}".encode("utf-8"))
            return 1
        old_handlers = {}

        def forward_signal(signum, _frame) -> None:
            if worker.poll() is None:
                try:
                    os.killpg(worker.pid, signum)
                except ProcessLookupError:
                    pass

        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, forward_signal)
        try:
            assert worker.stdout is not None
            output_fd = worker.stdout.fileno()
            pending = bytearray()
            last_output = time.monotonic()
            while True:
                ready, _, _ = select.select([output_fd], [], [], 0.5)
                if ready:
                    chunk = os.read(output_fd, 65536)
                    if not chunk:
                        break
                    pending.extend(chunk)
                    last_output = time.monotonic()
                    while b"\n" in pending:
                        line, _, remainder = pending.partition(b"\n")
                        emit(bytes(line))
                        pending = bytearray(remainder)
                    if len(pending) > 1024 * 1024:
                        emit(bytes(pending))
                        pending.clear()
                elif worker.poll() is not None and time.monotonic() - last_output > 2:
                    # A crashed Kit process can leave a helper holding the pipe open.
                    break
            if pending:
                emit(bytes(pending))
            return_code = worker.wait()
            emit(f"[record] exit_code={return_code} pid={worker.pid}".encode("utf-8"))
            return return_code
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            if worker.poll() is None:
                worker.terminate()
                worker.wait()
            if worker.stdout is not None:
                worker.stdout.close()
