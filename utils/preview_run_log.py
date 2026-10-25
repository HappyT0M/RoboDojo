"""Save each preview run's unmodified output in a Beijing-time-named log."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re
import select
import shlex
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


BEIJING_TIME = ZoneInfo("Asia/Shanghai")
WORKER_ENV_VAR = "ROBODOJO_PREVIEW_RECORD_WORKER"
ROPE_TASKS = frozenset({
    "put_rope_ball_in_basket",
    "pass_rope_through_ring",
    "route_rope_around_posts",
})


def _argument_value(argv: list[str], option: str) -> str | None:
    for index, argument in enumerate(argv):
        if argument == option:
            return argv[index + 1] if index + 1 < len(argv) else None
        if argument.startswith(option + "="):
            return argument[len(option) + 1:]
    return None


def _task_label(argv: list[str]) -> str:
    task = _argument_value(argv, "--task_name")
    if task is None:
        tasks = _argument_value(argv, "--tasks")
        if tasks:
            names = [name.strip() for name in tasks.split(",") if name.strip()]
            task = names[0] if len(names) == 1 else "multiple_tasks"
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", task or "deformable_tasks").strip("_")
    return safe_name[:80] or "deformable_tasks"


def _without_options(argv: list[str], options: set[str]) -> list[str]:
    remaining = []
    skip_value = False
    for argument in argv:
        if skip_value:
            skip_value = False
            continue
        if argument in options:
            skip_value = True
        elif not any(argument.startswith(option + "=") for option in options):
            remaining.append(argument)
    return remaining


def plan_preview_workers(project_root: str | Path, argv: list[str]) -> list[list[str]]:
    """Use a fresh process per CPU rope layout to avoid stale PhysX articulations."""
    if any(arg in ("-h", "--help", "--layout_id") or arg.startswith("--layout_id=") for arg in argv):
        return [argv]
    task_name = _argument_value(argv, "--task_name")
    tasks_arg = _argument_value(argv, "--tasks")
    if bool(task_name) == bool(tasks_arg):
        return [argv]  # Let the main argparse report missing/conflicting task options.
    tasks = [task_name] if task_name else [name.strip() for name in tasks_arg.split(",") if name.strip()]
    if not tasks:
        return [argv]

    base_argv = _without_options(argv, {"--task_name", "--tasks"})
    config_name = _argument_value(argv, "--env_cfg_type") or "arx_x5"
    seed = _argument_value(argv, "--seed") or "0"
    layout_dir = Path(project_root) / "Assets" / "Eval_Layout" / "RoboDojo" / config_name / seed
    workers = []
    for task in tasks:
        task_argv = [*base_argv, "--task_name", task]
        if task not in ROPE_TASKS or not layout_dir.is_dir():
            workers.append(task_argv)
            continue
        pattern = re.compile(rf"{re.escape(task)}_(\d+)\.json")
        layout_ids = sorted(
            int(match.group(1))
            for path in layout_dir.iterdir()
            if (match := pattern.fullmatch(path.name)) and path.is_file()
        )
        if not layout_ids:
            workers.append(task_argv)  # Preserve the worker's normal missing-layout error.
        else:
            workers.extend([*task_argv, "--layout_id", str(layout_id)] for layout_id in layout_ids)
    return workers


def _open_run_log(log_dir: Path, argv: list[str]):
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(BEIJING_TIME).strftime("%Y-%m-%dT%H:%M:%S")
    base_name = f"record_{_task_label(argv)}_{timestamp}"
    suffix = 1
    while True:
        name = f"{base_name}{'' if suffix == 1 else f'_{suffix}'}.log"
        log_path = log_dir / name
        try:
            descriptor = os.open(log_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            return log_path, os.fdopen(descriptor, "wb", buffering=0)
        except FileExistsError:
            suffix += 1


def _write_all(descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written == 0:
            raise OSError("write returned zero bytes")
        remaining = remaining[written:]


def run_with_timestamped_log(script_path: str, argv: list[str], log_dir: str | Path) -> int:
    """Run planned Isaac workers sequentially and stream output to one log."""
    worker_argvs = plan_preview_workers(Path(script_path).resolve().parents[1], argv)
    worker_env = os.environ.copy()
    worker_env[WORKER_ENV_VAR] = "1"
    worker_env["PYTHONUNBUFFERED"] = "1"
    log_path, log_file = _open_run_log(Path(log_dir), argv)

    with log_file:
        def emit(data: bytes) -> None:
            _write_all(log_file.fileno(), data)
            try:
                _write_all(1, data)
            except BrokenPipeError:
                pass

        print(f"[record] log_file={log_path}", flush=True)
        emit(f"[record] planned_workers={len(worker_argvs)}\n".encode("utf-8"))
        failed_workers = []
        for index, worker_argv in enumerate(worker_argvs, start=1):
            command = [sys.executable, script_path, *worker_argv]
            emit(f"\n[record] worker={index}/{len(worker_argvs)} command={shlex.join(command)}\n".encode("utf-8"))
            try:
                worker = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env=worker_env, start_new_session=True,
                )
            except OSError as exc:
                emit(f"[record] start_failed={exc}\n".encode("utf-8"))
                failed_workers.append(index)
                continue
            old_handlers = {}
            received_signal = 0

            def forward_signal(signum, _frame) -> None:
                nonlocal received_signal
                received_signal = signum
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
                last_output = time.monotonic()
                while True:
                    ready, _, _ = select.select([output_fd], [], [], 0.5)
                    if ready:
                        chunk = os.read(output_fd, 65536)
                        if not chunk:
                            break
                        emit(chunk)
                        last_output = time.monotonic()
                    elif worker.poll() is not None and time.monotonic() - last_output > 2:
                        # A crashed Kit process can leave a helper holding the pipe open.
                        break
                return_code = worker.wait()
                emit(f"\n[record] worker={index}/{len(worker_argvs)} exit_code={return_code} pid={worker.pid}\n".encode("utf-8"))
                if return_code != 0:
                    failed_workers.append(index)
            finally:
                for signum, handler in old_handlers.items():
                    signal.signal(signum, handler)
                if worker.poll() is None:
                    worker.terminate()
                    worker.wait()
                if worker.stdout is not None:
                    worker.stdout.close()
            if received_signal:
                return 128 + received_signal
        emit(f"[record] completed_workers={len(worker_argvs) - len(failed_workers)}/{len(worker_argvs)} failed_workers={failed_workers}\n".encode("utf-8"))
        return 1 if failed_workers else 0
