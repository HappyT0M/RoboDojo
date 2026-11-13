#!/usr/bin/env python3
"""Run task sweeps from JSON through the existing robodojo.sh eval entry point."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scripts/configs/pi05_deformable.json"
COMMON_FLAGS = (
    "policy_dir", "ckpt", "env_cfg", "action_type", "policy_env", "eval_env",
    "seed", "eval_num", "rigid_force_scale",
)
NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]*\Z")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = set(COMMON_FLAGS) | {"num_envs", "parallel", "tasks", "groups"}
    allowed = required | {"task_layout_ids"}
    if not isinstance(config, dict) or not required.issubset(config) or set(config) - allowed:
        raise ValueError(f"必需字段为：{', '.join(sorted(required))}；可选字段：task_layout_ids")
    for key in COMMON_FLAGS[:6]:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} 必须为非空字符串")
    for key, minimum in (("seed", 0), ("num_envs", 1), ("eval_num", 1)):
        value = config[key]
        if key == "eval_num" and value == "native":
            continue
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} 必须为不小于 {minimum} 的整数")
    scale = config["rigid_force_scale"]
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale < 0:
        raise ValueError("rigid_force_scale 必须为有限的非负数")
    if type(config["parallel"]) is not bool:
        raise ValueError("parallel 必须为 true 或 false")
    tasks = config["tasks"]
    if not isinstance(tasks, list) or not tasks or any(
        not isinstance(task, str) or not NAME.fullmatch(task) for task in tasks
    ) or len(set(tasks)) != len(tasks):
        raise ValueError("tasks 必须为不重复的任务名称列表，且不能为空")
    if not NAME.fullmatch(config["env_cfg"]):
        raise ValueError("env_cfg 必须为配置文件名（不含路径及 .yml 后缀）")
    task_layout_ids = config.get("task_layout_ids", {})
    if not isinstance(task_layout_ids, dict):
        raise ValueError("task_layout_ids 必须是 task_name 到布局编号列表的映射")
    unknown_layout_tasks = set(task_layout_ids) - set(tasks)
    if unknown_layout_tasks:
        raise ValueError(f"task_layout_ids 中的任务不在 tasks 列表：{sorted(unknown_layout_tasks)}")
    for task, layout_ids in task_layout_ids.items():
        if (not isinstance(layout_ids, list) or not layout_ids
                or any(type(layout_id) is not int or layout_id < 0 for layout_id in layout_ids)
                or len(set(layout_ids)) != len(layout_ids)):
            raise ValueError(f"task_layout_ids.{task} 必须是非空、不重复的非负整数列表")
    groups = config["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups 必须为非空列表")
    names = set()
    for group in groups:
        if not isinstance(group, dict) or set(group) != {"name", "wind", "policy_gpu", "env_gpu"}:
            raise ValueError("每个 group 必须包含 name、wind、policy_gpu、env_gpu")
        name = group["name"]
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in names:
            raise ValueError("group.name 必须为不重复的字母、数字、下划线或连字符名称")
        names.add(name)
        for key in ("policy_gpu", "env_gpu"):
            if type(group[key]) is not int or group[key] < 0:
                raise ValueError(f"{name}.{key} 必须为非负整数")
        wind = group["wind"]
        if not isinstance(wind, list) or len(wind) != 3 or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in wind
        ):
            raise ValueError(f"{name}.wind 必须为三个有限数值 [x, y, z]")
    return config


def build_command(config: dict, group: dict, task: str, root: Path) -> list[str]:
    command = ["bash", str(root / "scripts/robodojo.sh"), "eval", "--task", task]
    for key in COMMON_FLAGS:
        command.extend(["--" + key.replace("_", "-"), str(config[key])])
    command.extend(["--policy-gpu", str(group["policy_gpu"]), "--env-gpu", str(group["env_gpu"])])
    command.extend(["--wind", *map(str, group["wind"])])
    return command


def stop_processes(processes: list[subprocess.Popen]) -> None:
    """Stop each private process group, including policy and simulator descendants."""
    pending = {process.pid for process in processes}
    for pgid in pending:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    while pending and time.monotonic() < deadline:
        for process in processes:
            process.poll()  # Reap leaders even if descendants still need cleanup.
        for pgid in list(pending):
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                pending.remove(pgid)
        if pending:
            time.sleep(0.1)
    for pgid in pending:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        process.wait()


def run_groups(config: dict, root: Path) -> int:
    env = os.environ.copy()
    env.update(EVAL_ENV_TYPE="sim", ROBODOJO_NUM_ENVS=str(config["num_envs"]), PYTHONUNBUFFERED="1")
    # An inherited EVAL_NUM must not override JSON's explicit "native" choice.
    env.pop("EVAL_NUM", None)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%dT%H:%M:%S_%f")
    states = []
    interrupted = 0

    def on_signal(signum, _frame):
        nonlocal interrupted
        interrupted = signum

    previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for group in config["groups"]:
            path = log_dir / f"eval_{group['name']}_{stamp}.log"
            log = path.open("x", encoding="utf-8", buffering=1)
            states.append({"group": group, "log": log, "index": 0, "process": None, "rc": None})
            # Record effective settings for interpreting the results later.
            log.write(json.dumps({**config, "groups": [group]}, ensure_ascii=False, indent=2) + "\n")
            print(f"[eval] {group['name']} 日志：{path}", flush=True)

        while not interrupted and any(state["rc"] is None for state in states):
            for state in states:
                if interrupted:
                    break
                if state["rc"] is not None:
                    continue
                process = state["process"]
                if process is not None:
                    rc = process.poll()
                    if rc is None:
                        if not config["parallel"]:
                            break
                        continue
                    state["process"] = None
                    state["log"].write(f"[eval] task_exit_code={rc}\n")
                    if rc != 0:
                        state["rc"] = rc  # Stop this group; other groups continue.
                    else:
                        state["index"] += 1
                        if state["index"] == len(config["tasks"]):
                            state["rc"] = 0
                if state["rc"] is not None:
                    print(f"[eval] {state['group']['name']} 退出码={state['rc']}，"
                          f"完成任务={state['index']}/{len(config['tasks'])}", flush=True)
                    continue
                task = config["tasks"][state["index"]]
                command = build_command(config, state["group"], task, root)
                task_env = env.copy()
                task_env.pop("ROBODOJO_LAYOUT_IDS", None)
                layout_ids = config.get("task_layout_ids", {}).get(task)
                if layout_ids is not None:
                    task_env["ROBODOJO_LAYOUT_IDS"] = ",".join(map(str, layout_ids))
                state["log"].write(f"[eval] command={shlex.join(command)}\n")
                if layout_ids is not None:
                    state["log"].write(f"[eval] selected_layout_ids={layout_ids}\n")
                state["log"].flush()
                suffix = f"，布局={layout_ids}" if layout_ids is not None else "，全部布局"
                print(f"[eval] {state['group']['name']} 开始任务：{task}{suffix}", flush=True)
                state["process"] = subprocess.Popen(
                    command, cwd=root, env=task_env, stdout=state["log"],
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
                if not config["parallel"]:
                    break
            if not interrupted and any(state["rc"] is None for state in states):
                time.sleep(0.2)
    finally:
        processes = [state["process"] for state in states if state["process"] is not None]
        if processes:
            print("[eval] 正在停止模型和仿真进程…", flush=True)
            stop_processes(processes)
        for state in states:
            state["log"].close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if interrupted:
        return 128 + interrupted
    return 1 if any(state["rc"] != 0 for state in states) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON 配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="仅显示命令，不启动模型、不写日志")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        print(f"[eval] 配置：{args.config.resolve()}")
        print(f"[eval] 组间{'并行' if config['parallel'] else '串行'}；组内任务串行；"
              f"eval_num={config['eval_num']}，num_envs={config['num_envs']}", flush=True)
        if args.dry_run:
            for group in config["groups"]:
                for task in config["tasks"]:
                    layout_ids = config.get("task_layout_ids", {}).get(task)
                    layout_suffix = f" layout_ids={layout_ids}" if layout_ids is not None else " all_layouts"
                    print(f"[dry-run] {group['name']}: EVAL_ENV_TYPE=sim "
                          f"ROBODOJO_NUM_ENVS={config['num_envs']} "
                          f"ROBODOJO_LAYOUT_IDS={','.join(map(str, layout_ids)) if layout_ids is not None else '<unset>'}"
                          f"{layout_suffix} "
                          + shlex.join(build_command(config, group, task, ROOT)))
            return 0
        policy_dir = ROOT / config["policy_dir"]
        required_paths = [
            policy_dir / "eval.sh", policy_dir / "setup_eval_policy_server.sh",
            policy_dir / "setup_eval_env_client.sh", ROOT / "env_cfg" / f"{config['env_cfg']}.yml",
            *(ROOT / "task/RoboDojo/config" / f"{task}.yml" for task in config["tasks"]),
        ]
        for path in required_paths:
            if not path.is_file():
                raise FileNotFoundError(f"缺少文件：{path}")
        return run_groups(config, ROOT)
    except (OSError, ValueError) as exc:
        print(f"[eval] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
