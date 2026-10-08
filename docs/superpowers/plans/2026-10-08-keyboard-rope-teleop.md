# 键盘遥操绳球任务实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 RoboDojo 当前可运行的 `put_rope_ball_in_basket` 任务新增 Isaac Sim 图形窗口键盘遥操入口，能控制选择的 X5 手臂并保存视频和动作记录。

**Architecture:** 将键盘到笛卡尔末端目标的纯计算逻辑放入无 Isaac 依赖的 `utils/teleop_controls.py`，以单元测试验证。新增 `scripts/teleop_deformable_task.py` 负责启动 Isaac Sim、加载现有评测环境、订阅 Carb 键盘事件、调用 RoboDojo IK/动作入口及写出录像。通过 `--arm` 选择左/右手，另一只手每步维持当前末端目标和夹爪位置；可选复用 `utils.rigid_wind` 施加风力。

**Tech Stack:** Python、pytest、Isaac Sim/Isaac Lab AppLauncher、Carb Input、RoboDojo EvalEnv、NumPy、JSONL、现有视频采集工具。

---

## 文件职责

- 新建 `utils/teleop_controls.py`：按键状态映射、末端目标位置步进/边界限制、风速参数校验、JSONL 记录行构建。不得导入 Isaac Sim。
- 新建 `tests/test_teleop_controls.py`：覆盖平移方向、反向按键抵消、工作空间限位、夹爪开合优先级、风速校验和 JSON 序列化字段。
- 新建 `scripts/teleop_deformable_task.py`：运行图形应用、创建任务环境、输入事件订阅、左/右臂目标保持、可选刚体风力 hook、录像和数据清理。
- 修改 `scripts/README.md`：补充启动命令、图形模式要求及键位表。

### Task 1: 纯 Python 控制与记录辅助函数

**Files:**
- Create: `tests/test_teleop_controls.py`
- Create: `utils/teleop_controls.py`

- [x] **Step 1: 先写会失败的测试**

```python
import json

import numpy as np
import pytest

from utils.teleop_controls import apply_translation_keys, parse_wind, select_gripper_command, teleop_record_row


def test_translation_keys_move_xyz_without_changing_orientation():
    pose = [0.1, -0.2, 1.0, 1.0, 0.0, 0.0, 0.0]
    result = apply_translation_keys(pose, {"W", "D", "R"}, step_m=0.01, bounds=((-1, 1), (-1, 1), (0, 2)))
    np.testing.assert_allclose(result, [0.11, -0.19, 1.01, 1, 0, 0, 0])


def test_opposing_translation_keys_cancel_and_bounds_clip():
    result = apply_translation_keys([0.99, 0, 1, 1, 0, 0, 0], {"W", "S", "D"}, 0.05, ((-1, 1), (-1, 1), (0, 2)))
    np.testing.assert_allclose(result[:3], [0.99, 0.05, 1])


def test_gripper_key_selects_open_or_closed_fraction():
    assert select_gripper_command({"C"}, current=1.0) == 0.0
    assert select_gripper_command({"O"}, current=0.0) == 1.0
    assert select_gripper_command(set(), current=0.4) == 0.4


def test_parse_wind_requires_finite_three_vector():
    assert parse_wind((0, 0.5, 0)) == (0.0, 0.5, 0.0)
    with pytest.raises(ValueError):
        parse_wind((0, float("nan"), 0))
    with pytest.raises(ValueError):
        parse_wind((1, 2))


def test_record_row_is_json_serializable_and_contains_replay_context():
    row = teleop_record_row(
        step=3, held_keys={"W"}, arm="left", target_pose=[0, 0, 1, 1, 0, 0, 0],
        gripper=1.0, wind=(0, 0, 0), task="put_rope_ball_in_basket", seed=0,
    )
    decoded = json.loads(json.dumps(row))
    assert decoded["step"] == 3
    assert decoded["arm"] == "left"
    assert decoded["held_keys"] == ["W"]
    assert decoded["wind"] == [0, 0, 0]
```

- [x] **Step 2: 运行测试并确认因辅助模块不存在而失败**

Run: `python -m pytest tests/test_teleop_controls.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.teleop_controls'`.

- [x] **Step 3: 写最小辅助实现**

```python
from __future__ import annotations

from typing import Iterable, Sequence
import math
import numpy as np

_MOVE = {"W": (0, 1), "S": (0, -1), "D": (1, 1), "A": (1, -1), "R": (2, 1), "F": (2, -1)}


def apply_translation_keys(target_pose, held_keys, step_m, bounds):
    pose = np.asarray(target_pose, dtype=np.float64).copy()
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("target_pose must be a finite 7D pose")
    if not math.isfinite(step_m) or step_m <= 0:
        raise ValueError("step_m must be positive and finite")
    limits = np.asarray(bounds, dtype=np.float64)
    if limits.shape != (3, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("bounds must contain three finite (min, max) pairs")
    delta = np.zeros(3)
    for key in set(held_keys):
        if key in _MOVE:
            axis, sign = _MOVE[key]
            delta[axis] += sign * step_m
    pose[:3] = np.clip(pose[:3] + delta, limits[:, 0], limits[:, 1])
    return pose


def select_gripper_command(held_keys, current):
    keys = set(held_keys)
    if "C" in keys and "O" not in keys:
        return 0.0
    if "O" in keys and "C" not in keys:
        return 1.0
    return float(current)


def parse_wind(values):
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError("wind must be a finite 3D vector")
    return result


def teleop_record_row(step, held_keys, arm, target_pose, gripper, wind, task, seed):
    return {"step": int(step), "held_keys": sorted(set(held_keys)), "arm": str(arm),
            "target_pose": [float(v) for v in target_pose], "gripper": float(gripper),
            "wind": [float(v) for v in wind], "task": str(task), "seed": int(seed)}
```

- [x] **Step 4: 再运行纯控制测试**

Run: `python -m pytest tests/test_teleop_controls.py -q`
Expected: all tests PASS.

### Task 2: 图形窗口遥操入口

**Files:**
- Create: `scripts/teleop_deformable_task.py`

- [x] **Step 1: 按现有录制入口组装环境与空策略客户端**
  - 在 `AppLauncher` 前解析 `--task_name`（默认 `put_rope_ball_in_basket`）、`--arm {left,right}`、`--seed`、`--device_id`、`--out_dir`、`--step_m`、`--workspace_radius`、`--wind WX WY WZ`、`--rigid_force_scale`，并调用 `AppLauncher.add_app_launcher_args`。
  - 从 `record_deformable_tasks.py` 复用配置加载逻辑；将 `eval_env_module.WsModelClient` 替换为不连接服务的客户端，配置 `num_envs=1`，并调用 `create_eval_env`。
  - 从 `robot_manager.robot_list` 找到 `arm_name` 分别为 `left_arm` 和 `right_arm` 的目标机器人；动作字典使用 `left_ee_pose`、`right_ee_pose`、`left_ee_joint_state`、`right_ee_joint_state` 四个键（单臂关闭度通过 `ee_joint_state` 的第一个归一化数传入）。

- [x] **Step 2: 在 Carb 键盘事件中维护按住键集合**

```python
def on_keyboard_event(event):
    key = _keyboard_key_name(event.input)
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if key == "ESCAPE":
            stop_event.set()
        elif key in MOVEMENT_KEYS | {"O", "C"}:
            held_keys.add(key)
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        held_keys.discard(key)
    return True
```

  - 通过 `omni.appwindow.get_default_app_window()` 获得窗口键盘；若窗口或键盘不可用，输出明确的 GUI 模式提示并退出。
  - 在 `finally` 中调用 `unsubscribe_to_keyboard_events(keyboard, subscription_id)` 注销输入订阅。

- [x] **Step 3: 每帧更新目标、发送动作、保存记录**
  - Reset 后读取左右末端目标位姿 `robot_manager.get_delta_endpose(..., env_idx_list=[0])` 和夹爪开度；对所选臂初始化目标，另一臂目标与夹爪保持。
  - 将 `apply_translation_keys` 的工作空间边界设置为 reset 位姿 XYZ 各自 ± `workspace_radius`；每步只更新所选臂目标，四元数不变。
  - 仅对目标手调用 `robot_manager.solve_ik(target_pose, env_idx=0, robot=active_robot)`；成功时更新该手关节目标，失败时保留上次目标并提示。为 `env.take_action()` 构造包含左右 `*_arm_joint_state` 及 `*_ee_joint_state` 夹爪开度的完整 joint action，非活动手每步保持 reset 关节目标；随后调用 `env.get_obs()`。
  - 无风时不创建 wind controller；非零风速且 `rigid_force_scale > 0` 时创建 `create_rigid_wind_controller(env.scene_manager, wind, force_scale=...)` 并用 `install_sim_step_wind_hook` 挂入 `env.sim_step`。
  - 每个遥操控制步用 `teleop_record_row` 构造记录并附加所选臂的 IK 关节目标，写入 JSONL 后调用 `flush()`，保证异常退出时已有记录落盘。

- [x] **Step 4: 按退出键后收尾录像并释放资源**
  - 输出目录为 `out_dir/task_name/`；用时间戳、风开关和风速模长生成不会覆盖的录像基础名及 JSONL 文件名。
  - 设置 `env.end_flag`，通过现有 `env.save_video` 为环境 0 落盘；无论成功或异常都关闭 JSONL、注销键盘订阅、关闭环境和 `simulation_app`。
  - 启动时打印键位：`W/S` X、`A/D` Y、`R/F` Z、`C/O` 夹爪闭合/打开、`Esc` 保存并退出；保持末端姿态不变。

### Task 3: 文档、测试与运行说明

**Files:**
- Modify: `scripts/README.md`
- Test: `tests/test_teleop_controls.py`
- Check: `scripts/teleop_deformable_task.py`, `utils/teleop_controls.py`

- [x] **Step 1: README 加运行示例与键位表**

```bash
python scripts/teleop_deformable_task.py --task_name put_rope_ball_in_basket --arm left --seed 0 --device_id 0 --wind 0 0 0 --out_dir eval_result/teleop_rope
```

  - 说明必须在非 headless Isaac Sim 图形窗口中运行；默认无风，将 `--wind` 改为例如 `0 0.5 0` 可启用风。
  - 列出按键与 xyz 方向、夹爪状态，并说明视频与 JSONL 保存位置。

- [x] **Step 2: 跑单元测试和静态检查**

Run: `python -m pytest tests/test_teleop_controls.py tests/test_rigid_wind.py tests/test_cloth_wind.py -q`
Expected: all tests PASS.

Run: `python -m compileall -q scripts/teleop_deformable_task.py utils/teleop_controls.py`
Expected: exit code 0.

Run: `git diff --check`
Expected: no whitespace errors.

- [ ] **Step 3: 在目标 Isaac Sim 环境进行 GUI 集成验收**
  - 运行 README 命令，确认窗口显示篮子和绳球；按 `W` 后活动手末端目标向 +X 变化，另一只手不动；按 `C`/`O` 确认夹爪闭合/张开；按 `Esc` 后生成各摄像机 MP4 和 JSONL。
  - 再用非零风速重跑，确认 JSONL 中风向/风速正确，终端诊断显示至少一个动态任务刚体受力。
  - 本地缺少 Isaac Sim 时只报告纯 Python 验证结果，不声称 GUI/PhysX 集成已经通过。

- [x] **Step 4: 提交本次功能文件**

```bash
git add scripts/teleop_deformable_task.py scripts/README.md utils/teleop_controls.py tests/test_teleop_controls.py
git commit -m "feat: add keyboard teleoperation for rope task"
```

---

## 规格覆盖自检

- 图形窗口与键盘事件：Task 2 步骤 1–2。
- 左右手选择、末端 XYZ 平移、固定姿态、夹爪控制与另一只手保持：Task 2 步骤 3。
- 无策略服务、风力可选：Task 2 步骤 1 和步骤 3。
- 多摄像机视频、动作 JSONL、时间戳文件名和资源释放：Task 2 步骤 3–4。
- 纯 Python 测试与 Isaac Sim 目标环境集成验收：Task 1 和 Task 3。
