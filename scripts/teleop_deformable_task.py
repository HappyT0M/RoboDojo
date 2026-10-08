"""Keyboard teleoperation for RoboDojo deformable-object tasks.

Run from the RoboDojo repository root in a graphical Isaac Sim session. The
current supported task is ``put_rope_ball_in_basket``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import json
import math
import os
import sys
import threading

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_XPOLICYLAB_ROOT = os.path.join(_PROJECT_ROOT, "XPolicyLab")
for _import_root in (_PROJECT_ROOT, _XPOLICYLAB_ROOT):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Keyboard teleoperation for RoboDojo rope tasks.")
parser.add_argument(
    "--task_name",
    choices=("put_rope_ball_in_basket",),
    default="put_rope_ball_in_basket",
    help="任务名（当前首版仅支持绳球入篮）",
)
parser.add_argument("--arm", choices=("left", "right"), default="left", help="遥操哪只机械臂")
parser.add_argument("--seed", type=int, default=0, help="场景种子")
parser.add_argument("--device_id", type=int, default=0, help="Isaac Sim 使用的 GPU 序号")
parser.add_argument("--out_dir", default="eval_result/teleop_rope", help="视频和动作记录输出目录")
parser.add_argument("--step_m", type=float, default=0.005, help="每个控制步的末端平移距离，单位米")
parser.add_argument(
    "--workspace_radius",
    type=float,
    default=0.30,
    help="末端相对 reset 位置在各坐标方向的最大活动半径，单位米",
)
parser.add_argument("--wind", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("WX", "WY", "WZ"))
parser.add_argument("--rigid_force_scale", type=float, default=1.0, help="刚体风力缩放系数")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.headless:
    parser.error("遥操需要 Isaac Sim 图形窗口，请移除 --headless")
if args_cli.device_id < 0:
    parser.error("--device_id 必须是非负整数")
if not math.isfinite(args_cli.step_m) or args_cli.step_m <= 0:
    parser.error("--step_m 必须是正的有限数")
if not math.isfinite(args_cli.workspace_radius) or args_cli.workspace_radius <= 0:
    parser.error("--workspace_radius 必须是正的有限数")
if not math.isfinite(args_cli.rigid_force_scale) or args_cli.rigid_force_scale < 0:
    parser.error("--rigid_force_scale 必须是非负有限数")

from utils.teleop_controls import (  # noqa: E402
    GRIPPER_KEYS,
    MOVEMENT_KEYS,
    apply_translation_keys,
    parse_wind,
    select_gripper_command,
    teleop_record_row,
)

try:
    wind = parse_wind(args_cli.wind)
except ValueError as exc:
    parser.error(str(exc))

args_cli.device = f"cuda:{args_cli.device_id}"
print(f"[teleop] Isaac Sim device={args_cli.device}; task={args_cli.task_name}; arm={args_cli.arm}; wind={wind}")

from env.camera_manager.capture.render_sync import add_zero_delay_kit_args  # noqa: E402

add_zero_delay_kit_args(args_cli)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omegaconf import OmegaConf  # noqa: E402

from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR  # noqa: E402
import src.eval_client.eval_env as eval_env_module  # noqa: E402
from src.eval_client.eval_env import create_eval_env  # noqa: E402
from utils.load_file import load_yaml  # noqa: E402
from utils.pipeline_utils import process_config, process_randomization, resolve_random_task_num_envs  # noqa: E402
from utils.rigid_wind import create_rigid_wind_controller, install_sim_step_wind_hook  # noqa: E402

_BENCHMARK_PATH = os.path.join(ROOT_DIR, "task", BENCHMARK)
_task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")


class _NoopModelClient:
    """Satisfy EvalEnv's client interface without connecting to a model."""

    def __init__(self, *args, **kwargs):
        self.connected = False

    def connect(self, *args, **kwargs):
        return True

    def call(self, *args, **kwargs):
        return None

    def close(self):
        pass


def _build_env(task_name: str):
    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, "arx_x5.yml"))
    eval_cfg["task_name"] = task_name
    eval_cfg["num_envs"] = 1
    eval_cfg["device_id"] = args_cli.device_id
    eval_cfg["eval_batch"] = False
    eval_cfg["policy_name"] = "noop_teleop"
    eval_cfg["additional_info"] = "keyboard_teleop"
    eval_cfg["seed"] = args_cli.seed
    eval_cfg["physx_monitor_enabled"] = False

    deploy_cfg = {
        "policy_name": "noop_teleop",
        "port": 0,
        "host": "localhost",
        "protocol": "ws",
        "policy_server_url": "ws://localhost:0",
        "evaluation_id": "keyboard_teleop",
        "trial_id": f"{task_name}-keyboard-teleop",
        "action_case_id": f"{task_name}_keyboard_teleop",
        "repeat_index": None,
    }
    env_cfg = OmegaConf.create(
        {
            "sim": load_yaml(
                os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")
            ),
            "scene": load_yaml(
                os.path.join(ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")
            ),
            "camera": load_yaml(
                os.path.join(ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml")
            ),
            "robot": load_yaml(
                os.path.join(ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml")
            ),
            "task_env": load_yaml(
                _task_registry.task_config_path(os.path.join(_BENCHMARK_PATH, "config"), task_name)
            ),
            "eval_cfg": eval_cfg,
            "deploy_cfg": deploy_cfg,
        }
    )
    num_envs = resolve_random_task_num_envs(task_name, 1, env_cfg.sim)
    eval_cfg["num_envs"] = num_envs
    OmegaConf.update(env_cfg, "sim.scene.num_envs", num_envs, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.num_envs", num_envs, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, eval_num = process_config(env_cfg, task_name=task_name)
    eval_cfg["eval_num"] = eval_num
    OmegaConf.update(
        env_cfg,
        "camera.default_frequency",
        eval_cfg["observation"].get("collect_freq", 0),
        force_add=True,
    )
    env_cfg.sim.seed = [args_cli.seed for _ in range(num_envs)]

    eval_env_module.WsModelClient = _NoopModelClient
    return create_eval_env(env_cfg, simulation_app, resume_state=None)


def _get_target_robots(env):
    targets = {
        robot.arm_name: robot
        for robot in env.robot_manager.robot_list
        if robot.type == "target"
    }
    required = {"left_arm", "right_arm"}
    if not required.issubset(targets):
        raise RuntimeError(f"双臂任务需要目标机器人 {sorted(required)}，当前发现 {sorted(targets)}")
    return targets


def _video_stem(task_name: str, seed: int, wind_vector: tuple[float, float, float]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    speed = math.sqrt(sum(value * value for value in wind_vector))
    speed_token = f"{speed:.3f}".replace(".", "p")
    wind_token = "yes" if any(wind_vector) else "no"
    return f"teleop_{task_name}_seed-{seed:04d}_{timestamp}_wind-{wind_token}_speed-{speed_token}"


def run_teleop() -> None:
    env = None
    subscription = None
    input_interface = None
    keyboard = None
    record_file = None
    video_path = None
    video_saved = False
    try:
        env = _build_env(args_cli.task_name)
        env.reset(seed=[args_cli.seed])
        env.run_reward()
        if hasattr(env, "get_score"):
            env.get_score()

        robots = _get_target_robots(env)
        active_robot = robots[f"{args_cli.arm}_arm"]
        targets = {
            side: env.robot_manager.get_delta_endpose(robot, env_idx_list=[0])[0].astype(float).tolist()
            for side, robot in (("left", robots["left_arm"]), ("right", robots["right_arm"]))
        }
        current_joints = {
            side: env.robot_manager.get_joint(robot, env_idx_list=[0])[0].astype(float).tolist()
            for side, robot in (("left", robots["left_arm"]), ("right", robots["right_arm"]))
        }
        gripper = {"left": 1.0, "right": 1.0}
        active_side = args_cli.arm
        anchor = targets[active_side][:3]
        bounds = tuple(
            (coordinate - args_cli.workspace_radius, coordinate + args_cli.workspace_radius)
            for coordinate in anchor
        )

        output_dir = os.path.join(args_cli.out_dir, args_cli.task_name)
        os.makedirs(output_dir, exist_ok=True)
        stem = _video_stem(args_cli.task_name, args_cli.seed, wind)
        jsonl_path = os.path.join(output_dir, f"{stem}_actions.jsonl")
        video_path = os.path.join(output_dir, f"{stem}.mp4")
        record_file = open(jsonl_path, "w", encoding="utf-8")

        wind_controller = None
        wind_skipped = []
        if any(wind) and args_cli.rigid_force_scale > 0:
            wind_controller, wind_skipped = create_rigid_wind_controller(
                env.scene_manager,
                wind,
                force_scale=args_cli.rigid_force_scale,
            )
            if wind_controller is not None:
                install_sim_step_wind_hook(env, wind_controller)
            else:
                print("[teleop][wind][WARN] 未发现可施加风力的动态刚体")
            for item in wind_skipped:
                print(f"[teleop][wind][skip] {item['label']}: {item['reason']}")

        import carb
        import omni.appwindow

        app_window = omni.appwindow.get_default_app_window()
        if app_window is None or app_window.get_keyboard() is None:
            raise RuntimeError("没有获取到 Isaac Sim 图形窗口键盘；请使用带 GUI 的本地/远程桌面运行")
        keyboard = app_window.get_keyboard()
        held_keys: set[str] = set()
        stop_event = threading.Event()
        keyboard_inputs = {
            getattr(carb.input.KeyboardInput, key): key
            for key in MOVEMENT_KEYS | GRIPPER_KEYS
            if hasattr(carb.input.KeyboardInput, key)
        }
        escape_input = getattr(
            carb.input.KeyboardInput,
            "ESCAPE",
            getattr(carb.input.KeyboardInput, "ESC", None),
        )

        def on_keyboard_event(event):
            key = keyboard_inputs.get(event.input)
            if event.type in {
                carb.input.KeyboardEventType.KEY_PRESS,
                carb.input.KeyboardEventType.KEY_REPEAT,
            }:
                if event.input == escape_input:
                    stop_event.set()
                elif key is not None:
                    held_keys.add(key)
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                if key is not None:
                    held_keys.discard(key)
            return True

        input_interface = carb.input.acquire_input_interface()
        subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

        print("[teleop] 场景已启动。请点击 Isaac Sim 视口使键盘获得焦点。")
        print("[teleop] W/S: X+/X- | D/A: Y+/Y- | R/F: Z+/Z- | C: 闭合夹爪 | O: 打开夹爪 | Esc: 保存并退出")
        print(f"[teleop] arm={args_cli.arm}, step_m={args_cli.step_m}, workspace_radius={args_cli.workspace_radius}")
        print(f"[teleop] actions={jsonl_path}")

        step = 0
        last_ik_target = current_joints[active_side].copy()
        while (
            simulation_app.is_running()
            and not stop_event.is_set()
            and not env.end_flag[0]
            and step < env.step_lim
        ):
            simulation_app.update()
            keys = set(held_keys)
            candidate = apply_translation_keys(
                targets[active_side],
                keys,
                step_m=args_cli.step_m,
                bounds=bounds,
            )
            if not np.allclose(candidate, targets[active_side], rtol=0.0, atol=1e-12):
                ik_result = env.robot_manager.solve_ik(
                    target_pose=candidate,
                    env_idx=0,
                    robot=active_robot,
                )
                if ik_result.get("status") == "Success":
                    targets[active_side] = candidate.tolist()
                    last_ik_target = np.asarray(ik_result["joint_value"], dtype=float).tolist()
                    current_joints[active_side] = last_ik_target
                else:
                    print(f"[teleop][IK] {args_cli.arm} 手臂无法到达目标，保持上一目标")

            gripper[active_side] = select_gripper_command(keys, gripper[active_side])
            action = {}
            for side in ("left", "right"):
                robot = robots[f"{side}_arm"]
                arm_key = env.robot_manager.process_name(robot.arm_name)
                gripper_key = env.robot_manager.process_name(robot.gripper_name)
                action[arm_key] = current_joints[side]
                action[gripper_key] = [gripper[side]]

            env.take_action(action)
            env.get_obs()
            row = teleop_record_row(
                step=step,
                held_keys=keys,
                arm=active_side,
                target_pose=targets[active_side],
                gripper=gripper[active_side],
                wind=wind,
                task=args_cli.task_name,
                seed=args_cli.seed,
            )
            row["joint_target"] = list(last_ik_target)
            record_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            record_file.flush()
            step += 1
            if step % 50 == 0:
                print(f"[teleop] 已记录 {step} 个控制步")

        env.get_obs_batch(last_frame=True)
        env.end_flag = [True] * env.num_envs
        env.save_video(0, video_path, tag="keyboard_teleop")
        video_saved = True
        print(f"[teleop] 视频基础路径: {video_path}")
        print(f"[teleop] 动作记录: {jsonl_path}")
        if wind_controller is not None:
            print(f"[teleop][wind] body_count={wind_controller.last_diagnostics['body_count']}")
    finally:
        if record_file is not None:
            record_file.flush()
            record_file.close()
        if subscription is not None and input_interface is not None:
            try:
                input_interface.unsubscribe_to_keyboard_events(keyboard, subscription)
            except Exception as exc:
                print(f"[teleop][WARN] 注销键盘监听失败: {exc}")
        if env is not None:
            if video_path is not None and not video_saved:
                try:
                    env.end_flag = [True] * env.num_envs
                    env.get_obs_batch(last_frame=True)
                    env.save_video(0, video_path, tag="keyboard_teleop")
                    print(f"[teleop] 已尽力保存中断前的视频: {video_path}")
                except Exception as exc:
                    print(f"[teleop][WARN] 中断时视频未能完整保存: {exc}")
            try:
                env.close()
            except Exception as exc:
                print(f"[teleop][WARN] 关闭环境失败: {exc}")
        simulation_app.close()


if __name__ == "__main__":
    run_teleop()
