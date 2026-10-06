"""在不接入策略、不修改任务文件的情况下录制恒定风扰动任务。

在 RoboDojo 根目录运行，例如：

    python scripts/record_deformable_constant_wind.py \
        --tasks fold_clothes,fold_clothes_random \
        --duration_s 10 \
        --wind 0.5 0.0 0.0 \
        --device_id 1 \
        --out_dir eval_result/constant_wind_preview

    python scripts/record_deformable_constant_wind.py \
        --task_name put_rope_ball_in_basket \
        --duration_s 10 \
        --wind 0.0 0.5 0.0 \
        --rigid_force_scale 100 \
        --device_id 0 \
        --disable_xlens_snapshot \
        --out_dir eval_result/rope_basket_wind --headless

脚本只修改内存中的运行配置，不修改 ``task/RoboDojo/config`` 下的 YAML。
布料使用 Isaac Sim 粒子系统风速；可移动刚体和任务关节链 link 使用逐物体计算的
空气阻力。机器人与静态场景资产不受风。``--wind WX WY WZ`` 表示恒定环境风速
（单位 m/s），``--rigid_force_scale`` 用于调节简化刚体风阻的可观察程度。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import json
import math
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_XPOLICYLAB_ROOT = os.path.join(_PROJECT_ROOT, "XPolicyLab")
for _root in (_PROJECT_ROOT, _XPOLICYLAB_ROOT):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from isaaclab.app import AppLauncher


def _nonnegative_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


parser = argparse.ArgumentParser(description="Record RoboDojo tasks under constant wind perturbations.")
parser.add_argument("--task_name", type=str, default=None, help="单个任务名，与 --tasks 二选一")
parser.add_argument(
    "--tasks",
    type=str,
    default=None,
    help="逗号分隔的任务名；默认 fold_clothes,fold_clothes_random",
)
parser.add_argument("--duration_s", type=float, default=10.0, help="每个任务录制时长，单位秒")
parser.add_argument(
    "--wind",
    type=float,
    nargs=3,
    default=(0.5, 0.0, 0.0),
    metavar=("WX", "WY", "WZ"),
    help="恒定环境风速向量，单位 m/s；默认 0.5 0 0，传入 0 0 0 可关闭风力",
)
parser.add_argument(
    "--no-wind",
    action="store_true",
    help="关闭恒定风力（与 --wind 同时指定时以 --no-wind 为准）",
)
parser.add_argument(
    "--particle_drag",
    type=float,
    default=0.5,
    help="布料粒子材质 drag；默认 0.5，便于观察风力响应",
)
parser.add_argument(
    "--rigid_drag_coefficient",
    type=_nonnegative_finite_float,
    default=1.0,
    help="刚体风阻系数 Cd；默认 1.0",
)
parser.add_argument(
    "--rigid_force_scale",
    type=_nonnegative_finite_float,
    default=100.0,
    help="刚体风阻缩放系数；默认 100，越大扰动越明显，0 关闭刚体风力分量",
)
parser.add_argument("--num_envs", type=int, default=1, help="并行环境数，录制时建议保持 1")
parser.add_argument("--env_cfg_type", type=str, default="arx_x5", help="env_cfg 下的配置名，不含 .yml")
parser.add_argument(
    "--device_id",
    type=int,
    default=None,
    help="物理 GPU id；会在启动 Isaac Sim 前设置 CUDA_VISIBLE_DEVICES。未指定时尊重外部环境变量",
)
parser.add_argument("--seed", type=int, default=0, help="场景随机种子")
parser.add_argument(
    "--out_dir",
    type=str,
    default="eval_result/constant_wind_preview",
    help="视频输出根目录",
)
parser.add_argument("--additional_info", type=str, default="constant_wind_preview")
parser.add_argument(
    "--disable_xlens_snapshot",
    action="store_true",
    help="关闭 reset 后的 XLens 场景快照导出",
)
parser.add_argument(
    "--xlens_snapshot_dir",
    type=str,
    default="logs/xlens_geometry_constant_wind",
    help="XLens 场景快照输出根目录",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 必须在 AppLauncher(args_cli) 之前选择 GPU。仅把 device_id 写入 eval_cfg
# 不会改变 Isaac Sim 已经选择的默认卡；这里兼容集群上 CUDA_VISIBLE_DEVICES
# 已由作业调度器设置的情况。
if args_cli.device_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.device_id)
elif "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"[wind-record] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR  # noqa: E402
from env.camera_manager.capture.render_sync import add_zero_delay_kit_args  # noqa: E402

add_zero_delay_kit_args(args_cli)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omegaconf import OmegaConf  # noqa: E402

import src.eval_client.eval_env as eval_env_module  # noqa: E402
from env.observation_manager.obs_manager import ObsManager  # noqa: E402
from src.eval_client.eval_env import create_eval_env  # noqa: E402
from utils.load_file import load_yaml  # noqa: E402
from utils.pipeline_utils import process_config, process_randomization, resolve_random_task_num_envs  # noqa: E402
from utils.cloth_wind import apply_constant_cloth_wind_if_present, collect_cloth_wind_readbacks  # noqa: E402
from utils.rigid_wind import create_rigid_wind_controller, install_sim_step_wind_hook  # noqa: E402
from utils.xlens_snapshot import populate_manifest_objects, save_robodojo_snapshot  # noqa: E402

BENCHMARK_PATH = os.path.join(ROOT_DIR, "task", BENCHMARK)
task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")


class _NoopModelClient:
    """与评测入口兼容但不连接任何 VLA/WAM 服务的占位客户端。"""

    def __init__(self, *args, **kwargs):
        self.connected = False

    def connect(self, *args, **kwargs):
        return True

    def call(self, func_name, *args, **kwargs):
        return None

    def close(self):
        return None


def _configure_noop_mode() -> None:
    eval_env_module.WsModelClient = _NoopModelClient
    # 保持现有观察采集方式，同时让快照工具可以读取分割结果。
    ObsManager.ANNOTATORS_TO_COLLECT.update(
        {
            "instance_id_segmentation_fast": "instance_id",
            "instance_segmentation_fast": "instance",
            "semantic_segmentation": "semantic",
        }
    )


def _resolve_tasks() -> list[str]:
    if args_cli.tasks:
        tasks = [name.strip() for name in args_cli.tasks.split(",") if name.strip()]
    elif args_cli.task_name:
        tasks = [args_cli.task_name]
    else:
        tasks = ["fold_clothes", "fold_clothes_random"]
    if not tasks:
        raise ValueError("至少需要一个任务名")
    return tasks


def _inject_constant_wind(task_cfg: dict) -> None:
    """只修改内存中的任务配置，不触碰原始 YAML 文件。"""
    wind = _selected_wind()
    has_wind = any(value != 0.0 for value in wind)
    apply_constant_cloth_wind_if_present(
        task_cfg,
        wind,
        particle_drag=float(args_cli.particle_drag) if has_wind else None,
    )


def _selected_wind() -> list[float]:
    return [0.0, 0.0, 0.0] if args_cli.no_wind else [float(value) for value in args_cli.wind]


def _log_wind_readback(
    env,
    task_name: str,
    phase: str,
    log_path: str,
    rigid_wind_controller=None,
    rigid_wind_skipped=None,
) -> dict:
    """Record the requested wind and values read from live particle/rigid views."""
    requested_wind = _selected_wind()
    scene_manager = getattr(env, "scene_manager", None)
    garments_by_env = getattr(scene_manager, "_garment_objects", [])
    garments = collect_cloth_wind_readbacks(garments_by_env, requested_wind)
    matches = all(item["matches_request"] for item in garments) if garments else None
    rigid_wind = (
        rigid_wind_controller.last_diagnostics
        if rigid_wind_controller is not None
        else {"body_count": 0, "applied_body_count": 0, "bodies": []}
    )
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task_name": task_name,
        "phase": phase,
        "requested_wind": requested_wind,
        "requested_particle_drag": float(args_cli.particle_drag)
        if any(value != 0.0 for value in requested_wind)
        else None,
        "requested_rigid_drag_coefficient": float(args_cli.rigid_drag_coefficient),
        "requested_rigid_force_scale": float(args_cli.rigid_force_scale),
        "garment_count": len(garments),
        "all_match_request": matches,
        "garments": garments,
        "rigid_wind": rigid_wind,
        "rigid_wind_skipped": rigid_wind_skipped or [],
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
    except OSError as exc:
        print(f"[wind-record][WARN] 无法写入风力诊断日志 {log_path}: {exc}")

    print(
        f"[wind-record][wind-readback] task={task_name} phase={phase} "
        f"requested={requested_wind} garments={len(garments)} all_match={matches}"
    )
    for item in garments:
        print(
            f"[wind-record][wind-readback] env={item['env_id']} "
            f"garment={item['garment_name']} path={item['particle_system_path']} "
            f"actual={item['actual_wind']} matches={item['matches_request']} "
            f"error={item['readback_error']}"
        )
    if garments and not matches:
        print("[wind-record][WARN] 实际风速读回失败或与请求值不一致")
    if rigid_wind_controller is not None:
        print(
            f"[wind-record][rigid-wind] task={task_name} phase={phase} "
            f"bodies={rigid_wind['body_count']} applied={rigid_wind['applied_body_count']} "
            f"force_scale={args_cli.rigid_force_scale}"
        )
    if rigid_wind_skipped:
        for item in rigid_wind_skipped:
            print(f"[wind-record][rigid-wind][skip] {item['label']}: {item['reason']}")
    return record


def _build_env(task_name: str):
    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, args_cli.env_cfg_type + ".yml"))
    eval_cfg["task_name"] = task_name
    eval_cfg["num_envs"] = args_cli.num_envs
    eval_cfg["device_id"] = args_cli.device_id
    eval_cfg["eval_batch"] = False
    eval_cfg["policy_name"] = "noop_constant_wind"
    eval_cfg["additional_info"] = args_cli.additional_info
    eval_cfg["seed"] = args_cli.seed
    eval_cfg["physx_monitor_enabled"] = False

    deploy_cfg = {
        "policy_name": "noop_constant_wind",
        "port": 0,
        "host": "localhost",
        "protocol": "ws",
        "policy_server_url": "ws://localhost:0",
        "evaluation_id": "constant_wind_preview",
        "trial_id": f"{task_name}-constant-wind",
        "action_case_id": f"{task_name}_constant_wind",
        "repeat_index": None,
    }

    task_cfg = load_yaml(task_registry.task_config_path(os.path.join(BENCHMARK_PATH, "config"), task_name))
    _inject_constant_wind(task_cfg)
    env_cfg = OmegaConf.create(
        {
            "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")),
            "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")),
            "camera": load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml")),
            "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml")),
            "task_env": task_cfg,
            "eval_cfg": eval_cfg,
            "deploy_cfg": deploy_cfg,
        }
    )

    num_envs = resolve_random_task_num_envs(task_name, args_cli.num_envs, env_cfg.sim)
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

    _configure_noop_mode()
    return create_eval_env(env_cfg, simulation_app, resume_state=None), num_envs


def _neutral_action(env) -> dict:
    """保持机械臂初始关节目标，夹爪保持张开。"""
    action = {}
    for robot in env.robot_manager.robot_list:
        if robot.type != "target":
            continue
        arm_key = env.robot_manager.process_name(robot.arm_name)
        gripper_key = env.robot_manager.process_name(robot.gripper_name)
        joints = env.robot_manager.get_joint(robot, env_idx_list=[0])[0]
        if joints is not None:
            action[arm_key] = list(joints)
        action[gripper_key] = [1.0]
    return action


def _steps_for_duration(env) -> int:
    """按录像采集频率计算高层动作步数；每个动作对应一帧采集。"""
    collect_freq = float(getattr(env.obs_manager, "collect_freq", 0) or 0)
    if collect_freq <= 0:
        raise ValueError("当前相机 observation.collect_freq 必须大于 0 才能按秒数录制")
    return max(1, int(round(args_cli.duration_s * collect_freq)))


def _record_task(task_name: str) -> str:
    print(f"\n{'=' * 70}\n[wind-record] 开始任务: {task_name}\n{'=' * 70}")
    env, num_envs = _build_env(task_name)
    original_sim_step = None
    try:
        record_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        task_output_dir = os.path.join(args_cli.out_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        wind_log_path = os.path.join(task_output_dir, f"wind_diagnostics_{record_timestamp}.jsonl")
        env.reset(seed=[args_cli.seed])
        wind = _selected_wind()
        has_wind = any(value != 0.0 for value in wind)
        rigid_wind_controller = None
        rigid_wind_skipped = []
        if has_wind and args_cli.rigid_force_scale > 0.0:
            rigid_wind_controller, rigid_wind_skipped = create_rigid_wind_controller(
                env.scene_manager,
                wind,
                drag_coefficient=args_cli.rigid_drag_coefficient,
                force_scale=args_cli.rigid_force_scale,
            )
        if rigid_wind_controller is not None:
            original_sim_step = install_sim_step_wind_hook(env, rigid_wind_controller)
        garments_by_env = getattr(env.scene_manager, "_garment_objects", [])
        has_garments = any(bool(objects) for objects in garments_by_env)
        if has_wind and not has_garments and rigid_wind_controller is None:
            raise RuntimeError(
                "风力未应用：任务中没有布料，且没有找到可施力的动态任务刚体。"
                "请查看刚体筛选诊断或确认资产带有 RigidBodyAPI。"
            )
        _log_wind_readback(
            env,
            task_name,
            "after_reset",
            wind_log_path,
            rigid_wind_controller=rigid_wind_controller,
            rigid_wind_skipped=rigid_wind_skipped,
        )
        if not args_cli.disable_xlens_snapshot and args_cli.xlens_snapshot_dir:
            snapshot_dir = os.path.join(
                args_cli.xlens_snapshot_dir,
                f"{task_name}_episode_{args_cli.seed:04d}",
            )
            manifest_path = save_robodojo_snapshot(
                env,
                snapshot_dir,
                task_name=task_name,
                env_id=0,
                objects=[],
            )
            objects = populate_manifest_objects(manifest_path, min_pixels=50)
            print(f"[XLens] scene manifest: {manifest_path}")
            print(f"[XLens] visible instances: {len(objects)}")

        env.run_reward()
        if hasattr(env, "get_score"):
            env.get_score()
        action = _neutral_action(env)
        steps = _steps_for_duration(env)
        print(
            f"[wind-record] task={task_name} num_envs={num_envs} "
            f"duration≈{args_cli.duration_s:.2f}s steps={steps} "
            f"wind={wind} ({'on' if has_wind else 'off'})"
            f"{' cloth_drag=' + str(args_cli.particle_drag) if has_wind and has_garments else ''}"
            f"{' rigid_force_scale=' + str(args_cli.rigid_force_scale) if rigid_wind_controller else ''}"
        )

        for step in range(steps):
            env.take_action(action)
            env.get_obs()
            if step == 0:
                _log_wind_readback(
                    env,
                    task_name,
                    "after_first_step",
                    wind_log_path,
                    rigid_wind_controller=rigid_wind_controller,
                    rigid_wind_skipped=rigid_wind_skipped,
                )
            if (step + 1) % max(1, int(getattr(env.obs_manager, "collect_freq", 25))) == 0:
                print(f"[wind-record] {task_name}: {step + 1}/{steps}")

        env.end_flag = [True] * num_envs
        env.success = [True] * num_envs
        wind_speed = math.sqrt(sum(value ** 2 for value in wind))
        speed_token = f"{wind_speed:.3f}".replace("-", "m").replace(".", "p")
        wind_token = "yes" if has_wind else "no"
        video_tag = "constant_wind" if has_wind else "no_wind"
        # save_video 会在这个基础名后追加具体相机名和 tag。
        # 文件名示例：record_20260928_143012_123456_wind-yes_speed-0p500_cam_head_constant_wind.mp4
        video_name = f"record_{record_timestamp}_wind-{wind_token}_speed-{speed_token}.mp4"
        video_path = os.path.join(task_output_dir, video_name)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        env.save_video(0, video_path, tag=video_tag)
        print(f"[wind-record] 完成: {task_output_dir}")
        print(f"[wind-record] 风力诊断: {wind_log_path}")
        return task_output_dir
    finally:
        if original_sim_step is not None:
            env.sim_step = original_sim_step
        env.close()


def main() -> None:
    output_dirs = []
    try:
        for task_name in _resolve_tasks():
            try:
                output_dirs.append(_record_task(task_name))
            except Exception as exc:
                print(f"[wind-record] {task_name} 失败: {type(exc).__name__}: {exc}")
                import traceback

                traceback.print_exc()
        print("\n[wind-record] 输出目录:")
        for output_dir in output_dirs:
            print(f"  - {output_dir}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
