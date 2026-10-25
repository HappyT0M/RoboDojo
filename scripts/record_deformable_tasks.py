"""无需加载策略权重，预览可形变任务在无风或恒定风下的场景。

用法示例（在 RoboDojo 根目录下运行）::

    python scripts/record_deformable_tasks.py \
        --task_name fold_clothes \
        --num_envs 1 \
        --env_cfg_type arx_x5 \
        --device_id 0 \
        --seed 0 \
        --layout_id 0 \
        --steps 100 \
        --out_dir eval_result/preview

    # 以恒定 +Y 风速录制任务的所有布局，每个布局 10 秒（风速单位 m/s）
    python scripts/record_deformable_tasks.py \
        --task_name put_rope_ball_in_basket \
        --duration_s 10 --wind 0 0.5 0 \
        --rigid_force_scale 100 --out_dir eval_result/preview

    # 一次性录制两个任务
    python scripts/record_deformable_tasks.py --tasks fold_clothes,fold_clothes_random

说明：
    * 复用 RoboDojo 官方的 ``create_eval_env`` 与摄像头/渲染管线，仅把策略客户端
      替换为“空操作”占位对象，因此不接入任何 VLA / 动作模型。
    * 默认依次录制指定任务的全部已有布局；``--layout_id`` 可只录制一个布局。
      ``--seed`` 选择 Eval_Layout 下的目录，不是布局编号。
    * 每个 episode 机械臂保持初始位姿、夹爪保持张开，仅推进物理仿真并采集摄像头画面。
    * ``--wind WX WY WZ`` 设置世界坐标中的恒定风速；默认 0 0 0。布料使用粒子风速，
      可移动刚体及任务关节链 link 使用简化空气阻力；不修改任务 YAML。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import json
import math
import os
from pathlib import Path
import sys

# 将项目根目录加入 sys.path，确保无论从哪个目录运行都能导入 env/task/utils/src 等包。
# 官方入口（scripts/robodojo.sh）从项目根目录启动 Python，本脚本需自行兜底。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_XPOLICYLAB_ROOT = os.path.join(_PROJECT_ROOT, "XPolicyLab")
for _import_root in (_PROJECT_ROOT, _XPOLICYLAB_ROOT):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

from isaaclab.app import AppLauncher


def _nonnegative_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def _positive_finite_float(value: str) -> float:
    parsed = _nonnegative_finite_float(value)
    if parsed == 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


parser = argparse.ArgumentParser(description="Record deformable task preview videos without a policy.")
parser.add_argument("--task_name", type=str, default=None, help="单个任务名（与 --tasks 二选一）")
parser.add_argument(
    "--tasks",
    type=str,
    default=None,
    help="逗号分隔的多个任务名，例如 fold_clothes,fold_clothes_random",
)
parser.add_argument("--num_envs", type=int, default=1, help="并行环境数；逐布局录制要求为 1")
parser.add_argument("--env_cfg_type", type=str, default="arx_x5", help="eval 配置文件名（env_cfg 目录下，不含 .yml）")
parser.add_argument("--device_id", type=int, default=0, help="当前进程可见的 GPU 序号")
parser.add_argument("--seed", type=int, default=0, help="布局目录编号，例如 Assets/Eval_Layout/.../0")
parser.add_argument("--layout_id", type=int, default=None, help="只录制指定布局文件的编号；默认录制全部")
duration_group = parser.add_mutually_exclusive_group()
duration_group.add_argument("--steps", type=int, default=120, help="每个 episode 推进的动作步数；默认 120")
duration_group.add_argument("--duration_s", type=_positive_finite_float, help="每个 episode 录制时长（秒），按采集频率换算成动作步数")
parser.add_argument(
    "--wind", type=_finite_float, nargs=3, default=(0.0, 0.0, 0.0),
    metavar=("WX", "WY", "WZ"), help="恒定环境风速向量（m/s）；默认 0 0 0",
)
parser.add_argument("--no-wind", action="store_true", help="关闭风力；与 --wind 同时传入时优先")
parser.add_argument("--particle_drag", type=_nonnegative_finite_float, default=0.5, help="布料粒子 drag；默认 0.5")
parser.add_argument("--rigid_drag_coefficient", type=_nonnegative_finite_float, default=1.0, help="刚体风阻系数 Cd；默认 1.0")
parser.add_argument("--rigid_force_scale", type=_nonnegative_finite_float, default=100.0, help="刚体风阻缩放系数；默认 100，0 关闭刚体风阻")
parser.add_argument(
    "--out_dir",
    type=str,
    default="eval_result/preview",
    help="视频输出根目录",
)
parser.add_argument("--additional_info", type=str, default="preview", help="附加信息（用于输出目录命名）")
parser.add_argument(
    "--xlens_snapshot_dir",
    type=str,
    default="logs/xlens_geometry",
    help="XLens 场景快照输出根目录；默认在每次 reset 后自动导出",
)
parser.add_argument(
    "--xlens_min_mask_pixels",
    type=int,
    default=50,
    help="自动加入 scene.json 的实例至少需要覆盖的像素数",
)
parser.add_argument(
    "--disable_xlens_snapshot",
    action="store_true",
    help="关闭 reset 后的 XLens 场景快照导出",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.steps <= 0:
    parser.error("--steps must be a positive integer")
if args_cli.num_envs != 1:
    parser.error("逐布局录制要求 --num_envs 1")
if args_cli.layout_id is not None and args_cli.layout_id < 0:
    parser.error("--layout_id must be a non-negative integer")

# Do not rewrite CUDA_VISIBLE_DEVICES here. Isaac Sim/Omniverse has its own GPU
# enumeration and warns that masking CUDA devices can break Vulkan/PhysX device
# discovery. Select the already-visible device directly instead.
if args_cli.device_id is not None:
    if args_cli.device_id < 0:
        parser.error("--device_id must be a non-negative integer")
    args_cli.device = f"cuda:{args_cli.device_id}"
print(f"[record] Isaac Sim device={args_cli.device}")

# 在 AppLauncher 之前安全导入（env 为 namespace package，global_configs 仅依赖 os）
from env.global_configs import BENCHMARK, ROOT_DIR  # noqa: E402

# 摄像头画面必须与物理同步，避免帧滞后
from env.camera_manager.capture.render_sync import add_zero_delay_kit_args  # noqa: E402

add_zero_delay_kit_args(args_cli)

# 启动 Omniverse / Isaac Sim 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omegaconf import OmegaConf  # noqa: E402

from env.global_configs import *  # noqa: E402,F403
from env.observation_manager.obs_manager import ObsManager  # noqa: E402
import src.eval_client.eval_env as eval_env_module  # noqa: E402
from src.eval_client.eval_env import create_eval_env  # noqa: E402
from utils.load_file import load_yaml  # noqa: E402
from utils.cloth_wind import apply_constant_cloth_wind_if_present, collect_cloth_wind_readbacks  # noqa: E402
from utils.rigid_wind import create_rigid_wind_controller, install_sim_step_wind_hook  # noqa: E402
from utils.xlens_snapshot import populate_manifest_objects, save_robodojo_snapshot  # noqa: E402
from utils.pipeline_utils import (  # noqa: E402
    process_config,
    process_randomization,
    resolve_random_task_num_envs,
)

BENCHMARK_PATH = os.path.join(ROOT_DIR, "task", BENCHMARK)
task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")


class _NoopModelClient:
    """占位策略客户端：不连接任何策略服务器，所有调用均为空操作。"""

    def __init__(self, *args, **kwargs):
        # 与 WsModelClient 保持构造接口兼容，但不保存或使用连接参数。
        self.connected = False

    def connect(self, *args, **kwargs):
        return True

    def call(self, func_name, *args, **kwargs):
        # reset / get_action 等调用直接忽略
        return None

    def close(self):
        pass


def _configure_preview_mode() -> None:
    """Configure the recorder without a policy server and with segmentation output."""
    # EvalEnv.__init__ constructs WsModelClient before returning the environment.
    # Replace it first so preview recording never opens a WebSocket connection.
    eval_env_module.WsModelClient = _NoopModelClient

    # Segmentation is consumed directly by the XLens snapshot exporter.  It is
    # also present in the camera capture payload, so ObsManager must know how
    # to copy these annotators into the observation dictionary.
    ObsManager.ANNOTATORS_TO_COLLECT.update(
        {
            "instance_id_segmentation_fast": "instance_id",
            "instance_segmentation_fast": "instance",
            "semantic_segmentation": "semantic",
        }
    )


def _resolve_tasks() -> list[str]:
    if args_cli.tasks:
        tasks = [t.strip() for t in args_cli.tasks.split(",") if t.strip()]
    elif args_cli.task_name:
        tasks = [args_cli.task_name]
    else:
        tasks = ["fold_clothes", "fold_clothes_random"]
    return tasks


def _selected_wind() -> list[float]:
    return [0.0, 0.0, 0.0] if args_cli.no_wind else list(args_cli.wind)


def _log_wind_readback(
    env, task_name: str, phase: str, log_path: str,
    rigid_wind_controller=None, rigid_wind_skipped=None,
) -> None:
    """Save requested wind and values read from live cloth and rigid views."""
    wind = _selected_wind()
    garments_by_env = getattr(env.scene_manager, "_garment_objects", [])
    garments = collect_cloth_wind_readbacks(garments_by_env, wind)
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
        "requested_wind": wind,
        "requested_particle_drag": args_cli.particle_drag if garments else None,
        "requested_rigid_drag_coefficient": args_cli.rigid_drag_coefficient,
        "requested_rigid_force_scale": args_cli.rigid_force_scale,
        "garment_count": len(garments),
        "all_match_request": matches,
        "garments": garments,
        "rigid_wind": rigid_wind,
        "rigid_wind_skipped": rigid_wind_skipped or [],
    }
    try:
        with open(log_path, "a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[record][WARN] 无法写入风力诊断日志 {log_path}: {exc}")
    print(
        f"[record][wind] task={task_name} phase={phase} requested={wind} "
        f"garments={len(garments)} all_match={matches} "
        f"rigid_bodies={rigid_wind['body_count']} applied={rigid_wind['applied_body_count']}"
    )
    for item in garments:
        print(
            f"[record][wind] garment={item['garment_name']} actual={item['actual_wind']} "
            f"matches={item['matches_request']} error={item['readback_error']}"
        )
    if garments and not matches:
        print("[record][WARN] 布料实际风速读回失败或与请求值不一致")
    for item in rigid_wind_skipped or []:
        print(f"[record][wind][skip] {item['label']}: {item['reason']}")


def build_env(task_name: str):
    """按 main.py 的流程组装 env_cfg，创建 eval 环境并替换策略客户端。"""
    num_envs = args_cli.num_envs
    eval_cfg_name = args_cli.env_cfg_type
    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, eval_cfg_name + ".yml"))
    eval_cfg["task_name"] = task_name
    eval_cfg["num_envs"] = num_envs
    eval_cfg["device_id"] = args_cli.device_id
    eval_cfg["eval_batch"] = False
    eval_cfg["policy_name"] = "noop_preview"
    eval_cfg["additional_info"] = args_cli.additional_info
    eval_cfg["seed"] = args_cli.seed
    eval_cfg["physx_monitor_enabled"] = False

    # 空操作策略客户端配置（不真正连接）
    deploy_cfg = {
        "policy_name": "noop_preview",
        "port": 0,
        "host": "localhost",
        "protocol": "ws",
        "policy_server_url": "ws://localhost:0",
        "evaluation_id": "preview",
        "trial_id": f"{task_name}-preview",
        "action_case_id": f"{task_name}_case",
        "repeat_index": None,
    }

    task_cfg = load_yaml(task_registry.task_config_path(os.path.join(BENCHMARK_PATH, "config"), task_name))
    wind = _selected_wind()
    if any(wind):
        apply_constant_cloth_wind_if_present(task_cfg, wind, particle_drag=args_cli.particle_drag)

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

    capped_num_envs = resolve_random_task_num_envs(task_name, num_envs, env_cfg.sim)
    num_envs = capped_num_envs
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

    _configure_preview_mode()
    env = create_eval_env(env_cfg, simulation_app, resume_state=None)
    return env, num_envs


def _neutral_action(env) -> dict:
    """构造“保持当前位姿、夹爪张开”的空操作动作。

    双机械臂动作字典需要 left/right 前缀的 arm_joint_state 与 ee_joint_state。
    这里读取机械臂当前关节位置作为目标（保持不动），夹爪置为张开（1.0）。
    """
    action = {}
    for robot in env.robot_manager.robot_list:
        if robot.type != "target":
            continue
        arm_key = env.robot_manager.process_name(robot.arm_name)
        gripper_key = env.robot_manager.process_name(robot.gripper_name)
        # 保持当前关节位姿
        joints = env.robot_manager.get_joint(robot, env_idx_list=[0])[0]
        action[arm_key] = list(joints) if joints is not None else None
        # 夹爪张开（归一化 0~1，1 表示张开）
        action[gripper_key] = [1.0]
    # 过滤掉 None（理论上不会出现）
    action = {k: v for k, v in action.items() if v is not None}
    return action


def _recording_steps(env) -> int:
    if args_cli.duration_s is None:
        return args_cli.steps
    collect_freq = float(getattr(env.obs_manager, "collect_freq", 0) or 0)
    if not math.isfinite(collect_freq) or collect_freq <= 0:
        raise ValueError("observation.collect_freq 必须大于 0 才能按秒数录制")
    return max(1, int(round(args_cli.duration_s * collect_freq)))


def _available_layouts(env) -> list[tuple[int, int]]:
    """Return (SeedManager index, layout filename number) in recording order."""
    layouts = []
    for seed_index, info in env.seed_manager.seed_info.items():
        layout_path = Path(info["scene_layout"])
        layout_id = int(layout_path.stem.rsplit("_", 1)[-1])
        layouts.append((int(seed_index), layout_id))
    if not layouts:
        raise ValueError(f"任务 {env.task_name} 在 --seed {args_cli.seed} 的目录下没有布局文件")
    if len({layout_id for _, layout_id in layouts}) != len(layouts):
        raise ValueError(f"任务 {env.task_name} 有重复的布局编号")
    if args_cli.layout_id is not None:
        layouts = [entry for entry in layouts if entry[1] == args_cli.layout_id]
        if not layouts:
            raise ValueError(f"任务 {env.task_name} 没有布局 {args_cli.layout_id}")
    return layouts


def _record_layout(env, task_name: str, seed_index: int, layout_id: int, record_timestamp: str, task_output_dir: str) -> None:
    """Reset and record one saved layout; recreate wind views after each reset."""
    original_sim_step = None
    try:
        wind_log_path = os.path.join(task_output_dir, f"wind_diagnostics_{record_timestamp}_layout-{layout_id:03d}.jsonl")
        env.reset(seed=[seed_index])
        wind = _selected_wind()
        has_wind = any(wind)
        rigid_wind_controller = None
        rigid_wind_skipped = []
        if has_wind and args_cli.rigid_force_scale > 0:
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
        if has_wind:
            _log_wind_readback(
                env, task_name, "after_reset", wind_log_path,
                rigid_wind_controller, rigid_wind_skipped,
            )

        if not args_cli.disable_xlens_snapshot and args_cli.xlens_snapshot_dir:
            snapshot_dir = os.path.join(
                args_cli.xlens_snapshot_dir,
                f"{task_name}_seed_{args_cli.seed:04d}_layout_{layout_id:03d}",
            )
            manifest_path = save_robodojo_snapshot(
                env, snapshot_dir, task_name=task_name, env_id=0, objects=[],
            )
            objects = populate_manifest_objects(
                manifest_path, min_pixels=args_cli.xlens_min_mask_pixels,
            )
            print(f"[XLens] scene manifest: {manifest_path}")
            print(f"[XLens] discovered {len(objects)} visible segmentation instances")
            if not objects:
                print("[XLens] warning: no visible non-background instances passed the pixel threshold")

        # 与官方 run_eval 一致，避免空检查列表导致首步即判成功提前结束。
        env.run_reward()
        if hasattr(env, "get_score"):
            env.get_score()
        steps = _recording_steps(env)
        print(f"[record] 任务: {task_name} | 布局: {layout_id} | steps={steps} | wind={wind}")
        print(f"[record] 指令: {env.obs_manager.instruction}")
        action = _neutral_action(env)
        print(f"[record] 空操作动作: {action}")

        # 每步显式采集画面；刚体风阻钩子在每个底层物理步前施力。
        for step in range(steps):
            env.take_action(action)
            env.get_obs()
            if has_wind and step == 0:
                _log_wind_readback(
                    env, task_name, "after_first_step", wind_log_path,
                    rigid_wind_controller, rigid_wind_skipped,
                )
            if (step + 1) % 20 == 0:
                print(f"[record] {task_name} 布局 {layout_id} 已推进 {step + 1}/{steps} 步")

        env.end_flag = [True]
        env.success = [True]
        wind_speed = math.sqrt(sum(value ** 2 for value in wind))
        speed_token = f"{wind_speed:.3f}".replace(".", "p")
        video_name = f"record_{record_timestamp}_layout-{layout_id:03d}_wind-{'yes' if has_wind else 'no'}_speed-{speed_token}.mp4"
        video_path = os.path.join(task_output_dir, video_name)
        video_tag = "constant_wind" if has_wind else "preview"
        env.save_video(0, video_path, tag=video_tag)
        saved_videos = list(Path(task_output_dir).glob(f"{Path(video_name).stem}_*_{video_tag}.mp4"))
        if not saved_videos:
            raise RuntimeError(f"{task_name} 布局 {layout_id} 没有录制出视频")
        print(f"[record] 完成 {task_name} 布局 {layout_id}，视频数={len(saved_videos)}，输出目录: {task_output_dir}")
        if has_wind:
            print(f"[record] 风力诊断: {wind_log_path}")
    finally:
        if original_sim_step is not None:
            env.sim_step = original_sim_step


def record_task(task_name: str) -> str:
    print(f"\n{'=' * 70}\n[record] 开始创建任务环境: {task_name}\n{'=' * 70}")
    env, _ = build_env(task_name)
    try:
        layouts = _available_layouts(env)
        print(f"[record] 已找到 {len(env.seed_manager.seed_info)} 个布局文件；任务配置 eval_nums={env.eval_num}")
        print(f"[record] 本次录制布局: {[layout_id for _, layout_id in layouts]}")
        if args_cli.layout_id is None and len(layouts) < env.eval_num:
            print(f"[record][WARN] 只有 {len(layouts)} 个布局文件，少于配置的 {env.eval_num} 个")
        record_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        task_output_dir = os.path.join(args_cli.out_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        failed_layouts = []
        for seed_index, layout_id in layouts:
            try:
                _record_layout(env, task_name, seed_index, layout_id, record_timestamp, task_output_dir)
            except Exception as exc:
                import traceback

                failed_layouts.append(layout_id)
                print(f"[record] {task_name} 布局 {layout_id} 录制失败: {type(exc).__name__}: {exc}")
                traceback.print_exc()
        if failed_layouts:
            raise RuntimeError(f"{task_name} 未完成的布局: {failed_layouts}")
        return task_output_dir
    finally:
        env.close()


def main():
    out_dirs = []
    failures = []
    try:
        tasks = _resolve_tasks()
        print(f"[record] 待录制任务: {tasks}")
        for task_name in tasks:
            try:
                out_dirs.append(record_task(task_name))
            except Exception as exc:
                import traceback

                failures.append(task_name)
                print(f"[record] 任务 {task_name} 录制失败: {type(exc).__name__}: {exc}")
                traceback.print_exc()

        print("\n" + "=" * 70)
        print("[record] 全部完成，视频输出目录：")
        for output_dir in out_dirs:
            print(f"  - {output_dir}")
        print("=" * 70)
    finally:
        simulation_app.close()
    if failures:
        raise SystemExit(f"录制失败的任务: {', '.join(failures)}")


if __name__ == "__main__":
    main()
