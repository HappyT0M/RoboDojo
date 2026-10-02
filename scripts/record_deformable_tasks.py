"""无需策略（VLA/动作模型）即可创建可形变物体任务仿真环境并录制视频。

用法示例（在 RoboDojo 根目录下运行）::

    python scripts/record_deformable_tasks.py \
        --task_name fold_clothes \
        --num_envs 1 \
        --env_cfg_type arx_x5 \
        --device_id 0 \
        --seed 0 \
        --steps 100 \
        --out_dir eval_result/preview

    # 一次性录制两个任务
    python scripts/record_deformable_tasks.py --tasks fold_clothes,fold_clothes_random

说明：
    * 复用 RoboDojo 官方的 ``create_eval_env`` 与摄像头/渲染管线，仅把策略客户端
      替换为“空操作”占位对象，因此不接入任何 VLA / 动作模型。
    * 每个 episode 机械臂保持初始位姿、夹爪保持张开，仅推进物理仿真并采集摄像头
      画面，用于直观查看任务场景（衣物初始摆放、杂物、视角等）。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import os
import sys

# 将项目根目录加入 sys.path，确保无论从哪个目录运行都能导入 env/task/utils/src 等包。
# 官方入口（scripts/robodojo.sh）从项目根目录启动 Python，本脚本需自行兜底。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_XPOLICYLAB_ROOT = os.path.join(_PROJECT_ROOT, "XPolicyLab")
for _import_root in (_PROJECT_ROOT, _XPOLICYLAB_ROOT):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record deformable task preview videos without a policy.")
parser.add_argument("--task_name", type=str, default=None, help="单个任务名（与 --tasks 二选一）")
parser.add_argument(
    "--tasks",
    type=str,
    default=None,
    help="逗号分隔的多个任务名，例如 fold_clothes,fold_clothes_random",
)
parser.add_argument("--num_envs", type=int, default=1, help="并行环境数")
parser.add_argument("--env_cfg_type", type=str, default="arx_x5", help="eval 配置文件名（env_cfg 目录下，不含 .yml）")
parser.add_argument("--device_id", type=int, default=0, help="当前进程可见的 GPU 序号")
parser.add_argument("--seed", type=int, default=0, help="场景随机种子")
parser.add_argument("--steps", type=int, default=120, help="每个 episode 推进的步数（动作步数）")
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

    env_cfg = OmegaConf.create(
        {
            "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")),
            "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")),
            "camera": load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml")),
            "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml")),
            "task_env": load_yaml(task_registry.task_config_path(os.path.join(BENCHMARK_PATH, "config"), task_name)),
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


def record_task(task_name: str) -> str:
    print(f"\n{'=' * 70}\n[record] 开始创建任务环境: {task_name}\n{'=' * 70}")
    env, num_envs = build_env(task_name)

    record_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    seed = [args_cli.seed]
    env.reset(seed=seed)
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
        objects = populate_manifest_objects(
            manifest_path,
            min_pixels=args_cli.xlens_min_mask_pixels,
        )
        print(f"[XLens] scene manifest: {manifest_path}")
        print(f"[XLens] discovered {len(objects)} visible segmentation instances")
        if not objects:
            print("[XLens] warning: no visible non-background instances passed the pixel threshold")
    # 注册奖励/成功检查（与官方 run_eval 一致），避免空检查列表导致首步即判成功提前结束
    env.run_reward()
    if hasattr(env, "get_score"):
        env.get_score()

    print(f"[record] 任务: {task_name} | num_envs={num_envs} | steps={args_cli.steps}")
    print(f"[record] 指令: {env.obs_manager.instruction}")

    action = _neutral_action(env)
    print(f"[record] 空操作动作: {action}")

    # 推进若干步（机械臂保持不动，仅物理仿真 + 采集画面）
    # 每步显式调用 get_obs 以触发摄像头画面采集（官方流程中由策略循环负责）
    for step in range(args_cli.steps):
        env.take_action(action)
        env.get_obs()
        if (step + 1) % 20 == 0:
            print(f"[record] {task_name} 已推进 {step + 1}/{args_cli.steps} 步")

    # 结束当前 episode，写出视频
    env.end_flag = [True] * num_envs
    env.success = [True] * num_envs

    # 复用官方 save_video 机制，将已采集的摄像头流落盘
    # save_video 会在这个基础名后追加具体相机名和 tag。
    # 文件名示例：record_20260928_143012_123456_wind-no_speed-0p000_cam_head_preview.mp4
    video_name = f"record_{record_timestamp}_wind-no_speed-0p000.mp4"
    video_path = os.path.join(args_cli.out_dir, task_name, video_name)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    env.save_video(0, video_path, tag="preview")

    env.close()
    print(f"[record] 完成 {task_name}，视频输出目录: {os.path.dirname(video_path)}")
    return os.path.dirname(video_path)


def main():
    tasks = _resolve_tasks()
    print(f"[record] 待录制任务: {tasks}")
    out_dirs = []
    for task_name in tasks:
        try:
            out_dirs.append(record_task(task_name))
        except Exception as e:
            import traceback

            print(f"[record] 任务 {task_name} 录制失败: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("[record] 全部完成，视频输出目录：")
    for d in out_dirs:
        print(f"  - {d}")
    print("=" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()
