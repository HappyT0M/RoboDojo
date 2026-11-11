# RoboDojo scripts

## Public entry points

| Script | Purpose |
| --- | --- |
| [robodojo.sh](robodojo.sh) | Main CLI: `doctor`, `eval`, `client`, `smoke`, `benchmark`, `dimensions`, `summarize`, `tasks` |
| [install.sh](install.sh) | One-time environment setup (conda, Isaac Sim, submodules) |
| [init_assets.sh](init_assets.sh) | Download robot/object assets |
| [eval_policy.sh](eval_policy.sh) | Isaac Sim eval client (called by `robodojo.sh client` and XPolicyLab) |
| [eval_from_config.py](eval_from_config.py) | JSON-configured task sweeps; defaults to parallel Pi0.5 wind / no-wind groups |

## 使用 JSON 启动 Pi0.5 评测

修改 [`configs/pi05_deformable.json`](configs/pi05_deformable.json)，然后在服务器的
RoboDojo 根目录运行：

```bash
conda activate RoboDojo
python scripts/eval_from_config.py
```

默认无风 `[0, 0, 0]` 与有风 `[0, 0.5, 0]` 两组并行，每组依次评测十个任务，
每个任务评测十个布局。选择服务器上空闲的显卡后，修改各组 GPU 编号。

| JSON 字段 | 含义 |
| --- | --- |
| `policy_dir` / `ckpt` | 模型目录及权重名称或路径 |
| `policy_env` / `eval_env` | 模型运行环境（Pi0.5 为 `uv`）及仿真 Conda 环境 |
| `env_cfg` / `action_type` | 机器人配置及动作类型，默认 `arx_x5` / `joint` |
| `seed` | 布局目录种子，默认 `0` |
| `eval_num` | 每组每任务的评测数量；`10` 为十个布局，`1` 为一个；`"native"` 使用任务原生数量 |
| `num_envs` | 单个仿真进程同时创建的场景数，默认 `1`，多个布局依次 reset |
| `rigid_force_scale` | 刚体风力倍率，默认 `100`，与之前录制保持一致 |
| `parallel` | `true` 为组间并行，`false` 为组间串行；组内任务始终串行 |
| `tasks` | 要评测的任务列表及顺序 |
| `groups[].name` | 本组日志名称标识（调整风速时也可重命名） |
| `groups[].wind` | 世界坐标系风速 `[x, y, z]`，单位 m/s |
| `groups[].policy_gpu` / `groups[].env_gpu` | 本组模型 GPU / 仿真 GPU 编号 |

默认无风组使用 GPU 0/1，有风组使用 GPU 2/3。只运行一组时，从 `groups` 中删除
另一组即可。JSON 不支持注释和末尾多余的逗号；数字和布尔值不要加引号。
自定义配置可用 `python scripts/eval_from_config.py --config 路径.json` 加载；
配置中的相对 `policy_dir` 以仓库根目录为基准，其余模型参数沿用原评测入口的含义。

先查看将执行的命令（Mac 也可运行，不加载模型或 Isaac Sim）：

```bash
python scripts/eval_from_config.py --dry-run
```

本次 `num_envs` 通过环境变量 `ROBODOJO_NUM_ENVS` 传给评测客户端，不改动或备份
共享 YAML。未设置该环境变量的原有命令仍从 YAML 读取场景数。
日志固定保存在 `logs/eval_<组名>_<北京时间>.log`，开头记录本次参数，正文保留
原有日志格式；单任务日志和 `eval_result/` 中的结果、视频继续由原入口保存。
`--dry-run` 只校验 JSON 并展示命令，不验证服务器环境或权重。

任一任务进程失败后，该组停止后续任务，另一组继续；总命令返回非零退出码。
按 Ctrl+C 或收到 SIGTERM 时会停止当前模型和仿真进程组，等待最多十秒后强制终止
剩余进程。退出码为零仍需查看各任务 `_result.json` 的 `eval_time` 和 `success_rate`，
确认实际完成布局数和成功率。此入口沿用模型评测的 reset 流程，每个任务启动一次
模型和仿真；录制脚本的逐布局独立进程机制不适用于此入口。

## Keyboard teleoperation for deformable tasks

`teleop_deformable_task.py` opens a GUI Isaac Sim session for manually operating
the rope-and-ball task. It does not connect to a VLA/WAM policy server. Focus the
Isaac Sim viewport after startup so it receives keyboard input.

```bash
python scripts/teleop_deformable_task.py \
  --task_name put_rope_ball_in_basket \
  --arm left \
  --seed 0 \
  --device_id 0 \
  --wind 0 0 0 \
  --out_dir eval_result/teleop_rope
```

Hold the keys to move the selected end effector in small Cartesian increments:

| Keys | Motion |
| --- | --- |
| `W` / `S` | +X / -X |
| `D` / `A` | +Y / -Y |
| `R` / `F` | +Z / -Z |
| `C` / `O` | Close / open gripper |
| `Esc` | Save and exit |

Select the other arm with `--arm right`. The inactive arm holds its reset joint
positions. The end-effector orientation stays fixed. Use `--step_m` to change
movement increments and `--workspace_radius` to limit travel around the reset
position. Wind is disabled by default; for example, `--wind 0 0.5 0` enables a
constant 0.5 m/s wind in +Y.

The task directory under `--out_dir` receives per-camera MP4 files and a
timestamped `_actions.jsonl` control log after exit. This script requires a GUI
window and cannot be run with `--headless`.

## Scene preview without model weights

`record_deformable_tasks.py` advances the simulation with a neutral robot action
and records camera video for **every saved layout of the selected task**. The
layout number appears in each video and wind diagnostic filename. `--seed`
selects the layout directory (usually `0`); `--layout_id` selects one layout
file when a short preview is enough. The script defaults to no wind and 120
action steps per layout. Use `--steps` or `--duration_s` to choose the length.
`--wind WX WY WZ` sets constant world-space wind velocity in m/s; the default
is `0 0 0`. Video preview enables only RGB camera capture by default to reduce
GPU memory use. Add `--xlens_snapshot` if instance masks and XLens scene
snapshots are needed.
Every task launches a fresh Isaac Sim process for each layout, sequentially.
After saving the video, the worker releases camera resources, closes the
environment and Isaac Sim, and exits. The supervisor waits for process exit
before starting the next layout, so GPU resources cannot accumulate across
layouts in one run. Ten tasks with ten layouts each produce 100 workers.
`--tasks ... --layout_id N` starts one worker per task for layout N.
The single command and Beijing-time log cover the whole sequence. Failed
layouts are reported while later layouts still run; the command exits nonzero
if any worker fails or exits without reporting a completed recording.
After environment cleanup, the disposable worker uses Isaac Sim's immediate
exit (`close(wait_for_replicator=False, skip_cleanup=True)`) to avoid a second
Replicator cleanup pass. The supervisor allows 60 seconds from the start of
environment cleanup for the worker to exit, then sends SIGTERM and, after
another 5 seconds, SIGKILL if needed. It waits for exit before the next layout
and reports a shutdown timeout as a failed worker, even if videos were saved.
The known `Invalid NodeObj object` error during annotator detach is logged as
a warning while remaining capture resources are still released; other cleanup
errors remain failures.
Each layout requires Isaac Sim startup time. On a single GPU, run the no-wind
and wind commands one after the other; separate commands are not serialized
against each other.

```bash
python scripts/record_deformable_tasks.py \
  --task_name put_rope_ball_in_basket \
  --seed 0 --layout_id 0 --steps 120 \
  --out_dir eval_result/preview

python scripts/record_deformable_tasks.py \
  --task_name put_rope_ball_in_basket \
  --seed 0 --device_id 3 --headless --duration_s 10 --wind 0 0.5 0 \
  --rigid_force_scale 100 \
  --out_dir eval_result/preview
```

`--device_id N` selects GPU N for Isaac Sim rendering and camera capture;
`--device cuda:N` is equivalent. The rope task's configured physics device
remains CPU. The script does not change `CUDA_VISIBLE_DEVICES`; avoid an
external mask that remaps CUDA indices differently from the physical GPU order
used by Kit's `activeGpu` setting.

Each run saves stdout, stderr, and native Isaac Sim output to a separate file
under the repository's fixed `logs/` directory. The filename uses Beijing time
(`Asia/Shanghai`), for example
`logs/record_put_rope_ball_in_basket_2026-10-25T21:47:38.log`. Log messages
retain their original timestamps and formatting; the script prints the chosen
log path at startup. Runs started within the same second get a numeric filename
suffix rather than overwriting one another. With `--tasks` naming multiple
tasks, the filename uses `multiple_tasks`.

For the rope tasks, generate and verify layouts 0–9 with
`scripts/generate_rope_eval_layouts.py` before recording. The script prints the
number of discovered layouts and warns if it is below the task's `eval_nums`.

## Ten deformable evaluation tasks

The fixed evaluation set contains `put_rope_ball_in_basket`,
`pass_rope_through_ring`, `route_rope_around_posts`, `fold_clothes_deformable`,
`flatten_clothes`, `hang_clothes`, `put_clothes_in_bin`, `cover_object_with_cloth`,
`pull_rope_to_retrieve_ball`, and `wipe_marked_area_with_cloth`.
`drape_cloth_over_rack` has been retired in favor of `hang_clothes`.
Generate the three rope tasks' ten scene-varied layouts
with `python scripts/generate_rope_eval_layouts.py`. Generate the light-weight
fixtures (`folding_board`, `garment_rack`, `garment_bin`, `cloth_block`) and
ten layouts for each of the four garment tasks with
`python scripts/generate_deformable_task_assets.py`. Generate the three new tasks
with `python scripts/generate_additional_deformable_assets.py`; add `--check` for
read-only verification of their five assets and thirty layouts. All generators write into
`Assets/`; that entire directory is excluded from Git. Export the assets and
layouts for a separate Hugging Face dataset with `export_deformable_assets.py`.
See [the asset publishing guide](../docs/DEFORMABLE_ASSETS.md) and
[the ten-task expansion and sync commands](../docs/DEFORMABLE_TASK_EXPANSION.md).
The custom tasks also appear under `robodojo.sh tasks --dimension deformable`;
this is a fork-specific extension of the official capability groups.

For an existing installation, rebuild the two active rope fixtures with
`python scripts/create_rope_assets.py --update-fixtures`, then run the rope and original garment layout
generators above. This avoids resetting the rope and seed scenes. Custom
fixtures have an Xform root with a child render/collision Mesh; static triangle
collisions preserve the bin and target-slot openings. Each frame's height is
computed after rotation so its bottom stays 2 mm above the tabletop. Custom
garment layouts select muted matte colors and place the lowest cloth vertices
above the support surface. The native `Top_Long` assets are unchanged.
See [the scene repair and sync checklist](../docs/DEFORMABLE_SCENE_FIX.md).

`fold_clothes_deformable` reuses the native folding reward logic with its own
configuration, ten layouts, and output task name. The native `fold_clothes`
keeps its original layout files and default evaluation count of 25. Use
`--task_name fold_clothes_deformable` to record or evaluate the custom version.

Wind uses cloth particle settings and dynamic task-body drag, with diagnostics
saved beside the videos. `--rigid_force_scale` defaults to 100 for preview
recordings. `--no-wind` overrides `--wind` when both are supplied.

## Typical eval flow

```text
robodojo.sh eval
  -> scripts/internal/run_policy_eval.sh
    -> policy server (localhost) + sim client

Split / multi-machine (see docs/SPLIT_EVAL.md):

robodojo.sh server  ->  scripts/internal/run_policy_server.sh  ->  policy server (bind 0.0.0.0)
robodojo.sh client  ->  scripts/eval_policy.sh  ->  src/eval_client/main.py
```

Run one or more official capability dimensions in a benchmark sweep:

```bash
bash scripts/robodojo.sh dimensions
bash scripts/robodojo.sh benchmark \
  --dimension memory,long-horizon \
  --policy-dir XPolicyLab/policy/<POLICY> \
  --ckpt <CHECKPOINT> \
  --policy-env <ENV> \
  --eval-num native
```

Available dimensions are `generalization`, `memory`, `precision`,
`long-horizon`, and `open`. Generalization includes both the 12 standard tasks
and their 12 runnable `_random` layout variants. Combine `--dimension` with
`--only` or `--tasks-file` to narrow a dimension further.

## Auto multi-GPU grouping

`robodojo.sh smoke` and `robodojo.sh benchmark` now support balanced
multi-GPU execution driven by the runtime table in `../optimal_8group.txt`.
Pass a concrete GPU id list and RoboDojo will partition the selected tasks
online instead of relying on a hard-coded task group.

Dry-run example:

```bash
bash scripts/robodojo.sh benchmark \
  --policy-dir XPolicyLab/policy/ACT \
  --ckpt test_ckpt \
  --policy-env RoboDojo \
  --eval-num 1 \
  --gpu-ids 0,2,5,7 \
  --dry-run
```

Launch only a subset of tasks:

```bash
bash scripts/robodojo.sh benchmark \
  --policy-dir XPolicyLab/policy/ACT \
  --ckpt test_ckpt \
  --policy-env RoboDojo \
  --eval-num native \
  --only imitate_sorting_sequence,pour_by_language,play_tic_tac_toe \
  --gpu-ids 0,1,3
```

## Internal (`internal/`)

Not intended for direct daily use. Called by `robodojo.sh` or policy utilities.

| File | Called by |
| --- | --- |
| [verify_install.sh](internal/verify_install.sh) | `robodojo.sh doctor` |
| [task_inventory.py](internal/task_inventory.py) | `robodojo.sh tasks` |
| [smoke_all_tasks.sh](internal/smoke_all_tasks.sh) | `robodojo.sh smoke` / `benchmark` |
| [summarize_result.py](internal/summarize_result.py) | `robodojo.sh summarize` |
| [stat_score_distribution.py](internal/stat_score_distribution.py) | Offline score histogram analysis (manual) |

## Docker

Container install and smoke tests live under [../docker/](../docker/), not here.

## Policy-specific scripts

Training, data prep, and per-policy `eval.sh` live in [../XPolicyLab/policy/](../XPolicyLab/policy/) (submodule).

## Constant wind during policy evaluation

Every model using the standard RoboDojo evaluation client can use the same
world-space ambient wind velocity (m/s). Add these options to a working
`robodojo.sh eval` command; keep its checkpoint, environment and action type:

```bash
--wind 0.0 0.5 0.0 --rigid-force-scale 100
```

Rigid wind starts after scene settling and policy reset, before the first policy
observation, and is applied before **each physics substep**, including when
simulation decimation exceeds one. Views are rebuilt after each episode reset;
wind is disabled during reset and settling. Dynamic task bodies and articulation
links (including rope links and the attached ball) receive drag; robots,
static geometry, kinematic bodies and fixed-base task articulations are excluded.
Cloth retains its existing particle-system wind path and initialization timing.

`--rigid-force-scale` defaults to **1**; use **100** to reproduce the default
multiplier of `record_deformable_tasks.py`. This is a simplified drag
model using projected bounding-box areas, air density 1.225 kg/m³ and Cd=1.
Keep the multiplier fixed across models. Zero/omitted wind disables rigid drag
entirely, including still-air damping, preserving the original baseline.
The equivalent environment variable is `ROBODOJO_RIGID_FORCE_SCALE` (also usable
with the split client). Requesting wind without any eligible body fails clearly.

Each timestamped result directory contains `_wind.json` with the velocity,
multiplier, target paths, skipped bodies, physics-step count and last applied
forces. Completed `_result.json` also includes this under `wind`. Diagnostics
are saved after reset and after each completed batch, not on every substep.
A zero-body or force-submission failure invalidates the run rather than silently
continuing without wind. Start with `scene.num_envs: 1` for server validation.
# Evaluation logs and timestamps

`bash scripts/robodojo.sh eval ...` automatically streams stdout and stderr to
`logs/eval_YYYY-MM-DD_HH-MM-SS_<unique>.log` while displaying them in the terminal.
Each invocation creates a new log and run ID using Beijing time (`Asia/Shanghai`,
UTC+08:00); the result directory uses the same run ID. Internal recovery keeps
that ID. The unique suffix prevents concurrent runs from overwriting each other.
Do not append an old `| tee "$LOG_FILE"` command: automatic logging replaces it.
The log header records the exact arguments, run ID, and timezone. `--help` and
`--dry-run` do not create logs. Native engine timestamps ending in `Z` remain UTC.

Wind evaluation logs report application counts and maximum submitted force every
1000 physics steps. `_wind.json` records the final count and per-body forces.
Constant wind means constant air velocity; drag varies with body velocity and
orientation, and visible motion can stop under contact constraints.

### Stock-frame version of the rope crossing task

`pass_rope_through_ring` now uses the existing `Geometry/frame/00000/object.usdz`
asset as a fixed, orange rectangular frame. Its opening is 35.1 × 28.2 cm,
with 1.5 cm depth. It stands on the table facing the +Y crossing direction.
The ball must cross within the rectangle with 2.4 cm clearance from its edges,
then stay 4–22 cm beyond the plane and within 13 cm of its axis for 20 reward
checks. This is a new task variant; do not pool results with the old ring variant.

After syncing code to a server, update its ignored seed layout once:

```bash
python scripts/create_rope_assets.py --update-ring-layout
```

This updates only the ring-labelled target in `pass_rope_through_ring_0.json`;
it does not regenerate the rope or change the basket/posts layouts. The stock
frame asset must already exist. Run the same evaluation command and task name.
Verify the frame is visible and the opening is traversable in Isaac Sim before
collecting comparison results; local geometry tests do not validate PhysX contact.

For the ten-layout pilot across all three rope tasks, generate layouts 1–9
after preparing the three layout-0 scenes. See [rope eval layouts](../docs/rope_eval_layouts.md).

### Task three: native single-obstacle routing (variant A)

`route_rope_around_posts` now uses `Geometry/vase/00003` as a static obstacle
and `Geometry/mousemat/00000` as a static destination marker. The task name is
retained for command compatibility; its semantics differ from the old two-post
wrapping task. Keep results separate (`native_vase_route_v1`).

After syncing code, update this task's ignored seed layout on the server:

```bash
python scripts/create_rope_assets.py --update-posts-layout
```

This replaces the task's Articulation and Geometry layout sections, preserving
room/table settings and leaving other task layouts unchanged. The stock vase
and mousemat assets must be installed. No generated post/slot assets are used.

The rope ball starts at approximately (0, -0.20, 0.79). The vase is centered at
(0, 0, 0.84053), standing on the 0.765 m tabletop; the mat is at (0, 0.24, 0.76865).
After a detected rope grasp, the evaluator requires a near-to-far crossing of
Y=0 on the +X side (7.5–20 cm from the vase axis), with ball height 0.779–0.87 m.
The ball must then remain within the mat's inset goal region for 20 reward checks.
It need not wrap tightly around the vase or maintain a grasp at the destination.
The waypoint rules use environment axes; moving/rotating the task requires
reviewing these bounds. Start with a no-wind demo to validate visuals, contacts,
and reachability in Isaac Sim before collecting wind comparisons.

### Download RoboDojo evaluation checkpoints to one root

`scripts/download_vla_weights.py` calls the existing
`scripts/RoboDojo/download_ckpt.sh` for each selected model. It uses the same
RoboDojo dataset, remote checkpoint directories, policy aliases, and Git LFS
transfer logic. It never substitutes model-author base weights.

Edit `DEFAULT_ROOT` / `DEFAULT_MODELS` in the Python script, set `VLA_CKPT_ROOT`,
or pass `--root` / `--models`. The default root is `/data/xiebingsheng/ckpt`.
Defaults request DreamZero, Pi_0 and LingBot_VLA; this is a requested selection,
not a guarantee that all three are published in the RoboDojo dataset. The
existing downloader checks remote availability and installed policy adapters.
Missing models fail explicitly; other selected models continue. The batch exits
nonzero if any model fails. A dry run only previews requests, without validating
remote availability.

Requirements: Python 3, Bash 4+, git and git-lfs, and installed XPolicyLab policy
adapters. No GPU, huggingface_hub or Google Cloud CLI is required.

```bash
python scripts/download_vla_weights.py --dry-run
python scripts/download_vla_weights.py --root /data/xiebingsheng/ckpt
python scripts/download_vla_weights.py --models Pi_0 Pi_05 --root /data/xiebingsheng/ckpt
python scripts/download_vla_weights.py --source modelscope --root /root/autodl-tmp/ckpt
```

Files are stored in `<root>/.cache/robodojo_ckpt_<source>_repo`; model paths such
as `<root>/Pi_0` are symlinks into that cache, retaining the official checkpoint
subdirectory structure. Keep the cache when copying or moving downloads.
Existing target paths are archived as `.partial.<timestamp>` by the original
downloader before replacing them with symlinks. Reruns reuse the Git LFS cache.
This wrapper does not update `XPolicyLab/policy/<POLICY>/checkpoints` links; point
your evaluation setup at the downloaded directory separately.

The original downloader still defaults to its original output location when
`ROBO_DOJO_CKPT_OUTPUT_ROOT` is not set. Its repository/revision environment
variables remain supported. This replaces the previous base-weight downloader;
`--all`, `--list`, and `--workers` from that version are no longer supported.
