# CPU 物理路径下的绳索操作任务集实施规划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 在 RoboDojo 中实现三个完全使用 CPU 物理仿真的绳索操作任务，避免 FEM、布料粒子和流体粒子所走的 GPU PhysX 路径。

**架构：** 用多个带关节的胶囊刚体组成一条可弯曲但不可伸长的绳索，并通过现有 `ArticulationObject`、任务配置、初始布局、成功判定和录制流程接入 RoboDojo。三个任务共用同一类绳索资产，区别在操作目标和几何布置。物理仿真明确使用 CPU；应用渲染可继续使用 GPU，但渲染设备不能被当作物理仿真设备。

**技术栈：** Isaac Sim 5.1.0、当前集群使用的 Isaac Lab Python 扩展 0.54.3、PhysX CPU 刚体与关节、RoboDojo `ArticulationObject`、现有任务注册和 seed manager。

---

## 设计约束与适用范围

- 物理场景只使用 `Rigid`、`Dynamic` 和 `Articulation` 刚体对象；不使用 `Deformable` FEM、`Garment` 粒子布料或 `Fluid` 粒子系统。PhysX 107.3 已将旧粒子布料列为弃用，粒子模拟也不支持 CPU 物理路径；参见 [PhysX 迁移文档](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/deformables_beta/deformable_migration.html) 和 [粒子文档](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/109.0/dev_guide/particles/particles.html)。
- “绳索”由关节刚体链近似。它能弯曲和绕障，但不会像真实绳子一样连续伸长、压扁或发生材料内部形变。若这些现象是任务核心，则在当前“完全不用 GPU 物理”的约束下不纳入第一版。
- 仿真配置必须同时满足 `sim.device=cpu` 和 PhysX GPU dynamics 关闭。当前 `BaseEnv.setup_physics()` 无条件调用 `overwrite_gpu_setting(1)`，实施时要提供明确配置：任务注册项设置 `physics_device: cpu`、`gpu_dynamics_enabled: false`；`process_config()` 将两项写入 `env_cfg.sim`，`BaseEnv.setup_physics()` 再按 `gpu_dynamics_enabled` 调用 `overwrite_gpu_setting(0)`。既有任务未指定时保持原默认行为。不能只凭 `AppLauncher` 使用 GPU 或 `--device_id` 判断物理设备。
- 录制时不启动 VLA/WAM 策略服务。允许使用 GPU 做图像渲染；验收关注 PhysX 仿真本身保持 CPU。

## 三个任务定义

| 编号 | 任务 | 关键操作 | 成功判定 |
|---|---|---|---|
| T1 | 抓绳并把拴球放入篮子 | 夹爪闭合、夹爪附近的绳链胶囊段被抬起，作为抓持代理条件；之后移动绳链 | 抓持代理条件曾成立，且球心进入篮内边界并连续稳定 20 个物理步；仅碰球不计成功 |
| T2 | 把拴球绳穿过圆环 | 抓住绳链，将末端从圆环近侧引导到远侧目标区 | 指定末端越过圆环平面并进入目标区，连续稳定 20 步 |
| T3 | 绕过双立柱并送入终点槽 | 依次把绳链绕过两根立柱，再将拴球端送入终点区域 | 绳链中心线按顺序经过两柱规定侧，球进入终点槽并稳定 20 步 |

三项共享一个绳链物理资产，靠布局和目标参数区分任务。首轮原型使用 12–16 个胶囊链节；先确认 CPU 稳定性和可抓取性，再决定是否增加链节。相邻链节的碰撞需过滤，减少自碰撞抖动；非相邻链节只在稳定性验证后启用自碰撞。

## 文件职责

- 修改 `task/RoboDojo/config/_task.yml` 与 `utils/pipeline_utils.py`：支持任务级 `physics_device` 和 `gpu_dynamics_enabled` 字段，并将其写入 `env_cfg.sim`。
- 修改 `env/environment/base_env.py`：读取 `gpu_dynamics_enabled` 配置；默认值保持既有任务行为，CPU-only 绳索任务显式关闭 GPU dynamics。
- 保留 `env/scene_manager/objects/articulation.py`：复用已有 USD articulation 加载、摆放、复位接口，不新增 FEM 对象类型。
- 新增一个 ASCII USD 资产生成脚本及其生成文件：绳链放入 `Assets/Object/RoboDojo/Articulation/rope_chain/00000/`，环、柱和终点槽放入 `Assets/Object/RoboDojo/Geometry/`。篮子复用 RoboDojo 原有 `Geometry/basket/00002`，不再为 T1 生成自制篮子。绳链使用 12 个胶囊刚体、交替轴向的串联转动关节及末端刚性球；绳链为米白色，球为哑光彩色。
- 新增 `task/RoboDojo/config/rope_*.yml`：分别声明三个任务的对象、标签和目标几何；在 `_task.yml` 注册项设定 `physics_device: cpu` 与 `gpu_dynamics_enabled: false`。配置中不得出现 `Deformable`、`Garment` 或 `Fluid` 对象。
- 新增 `task/RoboDojo/tasks/rope_*.py`：复用 `TaskEnv` 和现有双臂控制接口，实现三个初始场景和语言指令；共用绳链任务基类，避免复制 reset 与状态读取逻辑。
- 修改 `task/RoboDojo/config/_task.yml`：注册三个任务。
- 新增三份 `Assets/Eval_Layout/RoboDojo/arx_x5/0/rope_*.json`：提供可复现的单环境起始布局。
- 新增 `tests/test_rope_tasks.py`：纯配置测试验证三任务都注册、都使用 Articulation、仿真设备为 CPU、GPU dynamics 关闭，并验证每个成功判定的几何边界和稳定计数。
- 保留 `scripts/record_deformable_tasks.py` 的通用无策略录制能力；移除 FEM 专属的仿真设备自动切换。录制的 `--device_id` 只用于应用/渲染，不得写入 `sim.device`。

## 分阶段实施

### 阶段 1：验证 CPU 物理开关

- [x] 在 `_task.yml` 与 `process_config()` 增加 `physics_device`、`gpu_dynamics_enabled` 任务字段；由 `BaseEnv.setup_physics()` 读取开关。缺省行为保持既有任务兼容。
- [x] 添加配置测试，确认 CPU-only 任务会调用 `overwrite_gpu_setting(0)` 且仿真设备为 `cpu`，既有默认任务仍保留原行为。
- [ ] 在集群运行 CPU-only 任务短 smoke test；日志需打印仿真 device 与 GPU dynamics 状态。
- [ ] 验收：CPU 测试能够推进至少 200 个 physics steps；PhysX 日志没有 GPU pipeline/fallback 初始化；GPU 可见性仅用于渲染时不改变物理设备。

### 阶段 2：建立绳链资产和单任务原型

- [x] 创建 12 链节的 capsule articulation 与末端球，关闭 articulation 自碰撞。
- [ ] 在集群验证固定场景：加载双 X5、桌面、篮子和绳链，CPU 仿真推进 500 步并复位两次。
- [x] 给 T1 建立 seed 0 初始布局与配置；实现“抓持代理条件成立，再将球放入篮子”的判定。
- [ ] 验收：录制脚本在不连接策略服务的情况下生成视频；抓持代理与仅触碰球的结果符合成功判定。

### 阶段 3：补齐 T2–T3 和评测接口

- [x] 复用绳链状态读取和稳定计数器，新增圆环穿越、双立柱绕行判定。
- [x] 为每个任务添加固定 seed 0 布局；初始绳链不得穿透桌面、篮筐、圆环或立柱。
- [x] 使用离线几何单元测试覆盖圆环前后侧、柱体绕行顺序和连续稳定步数。
- [ ] 验收：三个任务均能经任务注册器加载、reset、录制；服务器 smoke 确认 sim device 为 CPU。

### 阶段 4：验证并记录 CPU-only 结果

- [ ] 在服务器以 `--num_envs 1` 串行录制三个任务，各录制约 10 秒；策略客户端保持 no-op。
- [ ] 对比同一场景下的 CPU physics 日志，确认不存在 FEM、particle cloth、particle-set 或 GPU PhysX backend 初始化。
- [ ] 记录每任务初始化时间、500 步耗时、复位稳定性和成功判定统计，作为后续是否提高链节数的依据。
- [ ] 验收：三段视频能看清绳链弯曲、拴球运动和目标关系；CPU 仿真不崩溃且连续复位 3 次成功。

## 不纳入第一版的实现

- 不把海绵换成 `MeshCuboidCfg` FEM；FEM 仍是体积软体路径，违反 CPU-only 约束。
- 不用 Garment 做绳子或布袋；粒子布料仍是 GPU 粒子物理，也无法正确表现实心海绵。
- 不用流体/颗粒粒子模拟替代海绵；粒子物理仍要求 GPU。
- 不把绳链宣传为真实连续绳索。若后续必须模拟拉伸、压缩或绳内应力，需要重新评估 CPU 连续体求解器或接受 GPU 路径。
