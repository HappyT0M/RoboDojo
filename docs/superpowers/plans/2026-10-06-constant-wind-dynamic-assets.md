# 可移动任务物体恒定风力实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让无策略录制脚本中的恒定风速能够作用于布料、绳链各刚体节、拴球和可移动任务刚体，同时排除机器人与静态场景物体。

**Architecture:** 将与 Isaac Sim 无关的空气阻力计算、候选体筛选和刚体尺寸估算放在独立 helper 中，以单元测试覆盖。录制脚本在 reset 后构建 Isaac Sim `RigidPrim` view，围绕现有 `env.sim_step` 注入持续施力，并保留布料原生粒子风接口。

**Tech Stack:** Python、NumPy、pytest、Isaac Sim 5.1 `isaacsim.core.prims.RigidPrim`、RoboDojo 场景对象注册表。

---

### Task 1: 纯风阻计算与对象筛选

**Files:**
- Create: `utils/rigid_wind.py`
- Create: `tests/test_rigid_wind.py`

- [ ] **Step 1: 先写失败测试**

覆盖：计算结果方向与相对风速一致；物体速度等于风速时力为零；零风速时力为零；面积/密度/系数输入非法时报错；只有场景中的动态刚体与非固定任务 articulation link 被选中；robot、Geometry、固定 articulation 被排除；胶囊长轴迎风面积随朝向变化。

- [ ] **Step 2: 运行测试确认缺少实现**

运行：`python -m pytest tests/test_rigid_wind.py -q`

预期：因 `utils.rigid_wind` 不存在而失败。

- [ ] **Step 3: 实现最小纯函数**

实现 `compute_drag_forces(wind, velocities, areas, air_density, drag_coefficient, scale)`，按 `0.5 * rho * Cd * A * |v_rel| * v_rel * scale` 计算；实现 capsule、sphere、box 的投影面积函数；实现根据场景注册表类别、`fixed_base` 和对象 prim 路径筛选动态受力路径。

- [ ] **Step 4: 验证纯函数测试通过**

运行：`python -m pytest tests/test_rigid_wind.py -q`

预期：全部新增测试通过。

### Task 2: Isaac Sim 刚体风力控制器

**Files:**
- Modify: `utils/rigid_wind.py`
- Modify: `tests/test_rigid_wind.py`

- [ ] **Step 1: 先写失败测试**

用假的刚体 view 验证控制器读取速度、计算力、调用 `apply_forces`；空受力对象时不调用物理 API；零风时不施力；缺少 view 能力时产生可读诊断而不伪报成功。

- [ ] **Step 2: 运行测试确认控制器行为缺失**

运行：`python -m pytest tests/test_rigid_wind.py -q`

预期：控制器导入或断言失败。

- [ ] **Step 3: 添加 IsaacSim 延迟导入控制器**

创建一个收集 link prim 路径的 `RigidPrim` view，并用 `SimulationManager.get_physics_sim_view()` 初始化。每次更新读取 `get_linear_velocities()`，向受力路径调用 `apply_forces(..., is_global=True)`。Isaac Sim 导入留在控制器构造阶段，保持纯数学单测可在无 Isaac Sim 环境运行。

- [ ] **Step 4: 验证控制器单测通过**

运行：`python -m pytest tests/test_rigid_wind.py -q`

预期：全部新增测试通过。

### Task 3: 录制脚本接入绳链、球和可移动刚体

**Files:**
- Modify: `scripts/record_deformable_constant_wind.py`
- Modify: `tests/test_cloth_wind.py`
- Modify: `tests/test_rigid_wind.py`

- [ ] **Step 1: 先写无布料任务测试**

验证非零风速任务没有 `Garment` 时不再抛出“task has no Garment instances”，并且仍创建刚体风力控制器；有布料任务仍设置 `particle_system.wind` 和 drag。

- [ ] **Step 2: 运行相关测试确认旧行为失败**

运行：`python -m pytest tests/test_cloth_wind.py tests/test_rigid_wind.py -q`

预期：无布料任务测试因现有 `apply_constant_cloth_wind` 抛错而失败。

- [ ] **Step 3: 集成录制期施力与命令参数**

只在 task config 含 `Garment` 时调用 cloth helper。新增保守可调参数 `--rigid_drag_coefficient`、`--rigid_force_scale`。reset 后根据 scene manager 的刚体和 articulation 注册表收集路径，排除固定 articulation 与 robot_manager；在 `env.sim_step` 前包装控制器更新，确保每个底层物理推进步都施力。无风时清空/跳过 rigid controller。

- [ ] **Step 4: 扩展诊断日志**

在 JSONL 中记录受力刚体数、路径/标签、估算面积、质量（可读时）、最后施力范数和跳过原因。布料读回逻辑保留；没有布料但刚体已受力时不再输出“未找到布料”作为整体失败。

- [ ] **Step 5: 验证单测、语法与旧功能**

运行：`python -m pytest tests/test_cloth_wind.py tests/test_rigid_wind.py tests/test_rope_tasks.py -q`，然后运行 `python -m compileall -q scripts/record_deformable_constant_wind.py utils/rigid_wind.py`。

预期：测试通过，语法检查退出码为 0。

### Task 4: 录制冒烟测试说明与集群命令

**Files:**
- Modify: `scripts/record_deformable_constant_wind.py` 文档字符串
- Modify: `docs/superpowers/specs/2026-10-06-constant-wind-dynamic-assets-design.md`

- [ ] **Step 1: 更新用法示例**

加入 `put_rope_ball_in_basket` 风速录制命令，并注明风速是 m/s、`--rigid_force_scale` 用于把理想化空气阻力调到可观察范围。

- [ ] **Step 2: 在有 Isaac Sim 5.1 的集群做有风/无风 smoke run**

录制 `put_rope_ball_in_basket`，对比 `--wind 0 0 0` 与 `--wind 0 0.5 0`，核对日志：绳子 12 个 capsule 和球被选中；篮子、机械臂、桌面不在受力路径；视频及 JSONL 正常生成。

- [ ] **Step 3: 报告无法本地运行的仿真验证**

本地不启动 Isaac Sim 时，只报告静态测试结果；只有取得集群输出后才称运行时 smoke test 通过。
