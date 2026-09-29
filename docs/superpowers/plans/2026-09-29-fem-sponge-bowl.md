# RoboDojo FEM 海绵入碗任务实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 在 RoboDojo 当前 Isaac Sim 5.1 / Isaac Lab 2.3.2 环境中新增一个无风 FEM 海绵入碗操作任务，并提供可在服务器运行的 FEM 检查入口。

**架构：** 通过独立 `deformable` 对象类型接入 Isaac Lab 的 `DeformableObject`，扩展 SceneManager 和 LayoutManager 的生命周期及布局解析。用无 Isaac Sim 依赖的几何判定函数计算海绵节点相对碗的包含比例，再由通用奖励接口维护连续稳定计数；任务 YAML、初始布局和任务类沿用 RoboDojo 现有接口。

**技术栈：** Python 3.11、PyTorch、NumPy、Isaac Sim 5.1、Isaac Lab 2.3.2、pytest。

---

## 文件与职责

- 新增 `RoboDojo/utils/deformable_metrics.py`：纯数值函数，将世界坐标 FEM 节点变换到碗坐标系，计算中心和表面包含比例。
- 新增 `RoboDojo/tests/test_deformable_metrics.py`：验证坐标变换、包含阈值、碗沿和稳定计数边界。
- 新增 `RoboDojo/env/scene_manager/objects/deformable.py`：封装 Isaac Lab `DeformableObject` 的构造、状态读取、复位和销毁。
- 新增 `RoboDojo/Assets/Object/RoboDojo/Deformable/sponge/00000/object.usda`：项目自有的封闭海绵表面网格，由 Isaac Lab 在加载时应用 FEM 属性并生成仿真网格。
- 修改 `RoboDojo/env/scene_manager/scene_manager.py`：注册、初始化、查找、复位、清除和分类返回 `deformable` 对象。
- 修改 `RoboDojo/env/scene_manager/layout_manager.py`：注册 Deformable 布局记录、生成实例路径、读取海绵参数和保存的初始布局。
- 修改 `RoboDojo/env/reward_manager/reward_manager.py`：加入通用 FEM 软体进入容器判定及连续满足计数；使用与其他奖励条件相同的检查和分数通路。
- 新增 `RoboDojo/task/RoboDojo/tasks/put_sponge_in_bowl.py`：无风任务、成功条件、评分和语言指令。
- 新增 `RoboDojo/task/RoboDojo/config/put_sponge_in_bowl.yml`：海绵尺寸、FEM 参数、碗和桌面初始设置。
- 新增 `RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5/0/put_sponge_in_bowl_0.json`：单环境固定起始布局。
- 修改 `RoboDojo/task/RoboDojo/config/_task.yml`：注册任务和评测次数。
- 新增 `RoboDojo/scripts/test_fem_sponge.py`：在 Isaac Sim 服务器环境下单独验证 FEM 对象加载、节点状态和复位，并输出诊断值。

## Task 1：实现并测试海绵入碗几何判定

**文件：**
- 创建：`RoboDojo/utils/deformable_metrics.py`
- 创建：`RoboDojo/tests/test_deformable_metrics.py`

- [ ] **Step 1：先写失败测试**

测试目标：碗无旋转、单位四元数时，全部位于碗内部的节点返回 `center_inside=True` 和 `contained_fraction=1.0`；将点移到碗沿上方后比例下降；绕 Z 轴旋转的碗仍能正确判断局部坐标。

```python
def test_containment_uses_bowl_local_frame():
    metrics = compute_deformable_containment(
        points_w=np.array([[0.0, 0.0, 0.08], [0.01, 0.0, 0.08]]),
        bowl_pos_w=np.array([0.0, 0.0, 0.0]),
        bowl_quat_w=np.array([1.0, 0.0, 0.0, 0.0]),
        interior_bounds=((-0.05, -0.05, 0.02), (0.05, 0.05, 0.12)),
        rim_z=0.12,
    )
    assert metrics.center_inside
    assert metrics.contained_fraction == 1.0
```

- [ ] **Step 2：确认测试按预期失败**

运行：`pytest RoboDojo/tests/test_deformable_metrics.py -q`  
预期：因 `compute_deformable_containment` 尚未实现而失败。

- [ ] **Step 3：实现最小几何函数**

实现 `compute_deformable_containment(points_w, bowl_pos_w, bowl_quat_w, interior_bounds, rim_z)`；将四元数归一化，逆旋转变换到碗局部坐标系，返回中心点局部坐标、中心是否在水平口径内、节点包含比例及有效节点数。空节点、非有限坐标或无效四元数应返回明确错误。

- [ ] **Step 4：补齐边界测试并复跑**

覆盖旋转碗、碗外节点、节点在碗沿上方、零节点和 `contained_fraction` 取值边界。运行同一 pytest 命令，预期全部通过。

## Task 2：添加 FEM 对象封装

**文件：**
- 创建：`RoboDojo/env/scene_manager/objects/deformable.py`
- 创建或扩充：`RoboDojo/tests/test_deformable_metrics.py` 中的纯配置检查（不导入 Isaac Sim）

- [ ] **Step 1：测试对象配置解析的预期**

测试：尺寸、泊松比、杨氏模量、质量、颜色和重置姿态均来自对象配置；尺寸必须为三个正数，泊松比在 `(0, 0.5)`，杨氏模量和质量为正。

- [ ] **Step 2：运行测试并确认缺少配置解析函数**

运行：`pytest RoboDojo/tests/test_deformable_metrics.py -q`  
预期：新增配置解析测试失败，指出配置解析函数缺失。

- [ ] **Step 3：实现最小的 Isaac Lab 软体封装**

在新模块中构造 `DeformableObjectCfg`，使用 Isaac Lab 的 `UsdFileCfg` 加载项目自有封闭 USDA 海绵网格，并配置 `DeformableBodyPropertiesCfg`、`DeformableBodyMaterialCfg` 和预览材质。封装 `initialize()`、`get_nodal_positions_w()`、`get_state()`、`apply_saved_pose()`、`reset()`、`destroy()`。复位时写回 `default_nodal_state_w`，释放此前的节点运动学约束，再调用对象 `reset()`。

对象类型仅在 Isaac Sim 进程启动后导入。若设备不是 CUDA，使用明确错误说明 FEM 仅支持 GPU 仿真，不允许静默跳过海绵。

- [ ] **Step 4：验证配置单测和语法**

运行：`pytest RoboDojo/tests/test_deformable_metrics.py -q` 与 `python -m compileall RoboDojo/env/scene_manager/objects/deformable.py`。预期通过；Isaac Lab 导入和物理运行由服务器 smoke test 验证。

## Task 3：接入 SceneManager 与 LayoutManager

**文件：**
- 修改：`RoboDojo/env/scene_manager/scene_manager.py`
- 修改：`RoboDojo/env/scene_manager/layout_manager.py`
- 修改：`RoboDojo/tests/test_deformable_metrics.py`（只测试不依赖模拟器的类型名和配置规则）

- [ ] **Step 1：添加布局及对象分类失败检查**

测试 `OBJECT_CONFIG_TYPES` 包含 `Deformable`，对象类型集合明确包含 `deformable`，并检查 Deformable 记录按环境清理，不影响既有六类记录。

- [ ] **Step 2：运行测试确认新类型尚未注册**

运行：`pytest RoboDojo/tests/test_deformable_metrics.py -q`。预期新类型检查失败。

- [ ] **Step 3：实现类型注册和生命周期**

在 `SceneManager` 增加 `_deformable_objects`、spawnable 类型、wrapper 工厂分支、对象分类、初始化、保存姿态应用、clear/delete、查询类型映射。CUDA 环境重载时重建 Deformable 对象；遵循 Isaac Lab 的对象初始化顺序，不将其误归入刚体或布料集合。

在 `LayoutManager` 增加 `Deformable` 记录、配置合并、保存布局加载和实例名查询支持。为 Deformable 使用专用解析分支，直接从布局实例读取海绵形状及 FEM 参数，避免按刚体类别目录寻找模型文件。`spawn_category_objects()` 将 `physics.type: deformable` 和这些参数传给 wrapper。

- [ ] **Step 4：跑纯逻辑测试并做静态检查**

运行：`pytest RoboDojo/tests/test_deformable_metrics.py -q`、`python -m compileall RoboDojo/env/scene_manager/scene_manager.py RoboDojo/env/scene_manager/layout_manager.py`。预期全部通过。

## Task 4：添加通用软体容器谓词与任务

**文件：**
- 修改：`RoboDojo/env/reward_manager/reward_manager.py`
- 创建：`RoboDojo/task/RoboDojo/tasks/put_sponge_in_bowl.py`
- 创建：`RoboDojo/task/RoboDojo/config/put_sponge_in_bowl.yml`
- 修改：`RoboDojo/task/RoboDojo/config/_task.yml`
- 创建：`RoboDojo/tests/test_put_sponge_in_bowl_config.py`

- [ ] **Step 1：写成功判定配置和稳定窗口测试**

测试任务名、无风默认值、稳定帧数为正；测试 success 连续满足达到阈值才完成，中间一次不满足会将连续计数清零。

- [ ] **Step 2：运行测试确认任务配置和稳定状态逻辑缺失**

运行：`pytest RoboDojo/tests/test_put_sponge_in_bowl_config.py -q`。预期因任务 YAML 或逻辑未创建而失败。

- [ ] **Step 3：添加奖励谓词和任务实现**

扩展 `RewardManager`，新增 `is_deformable_in_container` 谓词，在每次 reward 检查中从 SceneManager 取得软体节点与刚体碗姿态，调用 Task 1 的几何函数，并按连续稳定窗口输出逐环境布尔结果。任务 `run_reward()` 检查该谓词；`get_score()` 以入碗作为最终 100 分条件，`reset()` 清零稳定计数。

任务 YAML 配置一个 `sponge` Deformable 和一个 `bowl` Rigid；海绵使用偏米黄色的哑光外观，初始位置在碗外、桌面上且在机械臂可达范围内。风力参数明确为关闭。

- [ ] **Step 4：验证任务定义与 YAML**

运行：`pytest RoboDojo/tests/test_put_sponge_in_bowl_config.py RoboDojo/tests/test_deformable_metrics.py -q`，并用 `python -m compileall` 检查新增任务与奖励代码。预期通过。

## Task 5：添加固定布局和服务器 FEM smoke test

**文件：**
- 创建：`RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5/0/put_sponge_in_bowl_0.json`
- 创建：`RoboDojo/scripts/test_fem_sponge.py`
- 修改：`RoboDojo/task/RoboDojo/config/_task.yml`（如前序步骤尚未覆盖）

- [ ] **Step 1：先验证布局解析的预期**

布局测试检查其同时包含一个 `Deformable/sponge` 实例和一个 `Rigid/bowl` 实例，并包含明确的世界位置、姿态、物理类型和海绵参数。

- [ ] **Step 2：确认固定布局测试按预期失败**

运行：`pytest RoboDojo/tests/test_put_sponge_in_bowl_config.py -q`。预期因布局文件缺失而失败。

- [ ] **Step 3：添加布局与服务器检查脚本**

布局使用固定位置确保海绵和碗都在双臂机器人可达区。服务器脚本启动 Isaac Lab/Isaac Sim 后创建一块 FEM 海绵，检查 `data.nodal_pos_w` 形状和有限值、将其复位并再次读取节点，打印网格节点数、初始包围盒、位移范数和复位误差。脚本异常时返回非零状态，关闭仿真应用。

- [ ] **Step 4：本地检查与服务器运行**

本地运行两个新增 pytest 文件和 `python -m compileall`。服务器运行 `python scripts/test_fem_sponge.py --device cuda`，再以 `put_sponge_in_bowl` 加载固定布局录制画面并人工确认压缩、回弹和入碗评分。由于本地没有 Isaac Sim，本地通过不能替代服务器验证。

## 最终回归

- [ ] 运行新增的纯逻辑测试：`pytest RoboDojo/tests/test_deformable_metrics.py RoboDojo/tests/test_put_sponge_in_bowl_config.py -q`。
- [ ] 运行 RoboDojo 现有快速测试：`pytest RoboDojo/tests -q`。
- [ ] 对所有改动的 Python 文件运行 `python -m compileall`。
- [ ] 在服务器执行 FEM smoke test 和无风任务画面验证；记录 Isaac Sim/Isaac Lab 版本、GPU、海绵 FEM 参数、成功判定阈值及视频路径。
- [ ] 确认已有 `fold_clothes` 和 `fold_clothes_random` 的布局、风力开关和场景载入未改变。
