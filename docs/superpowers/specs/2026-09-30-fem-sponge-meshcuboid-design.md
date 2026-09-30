# FEM 海绵改用程序化网格的设计

**日期：** 2026-09-30  
**状态：** 等待用户审阅  
**范围：** 修复 `put_sponge_in_bowl` 中 FEM 海绵 prim 未应用可形变 schema 的问题

## 决策

将海绵的生成方式从加载普通 USDA 表面网格切换为 Isaac Lab `MeshCuboidCfg` 程序化网格，同时继续将其配置为 PhysX FEM 体积软体。`MeshCuboidCfg` 只负责生成海绵几何；FEM 属性和软体材料继续由现有海绵配置提供。

海绵尺寸读取任务/布局中的 `size`（当前为 `0.06 × 0.05 × 0.03 m`），并保留布局位置、姿态、比例、颜色和现有 FEM 材料参数。海绵仍通过现有 RoboDojo `DeformableObject` 封装及场景生命周期创建、复位、销毁；碗、机械臂、任务判定和风力计划不变。

## 为什么这样改

当前错误显示加载后的目标 prim 上没有 `PhysxDeformableBodyAPI`。现有海绵 USDA 是普通 `Xform` 与表面 `Mesh`，当前 `UsdFileCfg` 加载路径没有将其转换为可形变对象。Isaac Lab 的 mesh spawner 会在生成网格时应用可形变属性，因此使用 `MeshCuboidCfg` 可以绕过这条失败的 USD schema 路径。

此修改不改变物理路线：海绵继续是 PhysX FEM 体积软体，不切换到刚体、布料粒子、PBD 或 Newton/VBD。第一版几何仍为长方体近似。

## 修改边界

- 修改 FEM 对象封装中的海绵 spawn 配置：改用 `MeshCuboidCfg`，传入尺寸、可形变属性、FEM 材料和视觉材质。
- 继续使用现有配置校验、设备检查、节点状态、复位和删除接口。
- 更新 FEM smoke test，使它验证程序化海绵创建、节点状态有限且可读取，以及复位后状态恢复。
- 更新相关单测/说明，避免把普通 USDA 文件存在作为生成海绵的必要条件。
- 不改任务布局、碗、机械臂、成功标准、扰动接口或 Isaac Lab/Isaac Sim 版本。

## 验收

1. 纯 Python 测试和语法检查通过。
2. 在 AutoDL 的 Isaac Sim 5.1 / 当前 Isaac Lab 环境中，`MeshCuboidCfg` 成功创建 FEM 海绵，目标网格 prim 包含可形变 schema，节点位置有限且可读。
3. FEM 对象复位能恢复其初始节点状态。
4. 之后通过录制任务画面检查海绵可见，并单独确认夹爪接触压缩和松开回弹；本地无 Isaac Sim 时不宣称物理验证已通过。

## 风险与边界

- `MeshCuboidCfg` 的具体参数签名要以服务器运行的 Isaac Lab checkout 为准；本地 Windows 没有完整 Isaac Sim 运行时，无法完成物理验证。
- 体积 FEM 网格生成依赖运行环境具备相应四面体化能力。若服务器报出依赖或网格生成错误，应基于完整错误调整兼容方式，不静默退回到普通刚体或静态 USD。
- 程序化几何为长方体，外观可通过现有视觉材质改善，但不包含海绵孔隙或圆角细节。
