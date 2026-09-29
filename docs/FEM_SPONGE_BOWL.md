# FEM 海绵入碗任务

这个任务使用 RoboDojo 当前的 Isaac Sim / Isaac Lab 依赖，新增一个实心 FEM 海绵和一个刚体碗。没有接入任何 MuJoCo 资产或实现方式，也不启用风力。

## 当前场景

- 碗：复用 `Assets/Object/RoboDojo/Rigid/bowl/00007`，外部尺寸约为 16 × 16 × 6 cm。
- 海绵：项目自有的封闭长方体网格，尺寸 6 × 5 × 3 cm，位于桌面上、初始时与碗分开。
- 海绵参数：密度 250 kg/m³、杨氏模量 120 kPa、泊松比 0.35、弹性阻尼 0.2、动摩擦系数 0.6。按外形体积估算，初始质量约 2.25 g。
- 成功判定：FEM 节点至少 85% 位于碗沿以下的保守内接区域，节点中心也在区域内，并连续满足 30 个检查步。
- 固定评测布局：`arx_x5` 的 seed 0，目前只配置一个布局样本。

## 在服务器上验证

先确认 FEM 资产能加载、推进并复位：

```bash
python scripts/test_fem_sponge.py --device cuda:0 --steps 30 --headless
```

再录制完整任务场景画面（不连接 VLA 或 WAM）：

```bash
python scripts/record_deformable_tasks.py \
  --task_name put_sponge_in_bowl \
  --num_envs 1 \
  --seed 0 \
  --steps 300 \
  --disable_xlens_snapshot \
  --out_dir eval_result/sponge_bowl \
  --headless
```

如果 smoke test 通过，再检查录制画面中的海绵位置、颜色、碗的可见性和机器人可达性。该脚本只能验证对象创建、节点读数和状态复位；实际抓取压缩效果与任务成功率仍需在服务器物理仿真中确认。
