# 自制可形变场景修复（2026-11-02）

当前任务集为 7 个任务、70 个固定布局。保留 `hang_clothes`，移除重复的
`drape_cloth_over_rack`。本次既修改 Git 管理的代码，也重新生成 Hugging Face
管理的 Assets；只同步其中一部分不能完整修复。

## 原因与修改

| 现象 | 原因与处理 |
| --- | --- |
| 柱子、目标槽、折衣板、收纳箱、横杆缺失 | 自制 USD 原来以 Mesh 为根；加载器在引用位置创建 Xform，会覆盖根节点类型。统一改为 Xform 根下的 `mesh` 子节点，并在子 Mesh 上配置材质和三角面碰撞。 |
| 部分环沉入桌下 | 原先旋转环后仍使用固定中心高度 0.915 m。现在由官方 frame 的包围盒、缩放和四元数计算最低点，定位到桌面上方 2 mm。十个布局的高度约为 0.917–0.986904 m。 |
| 衣物外观不协调 | 仅在四个自制衣物任务的布局中覆盖材质：低饱和配色、roughness=0.9、metallic=0；支撑物统一为哑光材质。原生衣物 USD、原生折衣任务和共享灯光配置不变。颜色最终效果须在服务器渲染验收。 |
| 恢复支撑物后出现的初始碰撞风险 | 原生衣物的最低点低于资产原点，旧生成高度会埋入桌面/板。按包围盒抬高到支撑面上方 1 cm；挂衣、收纳场景把衣物与目标物体分开。装饰障碍块落到桌面上，避开折衣板。 |
| 毛巾任务录制失败 | 毛巾也使用根 Mesh，符合相同的加载问题。按任务集精简决定删除任务入口、配置和生成/导出项，不继续维护这一重复任务。 |

加载器默认类型见 [Isaac Sim 5.1 add_reference_to_stage](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.utils/docs/index.html#isaacsim.core.utils.stage.add_reference_to_stage)；
材质参数见 [PreviewSurface](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.materials.PreviewSurface)。

## 需要同步的代码

修改：

- `env/scene_manager/objects/garment.py`
- `scripts/create_rope_assets.py`
- `scripts/generate_rope_eval_layouts.py`
- `scripts/generate_deformable_task_assets.py`
- `scripts/export_deformable_assets.py`
- `task/RoboDojo/config/_task.yml`
- `tests/test_deformable_layout_generation.py`
- `tests/test_preview_recording.py`
- `scripts/README.md`
- `docs/DEFORMABLE_ASSETS.md`

新增：

- `tests/test_deformable_scene_assets.py`
- `docs/DEFORMABLE_SCENE_FIX.md`

删除（使用 Git 同步时会自动反映到服务器）：

- `task/RoboDojo/config/drape_cloth_over_rack.yml`
- `task/RoboDojo/tasks/drape_cloth_over_rack.py`

## 需要同步的资产

以下 6 个目录内的 `object.usd` 和 `metadata.json` 已重新生成：

```text
Assets/Object/RoboDojo/Geometry/rope_post/00000/
Assets/Object/RoboDojo/Geometry/rope_slot/00000/
Assets/Object/RoboDojo/Geometry/folding_board/00000/
Assets/Object/RoboDojo/Geometry/garment_rack/00000/
Assets/Object/RoboDojo/Geometry/garment_bin/00000/
Assets/Object/RoboDojo/Geometry/cloth_block/00000/
```

以下任务的布局 0–9 在 `Assets/Eval_Layout/RoboDojo/arx_x5/0/` 中重新生成：

```text
put_rope_ball_in_basket
pass_rope_through_ring
route_rope_around_posts
fold_clothes_deformable
flatten_clothes
hang_clothes
put_clothes_in_bin
```

旧 `drape_cloth_over_rack_*.json` 和 `square_towel` 本地资产暂留作历史备份；不再
生成、导出或注册。更新批量录制命令时去掉旧任务名。完整新导出包含 18 个物体
实例目录及 70 个布局，保留本轮未修改但其他任务依赖的原生物体。

可以在已安装官方基础资产的服务器上，用更新后的代码重建本次场景，无需先下载新资产包：

```bash
python scripts/create_rope_assets.py --update-fixtures
python scripts/generate_rope_eval_layouts.py
python scripts/generate_deformable_task_assets.py
python scripts/generate_rope_eval_layouts.py --check
```

上述命令不修改原生 `fold_clothes` 布局。不要用不带参数的 `create_rope_assets.py`
替代第一条命令：它还会重建绳子并重置三个绳任务的布局 0。

本地用于发布到原有 Hugging Face dataset 的新导出目录：
`.cache/deformable_assets_20261102_scene_fix/`。其中 `manifest.json` 记录所有文件
路径、哈希和任务依赖，远端仍保持 `Assets/...` 结构。此次未自动上传或推送。

## 验证及服务器验收

本地通过 23 项测试：资产导出、布局生成、环的高度/法向、重复生成一致性，以及
70 个场景的串行录制规划与进程退出。另有 1 项真实 OpenUSD 引用/材质/碰撞测试因
本机缺少 `pxr` 跳过。按现有资产元数据检查，40 个衣物布局均位于桌面范围内，
挂衣和收纳布局的衣物包围盒与支撑物分开。

```bash
python -m unittest discover -s tests -p 'test_deformable*.py' -v
python -m unittest discover -s tests -p test_preview_recording.py -v
python -m unittest discover -s tests -p test_frame_aperture.py -v
```

Mac 上未运行 Isaac Sim、PhysX 或 RTX 视觉验收。项目 doctor 检查仍受已有任务
inventory 分类缺失、基础任务类被计入 inventory，以及本地缺少 PyYAML 的限制；
policy smoke dry-run 因本地没有 XPolicyLab/policy 而未通过，不代表场景仿真已通过。

同步后先录制各任务布局 0，检查物体可见、开口可穿过、衣物材质与落桌状态：

```bash
python scripts/record_deformable_tasks.py \
  --tasks put_rope_ball_in_basket,pass_rope_through_ring,route_rope_around_posts,fold_clothes_deformable,flatten_clothes,hang_clothes,put_clothes_in_bin \
  --seed 0 --layout_id 0 --steps 120 --device_id 0 --headless \
  --wind 0 0 0 --out_dir eval_result/scene_fix_preview
```

移除 `--layout_id 0` 即串行录制全部 70 个布局。新旧场景的任务成功率不宜直接混合：
后续有风/无风比较应使用相同的新代码、资产 revision 和布局编号。
