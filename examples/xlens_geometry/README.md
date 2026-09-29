# RoboDojo XLens 示例

`build_tower_manifest.example.json` 展示了一个 `build_tower` 场景的输入格式。真实运行时，建议用 `save_robodojo_snapshot()` 在环境 reset 后自动导出 `scene.json`，不要手工填写相机位姿。

当前仓库的无策略预览脚本已经接入该导出流程。先确保
`env_cfg/camera/camera_config.yml` 中目标相机开启了
`instance_id_segmentation_fast`，然后在 RoboDojo 根目录运行：

```powershell
python scripts\record_deformable_tasks.py `
  --task_name fold_clothes `
  --num_envs 1 `
  --env_cfg_type arx_x5 `
  --device_id 0 `
  --seed 0 `
  --steps 1
```

脚本会在 `env.reset()` 后自动生成：

```text
logs/xlens_geometry/fold_clothes_episode_0000/scene.json
```

并从实例 mask 中自动发现可见的非背景 label，写入 `scene.json` 的
`objects` 字段。`--xlens_min_mask_pixels` 可以调整过滤小区域的阈值；如果
只想录制视频而不导出 XLens 快照，可以增加 `--disable_xlens_snapshot`。

需要为每个相机保存：

- `rgb.png`：RGB 图像；
- `instance_mask.npy`：实例 ID 或语义 ID mask；
- `K`：相机内参；
- `c2w`：相机到 RoboDojo world 坐标系的 4×4 位姿矩阵。

然后从 `RoboDojo` 根目录运行：

```powershell
python scripts/run_xlens_geometry.py `
  --manifest path/to/scene.json `
  --ckpt path/to/XLens/checkpoints/xlens.pth `
  --xlens-root ..\XLens `
  --config ..\XLens\configs\xlens_vits.yaml `
  --out-dir path/to/geometry_out
```

`--config` is required when the downloaded checkpoint is a bare
`model.safetensors`; it can be omitted for a `.pth` checkpoint that contains
the architecture configuration.

输出的 `geometry.json` 可在任务开始前转为 LLM 文本、VLA state 或 WAM 静态 geometry token。
