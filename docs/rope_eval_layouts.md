# 绳索任务先导评测布局

三个任务在 `Assets/Eval_Layout/RoboDojo/arx_x5/0/` 下各有编号 0–9 的十个布局。编号 0 保留基准摆放；其他布局包含固定的场景和绳索起点变化。`generate_rope_eval_layouts.py` 对指定复测布局额外做小幅间距调整：

| 任务 / 布局 | 调整 |
| --- | --- |
| `put_rope_ball_in_basket` / 2 | 在上一轮调整基础上，绳索沿 X 正方向移动 2 cm、篮筐沿 X 正方向移动 1.5 cm；移除额外的 rope_post |
| `pass_rope_through_ring` / 2、9 | 在上一轮调整基础上，绳索沿 Y 负方向移动 2 cm、框沿 Y 正方向移动 2 cm；框保持基准直立朝向并保留桌面间隙；移除额外的 rope_post |
| `route_rope_around_posts` / 5、7 | 在上一轮调整基础上，绳索起点沿 Y 负方向再移动 3 cm；posts 和目标槽保持原位 |

这些调整保留任务目标和障碍布局，目的是减少初始摆放过近或绳索触碰导致的异常；是否能解决 PhysX 异常仍需在 Isaac Sim 中重跑验证。

仓库的 `.gitignore` 忽略 `Assets/`。在评测机上准备好原始资产和三个 `_0.json` 后，运行：

```bash
python3 scripts/generate_rope_eval_layouts.py
python3 scripts/generate_rope_eval_layouts.py --check
```

如果缺少原始绳链资产或 `_0.json`，先运行 `python3 scripts/create_rope_assets.py`。生成器核对三个基准布局的物体位置与穿框朝向，避免从不同版本的场景生成不可比较的布局。任务配置的 `eval_nums` 已设为 10。

生成器检查桌面边界、不同布局的唯一性，以及篮筐、穿框和绕花瓶任务的初始目标侧关系。这些是静态检查；正式测评前仍需在 Isaac Sim 中检查每个布局的初始化、稳定性、画面可见性和成功判定。
