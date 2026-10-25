# 绳索任务先导评测布局

三个任务在 `Assets/Eval_Layout/RoboDojo/arx_x5/0/` 下各有编号 0–9 的十个布局。编号 0 保留现有场景；其余布局对场景整体及绳索起点施加固定的桌面 XY 偏移。目标朝向、绳链关节初态、任务参数、相机和桌面不变。

仓库的 `.gitignore` 忽略 `Assets/`。在评测机上准备好原始资产和三个 `_0.json` 后，运行：

```bash
python3 scripts/generate_rope_eval_layouts.py
python3 scripts/generate_rope_eval_layouts.py --check
```

如果缺少原始绳链资产或 `_0.json`，先运行 `python3 scripts/create_rope_assets.py`。生成器核对三个基准布局的物体位置与穿框朝向，避免从不同版本的场景生成不可比较的布局。任务配置的 `eval_nums` 已设为 10。

生成器检查桌面边界、不同布局的唯一性，以及篮筐、穿框和绕花瓶任务的初始目标侧关系。这些是静态检查；正式测评前仍需在 Isaac Sim 中检查每个布局的初始化、稳定性、画面可见性和成功判定。
