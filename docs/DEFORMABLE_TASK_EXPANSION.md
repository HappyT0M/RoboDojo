# 十任务扩展：覆盖、拉绳取球、擦拭（2026-11-11）

现有七个任务保留，新增三个任务，每个十个布局（编号 0–9）。
`drape_cloth_over_rack` 继续停用。新增任务使用 `arx_x5`，每个 episode 上限为
900 个 action steps，布局种子为 0；评测成功率由任务判定返回。

## 新任务和物体

| 任务 | 文本指令 | 物体 |
| --- | --- | --- |
| `cover_object_with_cloth` | Cover the object with the cloth so that it is fully covered from above. | `cover_cloth` 覆盖布（48×40 cm）、`cover_block` 矮盒（基准 18×14×5 cm）；部分布局有 `cloth_block` 干扰物 |
| `pull_rope_to_retrieve_ball` | Pull the free end of the rope to bring its attached ball into the marked target area. | 复用 `rope_chain` 绳球、`rope_retrieval_zone` 目标标记（22×16 cm）；部分布局有 `cloth_block` 侧向障碍 |
| `wipe_marked_area_with_cloth` | Wipe the entire marked area on the table using the cloth. | `wiping_cloth` 擦拭布（20×16 cm）、`wipe_board` 标记工作面（40×28 cm）；部分布局有 `cloth_block` 干扰物 |

新建五个物体目录，每个包含 `object.usd` 和 `metadata.json`：

```text
Assets/Object/RoboDojo/Garment/cover_cloth/00000/
Assets/Object/RoboDojo/Garment/wiping_cloth/00000/
Assets/Object/RoboDojo/Geometry/cover_block/00000/
Assets/Object/RoboDojo/Geometry/rope_retrieval_zone/00000/
Assets/Object/RoboDojo/Geometry/wipe_board/00000/
```

30 个新增布局保存在 `Assets/Eval_Layout/RoboDojo/arx_x5/0/<任务名>_<0–9>.json`。
变化包含三种房间、桌面尺寸、背景光照、干扰物有无/尺寸、任务物体位置和角度。
布料使用五种低饱和度哑光颜色；覆盖任务还改变矮盒大小。
布料采用三角网格粒子布，绳球沿用 CPU 刚性链。两者接入现有风力实现，
分别受粒子风场和刚体风阻影响，`--wind 0 0.5 0` 的含义保持不变。

## 成功判定

- 覆盖：曾被闭合夹爪抓取，随后释放；目标顶面采样点至少 98% 被实际布料三角面
  从上方覆盖，且布面接近顶面。最大粒子位移不超过阈值，连续满足 15 次检查。
  不以包围盒或凸包代替布面，避免折叠空洞也计为覆盖。
- 拉绳取球：抓取自由端前三节之一，夹爪和自由端移动且小球同时靠近目标，累计有效
  拉动至少 12 cm；小球相对初始位置移动至少 18 cm，进入目标有效区域并保持
  15 次检查。夹爪靠近球体的直接操作会使该 episode 不满足此任务要求。
- 擦拭：夹爪闭合且靠近布料，布料接近工作面，夹爪和接触区域布点同时移动时，
  才累计实际三角面覆盖的区域；累计覆盖至少 90%、有效擦拭距离至少 15 cm。
  静置布料、悬空移动、无夹持的风吹运动不累计。这里模拟区域擦拭，不涉及液体或污渍材质变化。

抓取与接触采用位置、夹爪开合量和高度阈值的几何近似，没有引入接触力传感器。
X5 抓取点由腕部位姿沿局部 +X 加上机器人配置中的 `gripper_bias` 得到。
覆盖率是离散采样近似，并非连续面积的精确积分。阈值可在三个任务 YAML 的
`DeformableManipulation` 下调整；旋转/缩放后的目标在其局部坐标中判定。
这些阈值需要在服务器上通过实际成功与失败轨迹进一步标定。

每个环境独立保存布面拓扑、初始球位、累计覆盖及连续成功计数；reset/soft_reset
后清零，避免前一个布局的进度进入下一个布局。

## 代码同步清单

新增文件：

```text
docs/DEFORMABLE_TASK_EXPANSION.md
env/reward_manager/deformable_manipulation.py
utils/deformable_manipulation_metrics.py
scripts/generate_additional_deformable_assets.py
scripts/configs/pi05_deformable_no_wind.json
scripts/configs/pi05_deformable_wind_y_0p5.json
task/RoboDojo/config/cover_object_with_cloth.yml
task/RoboDojo/config/pull_rope_to_retrieve_ball.yml
task/RoboDojo/config/wipe_marked_area_with_cloth.yml
task/RoboDojo/tasks/cover_object_with_cloth.py
task/RoboDojo/tasks/pull_rope_to_retrieve_ball.py
task/RoboDojo/tasks/wipe_marked_area_with_cloth.py
tests/test_deformable_manipulation.py
```

修改文件：

```text
docs/DEFORMABLE_ASSETS.md
env/reward_manager/func_parser.py
scripts/README.md
scripts/configs/pi05_deformable.json
scripts/export_deformable_assets.py
scripts/internal/task_inventory.py
task/RoboDojo/config/_task.yml
tests/test_deformable_layout_generation.py
tests/test_preview_recording.py
```

两份单组 JSON 配置此前尚未提交，本次一并列入同步；三份 JSON 的任务列表都已更新为十个。
单组配置使用 GPU 0；合并配置继续使用 0/1 和 2/3 两组 GPU，运行前按服务器实际资源修改。
任务清单增加本分支的 `deformable` 分组；官方五组保持原任务归属。

## 本机上传代码和资产

当前分支为 `main`。以下代码提交包含当前工作区的修改，`Assets/`、`.cache/` 和日志均被 Git 忽略。
本次未执行 commit、push 或 Hugging Face 上传。

```bash
cd /Users/happyt0m/developer/robotics/RoboDojo
git status --short
git add -A
git commit -m "[Task] feat: add covering rope retrieval and wiping tasks"
git push origin main

hf upload HappyT0M/RoboDojo-Deformable \
  .cache/deformable_assets_20261111_ten_tasks . \
  --repo-type dataset
```

本次导出目录已生成，可以直接上传；包含全部十任务需要的 23 个物体目录、100 个布局、
150 个资产文件（约 25.40 MiB）及 manifest，不只是三个新任务的增量。
目录名只在本地使用，HF 顶层仍是 `Assets/`，不会新增一个名为日期的子仓库。

日后修改资产需重新生成并导出到一个新的目录，再上传该新目录。例如生成/校验本次新增内容：

```bash
python scripts/generate_additional_deformable_assets.py
python scripts/generate_additional_deformable_assets.py --check
```

导出器拒绝覆盖已存在的打包目录，防止上传混合版本。

## 服务器下载

先进入服务器现有的 RoboDojo 根目录。实验室常用路径为
`/home/xiebingsheng/workspace/Guo/RoboDojo`，最近使用的 AutoDL 路径为
`~/autodl-fs/RoboDojo`；以实际安装目录为准。

```bash
git pull --ff-only origin main
hf download HappyT0M/RoboDojo-Deformable \
  --repo-type dataset --revision main \
  --include 'Assets/**' --local-dir .

python scripts/generate_additional_deformable_assets.py --check
python scripts/generate_rope_eval_layouts.py --check
python scripts/internal/task_inventory.py --dimension deformable --only-runnable --check
```

新增生成器应打印 `Verified 5 assets and 30 layouts (40 files).`，任务清单应列出十个任务。
这里用真实存在的 `main` 分支下载当前最新资产，不需要填 SHA 占位符。
正式对比实验建议记录代码 commit 和 HF 实际 commit，使两组使用同一资产版本。

下载会按同名相对路径合并覆盖；机器人、房间、桌面材质和背景等官方基础资产仍需保留。
本子集包不能替代官方完整安装。如果镜像下载仍是旧内容，可在该次 `hf download` 前加
`HF_ENDPOINT=https://huggingface.co` 直接访问官方站点。
命令语法见 [HF CLI 官方文档](https://huggingface.co/docs/huggingface_hub/guides/cli)。

## 服务器验收

先不加载模型，录制三个新任务的布局 0：

```bash
python scripts/record_deformable_tasks.py \
  --tasks cover_object_with_cloth,pull_rope_to_retrieve_ball,wipe_marked_area_with_cloth \
  --seed 0 --layout_id 0 --steps 120 --device_id 0 --headless --no-wind
```

去掉 `--layout_id 0` 后顺序录制 30 个场景，每个场景使用独立进程并退出释放资源。
有风预览把 `--no-wind` 替换为 `--wind 0 0.5 0`；单 GPU 上依次执行。

十任务 Pi0.5 评测继续使用 JSON，分别启动：

```bash
python scripts/eval_from_config.py --config scripts/configs/pi05_deformable_no_wind.json
python scripts/eval_from_config.py --config scripts/configs/pi05_deformable_wind_y_0p5.json
```

相关单元测试共 39 项通过、2 项因缺少 OpenUSD 跳过。
本地验证覆盖：成功/失败几何轨迹、真实尺寸擦拭布的两次往返覆盖、夹爪偏移、环境坐标、
重置隔离、生成器确定性、任务/布局标签、资产哈希、100 场景录制调度、JSON 启动器。
`doctor --skip-isaac --skip-conda --skip-policy` 在本机带 PyYAML 的 Python 下通过，
JSON dry-run 生成 20 条任务命令。当前 Mac 未安装 Isaac Sim/OpenUSD，且本地
XPolicyLab 子模块未初始化；真实 USD 组合测试跳过，`robodojo.sh smoke --dry-run`
在检查 Pi_05 的 `eval.sh` 时被阻止，尚未执行真实物理仿真或模型评测。

服务器运行相关测试：

```bash
python -m unittest discover -s tests -p 'test_deformable*.py' -v
```

录制只能确认场景显示与静置稳定性；实际可抓取性、接触稳定性和成功判定仍需手动轨迹
或模型轨迹验收。评测验收同时检查进程退出码和 `_result.json` 的 `eval_time >= 1`。
