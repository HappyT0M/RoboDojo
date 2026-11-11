# 十个可形变任务的 Hugging Face 资产管理

GitHub 管理任务 Python/YAML、资产生成/导出脚本和下载说明。整个 `Assets` 目录由
Hugging Face 管理；本地仍使用 RoboDojo 原有目录结构，任务代码无需改资产根路径。

自定义资产使用 Dataset 仓库 `HappyT0M/RoboDojo-Deformable`。当前导出范围为
十个任务的 100 个布局和布局直接引用的 23 个物体实例目录。未使用的 `rope_ring`
不会被导出：当前穿环任务使用的是原生 `frame`。
`drape_cloth_over_rack` 已从任务集移除，保留 `hang_clothes`；新导出不再包含
`square_towel` 或旧的搭毛巾布局。旧文件可以留作历史备份，不会被当前任务集使用。

| 物体分类 | 目录及编号 |
| --- | --- |
| Articulation | `rope_chain/00000` |
| Garment | `Top_Long/{00001,00004,00006,00007,00009,00011}`；`cover_cloth/00000`、`wiping_cloth/00000` |
| Geometry | `basket/{00002,00003,00004,00005}`；`frame/00000` |
| Geometry | `cover_block/00000`、`wipe_board/00000`、`rope_retrieval_zone/00000` |
| Geometry | `rope_post/00000`、`rope_slot/00000`、`folding_board/00000`、`garment_rack/00000`、`garment_bin/00000`、`cloth_block/00000` |

这些目录都位于 `Assets/Object/RoboDojo/`。导出完整物体目录，保留模型、元数据、
纹理和其他附属文件；共享物体只保存一份。布局位于 `Assets/Eval_Layout/RoboDojo/arx_x5/0/`。

这是需要与官方基础资产合并使用的子集包。机器人、三个房间
`Simple_Room_nolight/orange/teal`、背景 `brown_photostudio_02_4k.hdr`、桌面材质
`material_0122` 及地面等通用依赖继续使用官方安装。导出器会记录这些依赖，
但不递归解析 USD 内部的跨目录引用，所以该子集不能单独替代完整官方安装。

2026-11-11 新增覆盖、拉绳取球和擦拭三个任务，各十个布局。生成脚本为
`scripts/generate_additional_deformable_assets.py`，`--check` 只校验、不改写。
具体成功判定、文件清单和本次同步命令见 [十任务扩展说明](DEFORMABLE_TASK_EXPANSION.md)。

## 本地导出

在 RoboDojo 根目录运行：

```bash
python scripts/export_deformable_assets.py --output .cache/deformable_assets_v1
```

如已知当前官方资产的确切 commit，可增加 `--base-revision 完整COMMIT_SHA`。
未指定时 manifest 会明确记录为 `null`，不会把未知来源版本写成已锁定版本。

输出目录必须是新目录；可使用日期命名，例如 `deformable_assets_20261111_ten_tasks`，避免
旧的无关文件混入。这只是本地打包目录名，上传目标仍是同一个 Hugging Face 仓库根目录。导出结果包括：

```text
deformable_assets_v1/
├── README.md
├── manifest.json                 # 任务→资产映射、大小、SHA-256、官方依赖
└── Assets/
    ├── Object/RoboDojo/...        # 23 个实例目录
    └── Eval_Layout/RoboDojo/arx_x5/0/...  # 100 个自定义任务布局
```

原生 `fold_clothes` 布局不在包内；自定义折衣任务使用 `fold_clothes_deformable`。
复用的官方资产应附带来源说明和对应的原许可证；导出模板不会替它们重新指定许可证。

2026-11-02 的支撑物缺失、穿环高度及衣物外观修复需要同时更新代码和资产。
重新生成命令、同步清单和验证限制见 [场景修复说明](DEFORMABLE_SCENE_FIX.md)。

## 创建并上传到 Hugging Face

把以下 `YOUR_NAME` 替换为你的用户名或组织名。先在本机登录，再创建 Dataset 仓库：

```bash
pip install -U huggingface_hub
hf auth login
hf repos create YOUR_NAME/RoboDojo-Deformable --repo-type dataset
hf upload YOUR_NAME/RoboDojo-Deformable .cache/deformable_assets_v1 . --repo-type dataset
```

这些是供用户执行的发布命令；导出脚本本身不会登录、创建远端仓库或上传。
上传整个导出目录，因此远端顶层仍是 `Assets/`，不要再额外嵌套一层 `Assets`。
更新现有仓库时普通上传会添加/覆盖文件；如 manifest 移除了资产，也要同步清理远端
对应的旧文件。首次发布前补全 README 中的项目介绍和来源/许可证信息。

## 在实验室服务器安装

先使用现有 `scripts/init_assets.sh` 安装官方基础资产，再在 RoboDojo 根目录下载
自定义资产包的 `Assets/` 子树：

```bash
bash scripts/init_assets.sh
hf download YOUR_NAME/RoboDojo-Deformable \
  --repo-type dataset --revision YOUR_ASSET_COMMIT_SHA \
  --include 'Assets/**' --local-dir .
```

`YOUR_ASSET_COMMIT_SHA` 应替换为此次发布的完整 commit SHA。保留目录结构，下载
后文件落在现有的 `Assets/Object/...` 和 `Assets/Eval_Layout/...`，其他官方资产保留。
不要把现有 `init_assets.sh` 的 `HF_REPO_ID` 直接替换为这个子集仓库：原脚本要求
完整的 `Robots/Object/Material/Eval_Layout` 目录，且 Assets 已存在时会跳过下载。

在 GitHub 的实验配置或一个资产版本记录文件里保存官方仓库 revision、自定义仓库
revision 和代码 commit，使无风/有风实验使用同一版场景。每轮更新先导出新的目录，
再上传并更新所记录的资产 revision。

## 从 Git 停止跟踪 Assets

`.gitignore` 已忽略整个 `/Assets`，当前 Git 索引不包含资产。用 `git ls-files Assets`
检查应没有输出。正常提交代码即可，旧 Git 提交中的历史资产不会因此删除。

参考：[HF CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)、
[HF 文件下载与版本参数](https://huggingface.co/docs/huggingface_hub/guides/download)。
