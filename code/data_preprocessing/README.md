# 数据预处理

共享数据预处理项目。源码直接放在本目录，后续预处理也在这里扩展。96 × 96 扩充图片集放在 `../data/`；其他处理产物和实验记录写入本项目 `runs/`。

## 96 × 96 扩充图片集

`expand_rodent96.py` 保留旧 `mouse_mixed` 的原始像素和 train/val/test 划分，向训练集补入实验室脑部 RARE，以及尚未使用的公开结构像。三个图片目录只放 PNG，当前数据集根目录另附 README：

```text
../data/rodent96_expanded_260917/
  README.md
  train/*.png
  val/*.png
  test/*.png
```

```bash
cd code/data_preprocessing
../../.venv/bin/python expand_rodent96.py --out ../data/rodent96_expanded_260917
```

输出路径必须是 `code/data/` 下尚不存在的新目录；重新生成时换一个新目录。`--dry-run` 仅检查来源，`--legacy` 和 `--raw` 可指定旧训练集及原始数据目录。脚本仅读取现有 JSON/NPY 以保留旧像素、识别已有扫描及排除验证/测试动物；不输出 JSON、NPY、NIfTI、MAT、缓存或原始采集文件。计数及排除原因打印到终端。成功前图片暂存于同级 `.partial` 目录，全部完成后才改为最终目录名。

新增图采用原生切片、每体积正值 q99.5 灰度归一化、保持物理比例的前景裁剪和抗混叠缩小；不放大、不补边，不复制增强视图。原生 120 × 160 等低于 192 的实验室数据可用于这次 96 导出。保留 12% 栈端筛除和既有公开来源窗口，排除低信号、裁框不完整及完全相同的输出像素。实验室扫描按原始重建哈希去重，回波分别读取。

`VISUAL_EXCLUSIONS` 保存人工预览确认有严重重影的两组原始重建哈希，所有回波整组排除。其余自动筛选仍不能代替逐切片人工质检；部分栈端图在体积统一灰度窗口下较暗。

新增数据只进入 `train/`，原 `val/test` 逐像素保持原样。公开来源按已有跨库动物映射及 COMR 分组排除原验证/测试动物，也不再次处理旧数据中已有的原始体积。实验室 M0427 与 `RAREImage1.mat` 参考同源，整组排除；运动实验、覆盖或方向待核的扫描、身体/肿瘤及 CEST 不进入此批次。实验室 `RAT` 目录的物种标签尚不能逐一扫描确认，文件名前缀 `lab-RAT` 仅表示来源集合，不能据此称为纯小鼠或纯大鼠。

图片为 **96 × 96 单通道 16 位 PNG**，读取用 `np.asarray(Image.open(path), dtype=np.float32) / 65535`；不要用 `convert('L')` 或默认的 8 位 RGB 转换。文件名保留来源、subject、原切片号/旧数组下标及回波或视图。旧图不重新归一化或裁剪，新增图的处理方式与旧图有所不同。原 `prior96/train_strong.py` 仍使用旧 JSON/NPY 数据接口；此脚本只负责图片集预处理，不启动训练。

`build_png_gallery.py` 直接扫描三个图片目录，生成包含全部图片的离线 HTML。支持划分、旧/新增数据、来源、序列筛选、文件名搜索、分页跳转和逐张放大查看。预览页放在数据目录之外，不向图片目录写入其他文件：

```bash
../../.venv/bin/python build_png_gallery.py \
  --data ../data/rodent96_expanded_260917 \
  --out ../../experiments/rodent96_preview_260917/index.html \
  --asset-mode symlink
```

按用户要求，`--asset-mode symlink` 将 HTML 同级的 `images/<数据集名称>/` 链接到 `code/data/` 中的原图目录，不复制 PNG。聊天预览不能跟随跨目录符号链接，因此使用同级 `preview_tiles/` 中的轻量 8 位 JPEG 拼图显示全部图片，原始 16 位 PNG 不受影响。分享预览时带上 HTML 和 `preview_tiles/`；原图访问还需要符号链接的目标目录。图片增删后重新运行上述命令。另保留 `--asset-mode copy` 独立复制模式。

## 192 × 192 鼠脑 PNG

输入为 [`../data/rodent_mri/`](../data/rodent_mri/README_260914.md)，读取公开 NIfTI 和实验室已有的 Bruker `pdata/1/2dseq` 图像。输入文件保持原样。

```bash
cd code/data_preprocessing
uv pip install --python ../../.venv/bin/python -r requirements.txt
../../.venv/bin/python export_rodent_brain.py --out runs/rodent_brain192_native_crop_260914
../../.venv/bin/python build_gallery.py --run runs/rodent_brain192_native_crop_260914
../../.venv/bin/python verify_export.py --run runs/rodent_brain192_native_crop_260914
```

输出必须使用 `runs/` 下的新目录。快速检查可加 `--preview-per-dataset 3`，按来源排序各抽前、中、后位置。`--sources public` 或 `--sources lab` 可选择数据分支。

图像名称为 `{num}_{xxx}_{序列}.png`。`num` 从 1 连续递增；`xxx` 包含数据集、受试者/扫描标识和原始切片编号，例如 `1_ds002868-sub-001-ses-1-sl008_T2w-RARE.png`。实验室扫描使用原重建哈希前 12 位作标识，回波另行区分。序列中的 TE 单位为毫秒；短 TE 的 RARE 不一概标成 T2 加权。

| 产物 | 内容 |
|---|---|
| `images/` | 192 × 192、单通道、16 位 PNG；`--bit-depth 8` 可改为 8 位 |
| `gallery/index.html` | 按数据集、序列筛选和翻页的预览 |
| `gallery/overview.jpg` | 各数据集代表来源的概览 |
| `gallery/source_pages/` | 每来源首、中、末已接受切片的质检拼图 |
| `manifest.jsonl` | 每张 PNG 的来源、切片/帧编号、回波、物理比例、归一化及哈希 |
| `selected_sources.json` / `processed_sources.json` | 来源选择、空间方向、切片范围与处理结果 |
| `excluded_sources.json` / `rejected_slices.jsonl` | 来源与切片的排除原因 |
| `native_resolution_exclusions.json` | 原始层内任一边小于 192 的来源、实际矩阵和排除原因 |
| `summary.json` / `failed_sources.json` | 数量统计与读取错误；有读取错误时程序返回非零状态 |
| `config.json` | 参数、命令、代码哈希与运行时间 |

公开数据读取器识别 ds002868、ds002870、ds005186、ds005236、ds006663 和 Aging 的 T2 / RARE 结构图像。当前默认 `--min-native-size 192`，原始层内矩阵任一边小于 192 就在导出前排除，因此实际保留 ds005186、ds005236、ds002870 的 256 × 256 系列，以及实验室满足矩阵要求的来源。ds002870 为大鼠；其 144 × 144 系列、ds002868、ds006663、187 × 225 的 Aging 和实验室 120 × 160 系列不进入当前批次。

实验室数据使用已有去重清单中的 RAT 与 M0427 脑部 RARE 候选，双回波分开。身体、肿瘤、CEST、motion/topup/hs_brain、DWI、mask、实虚分量、GRE，以及通道或方向尚未核清的其他来源暂不纳入。

保留原始层面，不沿厚层方向插值增加切片。公开数据使用源数组轴 2；不同发布库的方向约定不同，不根据轴标签一律重建成同一解剖平面。空间与强度信息由 [NiBabel](https://nipy.org/nibabel/coordinate_systems.html) 读取，原始 affine 和实际显示操作记录在清单中。

Aging 的多个代表来源经视觉核对后统一在层内旋转 180°；不改变层序，不据此声称已核验解剖左右方向。ds002868 的晚层在统一体积窗口下较暗，仍保留实际脑组织信号。

每个单回波体积以有限正值的 99.5 百分位作上限，负值截至 0，映射到 PNG 灰度范围。图像内不添加文字或色条。

默认 `--crop-mode foreground`：平滑后以该层 99 百分位的 40% 作定位阈值，形态开运算去掉细小连接，保留最大组织分量及附近较大的分量，围绕其包围框留每侧 10% 边距。定位掩膜只决定裁框，不把掩膜外的像素抹黑，也不改变图内组织。不同切片的裁框可以不同，逐图记录原生坐标和输出到输入的采样变换。

裁框每个方向至少覆盖 192 个原生像素。等间距且组织能被 192 × 192 覆盖时，直接整数裁剪，不插值；所需框更大时先抗混叠，再等比例缩小。原始 header 间距差异不超过 0.01% 时可保留直接裁剪，实际输出间距仍逐轴记录。明显非等间距的来源按物理正方形裁剪，例如 256 × 256、间距 0.08203125 × 0.109375 mm 会使用约 256 × 192 的原生区域，再将行方向缩到 192，以保持物理比例。

裁框限定在采集视野内，不新增黑边，不插值放大；无法满足几何约束或不能保留至少 98% **检测到的定位前景**时排除该层。这个保留率不是脑组织完整性的验证，阈值可能受脑室、强度不均和周边肌肉影响，仍须查看预览。图内原有空气背景会保留。`--crop-mode full` 保留旧版全视野补方方式供对照，仍执行原生尺寸筛选。

首尾切片按来源候选窗口和 `--trim-fraction`（默认每端 12%）筛除，再排除非有限数、低信号、近常数、前景过小和完全相同的输出像素。此筛选不是脑分割：仍可能包含颅骨、周边组织、病变和采集伪影，需要结合预览继续挑选。

192 × 192 是输出矩阵，不代表实测有效分辨率。当前批次要求原生矩阵两边均不少于 192，且不放大；原生矩阵、像素间距、`transform.upsampled`、`direct_native_crop` 和采样坐标逐图记录。旧 `rodent_brain192_260914` 批次采用全视野补边并包含低矩阵插值图，只用于历史对照。同一动物的切片、回波、纵向扫描及跨库记录不能当作独立动物；已知身份映射保存在 `subject_group`，当前不生成训练/验证/测试划分。

检查：安装 `pytest` 后运行 `../../.venv/bin/python -m pytest -q`。覆盖物理比例、补边、抗混叠、无效图像筛除、Bruker 字节序和每帧缩放、双回波次序、NIfTI 原始层编号及强度缩放。

## 非中央层面子集

`select_noncentral_slices.py` 从已有批次复制子集，不再次裁剪、缩放或改变灰度。默认按原始切片号除以原始总层数减一计算位置，选择栈两端各 25%，不是按已接受切片列表的位置计算；这也不等同于统一的解剖前后方向。

```bash
../../.venv/bin/python select_noncentral_slices.py \
  --run runs/rodent_brain192_native_crop_260914 \
  --out runs/rodent_brain192_native_crop_260914/noncentral4000_260914
../../.venv/bin/python build_gallery.py \
  --run runs/rodent_brain192_native_crop_260914/noncentral4000_260914
```

默认取 4,000 张，保留父批次示例编号 1442 和 10493；处理其他父批次时用 `--include-num` 指定其示例编号，传入空列表可不指定。仅以既有归一化后的 p99 ≥ 0.4、标准差 ≥ 0.08 排除本子集中较暗的图，不按脑区面积排除。余下按来源及栈端分层分配名额，保持来源覆盖，再按信号强度和示例位置排序。

子集 `images/` 保留父批次文件名和原 PNG 字节，`manifest.jsonl` 保留原编号，新增 `selection.subset_rank` 记录连续子集序号；编号不连续是预期行为。复制时逐张验证文件与像素 SHA256、位深、尺寸、唯一性和继承的原生尺寸约束，结果写入 `verification.json`。`not_selected.jsonl` 区分中央层面、较暗图和名额筛减，后两类不等同于原数据无效。全量批次的 `verify_export.py` 要求连续编号，不适用于此类保留原编号的子集。
