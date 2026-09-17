# SPEN / xSPEN 数据查看

[打开统一 HTML](runs/unified_review_260916/index.html) · [旧导出图](runs/unified_review_260916/legacy_figures.html) · [覆盖说明](runs/unified_review_260916/coverage.html)

当前页面有 **2,049 个影像条目，另保留 10 个选例和 91 张旧导出图**。此前的 24 份扫描并不是整个旧目录；这次补查了核心资料目录、压缩包及旧 Bruker 实测结果。条目可能是一个数组、一个序列或一个处理结果，不能据此推算独立受试者数量。

白底界面支持分类、来源搜索、分页、全部切片及附加轴、连续浏览和点击放大。直接在浏览器打开 `index.html`，无需服务器或网络；移动或分享时保留整个输出文件夹。大型数组分块加载，列表每页 48 项，连续浏览每页 64 帧。

## 实际范围

| 分类 | 数量 | 内容 |
|---|---:|---|
| 已核验 xSPEN 对照 | 8 份 / 6,812 帧 | RO-only、旧流程复算、Tikhonov、PhaseMap + InvA |
| 已核验 Hybrid SPEN | 16 份 / 1,898 帧 | 13 份有旧重建，3 份为 RO-only 输入；MID253 可逐帧前后对照 |
| 选例 | 10 个 | 保留此前选定观测及完整 ADC / 复数数据，不重复计入扫描帧数 |
| MAT / FIG 数值数组 | 593 个条目 | 可读二维数值数组、复数幅度，以及全部附加维度 |
| NIfTI | 586 个条目 | 原始体素网格、全部切片和附加维度，保留有符号数值 |
| DICOM | 45 个序列 | 应用原 slope/intercept；保留序列帧，未自动拆分 mosaic |
| Siemens 原始 ADC | 113 个数组组 | 全部有效采样的 coil RSS 幅度；保留过采样和原记录极性 |
| Bruker SPEN | 543 份 / 921 帧 | 已有输入、传统重建、Tikhonov、Diffusion 四阶段 |
| 其他参考序列 | 145 份 / 939 帧 | 144 份 Bruker EPI / FLASH / RARE，以及 1 份 FLASH 工具示例 |
| 旧导出图 | 91 张 | 原有彩色图、拼图、标注和 EPS 栅格预览，单独入口 |

核心来源是 `/home/data2/chk/workspace/2026/08/14/xSPEN_项目`：2,795 个逐文件记录中，2,657 个有显示条目，95 个按 SHA256 合并重复副本，34 个是非影像 DAT，9 个不含符合图像维度要求的数值数组；没有读取失败。135 个外层归档及 49 个递归容器均已审计。逐变量跳过原因、重复来源和压缩包记录可在[覆盖说明](runs/unified_review_260916/coverage.html)查看。

额外的 Bruker 数据来自旧 `spen_diffusion_recons/results/real_batch` 与 `spen_recons/data/scanner`。272 份 SPEN 采集的扫描名称标注小鼠，其余 271 份物种未核实；部分旧设备字段与实验背景有冲突，不能当作人脑数据。13 个 study 也不等于 13 个独立生物个体。另发现 5 个 SPEN 和 2 个参考扫描为空文件，没有虚构图像。

逐线圈 FID 文本、501 份旧 Bruker 缓存、外部 16 个标准化 H5 均已映射到对应来源，不重复算新采集。IXI 等训练来源、模拟训练缓存和软件图标未混入实测扫描。MAT / FIG 的标量、向量、小参数矩阵、符号对象单独记录；通用浏览要求至少两个维度达到 16，FIG 不复刻 MATLAB 绘图布局和色标。核心目录没有遗漏的 H5 / HDF5 / NPY / NPZ 文件。

## 读图说明

- 默认每帧独立窗；正值图使用 0–p99.5，有符号图使用 p0.5–p99.5。NaN / Inf 在主图中标灰，下载数组保留原值。窗宽与 Gamma 只影响显示。
- 数值数组按声明的原始轴显示，不猜测未知轴的解剖、线圈或扩散含义。明确物理 FOV 的条目按 FOV 比例呈现，其余按像素比例。
- “原始 ADC”是采样域幅度，尚未做 RO FFT、regrid 或 PE 重建；“RO-only 输入”已经经过读出方向处理。不能把两者都称为未经处理的原始图像。
- 新增通用条目按文件与变量独立浏览，未把未核验的来源自动配成重建前后对照。不同方法的数值尺度可能不同，共用窗可能使某些结果变暗。
- Bruker 的四阶段均读取旧结果，本轮未重新推理。输入与传统结果显示复数幅度，完整复数另外保存；Tikhonov / Diffusion 保留负值和大于 1 的值，未沿用旧 PNG 的截断或双轴翻转。
- 所有原始来源目录只读；输出写在本项目 `runs/`，查看入口为工作区 `tmp/spen统一可视化_260916/`。

已核验 xSPEN 是 MID112、MID114、MID27、MID51、MID613、MID615、MID106、MID108。MID27 的对照使用旧 H5 保留的 46 层 × 18 次；原始 counter 4、28 因采集不完整被旧导出排除。此次 ADC 分类另读取了该 DAT 的全部实际记录，不能把对照中的排除当作原始数据已丢失。

Hybrid 的具体序列为 `esrs_hyb_spen_Diff2`，使用二次相位编码，与 crossed-chirp xSPEN 算子不同。[MID253](runs/unified_review_260916/index.html#MID253) 对应用户指定的旧图：40 层 × 4 volume，默认 MATLAB 第 20 层，标签沿用 b0、DWI-RO、DWI-PE/SPEN、DWI-SS。MID1496 头部计划 10 层、10 次，实际只记录 6 层、1 个 Rep；GoogleDrive 50slice 的具体解剖部位未核实。

旧 xSPEN 的 `Img` 是 RO 滤波 / FFT / RSS，不能作为 PE 逆编码真值。10 个选例中 5 个与旧 Img 配对通过，其余保留同流程复算。Hybrid 复用已有 PE 重建；xSPEN 对照的 6,812 帧使用原有 `review.reconstruct`。来源和选例见 [selection.json](selection.json)。

## 生成与检查

使用 `/home/data2/chk/workspace/2026/.venv/bin/python`。以下命令在 `code/xspen_human_review/` 中运行，复用当前 `runs/` 内的已核验对照与审计清单。

```bash
# 重新盘点核心目录，复用成功导出，重试失败文件。
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/data2/chk/workspace/2026/.venv/bin/python catalog_all.py \
  --run runs/unified_review_260916 --workers 3 --resume --rescan

# 读取已有 Bruker 结果和参考扫描，不重新重建。
/home/data2/chk/workspace/2026/.venv/bin/python import_bruker.py \
  --run runs/unified_review_260916
/home/data2/chk/workspace/2026/.venv/bin/python import_bruker_reference.py \
  --run runs/unified_review_260916

# 合并完整目录，更新页面、旧导出图、覆盖说明和 selection。
/home/data2/chk/workspace/2026/.venv/bin/python finalize_catalog.py \
  --run runs/unified_review_260916

# 全部文件引用/数组头检查，以及代表条目和原34入口的离线浏览器检查。
/home/data2/chk/workspace/2026/.venv/bin/python verify_archive_catalog.py \
  --run runs/unified_review_260916 --all-old \
  --output runs/unified_review_260916/inventory/archive_browser_verification.json
```

成功导出缓存仅用于相同输入及处理代码。算法改变时应使用新输出目录；旧 Bruker 导入清单、参考清单和 FLASH 提取来源保存在 `inventory/` 与 `archive_sources/`。仅改主界面时，可运行 `build_unified.py --out runs/unified_review_260916 --refresh-view`。

`build_unified.py` 与 `expand_scans.py` 保留最初 24 份扫描对照的构建流程；`catalog_all.py` 导入核心资料；两个 Bruker 脚本负责额外真实扫描；`finalize_catalog.py` 合并展示。页面源码在 `web/`，全部产物集中在唯一的 `runs/unified_review_260916/`。

验证包括全部条目的引用文件与 NPZ 形状、显示量化误差、MAT / NIfTI / ADC / DICOM 来源抽查、全部 Bruker 源数组对应、原 10 个选例与完整扫描一致性，以及离线浏览、灰度、放大、分页和移动端。记录位于 `inventory/archive_browser_verification.json`、`inventory/bruker_verification.json`、`inventory/bruker_reference_entries.json` 和原有 `expanded_scans_verification.json`。归档数据中的 NaN / Inf 是保留的源值，不作为假定有限数组处理。
