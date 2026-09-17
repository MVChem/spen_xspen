# SPEN diffusion reconstruction

从 `2026/08/14/spen_diffusion_recons` 的轻量源码建立的独立工作副本，包含 rat96 EDM、mouse96 strong prior 和 192 网格超分辨率重建。原入口多数是指向旧项目的软链接；这里已复制实际内容，并修复代码导入、输入路径和默认输出位置。

## 目录

| 位置 | 用途 |
| --- | --- |
| `scripts/core/` | rat96 EDM、SPEN 算子、DiffPIR / DAPS、训练及评估 |
| `scripts/prior96/` | mouse96 strong prior、数据准备、训练锁及评估 |
| `scripts/prior192/` | 192 网格数据准备、训练和 SPEN 超分辨率重建 |
| `vendor/InverseBench/` | 算子基类及 DAPS 所需的最小上游源码和许可证 |
| `runs/` | 新产生的权重、运行日志、评估结果和数据准备输出，运行时创建 |

没有复制数据、权重、实验记录、缓存、虚拟环境或 Git 历史。

## 96 网格训练与对比图

- `scripts/prior96/train_pipeline.py`：单 GPU 两阶段训练，支持断点继续；完成后运行仿真评估。
- `scripts/prior96/train_strong.py`：实际训练循环，支持梯度累积。
- `scripts/prior96/evaluate_reconstruction.py`：指定 `--mode simulation` 或 `--mode real`，使用给定权重重建选定案例；通过 `--inputs` 指定数据与来源记录，输出到 `--out`。
- `scripts/prior96/render_comparison.py`：读取评估目录内的 NPZ 和指标，重新导出紧凑布局的 PNG/PDF，无需 GPU。

仿真图依次为 GT、输入、Tikhonov、Phase map + InvA、Diffusion；真实数据图省略 GT。Phase map + InvA 固定在倒数第二行，Diffusion 在最后一行。真实数据没有配对真值，不标注 PSNR/SSIM。

在本目录、使用已安装项目依赖的 Python 环境重绘：

```bash
python scripts/prior96/render_comparison.py --run 'runs/<运行名>/evaluation' --kind simulation
python scripts/prior96/render_comparison.py --run 'runs/<运行名>/evaluation_real' --kind real
```

仿真默认展示第 `3 8 17 15` 号数组病例（从零计数），完整 PE 和随机 50% PE 各四列；前三例沿用原图，新增的第 15 号来自 ds002868 的另一只留出小鼠。用 `--case-ids` 指定任意数量的病例，图宽与分组位置随列数调整。实采重绘默认展示数组中的全部病例，也可用 `--real-case-ids` 选取数组索引。两种重绘均输出 PNG/PDF，以及记录图中病例顺序的 `simulation_selection.json` / `real_selection.json`；仿真记录包含来源和逐例指标。

```bash
python scripts/prior96/render_comparison.py --run 'runs/<运行名>/evaluation' --kind simulation \
  --case-ids 3 8 17 15 --out 'runs/<运行名>/redraw'
python scripts/prior96/render_comparison.py --run 'runs/<运行名>/evaluation_real' --kind real \
  --real-case-ids 0 2 4 5 7 9 --out 'runs/<运行名>/redraw_subset'
```

实采推理默认使用 FOV 16 mm 的 `5 13 22 30 38` 和 FOV 24 mm 的 `3 7 11 15 19`，与 VAE + DiT 对照图的十例编号一致。`--real16-ids` / `--real24-ids` 接收 MAT 文件的导出编号（不是数组索引）；可传 `13 22 30` / `7 11 15` 重建原六例。新增病例需要先推理，再用绘图脚本展示；重绘不会自动生成缺失的重建数组。

```bash
PYTHONPATH=../spenpy python scripts/prior96/evaluate_reconstruction.py --mode real \
  --checkpoint 'runs/<运行名>/mouse_mixed/model_ema.pt' --inputs ../data/prior96_0911_260916 \
  --reuse-real 'runs/<运行名>/evaluation_real' \
  --real16-ids 5 13 22 30 38 --real24-ids 3 7 11 15 19 \
  --out 'runs/<运行名>/evaluation_real_expanded'
```

这里通过 `PYTHONPATH` 使用同级 spenpy 源码；`--out` 应为新目录。`--reuse-real` 会核对已保存结果的权重、重建参数、MAT 哈希和数组对应关系，复用已有病例，只推理新增病例；没有旧评估目录时省略此参数。各病例继续采用原生 96×96 网格、固定显示窗和原重建参数。

## 已确认的重建对照图

[SPEN_Reconstruction_Comparison_260915](../../experiments/SPEN_Reconstruction_Comparison_260915/) 保存仿真及真实采集的两张 PNG，包含实验参数、指标约定和本地数据位置说明。

- `scripts/prior192/rebuild_figures.py`：使用最终 192 网格先验完成仿真与真实重建。
- `scripts/prior192/render_rebuilt.py`：从保存的 NPZ 重新绘制两张 PNG，无需重新推理。
- `scripts/prior192/phase_inva.py`：相位拟合和传统加权 InvA 基线。

默认运行目录为 `runs/rodent192_spen2x_260914/figures_260915/`。权重、原始数据及 NPZ 保留在本地，Git 仅收录确认图、说明与源码。

## 环境与入口

项目根 `.venv` 链接到 `/home/data2/chk/workspace/2026/.venv`。本目录的 [requirements.txt](requirements.txt) 包含从 GitHub `MVChem/spenpy` 安装的 `v1.0.0`，运行时使用 pip 安装的包。旧算法明确使用 `spenpy._legacy.core` 的 `calcInvA` 和 `calcSRMatrixApprox`，保留原数学实现，未改为新协议算子。

以下命令从本目录执行，只显示帮助，不训练或读取数据：

```bash
../../.venv/bin/python scripts/core/train.py --help
../../.venv/bin/python scripts/core/evaluate.py --help
../../.venv/bin/python scripts/prior96/train_strong.py --help
../../.venv/bin/python scripts/prior96/evaluate_mouse.py --help
../../.venv/bin/python scripts/prior192/train_hr.py --help
../../.venv/bin/python scripts/prior192/evaluate_sr.py --help
../../.venv/bin/python scripts/prior192/evaluate_real_sr.py --help
```

`core/evaluate.py` 和 `prior96/evaluate_mouse.py` 要求显式传入 `--checkpoint` 与 `--out`；建议将输出设为本项目 `runs/` 下的新目录。其余训练与评估的默认输出由源码位置确定，均在本项目 `runs/` 内，与启动工作目录无关。

## 输入资料

输入路径集中在 [project_paths.py](scripts/core/project_paths.py)。默认指向本项目尚未创建的 `data/`，不会自动读旧目录。数据准备和真实重建需要另行配置，下列环境变量只控制输入：

| 环境变量 | 含义 |
| --- | --- |
| `SPEN_DATA_ROOT` | 本项目数据输入根目录 |
| `SPEN_REFERENCE_ROOT` | 旧 `spen_recons` 根目录，供 rat 图像、扫描 MAT 和相位缓存读取 |
| `SPEN_RAT_SPLIT` | rat96 按个体划分的 `split.json` |
| `SPEN_PRIOR96_DATA` | mouse96 数组和 manifest 目录，可被 `--data` 覆盖 |
| `SPEN_PRIOR192_DATA` | 192 数组和 manifest 目录，可被 `--data` 覆盖 |
| `SPEN_MOUSE_RAW` | 原始 mouse 数据及 download manifest 目录 |

例如，确实需要读取旧 rat / scanner 资料时设置：

```bash
export SPEN_REFERENCE_ROOT=/home/data2/chk/workspace/2026/08/14/01_工作项目/spen_recons
```

训练与评估的检查点通过 `--checkpoint`、`--init-from`、`--resume` 或 `--reference-checkpoint` 显式选择。mouse96 新训练继承了与 rat96 prior 的基线比较，因此还需要 `--reference-checkpoint`。数据准备生成到 `runs/prepared/`；继续训练前将相应目录传入 `--data` 或设置输入环境变量。准备脚本沿用原数据流程，本轮未下载或重建数据。

## 检查

```bash
../../.venv/bin/python -m pytest -q -p no:cacheprovider
```

测试检查实复数伴随、梯度、稠密参考 proximal 解、EDM 接口和训练锁。外部 scanner 参考未配置时，对应集成测试跳过。和其他两个项目分进程运行测试，避免历史脚本中 `model`、`data`、`operators` 等同名模块冲突。

本轮验证了 CPU 测试和九个训练、评估、数据准备入口的 `--help`，未运行训练、完整评估或下载。
