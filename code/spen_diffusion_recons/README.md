# SPEN diffusion reconstruction

从 `2026/08/14/spen_diffusion_recons` 的轻量源码建立的独立工作副本，包含 rat96 EDM、mouse96 strong prior 和 192 网格超分辨率重建。原入口多数是指向旧项目的软链接；这里已复制实际内容，并修复代码导入、输入路径和默认输出位置。

## 目录

| 位置 | 用途 |
| --- | --- |
| `scripts/core/` | rat96 EDM、SPEN 算子、DiffPIR / DAPS、训练及评估 |
| `scripts/prior96/` | mouse96 strong prior、数据准备、训练锁及评估 |
| `scripts/prior192/` | 192 网格数据准备、训练和 SPEN 超分辨率重建 |
| `vendor/InverseBench/` | 算子基类及 DAPS 所需的最小上游源码和许可证 |
| `archive/reference/` | 旧批处理、制图和下载源码，仅供查阅 |
| `runs/` | 新产生的权重、运行日志、评估结果和数据准备输出，运行时创建 |

没有复制数据、权重、实验记录、缓存、虚拟环境或 Git 历史。

## 环境与入口

使用新项目根目录的统一 `.venv`。本目录的 [requirements.txt](requirements.txt) 只声明第三方依赖；还需要安装同级 [spenpy 源码](../spenpy/)。旧算法明确使用 `spenpy._legacy.core` 的 `calcInvA` 和 `calcSRMatrixApprox`，保留原数学实现，未改为新协议算子。

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

本轮验证了 CPU 测试和九个训练、评估、数据准备入口的 `--help`，未运行训练、完整评估或下载。`archive/reference/` 中保留原文的 `.py.reference` / `.m.reference` 不属于可执行入口；使用前须处理原实验目录依赖。
