# xSPEN Diffusion Reconstruction

xSPEN 编码算子、EDM prior、DiffPIR 重建，以及原生分辨率和 2× 输出网格的研究代码。此目录从旧项目按源码迁移，保留协议配置和现有测试；没有复制数据、权重、结果、实验状态、环境或冻结源码快照。

## 代码入口

| 路径 | 用途 |
| --- | --- |
| `operators.py`、`scanner.py`、`solvers.py` | xSPEN 物理算子、扫描仪适配、DiffPIR |
| `model.py`、`tiny_unet.py`、`edm.py` | 独立的 128×128 EDM prior |
| `prepare_data.py`、`prepare_scanner.py` | IXI 数据准备、Siemens 原始数据导出 |
| `train.py`、`evaluate.py`、`pipeline.py` | 基础训练、评估及串行流水线 |
| `pipelines/native_resolution/` | 原生物理网格的数据准备、prior、训练与评估 |
| `pipelines/native_resolution/superres2x/` | 2× 输出网格重建及检查 |
| `pipelines/native_resolution/expanded_human/` | 扩展人脑扫描选择、导出和重建 |
| `pipelines/native_resolution/real_comparison/` | 传统相位校正与重建对照 |
| `pipelines/native_resolution/audit/`、`data_inventory/` | 保留的数据审计工具，使用前需准备输入清单 |

## 环境和检查

使用项目根目录的共享 `.venv`，第三方依赖见本目录的 [requirements.txt](requirements.txt)，并安装同级 [spenpy 源码](../spenpy/)。从当前目录执行：

```bash
../../.venv/bin/python train.py --help
../../.venv/bin/python pipelines/native_resolution/train_native.py --help
../../.venv/bin/python -m pytest -q -p no:cacheprovider
```

SPEN 与 xSPEN 当前仍有 `model.py`、`utils.py` 等同名顶层模块；应在各自目录启动独立 Python / pytest 进程。传统重建使用相邻的 `../spenpy` 源码，不依赖旧目录的 Python 包。当前未做包名重构。

## 输入与输出

数据本次暂不整理。旧数据可通过参数显式读取，例如：

```bash
../../.venv/bin/python prepare_data.py --source /path/to/ixi --out data/ixi128
../../.venv/bin/python prepare_scanner.py --source /path/to/siemens_raw --out scanner
../../.venv/bin/python evaluate.py --checkpoint /path/to/model_ema.pt \
  --data /path/to/prepared_ixi128 --scanner /path/to/scanner_exports \
  --out runs/new_evaluation
```

也可设置以下环境变量；它们只配置输入位置，不改变输出位置：

| 环境变量 | 默认输入位置 | 用途 |
| --- | --- | --- |
| `XSPEN_IXI_SOURCE` | `data/raw/ixi` | IXI 原始 NIfTI |
| `XSPEN_IXI_DATA` | `data/ixi128` | 基础训练/评估的已准备数据 |
| `XSPEN_RAW_SCANNER` | `data/raw/siemens` | Siemens `.dat` 原始数据 |
| `XSPEN_SCANNER_DATA` | `scanner` | 基础/原生评估的 HDF5 与 QC 清单 |
| `XSPEN_LEGACY_ROOT` | `data/legacy` | 可选：旧 MATLAB 来源和历史审计输入的根目录 |

默认路径相对于此代码目录解析。训练和重建的默认产物位于新目录的 `runs/` 或对应 pipeline 内；带 `--out` 的入口可显式指定其他位置。GPU 流水线必须用 `--gpus` 指定设备，旧机器的 GPU UUID 与 tmux 会话已移除。

原生网格准备支持 `--source-manifest`；原生训练/评估支持 `--data`、`--out`，训练可用 `--init-from` 指定基础 prior。扩展人脑重建支持 `--selection`、`--baseline`、`--out`。2× 和报告/审计工具保留既有输入清单结构，依赖相应的数据、权重和选择清单；这些产物没有迁移，不能直接复现整套历史实验。`campaign.py` 仍要求本轮 `startup_verified.json` 预检记录，不应拿旧状态充当本轮验证。

## 迁移与验证

来源：[旧项目](/home/data2/chk/workspace/2026/08/14/01_工作项目/xspen_diffusion_recons/README.md)。

迁移后已通过原有 30 项 CPU 测试（物理算子、原生网格、传统相位和缓存完整性），并检查常用 CLI 的 `--help`。未启动训练、重建或写入旧数据。现有测试通过不代表新环境、真实数据全流程或成像效果已经验证。
