# 代码

| 项目 | 职责与入口 |
|---|---|
| [data](data/README.md) | 共享数据入口、原始文件与来源记录；本机数据使用符号链接 |
| [spen_diffusion_recons](spen_diffusion_recons/README.md) | SPEN 的 96/192 先验训练、数据制备与重建；旧一次性脚本单独作为参考保留 |
| [xspen_diffusion_recons](xspen_diffusion_recons/README.md) | xSPEN diffusion 训练与评价、原生分辨率、超分和真实数据处理流程 |
| [spenpy](spenpy/README.md) | SPEN / xSPEN 物理模型、仿真、重建和数据读取基础库，当前版本 1.0.0 |

三个目录均为实际源码副本，由根 Git 统一维护。`spenpy` 保持标准 Python 包结构；两套 diffusion 项目暂时保留各自的脚本入口，以便对照原实现。两项目存在 `model.py`、`data.py` 等同名模块，测试和脚本应在各自进程中运行，不要把它们一起加入同一个 `PYTHONPATH`。

环境统一放在项目根 `.venv/`；依赖和运行入口见各项目的 README。输入数据和权重尚未迁入，常用 CLI 的 `--help` 可先检查参数；运行真实训练或重建前需提供实际输入路径，并选择新的输出目录。

后续代码项目各自放在 `code/<项目>/`。每个项目可以有自己的 `runs/`，用于保存实验记录和运行产物；`runs/` 由 Git 忽略。用户认为重要的结果会自行放入根目录的 `experiments/<主题>_YYMMDD/`，例如 `experiments/重建对比_260914/`。当前进展、临时笔记和展示材料先放根目录 `tmp/`，完整约定见 [AGENTS.md](../AGENTS.md)。
