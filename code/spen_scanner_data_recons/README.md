# SPEN 原始采集全量重建

新版白色工作台（FastAPI + React）：见 [启动说明](web/README.md)。支持实验搜索、连续切片、多方法对比、图像放大及下载，直接读取现有运行结果。

从 [实际采集数据](../data/spen_acquired_260915/README.md) 中的 Bruker `rawdata.job0` / `fid` 直接读取信号，逐扫描、逐 slice、逐 volume、逐 echo 运行传统重建。原始数据保持不变，结果保存在本项目 `runs/`。

**[浏览全部层面](runs/all_raw_260915/index.html)** · [逐扫描和逐帧结果](runs/all_raw_260915/summary.json) · [独立核验](runs/all_raw_260915/verification.json) · [全帧 CSV](runs/all_raw_260915/frame_index.csv)

## 数据范围

输入根目录：`../data/spen_acquired_260915/`。本次遍历 `raw/` 下 **45 组采集、782 个 SPEN/xSPEN 成像扫描记录**。其中 **751 个有非空原始信号，全部成功解码，共 1,713 帧**；另外 31 个记录的原始信号为空或缺失，仍保留在清单中并说明原因。

没有挑代表层；每个可读扫描都完整遍历 `slice × volume × echo`。多接收线圈保留为独立复数通道，不把线圈当作 slice。波谱目录、其他辅助序列及已有 MAT 副本不重复计入成像帧数。MAT 只用于检查，不作为重建输入。

两处头信息冲突经过独立验证：

- `lxj_motionRARE_SPEN_230904.lG2/8` 的两个 method 计数重复表示同一组 13 个 volume；acqp、原始字节数和扫描仪帧数均证实实际为 **13**，不是相乘得到的 169。因此原盘点的 1,869 帧修正为 **1,713**。
- 8 个多 shot 扫描共 86 帧，声明图像矩阵为 96×96，但实际采集为 **96 RO × 95 PE**。`PVM_EncMatrix`、`PVM_EpiMatrix`、19 条线/shot × 5 shot 和原始字节数相互吻合。保存全部实采线，不补一条虚构测量线。

每个扫描的 `cases/<id>/case.json` 记录源路径、SHA256、原始维度、计数修正、轨迹来源、每帧状态及输出路径。完整采集清单保存在 [inventory.json](runs/all_raw_260915/inventory.json)。

本次结果：**1,475 帧完成两种方法，86 帧保留未做相位校正的重建，152 帧保留原始预览**。751 个非空扫描没有读取失败或丢失帧；另 31 个记录为 6 个空信号和 25 个缺失信号。

## 两种传统方法

共有前处理：原始字节解码 → 分 shot/echo 排序与反向读出校正 → 使用有来源记录的轨迹重采样 → RO FFT → Phase Map 相位校正。原始 int32 ADC 以 complex128 保留到重采样结束，之后按历史流程存为 complex64；批量 FGG 保持原标量算法的累加顺序，避免提前舍入改变相位优化结果。

1. **Phase Map + InvA**：单 shot 使用本地 `spenpy` 的 PV360 相位校正流程；PV5 使用其对应的轨迹分支和平滑设置。多 shot 使用存档 MATLAB 源码：3/5 shot 使用 NewWhole + OddNum，后续 echo 复用同 slice、同 volume 的首 echo 自动 mask；4 shot 使用原层级合并分支。`InvA` 是历史函数名，实际为 Gaussian 加权伴随矩阵，不是直接求矩阵逆。
2. **Tikhonov**：使用同一帧相位校正后信号 `Y` 和正向编码矩阵 `A=tmpAFinal`，对每个复数线圈求解 `min_X ||A X - Y||² + 0.01 × ||A||₂² × ||X||²`。采用 complex128 SVD，逐帧检查正规方程残差。

完整图集两种方法均采用 RSS 合并线圈。多 shot 原 MATLAB 分支的实际 Gaussian 宽度随帧记录：奇数 shot 为 0.8，4 shot 分支为 0.5；校正前后 InvA 使用同一实际算子。Tikhonov 使用返回的同一正向矩阵。保留原算法的相位拟合预算与回退诊断；计算出有限数组不代表相位优化已收敛或图像质量合格。

MATLAB 仅运行相位校正，不启用旧顶层脚本另加的经验幅度校正。源代码、依赖、逐帧输入/输出 MAT、mask、日志和 SHA256 保存在运行目录的 `matlab_multishot_260915/`，边界错误修复和重试另有记录。

## 结果状态与限制

- `completed`：本帧已有 Phase Map + InvA 和 Tikhonov 两种结果；仍需结合图像和质量警告判断采集质量。
- `partial_reconstruction`：保留未校正的 InvA 和对应 Tikhonov。实际 95 PE 的 5 shot 数据，每 shot 奇偶线数量为 10/9，原相位校正脚本不能直接处理；这部分不标成 Phase Map 已完成。
- `preview_only`：保留全部原始采样、RO FFT 和可读取的扫描仪预览。原因包括缺失/失效轨迹、缺少必要编码参数、尚未验证的 xSPEN 模型。
- `raw_unavailable`：源信号为空或缺失，无法从原始信号重建。

在有原始信号的扫描中，230911 采集的 13 个轨迹经原算法截除边缘后不剩有效读出支撑，已禁止将全零输出算作重建完成；另外 7 个轨迹的有效读出支撑很低，图集明确提示质量限制。诊断清单另外保留 4 个原始信号不可用记录的轨迹问题。没有从相邻扫描猜测轨迹，也没有把扫描仪中间文件冒充完整多线圈原始数据。

## 如何看图

打开首页选择一个 **exp（数据组）**，该组所有扫描的 **全部 slice、volume 和 echo** 会直接在同一页展开，往下滚动即可看完。扫描下拉框只用于跳到对应位置，其他扫描仍保留在页面中。默认打开含 11 层 scan 24 的 `20231207_150817_lxj_spen_231207_1_1_1` 数据组。

“查看内容”可切换整页的四图对比、两种重建对比，或单独查看 ADC、RO FFT、重建和扫描仪预览。每帧标明 scan、slice、volume、echo，点击图像可放大。图像随滚动加载，无需逐层选择或翻页。

所有帧的索引直接嵌入 HTML，本地打开即可使用，不需要服务器。地址栏保留数据组、视图及跳转位置，旧的单帧链接也可定位到对应帧。原来的逐组分页入口保存在 [分页图集](runs/all_raw_260915/archive.html)，作为补充查看方式；浏览器验收记录见 [交互检查](runs/all_raw_260915/viewer_browser_checks.json)。

默认四列为原始 ADC 的 log-RSS、RO FFT 的 RSS、Phase Map + InvA、Tikhonov。详情链接到 NPZ、原生尺寸 PNG 及旧图集；不具备完整方法的面板明确留空并说明原因。

各面板独立设窗：ADC 用 `log(1+RSS)` 的 p1–p99.5，其他图用 0–p99.5。显示窗口不改变 NPZ 数值，不能用显示亮度比较绝对信号。PNG 不作平滑或锐化。RO FFT 和重建预览翻转两个显示轴以沿用旧预览约定，这不定义解剖方向；ADC 与扫描仪图保留自己的方向。扫描仪预览不是配准后的真值。

NPZ 中各帧数组为 `[PE, RO, coil]`；完整 raw 解码数组在流程内保留 `[RO, PE, slice, volume, coil, echo]`。Tikhonov 正规方程残差只表示数值求解精度，不能当作图像质量分数。

## 运行与核验

在本项目目录使用已有环境 `/home/data2/chk/workspace/2026/.venv/bin/python`。该环境有 PyTorch；根目录另一个 `.venv` 没有。CPU 即可处理，不占用 diffusion 训练 GPU。

只更新交互页面、复用已有图像时运行下面一条命令即可，不需要重新重建或生成 PNG：

```bash
/home/data2/chk/workspace/2026/.venv/bin/python scripts/build_viewer.py \
  --run runs/all_raw_260915
```

从原始数据完整运行：

```bash
/home/data2/chk/workspace/2026/.venv/bin/python scripts/inventory_all.py \
  --out runs/inventory_new.json

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/data2/chk/workspace/2026/.venv/bin/python scripts/run_collection.py \
  --inventory runs/inventory_new.json --out runs/all_raw_new --workers 16

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/data2/chk/workspace/2026/.venv/bin/python scripts/run_matlab_multishot_collection.py \
  --run runs/all_raw_new --workers 8

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/data2/chk/workspace/2026/.venv/bin/python scripts/retry_matlab_multishot_bounds.py \
  --run runs/all_raw_new --finite-support

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/data2/chk/workspace/2026/.venv/bin/python scripts/finalize_matlab_multishot_collection.py \
  --run runs/all_raw_new

/home/data2/chk/workspace/2026/.venv/bin/python scripts/trajectory_quality.py \
  --run runs/all_raw_new

/home/data2/chk/workspace/2026/.venv/bin/python scripts/render_collection.py \
  --run runs/all_raw_new

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/data2/chk/workspace/2026/.venv/bin/python scripts/verify_collection.py \
  --run-dir runs/all_raw_new
```

MATLAB 本机路径 `/usr/local/MATLAB/R2024a/bin/matlab`；新运行需要原库及存档 `raw_spectroscopy/.../SPENReco` 依赖。首次输出用新目录；Python 中断恢复可加 `--resume`，默认重试失败或未完成扫描。`--retry-status` 可指定需重算的状态，`--only-ids-file` 可在保持完整清单的前提下仅重算指定扫描。不要在 MATLAB 正在修改同一扫描时重跑它。

独立核验从每个 `case.json` 重新读取数组：检查完整帧索引、原始字节数、维度、有限值、InvA 方程、Tikhonov 正规方程，以及 **501 个旧 MAT 中的 811 帧**。MAT 的三项复数相对 L2 阈值固定为 `1e-6`；失败不会被改成通过。未被当前清单引用的旧尝试文件不计入结果。

本次最终核验全部通过：1,713 帧无遗漏，811 帧 MAT 回归全部通过（最大相对误差 `1.72e-11`）；InvA 方程最大相对误差 `3.87e-14`，1,561 帧 Tikhonov 的正规方程残差最大 `4.05e-15`。117 项测试通过；图集覆盖全部帧，173 个 HTML 页的 23,130 个本地链接均有效。87 个扫描包含多 slice，其中 38 个有 10 层、1 个有 11 层。

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/data2/chk/workspace/2026/.venv/bin/python -m pytest -q scripts/test_*.py
```

先前六扫描、11 帧的小规模验证仍在 [raw_pilot_260915](runs/raw_pilot_260915/index.html)，包含三个 Tikhonov 强度及 adaptive 合并补充图；全量入口使用固定 0.01 和 RSS。早期 pilot 经过公共 API 的 complex64 原始数据封装，正式全量入口已修正为保留 ADC 精度，核验基准采用原始标量流程和存档 MAT。
