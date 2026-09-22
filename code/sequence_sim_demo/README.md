# 序列层面仿真 demo

用 **PyPulseq 1.5.0.post1 → Pulseq `.seq` → komamripy 0.0.11 / KomaMRI 0.14.0** 跑四组物理演示。
Python 负责构造波形、样品和重建；KomaMRI 按 RF、梯度和采样时刻计算 Bloch 演化。

## 运行

在本目录执行：

```bash
uv sync --locked
uv run python demo_epi.py --output runs/epi_260917
uv run python demo_spen_chirp.py --output runs/spen_chirp_260917
uv run python demo_brain_epi.py --output runs/brain_epi_260917
uv run python demo_brain_spen_xspen.py --output runs/brain_spen_xspen_260917
uv run python render_spen_xspen.py --run runs/brain_spen_xspen_260917
```

环境独立保存在本项目的 `.venv/` 和 `.julia/`。首次访问仿真后端会下载 Julia 并安装、预编译依赖。
默认用 8 个 CPU 线程；本次 demo 使用 CPU。可通过 `JULIA_NUM_THREADS` 改线程数。
EPI 的 `--gpu` 是可选入口，本次没有验证 GPU 路径。Python 依赖版本由 `uv.lock` 固定；
komamripy 自带的 `juliapkg.json` 固定 KomaMRI 及其组件版本。

## Demo 1：EPI 原始信号与 B0 畸变

[demo_epi.py](demo_epi.py) 构造一次 64×64 单次激发 GE-EPI：

- FOV 220 mm；非选层 90° RF，时长 0.2 ms。
- 双极性读出梯度，平顶 ADC；dwell 10 μs，echo spacing 0.94 ms，TE 31.62 ms。
- 256×256 的程序生成椭圆样品，非零样品点作为 35,324 个独立自旋；每个重建像素最多 4×4 个采样点。
- T1 = 1 s、T2 = 100 ms；单一均匀接收线圈；无噪声。
- 三种条件：零 B0、均匀 +49.867 Hz、空间变化的 B0 频率图。

均匀 B0 的理论 PE 位移为 `Δf × Ny × echo_spacing = +3 pixels`。
空间变化的 B0 直接进入 Bloch 方程；图像上的变形由采样信号和重建产生。
所有条件用相同的重建、相同的显示强度范围，不做畸变校正。
非零 B0 还会在每条读出内部积累相位，因此均匀 B0 结果也不与简单平移逐像素完全相同。

输出：

- [comparison.png](runs/epi_260917/comparison.png)：样品、B0 和重建对比。
- [sequence.png](runs/epi_260917/sequence.png)：RF、Gx/Gy/Gz 和 ADC 时序。
- [trajectory.png](runs/epi_260917/trajectory.png)：按采样时间着色的轨迹。
- [signals.png](runs/epi_260917/signals.png)：复数 ADC 信号的幅度与实部。
- `epi.seq`、`data.npz`、`metadata.json`、`validation.json`：序列、数组、条件及验证。

`data.npz` 中的图像轴为 `[y, x]`，信号按 ADC 时间顺序存储。
重建使用实际采样坐标排列双极读出，不依赖盲目翻转奇偶行。
RF/梯度在 Pulseq 中使用 Hz、Hz/m；B0 输入使用 Hz，进入 KomaMRI 时转换为 rad/s。

运行时验证：

1. PyPulseq 检查序列 timing；ADC 必须一一覆盖目标 Cartesian 网格。
2. 零 B0 原始复数信号与独立的 Fourier + T2 衰减参考比较，相对误差须小于 0.5%。
   参考使用同一离散样品但不调用 KomaMRI；只拟合一个全局复数增益以消除接收相位和整体增益约定。
3. 均匀 B0 下测得的位移必须为 +3 个 y 像素。

这里没有选层、扩散、运动、涡流或真实扫描仪误差模型。B0 图为人为指定，并非磁化率求解结果。
T2 和空间分辨的 B0 演化显式建模，未另加经验 T2* 衰减。

## Demo 2：SPEN chirp 产生的空间二次相位

[demo_spen_chirp.py](demo_spen_chirp.py) 使用 16 ms、16 kHz 的线性扫频 WURST 脉冲，
时间带宽积 R = 256，RF 峰值 1000 Hz（约 23.5 μT），同时施加 y 编码梯度。
通过 `no_signal_scaling=True` 保留实际 RF 幅度，不用复数 RF 波形积分将其重新缩放为“180°”。

计算两个实验：

1. 初始 Mz = 1，检查 chirp 的实际反转效果。
2. 先施加有限时长 90° RF，再施加 chirp，观察最终 Mxy 的空间幅度与相位。

这是 **一维 SPEN 编码模块**；没有 ADC，没有完整 SPEN 图像，也没有 crossed-chirp xSPEN。
自旋数为 1201；T1 = T2 = 10⁶ s，以突出编码相位。拟合区域取扫频定义 FOV 的中央 70%，
避免把边缘反转过渡区纳入二次相位拟合。

理想扫频反转的二次项系数幅度为
`|a| = 2π × T_rf × BW / FOV²`。
有限幅度、有限时长的 RF 会对这一近似产生偏离，因此同时记录拟合残差与系数偏差。
RF 波形栅格为 0.5 μs；分别用 0.25 μs 和 0.125 μs 的仿真步长检查数值收敛。

输出：[chirp_encoding.png](runs/spen_chirp_260917/chirp_encoding.png)、两个 `.seq`、
`data.npz`、`metadata.json`、`validation.json`。
运行时检查反转、二次相位近似以及 RF 时间步减半后的复数磁化变化。

## Demo 3：真实脑部结构像作为样品的 EPI 仿真

[demo_brain_epi.py](demo_brain_epi.py) 读取本机已有的 `IXI020-Guys-0700-T2.nii.gz`，
选取 canonical 后第 40、58、78 层（从 0 编号），分别模拟三种 B0 条件。
直接读取原始 NIfTI，与其他重建项目的预测图、测量信号无关。

- 保留原始近轴位层面，canonical 操作只调换/翻转轴，没有将倾斜层面重新插值为标准轴位。
- 原始 256×256，面内间距约 0.9375 mm，模拟 FOV 约 240 mm。
- 重建 96×96；以 288×288 网格离散自旋，用原始物理坐标线性插值。
  仿真网格细化用于计算磁化演化，不表示输入获得了更高的实测分辨率。
- EPI echo spacing 1.26 ms，TE 62.18 ms；T1 = 1 s、T2 = 100 ms，均为假设的均匀参数。
- 均匀 B0 约 +24.80 Hz，对应 +3 个 PE 像素；空间 B0 使用明确给定的平滑函数，
  各层采用同一张图以便比较。所有 EPI 图共用显示强度范围。

**输入是 T2 加权结构像，用其归一化强度作为有效 M0 空间模板，不是定量质子密度图。**
原始图像的组织对比仍留在模板中，再叠加本次设定的均匀 T2 衰减。
此演示用于观察真实脑结构经过序列采样后如何成像、畸变，不声称预测该受试者的真实 EPI 对比。
没有从结构像反推组织 T1/T2 或 B0，也没有脑组织分割。

输出位于 [runs/brain_epi_260917/](runs/brain_epi_260917/)：

- [brain_comparison.png](runs/brain_epi_260917/brain_comparison.png)：三个层面的输入、无偏移 EPI、
  均匀偏移 EPI、空间 B0 畸变及差值。
- [b0_map.png](runs/brain_epi_260917/b0_map.png)：人为给定的 B0 和局部 PE 位移近似。
- `brain_epi.seq`、`sequence.png`、`trajectory.png`：实际模拟的序列与轨迹。
- `slice_040.npz`、`slice_058.npz`、`slice_078.npz`：有效 M0、自旋坐标、B0、原始复数 ADC、
  k-space、重建图和独立 Fourier 参考；数组图像方向为 `[y, x]`。
- `metadata.json`：原始文件 SHA256、原始/canonical affine、层号、世界坐标、重采样方式、
  软件版本、运行命令、仿真参数与源码哈希。
- `source/`：本次执行的脚本及 Python 依赖锁文件快照。
- `validation.json`：每一层独立核验无 B0 信号与 Fourier + T2 参考的误差，以及均匀 B0 的 3 像素位移。

2026-09-17 实际运行中，三个层面的无 B0 复数信号相对误差为 0.00246%、0.00273%、0.00274%，
均匀 B0 实测位移均为 +3 像素，验证通过。

换其他本地非负结构像时：

```bash
uv run python demo_brain_epi.py \
  --input /path/to/brain.nii.gz --slices 40 58 78 \
  --matrix 96 --oversampling 3 --t1 1.0 --t2 0.10 \
  --output runs/brain_epi_another_run
```

输出必须是本项目 `runs/` 下的新目录，以保留已有结果。
输入形状需为单个三维体积；默认序列使用人体梯度限制，小 FOV 鼠脑应另行调整序列采样与梯度参数。
本 demo 是 EPI；完整 SPEN / crossed-chirp xSPEN 成像见下一节。

## Demo 4：沿本地源码实现 SPEN180 / crossed-chirp xSPEN 脑部成像

[spen_waveforms.py](spen_waveforms.py) 生成完整 RF、梯度和 ADC 时序；
[demo_brain_spen_xspen.py](demo_brain_spen_xspen.py) 计算磁化、采样信号并重建脑图。
结果：[SPEN / xSPEN 对比图](runs/brain_spen_xspen_260917/brain_spen_xspen_comparison.png)。

优先参考了本机旧项目的四个文件，实际来源路径与 SHA256 在运行目录 `metadata.json`，
源码副本保存在 `source/`：

| 本地来源 | 本次沿用的内容 |
|---|---|
| `xSPEN1D.m` | `Tp=Ta/(4β)`、`BW=R/Tp`、Gy/Gz 和 yz 交叉项的定义 |
| Siemens `mo_SPEN_xSPEN/SBBChirp.cpp` | 线性扫频二次 RF 相位、WURST-40 包络 |
| Bruker `xSPEN_scan18/pulseprogram` | 两次相同扫频配相反 Gy；Gz 参与编码与读出 |
| `ss90_chirp180_diffMode1.m` | 90° 激发后延时、180° chirp 与读出之间的全重聚时序关系 |

本次将这两种编码结构移植为可运行的 Pulseq 演示。FOV、时长、RF 幅度、辅助激发、
坡道和采样参数在新脚本中明确设定；没有逐事件复刻某台扫描仪的二进制序列，
也没有把 MID253 的 Hybrid SPEN 当成 crossed-chirp xSPEN。

共同设置：IXI020 第 58 层；64×64；FOV 240 mm；R=64；RO dwell 10 μs；
echo spacing 0.84 ms；读出窗 Ta=53.76 ms；每个 chirp 26.88 ms、BW≈2380.95 Hz。
RF 峰值约 300.37 Hz（7.05 μT），为本次选择的绝热程度；并非已校准的扫描仪发射幅度。
T1=1 s、T2=100 ms、单接收线圈，比较 B0=0 和均匀 +80 Hz。

- **SPEN180**：90° → 编码前延时 → 单个 180° chirp + Gy → 预相位 → 反向 Gy 解码与双极 Gx 读出。
- **xSPEN**：sinc 90° 层选择激发 → chirp/-Gy → 延时 → chirp/+Gy → 双极 Gx 读出；
  Gz 从激发持续到采样结束。名义层厚 6 mm，样品在 z 方向均匀延伸至 1.5 倍层厚，
  实际激发和两次反转由有限 RF 计算，不直接套理想 sinc 核。

**样品与重建的离散化一致。** 先把结构像平均成 64×64 个有效 M0 体素，体素内按常数处理，
每个面内轴使用 8 点 Gauss–Legendre 积分解析快速编码相位。xSPEN 另用 384 点层厚积分。
这是完整信号流程的有限体素演示，图像误差不能当作从独立高分辨率样品恢复细节的证据。
保留最初细网格输入数组，来源结构像不被修改。早期细网格样品/粗网格重建试验也保留在 `runs/`，
其模型不匹配会被小正则化放大，因此没有将其当作正式展示的通过结果。

RF 阶段由 KomaMRI Bloch 积分。RF 结束后，使用 Bloch 自由演化的精确指数解，
保留每个 ADC 点的真实时刻、T2 衰减及 Gy/Gz 演化，包含单条 RO 内部的焦点移动。
由于此处 RF、B0、T1/T2 不随 x 变化，可复用 y/z 的 RF 响应并对 x 做 Fourier 积分；
另用随机 3D 自旋直接运行完整 `.seq`，验证这一分解，没有拟合全局增益或相位。

重建使用零 B0 时的**有限 RF 编码矩阵**，各 kx 按自己的 ADC 时刻求解复数 Tikhonov，
正则项为 `0.01 × σmax²`，与本项目既有传统重建的量级一致，没有按真值搜索参数。
再反演 RO Fourier 积分。重建包含已设定的 T2 和 RF 响应；+80 Hz 条件也使用同一个零 B0 算子。
RO-only 面板是采样信号，单独设窗；输入和最终重建使用共同的有效 M0 强度范围。

验证包括：Pulseq timing、重聚时序、SPEN 二次项/xSPEN 交叉项、RF 步长减半、
分解计算与直接完整 Bloch 的 ADC 一致性、xSPEN 层厚积分加倍、Tikhonov 正规方程残差。
理想相位公式仅作为近似检查；有限幅度 RF 会产生可记录的偏离。
这些检查不等同于与真实扫描信号的校准，也不构成两种方法的一般性能排名。

输出分为 `spen/` 和 `xspen/`，各有完整 `.seq`、RF 编码段 `.seq`、时序图、脑图、
编码响应图、原始复数信号、RF 末态、积分核、重建数组、参数与验证 JSON。
`render_spen_xspen.py` 生成共同尺度的总览。复跑时为 `--output` 指定新的 `runs/` 子目录。

## 参考实现

- [PyPulseq](https://github.com/pulseq/pypulseq)
- [komamripy](https://github.com/JuliaHealth/komamripy)
- [KomaMRI](https://juliahealth.org/KomaMRI.jl/stable/)
- [Open-SPEN](https://github.com/andih98/Open-SPEN)：参考其 chirped-RF SPEN 与编码算子验证思路；
  本 demo 自行构造 Python 波形，没有运行其 MATLAB 脚本，也没有宣称复现其完整序列或论文结果。
- [xSPEN 原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5184846/) 与
  [脑部 xSPEN 扩散论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5740132/)：两次扫频与交叉项编码的背景文献；
  Demo 4 的直接实现依据是上表中的本地源码。

本目录用于演示和物理验证，尚未对接现有 SPENPy 重建或扫描仪原始 ADC 数据。
Demo 3 使用真实结构像的解剖外形，但 ADC 信号由 Bloch 模型重新生成。
