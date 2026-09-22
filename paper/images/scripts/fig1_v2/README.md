# fig1_v2：真实数据与可编辑 PPT

原参考图保存在 `../../source/fig1_v2.png`。当前 `../../ppt/fig1_v2.pptx` 已用真实项目数据和程序生成素材替换所有原图裁切；`../../ppt/fig1_v2.png` 来自最终 PPT 的实际渲染。未新增 prompt。

## 重新生成

依赖 Python 3、python-pptx、Pillow、NumPy、Matplotlib；渲染依赖 LibreOffice、Poppler。

```sh
python -m pip install python-pptx Pillow numpy matplotlib
python build.py
python render.py
```

从任意工作目录运行均可。`build.py` 调用 `make_assets.py`，从随附的 `assets/figure_data.npz` 生成缩略图与色标，再生成 PPT。正常重建无需大型数据集或 checkpoint，整个 images 目录可移动。手工编辑 PPT 后应另存，避免重建覆盖。

## 实际数据来源

详见 `assets/provenance.json`（源文件路径、SHA256、训练记录、模型及处理说明）。

- 训练幅度图：项目 192×192 训练数组索引 **823**，ds005236 小鼠 T2w-TurboRARE 实测切片。按训练输入范围 `x=2m−1` 加入 Gaussian noise，σ=0.35、种子 20260922。
- 训练去噪图：`rodent192_spen2x_260914/train/model_ema.pt` 的 EMA U-Net，60000 step；对上面的加噪图实际执行一次 CPU 前向，无人工平滑。
- 真实扫描：`20240321_lxj_spen_mouse_240321_1_1_1/slice_22.mat`，FOV 16 mm，4 coil，96×96，已完成扫描仪相位校正及 RO FFT。
- 固定复数线圈分量：严格调用项目 `evaluate_real_sr.load_case`，由加权 InvA 及拟合复增益得到 z，按 RSS 归一化；实虚部双线性插值至192，再次 RSS 归一化。图中 `|Ŝc|` 和 `arg Ŝc` 直接来自实际算子 `op.coils`；不是独立线圈标定，也不是生成的光滑相位。
- 线圈堆叠显示4个实际 coil。每张位图和边框单独存在，按模块浅层分组。最前面显示零基索引 coil 3，该通道在此案例中的能量最高。
- 重建结果：`joint_phase_diffpir_260916/gpu1_real_residual/real_residual_fov16_export22.npz` 中的 **image_scanner**，即固定扫描仪校正与固定 coil 的 baseline diffusion 结果。
- “Clean estimate” 缩略图：对上述 baseline 结果在 σ=0.02 执行一次实际 Dθ 前向，作为方法示意。它**不是历史60步采样中保存的某个中间状态**，不用于报告性能。
- 测量缩略图为同次扫描 y；预测缩略图为实际算子对 baseline 重建计算的 F(x)。均显示 `log1p(30*abs(data)/q99.5(abs(y)))/log1p(30)`，使用相同窗口，没有把 measured data 当作普通 Cartesian k-space。
- 编码缩略图为真实归一化 A_PE 的幅度，96×192，保留2:1宽高比。

## Phase map 的含义与色标

**校准区域：固定 coil/object phase**，显示 `angle(op.coils[c])`，192×192，全视野，不抠脑、不平滑。色标为 Matplotlib `twilight_shifted`，范围 **−π～π**，各 coil 一致。

**实验扩展区域：实际拟合的偶数行残余相位**，使用同一保存结果中的 **phase_joint**，形状 **48×96**，纵轴是原始采集的偶数 PE 行、横轴是 RO。这不是解剖图上的像素相位，所以不画成彩色脑图、不旋转180°、不套脑掩膜。

该案例实际相位范围 **−0.229140～0.089522 rad**。用 `coolwarm` 和对称 **±0.3 rad** 显示，色标由同一 Python 函数生成。原始全精度数组保留在 NPZ；没有为了颜色丰富而改写数组。

图像空间的真实扫描 MRI 与 coil maps 仅展示时旋转180°，与项目显示约定一致；测量域、编码矩阵和 residual phase 保留原始数组方向。训练切片保留数据集方向。幅度 MRI 窗口固定 [0,1]。FOV24/export11 的保存幅度与 phase 数组也随附，供对照选择。

## 编辑与近似范围

文字、上下标、分式、箭头、边框、U-Net、迭代曲线、RO示意频谱和PE示意采样行均为原生 PPT 对象。两个 U-Net、每个图像堆叠各自小分组；标签保持独立。可在“选择窗格”中选中并替换单个 coil 图层，箭头端点随路径编辑。

MRI、数值热图和色标为 Python 从数组渲染的位图，内部像素不可编辑为PPT顶点。没有引用原参考图裁切，没有嵌入SVG。RO频谱和PE行示意只解释算子，不宣称恢复实验曲线。字体为 Arial 和 Liberation Serif，平色近似参考图渐变。

若需要从项目原始文件重新提取数值资产：

```sh
python extract_data.py --workspace /path/to/spen_xspen
```

该可选步骤另需项目环境中的 PyTorch、SciPy、spenpy 及模型依赖，读取真实数据与checkpoint，使用CPU，重新生成 NPZ 和来源记录；不训练模型、不重跑完整采样。
