# 96×96 PNG → pixel DiT → SPEN 逆问题

本入口直接读取 `code/data/rodent96_expanded_260917/{train,val,test}` 的 16 位 PNG，从头训练无条件 EDM DiT。没有 VAE，未加载旧 U-Net 或 latent DiT 权重。

- 参数：20,728,976；宽度 320，11 个 Transformer block，5 heads，patch 4×4，576 tokens。
- 训练：60,000 个 optimizer step；GPU 1；batch 96（microbatch 48 × 累积 2）；BF16；AdamW，峰值 LR 2e-4，预热 500 步，余弦下降到 2e-5；EMA 最大衰减 0.9995。
- 数据：28,160 张训练 PNG 等概率有放回采样，其中实验室 RAT 9,702 张（34.45%）。在线增强复用旧 96 模型。验证、测试分别 2,632 / 2,280 张，不进入训练。
- 验证：固定 256 张按来源/个体轮询选取的验证图，每 1,000 步保存验证指标和 checkpoint；每 5,000 步输出无条件样本。当前实验室数据没有独立验证集。
- 输出：`code/spen_diffusion_recons/runs/rodent96_dit20m_260917/`；图片数据目录不新增 JSON 或 NPY。数据哈希、配置、源码快照和训练日志均放 runs。

`run_campaign.py` 串联训练与最终逆问题测试，并绑定 GPU UUID；重复执行可从最后 checkpoint 恢复。训练入口 `train.py`；模型 `pixel_model.py`；PNG 读取及审计 `png_data.py`；测试入口 `evaluate_inverse.py`。

项目根 `.venv` 已链接到 `/home/data2/chk/workspace/2026/.venv`。SPENPy 从官方 GitHub 的 `v1.0.0` 安装，不依赖本地 `PYTHONPATH` 覆盖。在项目根目录执行：

```bash
.venv/bin/python \
  code/spen_diffusion_recons/scripts/dit96/run_campaign.py
```

完成 60,000 步后，自动使用最终 EMA 评估；`best_ema.pt` 仅保留供研究，不按测试指标选模型。重建沿用 `SPEN96_Reconstruction_Comparison_260916` 的 30 张留出图 × 两种采样条件，以及 10 个真实 MAT 案例。使用相同复数观测、DiffPIR 60 步、固定验证集参数和随机种子，图中并排展示原 U-Net / 新 DiT。真实采集无配对 GT，仅比较测量残差与图像；此次模型架构、数据及训练预算同时变化，不能当成单独的架构消融。

短程流程核验目录 `runs/rodent96_dit20m_smoke_260917` 只用于验证训练、恢复与全部逆问题代码可运行，不是正式训练结果。

## 均衡采样重训

`--sampling balanced` 使用 `balanced_sampling.py` 中预先固定的混合比例：旧 mouse 80%、旧 rat 10%、实验室来源 8%、新增 ds005186 和 Aging 各 1%。旧数据在物种内保留原 manifest 的个体/FOV 权重，因此 mouse FOV16/FOV24 总占比分别为 48%/32%；新增组内按标记个体、扫描、回波、切片分层分配，切片更多的扫描不会得到更多总权重。实验室物种仍标记为未确认。

所有训练 PNG 的概率均大于零，验证/测试像素及划分保持不变。训练目录保存 `sampling_audit.json`、按 PNG 顺序排列的 float64 `sampling_probabilities.npy`，并将策略和元数据哈希绑定到断点。均衡实验从头训练，保留模型、60k 步、batch 96、学习率和增强设置。

在项目根目录启动均衡训练及完成后的自动评估：

```bash
.venv/bin/python code/spen_diffusion_recons/scripts/dit96/run_campaign.py \
  --out code/spen_diffusion_recons/runs/rodent96_dit_balanced_260917 \
  --sampling balanced --select-lambda
```

`--select-lambda` 在每种仿真条件的 16 张验证图上搜索 λ=0.1/0.3/1/3/10，按个体平均 PSNR 选择，然后重建原 30 张测试图；真实数据仍沿用无 GT 的固定 λ=1 协议。检查点采用最终 60k EMA，不按测试结果选择。均衡训练可通过同一 campaign 命令从 `latest.pt` 继续。

本轮流程核验位于 `runs/rodent96_dit_balanced_smoke_260917/`；其中 16 步模型仅用于训练/恢复检查，`evaluation_existing60k/` 使用的是旧均匀采样 60k 权重，验证新增选参流程，不是均衡训练的结果。
