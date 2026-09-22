"""Render native-grid Hybrid SPEN pilot comparisons and a local report."""
import argparse
import csv
import html
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run/'prepared/manifest.json').read_text())
    names = {'mouse': 'Mouse prior (96)', 'human_old': 'Human prior (50k)', 'human_new': 'Human prior (+2k)'}
    summaries = {key: json.loads((run/key/'summary.json').read_text()) for key in names}
    metric_maps = {key: {r['id']: r['metrics'] for r in summary['rows']} for key, summary in summaries.items()}
    aggregate, flat = {}, []
    for kind in ('real', 'simulation'):
        cases = [r for r in manifest['cases'] if r['kind'] == kind]
        methods = [('input', 'RO FFT + RSS'), ('inva', 'PhaseMap + InvA' if kind == 'real' else 'InvA (zero phase)'),
                   ('tikhonov', 'Tikhonov')] + list(names.items())
        if kind == 'simulation':
            methods.insert(0, ('gt', 'Ground truth'))
        fig, axes = plt.subplots(len(methods), len(cases), figsize=(2.5*len(cases), 2.45*len(methods)), squeeze=False)
        for col, case in enumerate(cases):
            ident = case['id']
            source = dict(np.load(run/'prepared'/f'{ident}.npz'))
            for row, (method, label) in enumerate(methods):
                ax = axes[row, col]
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                image = np.load(run/method/f'{ident}.npz')['native'] if method in names else source[method]
                if method == 'input':
                    image = image/max(float(np.quantile(image, .995)), 1e-12)
                ax.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                if col == 0:
                    ax.set_ylabel(label, fontsize=10)
                if row == 0:
                    title = f'Slice {case["slice"]+1} / {case["label"]}' if kind == 'real' else case['label']
                    ax.set_title(title, fontsize=11)
                if method not in ('input', 'gt'):
                    metrics = metric_maps[method if method in names else 'human_old'][ident][
                        'diffusion' if method in names else method]
                    flat.append(dict(id=ident, kind=kind, method=method, **metrics))
                    caption = f'DC {metrics["residual"]:.3f}' if kind == 'real' else f'{metrics["psnr"]:.2f} dB / {metrics["ssim"]:.3f}'
                    ax.set_xlabel(caption, fontsize=9)
        fig.suptitle('MID253 native 60 x 64; same window per case' if kind == 'real' else
                     'Held-out IXI T2; native 60 x 64; PSNR / SSIM', fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, .98])
        fig.savefig(out/f'{kind}_comparison.png', dpi=150)
        fig.savefig(out/f'{kind}_comparison.pdf')
        plt.close(fig)
        aggregate[kind] = {}
        for method in ['inva', 'tikhonov', *names]:
            rows = [r for r in flat if r['kind'] == kind and r['method'] == method]
            aggregate[kind][method] = {key: float(np.mean([r[key] for r in rows]))
                                     for key in ['residual'] + (['psnr', 'ssim'] if kind == 'simulation' else [])}
    with (out/'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'kind', 'method', 'min', 'max', 'residual', 'psnr', 'ssim'])
        writer.writeheader(); writer.writerows(flat)
    old = json.loads((run/'human_finetune/baseline.json').read_text())['initial_model']
    vals = [json.loads(line) for line in (run/'human_finetune/val_metrics.jsonl').read_text().splitlines()]
    best = min(v['val_loss'] for v in vals)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot([v['step'] for v in vals], [v['val_loss'] for v in vals], 'o-', label='Continued training')
    ax.axhline(old, color='gray', linestyle='--', label='Existing 50k prior')
    ax.set(xlabel='Additional steps', ylabel='Held-out EDM loss')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'training.png', dpi=140); plt.close(fig)
    text_rows = []
    for method in ['inva', 'tikhonov', *names]:
        s, r = aggregate['simulation'][method], aggregate['real'][method]
        text_rows.append(f'<tr><td>{html.escape(names.get(method, method))}</td><td>{s["psnr"]:.2f}</td><td>{s["ssim"]:.4f}</td><td>{r["residual"]:.4f}</td></tr>')
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Hybrid SPEN 人脑先验初试</title>
<style>body{{max-width:1500px;margin:32px auto;padding:0 24px;font:16px/1.65 system-ui;color:#222}}img{{max-width:100%}}table{{border-collapse:collapse}}th,td{{padding:8px 20px;border-bottom:1px solid #ddd}}a{{color:#12649a}}code{{background:#eee}}</style>
<h1>Hybrid SPEN 人脑先验初试 · 260917</h1>
<p>已完成 IXI 人脑 EDM 2,000 步继续训练、8 帧 MID253 实采重建、4 位留出 IXI 受试者仿真；使用 GPU 2、3。旧目录已有 50,000 步人脑权重，本次从它初始化，并保留旧权重对照。</p>
<p>验证损失：旧权重 {old:.8f}；继续训练最佳 {best:.8f}。数值越低越好；本次短程训练没有改善该验证指标。IXI 为 T2/PD 幅度先验，没有把实采 DWI 当作干净训练目标。</p>
<table><tr><th>方法</th><th>仿真均值 PSNR / dB</th><th>仿真均值 SSIM</th><th>实采均值 DC 残差</th></tr>{''.join(text_rows)}</table>
<p>仿真只有 4 位受试者的中央轴位切片，理想二次编码、已知平滑线圈、2% 复数 RMS 噪声，无奇偶相位误差，不能代替 DWI 验证。真实数据没有配对真值，DC 残差越低只说明更符合当前近似观测模型，不代表解剖更准确。</p>
<h2>真实 Hybrid SPEN</h2><a href="real_comparison.png"><img src="real_comparison.png"></a>
<p>MATLAB 第 20、27 层，各包含 b0 / DWI-RO / DWI-PE / DWI-SS。全部方法显示在 60×64 原生网格；原始层序，无翻转或解剖左右重标。每例重建统一用 Tikhonov p99.5 标度，显示窗 0–1；RO 输入独立 p99.5 窗。InvA 用一个由复数信号拟合的全局接收增益调整量纲，未对真值拟合。</p>
<h2>留出 IXI 仿真</h2><a href="simulation_comparison.png"><img src="simulation_comparison.png"></a>
<p>PSNR/SSIM 在原生网格、固定 data_range=1 的未截断数组上计算；未单独缩放到真值。鼠脑先验在 96 网格、人脑先验在 128 网格工作，因此此对照同时包含训练域和网格差异，不能单独归因于物种。</p>
<h2>训练</h2><img src="training.png">
<h2>算子与限制</h2><p>使用 Hybrid SPEN 的二次相位 A，RO FFT 后使用单位 RO 算子。传统 PhaseMap 通过低分辨率解码、校正、重编码处理偶数行；重建的数据约束作用在这份校正后的信号。固定线圈/物体相位由同一观测的 Tikhonov 结果估计，没有独立灵敏度或 B0 标定。DiffPIR 用 60 步、lambda=1、xi=0、seed=917。实采 sigma_noise=0.02 为假设值；仿真按已知注入噪声除以 A 范数、信号标度和 sqrt(2)，转换到归一化观测的单分量标准差，未按测试指标选参。用精确像素重叠将先验网格投影回原生网格；更大的先验网格不代表已获得更高的实测分辨率。</p>
<p>InvA 与存档 MATLAB 矩阵相对误差约 7.5e-15。Python 与 MATLAB 的 RO 输入经单一标度对齐后 NRMSE 约 0.9–1.4%；最终传统图仍有约 5–9% 差异，保留实际复算结果，没有替换成旧图。首轮统一使用 0.02 噪声假设的结果保留在 *_fixednoise 目录；本页仿真使用解析噪声校准后的结果，实采参数没有改变。</p>
<p><a href="metrics.csv">逐例指标 CSV</a> · <a href="summary.json">汇总 JSON</a> · <a href="real_comparison.pdf">实采 PDF</a> · <a href="simulation_comparison.pdf">仿真 PDF</a></p>
<p>完整运行目录：<code>{html.escape(str(run))}</code>。复数输入、相位图、原始/校正信号、先验网格、权重、命令及源码快照均保存在该目录。没有移动到 experiments。</p></html>'''
    (out/'index.html').write_text(content)
    (out/'summary.json').write_text(json.dumps(dict(aggregate=aggregate, initial_validation_loss=old,
                                                   continued_validation_loss=best), indent=2)+'\n')
    shutil.copyfile(__file__, run/'render_hybrid_pilot_source.py')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
