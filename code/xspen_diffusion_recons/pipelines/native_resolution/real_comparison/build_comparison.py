"""Build a reproducible real xSPEN comparison from final cached reconstructions.

CPU only. Source H5/MAT, evaluation arrays and checkpoints are never modified.
All anatomical columns share one measurement-derived scale and display window.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.io import loadmat
import torch

from traditional_phase import reconstruct_traditional

OUT = Path(__file__).resolve().parent
EXP = OUT.parent
PROJECT = EXP.parents[1]
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': FONT.get_name(), 'axes.unicode_minus': False,
                     'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
MAIN = ['input_ro_rss', 'complex_tikhonov', 'magnitude_l2',
        'phasemap_inva', 'diffusion128', 'diffusion_native']
EXTENDED = ['input_ro_rss', 'original_mat', 'complex_tikhonov', 'magnitude_l2',
            'windowed_adjoint', 'phasemap_inva', 'diffusion128', 'diffusion_native']
TITLES = {
    'input_ro_rss': '输入：RO + RSS\nPE 编码未反演',
    'original_mat': '原 MATLAB · Img\nRO 加窗 + RSS',
    'complex_tikhonov': 'Tikhonov + RSS\n逐线圈复数 · α=0.01',
    'magnitude_l2': '幅度 L2 / CG\n固定线圈相位 · ρ=0.003',
    'windowed_adjoint': '加窗 InvA\n未做 PhaseMap 校正',
    'phasemap_inva': 'PhaseMap + InvA\nxSPEN sinc 算子适配',
    'diffusion128': '原 128² Diffusion\nEDM + DiffPIR',
    'diffusion_native': '原生网格 Diffusion\n本次训练 · 20,000 步',
}
REPRESENTATIVES = ['MID112_rep00_slice037', 'MID114_rep00_slice026', 'MID27_rep00_slice026']
VIEWS = {'MID112': '轴位 · 3 mm', 'MID114': '矢状位 · 3 mm', 'MID27': '轴位 · 约 4 mm'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def collect_cases():
    """Select all 36 already evaluated native-grid observations, no score filter."""
    files = []
    for job in ['p3mm_mm3', 'p4mm_mm4']:
        files.extend(sorted((EXP / 'evaluation' / job).glob('MID*.npz')))
    assert len(files) == 36, len(files)
    return files


def process_case(path, originals, scanner_metadata):
    stem = path.stem
    info = json.loads(path.with_suffix('.json').read_text())
    scan, repeat, sl = info['scan'], info['repeat'], info['slice_index']
    if scan not in scanner_metadata:
        meta = json.loads((PROJECT / 'scanner' / f'{scan}.json').read_text())
        scanner_metadata[scan] = meta
        originals[scan] = loadmat(Path(meta['source']).with_suffix('.mat'), variable_names=['Img'])['Img'][:, :, 0, :]
    meta = scanner_metadata[scan]
    scale = info['magnitude_scale']
    with np.load(path) as source:
        saved = {key: source[key].copy() for key in source.files}
    assert info['native_shape'] == info['output_shape']
    with h5py.File(PROJECT / 'scanner' / f'{scan}.h5', 'r') as source:
        raw = source['kspace'][repeat, sl]
    np.testing.assert_allclose(raw / scale, saved['original_measurement'][0], rtol=2e-6, atol=1e-7)
    traditional = reconstruct_traditional(raw, meta, magnitude_scale=scale)
    ro = traditional['ro_image']
    rss = np.sqrt(np.sum(np.abs(ro)**2, axis=0))
    np.testing.assert_allclose(rss, ((saved['degraded'] + 1) / 2)[0, 0], rtol=3e-5, atol=1e-6)
    original = originals[scan]
    ns = original.shape[-1] // meta['repeated_counter_occurrences']
    counter = meta['slice_order'][sl]
    original_slice = 2 * (counter - ns // 2) if counter >= ns // 2 else 2 * counter + 1
    frame = repeat * ns + original_slice
    mat = original[:, :, frame].astype(np.float64)
    corr = float(np.corrcoef(rss.ravel(), mat.ravel())[0, 1])
    if corr < .85:
        raise ValueError(f'Original MATLAB frame mapping failed: {stem}, {corr}')
    # MAT contains filtered RO data in a different receiver/FFT convention.
    # Fit one positive scalar to the input RSS only, never to any reconstruction.
    mat_gain = float(np.sum(mat * rss) / np.sum(mat**2))
    assert mat_gain > 0
    arrays = {
        'input_ro_rss': ((saved['degraded'] + 1) / 2)[0, 0],
        'original_mat': mat * mat_gain,
        'complex_tikhonov': ((saved['native_tikhonov'] + 1) / 2)[0, 0],
        'magnitude_l2': ((saved['tikhonov'] + 1) / 2)[0, 0],
        'windowed_adjoint': traditional['windowed_adjoint_only'],
        'phasemap_inva': traditional['magnitude'],
        'diffusion128': ((saved['baseline128'] + 1) / 2)[0, 0],
        'diffusion_native': ((saved['diffusion'] + 1) / 2)[0, 0],
    }
    for method, data in arrays.items():
        assert data.shape == tuple(info['native_shape']), (stem, method, data.shape)
        assert np.isfinite(data).all(), (stem, method)
    summary = json.loads((path.parent / 'summary.json').read_text())
    assert summary['checkpoint_step'] == 20000
    record = dict(case=stem, scan=scan, repeat=repeat, slice_index=sl,
                  native_shape=info['native_shape'], fov_mm=info['fov_mm'],
                  pixel_mm=(np.array(info['fov_mm']) / info['native_shape']).tolist(),
                  thickness_mm=info['thickness_mm'], magnitude_scale=scale,
                  source_evaluation=str(path), source_evaluation_sha256=sha256(path),
                  source_raw=meta['source'], source_raw_sha256=meta['source_sha256'],
                  sequence=meta['sequence'], original_mat_frame_zero_based=int(frame),
                  original_mat_gain_to_input=mat_gain, original_mat_input_correlation=corr,
                  checkpoint=summary['config']['checkpoint'], checkpoint_step=20000,
                  checkpoint_sha256=summary.get('checkpoint_sha256'),
                  baseline_checkpoint=summary['config']['baseline'],
                  reconstruction_steps=info['steps'], phase_method=traditional['metadata'],
                  display_window=[0, 1], independent_method_normalization=False,
                  display_clipped_fractions={key: {'below_zero': float(np.mean(value < 0)),
                                                 'above_one': float(np.mean(value > 1))}
                                             for key, value in arrays.items()},
                  main_png=f'cases/{stem}.png', extended_png=f'cases/{stem}_extended.png',
                  no_clean_ground_truth=True)
    np.savez_compressed(OUT / 'arrays' / f'{stem}.npz', **arrays,
                        raw_complex_coils_pe_ro=raw,
                        phase_map_rad=traditional['phase_map_rad'],
                        phasemap_coil_images=traditional['coil_images'],
                        phasemap_corrected_ro_image=traditional['corrected_ro_image'])
    write_json(OUT / 'arrays' / f'{stem}.json', record)
    return arrays, record


def draw_rows(images, records, names, methods, title):
    nrow, ncol = len(names), len(methods)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.65*ncol + 1.65, 2.72*nrow + 1.5), squeeze=False)
    fig.suptitle(title, fontsize=17, y=.97, fontweight='bold')
    for row, name in enumerate(names):
        rec = records[name]
        height, width = rec['fov_mm']
        dy = height / rec['native_shape'][0]
        for col, method in enumerate(methods):
            ax = axes[row, col]
            is_input = method in ('input_ro_rss', 'original_mat')
            shift = .5 * dy if is_input else 0
            ax.imshow(images[name][method], cmap='gray', vmin=0, vmax=1,
                      interpolation='nearest', extent=(0, width, height-shift, -shift))
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.set_facecolor('black')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(TITLES[method], fontsize=11.4, pad=10,
                             color='#115f63' if method == 'diffusion_native' else '#17232d')
            if col == 0:
                shape = rec['native_shape']
                label = f"{rec['scan']}\n{VIEWS[rec['scan']]}\n{shape[0]} × {shape[1]}\n层 {rec['slice_index']} · 采集 {rec['repeat']}"
                ax.text(-.08, .5, label, transform=ax.transAxes, ha='right', va='center', fontsize=11)
            if col == ncol-1:
                # Physical scale bar, common FOV display without upsampling.
                xend, ybar = width - 8, height - 8
                ax.plot([xend-30, xend], [ybar, ybar], color='white', lw=2)
                ax.text(xend-15, ybar-3, '30 mm', color='white', fontsize=8, ha='center')
    footer = '同一行：同一真实观测 · 原生采样网格 · 统一灰度窗 [0, 1] · 无真实参考图像（GT）'
    fig.text(.54, .036, footer, ha='center', fontsize=11, color='#36424d')
    fig.subplots_adjust(left=.102 if ncol == 6 else .080, right=.995,
                        top=1-.99/(2.72*nrow+1.5), bottom=.075 if nrow==1 else .07,
                        wspace=.055, hspace=.09)
    return fig


def save_comparison(images, records, names, methods, path, title, pdf=None):
    fig = draw_rows(images, records, names, methods, title)
    fig.savefig(path, dpi=155)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)


def raw_input_figure(records):
    fig, axes = plt.subplots(3, 3, figsize=(12, 11.3))
    fig.suptitle('真实 xSPEN 输入：复数测量与 RO + RSS 显示', fontsize=18, fontweight='bold', y=.975)
    for row, name in enumerate(REPRESENTATIVES):
        with np.load(OUT / 'arrays' / f'{name}.npz') as source:
            raw, rss = source['raw_complex_coils_pe_ro'], source['input_ro_rss']
        coil = int(np.argmax(np.sum(np.abs(raw)**2, axis=(1, 2))))
        mag = np.sqrt(np.sum(np.abs(raw)**2, axis=0))
        db = 20*np.log10(np.maximum(mag / mag.max(), 1e-6))
        a = axes[row, 0].imshow(db, cmap='magma', vmin=-60, vmax=0, interpolation='nearest', aspect='auto')
        b = axes[row, 1].imshow(np.angle(raw[coil]), cmap='twilight', vmin=-np.pi, vmax=np.pi,
                                interpolation='nearest', aspect='auto')
        axes[row, 2].imshow(rss, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        rec = records[name]
        for ax in axes[row, :2]:
            ax.set_xlabel('RO 采样点')
        axes[row, 0].set_ylabel(f"{rec['scan']}\n层 {rec['slice_index']} · 采集 {rec['repeat']}\nPE 编码行")
        axes[row, 1].set_ylabel('PE 编码行')
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])
        axes[row, 0].set_title('32 线圈 RSS 测量幅度 / dB')
        axes[row, 1].set_title(f'单线圈复数相位 / rad（coil {coil}）')
        axes[row, 2].set_title('输入图：RO + RSS（PE 未反演）')
        fig.colorbar(a, ax=axes[row, 0], fraction=.045, pad=.02)
        fig.colorbar(b, ax=axes[row, 1], fraction=.045, pad=.02, ticks=[-np.pi, 0, np.pi])
    fig.text(.52, .022, '左两列属于测量域；幅度以各观测峰值为 0 dB。右列使用重建对照的固定灰度窗。',
             ha='center', fontsize=11)
    fig.subplots_adjust(left=.12, right=.985, top=.91, bottom=.075, wspace=.32, hspace=.35)
    fig.savefig(OUT / 'raw_input.png', dpi=170)
    plt.close(fig)


def write_report(records, pages):
    lines = [
        '# 真实 xSPEN：输入、传统重建和 Diffusion 对照', '',
        '使用 3 组真实 crossed-chirp xSPEN 扫描的 36 个既有测试观测。'
        '每个观测对应一个 slice × repeat；不是 36 位独立受试者。'
        '全部深度学习结果来自训练结束后的 checkpoint，原生模型为 20,000 步。', '',
        '![三组扫描总览](overview.png)', '',
        '[八列完整总览](overview_extended.png) · [原始复数输入](raw_input.png) · '
        '[全部对照 PDF](all_comparisons.pdf) · [可离线浏览的图册](gallery.html)', '',
        '| 扫描 | 视图 | 采样矩阵 PE × RO | 面内像素 | 层厚 | 展示观测 |',
        '|---|---|---|---|---|---|',
        '| MID112 | 轴位 | 60 × 64 | 3 × 3 mm | 3 mm | 4 层 × 3 次采集 |',
        '| MID114 | 矢状位 | 60 × 64 | 3 × 3 mm | 3 mm | 4 层 × 3 次采集 |',
        '| MID27 | 轴位 | 46 × 48 | 4.02 × 4.02 mm | 4 mm | 4 层 × 3 次采集 |', '',
        '这是同一序列家族的两种采集几何。repeat 是时间顺序索引，尚未核实为 b0 或特定扩散方向。'
        '切片和 repeat 标签均从 0 起计。总览选择 MID112/37、MID114/26、MID27/26 的 repeat 0，'
        '依据脑部覆盖和视图选择；完整图册保留全部 36 例。', '',
        '## 方法与标签', '',
        '| 图中方法 | 实际处理 |', '|---|---|',
        '| 输入 RO + RSS | H5 中未做奇偶相位校正的 32 通道测量，仅逆变换 RO 并 RSS 合并；PE 编码未反演。H5 已完成读取阶段的 regrid、去 RO 过采样和 reflection 校正。 |',
        '| 原 MATLAB · Img | 原文件保存的 RO 高斯加窗、Fourier 变换、RSS 图像，已核实同一 slice/repeat。不是完整的 xSPEN PE 反演。 |',
        '| Tikhonov + RSS | 当前 sinc 编码算子；输入奇偶相位校正后逐线圈求复数正则逆 (AᴴA+0.01I)⁻¹Aᴴ，再 RSS。 |',
        '| 幅度 L2 / CG | 固定从同一观测估计的 coil/object phase/gain，求共同实幅度；x=2m−1 域 ρ=0.003，相当于 m 域 0.012。此列无学习先验。 |',
        '| 加窗 InvA | xSPEN sinc 核的高斯加窗伴随，宽度固定 0.8，未作 PhaseMap 校正。InvA 在此是历史名称，并非严格矩阵逆。 |',
        '| PhaseMap + InvA | 从原始输入独立进行 odd/even 部分 InvA 重建，拟合相位差并修正偶数行，再加窗 InvA + RSS。沿用项目 notebook 方法的 xSPEN sinc 适配；未复用 Diffusion 的奇偶相位场。 |',
        '| 原 128² Diffusion | 原 50,000 步 EDM 先验，经上下采样适配同一物理算子和原生输出网格；60 步 DiffPIR。 |',
        '| 原生网格 Diffusion | 本次 20,000 步微调的相应原生分辨率 EDM 先验；同一观测和物理算子，60 步 DiffPIR。 |', '',
        '## MATLAB 中其他传统方法', '',
        '同批 2016 crossed-chirp 采集的原始 MATLAB 只实现 RO 加窗与 RSS。'
        '其他目录确有单 chirp/Hybrid SPEN 的 L2 梯度平滑、TV/L1、GRAPPA、ESPIRiT/BART 等实现，'
        '依赖不同的编码算子或校准。没有将其直接列为这批 xSPEN 已复现结果。'
        '上表的 sinc Tikhonov、幅度 L2 和 PhaseMap + InvA 均如实标为当前 xSPEN 算子上的适配。'
        '具体源码位置和适配限制见 [MATLAB 方法审计](matlab_methods_audit.md)。', '',
        '## 显示与解读', '',
        '同一行所有图使用同一测量导出的强度尺度和 [0,1] 灰度窗，不做逐方法 min–max 或百分位归一化。'
        '尺度为该观测复数 Tikhonov RSS 的第 99.5 百分位，跨观测不是绝对信号标定。'
        '原 MATLAB Img 使用不同接收机/FFT 尺度，仅拟合一个到输入 RO+RSS 的正标量，逐例记录，'
        '没有按重建结果调整对比度。保存数组未裁剪。', '',
        '图像保留原生像素，nearest 显示，按 FOV 保持物理长宽比。'
        'RO+RSS 与 MATLAB Img 的 PE 采样中心比重建中心提前半像素，使用对应 extent 对齐，'
        '不额外插值。为使用共同物理视野，输入边缘半像素可能裁出/留黑。'
        '本页的分辨率是采样网格，不以更平滑的视觉表现证明实际分辨能力。', '',
        '真实数据没有干净真值，不能据图计算真实 PSNR/SSIM 或断言细节更准确。'
        '当前 sinc 模型由头文件推导，尚无独立波形/B0 与线圈标定；'
        '传统与学习方法采用的相位估计路径也并非完全相同。', '',
        '## 全部 36 个观测', '',
    ]
    for page in pages:
        lines.append(f"- [{page['scan']} · 采集 {page['repeat']} · 4 层]({page['main_png']})"
                     f" / [八列扩展]({page['extended_png']})")
    lines.extend(['', '[逐例清单与参数](manifest.json) · [数据显示审计](data_display_audit.md) · '
                  '[数值核验](data_display_audit.json) · [MATLAB 帧对应核验](original_mat_alignment_checks.json) · '
                  '[重现脚本](build_comparison.py)', '',
                  '图册生成仅使用 CPU，未重训模型、未占用 GPU、未改动原始扫描文件。'])
    (OUT / '真实数据对照.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse', action='store_true', help='Reuse already generated traditional arrays')
    args = parser.parse_args()
    torch.set_num_threads(2)
    for folder in ['arrays', 'cases', 'pages']:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    images, records = {}, {}
    originals, scanner_metadata = {}, {}
    for i, path in enumerate(collect_cases(), 1):
        if args.reuse and (OUT / 'arrays' / f'{path.stem}.json').exists():
            record = json.loads((OUT / 'arrays' / f'{path.stem}.json').read_text())
            assert record['source_evaluation_sha256'] == sha256(path)
            with np.load(OUT / 'arrays' / f'{path.stem}.npz') as source:
                arrays = {key: source[key].copy() for key in EXTENDED}
        else:
            arrays, record = process_case(path, originals, scanner_metadata)
        images[path.stem], records[path.stem] = arrays, record
        print(json.dumps({'reconstructed': i, 'total': 36, 'case': path.stem}), flush=True)
    title = '真实 xSPEN 人脑：输入、传统重建与 Diffusion'
    save_comparison(images, records, REPRESENTATIVES, MAIN, OUT / 'overview.png', title)
    save_comparison(images, records, REPRESENTATIVES, EXTENDED, OUT / 'overview_extended.png', title + ' · 完整八列')
    raw_input_figure(records)
    pages = []
    with PdfPages(OUT / 'all_comparisons.pdf') as pdf:
        for name, record in records.items():
            for methods, key in [(MAIN, 'main_png'), (EXTENDED, 'extended_png')]:
                save_comparison(images, records, [name], methods, OUT / record[key], title)
        for scan in ['MID112', 'MID114', 'MID27']:
            repeats = sorted({r['repeat'] for r in records.values() if r['scan'] == scan})
            for repeat in repeats:
                names = [n for n, r in records.items() if r['scan'] == scan and r['repeat'] == repeat]
                page = dict(scan=scan, repeat=repeat, cases=names,
                            main_png=f'pages/{scan}_rep{repeat:02d}.png',
                            extended_png=f'pages/{scan}_rep{repeat:02d}_extended.png')
                for methods, key in [(MAIN, 'main_png'), (EXTENDED, 'extended_png')]:
                    save_comparison(images, records, names, methods, OUT / page[key], title, pdf=pdf)
                pages.append(page)
                print(json.dumps({'rendered_page': page['main_png']}), flush=True)
    manifest = dict(records=list(records.values()), pages=pages,
                    overview_png='overview.png', extended_overview_png='overview_extended.png',
                    raw_input_png='raw_input.png', representative_cases=REPRESENTATIVES,
                    main_methods=MAIN, extended_methods=EXTENDED, titles=TITLES,
                    no_clean_ground_truth=True, main_display_window=[0, 1],
                    method_sources={p.name: sha256(p) for p in [Path(__file__), OUT / 'traditional_phase.py']})
    write_json(OUT / 'manifest.json', manifest)
    write_report(records, pages)
    print(json.dumps({'complete': True, 'cases': len(records), 'out': str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
