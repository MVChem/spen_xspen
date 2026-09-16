"""Compact simulation and real-data figures from saved arrays, without GPU use.

Layout and typography match the approved SPEN reconstruction figures.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
import numpy as np

CASE_IDS = (3, 8, 17)
CONDITIONS = ('fov16_R1', 'fov16_R2')
METHODS = ('target', 'raw_rss', 'tikhonov', 'phase_inva', 'diffusion')
ROW_LABELS = {'target': 'Ground truth (GT)', 'raw_rss': 'Degraded input',
              'phase_inva': 'Phase map\n+ InvA', 'tikhonov': 'Tikhonov',
              'diffusion': 'Diffusion prior'}


def draw_comparison(images, output_stem, labels, groups=(), annotations=None,
                    row_labels=None):
    """One layout for native-grid images; groups use an exclusive end index."""
    annotations = annotations or {}
    row_labels = {**ROW_LABELS, **(row_labels or {})}
    count = len(labels)
    if not count or not images:
        raise ValueError('At least one image and label are required')
    for name, values in images.items():
        if values.ndim != 3 or len(values) != count or not np.isfinite(values).all():
            raise ValueError(f'{name}: expected finite [N,H,W] images matching labels')
        if name in annotations and len(annotations[name]) != count:
            raise ValueError(f'{name}: metric count does not match images')
    side, gap, group_gap, left = 1.92, .055, .32, 1.90
    starts = {start for start, _, _ in groups if start > 0}
    xs, cursor = [], left
    for col in range(count):
        if col in starts:
            cursor += group_gap-gap
        xs.append(cursor)
        cursor += side+gap
    tops = [(.78 if groups else .42) + row*(side+.075)
            for row in range(len(images))]
    width, height = xs[-1]+side+.12, tops[-1]+side+.095
    metric_font = None
    if annotations:
        metric_font = font_manager.FontProperties(fname=font_manager.findfont(
            font_manager.FontProperties(family='Times New Roman', weight='bold'),
            fallback_to_default=False), size=13, weight='bold')
    style = {'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 11,
             'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42}
    stem = Path(output_stem)
    if stem.suffix in ('.png', '.pdf'):
        stem = stem.with_suffix('')
    stem.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style):
        fig = plt.figure(figsize=(width, height), layout=None)
        try:
            for row, (name, values) in enumerate(images.items()):
                for col, value in enumerate(values):
                    ax = fig.add_axes([xs[col]/width, 1-(tops[row]+side)/height,
                                       side/width, side/height])
                    ax.imshow(value, cmap='gray', vmin=0, vmax=1,
                              interpolation='nearest', aspect='equal')
                    ax.set_axis_off()
                    if name in annotations:
                        ax.text(.035, .965, annotations[name][col],
                                transform=ax.transAxes, ha='left', va='top',
                                color='white', fontproperties=metric_font,
                                path_effects=[patheffects.withStroke(
                                    linewidth=.8, foreground='black')])
                    if row == 0:
                        ax.set_title(labels[col], pad=7)
                h, w = values.shape[-2:]
                fig.text((left-.14)/width, 1-(tops[row]+side/2)/height,
                         f'{row_labels.get(name, name)}\n{h} × {w}',
                         ha='right', va='center', fontsize=11.5, linespacing=1.35)
            for start, end, heading in groups:
                fig.text((xs[start]+xs[end-1]+side)/2/width, 1-.12/height,
                         heading, ha='center', va='top', fontsize=13, fontweight='bold')
            paths = {}
            for extension in ('png', 'pdf'):
                path = stem.with_suffix('.'+extension)
                fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.08)
                paths[extension] = path
            return paths
        finally:
            plt.close(fig)


def render_simulation(outputs, reports, out, label='Diffusion prior'):
    images = {method: [] for method in METHODS}
    annotations = {method: [] for method in METHODS if method != 'target'}
    for key in CONDITIONS:
        for index in CASE_IDS:
            record = reports[key]['records'][index]
            for method in METHODS:
                value = outputs[key][method][index]
                images[method].append(np.rot90(value, 2) if record['rot180'] else value)
                if method in annotations:
                    score = reports[key]['methods'][method]['cases'][index]
                    annotations[method].append(f"{score['psnr']:.2f} / {score['ssim']:.3f}")
    return draw_comparison({k: np.stack(v) for k, v in images.items()},
        Path(out)/'comparison', ['Mouse 1', 'Mouse 2', 'Mouse 3']*2,
        [(0, 3, 'Full PE · σ = 0.01'), (3, 6, 'Random 50% PE · σ = 0.02')],
        annotations, {'diffusion': label})


def render_real(arrays, out):
    """Real arrays are already oriented for display; no GT-based metrics."""
    fovs = arrays['fov_mm']
    groups, start = [], 0
    for end in range(1, len(fovs)+1):
        if end == len(fovs) or fovs[end] != fovs[start]:
            groups.append((start, end, f'FOV {fovs[start]:g} mm'))
            start = end
    return draw_comparison({k: arrays[k] for k in METHODS if k != 'target'},
                           Path(out)/'real_comparison',
                           [str(s) for s in arrays['labels']], groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='Directory with saved evaluation arrays')
    p.add_argument('--kind', choices=['simulation', 'real', 'all'], default='all')
    p.add_argument('--out', type=Path, help='Defaults to the evaluation directory')
    args = p.parse_args()
    out, paths = args.out or args.run, {}
    if args.kind in ('simulation', 'all') and (args.run/'fov16_R1.npz').exists():
        outputs, reports = {}, {}
        for key in CONDITIONS:
            with np.load(args.run/f'{key}.npz', allow_pickle=False) as a:
                outputs[key] = {method: a[method] for method in METHODS}
            reports[key] = json.loads((args.run/f'{key}_metrics.json').read_text())
        config = json.loads((args.run/'config.json').read_text())
        label = ('Archived diffusion' if 'verification' in config.get('purpose', '')
                 else 'Diffusion prior')
        paths['simulation'] = render_simulation(outputs, reports, out, label)
    if args.kind in ('real', 'all') and (args.run/'real.npz').exists():
        with np.load(args.run/'real.npz', allow_pickle=False) as a:
            arrays = {k: a[k] for k in (*METHODS[1:], 'labels', 'fov_mm')}
        paths['real'] = render_real(arrays, out)
    if not paths or (args.kind != 'all' and args.kind not in paths):
        raise FileNotFoundError(f'No {args.kind} arrays in {args.run}')
    print(json.dumps(paths, default=str))


if __name__ == '__main__':
    main()
