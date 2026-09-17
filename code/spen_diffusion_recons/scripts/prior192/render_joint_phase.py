"""Render saved joint-phase pilot arrays without rerunning any inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = dict(uncorrected='Fixed even-line phase (0)', sequential='Phase fit, then diffusion',
              joint='Joint phase net + diffusion', oracle='Known phase + diffusion',
              scanner='Scanner correction + diffusion')


def load_records(runs):
    rows = []
    for run in runs:
        summary = json.loads((run / 'summary.json').read_text())
        for row in summary['cases']:
            row['run'] = run
            row['config'] = summary['config']
            rows.append(row)
    if len({row['name'] for row in rows}) != len(rows):
        raise ValueError('Duplicate cases across requested runs')
    return rows


def images(rows, path, title):
    real = rows[0]['kind'] == 'real'
    methods = list(rows[0]['methods'])
    keys = methods if real else ['target'] + methods
    fig, axes = plt.subplots(len(keys), len(rows), squeeze=False,
                             figsize=(2.55 * len(rows) + 1.5, 2.55 * len(keys)))
    for col, row in enumerate(rows):
        with np.load(row['run'] / (row['name'] + '.npz')) as data:
            for r, key in enumerate(keys):
                ax = axes[r, col]
                array = data['target' if key == 'target' else 'image_' + key]
                if real:
                    array = np.rot90(array, 2)
                ax.imshow(array, cmap='gray', vmin=0, vmax=1, interpolation='none')
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    case_title = f"FOV {row['fov_mm']} mm / export #{row['export']}" if real else row['dataset']
                    ax.set_title(case_title, fontsize=10)
                if col == 0:
                    ax.set_ylabel('Ground truth' if key == 'target' else LABELS[key], fontsize=10)
                if key != 'target':
                    m = row['methods'][key]
                    if real:
                        label = f"Residual {m['measurement_nrmse']:.3f}"
                    else:
                        label = f"{m['psnr']:.2f} dB / {m['ssim']:.3f}\nPhase {m['phase_rmse_rad']:.3f} rad"
                    ax.text(.025, .975, label, transform=ax.transAxes, va='top', color='white',
                            fontsize=9, bbox=dict(facecolor='black', alpha=.5, linewidth=0, pad=2))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def phases(rows, path):
    fig, axes = plt.subplots(4, len(rows), squeeze=False,
                             figsize=(2.7 * len(rows) + 1.4, 9.2))
    for col, row in enumerate(rows):
        with np.load(row['run'] / (row['name'] + '.npz')) as data:
            truth, phase = data['phase_true'], data['phase_joint']
            error = np.angle(np.exp(1j * (phase - truth)))
            arrays = [truth, data['phase_sequential'], phase, error]
            for r, arr in enumerate(arrays):
                ax = axes[r, col]
                im = ax.imshow(arr, cmap='coolwarm', vmin=-1.6 if r < 3 else -.3,
                               vmax=1.6 if r < 3 else .3, aspect='auto')
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(['Injected phase', 'Sequential estimate', 'Joint estimate',
                                   'Joint wrapped error'][r], fontsize=10)
                if r == 0:
                    ax.set_title(row['sampling'] + ' / ' + row['dataset'], fontsize=10)
            fig.colorbar(im, ax=axes[3, col], orientation='horizontal', fraction=.07, pad=.08)
    fig.suptitle('Acquired-even phase [rad], color range [-1.6, 1.6] | Error over all coordinates\n'
                 'Reported phase RMSE uses only acquired rows, weighted by clean signal energy', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_records(args.runs)
    for condition in ['full', 'half']:
        selected = [r for r in rows if r['kind'] == 'simulation' and r['sampling'] == condition]
        for kind in ['smooth2d', 'none']:
            group = [r for r in selected if r['phase_kind'] == kind]
            if group:
                steps = group[0]['config']['steps']
                images(group, args.out / f'simulation_{condition}_{kind}.png',
                    f"{condition.upper()} PE | {'Injected smooth phase' if kind == 'smooth2d' else 'No added phase control'} | {steps} DiffPIR steps\n"
                    'Same observation / initialization for all methods; original prior-training images')
        smooth = [r for r in selected if r['phase_kind'] == 'smooth2d']
        if smooth:
            phases(smooth, args.out / f'phase_{condition}.png')
    for real_input in ['raw', 'corrected']:
        real = [r for r in rows if r['kind'] == 'real' and r.get('real_input', 'raw') == real_input]
        if real:
            title = ('Raw real SPEN | Even-only phase model is insufficient\n'
                     'Scanner correction includes odd-line and coil-specific phase'
                     if real_input == 'raw' else 'Real SPEN | Joint estimation of residual phase\n'
                     'Fixed scanner correction and coils; no high-resolution truth')
            images(real, args.out / f'real_{real_input}.png', title)
    fields = ['case', 'method', 'psnr', 'ssim', 'phase_rmse_rad', 'measurement_nrmse',
              'elapsed_seconds', 'cg_nonconverged', 'cg_max_relative_residual']
    with (args.out / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method, result in row['methods'].items():
                writer.writerow(dict(case=row['name'], method=method,
                    **{k: result.get(k, '') for k in fields[2:]}))
    groups = {}
    for row in rows:
        group = ('real/' + row.get('real_input', 'raw') if row['kind'] == 'real'
                 else row['sampling'] + '/' + row['phase_kind'])
        for method, result in row['methods'].items():
            groups.setdefault(group, {}).setdefault(method, []).append(result)
    means = {group: {method: dict(count=len(values), **{
        k: float(np.mean([v[k] for v in values])) for k in
        ('psnr', 'ssim', 'phase_rmse_rad', 'measurement_nrmse', 'elapsed_seconds') if k in values[0]})
        for method, values in methods.items()} for group, methods in groups.items()}
    (args.out / 'aggregate.json').write_text(json.dumps(means, indent=2) + '\n')
    print(json.dumps(means, indent=2))


if __name__ == '__main__':
    main()
