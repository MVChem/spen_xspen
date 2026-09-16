"""Evaluate a retrained prior against the frozen observations in the 0911 report.

Uses the original validation-selected inverse parameters, without retuning on
test images. Raw RSS / PhaseMap rows are archived baselines on those observations;
Tikhonov and the supplied diffusion checkpoint are recomputed here.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from model_v2 import load_strong_prior
from evaluate_mouse import SCANS, cases, controlled_operator, observe, subject_mean
from evaluate import metrics, unit
from prepare_data import sha256
from solvers import diffpir


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def comparison_figure(outputs, reports, out, label='Retrained diffusion'):
    labels = [('target', 'Ground truth'), ('raw_rss', 'SPEN RSS'),
              ('phase_inva', 'PhaseMap + InvA'), ('tikhonov', 'Tikhonov'),
              ('diffusion', label)]
    fig, axes = plt.subplots(5, 6, figsize=(13, 10.8))
    for col in range(6):
        key = 'fov16_R1' if col < 3 else 'fov16_R2'
        i = (3, 8, 17)[col % 3]
        record = reports[key]['records'][i]
        for row, (method, label) in enumerate(labels):
            ax = axes[row, col]
            value = outputs[key][method][i]
            if record['rot180']:
                value = np.rot90(value, 2)
            ax.imshow(value, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            if row == 0:
                ax.set_title(f"{'Full sampling' if col < 3 else 'Half sampling'} | Mouse {col % 3 + 1}", fontsize=9)
            else:
                score = reports[key]['methods'][method]['cases'][i]
                ax.set_title(f"{score['psnr']:.2f} dB / {score['ssim']:.4f}", fontsize=9)
    fig.suptitle(f'0911 SPEN simulation | {label}', fontsize=13)
    fig.tight_layout(rect=(0, .03, 1, .96))
    fig.text(.5, .013, 'Same held-out slices and complex observations. RSS / PhaseMap: archived baselines; Tikhonov / diffusion: new inference.',
             ha='center', fontsize=8)
    fig.savefig(out/'comparison.png', dpi=200)
    fig.savefig(out/'comparison.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    SCANS[16] = args.inputs/'scanner_reference/data/mat/20240321_lxj_spen_mouse_240321_1_1_1'
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Use a fresh evaluation directory')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    old = args.inputs/'legacy_evaluation'
    data = args.inputs/'mouse_mixed'
    old_config = json.loads((old/'config.json').read_text())
    checkpoint_hash = sha256(args.checkpoint)
    verification_only = checkpoint_hash == old_config['checkpoint_sha256']
    label = 'Archived checkpoint check' if verification_only else 'Retrained diffusion'
    net, checkpoint = load_strong_prior(args.checkpoint, args.device)
    manifest_hash = sha256(data/'manifest.json')
    assert checkpoint['manifest_sha256'] == manifest_hash == old_config['dataset_manifest_sha256']
    write_json(args.out/'config.json', dict(
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
        purpose='evaluator verification only' if verification_only else 'retrained model evaluation',
        checkpoint_step=checkpoint['step'], manifest_sha256=manifest_hash,
        steps=60, seed=72, selection='Original validation-selected parameters, frozen before retraining',
        torch=torch.__version__, cuda=torch.version.cuda,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        started_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        metric='Fixed range [0,1], all pixels for PSNR; Gaussian SSIM sigma=1.5, truncate=3.5, crop=5; subject-balanced mean'))
    outputs, reports, summary, rows = {}, {}, {}, []
    for acceleration, noise in ((1, .01), (2, .02)):
        key = f'fov16_R{acceleration}'
        target, records = cases(data, 'test', 16, 30, args.device)
        old_report = json.loads((old/f'{key}_metrics.json').read_text())
        assert records == old_report['records']
        with np.load(old/f'{key}.npz') as frozen:
            np.testing.assert_array_equal(unit(target), frozen['target'])
            y = torch.tensor(frozen['observation'], device=args.device)
            old_diffusion = frozen['V2 mouse prior'].copy()
        op = controlled_operator(16, acceleration, noise, args.device)
        regenerated = observe(op, target, noise, 9200 + 16 + acceleration)
        error = float((regenerated - y).abs().max())
        if error > 2e-6:
            raise ValueError(f'{key}: regenerated observation differs by {error}')
        selected = json.loads((old/f'{key}_selection.json').read_text())['selected']
        start = time.monotonic()
        with torch.no_grad():
            tik = op.proximal(torch.full_like(target, -1), y, selected['tikhonov'])
            pred, trace = diffpir(net, op, y, steps=60, sigma_noise=noise,
                                 lamb=selected['V2 mouse prior'], seed=72)
        arrays = dict(target=unit(target), observation=y.cpu().numpy(),
                      tikhonov=unit(tik), diffusion=unit(pred),
                      diffusion_model_range=pred.cpu().numpy(), legacy_diffusion=old_diffusion)
        with np.load(args.inputs/'legacy_baselines'/f'{key}.npz') as baselines:
            arrays.update({m: baselines[m].copy() for m in ('raw_rss', 'phase_inva')})
        methods = {}
        for name in ('raw_rss', 'phase_inva', 'tikhonov', 'diffusion', 'legacy_diffusion'):
            image = torch.tensor(arrays[name], device=args.device)[:, None] * 2 - 1
            scores = metrics(image, target)
            methods[name] = dict(subject_mean=subject_mean(scores, records), cases=scores)
            for i, (rec, score) in enumerate(zip(records, scores)):
                rows.append(dict(condition=key, method=name, index=i, subject=rec['subject'], key=rec['key'], **score))
        reports[key] = dict(records=records, methods=methods, selected=selected,
                            regenerated_observation_max_error=error,
                            frozen_observation_file_sha256=sha256(old/f'{key}.npz'),
                            archived_baselines_sha256=sha256(args.inputs/'legacy_baselines'/f'{key}.npz'),
                            inference_seconds=time.monotonic()-start)
        summary[key] = {m: d['subject_mean'] for m, d in methods.items()}
        outputs[key] = arrays
        np.savez_compressed(args.out/f'{key}.npz', **arrays)
        write_json(args.out/f'{key}_metrics.json', reports[key])
        write_json(args.out/f'{key}_trace.json', trace)
        print(json.dumps(dict(condition=key, **summary[key])), flush=True)
    with (args.out/'all_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(args.out/'summary.json', summary)
    comparison_figure(outputs, reports, args.out, label=label)
    lines = ['# 0911 旧权重评估器校验（非重训结果）' if verification_only else '# 0911 仿真重训结果', '',
             '每种条件 15 只留出小鼠、30 张图；先在个体内平均，再对个体等权平均。', '',
             '| 条件 | 旧 Diffusion PSNR / SSIM | 本次推理 PSNR / SSIM |',
             '| --- | --- | --- |']
    for key, scores in summary.items():
        a, b = scores['legacy_diffusion'], scores['diffusion']
        lines.append(f"| {key} | {a['psnr']:.4f} / {a['ssim']:.6f} | {b['psnr']:.4f} / {b['ssim']:.6f} |")
    lines += ['', '原复数观测、测试个体、DiffPIR 60 步及验证集选定的参数保持一致；无测试集调参。',
              '图中 RSS、PhaseMap + InvA 复用同一观测的旧基线；Tikhonov 和重训 Diffusion 为本次推理。',
              '这些是受控仿真指标，不能代表真实扫描效果。', '', '[对比图](comparison.png)', '']
    (args.out/'RESULTS.md').write_text('\n'.join(lines))
    write_json(args.out/'completed.json', dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
