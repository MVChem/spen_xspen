"""Validate a native 192 prior on controlled 96-sample SPEN inverse problems.

Input is a native-image manifest and uint16 arrays. --prior-training-data
supports a prior trained on ALL images: the old val/test partitions then only
choose calibration/report cases, all seen during prior training, with no
holdout or generalization claim. The 16 mm trajectory is a SIMULATION geometry: source images
keep their recorded physical spacing, and are placed on that assumed FOV.
No old 96 prior or enlarged 96 image is used as high-resolution ground truth.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'core'))
sys.path.insert(0, str(HERE.parent / 'prior96'))
from model_v2 import load_strong_prior
from solvers import diffpir
from evaluate import metrics, unit
from sr_operator import (SpenSuperResolutionOperator, SpenMagnitudeOperator,
                         make_sr_operator, readout_resize, readout_adjoint)

METHODS = {'tikh96_up': 'Tikhonov 96 + bicubic',
           'tikh192': 'Tikhonov 192', 'diff192': 'Diffusion 192 + DiffPIR'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def stable_seed(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16) % (2**31)


def group(record):
    return str(record.get('subject_group') or record.get('split_group') or record['subject'])


def source(record):
    return str(record['dataset'])


def select_cases(manifest, split, subjects_per_source, slices_per_subject):
    """Source-balanced fixed selection, independent of targets or predictions."""
    by_source = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(manifest['records'][split]):
        by_source[source(record)][group(record)].append((index, record))
    chosen = []
    for dataset, subjects in sorted(by_source.items()):
        ordered = sorted(subjects, key=lambda s: (stable_seed(f'{split}:{s}'), s))
        for subject in ordered[:subjects_per_source]:
            candidates = sorted(subjects[subject], key=lambda p: str(p[1]['key']))
            n = min(slices_per_subject, len(candidates))
            positions = (np.asarray([len(candidates) // 2]) if n == 1 else
                         np.rint(np.linspace(0, len(candidates) - 1, n)).astype(int))
            for position in positions:
                index, record = candidates[int(position)]
                chosen.append(dict(record, array_index=index, evaluation_split=split,
                                   evaluation_subject_group=subject))
    if not chosen:
        raise ValueError(f'No selected {split} cases')
    return chosen


def load_cases(data, selected, device):
    split = selected[0]['evaluation_split']
    array = np.load(data / f'{split}.npy', mmap_mode='r', allow_pickle=False)
    if array.dtype != np.uint16 or tuple(array.shape[1:]) != (192, 192):
        raise ValueError(f'Expected native uint16 [N,192,192], got {array.dtype} {array.shape}')
    values = np.asarray(array[[r['array_index'] for r in selected]], dtype=np.float32) / 65535.
    if not np.isfinite(values).all() or np.any(values.max(axis=(1, 2)) <= 0):
        raise ValueError('Nonfinite or empty evaluation image')
    return torch.as_tensor(values[:, None] * 2 - 1, device=device)


def audit_manifest(manifest):
    groups = {split: {group(r) for r in manifest['records'][split]}
              for split in ('train', 'val', 'test')}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if groups[a] & groups[b]:
            raise ValueError(f'{a}/{b} subject-group leakage')
        planes_a = {r['physical_plane_key'] for r in manifest['records'][a]}
        planes_b = {r['physical_plane_key'] for r in manifest['records'][b]}
        if planes_a & planes_b:
            raise ValueError(f'{a}/{b} physical-plane leakage')
    for rows in manifest['records'].values():
        for record in rows:
            transform = record['transform']
            if (transform.get('upsampled') is not False
                    or transform.get('output_shape') != [192, 192]
                    or min(transform['native_plane_shape']) < 192
                    or min(transform['sampling_yx']) < 1 - 1e-6):
                raise ValueError(f"Not an eligible native 192 reference: {record['key']}")
    return {split: len(items) for split, items in groups.items()}


def audit_prior_membership(data, manifest, prior_data, selected):
    """Check every evaluation image against the all-image prior training array."""
    prior_manifest_path = prior_data / 'manifest.json'
    prior_manifest = json.loads(prior_manifest_path.read_text())
    if any(prior_manifest['records'].get(split) for split in ('val', 'test')):
        raise ValueError('--prior-training-data expects an all-image training manifest with no holdout rows')
    expected = prior_manifest['npy_sha256']['train']
    actual = sha(prior_data / 'train.npy')
    if actual != expected:
        raise ValueError('All-image prior training array differs from its manifest')
    training = prior_manifest['records']['train']
    indexed = {r['key']: (i, r) for i, r in enumerate(training)}
    if len(indexed) != len(training):
        raise ValueError('Duplicate keys in prior training manifest')
    all_pixels = np.load(prior_data / 'train.npy', mmap_mode='r', allow_pickle=False)
    if all_pixels.dtype != np.uint16 or all_pixels.shape != (len(training), 192, 192):
        raise ValueError('Unexpected all-image training array shape or dtype')
    checked = 0
    for split in ('train', 'val', 'test'):
        pixels = np.load(data / f'{split}.npy', mmap_mode='r', allow_pickle=False)
        for i, record in enumerate(manifest['records'][split]):
            if record['key'] not in indexed:
                raise ValueError(f"Evaluation image absent from prior training: {record['key']}")
            prior_index, prior_record = indexed[record['key']]
            for field in ('physical_plane_key', 'source_plane_key', 'pixel_sha256', 'source_sha256'):
                if record[field] != prior_record[field]:
                    raise ValueError(f"Prior training provenance differs at {record['key']}: {field}")
            if not np.array_equal(pixels[i], all_pixels[prior_index]):
                raise ValueError(f"Prior training pixels differ at {record['key']}")
            checked += 1
    for split, rows in selected.items():
        for record in rows:
            record.update(seen_in_prior_training=True,
                          prior_training_array_index=indexed[record['key']][0],
                          case_role='calibration' if split == 'val' else 'report',
                          no_holdout=True)
    return dict(prior_training_manifest_sha256=sha(prior_manifest_path),
                prior_training_npy_sha256=actual, prior_training_image_count=len(training),
                evaluation_images_checked=checked, all_evaluation_images_in_prior_training=True,
                provenance_fields_verified=True, all_evaluation_pixels_equal_prior_training=True)


def averages(rows, records):
    """Report both animal equal weight and source equal weight summaries."""
    subjects = defaultdict(list)
    for row, record in zip(rows, records):
        subjects[(source(record), group(record))].append(row)
    animal = {s: {k: float(np.mean([r[k] for r in values])) for k in rows[0]}
              for s, values in subjects.items()}
    datasets = defaultdict(list)
    for (dataset, subject), values in animal.items():
        datasets[dataset].append(values)
    per_source = {s: {k: float(np.mean([r[k] for r in values])) for k in rows[0]}
                  for s, values in datasets.items()}
    return dict(subject_mean={k: float(np.mean([r[k] for r in animal.values()])) for k in rows[0]},
                source_mean={k: float(np.mean([r[k] for r in per_source.values()])) for k in rows[0]},
                per_source=per_source,
                per_subject={f'{dataset}|{subject}': values
                             for (dataset, subject), values in animal.items()})


def make_operators(args, acceleration, device):
    high = make_sr_operator(fov=args.fov, image_size=192, device=device,
                           seed=args.coil_seed, cg_max_iter=args.cg_max_iter,
                           cg_rtol=args.cg_rtol)
    low = make_sr_operator(fov=args.fov, image_size=96, device=device, seed=args.coil_seed)
    mask = torch.ones(96, dtype=torch.bool)
    if acceleration == 2:
        mask[:] = False
        generator = torch.Generator().manual_seed(args.mask_seed)
        mask[torch.randperm(96, generator=generator)[:48]] = True
    mask = mask.to(device)
    op192 = SpenSuperResolutionOperator(high.a_full, high.coils, 96, mask,
        cg_max_iter=args.cg_max_iter, cg_rtol=args.cg_rtol)
    op96 = SpenMagnitudeOperator(low.a_full, low.coils, mask)
    return op192, op96, dict(high.metadata, selected_pe_rows=torch.where(mask)[0].cpu().tolist(),
                            mask_seed=args.mask_seed, same_coils_in_both_conditions=True)


@torch.no_grad()
def observe(op, targets, records, noise):
    observed = []
    for target, record in zip(targets, records):
        y = op.forward(target[None])
        generator = torch.Generator(device=target.device).manual_seed(stable_seed(f"observe:{record['key']}"))
        # Generate noise on the original 96-row grid then select acquired rows.
        # The same physical noise realization (rescaled sigma) spans R1/R2.
        shape = (1, y.shape[1], 96, 96)
        eps = torch.randn(shape, generator=generator, device=target.device)
        eps = eps + 1j * torch.randn(shape, generator=generator, device=target.device)
        observed.append(y + noise * eps[:, :, op.mask])
    return torch.cat(observed)


@torch.no_grad()
def reconstruct(method, parameter, op192, op96, observations, records, net, args, noise):
    output, diagnostics = [], []
    for index, (y, record) in enumerate(zip(observations, records)):
        op = op96 if method == 'tikh96_up' else op192
        before = len(op192.cg_diagnostics)
        if method.startswith('tikh'):
            z = torch.full((1, 1, *op.coils.shape[-2:]), -1., device=y.device)
            prediction = op.proximal(z, y[None], parameter)
            trace = []
        else:
            prediction, trace = diffpir(net, op, y[None], steps=args.steps,
                sigma_noise=noise, lamb=parameter, sigma_max=args.sigma_max,
                sigma_min=.02, seed=stable_seed(f"diffpir:{record['key']}"))
        if method == 'tikh96_up':
            prediction = F.interpolate(prediction, size=(192, 192), mode='bicubic', align_corners=False)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f'{method} produced nonfinite output')
        output.append(prediction)
        diagnostics.append(dict(key=record['key'], method=method, parameter=parameter,
                                sampler=trace, cg=op192.cg_diagnostics[before:]))
        del op192.cg_diagnostics[before:]
    return torch.cat(output), diagnostics


def cg_summary(diagnostics):
    solves = [row for case in diagnostics for row in case['cg']]
    relative = [r for row in solves for r in row['relative_normal_residual']]
    return dict(proximal_calls=len(solves), sample_solves=len(relative),
                not_converged=sum(not v for row in solves for v in row['converged']),
                maximum_relative_normal_residual=max(relative, default=0.))


def physical_checks():
    """CPU adjoints and an independently assembled dense quadratic reference."""
    generator = torch.Generator().manual_seed(411)
    dtype = torch.complex128
    x = torch.randn(2, 3, 16, generator=generator, dtype=dtype)
    y = torch.randn(2, 3, 8, generator=generator, dtype=dtype)
    torch.testing.assert_close((readout_resize(x, 8).conj() * y).sum(),
                               (x.conj() * readout_adjoint(y, 16)).sum())
    torch.testing.assert_close(readout_resize(torch.ones_like(x), 8), torch.ones_like(y))
    a = torch.randn(3, 6, generator=generator, dtype=dtype) / 5
    c = torch.randn(2, 6, 8, generator=generator, dtype=dtype) / 3
    errors = []
    for mask in (None, torch.tensor([True, False, True])):
        op = SpenSuperResolutionOperator(a, c, 4, mask, cg_max_iter=250,
                                         cg_rtol=1e-11, cg_atol=1e-12)
        basis = torch.eye(48, dtype=torch.float64).reshape(48, 1, 6, 8)
        columns = op.linear(basis).flatten(1).T
        matrix = torch.cat((columns.real, columns.imag))
        z = torch.randn(1, 1, 6, 8, generator=generator, dtype=torch.float64)
        obs = torch.randn(op.forward(z).shape, generator=generator, dtype=dtype)
        torch.testing.assert_close((op.linear(z).conj() * obs).sum().real,
                                   (z * op.adjoint(obs)).sum())
        y0 = (obs - op.forward(torch.zeros_like(z))).flatten()
        for rho in (.03, 1e-6):
            expected = torch.linalg.solve(matrix.T @ matrix + rho * torch.eye(48),
                matrix.T @ torch.cat((y0.real, y0.imag)) + rho * z.flatten())
            actual = op.proximal(z, obs, rho).flatten()
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)
            errors.append(float((actual - expected).abs().max()))
    return dict(passed=True, readout_adjoint=True, readout_constant_intensity=True,
                real_adjoint=True, dense_proximal_full_and_half=True,
                maximum_dense_reference_error=max(errors))


def plot_comparison(folder, target, predictions, records, rows, max_columns=8,
                    seen_in_prior_training=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Keep fixed source-balanced selection order, with no result-based selection.
    count = min(max_columns, len(records))
    fig, axes = plt.subplots(4, count, figsize=(2.35 * count, 9.3), squeeze=False)
    images = dict(target=unit(target), **{k: unit(v) for k, v in predictions.items()})
    labels = dict(target='Native 192 reference', **METHODS)
    for row, (key, values) in enumerate(images.items()):
        for col in range(count):
            ax = axes[row, col]
            ax.imshow(values[col], cmap='gray', vmin=0, vmax=1, interpolation='none')
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(labels[key], fontsize=10)
            if row == 0:
                ax.set_title(f"{source(records[col])}\n{group(records[col])[-30:]}", fontsize=7)
            else:
                metric = rows[key][col]
                ax.set_title(f"{metric['psnr']:.2f} dB / SSIM {metric['ssim']:.3f}", fontsize=8)
    protocol = ('In-sample report cases: all images seen in prior training; no holdout'
                if seen_in_prior_training else
                'Same observation and [0,1] display window for every method')
    fig.suptitle('Controlled SPEN: acquired 96 × 96 → reconstructed 192 × 192\n' + protocol, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(folder / 'comparison.png', dpi=160)
    fig.savefig(folder / 'comparison.pdf')
    plt.close(fig)


def write_report(out, config, results):
    in_sample = config.get('seen_in_prior_training', False)
    protocol = ('本次 prior 使用全部图像训练，没有留出集。校准组与报告组图像均已进入 prior 训练；'
                '两组只在仿真重建参数选择和结果报告时分开。这是训练集内的开发性重建实验，不能解释为留出测试或泛化表现。'
                if in_sample else '本次为代表性留出病例的开发性仿真。')
    text = ['# 原生 192 先验：SPEN 96 → 192 仿真', '',
            f"训练检查点 step {config['checkpoint_step']}；训练 manifest SHA256 `{config['prior_training_manifest_sha256']}`。", '',
            protocol, '',
            '参考图直接读取原生 192 图像数组。没有将 96 图放大充当真值。',
            f"各来源的图像放在假设的 {config['fov']} mm SPEN 仿真视野中；该视野不是其原始扫描视野。原始像素间距、裁剪与变换保存在 cases.json。", '',
            '完整采样为 96 条 PE，复噪声实部/虚部标准差各 0.01；半采样为固定随机 48/96 条 PE，标准差各 0.02。两条件使用相同模拟线圈和相位函数，RO 均为 96 点。',
            '参数仅按校准组的来源等权 PSNR 选择；先对动物内切片平均，再对每个来源内动物等权平均，最后对来源等权平均。报告病例在推理前固定。', '',
            '幅度指标统一裁剪到 [0,1]，无逐输出亮度拟合。未截断输出、测量残差和 CG 诊断另存。输出网格增加本身不能证明真实空间分辨率翻倍。', '']
    for condition, result in results.items():
        text += [f'## {condition}', '', '| 方法 | PSNR / dB | SSIM | 测量 NRMSE | CG 未收敛 / 求解数 |',
                 '| --- | ---: | ---: | ---: | ---: |']
        for method, row in result['methods'].items():
            m, cg = row['summary']['source_mean'], row['cg']
            text.append(f"| {METHODS[method]} | {m['psnr']:.3f} | {m['ssim']:.4f} | {m['measurement_nrmse']:.4f} | {cg['not_converged']} / {cg['sample_solves']} |")
        text += ['', f'![固定报告病例对照]({condition}/comparison.png)', '']
    (out / 'RESULTS.md').write_text('\n'.join(text) + '\n')


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--prior-training-data', type=Path,
                   help='All-image prior dataset; old data val/test only select in-sample calibration/report cases')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--fov', type=int, choices=[16, 24], default=16)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--sigma-max', type=float, default=2.)
    p.add_argument('--val-subjects-per-source', type=int, default=1)
    p.add_argument('--test-subjects-per-source', type=int, default=2)
    p.add_argument('--slices-per-subject', type=int, default=1)
    p.add_argument('--lambdas', nargs='+', type=float, default=[.1, .3, 1.])
    p.add_argument('--rhos', nargs='+', type=float, default=[.0003, .001, .003])
    p.add_argument('--coil-seed', type=int, default=4517)
    p.add_argument('--mask-seed', type=int, default=20260914)
    p.add_argument('--cg-max-iter', type=int, default=320)
    p.add_argument('--cg-rtol', type=float, default=1e-5)
    p.add_argument('--dry-run', action='store_true', help='CPU physical checks and actual-data forward/baseline smoke; no model required')
    args = p.parse_args()
    if min(args.val_subjects_per_source, args.test_subjects_per_source, args.slices_per_subject) < 1:
        raise ValueError('Case counts must be positive')
    if not args.dry_run and args.checkpoint is None:
        p.error('--checkpoint is required except for --dry-run')
    if args.dry_run:
        args.device = 'cpu'
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.data / 'manifest.json').read_text())
    counts = audit_manifest(manifest)
    array_hashes = {split: sha(args.data / f'{split}.npy') for split in ('train', 'val', 'test')}
    if array_hashes != manifest['npy_sha256']:
        raise ValueError('Saved array hashes differ from the frozen training manifest')
    records = {split: select_cases(manifest, split, getattr(args, f'{split}_subjects_per_source'),
                                   args.slices_per_subject) for split in ('val', 'test')}
    in_sample = args.prior_training_data is not None
    prior_audit = (audit_prior_membership(args.data, manifest, args.prior_training_data, records)
                   if in_sample else dict(prior_training_manifest_sha256=sha(args.data / 'manifest.json')))
    args.out.mkdir(parents=True, exist_ok=True)
    # A fresh directory prevents mixing a changed checkpoint, geometry or sample set.
    if any(args.out.iterdir()):
        raise FileExistsError(f'Use a fresh evaluation output directory: {args.out}')
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(manifest_sha256=sha(args.data / 'manifest.json'), subject_counts=counts,
                  verified_npy_sha256=array_hashes,
                  **prior_audit,
                  seen_in_prior_training=in_sample, no_holdout=in_sample,
                  evaluation_protocol=('in_sample_developmental_reconstruction' if in_sample else
                                       'held_out_developmental_reconstruction'),
                  case_roles={'val': 'calibration', 'test': 'report'},
                  role_note=('Legacy val/test keys select calibration/report groups only. All these images were included in prior training; there is no prior holdout.'
                             if in_sample else 'Calibration and report cases are excluded from prior training.'),
                  source_sha256={str(path): sha(path) for path in
                      (Path(__file__), HERE / 'sr_operator.py', HERE.parent / 'core/solvers.py')},
                  source_fov_policy='Recorded source images placed on assumed simulation FOV; no claim that this equals native acquisition FOV',
                  calibration_criterion='source-equal mean of subject-equal PSNR',
                  selection_policy='Stable hash ordering of subject groups within each source, evenly spaced ordered slices; no image-quality selection',
                  precision='FP32 EDM coordinates and physics; StrongPrior internal CUDA BF16 autocast')
    save_json(args.out / 'cases.json', records)
    save_json(args.out / 'physical_checks.json', physical_checks())
    net = None
    if not args.dry_run:
        net, checkpoint = load_strong_prior(args.checkpoint, args.device)
        if checkpoint.get('manifest_sha256') != config['prior_training_manifest_sha256']:
            raise ValueError('Checkpoint and prior training manifest differ')
        if checkpoint.get('img_resolution', 192) != 192:
            raise ValueError('Expected a 192-resolution prior')
        config.update(checkpoint_sha256=sha(args.checkpoint), checkpoint_step=checkpoint['step'])
    save_json(args.out / 'config.json', config)
    values = {split: load_cases(args.data, items[:1] if args.dry_run else items, args.device)
              for split, items in records.items()}
    results = {}
    for acceleration, noise in ((1, .01), (2, .02)):
        name = f'R{acceleration}'
        folder = args.out / name
        folder.mkdir()
        op192, op96, metadata = make_operators(args, acceleration, args.device)
        save_json(folder / 'operator.json', dict(metadata, noise_per_real_imag_component=noise))
        if args.dry_run:
            y = observe(op192, values['val'], records['val'][:1], noise)
            pred, diag = reconstruct('tikh192', .003, op192, op96, y, records['val'][:1], None, args, noise)
            save_json(folder / 'smoke.json', dict(observation_shape=list(y.shape),
                image_shape=list(pred.shape), finite=bool(torch.isfinite(pred).all()),
                measurement_nrmse=op192.relative_residual(pred, y).cpu().tolist(), cg=cg_summary(diag)))
            continue
        observed = {split: observe(op192, images, records[split], noise)
                    for split, images in values.items()}
        selections, result, predictions, case_metrics = {}, dict(methods={},
            seen_in_prior_training=in_sample, no_holdout=in_sample, case_role='report',
            evaluation_protocol=config['evaluation_protocol']), {}, {}
        for method in METHODS:
            trials = []
            for parameter in args.lambdas if method == 'diff192' else args.rhos:
                started = time.monotonic()
                prediction, diagnostics = reconstruct(method, parameter, op192, op96, observed['val'],
                                                       records['val'], net, args, noise)
                summary = averages(metrics(prediction, values['val']), records['val'])
                trial = dict(parameter=parameter, summary=summary, cg=cg_summary(diagnostics),
                             seconds=time.monotonic() - started)
                trials.append(trial)
                save_json(folder / f'calibration_{method}_{parameter:g}_trace.json', diagnostics)
                print(json.dumps(dict(event='sr_calibration', condition=name, method=method,
                                      parameter=parameter, **summary['source_mean'])), flush=True)
            best = max(trials, key=lambda trial: trial['summary']['source_mean']['psnr'])
            selections[method] = dict(best=best, trials=trials, case_role='calibration',
                                      seen_in_prior_training=in_sample, no_holdout=in_sample)
            save_json(folder / 'selection.json', selections)
            prediction, diagnostics = reconstruct(method, best['parameter'], op192, op96, observed['test'],
                                                   records['test'], net, args, noise)
            rows = metrics(prediction, values['test'])
            residual = op192.relative_residual(prediction, observed['test']).cpu().tolist()
            clipped = op192.relative_residual(prediction.clamp(-1, 1), observed['test']).cpu().tolist()
            for row, r, c, image in zip(rows, residual, clipped, prediction):
                row.update(measurement_nrmse=r, displayed_measurement_nrmse=c,
                           outside_range_fraction=float(((image < -1) | (image > 1)).float().mean()))
            result['methods'][method] = dict(selected_parameter=best['parameter'], cases=rows,
                summary=averages(rows, records['test']), cg=cg_summary(diagnostics))
            predictions[method], case_metrics[method] = prediction, rows
            save_json(folder / f'report_{method}_trace.json', diagnostics)
            save_json(folder / 'metrics.json', result)
            print(json.dumps(dict(event='sr_report', condition=name, method=method,
                                  **result['methods'][method]['summary']['source_mean'])), flush=True)
        np.savez_compressed(folder / 'reconstructions.npz', target=unit(values['test']),
                            observation=observed['test'].cpu().numpy(),
                            **{key: unit(value) for key, value in predictions.items()},
                            **{key + '_unclipped': value.cpu().numpy()[:, 0] for key, value in predictions.items()})
        plot_comparison(folder, values['test'], predictions, records['test'], case_metrics,
                        seen_in_prior_training=in_sample)
        results[name] = result
        write_report(args.out, config, results)
    save_json(args.out / ('dry_run_completed.json' if args.dry_run else 'completed.json'),
              dict(completed=True, dry_run=args.dry_run, conditions=['R1', 'R2'],
                   seen_in_prior_training=in_sample, no_holdout=in_sample))


if __name__ == '__main__':
    main()
