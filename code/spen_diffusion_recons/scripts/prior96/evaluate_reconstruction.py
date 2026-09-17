"""Evaluate a trained prior on frozen simulation and real scanner observations.

Uses the original validation-selected inverse parameters, without retuning on
test images. Simulation Raw RSS / PhaseMap rows are archived baselines on those
observations; real-data baselines and the supplied prior are recomputed here.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io
import torch

from model_v2 import load_strong_prior
from evaluate_mouse import SCANS, cases, controlled_operator, observe, subject_mean, real_case
from evaluate import metrics, unit
from prepare_data import sha256
from solvers import diffpir
from operators import scanner_matrices
from render_comparison import render_simulation, render_real


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


@torch.no_grad()
def evaluate_real(net, checkpoint, args):
    """Selected real acquisitions, using the unchanged native-96 scanner model."""
    original = json.loads((args.inputs/'legacy_evaluation/real_mouse_metrics.json').read_text())
    archived = {(case['fov_mm'], Path(case['path']).name): case for case in original['cases']}
    scans = {fov: args.scan_root/path.name for fov, path in SCANS.items()}
    selections = [(fov, scans[fov]/f'slice_{index}.mat')
                  for fov, indices in ((16, args.real16_ids), (24, args.real24_ids))
                  for index in indices]
    for _, path in selections:
        if not path.is_file():
            raise FileNotFoundError(path)
    cached = {}
    if args.reuse_real:
        saved = json.loads((args.reuse_real/'real.json').read_text())
        expected = dict(checkpoint_sha256=sha256(args.checkpoint), steps=60, seed=73,
                        rho=.003, lamb=1., sigma_noise=.02)
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(f'Real cache has incompatible {key}')
        with np.load(args.reuse_real/'real.npz', allow_pickle=False) as a:
            if len(saved['cases']) != len(a['labels']):
                raise ValueError('Real cache metadata and labels differ in length')
            for method in ('raw_rss', 'phase_inva', 'tikhonov', 'diffusion'):
                if a[method].shape != (len(saved['cases']), 96, 96) or not np.isfinite(a[method]).all():
                    raise ValueError(f'Invalid cached real images: {method}')
            for i, row in enumerate(saved['cases']):
                if a['fov_mm'][i] != row['fov_mm'] or a['labels'][i] != f"Acquisition #{row['export_index']}":
                    raise ValueError('Real cache order does not match metadata')
                identity = (row['fov_mm'], f"slice_{row['export_index']}.mat")
                if identity in cached:
                    raise ValueError(f'Duplicate cached acquisition: {identity}')
                cached[identity] = (row, {k: a[k][i].copy() for k in
                                          ('raw_rss', 'phase_inva', 'tikhonov', 'diffusion')})
    arrays = {k: [] for k in ('raw_rss', 'phase_inva', 'tikhonov', 'diffusion')}
    rows = []
    for fov, path in selections:
        case = archived.get((fov, path.name))
        if case is not None and sha256(path) != case['sha256']:
            raise ValueError(f'Scanner input differs from archived case: {path}')
        key = f'real_fov{fov}_{path.stem}'
        if (fov, path.name) in cached:
            meta, images = cached[(fov, path.name)]
            if sha256(path) != meta['sha256']:
                raise ValueError(f'Scanner input differs from cached case: {path}')
            with np.load(args.reuse_real/f'{key}.npz', allow_pickle=False) as raw:
                for method in ('phase_inva', 'tikhonov', 'diffusion'):
                    display = np.rot90(((raw[method][0, 0]+1)/2).clip(0, 1), 2)
                    np.testing.assert_array_equal(display, images[method])
            for suffix in ('.npz', '_trace.json'):
                shutil.copyfile(args.reuse_real/f'{key}{suffix}', args.out/f'{key}{suffix}')
            for method in arrays:
                arrays[method].append(images[method])
            rows.append(dict(meta, path=str(path), reused_from=str(args.reuse_real.resolve())))
            print(json.dumps(dict(event='real_case_reused', case=key)), flush=True)
            continue
        op, y, anchor, meta = real_case(path, args.device)
        observation_error = None
        if case is not None:
            with np.load(args.inputs/'legacy_evaluation'/f'{key}.npz') as frozen:
                observation_error = float(np.max(np.abs(y.cpu().numpy()-frozen['observation'])))
            if observation_error > 2e-6:
                raise ValueError(f'Real observation changed: {key}: {observation_error}')
        predictions = {'phase_inva': anchor,
                       'tikhonov': op.proximal(torch.full_like(anchor, -1), y, .003)}
        predictions['diffusion'], trace = diffpir(net, op, y, steps=60,
            sigma_noise=.02, lamb=1., seed=73)
        raw = np.asarray(scipy.io.loadmat(path, variable_names=['spen_original_signal_rofft'])
                         ['spen_original_signal_rofft'])
        if raw.shape != (96, 96, 1, 4):
            raise ValueError(f'Unexpected original signal axes: {raw.shape}')
        signal = torch.tensor(raw[:, :, 0].transpose(2, 0, 1),
                              dtype=torch.complex64, device=args.device)
        _, encoding, _ = scanner_matrices(path, args.device)
        smax = float(torch.linalg.svdvals(encoding).max())
        response = op.forward(torch.ones_like(anchor))
        raw_scale = float(response.abs().square().sum(1).sqrt()[0, 8:-8, 8:-8].median())
        if raw_scale <= 0:
            raise ValueError('Nonpositive unit-object response')
        raw_rss = signal.abs().square().sum(0).sqrt()/(meta['magnitude_scale']*smax*raw_scale)
        arrays['raw_rss'].append(np.rot90(raw_rss.cpu().numpy(), 2))
        unscaled = {}
        meta.update(fov_mm=fov, export_index=int(path.stem.split('_')[-1]),
                    original_observation_max_error=observation_error,
                    encoding_smax=smax, raw_rss_calibration_scale=raw_scale, methods={})
        for method, value in predictions.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f'{key}: {method}')
            arrays[method].append(np.rot90(unit(value)[0], 2))
            unscaled[method] = value.cpu().numpy()
            meta['methods'][method] = dict(
                measurement_nrmse=float(op.relative_residual(value, y)),
                displayed_measurement_nrmse=float(op.relative_residual(value.clamp(-1, 1), y)))
        np.savez_compressed(args.out/f'{key}.npz', observation=y.cpu().numpy(),
                            **unscaled)
        write_json(args.out/f'{key}_trace.json', trace)
        rows.append(meta)
        print(json.dumps(dict(event='real_case_complete', case=key)), flush=True)
    payload = {k: np.stack(v) for k, v in arrays.items()}
    payload.update(labels=np.asarray([f"Acquisition #{r['export_index']}" for r in rows]),
                   fov_mm=np.asarray([r['fov_mm'] for r in rows]))
    np.savez_compressed(args.out/'real.npz', **payload)
    write_json(args.out/'real.json', dict(cases=rows, checkpoint_step=checkpoint['step'],
        checkpoint_sha256=sha256(args.checkpoint), steps=60, seed=73,
        reused_cases=sum('reused_from' in r for r in rows),
        fresh_inference_cases=sum('reused_from' not in r for r in rows),
        rho=.003, lamb=1., sigma_noise=.02,
        phase='Existing MAT scanner phase correction; weighted InvA and measurement-derived gain',
        display='All rows rotate 180 degrees, fixed [0,1] window; native 96x96, no interpolation',
        input='Original RO-FFT coil RSS, calibrated by unit-object response and the same measurement scale',
        scope='Real acquired SPEN; no paired GT and no PSNR/SSIM'))
    render_real(payload, args.out)
    case_description = '；'.join(
        f"FOV {fov} mm：导出编号 " + '、'.join(str(r['export_index']) for r in rows if r['fov_mm'] == fov)
        for fov in (16, 24))
    (args.out/'RESULTS.md').write_text(
        '# SPEN 真实采集重建\n\n'
        f"使用指定权重（第 {checkpoint['step']:,} 步 EMA），重建选定的 {len(rows)} 个真实采集案例。\n\n"
        f"其中 {sum('reused_from' in r for r in rows)} 例复用同权重、同参数和同 MAT 的保存结果，其余重新推理。\n\n"
        f'{case_description}。'
        '各行依次为原始输入 RSS、Tikhonov、Phase map + InvA、Diffusion prior；'
        '全部为原生 96×96，统一显示窗 [0,1]，显示时旋转 180°。\n\n'
        'DiffPIR 为 60 步、λ=1、σ=0.02；Tikhonov ρ=0.003。'
        '沿用原 MAT 相位校正、线圈/相位估计及幅度标度。'
        '没有配对干净真值，不报告 PSNR/SSIM；测量残差见 real.json。\n\n'
        '[真实数据对比图](real_comparison.png)\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--mode', choices=['simulation', 'real'], default='simulation')
    p.add_argument('--scan-root', type=Path,
                   default=Path(__file__).resolve().parents[3]/'data/spen_acquired_260915/mat')
    p.add_argument('--real16-ids', type=int, nargs='+', default=[5, 13, 22, 30, 38],
                   help='FOV 16 mm MAT export numbers; default: 5 13 22 30 38')
    p.add_argument('--real24-ids', type=int, nargs='+', default=[3, 7, 11, 15, 19],
                   help='FOV 24 mm MAT export numbers; default: 3 7 11 15 19')
    p.add_argument('--reuse-real', type=Path,
                   help='Reuse matching cases from a saved real evaluation; infer only missing cases')
    args = p.parse_args()
    for indices in (args.real16_ids, args.real24_ids):
        if len(set(indices)) != len(indices) or any(i < 1 for i in indices):
            p.error('Real acquisition numbers must be positive and unique within each FOV')
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
    label = 'Archived diffusion' if verification_only else 'Diffusion prior'
    net, checkpoint = load_strong_prior(args.checkpoint, args.device)
    manifest_hash = sha256(data/'manifest.json')
    assert checkpoint['manifest_sha256'] == manifest_hash == old_config['dataset_manifest_sha256']
    write_json(args.out/'config.json', dict(
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
        purpose='evaluator verification only' if verification_only else 'retrained model evaluation',
        checkpoint_step=checkpoint['step'], manifest_sha256=manifest_hash,
        mode=args.mode, steps=60, seed=72 if args.mode=='simulation' else 73,
        real_case_selection={'16': args.real16_ids, '24': args.real24_ids} if args.mode=='real' else None,
        reuse_real=str(args.reuse_real.resolve()) if args.reuse_real else None,
        selection='Original validation-selected simulation parameters; fixed real-data parameters',
        torch=torch.__version__, cuda=torch.version.cuda,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        started_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        metric=('Fixed range [0,1], all pixels for PSNR; Gaussian SSIM sigma=1.5, truncate=3.5, crop=5; subject-balanced mean'
                if args.mode=='simulation' else 'No paired GT; measurement residual only')))
    if args.mode=='real':
        evaluate_real(net, checkpoint, args)
        write_json(args.out/'completed.json', dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat()))
        return
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
    render_simulation(outputs, reports, args.out, label=label)
    lines = ['# 旧权重评估器校验（非重训结果）' if verification_only else '# SPEN 仿真重训结果', '',
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
