"""Reconstruct the final pixel192 prior for simulation and real-data figures.

Simulation uses matched native192 images, full PE / random half PE, and the
original measured-data PhaseMap + InvA baseline. Outputs stay separate from
the shared training inputs. Rendering can be repeated from the saved NPZs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT.parent / 'spenpy'))
sys.path.insert(0, str(HERE.parent / 'core'))
sys.path.insert(0, str(HERE.parent / 'prior96'))
from model_v2 import load_strong_prior
from evaluate import metrics, unit
from solvers import diffpir
from evaluate_png_sr import physical_checks, observe, stable_seed, sha, save_json
from sr_operator import make_sr_operator, SpenSuperResolutionOperator, SCANS
from phase_inva import load_phase_matrices, phase_inva
from render_rebuilt import render_simulation, render_real

DEFAULT_RUN = PROJECT / 'runs/rodent192_spen2x_260914'
SCAN_NAMES = {16: '20240321_lxj_spen_mouse_240321_1_1_1',
              24: '20240115_lxj_SPEN_96_240115_1_1_1'}
REAL_SELECTIONS = [(16, n) for n in (5, 13, 22, 30, 38)] + [(24, n) for n in (3, 7, 11, 15, 19)]


def announce(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def first_per_source(records):
    selected = {}
    for record in records:
        selected.setdefault(record['dataset'], record)
    return list(selected.values())


def read_selected(args, net_checkpoint):
    manifest = json.loads((args.data / 'manifest.json').read_text())
    assert sha(args.data / 'manifest.json') == net_checkpoint['manifest_sha256']
    assert sha(args.data / 'train.npy') == manifest['npy_sha256']['train']
    index = {row['key']: (i, row) for i, row in enumerate(manifest['records']['train'])}
    array = np.load(args.data / 'train.npy', mmap_mode='r', allow_pickle=False)
    assert array.dtype == np.uint16 and array.shape == (len(index), 192, 192)
    old = json.loads(args.cases.read_text())
    selected = {role: first_per_source(old[part])
                for role, part in [('calibration', 'val'), ('report', 'test')]}
    assert not ({r['key'] for r in selected['calibration']} &
                {r['key'] for r in selected['report']})
    tensors = {}
    for role, rows in selected.items():
        for row in rows:
            i, original = index[row['key']]
            assert row['pixel_sha256'] == original['pixel_sha256']
            row['prior_training_array_index'] = i
        values = np.asarray(array[[index[r['key']][0] for r in rows]], np.float32) / 65535.
        tensors[role] = torch.from_numpy(2 * values[:, None] - 1).to(args.device)
    return selected, tensors


def predict(net, op, observations, records, method, parameter, args, noise):
    outputs, traces = [], []
    for y, record in zip(observations, records):
        op.cg_diagnostics.clear()
        with torch.no_grad():
            if method == 'tikhonov':
                x = op.proximal(torch.full((1, 1, 192, 192), -1., device=args.device),
                                y[None], parameter)
                trace = []
            else:
                x, trace = diffpir(net, op, y[None], steps=args.steps,
                    sigma_noise=noise, lamb=parameter, sigma_max=2.,
                    seed=stable_seed('rebuilt_diffpir:' + record['key']))
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError(method)
        outputs.append(x)
        traces.append(dict(key=record['key'], sampler=trace, cg=list(op.cg_diagnostics)))
    return torch.cat(outputs), traces


def score(prediction, target):
    return float(np.mean([row['psnr'] for row in metrics(prediction, target)]))


def simulation(args, net, ckpt):
    selected, targets = read_selected(args, ckpt)
    save_json(args.out / 'simulation_cases.json', selected)
    high = make_sr_operator(fov=16, image_size=192, device=args.device, seed=4517,
                            cg_max_iter=320)
    with torch.no_grad():
        response = high.forward(torch.ones((1, 1, 192, 192), device=args.device))
        raw_scale = float(response.abs().square().sum(1).sqrt()[0, 8:-8, 8:-8].median())
    assert raw_scale > 0
    inv, odd_inv, even_inv = load_phase_matrices(SCANS[16] / 'slice_7.mat', args.device)
    masks = [torch.ones(96, dtype=torch.bool), torch.zeros(96, dtype=torch.bool)]
    generator = torch.Generator().manual_seed(args.mask_seed)
    masks[1][torch.randperm(96, generator=generator)[:48]] = True
    arrays = {k: [] for k in ['degraded', 'phase_inva', 'tikhonov', 'diffusion',
        'phase_inva_unclipped', 'tikhonov_unclipped', 'diffusion_unclipped',
        'observation', 'phase', 'corrected_observation', 'receiver_gains']}
    metadata = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        checkpoint_step=ckpt['step'], physical_checks=physical_checks(), steps=args.steps,
        mask_seed=args.mask_seed, source_sha256={name: sha(HERE / name) for name in
            ['rebuild_figures.py', 'phase_inva.py', 'render_rebuilt.py']},
        geometry=high.metadata, conditions=[],
        reconstruction_grid=[192, 192], acquisition_grid=[96, 96],
        input_display='Zero-filled acquired-grid RSS / unit-object calibration; same scalar in both conditions',
        raw_rss_scale=raw_scale, phase_steps=args.phase_steps,
        phase_baseline='Measured-data odd/even phase fit + legacy weighted InvA at96; per-coil gain fitted on acquired rows; bicubic to192',
        tikhonov='192-grid min_x ||forward(x)-y||^2 + rho||x+1||^2; x=2*m-1',
        noise='Independent Gaussian real/imaginary components; paired underlying full-grid noise across conditions',
        phase_error='No extra odd/even phase error injected, matching the reference simulation',
        calibration='First pre-existing calibration case per source; select by source-equal PSNR before reconstructing report cases',
        sampling='Exactly 48/96 entire PE rows; all RO samples and coils share the mask',
        sample_scope='All calibration and report images participated in prior training; no held-out performance claim')
    for condition, (mask_cpu, noise) in enumerate(zip(masks, args.noise)):
        mask = mask_cpu.to(args.device)
        op = SpenSuperResolutionOperator(high.a_full, high.coils, 96, mask,
            sigma_noise=noise, cg_max_iter=320)
        with torch.no_grad():
            obs = {role: observe(op, target, selected[role], noise)
                   for role, target in targets.items()}
        trials, chosen = {}, {}
        for method, candidates in [('tikhonov', args.rhos), ('diffusion', args.lambdas)]:
            trials[method] = []
            for value in candidates:
                prediction, diagnostics = predict(net, op, obs['calibration'], selected['calibration'],
                                                   method, value, args, noise)
                value_score = score(prediction, targets['calibration'])
                solves = [solve for case in diagnostics for solve in case['cg']]
                trials[method].append(dict(parameter=value, mean_psnr=value_score,
                    cg_unconverged=sum(not ok for solve in solves for ok in solve['converged']),
                    cg_max_relative_residual=max(v for solve in solves for v in solve['relative_normal_residual'])))
                announce(stage='calibration', condition=condition, method=method,
                         parameter=value, psnr=value_score)
            chosen[method] = max(trials[method], key=lambda r: r['mean_psnr'])['parameter']
        condition_meta = dict(noise_sigma=noise, acquired_pe=int(mask.sum()),
            selected_pe_rows=torch.where(mask_cpu)[0].tolist(), trials=trials,
            selected=chosen, methods={}, phase_checks=[])
        for method in ['tikhonov', 'diffusion']:
            pred, diagnostics = predict(net, op, obs['report'], selected['report'],
                                       method, chosen[method], args, noise)
            arrays[method].append(unit(pred))
            arrays[method + '_unclipped'].append((pred[:, 0].cpu().numpy() + 1) / 2)
            condition_meta['methods'][method] = dict(metrics=metrics(pred, targets['report']),
                measurement_nrmse=op.relative_residual(pred, obs['report']).detach().cpu().tolist(),
                diagnostics=diagnostics)
        raw, classic, phases, corrected, gains = [], [], [], [], []
        for y, record in zip(obs['report'], selected['report']):
            mag, phase, corrected_y, _, gain, error = phase_inva(y, mask, high.native_encoding,
                inv, odd_inv, even_inv, stable_seed('rebuilt_phase:' + record['key']),
                steps=args.phase_steps)
            assert not bool(corrected_y[:, ~mask].abs().any())
            native = torch.zeros((96, 96), device=args.device)
            native[mask] = y.abs().square().sum(0).sqrt() / raw_scale
            up = F.interpolate(mag[None, None], size=(192, 192), mode='bicubic',
                               align_corners=False)[0, 0]
            raw.append(native.cpu().numpy()); classic.append(up.cpu().numpy())
            phases.append(phase.cpu().numpy()); corrected.append(corrected_y.cpu().numpy())
            gains.append(gain.cpu().numpy())
            condition_meta['phase_checks'].append(dict(key=record['key'],
                amplitude_preservation_error=error, phase_rms_rad=float(phase.square().mean().sqrt())))
        arrays['degraded'].append(np.stack(raw))
        arrays['phase_inva_unclipped'].append(np.stack(classic))
        arrays['phase_inva'].append(np.stack(classic).clip(0, 1))
        arrays['phase'].append(np.stack(phases))
        arrays['corrected_observation'].append(np.stack(corrected))
        arrays['receiver_gains'].append(np.stack(gains))
        phase_pred = torch.as_tensor(np.stack(classic), device=args.device)[:, None] * 2 - 1
        condition_meta['methods']['phase_inva'] = dict(metrics=metrics(phase_pred, targets['report']))
        full_observation = obs['report'].new_zeros((len(selected['report']), 4, 96, 96))
        full_observation[:, :, mask] = obs['report']
        arrays['observation'].append(full_observation.cpu().numpy())
        metadata['conditions'].append(condition_meta)
        save_json(args.out / 'simulation.json', metadata)
        announce(stage='simulation_condition_done', condition=condition, chosen=chosen)
    payload = {k: np.stack(v) for k, v in arrays.items()}
    payload.update(target=unit(targets['report']), mask=np.stack([m.numpy() for m in masks]),
        noise_sigma=np.asarray(args.noise),
        labels=np.asarray([r['dataset'] for r in selected['report']]),
        case_keys=np.asarray([r['key'] for r in selected['report']]))
    assert payload['degraded'].shape == (2, len(selected['report']), 96, 96)
    assert all(np.isfinite(a).all() for a in payload.values() if a.dtype.kind not in 'US')
    np.savez_compressed(args.out / 'simulation.npz', **payload)
    render_simulation(args.out / 'simulation.npz', args.out / 'figure1_simulation')


def real(args, net, ckpt):
    from evaluate_real_sr import load_case
    selections = REAL_SELECTIONS
    arrays = {k: [] for k in ['degraded', 'phase_inva', 'tikhonov', 'diffusion',
                             'phase_inva_unclipped', 'tikhonov_unclipped', 'diffusion_unclipped']}
    cached, previous_arrays, reused = {}, {}, []
    checkpoint_hash = sha(args.checkpoint)
    if args.extend_real:
        previous = json.loads((args.out / 'real.json').read_text())
        expected = dict(checkpoint_sha256=checkpoint_hash, checkpoint_step=ckpt['step'],
                        steps=args.steps, rho=.003, lamb=1., noise_sigma=.02)
        if any(previous.get(k) != value for k, value in expected.items()):
            raise ValueError('Existing real results use different weights or reconstruction parameters')
        with np.load(args.out / 'real.npz', allow_pickle=False) as saved:
            previous_arrays = {key: saved[key].copy() for key in arrays}
            for i, case in enumerate(previous['cases']):
                identity = (case['fov_mm'], case['export_index'])
                if identity in cached or identity not in selections:
                    raise ValueError(f'Existing case outside the fixed selection or duplicated: {identity}')
                if saved['fov_mm'][i] != identity[0] or saved['labels'][i] != f'Acquisition #{identity[1]}':
                    raise ValueError('Existing real arrays and case metadata disagree')
                for key, values in previous_arrays.items():
                    size = 96 if key == 'degraded' else 192
                    if values.shape != (len(previous['cases']), size, size) or not np.isfinite(values).all():
                        raise ValueError(f'Invalid saved real array: {key}')
                case.setdefault('reconstruction_source_sha256', previous['source_sha256'])
                cached[identity] = (i, case)
    rows = []
    for fov, number in selections:
        path = SCANS[fov] / f'slice_{number}.mat'
        if (fov, number) in cached:
            index, meta = cached[(fov, number)]
            if meta['source_sha256'] != sha(path):
                raise ValueError(f'Acquisition changed since reconstruction: {path}')
            for key in arrays:
                arrays[key].append(previous_arrays[key][index])
            rows.append(meta)
            reused.append(dict(fov_mm=fov, export_index=number))
            announce(stage='real_case_reused', fov=fov, export_index=number)
            continue
        with torch.no_grad():
            op96, op, y, anchor, meta = load_case(path, args.device)
            op.cg_max_iter = 320
            predictions = {'phase_inva': F.interpolate(anchor, (192, 192), mode='bicubic', align_corners=False)}
            predictions['tikhonov'] = op.proximal(torch.full_like(predictions['phase_inva'], -1), y, .003)
            predictions['diffusion'], sampler = diffpir(net, op, y, steps=args.steps,
                sigma_noise=.02, lamb=1., sigma_max=2., seed=73)
            raw = np.asarray(scipy.io.loadmat(path,
                variable_names=['spen_original_signal_rofft'])['spen_original_signal_rofft'])
            assert raw.shape == (96, 96, 1, 4)
            original = torch.as_tensor(raw[:, :, 0].transpose(2, 0, 1),
                                       dtype=torch.complex64, device=args.device)
            response = op96.forward(torch.ones_like(anchor))
            raw_scale = float(response.abs().square().sum(1).sqrt()[0, 8:-8, 8:-8].median())
            input_image = original.abs().square().sum(0).sqrt() / (
                meta['magnitude_scale'] * meta['source_encoding_smax'] * raw_scale)
            arrays['degraded'].append(np.rot90(input_image.cpu().numpy(), 2))
            meta.update(fov_mm=fov, export_index=number, raw_rss_calibration_scale=raw_scale,
                reconstruction_source_sha256=sha(Path(__file__)),
                phase_baseline='MAT already contains scanner phase correction; apply matched weighted InvA and measurement-derived gain',
                sampler=sampler, cg=list(op.cg_diagnostics), methods={})
            for name, x in predictions.items():
                if not bool(torch.isfinite(x).all()):
                    raise FloatingPointError(name)
                arrays[name].append(np.rot90(unit(x)[0], 2))
                arrays[name + '_unclipped'].append(np.rot90((x[0, 0].cpu().numpy() + 1) / 2, 2))
                meta['methods'][name] = dict(measurement_nrmse=float(op.relative_residual(x, y)))
            rows.append(meta)
        announce(stage='real_case_done', fov=fov, export_index=number)
    payload = {k: np.stack(v) for k, v in arrays.items()}
    payload.update(labels=np.asarray([f'Acquisition #{number}' for _, number in selections]),
                   fov_mm=np.asarray([fov for fov, _ in selections]))
    temporary = args.out / 'real.pending.npz'
    np.savez_compressed(temporary, **payload)
    temporary.replace(args.out / 'real.npz')
    save_json(args.out / 'real.json', dict(cases=rows, checkpoint_step=ckpt['step'],
        checkpoint=str(args.checkpoint), source_sha256=sha(Path(__file__)),
        checkpoint_sha256=checkpoint_hash, rho=.003, lamb=1., noise_sigma=.02,
        steps=args.steps, reused_cases=reused,
        case_selection='Five fixed export positions per FOV:10/30/50/70/90 percent; retained original six and added four, no score-based selection',
        scope='Real acquired data; no paired HR ground truth and no PSNR/SSIM',
        display='Rotate all rows180; common[0,1] magnitude window; input uses unit-object response calibration'))
    render_real(args.out / 'real.npz', args.out / 'figure2_real')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['simulation', 'real', 'all'], default='simulation')
    p.add_argument('--extend-real', action='store_true',
                   help='Reuse existing real arrays and reconstruct only missing fixed cases')
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_RUN / 'train/model_ema.pt')
    p.add_argument('--data', type=Path, default=DEFAULT_RUN / 'data_all')
    p.add_argument('--cases', type=Path, default=DEFAULT_RUN / 'simulation/cases.json')
    p.add_argument('--scan-root', type=Path, default=PROJECT.parent / 'data/spen_acquired_260915/mat')
    p.add_argument('--out', type=Path, default=DEFAULT_RUN / 'figures_260915')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--phase-steps', type=int, default=300)
    p.add_argument('--noise', type=float, nargs=2, default=[.01, .02])
    p.add_argument('--rhos', type=float, nargs='+', default=[.000003, .00001, .00003, .0001, .0003, .001, .003, .01])
    p.add_argument('--lambdas', type=float, nargs='+', default=[.03, .1, .3, 1., 3.])
    p.add_argument('--mask-seed', type=int, default=20260914)
    args = p.parse_args()
    if args.extend_real and args.mode != 'real':
        p.error('--extend-real requires --mode real')
    if args.extend_real and not all((args.out / name).exists() for name in ('real.json', 'real.npz')):
        p.error('--extend-real requires existing real.json and real.npz')
    if args.steps < 2 or args.phase_steps < 1 or not 0 < args.noise[0] < args.noise[1]:
        p.error('Require steps>=2, phase-steps>=1 and 0<left noise<right noise')
    if any(value <= 0 for value in args.rhos + args.lambdas):
        p.error('Regularization candidates must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    for mode in (['simulation', 'real'] if args.mode == 'all' else [args.mode]):
        if (args.out / f'{mode}.npz').exists():
            if not (mode == 'real' and args.extend_real):
                raise FileExistsError('Use a fresh output directory; use render_rebuilt.py to redraw saved arrays')
    SCANS.update({fov: args.scan_root / name for fov, name in SCAN_NAMES.items()})
    torch.set_num_threads(3)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    net, ckpt = load_strong_prior(args.checkpoint, args.device)
    assert ckpt['step'] == 60000 and ckpt['img_resolution'] == 192
    start = time.monotonic()
    if args.mode in ['simulation', 'all']:
        simulation(args, net, ckpt)
    if args.mode in ['real', 'all']:
        real(args, net, ckpt)
    announce(stage='completed', mode=args.mode, out=str(args.out), elapsed_sec=time.monotonic()-start)


if __name__ == '__main__':
    main()
