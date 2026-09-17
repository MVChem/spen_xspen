"""Controlled simulation and real-data pilot for joint phase/DiffPIR inference."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy.io
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT.parent / 'spenpy'))
sys.path.insert(0, str(HERE.parent / 'core'))
sys.path.insert(0, str(HERE.parent / 'prior96'))
from joint_phase_diffpir import PhaseConfig, PhaseOperator, joint_diffpir
from sr_operator import SCANS, SpenSuperResolutionOperator, make_sr_operator
from pilot_testtime_even_odd import fit, make_weights
from phase_inva import coords_grid, load_phase_matrices, wrap_phase
from model_v2 import load_strong_prior
from solvers import diffpir
from evaluate import metrics
from evaluate_png_sr import sha, stable_seed

REFERENCE = PROJECT / 'runs/rodent192_spen2x_260914'


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def announce(**data):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def phase_error(phase, truth, weights):
    return float(((weights * wrap_phase(phase - truth).square()).sum() /
                  weights.sum().clamp_min(1e-12)).sqrt())


def reconstruct(args, net, base, observed, phase, method, seed, sigma, lamb, name):
    start = time.monotonic()
    base.cg_diagnostics.clear()
    state = None
    if method == 'joint':
        x, phase, trace, state = joint_diffpir(net, base, observed, steps=args.steps,
            sigma_noise=sigma, lamb=lamb, seed=seed,
            phase_config=PhaseConfig(updates_per_step=args.phase_updates, seed=args.seed),
            progress=lambda row: announce(event='joint_step', case=name, **row))
    else:
        op = PhaseOperator(base, phase)
        x, trace = diffpir(net, op, observed, steps=args.steps, sigma_noise=sigma,
                          lamb=lamb, sigma_max=2., seed=seed)
    op = PhaseOperator(base, phase)
    checks = base.cg_diagnostics
    result = dict(elapsed_seconds=time.monotonic() - start,
        measurement_nrmse=float(op.relative_residual(x, observed)), sampler=trace,
        cg_calls=len(checks), cg_nonconverged=sum(not all(v['converged']) for v in checks),
        cg_max_relative_residual=max(max(v['relative_normal_residual']) for v in checks),
        cg_max_iterations=max(v['iterations'] for v in checks))
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError(name + '/' + method)
    return x, phase.detach(), result, state


def run_simulation(args, net, source, original_meta):
    SCANS[16] = Path(original_meta['geometry']['path']).parent
    # CPU/CUDA RNGs differ. Reference observations require CUDA-generated
    # coils; newly generated observations are paired with their actual device.
    high = make_sr_operator(fov=16, image_size=192, device=args.device, seed=4517,
                            cg_max_iter=320)
    inv, odd_inv, even_inv = load_phase_matrices(SCANS[16] / 'slice_7.mat', args.device)
    xy = coords_grid(48, 96, high.device).reshape(48, 96, 2)
    records = []
    for condition in args.conditions:
        ci = 0 if condition == 'full' else 1
        sigma = float(source['noise_sigma'][ci])
        lamb = original_meta['conditions'][ci]['selected']['diffusion']
        mask = torch.as_tensor(source['mask'][ci], device=args.device)
        base = SpenSuperResolutionOperator(high.a_full, high.coils, 96, mask,
                                          sigma_noise=sigma, cg_max_iter=320)
        for index in args.indices:
            target = torch.as_tensor(source['target'][index], device=args.device)[None, None] * 2 - 1
            key = str(source['case_keys'][index])
            dataset = str(source['labels'][index])
            with torch.no_grad():
                clean_full = high.forward(target)
                if args.observation_source == 'reference':
                    noisy = torch.as_tensor(source['observation'][ci, index], device=args.device)[None, :, mask]
                else:
                    gen = torch.Generator(device=args.device).manual_seed(stable_seed('joint_noise:' + key))
                    noise = torch.complex(torch.randn(clean_full.shape, device=args.device, generator=gen),
                                          torch.randn(clean_full.shape, device=args.device, generator=gen))
                    noisy = (clean_full + sigma * noise)[:, :, mask]
                noise_residual = noisy - clean_full[:, :, mask]
                noise_check = dict(real_mean=float(noise_residual.real.mean()),
                    imag_mean=float(noise_residual.imag.mean()), real_std=float(noise_residual.real.std()),
                    imag_std=float(noise_residual.imag.std()))
                if any(not .9 * sigma < noise_check[k] < 1.1 * sigma for k in ('real_std', 'imag_std')):
                    raise ValueError(f'Coil/observation mismatch: {noise_check}')
                if any(abs(noise_check[k]) > .05 * sigma for k in ('real_mean', 'imag_mean')):
                    raise ValueError(f'Nonzero noise mean: {noise_check}')
            image_seed = stable_seed(('rebuilt_diffpir:' if args.observation_source == 'reference'
                                      else 'joint_diffpir:') + key)
            for kind in args.phase_kinds:
                name = f'{condition}_{dataset}_{kind}'
                truth = torch.zeros_like(xy[..., 0]) if kind == 'none' else (
                    .6 + .7 * xy[..., 0] - .5 * xy[..., 1] + .25 * xy[..., 0] * xy[..., 1])
                oracle = PhaseOperator(base, truth)
                observed = oracle.factor * noisy
                phase_weights = clean_full[0, :, 1::2].abs().square().sum(0) * mask[1::2, None]
                phases = dict(uncorrected=torch.zeros_like(truth), oracle=truth)
                fit_trace = []
                if kind != 'none':
                    full = observed.new_zeros(4, 96, 96)
                    full[:, mask] = observed[0]
                    fit_scale = full.abs().square().mean().sqrt().clamp_min(1e-8)
                    _, train, _ = make_weights(full / fit_scale, odd_inv, even_inv)
                    phases['sequential'], fit_trace, _ = fit(full / fit_scale, odd_inv,
                        even_inv, train, args.sequential_steps, stable_seed('joint_phase:' + key))
                methods = ('uncorrected', 'joint') if kind == 'none' else (
                    'uncorrected', 'sequential', 'joint', 'oracle')
                arrays = dict(target=source['target'][index], observation=observed[0],
                    clean_observation=clean_full[0, :, mask], coils=high.coils[0],
                    encoding=high.a_full, mask=mask, phase_true=truth, phase_weights=phase_weights)
                row = dict(name=name, kind='simulation', sampling=condition, dataset=dataset,
                    phase_kind=kind, case_key=key, noise_sigma=sigma, lamb=lamb,
                    phase_fit_trace=fit_trace, noise_check=noise_check, image_seed=image_seed, methods={})
                announce(event='case_start', case=name)
                for method in methods:
                    x, phase, result, state = reconstruct(args, net, base, observed,
                        phases.get(method), method, image_seed,
                        sigma, lamb, name)
                    result.update(metrics(x, target)[0])
                    result['phase_rmse_rad'] = phase_error(phase, truth, phase_weights)
                    if args.observation_source == 'reference' and (method == 'oracle' or (
                            kind == 'none' and method == 'uncorrected')):
                        old = torch.as_tensor(source['diffusion_unclipped'][ci, index], device=args.device)
                        result['max_difference_to_reference_image'] = float(((x[0, 0] + 1) / 2 - old).abs().max())
                    row['methods'][method] = result
                    arrays['image_' + method] = (x[0, 0] + 1) / 2
                    arrays['phase_' + method] = phase
                    if state is not None:
                        torch.save(state, args.out / (name + '_phase_net.pt'))
                    announce(event='method_complete', case=name, method=method,
                        **{k: v for k, v in result.items() if k != 'sampler'})
                save_case(args.out, row, arrays)
                records.append(row)
                save_json(args.out / 'progress.json', dict(completed_cases=len(records), last_case=name))
    return records


def run_real(args, net):
    from evaluate_real_sr import load_case
    records = []
    scans = {16: '20240321_lxj_spen_mouse_240321_1_1_1',
             24: '20240115_lxj_SPEN_96_240115_1_1_1'}
    for fov, export in [(16, 22), (24, 11)]:
        path = PROJECT.parent / 'data/spen_acquired_260915/mat' / scans[fov] / f'slice_{export}.mat'
        _, base, corrected, anchor, meta = load_case(path, args.device)
        base.cg_max_iter = 320
        mat = scipy.io.loadmat(path, variable_names=['spen_original_signal_rofft'])
        raw = torch.as_tensor(mat['spen_original_signal_rofft'][:, :, 0].transpose(2, 0, 1),
                              device=args.device, dtype=torch.complex64)[None]
        raw /= meta['magnitude_scale'] * meta['source_encoding_smax']
        # These coils retain the scanner-corrected InvA anchor. This pilot
        # replaces only acquired-even correction, not the coil calibration.
        cross = (raw[0, :, 1::2] * corrected[0, :, 1::2].conj()).sum(0)
        scanner_phase = torch.angle(cross)
        scanner_model = PhaseOperator(base, scanner_phase)
        reconstructed_raw = scanner_model.factor * corrected
        correction_error = float((reconstructed_raw - raw).norm() / raw.norm())
        residual_mode = args.real_input == 'corrected'
        name = f"real_{'residual_' if residual_mode else ''}fov{fov}_export{export}"
        _, odd_inv, even_inv = load_phase_matrices(path, args.device)
        fit_scale = raw.abs().square().mean().sqrt().clamp_min(1e-8)
        _, train, _ = make_weights(raw[0] / fit_scale, odd_inv, even_inv)
        sequential, _, _ = fit(raw[0] / fit_scale, odd_inv, even_inv, train,
                                args.sequential_steps, args.seed)
        phases = dict(uncorrected=torch.zeros(48, 96, device=args.device),
                      sequential=sequential, scanner=scanner_phase)
        arrays = dict(observation=corrected[0] if residual_mode else raw[0], phase_scanner=scanner_phase,
                      anchor=(anchor[0, 0] + 1) / 2)
        row = dict(name=name, kind='real', fov_mm=fov, export=export, metadata=meta,
                   scanner_phase_factor_relative_error=correction_error,
                   real_input=args.real_input, methods={})
        methods = ('scanner', 'joint') if residual_mode else ('uncorrected', 'sequential', 'joint', 'scanner')
        for method in methods:
            # The scanner reference uses the actual saved corrected signal;
            # it is not assumed to equal an even-only phase factor exactly.
            observed = corrected if method == 'scanner' or residual_mode else raw
            phase = torch.zeros_like(scanner_phase) if method == 'scanner' else phases.get(method)
            x, phase, result, state = reconstruct(args, net, base, observed,
                phase, method, 73, .02, 1., name)
            arrays['image_' + method] = (x[0, 0] + 1) / 2
            arrays['phase_' + method] = scanner_phase if method == 'scanner' and not residual_mode else phase
            row['methods'][method] = result
            if state is not None:
                torch.save(state, args.out / (name + '_phase_net.pt'))
            announce(event='method_complete', case=name, method=method,
                     **{k: v for k, v in result.items() if k != 'sampler'})
        save_case(args.out, row, arrays)
        records.append(row)
    return records


def save_case(out, row, arrays):
    converted = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                 for k, v in arrays.items()}
    if not all(np.isfinite(v).all() for v in converted.values()):
        raise FloatingPointError('Nonfinite saved arrays')
    np.savez_compressed(out / (row['name'] + '.npz'), **converted)
    save_json(out / (row['name'] + '.json'), row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=REFERENCE / 'figures_260915')
    parser.add_argument('--checkpoint', type=Path, default=REFERENCE / 'train/model_ema.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--phase-updates', type=int, default=12)
    parser.add_argument('--sequential-steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--conditions', choices=['full', 'half'], nargs='+', default=['full', 'half'])
    parser.add_argument('--phase-kinds', choices=['smooth2d', 'none'], nargs='+', default=['smooth2d', 'none'])
    parser.add_argument('--mode', choices=['simulation', 'real'], default='simulation')
    parser.add_argument('--real-input', choices=['raw', 'corrected'], default='raw')
    parser.add_argument('--observation-source', choices=['new', 'reference'], default='new')
    args = parser.parse_args()
    if args.steps < 2 or args.phase_updates < 1 or args.sequential_steps < 1:
        parser.error('Invalid iteration counts')
    if args.mode == 'simulation' and args.observation_source == 'reference' and not args.device.startswith('cuda'):
        parser.error('Reference observations require CUDA-generated coil RNG; use new observations on CPU')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    net, ckpt = load_strong_prior(args.checkpoint, args.device)
    source_meta = json.loads((args.source / 'simulation.json').read_text())
    if sha(args.checkpoint) != source_meta['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from reference figures')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(checkpoint_sha256=sha(args.checkpoint), checkpoint_step=ckpt['step'],
        phase_config=asdict(PhaseConfig(updates_per_step=args.phase_updates, seed=args.seed)),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        device_name=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'CPU',
        source_npz_sha256=sha(args.source / 'simulation.npz'),
        source_sha256={p.name: sha(p) for p in [Path(__file__), HERE / 'joint_phase_diffpir.py',
            HERE / 'sr_operator.py', HERE / 'pilot_testtime_even_odd.py']},
        model='y = exp(i E phi_theta) * M D_RO A_PE(S*m); original scanner row parity',
        fitting='Per-acquisition MLP only, from complex acquired data each diffusion step; no GT in fitting',
        simulation_scope=('Original four in-training images, masks, geometry and checkpoint; '
                          + ('original saved observations and original diffusion seeds' if args.observation_source == 'reference'
                             else 'NEW device-specific coil/noise realization; compare only within this run')),
        real_scope='Two fixed middle reference cases; scanner-derived coil calibration retained; no anatomical ground truth')
    save_json(args.out / 'config.json', config)
    started = time.monotonic()
    if args.mode == 'simulation':
        with np.load(args.source / 'simulation.npz', allow_pickle=False) as data:
            source = {k: data[k].copy() for k in ('target', 'labels', 'case_keys', 'mask',
                      'noise_sigma', 'observation', 'diffusion_unclipped')}
        records = run_simulation(args, net, source, source_meta)
    else:
        records = run_real(args, net)
    save_json(args.out / 'summary.json', dict(config=config, cases=records,
                                            elapsed_seconds=time.monotonic() - started))
    announce(event='complete', cases=len(records), seconds=time.monotonic() - started)


if __name__ == '__main__':
    main()
