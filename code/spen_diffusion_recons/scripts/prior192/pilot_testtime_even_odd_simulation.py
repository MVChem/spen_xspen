"""Known even-line phase recovery on the original four SPEN simulation cases.

Fits each phase net from corrupted observations only. Known phase, noiseless
signal and image truth are reserved for validation and oracle comparisons.
The optional DiffPIR is sequential phase correction then fixed-prior inference,
not an alternating joint phase/magnitude algorithm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT.parent / 'spenpy'))
sys.path.insert(0, str(HERE.parent / 'core'))
sys.path.insert(0, str(HERE.parent / 'prior96'))
from pilot_testtime_even_odd import (correct, coords_grid, diagnostics, digest, fit,
                                    fit_tiny_phase_scanner_batch, load_phase_matrices,
                                    make_weights, wrap_phase)
from sr_operator import SCANS, make_sr_operator, SpenSuperResolutionOperator
from evaluate import metrics
from evaluate_png_sr import stable_seed, sha
from model_v2 import load_strong_prior
from solvers import diffpir

DEFAULT_RUN = PROJECT / 'runs/rodent192_spen2x_260914'
METHODS = ('uncorrected', 'legacy_tiny', 'tiny', 'oracle')


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def fitted_phases(observed, odd_inv, even_inv, args, seed):
    # A single observation-derived scalar improves numerical conditioning.
    # It does not change a phase estimate's physical amplitude convention.
    fit_scale = observed.abs().square().mean().sqrt().clamp_min(1e-8)
    fit_signal = observed / fit_scale
    weights, train, evaluation = make_weights(fit_signal, odd_inv, even_inv)
    phase, trace, state = fit(fit_signal, odd_inv, even_inv, train, args.steps, seed)
    scanner = fit_signal.permute(1, 2, 0).unsqueeze(2)[None]
    devices = [observed.device.index] if observed.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        old_phase = fit_tiny_phase_scanner_batch(scanner, odd_inv, even_inv, steps=args.steps)[0]
    return phase, old_phase, trace, state, (weights, train, evaluation), float(fit_scale)


def phase_metrics(phase, truth, energy):
    error = wrap_phase(phase - truth)
    return dict(phase_error_rms_rad=float(((energy * error.square()).sum() / energy.sum()).sqrt()))


def run_case(args, source, meta, high, op, matrices, net, index, phase_kind):
    started = time.monotonic()
    condition = 0 if args.condition == 'full' else 1
    dataset = str(source['labels'][index])
    case_key = str(source['case_keys'][index])
    name = f'{dataset}_{phase_kind}'
    target = torch.as_tensor(source['target'][index], device=args.device)[None, None] * 2 - 1
    baseline = torch.as_tensor(source['observation'][condition, index], device=args.device)
    mask = op.mask
    with torch.no_grad():
        clean = high.forward(target)[0]
        clean[:, ~mask] = 0
    noise_residual = (baseline - clean)[:, mask]
    sigma = float(source['noise_sigma'][condition])
    noise_check = dict(real_mean=float(noise_residual.real.mean()),
                       imaginary_mean=float(noise_residual.imag.mean()),
                       real_std=float(noise_residual.real.std()),
                       imaginary_std=float(noise_residual.imag.std()))
    for key in ('real_std', 'imaginary_std'):
        if not .9 * sigma < noise_check[key] < 1.1 * sigma:
            raise AssertionError(f'Forward model / original noise mismatch: {noise_check}')
    for key in ('real_mean', 'imaginary_mean'):
        if abs(noise_check[key]) > .05 * sigma:
            raise AssertionError(f'Nonzero original noise mean: {noise_check}')
    coords = coords_grid(48, 96, baseline.device).reshape(48, 96, 2)
    xx, yy = coords[..., 0], coords[..., 1]
    true_phase = dict(none=torch.zeros_like(xx), ro_only=.6 + .7 * xx,
                      smooth2d=.6 + .7 * xx - .5 * yy + .25 * xx * yy)[phase_kind]
    observed = correct(baseline, -true_phase)
    if bool(observed[:, ~mask].abs().any()):
        raise AssertionError('Injection filled unacquired rows')
    _, odd_inv, even_inv = matrices
    seed = stable_seed('testtime_sim_phase:' + case_key)
    phase, old_phase, trace, state, confidence, fit_scale = fitted_phases(
        observed, odd_inv, even_inv, args, seed)
    phases = dict(uncorrected=torch.zeros_like(true_phase), legacy_tiny=old_phase,
                  tiny=phase, oracle=true_phase)
    corrected = {method: correct(observed, value) for method, value in phases.items()}
    energy = clean[:, 1::2].abs().square().sum(0)
    assert not bool(energy[~mask[1::2]].any())
    oracle_error = float((corrected['oracle'] - baseline).abs().max())
    if oracle_error > 5e-7:
        raise AssertionError(f'Oracle failed to undo injection: {oracle_error}')
    arrays = dict(target=source['target'][index], phase_true=true_phase,
                  phase_weights=energy, mask=mask, observation_baseline=baseline,
                  observation_corrupted=observed, fit_weights=confidence[0],
                  spatial_holdout_weights=confidence[2])
    methods = {}
    rho = float(meta['conditions'][condition]['selected']['tikhonov'])
    lamb = float(meta['conditions'][condition]['selected']['diffusion'])
    for method in METHODS:
        signal = corrected[method]
        amplitude_error = float((signal.abs() - observed.abs()).abs().max())
        if amplitude_error > 5e-7 or bool(signal[:, ~mask].abs().any()):
            raise AssertionError('Phase-only/mask invariant failed')
        methods[method] = dict(**phase_metrics(phases[method], true_phase, energy),
            measurement_difference_to_oracle=float((signal - baseline).norm() / baseline.norm()),
            amplitude_preservation_error=amplitude_error,
            odd_even=diagnostics(signal / fit_scale, odd_inv, even_inv, *confidence))
        op.cg_diagnostics.clear()
        with torch.no_grad():
            image = op.proximal(torch.full_like(target, -1), signal[:, mask][None], rho)
        if not bool(torch.isfinite(image).all()):
            raise FloatingPointError('Nonfinite Tikhonov image')
        methods[method]['tikhonov'] = dict(**metrics(image, target)[0],
            measurement_nrmse=float(op.relative_residual(image, signal[:, mask][None])),
            cg=list(op.cg_diagnostics))
        arrays['phase_' + method] = phases[method]
        arrays['signal_' + method] = signal
        arrays['tikhonov_' + method] = (image[0, 0] + 1) / 2
        if net is not None and phase_kind == 'smooth2d' and method != 'legacy_tiny':
            op.cg_diagnostics.clear()
            with torch.no_grad():
                result, sampler = diffpir(net, op, signal[:, mask][None],
                    steps=args.diffusion_steps, sigma_noise=sigma, lamb=lamb, sigma_max=2.,
                    seed=stable_seed('rebuilt_diffpir:' + case_key))
            methods[method]['diffusion'] = dict(**metrics(result, target)[0],
                measurement_nrmse=float(op.relative_residual(result, signal[:, mask][None])),
                sampler=sampler, cg=list(op.cg_diagnostics))
            arrays['diffusion_' + method] = (result[0, 0] + 1) / 2
    old_tik = source['tikhonov_unclipped'][condition, index]
    oracle_tik = arrays['tikhonov_oracle'].cpu().numpy()
    oracle_image_difference = float(np.max(np.abs(old_tik - oracle_tik)))
    # Small iterative floating-point variations are allowed; a model mismatch is not.
    if oracle_image_difference > 2e-3:
        raise AssertionError(f'Original Tikhonov mismatch: {oracle_image_difference}')
    record = dict(name=name, dataset=dataset, case_key=case_key, case_index=index,
                  phase_kind=phase_kind, methods=methods, fit_trace=trace,
                  fit_scale=fit_scale, fit_seed=seed, noise_check=noise_check,
                  oracle_observation_max_error=oracle_error,
                  oracle_tikhonov_max_difference_to_original=oracle_image_difference,
                  elapsed_seconds=time.monotonic() - started)
    payload = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
               for k, v in arrays.items()}
    if not all(np.isfinite(value).all() for value in payload.values()):
        raise FloatingPointError('Nonfinite saved arrays')
    np.savez_compressed(args.out / (name + '.npz'), **payload)
    torch.save(dict(state_dict=state, steps=args.steps, seed=seed), args.out / (name + '_phase_net.pt'))
    save_json(args.out / (name + '.json'), record)
    print(json.dumps(dict(event='case_complete', sampling=args.condition, name=name,
        seconds=record['elapsed_seconds'], phase_rmse={k: v['phase_error_rms_rad'] for k, v in methods.items()},
        tikhonov_psnr={k: v['tikhonov']['psnr'] for k, v in methods.items()},
        diffusion_psnr={k: v['diffusion']['psnr'] for k, v in methods.items() if 'diffusion' in v})), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_RUN / 'figures_260915')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--condition', choices=('full', 'half'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--diffusion', action='store_true')
    parser.add_argument('--diffusion-steps', type=int, default=60)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_RUN / 'train/model_ema.pt')
    args = parser.parse_args()
    if args.steps < 1 or args.diffusion_steps < 2:
        parser.error('Invalid step count')
    if not args.device.startswith('cuda'):
        parser.error('Original synthetic coil RNG uses CUDA; run on a CUDA device')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    with np.load(args.source / 'simulation.npz', allow_pickle=False) as saved:
        source = {key: saved[key].copy() for key in saved.files}
    meta = json.loads((args.source / 'simulation.json').read_text())
    geometry = meta['geometry']
    scan_path = Path(geometry['path'])
    SCANS[16] = scan_path.parent
    high = make_sr_operator(fov=16, image_size=192, device=args.device,
                            seed=geometry['coil_seed'], cg_max_iter=320)
    matrices = load_phase_matrices(scan_path, args.device)
    condition = 0 if args.condition == 'full' else 1
    sigma = float(source['noise_sigma'][condition])
    mask = torch.as_tensor(source['mask'][condition], device=args.device)
    op = SpenSuperResolutionOperator(high.a_full, high.coils, 96, mask,
                                     sigma_noise=sigma, cg_max_iter=320)
    net = None
    checkpoint_hash = None
    if args.diffusion:
        checkpoint_hash = sha(args.checkpoint)
        if checkpoint_hash != meta['checkpoint_sha256']:
            raise ValueError('Different diffusion checkpoint from original figures')
        net, checkpoint = load_strong_prior(args.checkpoint, args.device)
        assert checkpoint['step'] == 60000 and checkpoint['img_resolution'] == 192
    config = dict(sampling=args.condition, noise_sigma=sigma, steps=args.steps,
        device=args.device, phase_network_parameters=1153, image_grid=[192, 192],
        observed_grid=[96, 96], selected_pe_rows=torch.where(mask)[0].cpu().tolist(),
        observed_odd_rows=int(mask[::2].sum()), observed_even_rows=int(mask[1::2].sum()),
        phase_grid='48x96 acquired-even row / RO coordinates, each axis in [-1,1]',
        phase_formulas_rad=dict(none='0', ro_only='.6+.7*x', smooth2d='.6+.7*x-.5*y+.25*x*y'),
        injection='Multiply the saved noisy even observations by exp(+i*phase); identical underlying noisy observation in all conditions',
        source=str(args.source), source_npz_sha256=sha(args.source / 'simulation.npz'),
        source_json_sha256=sha(args.source / 'simulation.json'),
        source_script_sha256={name: digest(HERE / name) for name in
            ['pilot_testtime_even_odd_simulation.py', 'pilot_testtime_even_odd.py', 'phase_inva.py', 'sr_operator.py']},
        fit='Per corrupted observation only; fixed 500-step configuration from real pilot, no tuning on truth',
        phase_error='Wrapped difference from known injected acquired-even phase; noiseless acquired-even coil energy weighting; unacquired rows excluded',
        spatial_holdout='20% decoded blocks excluded from alignment loss, not independent acquired data',
        original_parameters=meta['conditions'][condition]['selected'],
        diffraction_prior_scope='All four report images participated in the original prior training; no held-out generalization claim',
        coil_scope='Known synthetic object/coil phase fixed; only the extra odd/even acquisition phase is unknown',
        legacy_comparison='Original legacy tiny optimizer uses all confidence pixels; not a single-factor correction-order ablation',
        diffusion=args.diffusion, diffusion_steps=args.diffusion_steps,
        checkpoint_sha256=checkpoint_hash,
        diffusion_scope='Only smooth2d: uncorrected, tiny-corrected and true-phase-corrected; same initialization per case; sequential fitting then inference, not alternating')
    save_json(args.out / 'config.json', config)
    results = []
    for phase_kind in ('none', 'ro_only', 'smooth2d'):
        for index in range(len(source['target'])):
            results.append(run_case(args, source, meta, high, op, matrices, net, index, phase_kind))
            save_json(args.out / 'progress.json', dict(completed=len(results), total=12,
                                                     last=results[-1]['name']))
    save_json(args.out / 'summary.json', dict(config=config, cases=results))
    print(json.dumps(dict(event='complete', sampling=args.condition, cases=len(results))), flush=True)


if __name__ == '__main__':
    main()
