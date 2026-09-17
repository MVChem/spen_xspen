"""Replace only the final row of the approved SPEN comparison figures.

Reuse the original observations and all earlier rows. Simulation has no
additional phase perturbation. Real data retain the original scanner phase
correction and coil estimates; the new network fits residual even-line phase.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from evaluate_joint_phase import (PROJECT, REFERENCE, PhaseConfig, PhaseOperator,
    announce, joint_diffpir, load_strong_prior, save_json, sha)
from evaluate_real_sr import load_case
import render_rebuilt as renderer


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def check_fixed_arrays(original, replacement):
    fixed = [k for k in original if k not in ('diffusion', 'diffusion_unclipped')]
    for key in fixed:
        if not np.array_equal(original[key], replacement[key]):
            raise AssertionError(f'Changed reference field: {key}')
    return fixed


def check_only_final_row(reference_path, new_path, expected_rows):
    original = np.asarray(Image.open(reference_path).convert('RGB'))
    replacement = np.asarray(Image.open(new_path).convert('RGB'))
    if original.shape != replacement.shape:
        raise AssertionError(f'Figure geometry changed: {original.shape} vs {replacement.shape}')
    occupied = (original < 245).any(axis=-1).mean(axis=1) > .5
    starts = np.where(np.diff(np.r_[False, occupied].astype(int)) == 1)[0]
    if len(starts) != expected_rows:
        raise AssertionError(f'Cannot identify reference image rows: {starts}')
    last_start = int(starts[-1])
    changed = np.any(original != replacement, axis=-1)
    if changed[:last_start].any() or not changed[last_start:].any():
        raise AssertionError('Changes are not confined to the last row')
    yy, xx = np.where(changed)
    return dict(shape=list(original.shape), unchanged_top_rows=expected_rows - 1,
                last_row_start_pixel=last_start, changed_pixels=int(changed.sum()),
                changed_bbox_xyxy=[int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
                pixels_changed_above_last_row=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=PROJECT / 'runs/joint_phase_diffpir_260916')
    parser.add_argument('--reference', type=Path, default=REFERENCE / 'figures_260915')
    parser.add_argument('--approved', type=Path,
                        default=PROJECT.parents[1] / 'experiments/SPEN_Reconstruction_Comparison_260915')
    parser.add_argument('--checkpoint', type=Path, default=REFERENCE / 'train/model_ema.pt')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    reference_sim = load_npz(args.reference / 'simulation.npz')
    reference_real = load_npz(args.reference / 'real.npz')
    real_meta = json.loads((args.reference / 'real.json').read_text())
    cache_sim = args.cache / 'gpu1_simulation'
    cache_real = args.cache / 'gpu1_real_residual'
    sim_summary = json.loads((cache_sim / 'summary.json').read_text())
    real_summary = json.loads((cache_real / 'summary.json').read_text())
    checkpoint_hash = sha(args.checkpoint)
    for summary in (sim_summary, real_summary):
        c = summary['config']
        if (c['checkpoint_sha256'] != checkpoint_hash or c['steps'] != 60
                or c['phase_config'] != dict(updates_per_step=12, learning_rate=.01,
                    smooth_weight=.002, energy_weight=.0001, seed=20260916)):
            raise ValueError('Cached phase-net settings do not match')
    if sim_summary['config']['source_npz_sha256'] != sha(args.reference / 'simulation.npz'):
        raise ValueError('Cached simulation observations differ')
    if sim_summary['config']['observation_source'] != 'reference':
        raise ValueError('Simulation cache must use the original observations')
    if real_summary['config']['real_input'] != 'corrected':
        raise ValueError('Real cache must retain the original scanner correction')
    approved_hashes = {name: sha(args.approved / name) for name in
                       ('figure1_simulation.png', 'figure2_real.png', 'README.md')}
    simulation = {k: v.copy() for k, v in reference_sim.items()}
    real = {k: v.copy() for k, v in reference_real.items()}
    sim_rows = {(r['sampling'], r['case_key']): r for r in sim_summary['cases']
                if r['phase_kind'] == 'none'}
    sim_records = []
    for ci, condition in enumerate(('full', 'half')):
        for index, key in enumerate(reference_sim['case_keys']):
            row = sim_rows[(condition, str(key))]
            values = load_npz(cache_sim / (row['name'] + '.npz'))
            observed = reference_sim['observation'][ci, index][:, reference_sim['mask'][ci]]
            if (not np.array_equal(observed, values['observation'])
                    or not np.array_equal(reference_sim['target'][index], values['target'])
                    or np.any(values['phase_true'])):
                raise ValueError('Simulation replacement is not the original no-perturbation case')
            simulation['diffusion'][ci, index] = values['image_joint'].clip(0, 1)
            simulation['diffusion_unclipped'][ci, index] = values['image_joint']
            sim_records.append(dict(sampling=condition, key=str(key), source=str(cache_sim / (row['name'] + '.npz')),
                                    metrics=row['methods']['joint']))
    fixed_sim = check_fixed_arrays(reference_sim, simulation)
    np.savez_compressed(args.out / 'simulation.npz', **simulation)
    announce(event='simulation_row_ready', cases=len(sim_records))

    net, ckpt = load_strong_prior(args.checkpoint, args.device)
    cached = {(r['fov_mm'], r['export']): r for r in real_summary['cases']}
    real_records = []
    for i, original in enumerate(real_meta['cases']):
        started = time.monotonic()
        identity = original['fov_mm'], original['export_index']
        path = Path(original['path'])
        if sha(path) != original['source_sha256']:
            raise ValueError('Real acquisition changed since the approved comparison')
        _, op, y, anchor, meta = load_case(path, args.device)
        op.cg_max_iter = 320
        anchor_display = np.rot90((F.interpolate(anchor, (192, 192), mode='bicubic',
                                 align_corners=False)[0, 0].cpu().numpy() + 1) / 2, 2)
        anchor_error = float(np.max(np.abs(anchor_display - reference_real['phase_inva_unclipped'][i])))
        if anchor_error > 2e-5:
            raise AssertionError(f'Changed real-data scale/anchor: {anchor_error}')
        name = f'fov{identity[0]}_export{identity[1]}'
        if identity in cached:
            row = cached[identity]
            values = load_npz(cache_real / (row['name'] + '.npz'))
            if not np.allclose(values['observation'], y[0].cpu().numpy(), rtol=1e-6, atol=1e-7):
                raise ValueError('Cached real observation differs')
            native_image = values['image_joint']
            details = dict(reused_from=str(cache_real / (row['name'] + '.npz')),
                           metrics=row['methods']['joint'])
        else:
            x, phase, trace, state = joint_diffpir(net, op, y, steps=60,
                sigma_noise=.02, lamb=1., seed=73, phase_config=PhaseConfig())
            native_image = (x[0, 0].cpu().numpy() + 1) / 2
            torch.save(state, args.out / (name + '_phase_net.pt'))
            np.savez_compressed(args.out / (name + '.npz'), image=native_image,
                phase=phase.cpu().numpy(), observation=y[0].cpu().numpy())
            details = dict(sampler=trace, cg=list(op.cg_diagnostics),
                measurement_nrmse=float(PhaseOperator(op, phase).relative_residual(x, y)))
            if any(not all(c['converged']) for c in op.cg_diagnostics):
                raise AssertionError('Real PCG failed to converge')
        display = np.rot90(native_image, 2)
        if not np.isfinite(display).all():
            raise FloatingPointError(name)
        real['diffusion'][i] = display.clip(0, 1)
        real['diffusion_unclipped'][i] = display
        real_records.append(dict(fov_mm=identity[0], export_index=identity[1],
            metadata=meta, anchor_max_error=anchor_error, **details))
        announce(event='real_row_case_ready', name=name, reused=identity in cached,
                 seconds=time.monotonic() - started)
    fixed_real = check_fixed_arrays(reference_real, real)
    np.savez_compressed(args.out / 'real.npz', **real)
    renderer.ROW_LABELS = {**renderer.ROW_LABELS, 'diffusion': 'Diffusion prior\n+ PhaseNet'}
    renderer.render_simulation(args.out / 'simulation.npz', args.out / 'figure1_simulation')
    renderer.render_real(args.out / 'real.npz', args.out / 'figure2_real')
    figures = {name: check_only_final_row(args.approved / name, args.out / name, rows)
               for name, rows in [('figure1_simulation.png', 5), ('figure2_real.png', 4)]}
    for name, digest in approved_hashes.items():
        if sha(args.approved / name) != digest:
            raise AssertionError('Approved reference file changed')
    metadata = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
        checkpoint_step=ckpt['step'], phase_config=vars(PhaseConfig()), steps=60,
        source_sha256={p.name: sha(p) for p in [Path(__file__), Path(renderer.__file__)]},
        reference=str(args.reference), approved=str(args.approved), approved_sha256=approved_hashes,
        unchanged_simulation_fields=fixed_sim, unchanged_real_fields=fixed_real,
        simulation='Original observations, masks, noise and seeds; no extra injected phase',
        real='Original scanner-corrected observations and coil estimates; joint residual even-line phase',
        figure_pixel_checks=figures, simulation_cases=sim_records, real_cases=real_records)
    save_json(args.out / 'replacement.json', metadata)
    announce(event='complete', figure_pixel_checks=figures)


if __name__ == '__main__':
    main()
