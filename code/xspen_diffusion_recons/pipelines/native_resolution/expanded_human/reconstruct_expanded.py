"""Selected new real xSPEN frames: native EMA EDM + fixed DiffPIR, no retraining.

The native model and physical sampling grid must match exactly. The old 128
model is an explicit bilinear transfer baseline. No real ground truth exists.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PROJECT = EXP.parents[1]
for directory in (PROJECT, EXP, EXP / 'real_comparison'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from evaluate_native import Resampled128Prior
from model import load_strong_prior
from native_model import load_native_prior
from native_scanner import load_case
from solvers import diffpir
from traditional_phase import reconstruct_traditional
from utils import sha256

METHODS = ['degraded', 'native_tikhonov', 'tikhonov', 'phasemap_inva', 'baseline128', 'diffusion']
TITLES = ['RO Fourier + RSS\nPE still encoded', 'Complex Tikhonov\nalpha = 0.01',
          'Magnitude L2 / CG\nrho = 0.003', 'PhaseMap + windowed InvA\nxSPEN sinc adaptation',
          'Old 128 prior + DiffPIR\nbilinear transfer baseline', 'Native EMA EDM + DiffPIR\nacquired sampling grid']
BASELINE_NOTE = ('Existing 128x128 prior trained at 210x210 mm, bilinear up/down image transfer; '
                 'different physical scale and noise covariance. It is not a newly trained native prior.')
METRIC_NOTE = ('No clean measured ground truth: no PSNR or SSIM. Residual is fit to a fixed '
               'provisional nuisance model, not image accuracy. Coils/object phase/gain are '
               'estimated from the same observation, which favors the native anchor. '
               'PhaseMap magnitude residual also uses those fixed coil factors, not its own phase.')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def atomic_npz(path, values):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as target:
        np.savez_compressed(target, **values)
    os.replace(temp, path)


def array(tensor):
    return tensor.detach().cpu().resolve_conj().resolve_neg().numpy()


def source_hashes():
    paths = [Path(__file__), EXP / 'native_scanner.py', EXP / 'native_model.py', EXP / 'evaluate_native.py',
             EXP / 'real_comparison/traditional_phase.py']
    paths += [PROJECT / name for name in ('operators.py', 'scanner.py', 'solvers.py', 'model.py', 'edm.py', 'tiny_unet.py')]
    paths += [PROJECT.parent / 'spenpy/spenpy/recon' / name for name in ('even_odd.py', 'phasemap.py')]
    return {str(p.resolve()): sha256(p) for p in paths}


def completed_case(npz_path, json_path, signature):
    """Only skip an atomically completed artifact with the same computation."""
    if not json_path.exists():
        return None
    info = json.loads(json_path.read_text())
    if info.get('computation_sha256') != signature:
        raise FileExistsError(f'Existing case differs in input/model/code/settings: {json_path}; choose a new --out')
    if info.get('status') != 'complete' or not npz_path.exists():
        return None
    if sha256(npz_path) != info.get('npz_sha256'):
        raise ValueError(f'Completed case NPZ checksum mismatch: {npz_path}')
    return info


def validate_scan(entry, shape, net_shape):
    if tuple(shape[-2:]) != tuple(net_shape):
        raise ValueError(f"{entry['scan']}: checkpoint shape {net_shape} != native acquisition {shape[-2:]}; resizing is forbidden")
    if len(shape) != 5 or not entry.get('cases'):
        raise ValueError('Expected [repeat,slice,coil,PE,RO] H5 and nonempty selected cases')
    seen = set()
    for case in entry['cases']:
        sl, rep = case['slice_index'], case['repeat']
        if not isinstance(sl, int) or not isinstance(rep, int) or not 0 <= sl < shape[1] or not 0 <= rep < shape[0]:
            raise ValueError(f'Invalid case bounds: {case}, acquisition {shape}')
        if (sl, rep) in seen:
            raise ValueError(f'Duplicate selected case: {case}')
        seen.add((sl, rep))


def draw_page(cases, out, title, smoke):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(cases), 6, figsize=(17, 2.8 * len(cases) + 1.15), squeeze=False)
    for row, info in enumerate(cases):
        height, width = info['fov_mm']
        dy = height / info['native_shape'][0]
        with np.load(info['npz']) as stored:
            for col, method in enumerate(METHODS):
                magnitude = (stored[method][0, 0] + 1) / 2
                shift = .5 * dy if method == 'degraded' else 0
                ax = axes[row, col]
                ax.imshow(magnitude, cmap='gray', vmin=0, vmax=1, interpolation='nearest',
                          extent=(0, width, height-shift, -shift), aspect='equal')
                ax.set_xlim(0, width); ax.set_ylim(height, 0)
                ax.set_facecolor('black'); ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    ax.set_title(TITLES[col], fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"{info['scan']}\nslice {info['slice_index']} / rep {info['repeat']}\n{info['native_shape'][0]} x {info['native_shape'][1]}", fontsize=8)
    fig.suptitle(title + (' | SMOKE: 2 steps, execution check only' if smoke else ' | 60 fixed DiffPIR steps'), fontsize=12)
    fig.text(.5, .015, 'Same observation and [0,1] display window; physical FOV, nearest pixels; no real GT.\nRepeat = occurrence, not a verified b-value/direction. Output grid is not achieved resolution.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .95))
    temporary = out.with_name(out.stem + f'.{os.getpid()}.tmp.png')
    fig.savefig(temporary, dpi=140)
    plt.close(fig)
    os.replace(temporary, out)


def summarize(cases):
    return dict(case_count=len(cases), mean_measurement_residual={
        method: float(np.mean([case['measurement_residual'][method] for case in cases])) for method in METHODS},
        metric_note=METRIC_NOTE, independent_subject_count='Not established from scan/repeat/slice counts')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--smoke', action='store_true', help='Only the first selected case; 2 steps, not a quality assessment')
    parser.add_argument('--baseline', type=Path, default=PROJECT / 'runs/human128/model_ema.pt')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA explicitly requested but unavailable')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    selection = json.loads(args.selection.read_text())
    scans = selection['scans']
    if not scans or len({entry['scan'] for entry in scans}) != len(scans):
        raise ValueError('Require nonempty, uniquely named scans')
    for entry in scans:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', entry['scan']):
            raise ValueError('Scan name must be a safe unique path component')
    if args.smoke:
        scans = [dict(scans[0], cases=scans[0]['cases'][:1])]
    parameters = dict(steps=2 if args.smoke else 60, sigma_noise=.02, lamb=1., xi=0., seed=20260913,
                      complex_tikhonov_alpha=.01, magnitude_proximal_rho=.003,
                      phase_estimator='quadratic', phase_gaussian_width=.8, smoke=args.smoke,
                      native_image_resize=False, device=args.device, torch_version=torch.__version__)
    sources = source_hashes()
    baseline_hash = sha256(args.baseline)
    old_net, old_checkpoint = load_strong_prior(args.baseline, args.device)
    old = Resampled128Prior(old_net).eval().requires_grad_(False)
    config = dict(selection=str(args.selection.resolve()), selection_sha256=sha256(args.selection),
                  selected_scans=scans, parameters=parameters, source_sha256=sources,
                  baseline_checkpoint=str(args.baseline.resolve()), baseline_checkpoint_sha256=baseline_hash,
                  baseline_step=old_checkpoint.get('step'), baseline_limitations=BASELINE_NOTE,
                  note='Inference with existing trained EMA checkpoints; no retraining or achieved-resolution assertion.')
    config_path = args.out / 'config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if canonical_hash(previous) != canonical_hash(config):
            raise FileExistsError('Existing run config differs; choose a new --out')
    else:
        atomic_json(config_path, config)
    all_cases = []
    scan_summaries = []
    cached_checkpoint = None
    net = None
    run_start = time.monotonic()
    for entry in scans:
        scan = entry['scan']
        path = Path(entry['h5']).resolve()
        checkpoint_path = Path(entry['checkpoint']).resolve()
        with h5py.File(path) as handle:
            meta = json.loads(handle.attrs['metadata'])
            shape = tuple(handle['kspace'].shape)
        if cached_checkpoint != checkpoint_path:
            net, checkpoint = load_native_prior(checkpoint_path, args.device)
            cached_checkpoint = checkpoint_path
            checkpoint_hash = sha256(checkpoint_path)
        validate_scan(entry, shape, net.image_shape)
        h5_hash = sha256(path)
        checkpoint_manifest = checkpoint_path.parent.parent.parent / 'data' / checkpoint_path.parent.name / 'manifest.json'
        training_geometry = None
        if checkpoint_manifest.exists():
            manifest = json.loads(checkpoint_manifest.read_text())
            training_geometry = {key: manifest.get(key) for key in ('profile_id', 'grid_id', 'shape', 'fov_mm', 'thickness_mm', 'native_shape')}
        provenance = dict(scanner_h5=str(path), scanner_h5_sha256=h5_hash,
                          source_raw=str(meta.get('source', '')), source_raw_sha256=meta.get('source_sha256'),
                          source_hash_note='Raw SHA256 recorded by scanner export; H5 and model are hashed during this run.',
                          checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                          checkpoint_step=checkpoint.get('step'), checkpoint_weights='ema',
                          checkpoint_manifest_sha256=checkpoint.get('manifest_sha256'), training_geometry=training_geometry,
                          baseline_checkpoint=str(args.baseline.resolve()), baseline_checkpoint_sha256=baseline_hash,
                          baseline_step=old_checkpoint.get('step'), baseline_limitations=BASELINE_NOTE,
                          geometry_note=entry.get('geometry_note', ''), qc_note=entry.get('qc_note', ''),
                          case_selection=entry['cases'], source_metadata=meta, parameters=parameters,
                          source_code_sha256=sources)
        scan_out = args.out / scan
        scan_out.mkdir(exist_ok=True)
        cases = []
        for chosen in entry['cases']:
            sl, rep = chosen['slice_index'], chosen['repeat']
            name = f'{scan}_rep{rep:02d}_slice{sl:03d}'
            npz_path, json_path = scan_out / f'{name}.npz', scan_out / f'{name}.json'
            signature = canonical_hash(dict(provenance=provenance, case=chosen))
            info = completed_case(npz_path, json_path, signature)
            if info is not None:
                cases.append(info)
                print(json.dumps(dict(event='case_skipped_verified', case=name)), flush=True)
                continue
            started = time.monotonic()
            op, observation, l2, anchor, raw_rss, info = load_case(path, sl, rep, args.device, image_shape=None)
            if tuple(op.image_shape) != tuple(net.image_shape) or tuple(op.image_shape) != tuple(shape[-2:]):
                raise ValueError('Native operator/model/acquisition shape disagreement')
            with h5py.File(path) as handle:
                raw = handle['kspace'][rep, sl]
            traditional = reconstruct_traditional(raw, meta, info['magnitude_scale'])
            phase_x = torch.as_tensor(traditional['magnitude'], device=args.device)[None, None] * 2 - 1
            outputs = dict(degraded=raw_rss*2-1, native_tikhonov=anchor, tikhonov=l2, phasemap_inva=phase_x)
            traces = {}
            for method, prior in [('baseline128', old), ('diffusion', net)]:
                outputs[method], traces[method] = diffpir(prior, op, observation,
                    **{key: parameters[key] for key in ('steps', 'sigma_noise', 'lamb', 'xi', 'seed')})
            residuals = {key: float(op.relative_residual(value, observation)) for key, value in outputs.items()}
            clipped_residuals = {key: float(op.relative_residual(value.clamp(-1, 1), observation)) for key, value in outputs.items()}
            payload = {key: array(value) for key, value in outputs.items()}
            payload.update(raw_receiver=raw, measurement=array(observation), original_measurement=array(op.original_observation),
                           phase_correction=array(op.phase_correction), encoding=array(op.a), readout=array(op.f),
                           coils=array(op.coils), pe_projection=array(op.p), ro_projection=array(op.q), mask=array(op.mask),
                           phase_magnitude=traditional['magnitude'], phase_modelx=array(phase_x),
                           phase_map_rad=traditional['phase_map_rad'], phase_coefficients=traditional['phase_coefficients'],
                           phase_coil_images=traditional['coil_images'], phase_corrected_ro_image=traditional['corrected_ro_image'],
                           phase_ro_image=traditional['ro_image'], phase_inv_a=traditional['inv_a'],
                           phase_inv_odd=traditional['inv_odd'], phase_inv_even=traditional['inv_even'],
                           windowed_adjoint_only=traditional['windowed_adjoint_only'])
            if any(not np.isfinite(value).all() for value in payload.values()):
                raise FloatingPointError(f'Nonfinite saved array in {name}')
            atomic_npz(npz_path, payload)
            info.update(scan=scan, provenance=provenance, computation_sha256=signature,
                        npz=str(npz_path), npz_sha256=sha256(npz_path), status='complete',
                        measurement_residual=residuals, clipped_measurement_residual=clipped_residuals,
                        traces=traces, steps=parameters['steps'], sigma_noise=.02, lambda_fixed=1., xi=0.,
                        noise_note='Fixed assumed normalized complex-measurement noise scale for DiffPIR; not a measured noise estimate or synthetic noise injection.',
                        phase_method=traditional['metadata'], metric_note=METRIC_NOTE,
                        model_method='Native-grid EDM EMA prior + DiffPIR with complex-coil CG data consistency',
                        intensity='Model x=2*m-1 stored without clipping. All magnitudes share raw/magnitude_scale; display clips only via fixed [0,1] window.',
                        spatial_display='Native pixels, physical FOV aspect, nearest; only raw RO+RSS extent shifted by -0.5 PE pixel. No reconstructed image resampling.',
                        smoke=args.smoke, elapsed_seconds=time.monotonic()-started)
            atomic_json(json_path, info)
            cases.append(info)
            print(json.dumps(dict(event='case_complete', case=name, checkpoint_step=checkpoint.get('step'),
                                  seconds=round(info['elapsed_seconds'], 2), residuals=residuals)), flush=True)
        groups = collections.defaultdict(list)
        for info in cases:
            groups[info['repeat']].append(info)
        pages = []
        for rep, group in sorted(groups.items()):
            group.sort(key=lambda item: item['slice_index'])
            for start in range(0, len(group), 4):
                page = scan_out / f'{scan}_rep{rep:02d}_page{start//4+1:02d}.png'
                draw_page(group[start:start+4], page, f'{scan}: measured xSPEN, native {shape[-2]} x {shape[-1]}', args.smoke)
                pages.append(str(page))
        summary = dict(scan=scan, **summarize(cases), cases=cases, pages=pages,
                       checkpoint_step=checkpoint.get('step'), checkpoint_sha256=checkpoint_hash,
                       scanner_h5_sha256=h5_hash, geometry_note=entry.get('geometry_note', ''), qc_note=entry.get('qc_note', ''),
                       parameters=parameters, status='complete')
        atomic_json(scan_out / 'summary.json', summary)
        scan_summaries.append({key: value for key, value in summary.items() if key != 'cases'})
        all_cases.extend(cases)
        print(json.dumps(dict(event='scan_complete', scan=scan, cases=len(cases))), flush=True)
    result = dict(config=config, scans=scan_summaries, **summarize(all_cases), status='complete',
                  elapsed_seconds=time.monotonic()-run_start, smoke=args.smoke)
    atomic_json(args.out / 'summary.json', result)
    atomic_json(args.out / 'completed.json', dict(status='complete', summary=str(args.out/'summary.json'),
                                               case_count=len(all_cases), smoke=args.smoke))
    print(json.dumps(dict(event='expanded_reconstruction_complete', out=str(args.out), cases=len(all_cases))), flush=True)


if __name__ == '__main__':
    main()
