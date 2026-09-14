"""Exactly 2x in both in-plane dimensions, with frozen measured xSPEN calibration.

Reuse existing native images and verified matching 2x reconstructions; infer the
remaining cases with existing best EMA checkpoints. No weights are copied.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PROJECT = EXP.parents[1]
for p in (PROJECT, EXP):
    sys.path.insert(0, str(p))
from native_model import load_native_prior
from operators import NativeGridXSPENOperator
from solvers import diffpir
from utils import sha256

METHODS = ('degraded', 'native_tikhonov', 'native_diffusion',
           'tikhonov_bicubic2x', 'native_diffusion_bicubic2x', 'diffusion2x')
SR_JOBS = {(60, 64): 'p3mm_mm1p5', (46, 48): 'p4mm_mm2'}
PARAMETERS = dict(steps=60, sigma_noise=.02, lamb=1., xi=0., seed=20260913)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def write_npz(path, values):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    os.replace(tmp, path)


def array(x):
    return x.detach().cpu().resolve_conj().resolve_neg().numpy()


def build_selection():
    old = json.loads((EXP / 'real_comparison/manifest.json').read_text())
    new = json.loads((EXP / 'expanded_human/selection.json').read_text())
    records = []
    for r in old['records']:
        job = SR_JOBS[tuple(r['native_shape'])]
        records.append(dict(scan=r['scan'], repeat=r['repeat'], slice_index=r['slice_index'],
                            native_reference_npz=r['source_evaluation'],
                            native_checkpoint=r['checkpoint'], sr_job=job,
                            existing_sr_npz=str(EXP / 'evaluation' / job / (r['case'] + '.npz')),
                            selection_note='Same 36 fixed cases as original real comparison.'))
    for scan in new['scans']:
        for r in scan['cases']:
            name = f"{scan['scan']}_rep{r['repeat']:02d}_slice{r['slice_index']:03d}"
            records.append(dict(scan=scan['scan'], repeat=r['repeat'], slice_index=r['slice_index'],
                                native_reference_npz=str(EXP / 'expanded_human/evaluation' / scan['scan'] / (name + '.npz')),
                                native_checkpoint=scan['checkpoint'], sr_job='p4mm_mm2',
                                existing_sr_npz=None, selection_note=scan['slice_selection_basis'],
                                view_note=scan['header_view']))
    if len(records) != 96 or len({(r['scan'], r['repeat'], r['slice_index']) for r in records}) != 96:
        raise ValueError('Expected the 96 fixed, unique cases from 8 scans')
    selection = dict(records=records, scan_count=8, case_count=96,
                     note='No result-based selection. Same slices and occurrences as the prior native comparison.')
    path = HERE / 'selection.json'
    if path.exists() and canonical_hash(json.loads(path.read_text())) != canonical_hash(selection):
        raise FileExistsError('Selection differs from existing experiment')
    write_json(path, selection)
    return records


def source_hashes():
    files = [Path(__file__), EXP / 'native_model.py', EXP / 'native_scanner.py']
    files += [PROJECT / n for n in ('operators.py', 'solvers.py', 'model.py', 'edm.py', 'tiny_unet.py', 'utils.py')]
    return {str(p): sha256(p) for p in files}


def make_operator(stored, shape, device):
    t = lambda key: torch.as_tensor(stored[key], device=device)
    mask = t('mask') if 'mask' in stored else None
    op = NativeGridXSPENOperator(t('encoding'), t('readout'), t('coils'),
                               image_shape=shape, mask=mask)
    return op, t('measurement')


def verified_existing(entry, stored, native_info, checkpoint_hash, parameters, sources):
    """Existing output is reusable only with identical data, calibration and recipe."""
    if not entry.get('existing_sr_npz'):
        return None
    p = Path(entry['existing_sr_npz'])
    info = json.loads(p.with_suffix('.json').read_text())
    config_path = p.parent / 'config.json'
    config = json.loads(config_path.read_text())
    if config['checkpoint_sha256'] != checkpoint_hash or config['checkpoint_step'] != 20000:
        raise ValueError(f'Existing SR checkpoint mismatch: {p}')
    if config['config']['steps'] != parameters['steps'] or config.get('exploratory_smoke'):
        raise ValueError(f'Existing SR steps mismatch: {p}')
    if info['steps'] != parameters['steps'] or info['sigma_noise'] != parameters['sigma_noise'] or info['lambda_fixed'] != parameters['lamb']:
        raise ValueError(f'Existing SR parameters mismatch: {p}')
    for source, digest in sources.items():
        name = Path(source).name
        if name != Path(__file__).name and config['source_sha256'].get(name) != digest:
            raise ValueError(f'Existing SR source changed: {name}')
    # Original evaluate_native source pins seed=20260913 and default xi=0.
    original_evaluator = p.parent / 'source_snapshot/evaluate_native.py'
    if sha256(original_evaluator) != config['source_sha256']['evaluate_native.py']:
        raise ValueError('Existing evaluator snapshot hash mismatch')
    evaluator_text = original_evaluator.read_text()
    if 'sigma_noise=.02, lamb=1., seed=20260913)' not in evaluator_text or parameters['xi'] != 0 or parameters['seed'] != 20260913:
        raise ValueError('Existing evaluator seed/xi recipe cannot be established')
    for key in ('scan', 'repeat', 'slice_index', 'native_shape', 'fov_mm', 'r_value', 'beta', 'magnitude_scale', 'source_sha256'):
        if info[key] != native_info[key]:
            raise ValueError(f'Existing SR metadata differs: {key}')
    with np.load(p) as sr:
        for key in ('measurement', 'original_measurement', 'encoding', 'readout', 'coils', 'phase_correction'):
            if not np.array_equal(sr[key], stored[key]):
                raise ValueError(f'Existing SR calibration differs: {key}')
        x = sr['diffusion'].copy()
    expected = tuple(2 * n for n in native_info['native_shape'])
    if tuple(x.shape) != (1, 1, *expected) or info['output_shape'] != list(expected):
        raise ValueError('Existing output is not exact 2x')
    return dict(image=x, trace=info['traces']['diffusion'], source_npz=str(p),
                source_npz_sha256=sha256(p), source_json_sha256=sha256(p.with_suffix('.json')),
                source_config_sha256=sha256(config_path))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--out', type=Path, default=HERE / 'evaluation')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--scan', help='Optional single scan; partial result is explicitly labelled')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    records = build_selection()
    if args.prepare_only:
        print(json.dumps(dict(event='selection_ready', count=len(records))), flush=True)
        return
    if args.scan:
        records = [r for r in records if r['scan'] == args.scan]
    if args.limit:
        records = records[:args.limit]
    if args.smoke:
        records = records[:1]
    if not records:
        raise ValueError('Empty selection')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    params = dict(PARAMETERS, steps=2 if args.smoke else 60)
    sources = source_hashes()
    checkpoints = {}
    for job in {r['sr_job'] for r in records}:
        p = EXP / 'runs' / job / 'model_ema.pt'
        manifest_path = EXP / 'data' / job / 'manifest.json'
        cfg = json.loads((EXP / 'evaluation' / job / 'config.json').read_text())
        digest = sha256(p)
        if digest != cfg['checkpoint_sha256'] or sha256(manifest_path) != cfg['manifest_sha256']:
            raise ValueError('Best EMA or training manifest differs from completed evaluated model')
        checkpoints[job] = dict(path=str(p), sha256=digest, step=cfg['checkpoint_step'],
                                training_fov_mm=cfg['profile']['fov_mm'], image_shape=cfg['grid']['shape'],
                                training_manifest_sha256=cfg['manifest_sha256'])
    config = dict(parameters=params, device=args.device, smoke=args.smoke,
                  selected_records=records, checkpoints=checkpoints, source_sha256=sources,
                  note='Exact 2x per axis, original FOV and through-plane thickness. Existing best EMA, no new weights.')
    config_path = args.out / 'config.json'
    if config_path.exists() and canonical_hash(json.loads(config_path.read_text())) != canonical_hash(config):
        raise FileExistsError('Run configuration differs; use a new output directory')
    write_json(config_path, config)
    summaries = []
    models = {}
    start = time.monotonic()
    for entry in records:
        case = f"{entry['scan']}_rep{entry['repeat']:02d}_slice{entry['slice_index']:03d}"
        native_path = Path(entry['native_reference_npz'])
        native_info = json.loads(native_path.with_suffix('.json').read_text())
        native_hash = sha256(native_path)
        native_json_hash = sha256(native_path.with_suffix('.json'))
        if native_info.get('npz_sha256') and native_info['npz_sha256'] != native_hash:
            raise ValueError(f'Native reference checksum mismatch: {case}')
        ckpt = checkpoints[entry['sr_job']]
        signature = canonical_hash(dict(entry=entry, native_npz=native_hash, native_json=native_json_hash,
                                        checkpoint=ckpt, parameters=params, sources=sources))
        directory = args.out / entry['scan']
        directory.mkdir(exist_ok=True)
        npz_path, json_path = directory / f'{case}.npz', directory / f'{case}.json'
        if json_path.exists():
            previous = json.loads(json_path.read_text())
            if previous.get('computation_sha256') != signature:
                raise FileExistsError(f'Existing output differs: {case}')
            if previous.get('status') == 'complete' and npz_path.exists() and sha256(npz_path) == previous['npz_sha256']:
                summaries.append(previous)
                print(json.dumps(dict(event='case_skipped_verified', case=case)), flush=True)
                continue
        started = time.monotonic()
        native_shape = tuple(native_info['native_shape'])
        high_shape = tuple(2 * n for n in native_shape)
        if ckpt['image_shape'] != list(high_shape):
            raise ValueError('Checkpoint must exactly match the 2x grid; no prior resizing allowed')
        with np.load(native_path) as source:
            stored = {key: source[key] for key in source.files}
        if tuple(stored['diffusion'].shape[-2:]) != native_shape or native_info['steps'] != 60:
            raise ValueError('Reference must be completed native-grid 60-step reconstruction')
        op, y = make_operator(stored, high_shape, args.device)
        native_op, _ = make_operator(stored, native_shape, args.device)
        t = lambda value: torch.as_tensor(value, device=args.device)
        output = dict(degraded=t(stored['degraded']), native_tikhonov=t(stored['native_tikhonov']),
                      native_diffusion=t(stored['diffusion']))
        for name, original in [('tikhonov_bicubic2x', 'native_tikhonov'), ('native_diffusion_bicubic2x', 'native_diffusion')]:
            output[name] = F.interpolate(output[original], size=high_shape, mode='bicubic', align_corners=False)
        reused = None if args.smoke else verified_existing(entry, stored, native_info, ckpt['sha256'], params, sources)
        if reused:
            output['diffusion2x'] = t(reused.pop('image'))
            trace = reused.pop('trace')
            origin = 'verified_existing'
        else:
            if entry['sr_job'] not in models:
                net, checkpoint = load_native_prior(ckpt['path'], args.device)
                if tuple(net.image_shape) != high_shape or checkpoint['step'] != ckpt['step']:
                    raise ValueError('Loaded checkpoint metadata mismatch')
                models[entry['sr_job']] = net.eval().requires_grad_(False)
            output['diffusion2x'], trace = diffpir(models[entry['sr_job']], op, y, **params)
            origin = 'new_inference'
        residual = {name: float((native_op if tuple(value.shape[-2:]) == native_shape else op).relative_residual(value, y))
                    for name, value in output.items()}
        projected = op.project(output['diffusion2x'])
        projected_residual = float(native_op.relative_residual(projected, y))
        if abs(projected_residual - residual['diffusion2x']) > 2e-5:
            raise AssertionError('2x projection and raw-measurement residual disagree')
        payload = {key: array(value) for key, value in output.items()}
        payload['diffusion2x_projected_native'] = array(projected)
        payload['pe_projection'] = array(op.p)
        payload['ro_projection'] = array(op.q)
        if any(not np.isfinite(v).all() for v in payload.values()):
            raise FloatingPointError(f'Nonfinite output: {case}')
        write_npz(npz_path, payload)
        fov = native_info['fov_mm']
        info = dict(scan=entry['scan'], repeat=entry['repeat'], slice_index=entry['slice_index'],
                    native_shape=list(native_shape), output_shape=list(high_shape), fov_mm=fov,
                    thickness_mm=native_info['thickness_mm'], r_value=native_info['r_value'], beta=native_info['beta'],
                    position_lps_mm=native_info['position_lps_mm'], magnitude_scale=native_info['magnitude_scale'],
                    native_pixel_mm=[fov[i]/native_shape[i] for i in range(2)],
                    output_pixel_mm=[fov[i]/high_shape[i] for i in range(2)],
                    source_raw_sha256=native_info['source_sha256'],
                    native_reference_npz=str(native_path), native_reference_sha256=native_hash,
                    native_reference_json_sha256=native_json_hash, native_checkpoint=entry['native_checkpoint'],
                    sr_checkpoint=ckpt['path'], sr_checkpoint_sha256=ckpt['sha256'], sr_checkpoint_step=ckpt['step'],
                    training_fov_mm=ckpt['training_fov_mm'],
                    max_relative_fov_difference=max(abs(fov[i]/ckpt['training_fov_mm'][i]-1) for i in range(2)),
                    inference_origin=origin, reused_result=reused, parameters=params, smoke=args.smoke,
                    measurement_residual=residual, projected_measurement_residual=projected_residual,
                    relative_image_change_from_bicubic=float(torch.linalg.vector_norm(output['diffusion2x']-output['native_diffusion_bicubic2x']) /
                                                              torch.linalg.vector_norm(output['native_diffusion_bicubic2x']+1).clamp_min(1e-12)),
                    traces=trace, npz=str(npz_path), npz_sha256=sha256(npz_path), status='complete',
                    computation_sha256=signature, source_code_sha256=sources,
                    selection_note=entry['selection_note'], view_note=entry.get('view_note', 'Original acquisition view retained'),
                    intensity='All image arrays are unclipped model x=2*m-1, shared original native receiver scale. Display [0,1].',
                    interpolation_baselines='PyTorch bicubic, align_corners=False, no clipping; interpolation only, not new information.',
                    model_method='Existing exact-grid best EDM EMA + 60-step DiffPIR with frozen native raw-data calibration.',
                    limitations='2x output sampling is not measured effective resolution. Under the current averaging operator 75% of high-grid degrees of freedom are unobserved. No real GT, independent coil/phase or measured noise calibration; occurrences are not verified diffusion labels.',
                    elapsed_seconds=time.monotonic()-started)
        write_json(json_path, info)
        summaries.append(info)
        write_json(args.out / 'progress.json', dict(completed=len(summaries), expected=len(records), last_case=case))
        print(json.dumps(dict(event='case_complete', case=case, origin=origin,
                              seconds=round(info['elapsed_seconds'], 2), residuals=residual)), flush=True)
    counts = collections.Counter(r['inference_origin'] for r in summaries)
    result = dict(status='complete', case_count=len(summaries), scan_count=len({r['scan'] for r in summaries}),
                  full_selection=len(summaries)==96, smoke=args.smoke, records=summaries,
                  inference_counts=dict(counts), elapsed_seconds=time.monotonic()-start, parameters=params,
                  no_real_ground_truth=True, new_checkpoint_files=0,
                  source_selection_sha256=sha256(HERE/'selection.json'))
    write_json(args.out / 'summary.json', result)
    write_json(args.out / 'completed.json', {key:value for key,value in result.items() if key!='records'})
    print(json.dumps(dict(event='run_complete', count=len(summaries), origins=dict(counts), seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
