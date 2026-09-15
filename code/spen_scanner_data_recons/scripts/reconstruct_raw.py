#!/usr/bin/env python3
"""Reconstruct measured SPEN raw scans with Phase Map + InvA and Tikhonov.

Data are read from Bruker binary payloads. Existing MAT exports are optional
regression references, never reconstruction inputs. Every slice and volume
is reconstructed independently at its native matrix size.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np
from scipy.io import loadmat
import torch

PROJECT = Path(__file__).resolve().parents[1]
SPENPY = PROJECT.parent / 'spenpy'
sys.path.insert(0, str(SPENPY))
from spenpy.io import read_bruker
from spenpy.recon import reconstruct_bruker
from spenpy._legacy.bruker.param import read_pv_param
from spenpy._legacy.bruker.image import read_bruker_2dseq


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().resolve_conj().numpy()
    return np.asarray(value)


def parameter(scan_dir, name, default=None):
    value = read_pv_param(str(scan_dir), name)
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def scalar(scan_dir, name, default=None):
    value = parameter(scan_dir, name, default)
    if value is None:
        raise ValueError(f'Missing required parameter {name}: {scan_dir}')
    return np.asarray(value).reshape(-1)[0].item()


def tikhonov_reconstruct(encoding, observation, lambda_relative):
    """Solve min_X ||AX-Y||_F^2 + lambda_relative*||A||_2^2*||X||_F^2.

    A is the PE forward matrix, Y has [acquired_PE, RO, coil] axes. The
    identity penalty acts on complex coil images independently. No estimated
    sensitivity maps, image clipping, or magnitude-only constraint is used.
    """
    a = np.asarray(encoding, dtype=np.complex128)
    y = np.asarray(observation, dtype=np.complex128)
    if not np.isfinite(lambda_relative) or lambda_relative <= 0:
        raise ValueError('lambda_relative must be finite and positive')
    if a.ndim != 2 or y.ndim != 3 or a.shape[0] != y.shape[0]:
        raise ValueError('Expected A[acquired_PE,image_PE] and Y[acquired_PE,RO,coil]')
    if not np.isfinite(a).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite Tikhonov inputs')
    u, singular, vh = np.linalg.svd(a, full_matrices=False)
    smax = float(singular[0])
    if smax <= 0:
        raise ValueError('Zero encoding matrix')
    alpha = float(lambda_relative) * smax**2
    rhs = y.reshape(y.shape[0], -1)
    x = (vh.conj().T * (singular / (singular**2 + alpha))) @ (u.conj().T @ rhs)
    prediction = a @ x
    normal_error = a.conj().T @ (prediction - rhs) + alpha * x
    tiny = np.finfo(np.float64).tiny
    relative_normal = float(np.linalg.norm(normal_error) / max(np.linalg.norm(a.conj().T @ rhs), tiny))
    if not np.isfinite(x).all() or relative_normal > 1e-9:
        raise FloatingPointError(f'Tikhonov solve failed normal-equation check: {relative_normal}')
    return x.reshape(a.shape[1], *y.shape[1:]), {
        'lambda_relative': float(lambda_relative), 'lambda_absolute': alpha,
        'encoding_spectral_norm': smax, 'solver': 'complex128 SVD, identity L2 penalty per coil',
        'relative_residual': float(np.linalg.norm(prediction - rhs) / max(np.linalg.norm(rhs), tiny)),
        'relative_normal_residual': relative_normal,
    }


def regression(reference, arrays, slice_index, volume_index, slices, volumes, coils):
    comparisons = {}
    for ours, previous in [('rofft_original', 'spen_original_signal_rofft'),
                           ('rofft_corrected', 'spen_phase_corrected_signal_rofft'),
                           ('inva_corrected', 'traditional_sr_data')]:
        old = reference[previous]
        expected = old.shape[:2] + (slices, volumes, coils)
        old_frame = old[:, :, 0, :].reshape(expected, order='F')[:, :, slice_index, volume_index, :]
        new_frame = arrays[ours]
        if old_frame.shape != new_frame.shape:
            raise ValueError(f'MAT reference shape mismatch: {old_frame.shape}, {new_frame.shape}')
        error = float(np.linalg.norm(new_frame - old_frame) / max(np.linalg.norm(old_frame), 1e-30))
        comparisons[previous] = {'relative_l2_error': error, 'passed': error < 1e-6}
    if not all(item['passed'] for item in comparisons.values()):
        raise AssertionError(f'Raw reconstruction differs from prior MAT: {comparisons}')
    return comparisons


def run_case(spec, data_root, out, scan_inventory, file_inventory, mat_by_scan, lambdas, main_lambda):
    scan_key = f"raw/{spec['experiment_name']}/{int(spec['scan_id'])}"
    raw_scan = scan_inventory[scan_key]
    if raw_scan['classification'] != 'spen_imaging' or raw_scan['raw_status'] != 'nonempty':
        raise ValueError('Only nonempty quadratic SPEN imaging scans are supported here')
    scan_dir = data_root / scan_key
    reference_record = mat_by_scan.get(scan_key)
    metadata = reference_record['metadata'] if reference_record else {}
    flavor = spec.get('regrid_flavor', metadata.get('recon_flavor'))
    if flavor not in ('pv360', 'pv5'):
        raise ValueError('Set regrid_flavor=pv360 or pv5 explicitly for scans without an export reference')
    trajectory_id = spec.get('trajectory_scan_id', metadata.get('trajectory_scan_id', -1))
    trajectory_dir = scan_dir.parent / str(trajectory_id) if trajectory_id is not None and int(trajectory_id) >= 0 else scan_dir
    n_segments = int(scalar(scan_dir, 'NSegments'))
    echoes = int(scalar(scan_dir, 'PVM_NEchoImages'))
    if n_segments != 1 or echoes != 1:
        raise NotImplementedError('This initial pipeline supports single-shot, single-echo SPEN; multi-shot/echo require their own validated handling')
    for required in ['PVM_Matrix', 'PVM_Fov', 'SpenGyGaussStren', 'SpatEncDuration']:
        if parameter(scan_dir, required) is None:
            raise ValueError(f'Missing reconstruction parameter: {required}')
    trajectory = parameter(trajectory_dir, 'PVM_EpiTrajAdjkx')
    if trajectory is None or not np.any(np.asarray(trajectory) > 0):
        raise ValueError('No verified nonzero readout trajectory; do not silently assume uniform sampling')

    source_hashes = {}
    for path in [scan_dir / p['file'] for p in raw_scan['raw_payloads']] + [scan_dir/'method', scan_dir/'acqp', trajectory_dir/'method']:
        relative = str(path.relative_to(data_root))
        digest = sha256(path)
        if digest != file_inventory[relative]['sha256']:
            raise ValueError(f'Raw source changed since import: {path}')
        source_hashes[relative] = digest
    acquisition = read_bruker(scan_dir, device='cpu')
    pe, ro, coils, slices, volumes = acquisition.data.shape
    if not torch.isfinite(acquisition.data).all():
        raise ValueError('Nonfinite sorted raw data')
    params = {
        'method': str(parameter(scan_dir, 'Method')),
        'matrix_ro_pe': parameter(scan_dir, 'PVM_Matrix'),
        'fov_mm': parameter(scan_dir, 'PVM_Fov'), 'coils': coils,
        'slices': slices, 'volumes': volumes, 'n_segments': n_segments,
        'echoes': echoes, 'regrid_flavor': flavor,
        'trajectory_scan_id': trajectory_id,
        'slice_thickness_mm': parameter(scan_dir, 'PVM_SliceThick'),
        'effective_b_values_s_mm2': parameter(scan_dir, 'PVM_DwEffBval'),
    }
    case = {**spec, 'raw_scan_path': scan_key, 'raw_shape': list(acquisition.data.shape),
            'raw_axes': list(acquisition.axes), 'parameters': params,
            'source_sha256': source_hashes, 'frames': [],
            'reference_mat_path': reference_record['mat_path'] if reference_record else None}
    old_mat = None
    if reference_record:
        ref_path = data_root / reference_record['mat_path']
        if sha256(ref_path) != reference_record['mat_sha256']:
            raise ValueError('Reference MAT changed since import')
        old_mat = loadmat(ref_path)
    # Retain the scanner's own reconstructed data as a separate native-frame
    # preview. Frame geometry is not registered to the new reconstruction.
    scanner = None
    scanner_path = scan_dir / 'pdata/1'
    if (scanner_path/'2dseq').is_file() and (scanner_path/'visu_pars').is_file():
        scanner = read_bruker_2dseq(str(scanner_path))
        if not np.isfinite(scanner).all():
            raise ValueError('Nonfinite scanner 2dseq preview')
        case['scanner_preview'] = {'path': str(scanner_path.relative_to(data_root)),
            'shape': list(scanner.shape), 'frame_mapping': 'Scanner frame order retained; no anatomical registration to reconstructed slice/volume'}
        np.save(out/'arrays'/f"{spec['id']}_scanner_2dseq.npy", scanner)
    start = time.monotonic()
    for volume_index in range(volumes):
        for slice_index in range(slices):
            frame_id = f"{spec['id']}_v{volume_index:02d}_s{slice_index:02d}"
            frame_input = acquisition.select(slice=slice_index, volume=volume_index)
            result = reconstruct_bruker(scan_dir, frame_input,
                traj_dir=str(trajectory_dir), regrid_flavor=flavor,
                process_with_pre_phase_corr=True,
                smooth_motion_phase_between_shots=flavor != 'pv5', device='cpu')
            arrays = {
                'sorted_samples': to_numpy(frame_input.data),
                'rofft_original': to_numpy(result.roffted_data_origin)[:, :, 0, :],
                'rofft_corrected': to_numpy(result.roffted_data_corrected)[:, :, 0, :],
                'inva_corrected': to_numpy(result.sr_data)[:, :, 0, :],
                'encoding': to_numpy(result.spen_az['tmpAFinal']),
                'inva_weighted_adjoint': to_numpy(result.spen_az['tmpInvAZ']),
                'traditional_adaptive': np.abs(to_numpy(result.images)),
            }
            arrays['inva_uncorrected'] = np.einsum('ij,jrc->irc', arrays['inva_weighted_adjoint'], arrays['rofft_original'])
            tikhonov, diagnostics = [], []
            for lam in lambdas:
                x, d = tikhonov_reconstruct(arrays['encoding'], arrays['rofft_corrected'], lam)
                tikhonov.append(x)
                diagnostics.append(d)
            main_index = lambdas.index(main_lambda)
            arrays['tikhonov_coils'] = tikhonov[main_index]
            arrays['tikhonov_sweep_coils'] = np.stack(tikhonov)
            arrays['tikhonov_lambdas'] = np.asarray(lambdas)
            # This is an explicit preview index, not a claim of spatial
            # correspondence. The complete scanner stack is also saved.
            if scanner is not None and scanner.ndim == 3 and scanner.shape[2] == slices*volumes:
                arrays['scanner_preview'] = np.abs(scanner[:, :, slice_index + slices*volume_index])
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise FloatingPointError(f'Nonfinite result in {frame_id}')
            frame = {'id': frame_id, 'slice_index': slice_index, 'volume_index': volume_index,
                     'arrays_path': f'arrays/{frame_id}.npz',
                     'native_image_shape': list(arrays['inva_corrected'].shape[:2]),
                     'tikhonov': diagnostics[main_index], 'tikhonov_sweep': diagnostics}
            if old_mat is not None:
                frame['mat_regression'] = regression(old_mat, arrays, slice_index, volume_index, slices, volumes, coils)
            np.savez_compressed(out / frame['arrays_path'], **arrays)
            case['frames'].append(frame)
            print(json.dumps({'event': 'frame_complete', 'id': frame_id,
                'normal_residual': frame['tikhonov']['relative_normal_residual'],
                'reference_passed': old_mat is not None}), flush=True)
    case['elapsed_seconds'] = time.monotonic() - start
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=PROJECT.parent/'data/spen_acquired_260915')
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/pilot_260915.json')
    parser.add_argument('--out', type=Path, default=PROJECT/'runs/raw_pilot_260915')
    parser.add_argument('--lambda-relative', type=float, default=0.01)
    parser.add_argument('--lambda-sweep', type=float, nargs='+', default=[0.001, 0.01, 0.1])
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    lambdas = sorted(set([args.lambda_relative, *args.lambda_sweep]))
    if any(not np.isfinite(x) or x <= 0 for x in lambdas):
        parser.error('All regularization coefficients must be finite and positive')
    if args.threads < 1:
        parser.error('--threads must be positive')
    args.out = args.out.resolve()
    args.data = args.data.resolve()
    if args.out == args.data or args.out.is_relative_to(args.data):
        parser.error('Output cannot be inside the source data collection')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Use a new or empty output directory')
    config = json.loads(args.config.read_text())
    names = [c['id'] for c in config['cases']]
    if len(names) != len(set(names)) or any(not re.fullmatch('[a-zA-Z0-9_-]+', n) for n in names):
        parser.error('Case IDs must be unique and contain only letters, digits, hyphens and underscores')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    raw_manifest = json.loads((args.data/'raw_manifest.json').read_text())
    mat_manifest = json.loads((args.data/'manifest.json').read_text())
    scans = {s['destination_relative_path']: s for e in raw_manifest['experiments'] for s in e['scans']}
    files = {r['relative_path']: r for r in raw_manifest['records']}
    mat_by_scan = {r['raw_scan_path']: r for r in mat_manifest['records']}
    (args.out/'arrays').mkdir(parents=True)
    summary = {'created_at': datetime.now().astimezone().isoformat(),
        'data_root': str(args.data), 'run_dir': str(args.out), 'device': 'cpu',
        'status': 'running', 'planned_cases': len(config['cases']),
        'methods': ['Phase Map + InvA (PV360/PV5 reference)', 'complex per-coil Tikhonov'],
        'lambda_relative': args.lambda_relative, 'lambda_sweep': lambdas,
        'array_axes': 'PE, RO, coil; each slice and volume stored separately; ADC RO size may differ',
        'data_stage': 'Bruker binary -> sorting/reflection correction -> trajectory regridding -> RO FFT -> Phase Map -> InvA or Tikhonov',
        'scope': 'Native matrix reconstruction; no learned prior, no super-resolution, no anatomical ground truth',
        'cases': [], 'failures': []}
    write_json(args.out/'config.json', config)
    write_json(args.out/'provenance.json', {'python_executable': sys.executable,
        'torch': torch.__version__, 'numpy': np.__version__,
        'spenpy_root': str(SPENPY), 'runner_sha256': sha256(Path(__file__)),
        'spenpy_source_sha256': {str(p.relative_to(SPENPY)): sha256(p) for p in sorted((SPENPY/'spenpy').rglob('*.py'))},
        'raw_manifest_sha256': sha256(args.data/'raw_manifest.json'),
        'mat_manifest_sha256': sha256(args.data/'manifest.json')})
    write_json(args.out/'summary.json', summary)
    for spec in config['cases']:
        try:
            summary['cases'].append(run_case(spec, args.data, args.out, scans, files, mat_by_scan, lambdas, args.lambda_relative))
        except Exception as error:
            summary['failures'].append({'case': spec, 'error': str(error), 'traceback': traceback.format_exc()})
            print(json.dumps({'event': 'case_failed', 'id': spec['id'], 'error': str(error)}), flush=True)
        write_json(args.out/'summary.json', summary)
    summary['status'] = 'completed' if not summary['failures'] else 'completed_with_failures'
    summary['frame_count'] = sum(len(c['frames']) for c in summary['cases'])
    summary['finished_at'] = datetime.now().astimezone().isoformat()
    write_json(args.out/'summary.json', summary)
    print(json.dumps({'status': summary['status'], 'scans': len(summary['cases']), 'frames': summary['frame_count']}), flush=True)
    if summary['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
