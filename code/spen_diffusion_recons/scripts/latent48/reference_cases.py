"""Load the frozen 260915 SPEN figure cases for a new reconstruction method.

Both public loaders return a dictionary containing ``report`` (case list),
``arrays`` (unmodified copies of the old display NPZ), and ``metadata``. The
simulation result also has a disjoint ``calibration`` case list. Each case
contains an operator, complex ``observation`` [1,4,PE,96], ``initial_image``
[1,1,192,192] in [-1,1], ``gt`` (same coordinates; metrics only), and an
``array_index`` identifying its old display arrays. Physics is never rotated;
``display_rot180`` tells the caller how to render a new prediction.

The initial image is a native 96-grid Tikhonov solution (rho=.003), bicubic
upsampled and clipped to the VAE training interval. It uses no reference GT.
Saved baselines are returned verbatim and are never replaced by this start.

Simulation observations are read from the old NPZ. The synthetic operator and
calibration observations must use CUDA: the original torch CUDA random stream
differs from CPU even for the same seed. Replayed report observations are used
only to verify that the reconstructed operator/random recipe matches the NPZ.
Real observations were not saved in that NPZ; they are reconstructed from the
hash-verified original phase-corrected MAT data and saved normalization scalars.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
for directory in (PROJECT.parent / 'spenpy', HERE.parent / 'core',
                  HERE.parent / 'prior96', HERE.parent / 'prior192'):
    sys.path.insert(0, str(directory))

from evaluate_png_sr import observe
from operators import SpenMagnitudeOperator, scanner_matrices
from sr_operator import (SCANS, SpenSuperResolutionOperator, encoding_from_params,
                         make_sr_operator)

SCAN_ROOT = PROJECT.parent / 'data/spen_acquired_260915/mat'
SCAN_NAMES = {16: '20240321_lxj_spen_mouse_240321_1_1_1',
              24: '20240115_lxj_SPEN_96_240115_1_1_1'}
REAL_SELECTIONS = [(16, n) for n in (5, 13, 22, 30, 38)] + [
    (24, n) for n in (3, 7, 11, 15, 19)]
INITIAL_RHO = .003


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _archive(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: value.copy() for key, value in archive.items()}
    for key, value in arrays.items():
        if value.dtype.kind not in 'US' and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite reference array: {path}/{key}')
    return arrays


def _scan_paths():
    # Do not rely on the legacy SPEN_REFERENCE_ROOT defaults after migration.
    paths = {fov: SCAN_ROOT / name for fov, name in SCAN_NAMES.items()}
    SCANS.update(paths)
    return paths


@torch.no_grad()
def _initial(op96, observation):
    native = op96.proximal(torch.full((1, 1, 96, 96), -1.,
                                     device=observation.device),
                           observation, INITIAL_RHO)
    return F.interpolate(native, (192, 192), mode='bicubic',
                         align_corners=False).clamp(-1, 1)


def _simulation_targets(reference_dir, records, arrays, device):
    data = reference_dir.parent / 'data_all'
    manifest_path = data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    training_path = data / 'train.npy'
    if _sha(training_path) != manifest['npy_sha256']['train']:
        raise ValueError('Training array hash differs from the saved manifest')
    train = np.load(training_path, mmap_mode='r', allow_pickle=False)
    if train.dtype != np.uint16 or train.shape[1:] != (192, 192):
        raise ValueError('Expected uint16 native 192 training images')
    index = {record['key']: i for i, record in enumerate(manifest['records']['train'])}
    report_keys = [row['key'] for row in records['report']]
    if report_keys != arrays['case_keys'].tolist():
        raise ValueError('Saved display case order differs from simulation_cases.json')
    if set(report_keys) & {row['key'] for row in records['calibration']}:
        raise ValueError('Report and calibration cases overlap')
    targets = {}
    for role, rows in records.items():
        indices = []
        for row in rows:
            i = index[row['key']]
            if (i != row['prior_training_array_index'] or
                    row['pixel_sha256'] != manifest['records']['train'][i]['pixel_sha256']):
                raise ValueError(f'Frozen case changed: {row["key"]}')
            indices.append(i)
        unit = np.asarray(train[indices], dtype=np.float32) / 65535.
        targets[role] = torch.as_tensor(2 * unit[:, None] - 1, device=device)
    restored_report = ((targets['report'][:, 0].cpu().numpy() + 1) / 2).clip(0, 1)
    if not np.array_equal(restored_report, arrays['target']):
        raise ValueError('Training report pixels differ from the exact saved figure GT')
    provenance = dict(training_array=str(training_path.resolve()),
                      training_array_sha256=manifest['npy_sha256']['train'],
                      training_manifest_sha256=_sha(manifest_path))
    return targets, provenance


@torch.no_grad()
def load_simulation_reference(reference_dir, device):
    """Return the original 8 report and 8 calibration condition/case pairs.

    ``array_index`` is (condition_index, case_index). R1 is index 0, R2 is
    index 1. Report ``observation`` comes directly from simulation.npz; its
    full zero-filled counterpart is ``reference_arrays['observation']``.
    ``gt`` is provided for metric calculation and must not initialize a solver.
    """
    reference_dir, device = Path(reference_dir).resolve(), torch.device(device)
    if device.type != 'cuda':
        raise ValueError('Exact reference simulation requires a CUDA device: the old synthetic coils/noise used CUDA RNG')
    arrays = _archive(reference_dir / 'simulation.npz')
    metadata = json.loads((reference_dir / 'simulation.json').read_text())
    records = json.loads((reference_dir / 'simulation_cases.json').read_text())
    if arrays['observation'].shape != (2, 4, 4, 96, 96):
        raise ValueError('Expected two conditions and four frozen simulation cases')
    _scan_paths()
    targets, provenance = _simulation_targets(reference_dir, records, arrays, device)
    high = make_sr_operator(fov=16, image_size=192, device=device, seed=4517,
                            cg_max_iter=320)
    low = make_sr_operator(fov=16, image_size=96, device=device, seed=4517,
                           cg_max_iter=320)
    if not np.allclose(high.scanner_params, metadata['geometry']['full_args'],
                       rtol=0, atol=1e-12):
        raise ValueError('Scanner encoding parameters differ from the reference')
    result = dict(report=[], calibration=[], arrays=arrays, records=records,
                  metadata=metadata, provenance=provenance)
    replay_errors = []
    for condition_index, noise in enumerate(arrays['noise_sigma']):
        mask = torch.as_tensor(arrays['mask'][condition_index], device=device)
        condition_metadata = metadata['conditions'][condition_index]
        if torch.where(mask)[0].cpu().tolist() != condition_metadata['selected_pe_rows']:
            raise ValueError('Saved sampling mask and metadata disagree')
        op = SpenSuperResolutionOperator(high.a_full, high.coils, 96, mask,
                                         sigma_noise=float(noise), cg_max_iter=320)
        op96 = SpenMagnitudeOperator(low.a_full, low.coils, mask,
                                    sigma_noise=float(noise))
        full_saved = torch.as_tensor(arrays['observation'][condition_index], device=device)
        if bool(full_saved[:, :, ~mask].abs().any()):
            raise ValueError('Saved missing PE rows are not zero-filled')
        saved_observations = full_saved[:, :, mask]
        replay = observe(op, targets['report'], records['report'], float(noise))
        replay_error = float((replay - saved_observations).abs().max())
        replay_errors.append(replay_error)
        if replay_error > 2e-5:
            raise ValueError(f'Original simulation operator/noise replay failed: {replay_error:.6g}')
        calibration_observations = observe(op, targets['calibration'],
                                           records['calibration'], float(noise))
        for role, observations in [('report', saved_observations),
                                   ('calibration', calibration_observations)]:
            for case_index, record in enumerate(records[role]):
                observation = observations[case_index:case_index + 1]
                references = ({name: values[condition_index, case_index].copy()
                               for name, values in arrays.items()
                               if values.ndim >= 4 and values.shape[:2] == (2, 4)}
                              if role == 'report' else {})
                result[role].append(dict(
                    key=record['key'], casekey=record['key'], role=role,
                    condition=f'R{condition_index + 1}', condition_index=condition_index,
                    case_index=case_index, array_index=(condition_index, case_index),
                    sigma_noise=float(noise), op=op, op96=op96,
                    observation=observation, initial_image=_initial(op96, observation),
                    gt=targets[role][case_index:case_index + 1], record=record,
                    display_rot180=False, reference_arrays=references,
                    metadata=dict(initialization='native96 Tikhonov, bicubic to192, clip[-1,1]',
                                  initial_rho=INITIAL_RHO, display_window=[0, 1],
                                  observation_source='saved NPZ' if role == 'report' else 'original CUDA stable_seed recipe',
                                  report_observation_replay_max_abs=replay_error)))
    result['provenance'].update(reference_dir=str(reference_dir),
        simulation_npz_sha256=_sha(reference_dir / 'simulation.npz'),
        observation_replay_max_abs=replay_errors,
        initialization_rho=INITIAL_RHO,
        scope='Calibration/report disjoint; both groups were included in prior training')
    return result


@torch.no_grad()
def _real_physics(path, saved, device):
    """Mirror evaluate_real_sr.load_case, freezing its saved scalar estimates.

    The original figure NPZ omitted y/coils. Reuse the original InvA/encoding
    implementation and fixed MAT phase correction; saved gain/scale avoid
    hardware-dependent reductions changing the reference normalization.
    """
    if _sha(path) != saved['source_sha256']:
        raise ValueError(f'Original real-data MAT hash changed: {path}')
    inv, a, params = scanner_matrices(path, device)
    if not np.allclose(params, saved['full_args'], rtol=0, atol=1e-12):
        raise ValueError('Original real-data encoding parameters changed')
    raw = np.asarray(scipy.io.loadmat(path,
        variable_names=['spen_phase_corrected_signal_rofft'])['spen_phase_corrected_signal_rofft'])
    if raw.shape != (96, 96, 1, 4):
        raise ValueError(f'Unexpected scanner signal axes: {raw.shape}')
    signal = torch.as_tensor(raw[:, :, 0, :], dtype=torch.complex64,
                             device=device).permute(2, 0, 1)[None]
    scale = torch.tensor(saved['magnitude_scale'], dtype=torch.float32, device=device)
    smax = torch.tensor(saved['source_encoding_smax'], dtype=torch.float32, device=device)
    gain = torch.tensor(complex(*saved['scalar_gain']), dtype=torch.complex64, device=device)
    z = torch.einsum('ij,bcjw->bciw', inv, signal) * gain
    rss = z.abs().square().sum(1, keepdim=True).sqrt()
    coils = z / rss.clamp_min(scale * 1e-8)
    observation = signal / (scale * smax)
    anchor = 2 * rss / scale - 1
    op96 = SpenMagnitudeOperator(a / smax, coils, sigma_noise=.02)
    a192 = encoding_from_params(params, 192, device) / smax
    c192 = F.interpolate(coils.real, (192, 192), mode='bilinear', align_corners=False)
    c192 = c192 + 1j * F.interpolate(coils.imag, (192, 192), mode='bilinear', align_corners=False)
    c192 = c192 / c192.abs().square().sum(1, keepdim=True).sqrt().clamp_min(1e-8)
    op = SpenSuperResolutionOperator(a192, c192, measurement_size=96,
                                     sigma_noise=.02, cg_max_iter=320)
    return op96, op, observation, anchor


@torch.no_grad()
def load_real_reference(reference_dir, device):
    """Return the 10 frozen real cases in figure order, with no GT.

    Existing ``arrays``/``reference_arrays`` are already rotated 180 degrees.
    New physics outputs and initial images are native orientation, so callers
    must rotate new predictions for display (``display_rot180=True``).
    """
    reference_dir, device = Path(reference_dir).resolve(), torch.device(device)
    arrays = _archive(reference_dir / 'real.npz')
    metadata = json.loads((reference_dir / 'real.json').read_text())
    paths = _scan_paths()
    identities = [(row['fov_mm'], row['export_index']) for row in metadata['cases']]
    if identities != REAL_SELECTIONS or arrays['degraded'].shape != (10, 96, 96):
        raise ValueError('Real reference must contain the original 10 cases in figure order')
    cases = []
    for case_index, (saved, (fov, number)) in enumerate(zip(metadata['cases'], identities)):
        if (arrays['fov_mm'][case_index] != fov or
                arrays['labels'][case_index] != f'Acquisition #{number}'):
            raise ValueError('Real NPZ labels disagree with source metadata')
        path = paths[fov] / f'slice_{number}.mat'
        op96, op, observation, anchor = _real_physics(path, saved, device)
        restored_anchor = F.interpolate(anchor, (192, 192), mode='bicubic', align_corners=False)
        restored_display = np.rot90((restored_anchor[0, 0].cpu().numpy() + 1) / 2, 2)
        reference_anchor = arrays['phase_inva_unclipped'][case_index]
        anchor_error = float(np.max(np.abs(restored_display - reference_anchor)))
        if anchor_error > 2e-5:
            raise ValueError(f'Real phase/coil normalization replay failed: {path}: {anchor_error:.6g}')
        # The old Tikhonov image independently checks operator/measurement scale.
        old_tikh = torch.as_tensor(np.rot90(arrays['tikhonov_unclipped'][case_index], 2).copy(),
                                   device=device)[None, None] * 2 - 1
        residual = float(op.relative_residual(old_tikh, observation))
        expected_residual = saved['methods']['tikhonov']['measurement_nrmse']
        if abs(residual - expected_residual) > 2e-5:
            raise ValueError(f'Real forward-model replay failed: {path}')
        references = {name: values[case_index].copy() for name, values in arrays.items()
                      if values.ndim == 3}
        key = f'fov{fov}_slice_{number}'
        cases.append(dict(key=key, casekey=key, role='report', case_index=case_index,
            array_index=case_index, condition_index=0 if fov == 16 else 1,
            fov_mm=fov, export_index=number, sigma_noise=float(metadata['noise_sigma']),
            op=op, op96=op96, observation=observation,
            initial_image=_initial(op96, observation), gt=None,
            display_rot180=True, reference_arrays=references,
            metadata=dict(saved, path=str(path), initial_rho=INITIAL_RHO,
                initialization='native96 Tikhonov, bicubic to192, clip[-1,1]',
                anchor_replay_max_abs=anchor_error,
                tikhonov_residual_replay_abs=abs(residual - expected_residual),
                observation_source='hash-verified MAT phase-corrected signal with saved normalization scalars')))
    return dict(report=cases, cases=cases, arrays=arrays, metadata=metadata,
        provenance=dict(reference_dir=str(reference_dir),
            real_npz_sha256=_sha(reference_dir / 'real.npz'), scan_root=str(SCAN_ROOT),
            initialization_rho=INITIAL_RHO,
            observation_note='Original real.npz did not store observations; read exact MAT and freeze saved gain/magnitude/encoding scale',
            scope='Real acquisitions; no paired high-resolution ground truth'))
