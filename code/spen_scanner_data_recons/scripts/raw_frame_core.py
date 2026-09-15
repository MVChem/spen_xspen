"""Frame-preserving Bruker SPEN reconstruction used by the collection runner.

Public API: ``prepare_scan(path, flavor, trajectory_path)`` followed by
``reconstruct_frame(context, slice_index, volume_index, echo_index)``.
Scanner acquisition axes are kept separate. Multi-shot phase correction and
the xSPEN forward model are explicitly outside the validated Python model.
"""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import sys

import numpy as np
import torch

SPENPY = Path(__file__).resolve().parents[2] / 'spenpy'
if str(SPENPY) not in sys.path:
    sys.path.insert(0, str(SPENPY))
from spenpy._legacy.bruker.param import read_pv_param
from spenpy._legacy.bruker.raw import read_bruker_kspace_pv360_fid_multichannel
from spenpy._legacy.core.matrix import calcInvA
from spenpy._legacy.fft.transform import fft_kspace_to_xspace
from spenpy._legacy.recon.gridding import smooth_trajectory, _place_regridded_block
from spenpy._legacy.recon.phase import apply_pv360_one_shot_phase_correction
from segmented_raw_reader import read_segmented_raw
from batched_gridding import regrid_all_preserving_scalar


def _param(path, name, default=None):
    value = read_pv_param(str(path), name)
    if value is None:
        return default
    return value.tolist() if isinstance(value, np.ndarray) else value


def _scalar(path, name, default=None):
    value = _param(path, name, default)
    return None if value is None else np.asarray(value).reshape(-1)[0].item()


def _np(tensor):
    return tensor.detach().cpu().resolve_conj().numpy()


def _normalize_legacy(raw, coils, echoes, segments):
    """Preserve volume/receiver order; never merge receivers into the echo axis."""
    raw = np.asarray(raw)
    if echoes == 1:
        if raw.ndim != 5 or raw.shape[-1] != 1:
            raise ValueError(f'Unsupported single-echo reader layout {raw.shape}')
        flat = raw
    elif segments == 1:
        # Odd-single reader: RO, PE, slice, volume*coil, echo, singleton.
        if raw.ndim != 6 or raw.shape[-2:] != (echoes, 1):
            raise ValueError(f'Unsupported single-shot multi-echo layout {raw.shape}')
        flat = raw[..., 0]
    else:
        # Corrected segmented reader has its own canonical output below.
        raise ValueError('Use the segmented multi-echo reader for this layout')
    ro, pe, slices, frames, _ = flat.shape
    if frames % coils:
        raise ValueError(f'{frames} receiver frames cannot be split into {coils} coils')
    return flat.reshape(ro, pe, slices, frames // coils, coils, echoes, order='F')


def _read_segmented_multiecho(scan_dir, params):
    """Unpack the same acquisition order as the preserved MATLAB reader.

    Fixes the Python translation's missing singleton and incorrectly ordered
    slice expression for segment interleaving. All echoes remain explicit.
    This is raw sorting only; it does not implement inter-shot phase mapping.
    """
    matrix = params['matrix_ro_pe']
    pe, segments = int(matrix[1]), params['n_segments']
    slices, volumes, coils, echoes = (params[k] for k in
                                    ('slices', 'volumes', 'coils', 'echoes'))
    if pe % segments:
        raise ValueError('PE matrix must be divisible by NSegments')
    raw_path = scan_dir / 'rawdata.job0'
    if not raw_path.exists():
        raw_path = scan_dir / 'fid'
    data = np.fromfile(raw_path, dtype='<i4').astype(np.float64)
    denominator = 2 * pe * slices * volumes * coils * echoes
    if not data.size or data.size % denominator:
        raise ValueError('Raw byte count does not match segmented acquisition dimensions')
    ro = data.size // denominator
    # ADC complex pair, readout, PE within shot, receiver, echo, slice,
    # shot, volume. This includes singleton receiver axes explicitly.
    packed = data.reshape(2, ro, pe // segments, coils, echoes,
                          slices, segments, volumes, order='F')
    acquired = (packed[0] + 1j * packed[1]).transpose(0, 1, 4, 5, 6, 2, 3)
    out = np.empty((ro, pe, slices, volumes, coils, echoes), np.complex128)
    for shot in range(segments):
        block = acquired[:, :, :, shot].copy()
        # Same MATLAB reflected-line convention as the single-echo reader.
        start = 0 if segments % 2 == 0 or shot % 2 else 1
        block[:, start::2] = block[::-1, start::2].copy()
        out[:, shot::segments] = block
    target_ro = int(matrix[0])
    if ro < target_ro:
        padded = np.zeros((target_ro, *out.shape[1:]), out.dtype)
        start = (target_ro - ro) // 2
        padded[start:start + ro] = out
        out = padded
    order = np.asarray(_param(scan_dir, 'PVM_ObjOrderList', list(range(slices)))).reshape(-1)
    if order.size != slices or set(order.tolist()) != set(range(slices)):
        raise ValueError('Invalid PVM_ObjOrderList; cannot label acquired slices')
    reordered = np.empty_like(out)
    reordered[:, :, order.astype(int)] = out
    return reordered


def _nufft_linear_operator(knots, n_out, input_size, accuracy=6):
    """Exact linear form of the existing FGG operator, evaluated in a batch.

    The preserved scalar implementation repeats the same spread/FFT for each
    readout line. Building its matrix once makes all slices and coils share
    the same numerical operator, without approximating the trajectory.
    """
    knots = np.asarray(knots, np.float64).reshape(-1)
    if knots.size != input_size:
        raise ValueError(f'Trajectory has {knots.size} samples but ADC has {input_size}')
    if not np.isfinite(knots).all() or np.ptp(knots) <= 0 or n_out < 2:
        raise ValueError('Invalid readout trajectory')
    m_sp, oversampling = accuracy, 2
    tau = np.pi * m_sp / (n_out * n_out * oversampling * (oversampling - .5))
    m_r = oversampling * n_out
    kmin, kmax = float(knots.min()), float(knots.max())
    scale = (n_out - 1) / (kmax - kmin)
    knots = np.mod(2 * np.pi * (scale * knots - n_out / 2 - kmin * scale) / n_out, 2 * np.pi)
    e3_pos = np.exp(-((np.pi * np.arange(1, m_sp + 1) / m_r) ** 2) / tau)
    e3 = np.r_[e3_pos[:m_sp - 1][::-1], 1., e3_pos]
    spread = np.zeros((m_r, input_size), np.complex128)
    for column, knot in enumerate(knots):
        m1 = int(np.floor(m_r * knot / (2 * np.pi)))
        x = knot - m1 * np.pi / (m_r / 2)
        e1 = np.exp(-(x*x) / (4*tau))
        e2_dummy = np.exp(x * np.pi / (m_r * tau))
        e2 = np.empty(2*m_sp)
        e2[m_sp - 1] = 1
        for j in range(m_sp, 2*m_sp):
            e2[j] = e2_dummy * e2[j-1]
        for j in range(m_sp - 2, -1, -1):
            e2[j] = e2[j+1] / e2_dummy
        for l1 in range(1-m_sp, m_sp+1):
            lx = int((m1 + l1 + m_r/2) >= 0)
            rx = int((m1 + l1) < m_r/2)
            row = int(m1 + l1 + (rx-lx)*m_r + m_r/2)
            spread[row, column] += e1 * e2[m_sp+l1-1] * e3[m_sp+l1-1]
    transformed = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(spread, axes=0), axis=0), axes=0)
    chop = int(np.floor(.5 * (oversampling-1)*n_out + .5))
    transformed = transformed[chop:chop+n_out]
    kx = np.arange(-n_out/2, n_out/2)
    transformed *= (np.sqrt(np.pi/tau) * np.exp(tau*kx*kx))[:, None] / (input_size*oversampling)
    # Legacy gridding converts the FGG image back to readout k-space.
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(transformed, axes=0), axis=0), axes=0)


@lru_cache(maxsize=128)
def _readout_operators(trajectory_tuple, input_ro, output_ro, flavor):
    trajectory, values = smooth_trajectory(np.asarray(trajectory_tuple))
    max_traj = int(np.floor(float(trajectory.max()) + .5))
    if max_traj <= 6:
        raise ValueError(f'Readout trajectory yields only {max_traj} grid points; the reference 3+3 edge removal leaves no measured signal')
    offset = float(values.max() if flavor == 'pv360' else trajectory.max())
    placement = int(np.floor(offset + .5))
    operators = []
    for knots in (trajectory, -trajectory[::-1] + offset):
        op = _nufft_linear_operator(knots, max_traj, input_ro)
        op[:3] = 0
        op[-3:] = 0
        operators.append(_place_regridded_block(op, output_ro, placement))
    return tuple(operators)


def _regrid_linear_operator_reference(raw, trajectory, matrix, segments, flavor):
    forward, reverse = _readout_operators(tuple(np.asarray(trajectory, float).reshape(-1)),
                                         raw.shape[0], int(matrix[0]), flavor)
    out = np.empty((int(matrix[0]), *raw.shape[1:]), np.complex128)
    for pe in range(raw.shape[1]):
        reverse_line = ((pe // segments) % 2 == 0) if segments % 2 == 0 else (pe % 2 == 1)
        operator = reverse if reverse_line else forward
        out[:, pe] = (operator @ raw[:, pe].reshape(raw.shape[0], -1)).reshape(out[:, pe].shape)
    return out.astype(np.complex64)


def _regrid_all(raw, trajectory, matrix, segments, flavor):
    return regrid_all_preserving_scalar(raw, trajectory, matrix, segments, flavor)


def prepare_scan(scan_dir, regrid_flavor='pv360', trajectory_scan_dir=None):
    """Read *every* slice, volume, coil and echo; assess supported recon stages.

    Missing calibration or unsupported physics keeps the raw frame data
    available as previews. Corrupt or dimensionally ambiguous raw data raises.
    """
    scan_dir = Path(scan_dir).resolve()
    trajectory_dir = Path(trajectory_scan_dir or scan_dir).resolve()
    if regrid_flavor not in ('pv360', 'pv5'):
        raise ValueError('regrid_flavor must be pv360 or pv5')
    slices = int(np.sum(_param(scan_dir, 'PVM_SPackArrNSlices', [1])))
    diff = _scalar(scan_dir, 'PVM_DwNDiffExp')
    if diff is None or diff < 1:
        diff = _scalar(scan_dir, 'DwNDiffExp', 1)
    params = {
        'method': str(_param(scan_dir, 'Method', '')),
        'matrix_ro_pe': _param(scan_dir, 'PVM_Matrix'),
        'fov_mm': _param(scan_dir, 'PVM_Fov'),
        'slices': slices, 'volumes': int(diff) * int(_scalar(scan_dir, 'PVM_NRepetitions', 1)),
        'coils': int(_scalar(scan_dir, 'PVM_EncNReceivers', 1)),
        'echoes': int(_scalar(scan_dir, 'PVM_NEchoImages', 1)),
        'n_segments': int(_scalar(scan_dir, 'NSegments', 1)),
        'spen_gy_gauss_cm': _scalar(scan_dir, 'SpenGyGaussStren'),
        'spatial_encoding_duration_ms': _scalar(scan_dir, 'SpatEncDuration'),
        'phase1_offset_mm': _scalar(scan_dir, 'PVM_SPackArrPhase1Offset', 0.),
        'slice_thickness_mm': _param(scan_dir, 'PVM_SliceThick'),
        'effective_b_values_s_mm2': _param(scan_dir, 'PVM_DwEffBval'),
        'regrid_flavor': regrid_flavor,
    }
    if any(params[k] < 1 for k in ('slices', 'volumes', 'coils', 'echoes', 'n_segments')):
        raise ValueError('Invalid acquisition dimension in source parameters')
    dimension_correction = None
    # This archived scan repeats the same 13 acquisitions in two method
    # counters. Its independent acquisition header and exact byte count show
    # 13 volumes, not the 169 obtained by multiplying those counters. Keep the
    # source method untouched and restrict the correction to this exact case.
    if (scan_dir.parent.name, scan_dir.name) == ('lxj_motionRARE_SPEN_230904.lG2', '8'):
        evidence = (params['matrix_ro_pe'] == [64, 64] and params['slices'] == 1
                    and params['coils'] == 4 and params['echoes'] == 1
                    and params['n_segments'] == 1 and params['volumes'] == 169
                    and _scalar(scan_dir, 'NR') == 13 and _scalar(scan_dir, 'NI') == 1
                    and (scan_dir/'fid').stat().st_size == 64*64*4*8*13)
        if not evidence:
            raise ValueError('Known scan-8 dimension conflict no longer matches independently verified header/byte evidence')
        dimension_correction = {'declared_method_volumes': 169, 'used_volumes': 13,
                                'reason': 'PVM_DwNDiffExp=13 and PVM_NRepetitions=13 double-count the same series',
                                'evidence': 'acqp NR=13, NI=1; fid bytes=64*64*4 receivers*8 bytes*13 volumes'}
        params['volumes'] = 13
    reader_info = None
    if params['n_segments'] > 1:
        raw, reader_info = read_segmented_raw(scan_dir, params)
        dimension_correction = reader_info['dimension_correction']
        reader = reader_info['reader']
    elif dimension_correction is not None:
        raw = _read_segmented_multiecho(scan_dir, params)
        reader = 'explicit MATLAB-order acquisition-packet unpacking'
    else:
        original = read_bruker_kspace_pv360_fid_multichannel(str(scan_dir))
        raw = _normalize_legacy(original, params['coils'], params['echoes'], params['n_segments'])
        reader = 'preserved Bruker reader with explicit volume/coil/echo separation'
    expected = tuple(params[k] for k in ('slices', 'volumes', 'coils', 'echoes'))
    if raw.shape[2:] != expected:
        raise ValueError(f'Read shape {raw.shape[2:]} differs from slice/volume/coil/echo {expected}')
    # Int32 ADC samples can exceed complex64's exact integer range. Preserve
    # them through regridding, matching the reference raw -> double -> grid
    # -> complex64 path; early rounding can change PhaseMap optimizer minima.
    raw = np.ascontiguousarray(raw, dtype=np.complex128)
    if not np.isfinite(raw).all():
        raise ValueError('Raw sorting produced nonfinite values')
    reasons = []
    trajectory = _param(trajectory_dir, 'PVM_EpiTrajAdjkx')
    regridded = None
    regrid_error = None
    if trajectory is not None and np.any(np.asarray(trajectory) > 0):
        try:
            regridded = _regrid_all(raw, trajectory, params['matrix_ro_pe'], params['n_segments'], regrid_flavor)
        except (ValueError, IndexError, FloatingPointError) as exc:
            regrid_error = str(exc)
            reasons.append('readout_trajectory_regridding_failed: ' + str(exc))
    else:
        reasons.append('no_nonzero_readout_trajectory; uniform ADC sampling is not established')
    method = params['method'].lower()
    quadratic = 'spen' in method and 'xspen' not in method and 'spec' not in method
    if not quadratic:
        reasons.append('xSPEN/non-quadratic acquisition has no validated local forward model')
    physical = all(params[k] is not None for k in
                   ('matrix_ro_pe', 'fov_mm', 'spen_gy_gauss_cm', 'spatial_encoding_duration_ms'))
    if not physical:
        reasons.append('missing quadratic-SPEN physical parameters')
    encoding_supported = quadratic and physical and regridded is not None
    if params['n_segments'] != 1:
        reasons.append('multi-shot inter-shot PhaseMap is present in MATLAB but not validated in Python')
    if raw.shape[1] % 2:
        reasons.append('reference odd/even PhaseMap requires equal odd/even PE counts; all actually acquired PE lines retained')
    phase_supported = encoding_supported and params['n_segments'] == 1 and raw.shape[1] % 2 == 0
    scope = {
        'phase_map_supported': bool(phase_supported),
        'encoding_supported': bool(encoding_supported),
        'regridded': regridded is not None,
        'status': 'phase_map_inva' if phase_supported else ('partial_inva_without_phase' if encoding_supported else 'preview_only'),
        'reason': '; '.join(reasons),
        'regrid_error': regrid_error,
    }
    return {'scan_dir': str(scan_dir), 'trajectory_dir': str(trajectory_dir),
            'sorted_samples': raw, 'regridded_samples': regridded,
            'counts': {k: params[k] for k in ('slices', 'volumes', 'coils', 'echoes')},
            'parameters': params, 'scope': scope, 'reader': reader,
            'reader_info': reader_info,
            'dimension_correction': dimension_correction,
            'axes': ['RO', 'PE', 'slice', 'volume', 'coil', 'echo'], '_matrix_cache': {}}


def _matrices(context, echo_index):
    parity = echo_index % 2
    if parity not in context['_matrix_cache']:
        params = context['parameters']
        pe = context['sorted_samples'].shape[1]
        lpe = float(params['fov_mm'][1]) / 10
        tp = float(params['spatial_encoding_duration_ms']) / 1000
        phase_factor = -1 if parity else 1
        a = -phase_factor * 2 * math.pi * 4.2574e3 * float(params['spen_gy_gauss_cm']) * tp / lpe
        shift = float(params['phase1_offset_mm'])
        inv_a, a_final = calcInvA(a, lpe, pe, shift, 1, 0, .8)
        odd_inv = even_inv = None
        if pe % 2 == 0:
            odd_inv, _ = calcInvA(a, lpe, pe//2, shift, 1, 0, .8)
            even_inv, _ = calcInvA(a, lpe, pe//2, shift, 1, .5, .8)
        context['_matrix_cache'][parity] = (inv_a, a_final, odd_inv, even_inv)
    return context['_matrix_cache'][parity]


def reconstruct_frame(context, slice_index, volume_index, echo_index=0):
    """Return arrays in PE, RO, coil order and honest per-method status.

    Supported single-shot frames have all six pilot keys. Partial segmented
    frames expose ``inva_uncorrected`` and ``encoding`` only; consumers must
    label Tikhonov on ``rofft_original`` as lacking PhaseMap correction.
    """
    selector = (slice(None), slice(None), slice_index, volume_index, slice(None), echo_index)
    raw = context['sorted_samples'][selector]
    source = context['regridded_samples']
    sampled = raw if source is None else source[selector]
    # MATLAB's echo loop reverses PE and the chirp sign on echoes 2,4,... .
    if echo_index % 2:
        sampled = sampled[:, ::-1]
    cmplx = torch.from_numpy(np.ascontiguousarray(sampled.transpose(1, 0, 2)[:, :, None, :], dtype=np.complex128))
    rofft = fft_kspace_to_xspace(cmplx, dim=1)
    arrays = {'sorted_samples': raw.transpose(1, 0, 2),
              'rofft_original': _np(rofft)[:, :, 0, :]}
    metadata = {'slice_index': int(slice_index), 'volume_index': int(volume_index),
                'echo_index': int(echo_index), 'echo_number': int(echo_index+1),
                'even_echo_pe_reversed': bool(echo_index % 2),
                'phase_map_status': 'applied' if context['scope']['phase_map_supported'] else 'not_applied',
                'reconstruction_status': context['scope']['status'],
                'scope_reason': context['scope']['reason'],
                'rofft_stage': 'trajectory_regridded' if source is not None else 'unregridded_ADC_preview',
                'coil_combination': 'RSS preview; complex per-coil arrays retained'}
    if context['scope']['encoding_supported']:
        inv_a, encoding, odd, even = _matrices(context, echo_index)
        arrays['encoding'] = _np(encoding)
        arrays['inva_weighted_adjoint'] = _np(inv_a)
        arrays['inva_uncorrected'] = np.einsum('ij,jrc->irc', arrays['inva_weighted_adjoint'], arrays['rofft_original'])
        if context['scope']['phase_map_supported']:
            corrected = apply_pv360_one_shot_phase_correction(
                rofft, inv_a, odd, even, optimize=True,
                smooth_motion_phase_between_shots=context['parameters']['regrid_flavor'] != 'pv5')
            arrays['rofft_corrected'] = _np(corrected)[:, :, 0, :]
            arrays['inva_corrected'] = np.einsum('ij,jrc->irc', arrays['inva_weighted_adjoint'], arrays['rofft_corrected'])
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise FloatingPointError('Nonfinite frame reconstruction')
    return arrays, metadata
