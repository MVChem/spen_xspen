"""Grid-independent measured nuisance calibration for rectangular xSPEN images.

The receiver scale, complex gain and phase are estimated once on the acquired
grid. Only the finite-volume magnitude projection depends on the output grid.
The reduced encoding model is inherited; this is not waveform calibration.
"""
import json
import sys
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from operators import NativeGridXSPENOperator, XSPENOperator, encoding_matrices
from scanner import correct_even_odd


def resize_magnitude(x, shape):
    shape = tuple(int(n) for n in shape)
    if x.shape[-2:] == shape:
        return x
    return F.interpolate(x, size=shape, mode='bilinear', align_corners=False,
                         antialias=True)


@torch.no_grad()
def load_case(path, slice_index, repeat=0, device='cpu', image_shape=None):
    with h5py.File(path) as source:
        meta = json.loads(source.attrs['metadata'])
        raw = torch.as_tensor(source['kspace'][repeat, slice_index], device=device)[None]
    m, k = raw.shape[-2:]
    shape = (m, k) if image_shape is None else tuple(int(n) for n in image_shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError('image_shape must contain two positive dimensions')
    a, f = encoding_matrices(m, k, m, k, meta['r_value'], meta['beta'], device=device)
    encoding_norm = torch.linalg.svdvals(a).max()
    a = a / encoding_norm
    ro_original = raw @ f.conj()
    ro, phase_field, phase_coeff = correct_even_odd(ro_original)
    observation_raw = ro @ f.T
    regularizer = .01
    inverse = torch.linalg.solve(a.mH @ a + regularizer * torch.eye(m, device=device), a.mH)
    native = (inverse @ ro)[0]
    native_rss = native.abs().square().sum(0).sqrt()
    scale = native_rss.quantile(.995).clamp_min(1e-8)
    coils = native / native_rss.clamp_min(scale * 1e-4)
    observation = observation_raw / scale
    native_anchor = (native_rss / scale * 2 - 1)[None, None]
    native_op = XSPENOperator(a, f, coils)
    modeled = native_op.forward(native_anchor)
    gain = (modeled.conj() * observation).sum() / modeled.abs().square().sum().clamp_min(1e-12)
    op = NativeGridXSPENOperator(a, f, coils * gain, image_shape=shape)
    anchor = resize_magnitude(native_rss[None, None] / scale, shape) * 2 - 1
    baseline = op.proximal(torch.full_like(anchor, -1), observation, .003)
    raw_rss = resize_magnitude(ro_original.abs().square().sum(1, keepdim=True).sqrt() / scale, shape)
    op.original_observation = raw / scale
    op.phase_correction = phase_field
    info = dict(scan=Path(path).stem, slice_index=int(slice_index), repeat=int(repeat),
                native_shape=[m, k], output_shape=list(shape),
                position_lps_mm=meta['positions_lps_mm'][slice_index],
                fov_mm=meta['fov_mm'], thickness_mm=meta['thickness_mm'],
                r_value=meta['r_value'], beta=meta['beta'],
                source_sha256=meta['source_sha256'], magnitude_scale=float(scale),
                encoding_spectral_norm=float(encoding_norm), gain=[float(gain.real), float(gain.imag)],
                native_regularization=regularizer, magnitude_regularization=.003,
                even_odd_phase_polynomial=phase_coeff,
                anchor_residual=float(op.relative_residual(anchor, observation)),
                baseline_residual=float(op.relative_residual(baseline, observation)),
                model_status=meta['calibration_status'],
                normalization='Scale and complex gain fixed on acquired grid; identical for every output grid.',
                discretization='A_native(S_native * (P * magnitude * Q.T)) * F_native.T',
                phase_correction='Unit-modulus quadratic parity phase in RO-image domain; DC uses corrected observations.',
                nuisance='Coil/object phase and receiver gain estimated from measured native Tikhonov anchor.',
                limitations='No clean real GT, independent coil calibration, verified direction table or waveform/B0 calibration. Output spacing is not achieved resolution.')
    return op, observation, baseline, anchor, raw_rss, info
