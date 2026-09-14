"""Real scanner adapter. Fixed phase/coil estimates use only each measured case."""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares
from operators import NativeGridXSPENOperator, encoding_matrices

def correct_even_odd(ro):
    """Fit one smooth parity phase from neighboring RO-image rows, pooling coils.

    Spatially varying object/coil phase is approximately cancelled by the
    symmetric neighbor interpolation. The transform is unit modulus and uses
    only the observed complex signal. Its assumptions remain a nuisance model.
    """
    b, c, m, k = ro.shape
    if b != 1 or m < 6:
        raise ValueError('Single scanner case with at least six PE rows required')
    rows = torch.arange(1, m-1, 2, device=ro.device)
    target = (ro[:, :, rows-1]+ro[:, :, rows+1])*.5
    cross = (ro[:, :, rows]*target.conj()).sum(1)[0]
    # Circular fitting avoids unwrapping low-SNR background into large 2pi ramps.
    phase = np.angle(cross.cpu().numpy())
    weight = cross.abs().cpu().numpy()
    yy, xx = np.meshgrid((rows.cpu().numpy()-m/2)/m*2, (np.arange(k)-k/2)/k*2, indexing='ij')
    design = np.stack([np.ones_like(xx), xx, yy, xx*xx, xx*yy, yy*yy], axis=-1)
    mask = weight > np.percentile(weight, 75)*.2
    w = np.sqrt(weight[mask]/max(float(weight.max()), 1e-12))
    d, observed = design[mask], phase[mask]
    start = np.zeros(6)
    # Readout timing mismatch may wrap by multiple turns across the FOV.
    # Find a coarse global linear phase before the local circular quadratic fit.
    best = -1.
    for slope_y in np.arange(-4., 4.1, 1.):
        slope_x = np.arange(-20., 20.1, 1.)
        ph = observed[:, None]-d[:, 1, None]*slope_x-d[:, 2, None]*slope_y
        coherence = (weight[mask, None]*np.exp(1j*ph)).sum(0)
        j = int(np.abs(coherence).argmax())
        if abs(coherence[j]) > best:
            best = abs(coherence[j])
            start[:3] = [np.angle(coherence[j]), slope_x[j], slope_y]
    def residual(coeff):
        fit = d@coeff
        return np.concatenate([w*(np.cos(fit)-np.cos(observed)), w*(np.sin(fit)-np.sin(observed))])
    coeff = least_squares(residual, start, max_nfev=100).x
    yy, xx = np.meshgrid((np.arange(m)-m/2)/m*2, (np.arange(k)-k/2)/k*2, indexing='ij')
    full = np.stack([np.ones_like(xx), xx, yy, xx*xx, xx*yy, yy*yy], axis=-1)@coeff
    full[::2] = 0
    field = torch.as_tensor(full, device=ro.device, dtype=ro.real.dtype)
    return ro*torch.exp(-1j*field), field, coeff.tolist()

def resize_complex(x, size):
    return torch.complex(F.interpolate(x.real[None], size=size, mode='bilinear', align_corners=False)[0],
                         F.interpolate(x.imag[None], size=size, mode='bilinear', align_corners=False)[0])

@torch.no_grad()
def load_case(path, slice_index, repeat=0, device='cuda', size=128):
    with h5py.File(path) as h5:
        meta = json.loads(h5.attrs['metadata'])
        raw = torch.as_tensor(h5['kspace'][repeat, slice_index], device=device)[None]
    m, k = raw.shape[-2:]
    an, fn = encoding_matrices(m, k, m, k, meta['r_value'], meta['beta'], device=device)
    norm = torch.linalg.svdvals(an).max()
    an = an/norm
    # A native, mildly regularized complex inverse provides measured coil/phase factors.
    ro_original = torch.matmul(raw, fn.conj())
    ro, phase_field, phase_coeff = correct_even_odd(ro_original)
    corrected_raw = torch.matmul(ro, fn.T)
    regularizer = .01
    inverse = torch.linalg.solve(an.conj().T@an+regularizer*torch.eye(m, device=device), an.conj().T)
    native = torch.matmul(inverse, ro)[0]
    native_rss = native.abs().square().sum(0).sqrt()
    rss = F.interpolate(native_rss[None, None], (size, size), mode='bilinear', align_corners=False)[0, 0]
    scale = rss.quantile(.995).clamp_min(1e-8)
    native_coils = native/native_rss.clamp_min(scale*1e-4)
    # No independent sensitivity/phase calibration scan is available for this adapter.
    # Keep nuisance factors at native resolution. Unknown HR phase is not inferred
    # by interpolating wrapped complex values; use exact overlap averages instead.
    op = NativeGridXSPENOperator(an, fn, native_coils, image_shape=(size, size))
    obs = corrected_raw/scale
    x_anchor = (rss/scale*2-1)[None, None]
    modeled = op.forward(x_anchor)
    # A global complex receiver gain reconciles native and HR quadrature units.
    gain = (modeled.conj()*obs).sum()/modeled.abs().square().sum().clamp_min(1e-12)
    op = NativeGridXSPENOperator(an, fn, native_coils*gain, image_shape=(size, size))
    op.phase_correction = phase_field
    op.original_observation = raw/scale
    baseline = op.proximal(torch.full_like(x_anchor, -1.), obs, .003)
    info = dict(scan=Path(path).stem, slice_index=int(slice_index), repeat=int(repeat),
                position_lps_mm=meta['positions_lps_mm'][slice_index],
                native_shape=[m, k], output_shape=[size, size], fov_mm=meta['fov_mm'],
                r_value=meta['r_value'], beta=meta['beta'], source_sha256=meta['source_sha256'],
                magnitude_scale=float(scale), gain=[float(gain.real), float(gain.imag)],
                native_regularization=regularizer, magnitude_regularization=.003,
                anchor_residual=float(op.relative_residual(x_anchor, obs)),
                baseline_residual=float(op.relative_residual(baseline, obs)),
                sigma_noise_assumption=.02,
                even_odd_phase_polynomial=phase_coeff,
                phase_correction='Unit-modulus polynomial in RO image domain, estimated from symmetric adjacent rows. All real DC is in this corrected measurement domain.',
                model_status=meta['calibration_status'],
                discretization='Native complex coil/object-phase grid with exact overlap projection from HR magnitude: A_native(S_native*(P*m*Q.T))*F_native.T',
                nuisance='Complex coil/object-phase factors estimated from the same measured native Tikhonov image. No clean target.',
                limitations='No independently validated WIP mapping, slice profile, B0/waveforms or direction table; output grid is not proof of achieved spatial resolution.')
    raw_rss = ro_original.abs().square().sum(1, keepdim=True).sqrt()
    raw_rss = F.interpolate(raw_rss, size=(size, size), mode='bilinear', align_corners=False)
    return op, obs, baseline, x_anchor, raw_rss/scale, info
