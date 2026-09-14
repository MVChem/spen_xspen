"""Independent native xSPEN PhaseMap + windowed InvA, with explicit provenance.

This preserves the 2026-09-09 comparison method: reconstruct odd/even partial
coil images with distinct windowed sinc adjoints, fit their cross-product phase,
correct original even RO rows, then apply the full windowed adjoint and RSS.
It does not reuse the diffusion adapter's neighboring-row parity nuisance fit.

The same-acquisition xSPEN_Siemens.m contains RO filtering/FFT/RSS only, not
PhaseMap/InvA. Therefore this method is an xSPEN sinc adaptation of the local
notebook PhaseMap method, not a port of an original calibrated xSPEN MATLAB
PhaseMap pipeline. The separate hybrid-SPEN MATLAB polynomial/unwrap and
decode/correct/re-encode algorithm is deliberately not silently substituted.
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
WORKSPACE = Path(os.environ.get('XSPEN_LEGACY_ROOT', str(PROJECT/'data/legacy'))).expanduser().resolve()
SPENPY = PROJECT.parent / 'spenpy'
for path in (PROJECT, SPENPY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from operators import encoding_matrices
from spenpy.recon import reconstruct_even_odd_inva

ORIGINAL_MATLAB = WORKSPACE / 'xSPEN_项目/07_Eddy_Siemens_Source_Code/esaszz_xSPEN_180c180c_Diff_CODE_ONLY/xSPEN_Siemens.m'
HYBRID_MATLAB = WORKSPACE / 'xSPEN_项目/07_Eddy_Siemens_Source_Code/DATA_Siemens_code/RunSiemensRawReffless1ShotHybridSPENEvenOddFix/Reffless1ShotHybridSPENEvenOddFix.m'
OLD_COMPARISON = PROJECT / 'runs/visual_review_20260909/extend_real.py'
OLD_MATRICES = PROJECT / 'runs/visual_review_20260909/extend_synthetic.py'
PHASE_IMPLEMENTATION = SPENPY / 'spenpy/recon/even_odd.py'


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _numpy(x):
    return x.detach().cpu().resolve_conj().resolve_neg().numpy()


def build_matrices(m, k, r_value, beta=.5, gaussian_width=.8, dtype=torch.complex64):
    """Full/partial windowed sinc adjoints in the diffusion experiment convention.

    Native A is divided by its spectral norm. PE focus is (row-floor(m/2))/m;
    output pixels are center samples (col+.5)/size-.5. No one-row conversion
    to the separate spenpy EncodingOperator origin is performed. InvA denotes
    a Gaussian-windowed adjoint, not an algebraic or regularized inverse.
    """
    if m < 4 or m % 2 or k < 2:
        raise ValueError('Full odd/even acquisition requires even PE >=4 and RO >=2')
    if not all(math.isfinite(float(v)) for v in (r_value, beta, gaussian_width)):
        raise ValueError('Nonfinite geometry/window parameter')
    if r_value <= 0 or not 0 < beta < 1 or gaussian_width <= 0:
        raise ValueError('Require R>0, beta in (0,1), Gaussian width>0')
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError('Complex matrix dtype required')
    native, readout = encoding_matrices(m, k, m, k, r_value, beta, dtype=dtype)
    norm = torch.linalg.svdvals(native).max()
    real = native.real.dtype
    focus = (torch.arange(m, dtype=real) - m // 2) / m
    inverses = []
    for size, parity in ((m, None), (m // 2, 0), (m // 2, 1)):
        kernel, _ = encoding_matrices(m, k, size, k, r_value, beta, dtype=dtype)
        kernel = kernel / norm
        positions = focus
        if parity is not None:
            kernel, positions = kernel[parity::2], focus[parity::2]
        centers = (torch.arange(size, dtype=real) + .5) / size - .5
        distance = size * (positions[:, None] - centers[None])
        sigma = gaussian_width * size**2 / (2 * r_value)
        inverses.append((kernel * torch.exp(-distance.square() / (2 * sigma**2))).mH)
    return dict(inv_a=inverses[0], inv_odd=inverses[1], inv_even=inverses[2],
                encoding=native / norm, readout=readout, encoding_norm=float(norm))


def reconstruct_traditional(raw, meta, magnitude_scale=1.0, estimator='quadratic', gaussian_width=.8):
    """Return independent PhaseMap/InvA arrays from raw complex [coil,PE,RO].

    Inputs must precede the diffusion adapter's parity correction. All returned
    coil signals and magnitudes use raw/magnitude_scale units, with no fitted
    gain, image clipping, output resampling, or independent display normalization.
    Every output is CPU NumPy except the JSON-serializable metadata dictionary.
    """
    if estimator != 'quadratic':
        raise ValueError('Only the provenance-checked historical quadratic PhaseMap is exposed')
    if not math.isfinite(float(magnitude_scale)) or magnitude_scale <= 0:
        raise ValueError('magnitude_scale must be finite and positive')
    sequence = meta.get('sequence', '')
    if sequence and 'esaszz_xSPEN_180c180c_bipolarDiff' not in sequence:
        raise ValueError('This adapter has been checked only for crossed-chirp bipolarDiff')
    tensor = torch.as_tensor(raw)
    if tensor.is_cuda:
        raise ValueError('This traditional comparison is CPU-only; pass CPU raw data')
    if tensor.ndim != 3 or not torch.is_complex(tensor) or not torch.isfinite(tensor).all():
        raise ValueError('Expected finite complex raw [coil, PE, RO]')
    c, m, k = tensor.shape
    if c < 1:
        raise ValueError('At least one coil is required')
    dtype = torch.complex128 if tensor.dtype == torch.complex128 else torch.complex64
    matrices = build_matrices(m, k, float(meta['r_value']), float(meta.get('beta', .5)), gaussian_width, dtype=dtype)
    raw_scaled = tensor.to(dtype) / magnitude_scale
    # Positive-sign forward F: y = image @ F.T; inverse RO is y @ conj(F).
    # At the native grid this is centered FFT with norm='ortho', not IFFT.
    ro = raw_scaled @ matrices['readout'].conj()
    frame = ro.permute(1, 2, 0).contiguous()
    reconstruction = reconstruct_even_odd_inva(frame, matrices['inv_a'], matrices['inv_odd'],
                                               matrices['inv_even'], estimator=estimator,
                                               coil_combination='rss')
    uncorrected = torch.einsum('ym,mxc->yxc', matrices['inv_a'], frame)
    # The historical fitter evaluates exp(-i*phase) in float32 even when the
    # supplied signal/matrices are complex128; allow its unit-modulus rounding.
    torch.testing.assert_close(reconstruction.corrected_ro_image.abs(), frame.abs(), atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(reconstruction.corrected_ro_image[::2], frame[::2], atol=0, rtol=0)
    sources = [ORIGINAL_MATLAB, HYBRID_MATLAB, OLD_COMPARISON, OLD_MATRICES, PHASE_IMPLEMENTATION]
    details = dict(reconstruction.metadata)
    details.update(method='PhaseMap + Gaussian-windowed InvA, crossed-chirp sinc adaptation',
                   implementation='Independent partial-InvA odd/even phase; circular quadratic fit plus masked 11x11 wrapped-residual smoothing.',
                   phase_convention='angle(sum_coils(even_partial * conj(odd_partial))); MATLAB even rows (zero-based indices 1,3,...) multiplied by exp(-i*phase).',
                   correction_rows_zero_based=list(range(1, m, 2)),
                   native_shape=[m, k], coil_count=c, r_value=float(meta['r_value']), beta=float(meta.get('beta', .5)),
                   gaussian_width=gaussian_width, gaussian_sigma_formula='width * number_of_output_PE_pixels^2 / (2*R)',
                   encoding_spectral_norm=matrices['encoding_norm'], magnitude_scale=float(magnitude_scale),
                   pe_focus_origin='(row-floor(Npe/2))/Npe; output voxel centers (column+.5)/Npixels-.5',
                   readout='Same positive-exponent forward F as encoding_matrices; RO inverse raw @ conj(F), centered FFT/ortho at native grid.',
                   inv_a_kind='Gaussian-windowed conjugate transpose, not a Tikhonov solve and not a strict inverse.',
                   windowed_adjoint_only='Same full windowed InvA applied before phase correction, RSS; explicit no-phase ablation.',
                   intensity='Raw receiver units divided only by supplied magnitude_scale; no target-based gain or clipping.',
                   original_same_acquisition_matlab='xSPEN_Siemens.m implements RO Gaussian filtering, sorting and FFT+RSS; it does not implement PhaseMap/InvA.',
                   matlab_algorithm_distinction='Historical hybrid-SPEN Reffless1ShotHybridSPENEvenOddFix uses polynomial/unwrap and low-resolution decode/correct/re-encode. This crossed-chirp adapter does not claim to be that pipeline.',
                   diffusion_nuisance_distinction='No use of scanner.correct_even_odd or its adjacent raw-row phase field. This PhaseMap is estimated after separate odd/even InvA reconstruction.',
                   limitations='Sinc kernel and window are an existing local xSPEN adaptation; no independent waveform/B0 or original-MATLAB xSPEN PhaseMap calibration.',
                   source_sha256={str(path): _sha256(path) for path in sources if path.exists()})
    output = dict(magnitude=_numpy(reconstruction.magnitude),
                  windowed_adjoint_only=_numpy(uncorrected.abs().square().sum(-1).sqrt()),
                  coil_images=_numpy(reconstruction.coil_images.permute(2, 0, 1)),
                  phase_map_rad=_numpy(reconstruction.phase_map_rad),
                  phase_coefficients=_numpy(reconstruction.coefficients),
                  corrected_ro_image=_numpy(reconstruction.corrected_ro_image.permute(2, 0, 1)),
                  ro_image=_numpy(ro),
                  inv_a=_numpy(matrices['inv_a']), inv_odd=_numpy(matrices['inv_odd']), inv_even=_numpy(matrices['inv_even']),
                  metadata=details)
    if any(not np.isfinite(value).all() for value in output.values() if isinstance(value, np.ndarray)):
        raise FloatingPointError('Nonfinite traditional output')
    return output


def reconstruct_scanner(path, slice_index, repeat=0, magnitude_scale=1.0, **kwargs):
    """Read one original uncorrected scanner H5 frame without changing counters."""
    with h5py.File(path) as source:
        metadata = json.loads(source.attrs['metadata'])
        raw = source['kspace'][repeat, slice_index]
    result = reconstruct_traditional(raw, metadata, magnitude_scale=magnitude_scale, **kwargs)
    result['metadata'].update(scan=Path(path).stem, slice_index=int(slice_index), repeat=int(repeat),
                              scanner_h5=str(Path(path).resolve()), source_raw_sha256=metadata.get('source_sha256'))
    return result
