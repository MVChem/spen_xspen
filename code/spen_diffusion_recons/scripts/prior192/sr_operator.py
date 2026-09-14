"""Fixed 96-sample SPEN acquisition with a finer PE/RO reconstruction grid.

The image is a real magnitude in [-1, 1]. Complex coil fractions include fixed
object/receiver phase. The measured array is AFTER the scanner's readout FFT:
    y_c = D_RO A_PE [S_c (x + 1) / 2].
Changing the image grid never changes the scanner ky trajectory. D_RO retains
the acquired centered Fourier band and preserves constant image intensity.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

V2 = Path(__file__).resolve().parents[1] / 'prior96'
V1 = V2.parent / 'core'
sys.path.insert(0, str(V1))
from project_paths import REFERENCE_ROOT
PROJECT = REFERENCE_ROOT
sys.path.insert(0, str(V1))
from operators import SpenMagnitudeOperator, scanner_matrices
from spenpy._legacy.core.matrix import calcSRMatrixApprox

SCANS = {
    16: PROJECT / 'data/mat/20240321_lxj_spen_mouse_240321_1_1_1',
    24: PROJECT / 'data/mat/20240115_lxj_SPEN_96_240115_1_1_1',
}


def _fftc(x):
    return torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(x, dim=-1),
                                          dim=-1, norm='ortho'), dim=-1)


def _ifftc(x):
    return torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x, dim=-1),
                                           dim=-1, norm='ortho'), dim=-1)


def _band_phase(n_high, n_low, like):
    # Both grids use physical pixel centers (j - (N - 1)/2) / N, matching
    # prepare_data.physical_slice. Centered FFT grids instead use j - N/2.
    k = torch.arange(-(n_low // 2), n_low - n_low // 2,
                     device=like.device, dtype=like.real.dtype)
    return torch.exp(1j * math.pi * k * (1 / n_high - 1 / n_low))


def readout_resize(x, output_size):
    """Bandlimit complex RO image N -> M, M <= N; constant one maps to one.

    Scanner convention is image->kspace IFFT and kspace->image FFT (see
    spenpy/fft/transform.py). With unitary FFTs the amplitude factor is sqrt(M/N).
    The retained even-length band has frequencies [-M/2, ..., M/2-1].
    """
    n, m = x.shape[-1], int(output_size)
    if not 0 < m <= n or n % 2 or m % 2:
        raise ValueError('Readout dimensions must be positive even integers, M <= N')
    if n == m:
        return x
    start = (n - m) // 2
    spectrum = _ifftc(x)[..., start:start + m] * _band_phase(n, m, x)
    return _fftc(spectrum) * math.sqrt(m / n)


def readout_adjoint(y, input_size):
    """Exact complex adjoint of readout_resize, including amplitude and phase.

    This is not intensity-preserving interpolation: for 2x grids D*1 = 0.5.
    The constant-preserving Fourier interpolation is (N/M) D*.
    """
    n, m = int(input_size), y.shape[-1]
    if not 0 < m <= n or n % 2 or m % 2:
        raise ValueError('Readout dimensions must be positive even integers, M <= N')
    if n == m:
        return y
    spectrum = _ifftc(y) * _band_phase(n, m, y).conj()
    padded = spectrum.new_zeros(*spectrum.shape[:-1], n)
    start = (n - m) // 2
    padded[..., start:start + m] = spectrum
    return _fftc(padded) * math.sqrt(m / n)


def encoding_from_params(params, image_size, device='cpu'):
    """Unnormalized [original NumPE, image_size] matrix on the same trajectory.

    Reuses the exact calcSRMatrixApprox implementation paired with scanner
    exports. In particular b is calculated from the ORIGINAL acquisition-grid
    partitions; substituting the high-resolution partitions changes the scan.
    """
    a, length, num_pe, shift, sign, ky1, _ = map(float, params)
    m, h = int(num_pe), int(image_size)
    if h <= 0:
        raise ValueError('image_size must be positive')
    dtype = torch.float64
    p_native = sign * torch.linspace(-length / 2, length / 2, m + 1,
                                     dtype=dtype, device=device) + shift / 10
    p_image = sign * torch.linspace(-length / 2, length / 2, h + 1,
                                    dtype=dtype, device=device) + shift / 10
    ky = -2 * sign * a * torch.arange(m, dtype=dtype, device=device) * length / m
    b = -ky[0] - 2 * a * (p_native[0] + (p_native[1] - p_native[0]) * ky1)
    result = calcSRMatrixApprox(a * length ** 2, h, ky, p_image, b)[0]
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite SPEN encoding')
    return result.to(torch.complex64)


def scanner_sr_matrices(path, image_size=192, device='cpu'):
    """Return (A_high, A_native, params, metadata), using one native A norm."""
    _, native, params = scanner_matrices(path, device)
    high = encoding_from_params(params, image_size, device)
    norm = torch.linalg.svdvals(native).max()
    metadata = dict(path=str(path), full_args=params, image_size=int(image_size),
                    acquired_pe=int(native.shape[0]), encoding_native_smax=float(norm),
                    matrix_shape=list(high.shape),
                    normalization='Both PE matrices divided by spectral norm of native A96',
                    readout='centered IFFT -> central band -> FFT; cell-center phase; sqrt(M/N) unitary scaling')
    return high / norm, native / norm, params, metadata


class SpenSuperResolutionOperator(SpenMagnitudeOperator):
    """Real-domain SPEN operator with RO bandwidth reduction and a CG proximal.

    Input [B,1,H,W], coils [C,H,W] or [B,C,H,W], encoding [M,H], output
    [B,C,selected_M,measurement_size]. CG solves the exact affine quadratic;
    its finite-iteration error is reported in cg_diagnostics on every call.
    """
    def __init__(self, encoding, coils, measurement_size=96, pe_mask=None,
                 sigma_noise=.01, cg_max_iter=160, cg_rtol=1e-5, cg_atol=1e-7):
        super().__init__(encoding, coils, pe_mask, sigma_noise)
        self.measurement_size = int(measurement_size)
        n = self.coils.shape[-1]
        if not 0 < self.measurement_size <= n or n % 2 or self.measurement_size % 2:
            raise ValueError('RO measurement/image sizes must be positive and even, M <= N')
        self.cg_max_iter = int(cg_max_iter)
        self.cg_rtol, self.cg_atol = float(cg_rtol), float(cg_atol)
        if self.cg_max_iter < 1 or self.cg_rtol <= 0 or self.cg_atol < 0:
            raise ValueError('Invalid CG stopping configuration')
        self.cg_diagnostics = []
        self.metadata = {}
        # diag(Re L*L); D_RO*D_RO has constant diagonal (M/N)^2.
        pe_energy = self.a.abs().square().sum(0)[None, None, :, None]
        coil_energy = self.coils.abs().square().sum(1, keepdim=True)
        self.normal_diagonal = .25 * pe_energy * coil_energy * (self.measurement_size / n) ** 2

    def linear(self, x):
        z = self.coils * (x * .5)
        encoded = torch.einsum('mh,bchw->bcmw', self.a, z.to(self.a.dtype))
        return readout_resize(encoded, self.measurement_size)

    def adjoint(self, y):
        expanded = readout_adjoint(y, self.coils.shape[-1])
        z = torch.einsum('mh,bcmw->bchw', self.a.conj(), expanded)
        return .5 * (self.coils.conj() * z).sum(1, keepdim=True).real

    @torch.no_grad()
    def proximal(self, z, observation, rho):
        """Solve min ||forward(x)-y||² + rho||x-z||² by batched PCG.

        Start at z, so unconstrained/nullspace image content stays supplied by
        the prior. No persistent iterate crosses diffusion steps. This matters
        when the early schedule's rho is very small. No image clipping occurs.
        """
        rho = torch.as_tensor(rho, device=z.device, dtype=z.dtype)
        if rho.numel() != 1 or not bool(torch.isfinite(rho)) or float(rho) <= 0:
            raise ValueError('rho must be one positive finite scalar')
        inner = lambda a, b: (a * b).flatten(1).sum(1).reshape(-1, 1, 1, 1)
        normal = lambda v: self.adjoint(self.linear(v)) + rho * v
        # Solve for the correction x-z instead of subtracting two large RHSs.
        initial = self.adjoint(observation - self.forward(z))
        offset = self.forward(torch.zeros_like(z))
        rhs = self.adjoint(observation - offset) + rho * z
        rhs_norm = inner(rhs, rhs).sqrt()
        threshold = self.cg_atol + self.cg_rtol * rhs_norm
        r = initial.clone()
        correction = torch.zeros_like(z)
        preconditioner = self.normal_diagonal.to(z.dtype) + rho
        preconditioned = r / preconditioner
        p = preconditioned.clone()
        rz = inner(r, preconditioned)
        tiny = torch.finfo(z.dtype).tiny
        iterations = 0
        for i in range(self.cg_max_iter):
            active = inner(r, r).sqrt() > threshold
            if not bool(active.any()):
                break
            ap = normal(p)
            pap = inner(p, ap)
            if bool(((pap <= 0) & active).any()):
                raise FloatingPointError('PCG normal operator lost positive definiteness')
            alpha = torch.where(active, rz / pap.clamp_min(tiny), torch.zeros_like(rz))
            correction = correction + alpha * p
            r = r - alpha * ap
            iterations = i + 1
            preconditioned = r / preconditioner
            rz_next = inner(r, preconditioned)
            beta = torch.where(active, rz_next / rz.clamp_min(tiny), torch.zeros_like(rz))
            p = preconditioned + beta * p
            rz = rz_next
        out = z + correction
        # Measure a freshly applied normal equation, not only recurrence error.
        true_residual = self.adjoint(self.forward(out) - observation) + rho * (out - z)
        residual_norm = inner(true_residual, true_residual).sqrt()
        relative = residual_norm / rhs_norm.clamp_min(tiny)
        if not torch.isfinite(out).all() or not torch.isfinite(relative).all():
            raise FloatingPointError('Nonfinite PCG proximal output')
        self.cg_diagnostics.append(dict(rho=float(rho), iterations=iterations,
            max_iter=self.cg_max_iter, relative_normal_residual=relative.flatten().cpu().tolist(),
            absolute_normal_residual=residual_norm.flatten().cpu().tolist(),
            converged=(residual_norm <= threshold).flatten().cpu().tolist()))
        return out


def make_sr_operator(fov=16, image_size=192, measurement_size=96, noise=.01,
                     device='cpu', seed=None, phase_strength=.7, coils=None,
                     cg_max_iter=160, cg_rtol=1e-5, cg_atol=1e-7):
    """Controlled known-coil SR case; image_size=96 gives the native model.

    Default coils match evaluate_mouse.controlled_operator's deterministic R1
    coil/gain/phase recipe at 96 and evaluate the same smooth functions at 192.
    Supply coils explicitly for scanner nuisance estimates or custom controls.
    """
    if int(fov) not in SCANS:
        raise ValueError('Supported mouse scanner FOVs are 16 and 24 mm')
    high, native, params, metadata = scanner_sr_matrices(
        SCANS[int(fov)] / 'slice_7.mat', image_size, device)
    if measurement_size != native.shape[0]:
        raise ValueError('This scanner factory fixes the original 96 x 96 acquisition')
    if seed is None:
        seed = 4500 + int(fov) + 1
    supplied_coils = coils is not None
    if coils is None:
        gen = torch.Generator(device=device).manual_seed(int(seed))
        # The native recipe places linspace(-1,1,96) at its 96 pixel centers.
        # Evaluate those same physical functions at the finer grid centers,
        # instead of moving the coil pattern when image_size changes.
        axis = torch.linspace(-1, 1, image_size, device=device)
        axis = axis * (96 * (image_size - 1) / (95 * image_size))
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        fields = []
        for i in range(4):
            angle = 2 * math.pi * i / 4
            amp = torch.exp(-((xx - .6 * math.cos(angle)) ** 2
                              + (yy - .6 * math.sin(angle)) ** 2) / 1.8)
            local_phase = phase_strength * (xx * math.cos(angle)
                            + yy * math.sin(angle) + .3 * xx * yy) + angle * .2
            fields.append(amp * torch.exp(1j * local_phase))
        coils = torch.stack(fields)
        coils = coils / coils.abs().square().sum(0, keepdim=True).sqrt()
        coeff = torch.randn(4, 3, device=device, generator=gen) * .5
        phase = coeff[:, 0, None, None] + coeff[:, 1, None, None] * xx + coeff[:, 2, None, None] * yy
        gain = torch.exp(.2 * torch.randn(4, 1, 1, device=device, generator=gen))
        coils = coils * gain * torch.exp(1j * phase)
        coils = coils / coils.abs().square().sum(0, keepdim=True).sqrt()
    op = SpenSuperResolutionOperator(high, coils, measurement_size,
        sigma_noise=noise, cg_max_iter=cg_max_iter, cg_rtol=cg_rtol, cg_atol=cg_atol)
    op.native_encoding = native
    op.scanner_params = params
    op.metadata = dict(metadata, fov_mm=int(fov), coil_seed=int(seed),
                       coil_phase=('caller-supplied fixed complex coil fractions' if supplied_coils
                                   else 'known synthetic smooth complex coil fractions'))
    return op


def make_low_resolution_operator(fov=16, noise=.01, device='cpu', seed=None, coils=None):
    """Original 96-grid baseline with the exact columnwise eigensystem solve."""
    op = make_sr_operator(fov=fov, image_size=96, noise=noise, device=device,
                          seed=seed, coils=coils)
    native = SpenMagnitudeOperator(op.a, op.coils, sigma_noise=noise)
    native.metadata = op.metadata
    return native
