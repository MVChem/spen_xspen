"""SPEN complex multi-coil operator for a real magnitude diffusion prior.

Physical model: y_c = M A (S_c m), m=(x+1)/2. A acts on PE;
RO has already been Fourier transformed. S includes fixed object/coil phase.
The affine offset from [-1,1] normalization is handled explicitly.
"""
import sys
from pathlib import Path
import numpy as np
import scipy.io
import torch
from data import PROJECT, ROOT, sha256

# Use the exact calcInvA implementation paired with the existing scanner export.
from project_paths import PACKAGE_ROOT
sys.path.insert(0, str(PACKAGE_ROOT/'vendor/InverseBench'))
from spenpy._legacy.core import calcInvA
from inverse_problems.base import BaseOperator

SCANNER = PROJECT/'data/mat/20231011_lxj_SPEN_data_231011_3_1_3'
CACHE = PROJECT/'log/0818_test_primary_strict_s7_real_scanner_rat/arrays'


def scanner_matrices(path, device='cpu'):
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False,
                           variable_names=['calc_inva_params'])
    v = torch.as_tensor(mat['calc_inva_params'].full_args, dtype=torch.float64).flatten()
    inv, a = calcInvA(v[0], v[1], int(v[2]), v[3], int(v[4]), v[5], v[6])
    return inv.to(device=device, dtype=torch.complex64), a.to(device=device, dtype=torch.complex64), v.tolist()


class SpenMagnitudeOperator(BaseOperator):
    """InverseBench API plus a real adjoint and exact quadratic proximal solve.

    Coils [C,H,W] or [B,C,H,W]; encoding [M,H]. Different RO columns are
    independent. Cache an H x H real eigensystem per column instead of
    constructing the enormous full image matrix. Input/output: [B,1,H,W]
    normalized real images / [B,C,M,W] complex measurements.
    """
    def __init__(self, encoding, coils, pe_mask=None, sigma_noise=.01):
        super().__init__(sigma_noise=sigma_noise, unnorm_shift=1., unnorm_scale=.5,
                         device=encoding.device)
        if encoding.ndim != 2 or not torch.is_complex(encoding):
            raise ValueError('encoding must be a complex [M,H] tensor')
        self.a_full = encoding
        self.coils = coils.unsqueeze(0) if coils.ndim == 3 else coils
        if self.coils.ndim != 4 or self.coils.shape[-2] != encoding.shape[1]:
            raise ValueError('coils must be [C,H,W] or [B,C,H,W]')
        self.mask = torch.ones(encoding.shape[0], dtype=torch.bool, device=self.device) if pe_mask is None else pe_mask.to(device=self.device, dtype=torch.bool)
        if self.mask.shape != (encoding.shape[0],) or not self.mask.any():
            raise ValueError('PE mask must select at least one encoding row')
        if not torch.isfinite(encoding).all() or not torch.isfinite(self.coils).all():
            raise ValueError('nonfinite encoding/coils')
        self.a = encoding[self.mask]
        self._eig = None

    def linear(self, x):
        z = self.coils * (x * .5)
        return torch.einsum('mh,bchw->bcmw', self.a, z.to(self.a.dtype))

    def forward(self, x, **kwargs):
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[-2:] != self.coils.shape[-2:]:
            raise ValueError(f'Expected [B,1,{self.coils.shape[-2]},{self.coils.shape[-1]}]')
        return self.linear(x + 1.)

    def adjoint(self, y):
        z = torch.einsum('mh,bcmw->bchw', self.a.conj(), y)
        return .5 * (self.coils.conj()*z).sum(1, keepdim=True).real

    def loss(self, pred, observation, **kwargs):
        return (self.forward(pred)-observation).abs().square().flatten(1).sum(1)

    def loss_m(self, measurements, observation):
        return (measurements-observation).abs().square().flatten(1).sum(1)

    def gradient(self, pred, observation, return_loss=False):
        residual = self.forward(pred)-observation
        grad = 2*self.adjoint(residual)
        if return_loss:
            return grad, residual.abs().square().flatten(1).sum(1)
        return grad

    def _eigensystem(self):
        if self._eig is None:
            # Form in double precision; return float32 for repeated GPU inference.
            a = self.a.to(torch.complex128)
            c = self.coils.to(torch.complex128)
            ah_a = a.conj().T @ a
            cc = torch.einsum('bciw,bcjw->bwij', c.conj(), c)
            gram = (.25 * cc * ah_a[None, None]).real
            vals, vecs = torch.linalg.eigh(gram)
            self._eig = vals.clamp_min(0).float(), vecs.float()
        return self._eig

    def proximal(self, z, observation, rho):
        """min_x ||forward(x)-y||^2 + rho ||x-z||^2, without clipping."""
        rho = torch.as_tensor(rho, device=z.device, dtype=z.dtype)
        if rho.numel() != 1 or float(rho) <= 0:
            raise ValueError('rho must be a positive scalar')
        vals, vecs = self._eigensystem()
        offset = self.forward(torch.zeros_like(z))
        rhs = self.adjoint(observation-offset) + rho*z
        rhs = rhs[:, 0].transpose(-1, -2).unsqueeze(-1)  # [B,W,H,1]
        coeff = vecs.transpose(-1,-2) @ rhs
        out = vecs @ (coeff / (vals+rho).unsqueeze(-1))
        return out.squeeze(-1).transpose(-1,-2).unsqueeze(1)

    def relative_residual(self, x, observation):
        return self.loss(x, observation).sqrt() / observation.abs().square().flatten(1).sum(1).sqrt().clamp_min(1e-12)


def synthetic_coils(size=96, count=4, device='cpu', phase_strength=.3):
    xy = torch.linspace(-1,1,size,device=device)
    yy, xx = torch.meshgrid(xy,xy,indexing='ij')
    coils = []
    for i in range(count):
        angle = 2*np.pi*i/count
        amp = torch.exp(-((xx-.6*np.cos(angle))**2+(yy-.6*np.sin(angle))**2)/1.8)
        phase = phase_strength*(xx*np.cos(angle)+yy*np.sin(angle)+.3*xx*yy) + angle*.2
        coils.append(amp*torch.exp(1j*phase))
    coils = torch.stack(coils)
    return coils / coils.abs().square().sum(0,keepdim=True).sqrt()


def make_synthetic_operator(device='cpu', acceleration=1, noise=.01):
    _, a, _ = scanner_matrices(SCANNER/'slice_13.mat', device)
    a = a / torch.linalg.svdvals(a).max()
    mask = torch.arange(a.shape[0], device=device) % acceleration == 0
    return SpenMagnitudeOperator(a, synthetic_coils(device=device), mask, noise)


def load_scanner_case(slice_id, device='cuda'):
    """Reuse phase-fit signal/anchor from the existing scan; flow only as a comparator.

    Coil fractions and phase are estimated from the traditional anchor. They
    are fixed nuisance estimates, not independently measured sensitivity maps.
    All physics stays in native orientation; rotate 180 degrees only to plot.
    """
    mat_path, cache_path = SCANNER/f'slice_{slice_id}.mat', CACHE/f'slice_{slice_id}.npz'
    inv, a, params = scanner_matrices(mat_path, device)
    with np.load(cache_path, allow_pickle=False) as z:
        saved = {k: z[k].copy() for k in z.files}
    signal = torch.as_tensor(saved['corrected_signal'], device=device).permute(2,0,1)[None]
    anchor = torch.as_tensor(saved['anchor_coils'], device=device).permute(2,0,1)[None]
    scale = float(saved['magnitude_scale'])
    smax = float(torch.linalg.svdvals(a).max())
    rss = anchor.abs().square().sum(1,keepdim=True).sqrt()
    coils = anchor / rss.clamp_min(scale*1e-8)
    op = SpenMagnitudeOperator(a/smax, coils)
    y = signal / (scale*smax)
    x_anchor = 2*rss/scale-1  # Unclipped; essential for residual consistency.
    m = scipy.io.loadmat(mat_path, squeeze_me=True, variable_names=['traditional_recon'])
    traditional = np.abs(m['traditional_recon']).astype(np.float32)
    traditional = np.clip(traditional/max(float(np.quantile(traditional,.995)),1e-6),0,1)
    # Validate exact scalar handling against the cached complex-coil formulation.
    native_pred = torch.einsum('ij,bcjw->bciw', a, anchor)/(scale*smax)
    discrepancy = float((op.forward(x_anchor)-native_pred).abs().max())
    if discrepancy > 2e-5:
        raise ValueError(f'Cached scanner scaling mismatch: {discrepancy}')
    metadata = dict(slice_id=slice_id, mat_path=str(mat_path), mat_sha256=sha256(mat_path),
                    phase_cache=str(cache_path), phase_cache_sha256=sha256(cache_path),
                    full_args=params, magnitude_scale=scale, encoding_smax=smax,
                    normalization_error=discrepancy,
                    anchor_residual=float(op.relative_residual(x_anchor,y)),
                    nuisance='fixed complex coil fractions from phase+InvA anchor',
                    orientation='native for physics; rot180 for plotted prior reconstructions')
    return op, y, x_anchor, saved, traditional, metadata
