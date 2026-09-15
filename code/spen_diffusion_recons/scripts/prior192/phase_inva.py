"""Legacy measured-data PhaseMap + weighted InvA reconstruction.

The seven phase-fitting functions below are copied verbatim from
spen_phase_inva_0527_common.py in the original project's
log/0710_primary_strict_source_robust_scannerop_split20260710_seed7/source_snapshot/scripts/.
Source SHA-256: 662c73da2368c602007de63b457479fba6fb5e0ec85333c242eea91217935de3
The reconstruction follows the 2026-09-10 simulation's additional_baselines.py,
with dynamic coil count and CPU-compatible random-state handling. No external
source file is imported at runtime.

InvA is the scanner's Gaussian-weighted adjoint, not a numerical matrix inverse.
Missing PE rows are zero-filled for phase fitting and InvA; complex receiver
gains are fitted using acquired rows only. No clean target is used. The original
controlled simulation injected no odd/even phase error; fitting a PhaseMap does
not establish correction of an independently simulated scanner phase error.
"""
import math

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

from spenpy._legacy.core import calcInvA


def coords_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1, 1, height, device=device)
    x = torch.linspace(-1, 1, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


def _batched_linear_params(batch: int, out_dim: int, in_dim: int, device: torch.device, zero: bool = False):
    weight = torch.empty(batch, out_dim, in_dim, device=device)
    bias = torch.empty(batch, out_dim, device=device)
    if zero:
        weight.zero_()
        bias.zero_()
    else:
        bound = 1 / math.sqrt(in_dim)
        weight.uniform_(-bound, bound)
        bias.uniform_(-bound, bound)
    return weight.requires_grad_(), bias.requires_grad_()


def _batched_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bni,boi->bno", x, weight) + bias[:, None, :]


def wrap_phase(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def scanner_batch_phase_inputs(y: torch.Tensor, odd_inv: torch.Tensor, even_inv: torch.Tensor):
    odd = torch.einsum("ij,bjrcn->bircn", odd_inv, y[:, 0::2])
    even = torch.einsum("ij,bjrcn->bircn", even_inv, y[:, 1::2])
    odd = odd[:, :, :, 0, :]
    even = even[:, :, :, 0, :]
    cross = torch.sum(even * torch.conj(odd), dim=-1)
    raw = torch.angle(cross).float()
    odd_mag = torch.sqrt(torch.sum(odd.abs().square(), dim=-1))
    even_mag = torch.sqrt(torch.sum(even.abs().square(), dim=-1))
    mag_std = torch.std(even_mag - odd_mag, dim=(1, 2), keepdim=True)
    mask = (odd_mag > 2 * mag_std) & (even_mag > 2 * mag_std) & torch.isfinite(raw)
    strength = cross.abs()
    cutoffs = [
        torch.quantile(strength[i][mask[i]], 0.2) if torch.any(mask[i]) else strength.new_tensor(0)
        for i in range(y.shape[0])
    ]
    mask = mask & (strength >= torch.stack(cutoffs).view(-1, 1, 1))
    weights = mask.float() * torch.minimum(odd_mag, even_mag).float()
    return odd, even, weights / weights.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)


def fit_tiny_phase_scanner_batch(
    y: torch.Tensor,
    odd_inv: torch.Tensor,
    even_inv: torch.Tensor,
    steps: int = 300,
    hidden: int = 16,
    lr: float = 3e-2,
    smooth_weight: float = 2e-3,
) -> torch.Tensor:
    odd, even, weights = scanner_batch_phase_inputs(y, odd_inv, even_inv)
    batch, height, width = weights.shape
    coords = coords_grid(height, width, y.device).expand(batch, -1, -1)
    with torch.enable_grad():
        w1, b1 = _batched_linear_params(batch, hidden, 2, y.device)
        w2, b2 = _batched_linear_params(batch, hidden, hidden, y.device)
        w3, b3 = _batched_linear_params(batch, hidden, hidden, y.device)
        w4, b4 = _batched_linear_params(batch, hidden, hidden, y.device)
        w5, b5 = _batched_linear_params(batch, hidden, hidden, y.device)
        w6, b6 = _batched_linear_params(batch, 1, hidden, y.device, zero=True)
        params = [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        def forward() -> torch.Tensor:
            h = torch.tanh(_batched_linear(coords, w1, b1))
            h = torch.tanh(_batched_linear(h, w2, b2))
            h = F.relu(_batched_linear(h, w3, b3))
            h = torch.tanh(_batched_linear(h, w4, b4))
            h = F.relu(_batched_linear(h, w5, b5))
            return _batched_linear(h, w6, b6).squeeze(-1).reshape(batch, height, width)

        for _ in range(steps):
            phase = forward()
            corrected = even * torch.exp(-1j * phase[:, :, :, None])
            cross = torch.sum(corrected * torch.conj(odd), dim=-1)
            align = 1 - (weights * cross.real / cross.abs().clamp_min(1e-8)).sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1e-8)
            smooth = wrap_phase(phase[:, :, 1:] - phase[:, :, :-1]).square().mean(dim=(1, 2))
            smooth = smooth + wrap_phase(phase[:, 1:] - phase[:, :-1]).square().mean(dim=(1, 2))
            opt.zero_grad(set_to_none=True)
            (align + smooth_weight * smooth).mean().backward()
            opt.step()

        return forward().detach()


def apply_even_phase(y: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    out = y.clone()
    if y.ndim == 2:
        out[1::2] *= torch.exp(-1j * phase).to(out.dtype)
    elif y.ndim == 3:
        out[:, 1::2] *= torch.exp(-1j * phase).to(out.dtype)
    else:
        out[1::2] *= torch.exp(-1j * phase[:, :, None, None]).to(out.dtype)
    return out

def load_phase_matrices(mat_path, device='cpu'):
    """Return native-grid (InvA, odd InvA, even InvA) from scanner parameters."""
    params = scipy.io.loadmat(
        mat_path, squeeze_me=True, struct_as_record=False,
        variable_names=['calc_inva_params'])['calc_inva_params']
    matrices = []
    for field in ('full_args', 'odd_args', 'even_args'):
        values = np.asarray(getattr(params, field), dtype=np.float64).ravel()
        if values.size != 7:
            raise ValueError(f'Expected seven scanner parameters in {field}')
        matrix, _ = calcInvA(values[0], values[1], int(values[2]), values[3],
                             int(values[4]), values[5], values[6])
        matrices.append(matrix.to(device=device, dtype=torch.complex64))
    return tuple(matrices)


def phase_inva(y, mask, a, inv, odd_inv, even_inv, seed, steps=300):
    """Fit phase and reconstruct one observation [C, acquired PE, native RO].

    ``a`` is the full native PE encoding normalized exactly as the forward
    acquisition model. The result tuple is (magnitude, phase, corrected,
    coil_images, complex_gains, phase_correction_amplitude_error). Magnitudes
    remain unclipped and on the native 96 grid; display upsampling is separate.
    """
    if y.ndim != 3 or not torch.is_complex(y):
        raise ValueError('Expected complex observation [C, acquired PE, RO]')
    mask = torch.as_tensor(mask, device=y.device, dtype=torch.bool)
    native_size = mask.numel()
    if mask.ndim != 1 or native_size != 96 or y.shape[-1] != native_size:
        raise ValueError('PhaseMap + InvA expects the native 96 x 96 scanner grid')
    if int(mask.sum()) != y.shape[1] or not bool(mask.any()):
        raise ValueError('PE mask does not match the acquired observation rows')
    if tuple(a.shape) != (native_size, native_size) or tuple(inv.shape) != tuple(a.shape):
        raise ValueError('Expected full native encoding and InvA matrices')
    if int(steps) < 0:
        raise ValueError('Phase fitting steps must be nonnegative')
    full = y.new_zeros(y.shape[0], native_size, native_size)
    full[:, mask] = y
    scanner = full.permute(1, 2, 0).unsqueeze(2)
    cuda_devices = []
    if y.device.type == 'cuda':
        cuda_devices = [y.device.index if y.device.index is not None
                        else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(int(seed))
        if cuda_devices:
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(int(seed))
        phase = fit_tiny_phase_scanner_batch(
            scanner[None], odd_inv, even_inv, steps=int(steps))[0]
    corrected_scanner = apply_even_phase(scanner, phase)
    corrected = corrected_scanner[:, :, 0].permute(2, 0, 1)
    u = torch.einsum('ij,cjw->ciw', inv, corrected)
    projected = torch.einsum('ij,cjw->ciw', a[mask], u)
    measured = corrected[:, mask]
    gain = ((projected.conj() * measured).sum((1, 2))
            / projected.abs().square().sum((1, 2)).clamp_min(1e-12))
    coils = u * gain[:, None, None]
    magnitude = coils.abs().square().sum(0).sqrt()
    amplitude_error = float((corrected.abs() - full.abs()).abs().max())
    if amplitude_error >= 1e-6:
        raise FloatingPointError('Phase correction changed observation magnitudes')
    if bool(corrected[:, ~mask].abs().any()):
        raise AssertionError('Phase correction populated unacquired PE rows')
    return magnitude, phase, corrected, coils, gain, amplitude_error
