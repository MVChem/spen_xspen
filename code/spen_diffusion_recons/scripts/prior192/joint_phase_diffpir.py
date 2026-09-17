"""Test-time acquired-even phase learning inside a frozen-prior DiffPIR loop.

F_theta(m) = exp(i E phi_theta) * D_RO A_PE(S m).
E puts phase on original even scanner rows (Python indices 1,3,...). The
sampling mask is applied AFTER E, so random PE masks preserve scanner parity.
Only the phase factor is learned. Encoding, coils, and the denoiser stay fixed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import torch

from pilot_testtime_even_odd import TinyPhase
from phase_inva import coords_grid, wrap_phase
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from model import sigma_schedule


@dataclass
class PhaseConfig:
    updates_per_step: int = 12
    learning_rate: float = .01
    smooth_weight: float = .002
    energy_weight: float = .0001
    seed: int = 20260916


class PhaseOperator:
    """Unit-modulus acquisition factor with exact adjoint and quadratic solve."""

    def __init__(self, base, phase):
        if base.a_full.shape[0] % 2:
            raise ValueError('Expected even acquisition PE size')
        self.base = base
        self.device, self.coils = base.device, base.coils
        self.set_phase(phase)

    def set_phase(self, phase):
        expected = (self.base.a_full.shape[0] // 2, self.base.measurement_size)
        if tuple(phase.shape) != expected or not bool(torch.isfinite(phase).all()):
            raise ValueError(f'Expected finite acquired-even phase {expected}')
        full = phase.new_zeros(self.base.a_full.shape[0], phase.shape[-1])
        full[1::2] = phase
        self.factor = torch.exp(1j * full[self.base.mask])[None, None]

    def linear(self, x):
        return self.factor * self.base.linear(x)

    def forward(self, x):
        return self.factor * self.base.forward(x)

    def adjoint(self, y):
        return self.base.adjoint(self.factor.conj() * y)

    def proximal(self, z, observation, rho):
        return self.base.proximal(z, self.factor.conj() * observation, rho)

    def relative_residual(self, x, observation):
        residual = (self.forward(x) - observation).flatten(1).norm(dim=1)
        return residual / observation.flatten(1).norm(dim=1).clamp_min(1e-12)


class PhaseFitter:
    """A new 1,153-parameter coordinate MLP for each single acquisition."""

    def __init__(self, base, config):
        if config.updates_per_step < 0 or config.learning_rate <= 0:
            raise ValueError('Invalid phase optimization configuration')
        if min(config.smooth_weight, config.energy_weight) < 0:
            raise ValueError('Phase penalties must be nonnegative')
        self.base, self.config = base, config
        with torch.random.fork_rng(devices=[base.device.index] if base.a.is_cuda else []):
            torch.manual_seed(config.seed)
            self.net = TinyPhase().to(base.device)
        self.shape = (base.a_full.shape[0] // 2, base.measurement_size)
        self.coords = coords_grid(*self.shape, base.device)
        self.even = torch.arange(base.a_full.shape[0], device=base.device)[base.mask] % 2 == 1
        self.even_mask = base.mask[1::2]
        if not bool(self.even.any()):
            raise ValueError('No acquired even lines available to fit phase')
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.learning_rate)
        self.total_updates = 0

    def phase(self):
        return self.net(self.coords, self.shape)

    def update(self, prediction, observation):
        """Fit raw complex acquired data; no truth, decoded alignment, or oracle."""
        prediction, observation = prediction.detach(), observation.detach()
        p, y = prediction[:, :, self.even], observation[:, :, self.even]
        scale = y.abs().square().mean().clamp_min(1e-12)
        report = {}
        with torch.enable_grad():
            for _ in range(self.config.updates_per_step):
                phase = self.phase()
                residual = p * torch.exp(1j * phase[self.even_mask])[None, None] - y
                data = residual.abs().square().mean() / scale
                smooth = (wrap_phase(phase[1:] - phase[:-1]).square().mean()
                          + wrap_phase(phase[:, 1:] - phase[:, :-1]).square().mean())
                energy = phase.square().mean()
                loss = data + self.config.smooth_weight * smooth + self.config.energy_weight * energy
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite phase-fitting loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.)
                self.optimizer.step()
                self.total_updates += 1
                report = dict(data_loss=float(data.detach()), smoothness=float(smooth.detach()),
                              phase_energy=float(energy.detach()), objective=float(loss.detach()))
        return report


@torch.no_grad()
def joint_diffpir(net, base, observation, steps=60, sigma_noise=.01, lamb=1.,
                  seed=7, sigma_max=2., sigma_min=.02, phase_config=None, progress=None):
    """Alternate denoising, phase fitting, and image data consistency each step.

The clean denoiser prediction is detached and clipped only for fitting phase.
The image proximal uses the unclipped denoiser output, matching core DiffPIR.
Gradients are enabled only for the small MLP, never through the image prior.
"""
    if observation.shape[0] != 1 or sigma_noise <= 0 or lamb <= 0:
        raise ValueError('One acquisition and positive noise/lambda required')
    config = phase_config or PhaseConfig()
    fitter = PhaseFitter(base, config)
    op = PhaseOperator(base, fitter.phase().detach())
    gen = torch.Generator(device=base.device).manual_seed(seed)
    sigmas = sigma_schedule(steps, sigma_max, sigma_min, base.device)
    x = torch.randn(1, 1, *base.coils.shape[-2:], device=base.device, generator=gen) * sigmas[0]
    trace = []
    for i, (s, sn) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        x0 = net(x, s)
        prediction = base.forward(x0.clamp(-1, 1))
        fit_report = fitter.update(prediction, observation)
        phase = fitter.phase().detach()
        op.set_phase(phase)
        rho = 2 * lamb * sigma_noise**2 / s.square()
        x0y = op.proximal(x0, observation, rho)
        # Consume the same random stream as core diffpir(xi=0).
        torch.randn(x.shape, device=x.device, generator=gen)
        x = x0y + (x - x0y) / s * sn
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError(f'Nonfinite image at step {i}')
        row = dict(step=i, sigma=float(s), rho=float(rho), phase_updates=fitter.total_updates,
                   residual=float(op.relative_residual(x0y, observation)), **fit_report)
        trace.append(row)
        if progress is not None and (i % 10 == 0 or i == steps - 1):
            progress(row)
    state = dict(state_dict={k: v.detach().cpu() for k, v in fitter.net.state_dict().items()},
                 config=asdict(config), parameter_count=sum(p.numel() for p in fitter.net.parameters()))
    return x, phase, trace, state
