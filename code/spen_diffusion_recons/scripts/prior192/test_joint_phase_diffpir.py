"""Physical identities and solver controls for acquired-domain phase learning."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_phase_diffpir import PhaseConfig, PhaseFitter, PhaseOperator, joint_diffpir
from sr_operator import SpenSuperResolutionOperator
from solvers import diffpir


def small_operator():
    torch.manual_seed(17)
    a = torch.randn(8, 12, dtype=torch.complex64) / 4
    coils = torch.randn(2, 12, 12, dtype=torch.complex64) / 2
    mask = torch.tensor([True, True, False, True, True, False, False, True])
    return SpenSuperResolutionOperator(a, coils, 8, mask, cg_max_iter=200,
                                       cg_rtol=1e-6, cg_atol=1e-7)


def test_phase_mask_adjoint_and_proximal():
    base = small_operator()
    phase = torch.randn(4, 8)
    op = PhaseOperator(base, phase)
    x = torch.randn(1, 1, 12, 12)
    y = torch.randn_like(base.forward(x))
    expected = base.forward(x).clone()
    full_rows = torch.where(base.mask)[0]
    for i, row in enumerate(full_rows):
        if row % 2:
            expected[:, :, i] *= torch.exp(1j * phase[row // 2])
    torch.testing.assert_close(op.forward(x), expected)
    torch.testing.assert_close(op.forward(x).abs(), base.forward(x).abs())
    lhs = (op.linear(x).conj() * y).sum().real
    rhs = (x * op.adjoint(y)).sum()
    torch.testing.assert_close(lhs, rhs, rtol=2e-5, atol=2e-5)
    z = torch.randn_like(x)
    solution = op.proximal(z, y, .1)
    normal_error = op.adjoint(op.forward(solution) - y) + .1 * (solution - z)
    assert float(normal_error.norm()) < 1e-4


def test_zero_updates_reproduces_original_diffpir():
    base = small_operator()
    class Denoiser:
        def __call__(self, x, sigma):
            return x / (1 + sigma.square())
    net = Denoiser()
    y = base.forward(torch.zeros(1, 1, 12, 12))
    expected, _ = diffpir(net, base, y, steps=4, sigma_max=2., seed=24)
    result, phase, trace, state = joint_diffpir(net, base, y, steps=4, seed=24,
                                               phase_config=PhaseConfig(updates_per_step=0))
    torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-5)
    assert not bool(phase.any())
    assert state['parameter_count'] == 1153
    assert trace[-1]['phase_updates'] == 0


def test_complex_loss_fits_known_phase_without_image_gradients():
    torch.set_num_threads(2)
    base = small_operator()
    fitter = PhaseFitter(base, PhaseConfig(updates_per_step=200))
    xy = fitter.coords.reshape(*fitter.shape, 2)
    truth = .4 + .3 * xy[..., 0] - .2 * xy[..., 1]
    x = torch.randn(1, 1, 12, 12, requires_grad=True)
    clean = base.forward(x)
    observed = PhaseOperator(base, truth).forward(x).detach()
    fitter.update(clean, observed)
    estimate = fitter.phase().detach()
    residual = PhaseOperator(base, estimate).relative_residual(x.detach(), observed)
    assert float(residual) < .025
    assert x.grad is None
