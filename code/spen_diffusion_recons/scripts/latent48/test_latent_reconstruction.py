"""CPU tests for decoder-chain gradients, finite nonlinear solves and seeding."""
import json

import pytest
import torch
from torch import nn

from latent_reconstruction import latent_objective, latent_proximal, latent_reconstruct


class AffineCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.7))
        self.autocast_dtype = torch.bfloat16

    def encode(self, x, sample=False):
        assert not sample
        return (x - .2) / self.weight

    def decode(self, z, clamp=True):
        x = self.weight * z + .2
        return x.clamp(-1, 1) if clamp else x


class ToyOperator:
    def __init__(self, scale=2.3):
        self.scale = scale

    def forward(self, x):
        # Includes [-1,1] affine normalization and complex coil phase.
        return self.scale * (1 + .4j) * (x + 1) / 2


class ToyDenoiser(nn.Module):
    img_resolution = 4
    img_channels = 1

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.7))

    def forward(self, z, sigma):
        return z / (1 + sigma.square()) * self.weight


def test_affine_decoder_operator_gradient_and_coordinate_scaling():
    codec = AffineCodec().requires_grad_(False)
    op = ToyOperator()
    mean, std = torch.tensor(.3), torch.tensor(.8)
    u = torch.linspace(-.7, .6, 16).reshape(1, 1, 4, 4).requires_grad_()
    z0, y, rho = torch.zeros_like(u), torch.full_like(u, 1.1 + .7j, dtype=torch.complex64), .6
    objective, _, _ = latent_objective(u, z0, codec, mean, std, op, y, rho)
    gradient, = torch.autograd.grad(objective, u)
    slope = op.scale * (1 + .4j) * codec.weight * std / 2
    offset = op.scale * (1 + .4j) * (codec.weight * mean + .2 + 1) / 2
    expected = 2 * (slope.conj() * (slope * u + offset - y)).real + 2 * rho * (u - z0)
    torch.testing.assert_close(gradient, expected)
    # This catches a missing measurement factor of 1/2 or latent std factor.
    epsilon = 1e-3
    direction = torch.ones_like(u)
    plus = latent_objective(u + epsilon * direction, z0, codec, mean, std, op, y, rho)[0]
    minus = latent_objective(u - epsilon * direction, z0, codec, mean, std, op, y, rho)[0]
    torch.testing.assert_close((plus - minus) / (2 * epsilon), (gradient * direction).sum(), rtol=5e-4, atol=1e-3)


def test_proximal_monotonicity_and_known_affine_minimizer():
    codec = AffineCodec().requires_grad_(False)
    op, mean, std, rho = ToyOperator(), torch.tensor(.3), torch.tensor(.8), .6
    z0 = torch.linspace(-.5, .5, 16).reshape(1, 1, 4, 4)
    y = op.forward(torch.full_like(z0, .4))
    result, trace = latent_proximal(z0, codec, mean, std, op, y, rho,
                                    inner_steps=100, objective_rtol=0, grad_rtol=1e-5)
    slope = op.scale * (1 + .4j) * codec.weight * std / 2
    offset = op.scale * (1 + .4j) * (codec.weight * mean + .2 + 1) / 2
    solution = ((slope.conj() * (y - offset)).real + rho * z0) / (slope.abs().square() + rho)
    torch.testing.assert_close(result, solution, atol=2e-4, rtol=2e-4)
    assert trace['objective_after'] < trace['objective_before']
    assert trace['accepted_steps'] > 0
    assert all(row['after']['objective'] <= row['before']['objective'] for row in trace['inner_trace'])
    assert trace['gradient_norm_after'] < trace['gradient_norm_before']


def test_reconstruction_no_grad_reproducibility_precision_restoration_and_frozen_weights():
    codec, net, op = AffineCodec(), ToyDenoiser(), ToyOperator()
    initial = torch.linspace(-.7, .7, 16).reshape(1, 1, 4, 4)
    y = op.forward(initial)
    config = dict(initial_image=initial, steps=4, inner_steps=3, sigma_max=.8, sigma_min=.02, seed=8)
    with torch.no_grad():
        result, diagnostics = latent_reconstruct(net, codec, [0.], [1.], op, y, **config)
        repeat, _ = latent_reconstruct(net, codec, [0.], [1.], op, y, **config)
        different, _ = latent_reconstruct(net, codec, [0.], [1.], op, y, **dict(config, seed=9))
    torch.testing.assert_close(result, repeat, rtol=0, atol=0)
    assert not torch.equal(result, different)
    assert torch.isfinite(result).all()
    assert len(diagnostics['trace']) == 4
    assert diagnostics['initialization'].startswith('observation_image')
    assert diagnostics['sigmas'][-1] == 0
    assert all(row['objective_after'] <= row['objective_before'] for row in diagnostics['trace'])
    assert codec.autocast_dtype == torch.bfloat16
    assert not codec.weight.requires_grad and not net.weight.requires_grad
    assert codec.weight.grad is None and net.weight.grad is None
    json.dumps(diagnostics, allow_nan=False)


def test_inference_mode_and_invalid_normalization_are_rejected():
    args = (ToyDenoiser(), AffineCodec(), [0.], [0.], ToyOperator(), torch.zeros(1, 1, 4, 4))
    with pytest.raises(ValueError, match='standard deviations'):
        latent_reconstruct(*args)
    with torch.inference_mode(), pytest.raises(ValueError, match='decoder input gradients'):
        latent_reconstruct(*args)
