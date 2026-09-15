"""Unconditional DiT with continuous EDM preconditioning for 48 x 48 latents.

Architecture and initialization follow the DiT paper and official implementation:
https://arxiv.org/abs/2212.09748
https://github.com/facebookresearch/DiT/blob/main/models.py
This implementation uses PyTorch SDPA, omits label conditioning / learned variance,
and predicts the EDM residual instead of discrete-time diffusion epsilon.

EDM coefficients and the deterministic Heun sampler follow:
https://github.com/NVlabs/edm/blob/main/training/networks.py
https://github.com/NVlabs/edm/blob/main/generate.py
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def fixed_2d_positions(grid_size: int, hidden_size: int) -> torch.Tensor:
    """Return row-major [1, grid_size**2, hidden_size] fixed sin/cos positions."""
    quarter = hidden_size // 4
    frequency = torch.exp(-math.log(10000.) * torch.arange(quarter) / quarter)
    yy, xx = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
    x_phase = xx.reshape(-1, 1) * frequency
    y_phase = yy.reshape(-1, 1) * frequency
    return torch.cat((x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()), dim=-1).unsqueeze(0)


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256):
        super().__init__()
        half = frequency_size // 2
        self.register_buffer('frequencies', torch.exp(-math.log(10000.) * torch.arange(half) / half))
        self.mlp = nn.Sequential(nn.Linear(frequency_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        phase = time.float().reshape(-1, 1) * self.frequencies.float()
        return self.mlp(torch.cat((phase.cos(), phase.sin()), dim=-1))


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, width = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.num_heads, width // self.num_heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.)
        return self.projection(attended.transpose(1, 2).reshape(batch, length, width))


def modulate(tokens: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Keep the residual stream in the projection dtype under BF16 autocast.
    return tokens * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attention = SelfAttention(hidden_size, num_heads)
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
                                 nn.GELU(approximate='tanh'),
                                 nn.Linear(int(hidden_size * mlp_ratio), hidden_size))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(condition).chunk(6, dim=-1)
        tokens = tokens + gate_a[:, None] * self.attention(modulate(self.norm_attention(tokens), shift_a, scale_a))
        tokens = tokens + gate_m[:, None] * self.mlp(modulate(self.norm_mlp(tokens), shift_m, scale_m))
        return tokens


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, in_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.projection = nn.Linear(hidden_size, patch_size ** 2 * in_channels)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.projection(modulate(self.norm(tokens), shift, scale))


class LatentDiT(nn.Module):
    """DiT-S/2 by default: 12 blocks, width 384, 6 heads, 576 latent tokens."""
    def __init__(self, input_size=48, patch_size=2, in_channels=4, hidden_size=384,
                 depth=12, num_heads=6, mlp_ratio=4., activation_checkpoint=False):
        super().__init__()
        if (min(input_size, patch_size, in_channels, hidden_size, depth, num_heads) < 1
                or input_size % patch_size or hidden_size % num_heads or hidden_size % 4
                or not math.isfinite(mlp_ratio) or mlp_ratio <= 0):
            raise ValueError('Require positive dimensions, patch-divisible input, and width divisible by heads and 4')
        self.input_size, self.patch_size, self.in_channels = input_size, patch_size, in_channels
        self.grid_size = input_size // patch_size
        self.activation_checkpoint = activation_checkpoint
        self.patch_embed = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.register_buffer('pos_embed', fixed_2d_positions(self.grid_size, hidden_size))
        self.time_embed = TimeEmbedding(hidden_size)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.patch_embed.weight.view(self.patch_embed.weight.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        for index in (0, 2):
            nn.init.normal_(self.time_embed.mlp[index].weight, std=.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.modulation[-1].weight)
        nn.init.zeros_(self.final_layer.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.projection.weight)
        nn.init.zeros_(self.final_layer.projection.bias)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.in_channels, self.input_size, self.input_size):
            raise ValueError(f'Expected [B,{self.in_channels},{self.input_size},{self.input_size}], got {tuple(x.shape)}')
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed.to(dtype=tokens.dtype)
        condition = self.time_embed(time)
        for block in self.blocks:
            if self.activation_checkpoint and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, condition, use_reentrant=False)
            else:
                tokens = block(tokens, condition)
        patches = self.final_layer(tokens, condition)
        batch, grid, patch = x.shape[0], self.grid_size, self.patch_size
        patches = patches.reshape(batch, grid, grid, patch, patch, self.in_channels)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(batch, self.in_channels, self.input_size, self.input_size)


class LatentEDM(nn.Module):
    """EDM denoiser; FP32 parameters, coefficients and loss, optional BF16 DiT.

    ``time_scale=1000`` maps EDM c_noise=log(sigma)/4 into the sinusoidal
    encoder's usual timestep range, matching the existing pixel-prior labels.
    DDP trainers should call their wrapped ``model(noisy, sigma)`` and compute
    the weighted MSE outside it; calling ``model.module.loss`` bypasses DDP.
    """
    sigma_min = .002
    sigma_max = 80.

    def __init__(self, input_size=48, patch_size=2, in_channels=4, hidden_size=384,
                 depth=12, num_heads=6, mlp_ratio=4., sigma_data=1., p_mean=-1.2,
                 p_std=1.2, activation_checkpoint=False, use_bf16=True, time_scale=1000.):
        super().__init__()
        if (not all(math.isfinite(v) for v in (sigma_data, p_mean, p_std, time_scale))
                or sigma_data <= 0 or p_std < 0 or time_scale <= 0):
            raise ValueError('Require finite EDM settings, positive sigma_data/time_scale and nonnegative p_std')
        self.config = dict(input_size=input_size, patch_size=patch_size, in_channels=in_channels,
                           hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                           sigma_data=sigma_data, p_mean=p_mean, p_std=p_std,
                           activation_checkpoint=activation_checkpoint, use_bf16=use_bf16, time_scale=time_scale)
        self.img_resolution, self.img_channels = input_size, in_channels
        self.sigma_data, self.p_mean, self.p_std = sigma_data, p_mean, p_std
        self.use_bf16, self.time_scale = use_bf16, time_scale
        self.net = LatentDiT(input_size, patch_size, in_channels, hidden_size, depth,
                             num_heads, mlp_ratio, activation_checkpoint)

    @staticmethod
    def _sigma(sigma, x):
        value = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).reshape(-1, 1, 1, 1)
        if value.shape[0] not in (1, x.shape[0]):
            raise ValueError('sigma must be scalar or have one value per example')
        return value.expand(x.shape[0], 1, 1, 1).clamp_min(1e-8)

    def forward(self, x, sigma):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            sigma = self._sigma(sigma, x)
            denominator = sigma.square() + self.sigma_data ** 2
            c_skip = self.sigma_data ** 2 / denominator
            c_out = sigma * self.sigma_data * denominator.rsqrt()
            c_in = denominator.rsqrt()
            scaled_input = x * c_in
            labels = .25 * self.time_scale * sigma.log().flatten()
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16,
                                enabled=self.use_bf16 and x.is_cuda):
                residual = self.net(scaled_input, labels)
            return c_skip * x + c_out * residual.float()

    def loss(self, clean, sigma=None, noise=None, generator=None):
        """FP32 scalar weighted denoising loss; supplied sigma/noise permit fixed validation."""
        with torch.autocast(device_type=clean.device.type, enabled=False):
            clean = clean.float()
            if sigma is None:
                sigma = (torch.randn(len(clean), device=clean.device, generator=generator) * self.p_std + self.p_mean).exp()
            sigma = self._sigma(sigma, clean)
            if noise is None:
                noise = torch.randn(clean.shape, device=clean.device, dtype=torch.float32, generator=generator)
            elif noise.shape != clean.shape:
                raise ValueError('noise must have the same shape as clean')
            noisy = clean + sigma * noise.float()
            weight = (sigma.square() + self.sigma_data ** 2) / (sigma * self.sigma_data).square()
            return (weight * (self(noisy, sigma) - clean).square()).mean()

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def sigma_schedule(steps=64, sigma_max=80., sigma_min=.002, rho=7., device='cpu'):
    if (steps < 2 or not all(math.isfinite(v) for v in (sigma_min, sigma_max, rho))
            or not 0 < sigma_min < sigma_max or rho <= 0):
        raise ValueError('Require steps >= 2, 0 < sigma_min < sigma_max, and positive rho')
    ramp = torch.linspace(0, 1, steps, dtype=torch.float64, device=device)
    sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1))).float()


@torch.no_grad()
def sample_latent(net, count=8, steps=64, seed=123, sigma_max=80., sigma_min=.002, rho=7.):
    """Deterministic EDM Heun integration, returning normalized diffusion latents."""
    device = next(net.parameters()).device
    sigmas = sigma_schedule(steps, sigma_max, sigma_min, rho, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(count, net.img_channels, net.img_resolution, net.img_resolution,
                    device=device, dtype=torch.float32, generator=generator) * sigmas[0]
    was_training = net.training
    net.eval()
    try:
        for index in range(steps):
            sigma, next_sigma = sigmas[index], sigmas[index + 1]
            derivative = (x - net(x, sigma)) / sigma
            proposal = x + (next_sigma - sigma) * derivative
            if index < steps - 1:
                next_derivative = (proposal - net(proposal, next_sigma)) / next_sigma
                proposal = x + (next_sigma - sigma) * (derivative + next_derivative) * .5
            x = proposal
        return x
    finally:
        net.train(was_training)
