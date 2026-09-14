"""Unconditional EDM magnitude prior; continuous denoiser D(x, sigma)."""
import torch
from torch import nn
from tiny_unet import TinyUNet


class EDMPrior(nn.Module):
    img_resolution = 96
    img_channels = 1
    sigma_min = 0.002
    sigma_max = 80.0

    def __init__(self, base_ch=32, dropout=0.05, sigma_data=0.5):
        super().__init__()
        self.config = dict(base_ch=base_ch, dropout=dropout, sigma_data=sigma_data)
        self.sigma_data = sigma_data
        self.net = TinyUNet(base_ch=base_ch, dropout=dropout)
        nn.init.zeros_(self.net.out[-1].weight)
        nn.init.zeros_(self.net.out[-1].bias)

    def forward(self, x, sigma):
        x = x.float()
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1, 1, 1, 1)
        sigma = sigma.expand(x.shape[0], 1, 1, 1).clamp_min(1e-6)
        sd = self.sigma_data
        c_skip = sd**2 / (sigma.square() + sd**2)
        c_out = sigma * sd / (sigma.square() + sd**2).sqrt()
        c_in = (sigma.square() + sd**2).rsqrt()
        # Fixed frequency scaling for the inherited discrete-time sinusoidal encoder.
        labels = 250.0 * sigma.log().flatten()
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            residual = self.net(c_in * x, labels)
        return c_skip * x + c_out * residual.float()

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def sigma_schedule(steps=64, sigma_max=80., sigma_min=0.002, device='cpu'):
    if steps < 2 or not 0 < sigma_min < sigma_max:
        raise ValueError('Require steps >= 2 and 0 < sigma_min < sigma_max')
    ramp = torch.linspace(0, 1, steps, device=device, dtype=torch.float64)
    sigmas = (sigma_max**(1/7) + ramp * (sigma_min**(1/7) - sigma_max**(1/7)))**7
    return torch.cat([sigmas, sigmas.new_zeros(1)]).float()


@torch.no_grad()
def sample_prior(net, count=8, steps=64, seed=123):
    device = next(net.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    sigmas = sigma_schedule(steps, device=device)
    x = torch.randn(count, 1, net.img_resolution, net.img_resolution, device=device, generator=gen) * sigmas[0]
    for i, (s, sn) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        d = (x - net(x, s)) / s
        xn = x + (sn - s) * d
        if i < len(sigmas) - 2:
            dn = (xn - net(xn, sn)) / sn
            xn = x + (sn - s) * (d + dn) / 2
        x = xn
    return x


def load_prior(path, device='cuda'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    net = EDMPrior(**checkpoint['model_config']).to(device)
    net.load_state_dict(checkpoint['ema'])
    net.eval().requires_grad_(False)
    return net, checkpoint
