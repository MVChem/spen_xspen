"""Tiny 2D U-Net epsilon predictor for 96x96 magnitude diffusion."""

from __future__ import annotations

import math

import torch
from torch import nn


def _groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half - 1, 1)
        )
        args = t.to(torch.float32)[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time(self.act(emb))[:, :, None, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class TinyUNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base_ch: int = 32,
        ch_mults: tuple[int, ...] = (1, 2, 2, 4),
        dropout: float = 0.05,
        residual_blocks: int = 1,
    ) -> None:
        super().__init__()
        time_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.init = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        ch = base_ch
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels: list[int] = []
        for idx, mult in enumerate(ch_mults):
            out = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(residual_blocks):
                blocks.append(ResBlock(ch, out, time_dim, dropout))
                ch = out
                self.skip_channels.append(ch)
            self.down_blocks.append(blocks)
            if idx != len(ch_mults) - 1:
                self.downsamples.append(Downsample(ch))

        self.mid1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for idx, mult in enumerate(reversed(ch_mults)):
            out = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(residual_blocks):
                skip_ch = self.skip_channels.pop()
                blocks.append(ResBlock(ch + skip_ch, out, time_dim, dropout))
                ch = out
            self.up_blocks.append(blocks)
            if idx != len(ch_mults) - 1:
                self.upsamples.append(Upsample(ch))

        self.out = nn.Sequential(
            nn.GroupNorm(_groups(ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        emb = self.time_mlp(t)
        h = self.init(x)
        skips: list[torch.Tensor] = []

        for idx, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, emb)
                skips.append(h)
            if idx < len(self.downsamples):
                h = self.downsamples[idx](h)

        h = self.mid2(self.mid1(h, emb), emb)

        for idx, blocks in enumerate(self.up_blocks):
            for block in blocks:
                skip = skips.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    raise RuntimeError(f"skip shape {skip.shape} does not match {h.shape}")
                h = block(torch.cat([h, skip], dim=1), emb)
            if idx < len(self.upsamples):
                h = self.upsamples[idx](h)

        return self.out(h)
