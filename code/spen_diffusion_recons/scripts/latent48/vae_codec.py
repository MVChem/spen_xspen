"""Frozen public f4 VAE, with a differentiable grayscale adapter.

The downloaded config/weights are used without architectural modifications.
All weights stay in FP32; optional autocast only changes operation precision.
Latents include the checkpoint's official scaling factor, but no dataset
normalization. Dataset-specific channel normalization belongs to the prior.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from diffusers import AutoencoderKL


class FrozenVAE(nn.Module):
    def __init__(self, local_path, device="cpu", *, autocast_dtype=None,
                 encode_batch_size=None, decode_batch_size=None):
        super().__init__()
        self.local_path = str(Path(local_path).resolve())
        model_path = Path(self.local_path)
        if (model_path / "vae" / "config.json").exists():
            model_path = model_path / "vae"
        self.vae = AutoencoderKL.from_pretrained(
            str(model_path), local_files_only=True, use_safetensors=True,
            torch_dtype=torch.float32,
        ).to(device)
        self.vae.requires_grad_(False).eval()
        self.latent_channels = int(self.vae.config.latent_channels)
        self.downsample_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        if self.downsample_factor != 4:
            raise ValueError(f"Expected a genuine f4 VAE; got f{self.downsample_factor}")
        self.scaling_factor = float(self.vae.config.scaling_factor)
        if self.vae.config.in_channels != 3 or self.vae.config.out_channels != 3:
            raise ValueError("This grayscale adapter expects an RGB checkpoint")
        self.autocast_dtype = autocast_dtype
        self.encode_batch_size = encode_batch_size
        self.decode_batch_size = decode_batch_size
        super().train(False)

    def train(self, mode=True):
        """The pretrained VAE remains frozen and in evaluation mode."""
        return super().train(False)

    def _precision(self, tensor):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(tensor.device.type, dtype=self.autocast_dtype)

    @staticmethod
    def _chunks(tensor, batch_size):
        if tensor.shape[0] < 1:
            raise ValueError("Empty VAE batches are not supported")
        if batch_size is None:
            return [tensor]
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return tensor.split(batch_size)

    def encode(self, x, sample=False, *, batch_size=None):
        """B1HW in [-1,1] -> scaled B4(H/4)(W/4), posterior mode by default.

        No input resizing, learned adapter, or image-dependent rescaling occurs.
        Use an outer torch.no_grad() during prior training to save memory.
        """
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected B1HW grayscale images, got {tuple(x.shape)}")
        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError("Image dimensions must be divisible by 4")
        chunk_size = self.encode_batch_size if batch_size is None else batch_size
        output = []
        for chunk in self._chunks(x, chunk_size):
            with self._precision(chunk):
                posterior = self.vae.encode(chunk.float().repeat(1, 3, 1, 1)).latent_dist
                z = posterior.sample() if sample else posterior.mode()
            output.append(z.float() * self.scaling_factor)
        result = torch.cat(output)
        if result.shape[-2:] != (x.shape[-2] // 4, x.shape[-1] // 4):
            raise RuntimeError("The loaded VAE did not produce the expected f4 shape")
        return result

    def decode(self, z, *, batch_size=None, clamp=True):
        """Scaled latents -> grayscale B1HW; gradients can flow back into z.

        RGB channels are averaged with fixed equal weights. Clipping enforces
        the established image range; clamp=False exposes raw decoder values.
        """
        if z.ndim != 4 or z.shape[1] != self.latent_channels:
            raise ValueError(f"Expected B{self.latent_channels}HW latents, got {tuple(z.shape)}")
        chunk_size = self.decode_batch_size if batch_size is None else batch_size
        output = []
        for chunk in self._chunks(z, chunk_size):
            with self._precision(chunk):
                rgb = self.vae.decode(chunk.float() / self.scaling_factor).sample
            output.append(rgb.float().mean(dim=1, keepdim=True))
        result = torch.cat(output)
        return result.clamp(-1, 1) if clamp else result

    def forward(self, x, sample=False):
        return self.decode(self.encode(x, sample=sample))
