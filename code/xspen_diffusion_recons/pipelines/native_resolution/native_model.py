"""Resolution-specific EDM prior using the existing SPEN strong U-Net.

Pad the noisy image with the known background (-1), evaluate EDM, then crop.
The physical image, loss, sampling state, and inverse operator are never resized.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from model import StrongPrior


class NativePrior(StrongPrior):
    def __init__(self, image_shape, **kwargs):
        super().__init__(**kwargs)
        self.image_shape = tuple(int(n) for n in image_shape)
        if len(self.image_shape) != 2 or min(self.image_shape) < 16:
            raise ValueError('Expected a two-dimensional physical image grid >= 16')
        self.config = dict(self.config, image_shape=list(self.image_shape))
        self.img_resolution = max(self.image_shape)

    def forward(self, x, sigma):
        if tuple(x.shape[-2:]) != self.image_shape:
            raise ValueError(f'Expected physical grid {self.image_shape}, got {x.shape[-2:]}')
        h, w = self.image_shape
        dh, dw = (-h) % 8, (-w) % 8
        top, left = dh // 2, dw // 2
        padded = F.pad(x, (left, dw-left, top, dh-top), value=-1.)
        pred = super().forward(padded, sigma)
        return pred[..., top:top+h, left:left+w]


def load_native_prior(path, device='cuda'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    net = NativePrior(**checkpoint['model_config']).to(device)
    net.load_state_dict(checkpoint['ema'])
    net.eval().requires_grad_(False)
    return net, checkpoint
