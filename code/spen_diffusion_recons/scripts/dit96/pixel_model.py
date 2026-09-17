"""20.73M-parameter pixel-space 96x96 EDM DiT; no VAE or latent encoder."""
from pathlib import Path
import importlib.util
import torch

_IMPL = Path(__file__).resolve().parents[1] / 'latent48' / 'dit.py'
_spec = importlib.util.spec_from_file_location('spen_shared_dit', _IMPL)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)


class PixelDiT(_shared.LatentEDM):
    def __init__(self, **kwargs):
        config = dict(input_size=96, patch_size=4, in_channels=1, hidden_size=320,
                      depth=11, num_heads=5, mlp_ratio=4., sigma_data=.5,
                      p_mean=-1.2, p_std=1.2, activation_checkpoint=False,
                      use_bf16=True, time_scale=1000.)
        config.update(kwargs)
        if config['input_size'] != 96 or config['in_channels'] != 1:
            raise ValueError('PixelDiT requires native 96x96 grayscale images')
        super().__init__(**config)


def load_pixel_dit(path, device='cuda'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('architecture') != 'pixel96_dit':
        raise ValueError('Expected a pixel96 DiT checkpoint')
    net = PixelDiT(**state['model_config']).to(device)
    net.load_state_dict(state['ema'], strict=True)
    return net.eval().requires_grad_(False), state


sample_prior = _shared.sample_latent
