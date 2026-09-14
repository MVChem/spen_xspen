import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

class PriorData:
    def __init__(self, root, device='cuda'):
        self.root = Path(root)
        self.device = device
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.arrays = {sp: np.load(self.root/f'{sp}.npy', mmap_mode='r') for sp in ('train', 'val')}
        records = self.manifest['records']['train']
        counts = Counter((r['subject'], r['modality'], r['view']) for r in records)
        self.weights = torch.tensor([self.manifest['modality_probability'][r['modality']]/counts[(r['subject'], r['modality'], r['view'])] for r in records])

    def sample(self, batch, augment=True):
        ids = torch.multinomial(self.weights, batch, replacement=True).numpy()
        x = torch.from_numpy(self.arrays['train'][ids].astype(np.float32)/65535.)[:, None].to(self.device)
        if augment:
            x = augment_magnitude(x)
        return x*2-1

    def validation(self, limit=256):
        records = self.manifest['records']['val']
        each = max(1, limit//len(self.manifest['subjects']['val']))
        selected = []
        for subject in self.manifest['subjects']['val']:
            choices = [i for i, r in enumerate(records) if r['subject'] == subject and r['modality'] == 'T2']
            selected.extend(choices[i] for i in np.linspace(0, len(choices)-1, each, dtype=int))
        x = torch.from_numpy(self.arrays['val'][selected].astype(np.float32)/65535.)[:, None].to(self.device)
        return x*2-1, [records[i]['key'] for i in selected]

def augment_magnitude(x):
    b, _, h, w = x.shape
    dev = x.device
    # Discrete scanner orientation changes apply to images only, never raw measurements.
    x = torch.rot90(x, int(torch.randint(4, (1,), device=dev)), (-2, -1))
    angle = torch.empty(b, device=dev).uniform_(-.12, .12)
    scale = torch.empty(b, device=dev).uniform_(.85, 1.10)
    flip = torch.where(torch.rand(b, device=dev) < .5, -1., 1.)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = angle.cos()*scale*flip
    theta[:, 0, 1] = -angle.sin()*scale
    theta[:, 1, 0] = angle.sin()*scale*flip
    theta[:, 1, 1] = angle.cos()*scale
    theta[:, :, 2] = torch.empty(b, 2, device=dev).uniform_(-.08, .08)
    out = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    gamma = torch.empty(b, 1, 1, 1, device=dev).uniform_(.85, 1.15)
    gain = torch.empty(b, 1, 1, 1, device=dev).uniform_(.90, 1.08)
    field = F.interpolate(torch.randn(b, 1, 4, 4, device=dev)*.06, size=(h, w), mode='bicubic', align_corners=False).exp()
    return (out.clamp_min(0).pow(gamma)*gain*field).clamp(0, 1)
