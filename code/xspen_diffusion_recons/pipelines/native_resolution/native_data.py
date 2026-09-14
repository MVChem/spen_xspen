"""Subject-balanced rectangular image batches from physical IXI resampling."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def augment_native(x):
    # A 90-degree rotation would exchange unequal PE/RO geometry; do not use it.
    b, _, h, w = x.shape
    angle = torch.empty(b, device=x.device).uniform_(-.08, .08)
    scale = torch.empty(b, device=x.device).uniform_(.94, 1.06)
    flip = torch.where(torch.rand(b, device=x.device) < .5, -1., 1.)
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = angle.cos()*scale*flip
    theta[:, 0, 1] = -angle.sin()*scale*h/w
    theta[:, 1, 0] = angle.sin()*scale*flip*w/h
    theta[:, 1, 1] = angle.cos()*scale
    theta[:, :, 2] = torch.empty(b, 2, device=x.device).uniform_(-.03, .03)
    out = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    gamma = torch.empty(b, 1, 1, 1, device=x.device).uniform_(.9, 1.1)
    gain = torch.empty(b, 1, 1, 1, device=x.device).uniform_(.95, 1.05)
    field = F.interpolate(torch.randn(b, 1, 4, 4, device=x.device)*.03, (h, w), mode='bicubic', align_corners=False).exp()
    return (out.clamp_min(0).pow(gamma)*gain*field).clamp(0, 1)


class NativeData:
    def __init__(self, root, device='cuda'):
        self.root = Path(root)
        self.device = device
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.arrays = {s: np.load(self.root/f'{s}.npy', mmap_mode='r') for s in ('train', 'val')}
        self.image_shape = tuple(self.arrays['train'].shape[-2:])
        subjects = self.manifest['subjects']
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            if set(subjects[a]) & set(subjects[b]):
                raise ValueError('Cross-subject split leakage')
        records = self.manifest['records']['train']
        if len(records) != len(self.arrays['train']):
            raise ValueError('Training records/array mismatch')
        counts = Counter((r['subject'], r['modality'], r['view']) for r in records)
        probability = self.manifest.get('modality_probability', {'T2': .8, 'PD': .2})
        self.weights = torch.tensor([probability[r['modality']]/counts[(r['subject'], r['modality'], r['view'])] for r in records])

    def sample(self, batch):
        ids = torch.multinomial(self.weights, batch, replacement=True).numpy()
        x = torch.from_numpy(self.arrays['train'][ids].astype(np.float32)/65535.)[:, None].to(self.device)
        return augment_native(x)*2-1

    def validation(self, limit=116):
        records = self.manifest['records']['val']
        indices = []
        for subject in self.manifest['subjects']['val']:
            for view in ('axial', 'sagittal'):
                choices = [i for i, r in enumerate(records) if r['subject'] == subject and r['modality'] == 'T2' and r['view'] == view]
                if choices:
                    indices.append(choices[len(choices)//2])
        if len(indices) > limit:
            indices = [indices[i] for i in np.linspace(0, len(indices)-1, limit, dtype=int)]
        if not indices:
            raise ValueError('No validation examples')
        x = torch.from_numpy(self.arrays['val'][indices].astype(np.float32)/65535.)[:, None].to(self.device)*2-1
        return x, [records[i]['key'] for i in indices]
