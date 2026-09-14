import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)

def save_checkpoint(path, value):
    path = Path(path)
    temp = path.with_suffix('.pt.tmp')
    torch.save(value, temp)
    os.replace(temp, path)

def save_grid(x, path, columns=6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a = ((x.detach().float().cpu().numpy()[:, 0] + 1) / 2).clip(0, 1)
    rows = math.ceil(len(a) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns*2, rows*2), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, im in zip(axes.flat, a):
        ax.imshow(im, cmap='gray', vmin=0, vmax=1)
    fig.tight_layout(pad=.1)
    fig.savefig(path, dpi=120)
    plt.close(fig)

@torch.no_grad()
def validate(net, clean):
    gen = torch.Generator(device=clean.device).manual_seed(20260909)
    sigma = (torch.randn(len(clean), 1, 1, 1, device=clean.device, generator=gen)*1.2-1.2).exp()
    noise = torch.randn(clean.shape, device=clean.device, generator=gen)
    weight = (sigma.square()+net.sigma_data**2)/(sigma*net.sigma_data).square()
    total = 0.
    net.eval()
    for i in range(0, len(clean), 16):
        sl = slice(i, i+16)
        pred = net(clean[sl]+sigma[sl]*noise[sl], sigma[sl])
        total += float(((pred-clean[sl]).square()*weight[sl]).flatten(1).mean(1).sum())
    return total/len(clean)
