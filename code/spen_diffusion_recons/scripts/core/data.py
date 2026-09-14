"""Reuse the existing subject split and audit the image provenance."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from project_paths import CORE, REFERENCE_ROOT, RAT_SPLIT

ROOT = CORE
PROJECT = REFERENCE_ROOT
DATA = PROJECT / 'data/0428_rat/hr'
SPLIT = RAT_SPLIT


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_split():
    split = json.loads(SPLIT.read_text())
    names = {k: split[k + '_files'] for k in ('train', 'val', 'test')}
    subjects = {k: {re.search(r'sub-\d+', n).group() for n in v} for k, v in names.items()}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if set(names[a]) & set(names[b]) or subjects[a] & subjects[b]:
            raise ValueError(f'Subject/file leakage: {a}, {b}')
    return names


def load_images(names):
    images = []
    for name in names:
        with Image.open(DATA / name) as im:
            x = np.asarray(im.convert('L'), dtype=np.float32) / 255.
        if x.shape != (96, 96) or not np.isfinite(x).all():
            raise ValueError(f'Invalid image: {name}')
        scale = max(float(np.quantile(x, .995)), 1e-6)
        images.append(np.clip(x / scale, 0, 1))
    return torch.from_numpy(np.stack(images))[:, None] * 2 - 1


def audit(output):
    names = read_split()
    hashes = {}
    for part, files in names.items():
        hashes[part] = {n: sha256(DATA / n) for n in files}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if set(hashes[a].values()) & set(hashes[b].values()):
            raise ValueError(f'Identical image bytes in {a} and {b}')
    report = dict(data_root=str(DATA), source_split=str(SPLIT), source_split_sha256=sha256(SPLIT),
                  counts={k: len(v) for k, v in names.items()}, species='rat', resolution=[96, 96],
                  normalization='per-image q0.995, clip to [0,1], then 2*x-1',
                  split_by='subject', legacy_excluded=926, files=hashes)
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    return names
