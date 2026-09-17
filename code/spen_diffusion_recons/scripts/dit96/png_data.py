"""Read uint16 PNGs directly; keep audits in runs, never in the image dataset."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import importlib.util
import numpy as np
from PIL import Image
import torch

_spec = importlib.util.spec_from_file_location('prior96_augmentation',
    Path(__file__).resolve().parents[1] / 'prior96/data_v2.py')
_aug = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aug)
augment_magnitude = _aug.augment_magnitude


def parse_name(path):
    fields = path.stem.split('__')
    if len(fields) != 5:
        raise ValueError(f'Unexpected image name: {path}')
    if fields[0] == 'old':
        _, source, subject, index, view = fields
        return dict(source=source, subject=subject, view=view, old_index=int(index), batch='old')
    source, subject, scan, index, view = fields
    return dict(source=source, subject=subject, view=view, scan=scan,
                batch='lab' if source == 'lab-RAT' else 'public')


def read_png(path):
    raw = path.read_bytes()
    # Check the actual PNG IHDR bit depth, not Pillow's platform-dependent mode.
    if raw[:8] != b'\x89PNG\r\n\x1a\n' or raw[24:26] != bytes([16, 0]):
        raise ValueError(f'Expected 16-bit grayscale PNG: {path}')
    with Image.open(BytesIO(raw)) as image:
        array = np.asarray(image, dtype=np.uint16)
    if array.shape != (96, 96):
        raise ValueError(f'Expected 96x96 image: {path}')
    return array, sha256(raw).hexdigest(), sha256(array.astype('<u2').tobytes()).hexdigest()


class PNGData:
    def __init__(self, root, device='cuda', validation_limit=256, *, sampling='uniform', legacy_manifest=None):
        if sampling not in ('uniform', 'balanced'):
            raise ValueError('Unknown sampling policy')
        self.root = Path(root).resolve()
        self.device = torch.device(device)
        self.records, self.arrays, self.audit = {}, {}, {'root': str(self.root), 'splits': {}}
        fingerprints, subjects, pixel_hashes = {}, {}, {}
        for part in ('train', 'val', 'test'):
            directory = self.root / part
            paths = sorted(directory.glob('*.png'))
            if not paths or set(directory.iterdir()) != set(paths):
                raise ValueError(f'{directory} must contain PNG images only')
            with ThreadPoolExecutor(max_workers=4) as pool:
                decoded = list(pool.map(read_png, paths))
            records = [dict(filename=p.name, **parse_name(p)) for p in paths]
            self.records[part] = records
            audit_rows = [dict(filename=p.name, sha256=v[1], pixel_sha256=v[2])
                          for p, v in zip(paths, decoded)]
            fingerprints[part] = sha256(json.dumps(audit_rows, sort_keys=True).encode()).hexdigest()
            subjects[part] = {(r['source'], r['subject']) for r in records}
            pixel_hashes[part] = {v[2] for v in decoded}
            self.audit['splits'][part] = dict(count=len(paths), fingerprint=fingerprints[part],
                sources=dict(Counter(r['source'] for r in records)), images=audit_rows)
            if part != 'test':
                # GPU resident FP32 preserves uint16 precision. No NPY cache on disk.
                values = np.stack([v[0] for v in decoded]).astype(np.float32) / 65535.
                self.arrays[part] = torch.from_numpy(values[:, None]).to(self.device)
            del decoded
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            if subjects[a] & subjects[b] or pixel_hashes[a] & pixel_hashes[b]:
                raise ValueError(f'Subject or exact pixel overlap between {a} and {b}')
        self.fingerprint = sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
        self.audit['fingerprint'] = self.fingerprint
        # All train images are eligible, with equal per-image sampling probability.
        # This gives lab RARE 34.45% naturally, without assigning unknown species.
        self.audit['sampling'] = 'uniform over training PNG images, with replacement'
        self.weights = None
        self.sampling_probabilities = None
        if sampling == 'balanced':
            if legacy_manifest is None:
                raise ValueError('Balanced sampling requires the legacy training manifest')
            from balanced_sampling import balanced_weights
            weights, sampling_audit = balanced_weights(self.records['train'], legacy_manifest)
            self.sampling_probabilities = weights
            self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            self.audit['sampling'] = sampling_audit
        groups = defaultdict(list)
        for i, r in enumerate(self.records['val']):
            groups[(r['source'], r['subject'])].append(i)
        queues = [list(np.asarray(ids)[np.linspace(0, len(ids)-1,
                   min(len(ids), max(1, (validation_limit+len(groups)-1)//len(groups))), dtype=int)])
                  for _, ids in sorted(groups.items())]
        selected = []
        for position in range(max(map(len, queues))):
            for queue in queues:
                if position < len(queue):
                    selected.append(int(queue[position]))
                    if len(selected) == validation_limit:
                        break
            if len(selected) == validation_limit:
                break
        self.validation_ids = selected
        self.audit['validation'] = dict(filenames=[self.records['val'][i]['filename'] for i in selected],
            selection='subject round-robin, fixed order; every second image rotated 180 degrees',
            lab_holdout=False)

    def sample(self, batch):
        ids = (torch.randint(len(self.arrays['train']), (batch,), device=self.device)
               if self.weights is None else torch.multinomial(self.weights, batch, replacement=True))
        return augment_magnitude(self.arrays['train'][ids], rotate180=True) * 2 - 1

    def validation(self):
        x = self.arrays['val'][self.validation_ids].clone()
        x[::2] = torch.rot90(x[::2], 2, (-2, -1))
        return x * 2 - 1
