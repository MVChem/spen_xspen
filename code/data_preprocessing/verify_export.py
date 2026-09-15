"""Verify every PNG, manifest hash, unique pixel array and contiguous filename ID."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def verify(run):
    run = Path(run)
    summary = json.loads((run / 'summary.json').read_text())
    records = [json.loads(line) for line in (run / 'manifest.jsonl').read_text().splitlines()]
    errors, names, pixels_seen = [], set(), set()
    datasets, sequences = Counter(), Counter()
    expected_depth = summary['bit_depth']
    for index, record in enumerate(records, 1):
        name = record['filename']
        try:
            if record['num'] != index or not name.startswith(f'{index}_') or name in names:
                raise ValueError('Noncontiguous or duplicate filename ID')
            names.add(name)
            path = run / record['image']
            if path != run / 'images' / name:
                raise ValueError('Manifest image path differs from filename')
            binary = path.read_bytes()
            if binary[:8] != b'\x89PNG\r\n\x1a\n' or binary[24] != expected_depth or binary[25] != 0:
                raise ValueError('Unexpected PNG type or bit depth')
            if hashlib.sha256(binary).hexdigest() != record['png_sha256']:
                raise ValueError('PNG file SHA256 mismatch')
            with Image.open(path) as image:
                if list(image.size) != summary['size']:
                    raise ValueError('Wrong pixel dimensions')
                array = np.asarray(image, dtype=np.uint16 if expected_depth == 16 else np.uint8)
            transform = record['transform']
            minimum = summary.get('minimum_native_size')
            if minimum and min(transform['native_plane_shape']) < minimum:
                raise ValueError('Native matrix is below requested minimum')
            if minimum and transform['upsampled']:
                raise ValueError('Filtered export contains upsampling')
            if summary.get('crop_mode') == 'foreground':
                if transform.get('added_padding'):
                    raise ValueError('Foreground crop contains added padding')
                sampling = np.asarray(transform['sampling_yx'])
                offset = np.asarray(transform['offset_yx'])
                last = offset + sampling * (np.asarray(transform['output_shape']) - 1)
                if (offset < -1e-6).any() or (last > np.asarray(transform['native_plane_shape']) - 1 + 1e-6).any():
                    raise ValueError('Crop samples fall outside acquired pixels')
                if transform['foreground_retained_fraction'] < 0.98:
                    raise ValueError('Crop excludes excessive detected foreground')
            if array.ndim != 2 or array.max() == array.min():
                raise ValueError('Not a nonconstant grayscale image')
            digest = hashlib.sha256(array.tobytes()).hexdigest()
            if digest != record['pixel_sha256'] or digest in pixels_seen:
                raise ValueError('Pixel hash mismatch or duplicate image')
            pixels_seen.add(digest)
            datasets[record['dataset']] += 1
            sequences[record['sequence']] += 1
        except Exception as error:
            errors.append(dict(filename=name, error=str(error)))
    if summary['image_count'] != len(records):
        errors.append(dict(error='Summary count differs from manifest'))
    if names != {p.name for p in (run / 'images').iterdir()}:
        errors.append(dict(error='Images directory differs from manifest'))
    if dict(datasets) != summary['images_by_dataset'] or dict(sequences) != summary['images_by_sequence']:
        errors.append(dict(error='Summary dataset/sequence counts differ from images'))
    if json.loads((run / 'failed_sources.json').read_text()):
        errors.append(dict(error='Run contains source read failures'))
    report = dict(passed=not errors, checked_images=len(records), size=summary['size'],
                  bit_depth=expected_depth, file_hashes_checked=True, pixel_hashes_checked=True,
                  contiguous_numbering=True if not errors else None, errors=errors)
    if summary.get('minimum_native_size'):
        report['minimum_native_size_checked'] = summary['minimum_native_size']
        report['no_upsampling_checked'] = True
    if summary.get('crop_mode') == 'foreground':
        report['crop_sampling_inside_native_fov_checked'] = True
    (run / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    report = verify(parser.parse_args().run)
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(0 if report['passed'] else 1)
