"""Append a phase-net row while preserving every approved comparison row."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import render_rebuilt as renderer


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(arrays, simulation, output):
    keys = [*renderer.IMAGE_KEYS, 'phase_net']
    labels = list(map(str, arrays['labels']))
    if simulation:
        targets = np.concatenate([arrays['target'], arrays['target']])
        images = [targets] + [np.concatenate(arrays[key], axis=0) for key in keys]
        annotations = {key: renderer._metric_labels(values, targets)
                       for key, values in zip(keys, images[1:]) if key != 'degraded'}
        annotations['degraded'] = renderer._metric_labels(
            np.repeat(np.repeat(images[1], 2, -2), 2, -1), targets)
        n = len(labels)
        full, half = arrays['noise_sigma']
        groups = [(0, n, f'Full PE · σ = {full:g}'),
                  (n, 2 * n, f'Random 50% PE · σ = {half:g}')]
        return renderer._draw_grid(images, ['target', *keys], labels * 2, groups,
                                   output, annotations)
    groups = []
    start = 0
    fov = arrays['fov_mm']
    for stop in range(1, len(labels) + 1):
        if stop == len(labels) or fov[stop] != fov[start]:
            groups.append((start, stop, f'FOV {fov[start]:g} mm'))
            start = stop
    return renderer._draw_grid([arrays[key] for key in keys], keys, labels, groups, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=renderer.DEFAULT_FIGURES)
    parser.add_argument('--phase-results', type=Path, required=True)
    parser.add_argument('--approved', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    renderer.ROW_LABELS = {**renderer.ROW_LABELS, 'phase_net': 'Diffusion prior\n+ PhaseNet'}
    records = {}
    for kind, name in [('simulation', 'figure1_simulation'), ('real', 'figure2_real')]:
        original = read_arrays(args.reference / f'{kind}.npz')
        phase_result = read_arrays(args.phase_results / f'{kind}.npz')
        result = {**original, 'phase_net': phase_result['diffusion'],
                  'phase_net_unclipped': phase_result['diffusion_unclipped']}
        for key, values in original.items():
            if not np.array_equal(result[key], values):
                raise AssertionError(f'Changed original field {kind}/{key}')
        np.savez_compressed(args.out / f'{kind}.npz', **result)
        draw(result, kind == 'simulation', args.out / name)
        reference_image = np.asarray(Image.open(args.approved / f'{name}.png').convert('RGB'))
        new_image = np.asarray(Image.open(args.out / f'{name}.png').convert('RGB'))
        if new_image.shape[1] != reference_image.shape[1] or new_image.shape[0] <= reference_image.shape[0]:
            raise AssertionError('Unexpected figure geometry after adding a row')
        # Exclude the old bottom padding, now occupied by the next row's gap.
        visible = (reference_image < 245).any(axis=-1).mean(axis=1) > .5
        original_bottom = int(np.where(visible)[0][-1]) + 1
        changed = np.any(reference_image[:original_bottom] != new_image[:original_bottom], axis=-1)
        records[kind] = dict(original_fields_unchanged=list(original),
            original_rows=5 if kind == 'simulation' else 4,
            new_rows=6 if kind == 'simulation' else 5,
            reference_npz_sha256=digest(args.reference / f'{kind}.npz'),
            phase_result_npz_sha256=digest(args.phase_results / f'{kind}.npz'),
            approved_png_sha256=digest(args.approved / f'{name}.png'),
            old_image_shape=list(reference_image.shape), new_image_shape=list(new_image.shape),
            changed_pixels_in_original_rows=int(changed.sum()))
    (args.out / 'comparison.json').write_text(json.dumps(dict(
        change='Append diffusion + phase net; retain the original diffusion row and all earlier rows',
        simulation='Original observations without additional injected phase',
        real='Original scanner-corrected observations and coils; jointly learned residual phase',
        phase_results=str(args.phase_results), checks=records,
        source_sha256=digest(Path(__file__))), indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
