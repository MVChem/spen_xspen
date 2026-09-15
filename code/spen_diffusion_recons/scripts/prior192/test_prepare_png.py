"""CPU checks for animal isolation and lossless 16-bit prior preparation."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from prepare_png import assign_groups, prepare, sha256


def make_export(root):
    (root / 'images').mkdir(parents=True)
    records, sources, expected = [], [], {}
    for group in range(12):
        digest = hashlib.sha256(f'volume-{group}'.encode()).hexdigest()
        source_id = f'source_{group}'
        sources.append(dict(source_id=source_id, source_sha256=digest,
                            metadata={'slice_axis': 2}))
        for plane in range(2):
            number = len(records) + 1
            pixels = (np.arange(192 * 192, dtype=np.uint32).reshape(192, 192)
                      + number * 512).astype(np.uint16)
            pixels[0, 0], pixels[0, 1], pixels[0, 2] = 0, 65535, 256
            name = f'{number}_group{group}_slice{plane}.png'
            path = root / 'images' / name
            Image.fromarray(pixels).save(path)
            expected[name] = pixels
            records.append(dict(num=number, filename=name, image=f'images/{name}',
                subject=f'animal_{group}', subject_group=f'lab:animal_{group}',
                species='rodent_unspecified', dataset='lab_mouse',
                sequence='RARE_TE20ms', source_id=source_id,
                source_path=str(root / f'raw_volume_{group}.nii'), source_sha256=digest,
                slice_index=plane, echo_index=None, png_sha256=sha256(path),
                pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                transform=dict(native_plane_shape=[192, 192], output_shape=[192, 192],
                    bit_depth=16, upsampled=False, added_padding=False,
                    sampling_yx=[1., 1.], offset_yx=[0., 0.],
                    normalization_window=[0., 65535.], normalization='per-source positive q99.5',
                    native_spacing_yx=[.1, .1], output_spacing_yx=[.1, .1], square_fov=19.2)))
    (root / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
    values = {'summary.json': dict(status='complete', image_count=len(records), size=[192, 192],
                                  bit_depth=16, upsampled_images=0, images_by_dataset={'lab_mouse': len(records)}),
              'verification.json': dict(passed=True, checked_images=len(records)),
              'processed_sources.json': sources, 'selected_sources.json': sources, 'config.json': {}}
    for name, value in values.items():
        (root / name).write_text(json.dumps(value))
    return records, expected


class PreparePNGTest(unittest.TestCase):
    def test_cross_dataset_animals_are_indivisible_and_order_independent(self):
        rows = []
        for animal in range(30):
            for dataset in ('dataset_a', 'dataset_b'):
                rows.append(dict(subject_group=f'shared-animal-{animal}',
                                 subject=f'{dataset}-local-id-{animal}', dataset=dataset, species='mouse'))
        assignment, _ = assign_groups(rows, 20260914, (.8, .1, .1))
        reordered, _ = assign_groups(list(reversed(rows)), 20260914, (.8, .1, .1))
        self.assertEqual(assignment, reordered)
        self.assertEqual(len(assignment), 30)
        self.assertEqual({info['part'] for info in assignment.values()}, {'train', 'val', 'test'})
        for info in assignment.values():
            self.assertEqual(info['datasets'], ['dataset_a', 'dataset_b'])
            self.assertEqual(info['images'], 2)

    def test_pack_preserves_all_uint16_pixels_and_normalization(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory) / 'export'
            _, expected = make_export(root)
            out = Path(directory) / 'packed'
            prepare(root, out, 20260914, (.8, .1, .1), 8)
            manifest = json.loads((out / 'manifest.json').read_text())
            observed, group_parts = {}, {}
            for part in ('train', 'val', 'test'):
                array = np.load(out / f'{part}.npy', allow_pickle=False)
                self.assertEqual(array.dtype, np.uint16)
                for image, record in zip(array, manifest['records'][part]):
                    np.testing.assert_array_equal(image, expected[record['filename']])
                    normalized = image.astype(np.float32) / 65535.
                    self.assertEqual(float(normalized.min()), 0.)
                    self.assertEqual(float(normalized.max()), 1.)
                    np.testing.assert_array_equal((normalized * 65535).round().astype(np.uint16), image)
                    self.assertEqual(record['species'], 'rodent_unspecified')
                    group = record['subject_group']
                    self.assertEqual(group_parts.setdefault(group, part), part)
                    observed[record['filename']] = part
            self.assertEqual(set(observed), set(expected))
            self.assertFalse((out / 'images').exists())

    def test_rejects_8bit_png_even_if_its_file_hash_is_consistent(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory) / 'export'
            records, expected = make_export(root)
            row = records[0]
            path = root / row['image']
            Image.fromarray((expected[row['filename']] >> 8).astype(np.uint8)).save(path)
            row['png_sha256'] = sha256(path)
            (root / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
            out = Path(directory) / 'packed'
            with self.assertRaisesRegex(ValueError, 'PNG bit depth'):
                prepare(root, out, 20260914, (.8, .1, .1), 8)
            self.assertFalse(out.exists())
            self.assertEqual(list(out.parent.glob('.packed.preparing-*')), [])


if __name__ == '__main__':
    unittest.main()
