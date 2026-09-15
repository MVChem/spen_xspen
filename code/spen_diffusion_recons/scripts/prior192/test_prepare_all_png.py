"""Check complete cohort merging and the preservation of the previous dataset."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prepare_all_png import prepare_all, sha256
from prepare_png import prepare
from test_prepare_png import make_export


class PrepareAllPNGTest(unittest.TestCase):
    def test_full_merge_restores_export_order_and_preserves_parent_and_pixels(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            base = Path(directory)
            _, expected = make_export(base / 'export')
            prepare(base / 'export', base / 'split', 20260914, (.8, .1, .1), 8)
            parent_paths = [base / 'split' / name for name in ('manifest.json', 'train.npy', 'val.npy', 'test.npy')]
            before = {path: sha256(path) for path in parent_paths}
            prepare_all(base / 'split', base / 'all', expected_count=24)
            merged = json.loads((base / 'all/manifest.json').read_text())
            old = json.loads((base / 'split/manifest.json').read_text())
            pixels = np.load(base / 'all/train.npy', allow_pickle=False)
            self.assertEqual(pixels.dtype, np.uint16)
            self.assertEqual(pixels.shape, (24, 192, 192))
            self.assertEqual(merged['image_counts'], dict(train=24, val=0, test=0))
            self.assertEqual(merged['subject_counts'], dict(train=12, val=0, test=0))
            self.assertEqual(merged['records']['val'], [])
            self.assertEqual(merged['records']['test'], [])
            self.assertEqual(merged['selection_indices']['val'], [])
            self.assertFalse((base / 'all/val.npy').exists())
            self.assertFalse((base / 'all/test.npy').exists())
            for index, row in enumerate(merged['records']['train']):
                self.assertEqual(row['source_manifest_index'], index)
                self.assertEqual(row['num'], index + 1)
                self.assertEqual(row['split'], 'train')
                self.assertAlmostEqual(row['sample_weight'], 1 / 24)
                original = old['records'][row['original_split']][row['original_split_index']]
                self.assertEqual(row['key'], original['key'])
                np.testing.assert_array_equal(pixels[index], expected[row['filename']])
            self.assertEqual(before, {path: sha256(path) for path in parent_paths})

    def test_rejects_changed_parent_array_before_creating_new_dataset(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            base = Path(directory)
            make_export(base / 'export')
            prepare(base / 'export', base / 'split', 20260914, (.8, .1, .1), 8)
            pixels = np.load(base / 'split/test.npy', mmap_mode='r+')
            pixels[0, 0, 0] = 123
            pixels.flush()
            del pixels
            with self.assertRaisesRegex(ValueError, 'Parent test.npy changed'):
                prepare_all(base / 'split', base / 'all', expected_count=24)
            self.assertFalse((base / 'all').exists())


if __name__ == '__main__':
    unittest.main()
