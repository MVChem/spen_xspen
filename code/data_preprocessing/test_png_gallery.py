"""Regression checks for HTML-directory-only preview asset serving."""
from pathlib import Path
from urllib.parse import unquote
import re

import numpy as np
from PIL import Image
import pytest

from build_png_gallery import build_gallery


def dataset(tmp_path):
    root = tmp_path / 'data' / 'sample96'
    for part in ('train', 'val', 'test'):
        (root / part).mkdir(parents=True)
        pixels = (np.arange(96 * 96, dtype=np.uint16).reshape(96, 96) * 7)
        Image.fromarray(pixels).save(root / part / 'old__ds005236__sub-01__000000__mouse-fov16.png')
    return root


def test_preview_assets_are_local_independent_and_lossless(tmp_path):
    root = dataset(tmp_path)
    output = tmp_path / 'preview' / 'index.html'
    build_gallery(root, output)
    content = output.read_text()
    url_root = re.search(r'const root="([^"]+)"', content)[1]
    assert not url_root.startswith('/') and '..' not in Path(url_root).parts
    for source in root.rglob('*.png'):
        asset = output.parent / unquote(url_root) / source.relative_to(root)
        assert asset.resolve().is_relative_to(output.parent)
        assert asset.read_bytes() == source.read_bytes()
        assert not asset.is_symlink() and not asset.samefile(source)
    assert all(p.suffix == '.png' for p in root.rglob('*') if p.is_file())
    build_gallery(root, output)  # Rebuilding reuses already-copied bytes.


def test_reject_asset_symlink_escaping_preview(tmp_path):
    root = dataset(tmp_path)
    preview = tmp_path / 'preview'
    preview.mkdir()
    (preview / 'images').symlink_to(root.parent, target_is_directory=True)
    with pytest.raises(ValueError, match='inside the preview'):
        build_gallery(root, preview / 'index.html')


def test_reject_html_inside_image_dataset(tmp_path):
    root = dataset(tmp_path)
    with pytest.raises(ValueError, match='outside the image-only'):
        build_gallery(root, root / 'index.html')


def test_symlink_gallery_uses_original_data_and_preview_sheets(tmp_path):
    root = dataset(tmp_path)
    original = {str(p): p.read_bytes() for p in root.rglob('*.png')}
    output = tmp_path / 'preview' / 'index.html'
    build_gallery(root, output, asset_mode='symlink')
    link = output.parent / 'images' / root.name
    assert link.is_symlink() and link.resolve() == root.resolve()
    content = output.read_text()
    assert 'const atlas={"root": "preview_tiles/"' in content
    assert 'CSS sprites work in an opaque sandbox origin' in content
    assert '轻量 8 位 JPEG' in content
    sheets = sorted((output.parent / 'preview_tiles').glob('*.jpg'))
    assert len(sheets) == 1
    # The atlas tile ordering matches collect_images: train, val, test.
    with Image.open(sheets[0]) as image:
        assert image.mode == 'L' and image.size == (1536, 768)
        decoded = np.asarray(image, dtype=np.float32)
    source = next((root / 'train').glob('*.png'))
    with Image.open(source) as image:
        expected = np.rint(np.asarray(image, dtype=np.float32) / 257.)
    for index in range(3):
        error = decoded[:96, index*96:(index+1)*96] - expected
        assert np.sqrt(np.mean(error**2)) < 4
    assert original == {str(p): p.read_bytes() for p in root.rglob('*.png')}
    build_gallery(root, output, asset_mode='symlink')
