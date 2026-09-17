from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from pixel_model import PixelDiT, load_pixel_dit
from png_data import PNGData, read_png


def test_model_budget_and_edm_initialization():
    torch.set_num_threads(2)
    model = PixelDiT(use_bf16=False)
    assert sum(p.numel() for p in model.parameters()) == 20_728_976
    x = torch.randn(1, 1, 96, 96)
    with torch.no_grad():
        torch.testing.assert_close(model(x, .5), .5*x)
    with pytest.raises(ValueError): PixelDiT(in_channels=4)


def test_transformer_trains_and_checkpoint_reloads(tmp_path):
    torch.set_num_threads(2)
    model = PixelDiT(hidden_size=32, depth=2, num_heads=2, patch_size=16, use_bf16=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
    clean = torch.rand(2, 1, 96, 96)*2-1
    sigma, noise = torch.tensor([.1, 1.]), torch.randn_like(clean)
    original = model.net.blocks[0].attention.qkv.weight.detach().clone()
    for _ in range(4):
        optimizer.zero_grad(); loss = model.loss(clean, sigma, noise)
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step()
    assert not torch.equal(original, model.net.blocks[0].attention.qkv.weight)
    path = tmp_path/'checkpoint.pt'
    torch.save(dict(architecture='pixel96_dit', model_config=model.config, ema=model.state_dict()), path)
    loaded, _ = load_pixel_dit(path, 'cpu')
    torch.testing.assert_close(loaded(clean, sigma), model(clean, sigma))
    torch.save(dict(architecture='unet'), path)
    with pytest.raises(ValueError, match='pixel96'): load_pixel_dit(path, 'cpu')


def make_data(root, duplicate=False):
    for i, part in enumerate(('train', 'val', 'test')):
        directory = root/part; directory.mkdir(parents=True)
        values = np.full((96, 96), 1234 if duplicate else 1234+i, dtype=np.uint16)
        Image.fromarray(values).save(directory/f'old__source__animal-{i}__000000__view.png')


def test_png_precision_and_holdout(tmp_path):
    make_data(tmp_path)
    data = PNGData(tmp_path, 'cpu', validation_limit=1)
    assert float(data.arrays['train'][0, 0, 0, 0]) == np.float32(1234/65535.)
    assert data.audit['splits']['test']['count'] == 1
    assert 'test' not in data.arrays  # Test images never become training targets.
    assert data.sample(2).shape == (2, 1, 96, 96)


def test_reject_cross_split_pixels(tmp_path):
    make_data(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match='overlap'): PNGData(tmp_path, 'cpu')


def test_reject_8bit_png(tmp_path):
    path = tmp_path/'image.png'
    Image.fromarray(np.zeros((96, 96), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match='16-bit'): read_png(path)
