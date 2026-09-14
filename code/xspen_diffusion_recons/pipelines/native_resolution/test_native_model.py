"""Physical-grid invariants for native diffusion adaptation."""
import pytest
import torch
import torch.nn.functional as F
from native_model import NativePrior
from model import StrongPrior


@pytest.mark.parametrize('shape', [(46,48), (60,64), (62,64), (90,96), (92,96), (120,128), (124,128)])
def test_explicit_padding_and_gradient(shape):
    torch.set_num_threads(2)
    native = NativePrior(shape, base_ch=8, attention=True, dropout=0.)
    reference = StrongPrior(base_ch=8, attention=True, dropout=0.)
    reference.load_state_dict(native.state_dict())
    # Nonzero output makes the test sensitive to padding/cropping and network path.
    torch.nn.init.normal_(native.net.out[-1].weight, std=.01)
    reference.load_state_dict(native.state_dict())
    x = torch.randn(2, 1, *shape, requires_grad=True)
    dh, dw = (-shape[0]) % 8, (-shape[1]) % 8
    top, left = dh//2, dw//2
    expected = reference(F.pad(x, (left, dw-left, top, dh-top), value=-1.), .3)
    expected = expected[..., top:top+shape[0], left:left+shape[1]]
    actual = native(x, .3)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert native.net.init.weight.grad.abs().sum() > 0


def test_rejects_silent_physical_grid_change():
    net = NativePrior((46, 48), base_ch=8)
    with pytest.raises(ValueError, match='physical grid'):
        net(torch.zeros(1, 1, 48, 46), .1)
