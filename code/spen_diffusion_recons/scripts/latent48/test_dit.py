"""CPU checks for latent shape, EDM arithmetic, learning, and checkpoint gradients."""
import copy

import pytest
import torch

from dit import LatentDiT, LatentEDM, sample_latent


def tiny_model(**kwargs):
    config = dict(input_size=8, hidden_size=32, depth=2, num_heads=4, use_bf16=False)
    config.update(kwargs)
    return LatentEDM(**config)


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_native_48_shape_zero_output_and_identity_blocks():
    model = LatentDiT(input_size=48, hidden_size=32, depth=2, num_heads=4)
    x = torch.randn(2, 4, 48, 48)
    assert model.pos_embed.shape == (1, 576, 32)
    output = model(x, torch.tensor([-5., 3.]))
    assert output.shape == x.shape
    assert torch.count_nonzero(output) == 0
    tokens, condition = torch.randn(2, 9, 32), torch.randn(2, 32)
    torch.testing.assert_close(model.blocks[0](tokens, condition), tokens, rtol=0, atol=0)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert not model.pos_embed.requires_grad


def test_initial_denoiser_matches_edm_closed_form_and_fp32_loss():
    model = tiny_model(sigma_data=1.3)
    clean, noise = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
    sigma = torch.tensor([.002, .5, 80.]).reshape(3, 1, 1, 1)
    noisy = clean + sigma * noise
    expected = 1.3 ** 2 / (sigma.square() + 1.3 ** 2) * noisy
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(noisy, sigma)
        loss = model.loss(clean, sigma=sigma, noise=noise)
    torch.testing.assert_close(output, expected)
    expected_loss = ((sigma.square() + 1.3 ** 2) / (sigma * 1.3).square() * (expected - clean).square()).mean()
    torch.testing.assert_close(loss, expected_loss)
    assert loss.ndim == 0 and loss.dtype == torch.float32 and output.dtype == torch.float32


def test_all_parameters_participate_and_transformer_learns_after_zero_init():
    torch.manual_seed(6)
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0.)
    clean, noise = torch.randn(4, 4, 8, 8), torch.randn(4, 4, 8, 8)
    sigma = torch.full((4,), .8)
    qkv_before = model.net.blocks[0].attention.qkv.weight.detach().clone()
    loss_before = float(model.loss(clean, sigma, noise).detach())
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(clean, sigma, noise)
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        optimizer.step()
    assert not torch.equal(model.net.blocks[0].attention.qkv.weight, qkv_before)
    assert float(model.loss(clean, sigma, noise).detach()) < loss_before


def test_activation_checkpoint_preserves_nonzero_gradients():
    torch.manual_seed(12)
    original = tiny_model()
    # Exercise the entire transformer, bypassing the initial zero gates / head.
    with torch.no_grad():
        for block in original.net.blocks:
            block.modulation[-1].weight.normal_(std=.02)
        original.net.final_layer.projection.weight.normal_(std=.02)
    checkpointed = tiny_model(activation_checkpoint=True)
    checkpointed.load_state_dict(original.state_dict())
    clean, noise = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    losses = [model.loss(clean, sigma=.5, noise=noise) for model in (original, checkpointed)]
    for loss in losses:
        loss.backward()
    torch.testing.assert_close(*losses)
    for (name, reference), candidate in zip(original.named_parameters(), checkpointed.parameters()):
        assert reference.grad is not None and candidate.grad is not None, name
        torch.testing.assert_close(reference.grad, candidate.grad, rtol=1e-5, atol=1e-7)


def test_heun_sampling_reproducible_finite_and_restores_mode():
    model = tiny_model()
    sample = sample_latent(model, count=2, steps=4, seed=21, sigma_max=2.)
    assert model.training
    assert sample.shape == (2, 4, 8, 8) and torch.isfinite(sample).all()
    torch.testing.assert_close(sample, sample_latent(model, count=2, steps=4, seed=21, sigma_max=2.), rtol=0, atol=0)
    assert not torch.equal(sample, sample_latent(model, count=2, steps=4, seed=22, sigma_max=2.))
    restored = LatentEDM(**model.config)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(restored(sample, .4), model(sample, .4))


def test_invalid_dimensions_and_sigma_batch_are_rejected():
    with pytest.raises(ValueError):
        tiny_model(input_size=7)
    with pytest.raises(ValueError):
        tiny_model(hidden_size=34)
    model = tiny_model()
    with pytest.raises(ValueError):
        model(torch.randn(2, 4, 8, 8), torch.ones(3))
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 8, 8), .5)
