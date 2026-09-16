"""Check that single-GPU microbatches preserve the effective batch objective."""
import copy

import torch

from train_strong import accumulated_backward


class ToyPrior(torch.nn.Module):
    sigma_data = .5

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x, sigma):
        return self.layer(x) / (1 + sigma)


class ToyData:
    def sample(self, count):
        return torch.rand(count, 1, 3, 3) * 2 - 1


def test_accumulated_gradient_equals_full_batch_objective():
    model = ToyPrior()
    reference = copy.deepcopy(model)
    data = ToyData()
    torch.manual_seed(19)
    loss = accumulated_backward(model, model, data, 3, 4, 'cpu')
    torch.manual_seed(19)
    clean, noisy, sigmas = [], [], []
    for _ in range(4):
        x = data.sample(3)
        sigma = (torch.randn(3, 1, 1, 1) * 1.2 - 1.2).exp()
        clean.append(x)
        noisy.append(x + sigma * torch.randn_like(x))
        sigmas.append(sigma)
    x, y, sigma = map(torch.cat, (clean, noisy, sigmas))
    weight = (sigma.square() + .5**2) / (sigma * .5).square()
    expected = (weight * (reference(y, sigma) - x).square()).mean()
    expected.backward()
    torch.testing.assert_close(loss, expected.detach())
    for actual, wanted in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, wanted.grad)
