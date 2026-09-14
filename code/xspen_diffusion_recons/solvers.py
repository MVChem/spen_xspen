"""EDM DiffPIR adaptation with complex xSPEN/Fourier CG data consistency."""
import torch
from edm import sigma_schedule

@torch.no_grad()
def diffpir(net, op, observation, steps=60, sigma_noise=.02, lamb=1., xi=0., seed=7):
    if sigma_noise <= 0 or lamb <= 0 or not 0 <= xi <= 1:
        raise ValueError('Positive noise/lambda and xi in [0,1] required')
    gen = torch.Generator(device=op.device).manual_seed(seed)
    sigmas = sigma_schedule(steps, 80., .02, op.device)
    x = torch.randn(len(observation), 1, *op.image_shape, device=op.device, generator=gen)*sigmas[0]
    trace = []
    for i, (s, sn) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        x0 = net(x, s)
        rho = 2*lamb*sigma_noise**2/s.square()
        x0y = op.proximal(x0, observation, rho)
        effect = (x-x0y)/s
        noise = torch.randn(x.shape, device=x.device, generator=gen)
        x = x0y+((1-xi)**.5*effect+xi**.5*noise)*sn
        if i % 10 == 0 or i == len(sigmas)-2:
            trace.append(dict(step=i, sigma=float(s), rho=float(rho), residual=op.relative_residual(x0y, observation).cpu().tolist()))
        if not torch.isfinite(x).all():
            raise FloatingPointError(f'DiffPIR nonfinite at step {i}')
    return x, trace
