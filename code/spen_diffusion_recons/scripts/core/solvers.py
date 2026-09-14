"""DiffPIR with exact SPEN proximal updates, and upstream InverseBench DAPS."""
import torch
from model import sigma_schedule


@torch.no_grad()
def diffpir(net, op, observation, steps=80, sigma_noise=.01, lamb=1., xi=0., seed=7,
            sigma_max=80., sigma_min=.02):
    """EDM-coordinate DiffPIR; rho=2*lambda*sigma_noise^2/sigma^2.

    Mirrors InverseBench/algo/diffpir.py's unscaled EDM branch, replacing its
    dense real matrix inverse with the exact complex-SPEN real-domain solve.
    No anchor blend, no warm start, and no ground truth enter this sampler.
    """
    if sigma_noise <= 0 or lamb <= 0 or not 0 <= xi <= 1:
        raise ValueError('Require noise/lambda > 0 and xi in [0,1]')
    gen = torch.Generator(device=op.device).manual_seed(seed)
    batch = observation.shape[0]
    sigmas = sigma_schedule(steps, sigma_max, sigma_min, op.device)
    x = torch.randn(batch, 1, *op.coils.shape[-2:], device=op.device, generator=gen)*sigmas[0]
    trace = []
    for i, (s, sn) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        x0 = net(x, s)
        rho = 2*lamb*sigma_noise**2/s.square()
        x0y = op.proximal(x0, observation, rho)
        effect = (x-x0y)/s
        noise = torch.randn(x.shape, device=x.device, generator=gen)
        x = x0y + ((1-xi)**.5*effect + xi**.5*noise)*sn
        if i % 10 == 0 or i == len(sigmas)-2:
            trace.append(dict(step=i, sigma=float(s), rho=float(rho),
                              residual=op.relative_residual(x0y, observation).cpu().tolist()))
        if not torch.isfinite(x).all():
            raise FloatingPointError(f'DiffPIR nonfinite state at step {i}')
    return x, trace


def daps(net, op, observation, steps=40, inner_steps=5, langevin_steps=50,
         tau=.02, lr=1e-4, seed=7):
    # Imported through the pinned repository path registered by operators.py.
    from algo.daps import DAPS
    algo = DAPS(net, op,
                annealing_scheduler_config=dict(num_steps=steps, sigma_max=80., sigma_min=.08,
                                                 sigma_final=0., schedule='linear', timestep='poly-7'),
                diffusion_scheduler_config=dict(num_steps=inner_steps, sigma_min=.002, sigma_final=0.,
                                                 schedule='linear', timestep='poly-7'),
                lgvd_config=dict(num_steps=langevin_steps, lr=lr, tau=tau, lr_min_ratio=.01))
    # Upstream DAPS num_samples means posterior draws of ONE observation.
    # Run distinct cases separately to avoid repeating a multi-case batch.
    if observation.shape[0] != 1:
        raise ValueError('DAPS adapter accepts one observation at a time')
    devices = [torch.cuda.current_device()] if observation.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        with torch.no_grad():
            result = algo.inference(observation, num_samples=1, verbose=False)
    if not torch.isfinite(result).all() or not bool(result.abs().sum() > 0):
        raise FloatingPointError('DAPS failed; result is nonfinite or upstream zero fallback')
    return result
