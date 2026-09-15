"""Approximate latent DiffPIR-style reconstruction with a nonlinear data step.

The denoiser and all diffusion states use standardized VAE coordinates. Each
denoised estimate z0 is corrected by approximately minimizing

    ||A(D(u * std + mean)) - y||_2^2 + rho_z ||u - z0||_2^2,
    rho_z = 2 * lambda_z * sigma_noise^2 / sigma^2.

D is the UNCLIPPED frozen decoder; A includes the physical image normalization
and affine offset. This is not the old pixel-domain quadratic proximal, an
exact CG solve, the published DiffPIR algorithm, or exact posterior sampling.
lambda_z is a new latent regularization parameter, not equivalent to pixel
lambda. A warm start is an observation-derived image followed by posterior-
mode encoding and standardized latent Gaussian perturbation. No GT is used.

The adaptation is motivated by the DiffPIR PnP splitting framework:
https://arxiv.org/abs/2305.08995
Latent inverse problems require explicitly accounting for the decoder:
https://arxiv.org/abs/2307.00619
The power-law noise schedule and Euler coordinates follow EDM:
https://github.com/NVlabs/edm/blob/main/generate.py
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import time

import torch


@contextmanager
def _decoder_precision(codec, force_fp32):
    """Preserve the training encoder precision while selecting decoder precision."""
    has_setting = hasattr(codec, 'autocast_dtype')
    original = getattr(codec, 'autocast_dtype', None)
    if force_fp32 and has_setting:
        codec.autocast_dtype = None
    try:
        yield
    finally:
        if has_setting:
            codec.autocast_dtype = original


def latent_objective(u, z0, codec, mean, std, op, observation, rho):
    """Differentiable unnormalized sums, including complex measurement residuals."""
    image = codec.decode(u * std + mean, clamp=False)
    residual = op.forward(image) - observation
    data = residual.abs().square().sum()
    regularization = rho * (u - z0).square().sum()
    return data + regularization, data, regularization


def _numbers(values):
    total, data, regularization = values
    return dict(objective=float(total.detach()), data_term=float(data.detach()),
                regularization=float(regularization.detach()))


def latent_proximal(z0, codec, mean, std, op, observation, rho, *, inner_steps=8,
                    prox_lr=.1, max_backtracks=8, grad_rtol=1e-4,
                    objective_rtol=1e-6, armijo=1e-4):
    """Finite RMS-preconditioned gradient descent with monotone Armijo steps.

    The second-moment preconditioner has no momentum, so each direction is a
    descent direction. All objective values are sums, not averages. Stagnation
    and exhausted inner budgets are reported separately from gradient-based
    convergence. The start is z0 for every subproblem.
    """
    z0 = z0.detach()
    u = z0.clone()
    second_moment = torch.zeros_like(u)
    rows, accepted_steps, function_evaluations, gradient_evaluations = [], 0, 0, 0
    initial_gradient_norm = None
    stop_reason = 'inner_budget'
    step_size = float(prox_lr)
    first_values = last_values = None
    final_gradient_norm = None
    # Explicit enable_grad works even when the evaluation driver uses no_grad.
    with torch.enable_grad():
        for index in range(inner_steps + 1):
            u = u.detach().requires_grad_(True)
            values = latent_objective(u, z0, codec, mean, std, op, observation, rho)
            function_evaluations += 1
            if not all(bool(torch.isfinite(value)) for value in values):
                raise FloatingPointError('Nonfinite latent proximal objective')
            gradient, = torch.autograd.grad(values[0], u)
            gradient_evaluations += 1
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError('Nonfinite latent decoder/operator gradient')
            current = _numbers(values)
            gradient_norm = float(gradient.norm())
            final_gradient_norm = gradient_norm
            last_values = current
            if first_values is None:
                first_values = current
                initial_gradient_norm = gradient_norm
            # Free decoder activations before evaluating trial steps.
            del values
            if gradient_norm <= max(1e-9, grad_rtol * initial_gradient_norm):
                stop_reason = 'gradient_tolerance'
                break
            if index == inner_steps:
                break
            second_moment.mul_(.9).addcmul_(gradient, gradient, value=.1)
            corrected = second_moment / (1 - .9 ** (index + 1))
            direction = -gradient / (corrected.sqrt() + 1e-8)
            slope = float((gradient * direction).sum())
            if not math.isfinite(slope) or slope >= 0:
                raise FloatingPointError('Latent proximal direction is not a descent direction')
            trial_step = min(step_size, float(prox_lr))
            accepted, trial_numbers = False, None
            backtracks = 0
            with torch.no_grad():
                for backtracks in range(max_backtracks + 1):
                    candidate = u.detach() + trial_step * direction
                    trial_values = latent_objective(candidate, z0, codec, mean, std,
                                                    op, observation, rho)
                    function_evaluations += 1
                    if all(bool(torch.isfinite(value)) for value in trial_values):
                        trial_numbers = _numbers(trial_values)
                        if trial_numbers['objective'] <= current['objective'] + armijo * trial_step * slope:
                            accepted = True
                            break
                    trial_step *= .5
            row = dict(iteration=index, before=current,
                       after=trial_numbers if accepted else current,
                       gradient_norm=gradient_norm,
                       gradient_rms=gradient_norm / math.sqrt(u.numel()),
                       directional_derivative=slope, accepted=accepted,
                       step_size=trial_step if accepted else 0., backtracks=backtracks)
            rows.append(row)
            if not accepted:
                stop_reason = 'line_search_stalled'
                break
            accepted_steps += 1
            u = candidate.detach()
            last_values = trial_numbers
            step_size = min(float(prox_lr), trial_step * 2.)
            decrease = current['objective'] - trial_numbers['objective']
            if decrease <= objective_rtol * max(abs(current['objective']), 1e-12):
                # Record the actual final gradient on the accepted candidate.
                with torch.enable_grad():
                    u = u.detach().requires_grad_(True)
                    final_values = latent_objective(u, z0, codec, mean, std, op, observation, rho)
                    final_gradient, = torch.autograd.grad(final_values[0], u)
                    function_evaluations += 1
                    gradient_evaluations += 1
                    final_gradient_norm = float(final_gradient.norm())
                stop_reason = ('gradient_tolerance' if final_gradient_norm <=
                               max(1e-9, grad_rtol * initial_gradient_norm)
                               else 'objective_stagnation')
                break
    diagnostics = dict(solver='RMS-preconditioned GD with Armijo backtracking',
                       objective_before=first_values['objective'],
                       objective_after=last_values['objective'],
                       data_before=first_values['data_term'], data_after=last_values['data_term'],
                       regularization_after=last_values['regularization'],
                       gradient_norm_before=initial_gradient_norm,
                       gradient_norm_after=final_gradient_norm,
                       relative_gradient_norm=final_gradient_norm / max(initial_gradient_norm, 1e-30),
                       accepted_steps=accepted_steps, inner_budget=inner_steps,
                       converged=stop_reason == 'gradient_tolerance', stop_reason=stop_reason,
                       function_evaluations=function_evaluations,
                       gradient_evaluations=gradient_evaluations, inner_trace=rows)
    return u.detach(), diagnostics


@torch.no_grad()
def latent_reconstruct(net, codec, mean, std, op, observation, initial_image=None,
                       steps=60, inner_steps=8, lamb=1., sigma_noise=.01,
                       sigma_max=1., sigma_min=.02, seed=20260915, prox_lr=.1, *,
                       decoder_fp32=True, max_backtracks=8, grad_rtol=1e-4,
                       objective_rtol=1e-6, schedule_rho=7., progress_callback=None):
    """Return raw decoded image (unclipped [-1,1] coordinates) and JSON diagnostics.

    Use one call and a distinct deterministic seed per case. ``initial_image``
    must be obtained from observation-side reconstruction, e.g. bicubic Tikh96;
    it is encoded at the codec's configured (training) precision. Omitting it
    uses Gaussian noise initialization. Decoder input gradients run in FP32 by
    default; checkpoint tensors and training files are never modified.

    progress_callback receives each completed outer-step diagnostic dictionary.
    This function supports no_grad callers, but must not be run in inference_mode.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError('Use torch.no_grad, not inference_mode: decoder input gradients are required')
    if steps < 2 or inner_steps < 1 or max_backtracks < 0:
        raise ValueError('Require steps >= 2, inner_steps >= 1, max_backtracks >= 0')
    positive = (lamb, sigma_noise, sigma_max, sigma_min, prox_lr, schedule_rho, grad_rtol)
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in positive):
        raise ValueError('Regularization, noise schedule, learning rate and gradient tolerance must be positive finite')
    if sigma_max < sigma_min or not 0 <= objective_rtol < 1:
        raise ValueError('Require sigma_max >= sigma_min and 0 <= objective_rtol < 1')
    if not bool(torch.isfinite(observation).all()) or observation.shape[0] != 1:
        raise ValueError('One finite case is required per call to preserve independent case seeds')
    device = observation.device
    channels, resolution = int(net.img_channels), int(net.img_resolution)
    mean = torch.as_tensor(mean, device=device, dtype=torch.float32).reshape(1, channels, 1, 1)
    std = torch.as_tensor(std, device=device, dtype=torch.float32).reshape(1, channels, 1, 1)
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        raise ValueError('Finite means and positive finite standard deviations are required')
    net.requires_grad_(False).eval()
    codec.requires_grad_(False).eval()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn(1, channels, resolution, resolution, generator=generator, device=device)
    if initial_image is None:
        encoded = torch.zeros_like(noise)
        initialization = 'standardized_latent_gaussian_noise'
    else:
        if initial_image.ndim != 4 or initial_image.shape[:2] != (1, 1):
            raise ValueError('initial_image must be one [1,1,H,W] observation-derived image')
        if not bool(torch.isfinite(initial_image).all()):
            raise ValueError('Nonfinite warm-start image')
        encoded = (codec.encode(initial_image.to(device=device, dtype=torch.float32), sample=False) - mean) / std
        if encoded.shape != noise.shape:
            raise ValueError(f'Warm-start latent {tuple(encoded.shape)} differs from model {tuple(noise.shape)}')
        initialization = 'observation_image_posterior_mode_encode_plus_latent_gaussian_noise'
    z = encoded + float(sigma_max) * noise
    ramp = torch.linspace(0, 1, steps, dtype=torch.float64, device=device)
    sigmas = (sigma_max ** (1 / schedule_rho) + ramp *
              (sigma_min ** (1 / schedule_rho) - sigma_max ** (1 / schedule_rho))) ** schedule_rho
    sigmas = torch.cat((sigmas.float(), sigmas.new_zeros(1).float()))
    started = time.monotonic()
    original_codec_dtype = getattr(codec, 'autocast_dtype', None)
    diagnostics = dict(algorithm='latent DiffPIR-style approximate nonlinear proximal, deterministic EDM Euler',
                       exact_posterior_sampler=False, exact_proximal=False,
                       objective='sum(abs(A(D(u*std+mean))-y)^2) + rho_z*sum((u-z0)^2)',
                       rho_z='2*lambda_z*sigma_noise^2/sigma^2',
                       lambda_coordinate='standardized latent; not equivalent to previous pixel lambda',
                       initialization=initialization, initialization_uses_ground_truth=False,
                       initialization_encoder_autocast=str(original_codec_dtype),
                       decoder_precision='float32' if decoder_fp32 else str(original_codec_dtype),
                       decoder_clamp=False, seed=int(seed), steps=int(steps), inner_steps=int(inner_steps),
                       lambda_z=float(lamb), sigma_noise=float(sigma_noise), sigma_max=float(sigma_max),
                       sigma_min=float(sigma_min), schedule_rho=float(schedule_rho), prox_lr=float(prox_lr),
                       max_backtracks=int(max_backtracks), grad_rtol=float(grad_rtol),
                       objective_rtol=float(objective_rtol), sigmas=sigmas.tolist(),
                       references=['https://arxiv.org/abs/2305.08995', 'https://arxiv.org/abs/2307.00619',
                                   'https://github.com/NVlabs/edm/blob/main/generate.py'], trace=[])
    with _decoder_precision(codec, decoder_fp32), torch.autocast(device.type, enabled=False):
        for index, (sigma, next_sigma) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            z0 = net(z, sigma).float().detach()
            if not bool(torch.isfinite(z0).all()):
                raise FloatingPointError(f'Nonfinite DiT estimate at step {index}')
            rho = 2 * float(lamb) * float(sigma_noise) ** 2 / float(sigma) ** 2
            corrected, prox = latent_proximal(z0, codec, mean, std, op, observation, rho,
                                              inner_steps=inner_steps, prox_lr=prox_lr,
                                              max_backtracks=max_backtracks, grad_rtol=grad_rtol,
                                              objective_rtol=objective_rtol)
            z = corrected + (z - corrected) * (next_sigma / sigma)
            if not bool(torch.isfinite(z).all()):
                raise FloatingPointError(f'Nonfinite latent diffusion state at step {index}')
            row = dict(step=index, sigma=float(sigma), next_sigma=float(next_sigma), rho_z=rho,
                       latent_rms=float(z.square().mean().sqrt()), elapsed_seconds=time.monotonic() - started,
                       **prox)
            diagnostics['trace'].append(row)
            if progress_callback is not None:
                progress_callback(row)
        raw_image = codec.decode(z * std + mean, clamp=False).detach()
    diagnostics.update(elapsed_seconds=time.monotonic() - started,
                       accepted_steps=sum(row['accepted_steps'] for row in diagnostics['trace']),
                       converged_subproblems=sum(row['converged'] for row in diagnostics['trace']),
                       total_subproblems=steps, outside_range_fraction=float(
                           ((raw_image < -1) | (raw_image > 1)).float().mean()),
                       final_data_term=float((op.forward(raw_image) - observation).abs().square().sum()))
    if not bool(torch.isfinite(raw_image).all()):
        raise FloatingPointError('Nonfinite final decoded reconstruction')
    return raw_image, diagnostics
