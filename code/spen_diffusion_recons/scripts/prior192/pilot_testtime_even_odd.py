"""Per-acquisition tiny phase-network pilot; no diffusion weights are updated.

The new objective corrects acquired even ROFFT lines BEFORE partial decoding.
It aligns relative odd/even phase, not their differently encoded raw signals.
Block-held-out decoded pixels assess spatial fitting, not independent MRI data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy.io
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT.parent / 'spenpy'))
from phase_inva import (coords_grid, fit_tiny_phase_scanner_batch,
                        load_phase_matrices, scanner_batch_phase_inputs, wrap_phase)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(y, odd_inv, even_inv):
    return (torch.einsum('ij,cjw->ciw', odd_inv, y[:, ::2]),
            torch.einsum('ij,cjw->ciw', even_inv, y[:, 1::2]))


def correct(y, phase):
    result = y.clone()
    result[:, 1::2] = y[:, 1::2] * torch.exp(-1j * phase)
    return result


def phase_cost(odd, even, weights):
    cross = (even * odd.conj()).sum(0)
    return (weights * (1 - cross.real / cross.abs().clamp_min(1e-10))).sum() / weights.sum().clamp_min(1e-10)


def make_weights(y, odd_inv, even_inv):
    scanner = y.permute(1, 2, 0).unsqueeze(2)[None]
    weights = scanner_batch_phase_inputs(scanner, odd_inv, even_inv)[2][0]
    rows, cols = torch.meshgrid(torch.arange(weights.shape[0], device=y.device),
                               torch.arange(weights.shape[1], device=y.device), indexing='ij')
    heldout = ((rows // 6 + 2 * (cols // 8)) % 5 == 0)
    train, evaluation = weights * ~heldout, weights * heldout
    if not bool(train.sum() > 0) or not bool(evaluation.sum() > 0):
        raise ValueError('Insufficient foreground for fixed spatial split')
    return weights, train, evaluation


class TinyPhase(nn.Module):
    def __init__(self):
        super().__init__()
        widths = [2, 16, 16, 16, 16, 16, 1]
        layers = []
        for i, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(left, right))
            if i < len(widths) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, coords, shape):
        return self.net(coords).reshape(shape)


def fit(y, odd_inv, even_inv, train_weights, steps, seed):
    # Same initialization is used for each member of the perturbation pair.
    with torch.random.fork_rng(devices=[y.device.index] if y.is_cuda else []):
        torch.manual_seed(seed)
        net = TinyPhase().to(y.device)
    shape = train_weights.shape
    coords = coords_grid(*shape, y.device)
    odd = decode(y, odd_inv, even_inv)[0].detach()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.02, weight_decay=1e-4)
    trace = []
    for step in range(steps):
        phase = net(coords, shape)
        even = torch.einsum('ij,cjw->ciw', even_inv,
                            y[:, 1::2] * torch.exp(-1j * phase))
        alignment = phase_cost(odd, even, train_weights)
        smoothness = (wrap_phase(phase[1:] - phase[:-1]).square().mean()
                      + wrap_phase(phase[:, 1:] - phase[:, :-1]).square().mean())
        loss = alignment + .002 * smoothness + .0001 * phase.square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite phase objective')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            trace.append(dict(step=step, objective=float(loss.detach()),
                              alignment=float(alignment.detach())))
    with torch.no_grad():
        phase = net(coords, shape).detach()
    return phase, trace, {k: v.detach().cpu() for k, v in net.state_dict().items()}


@torch.no_grad()
def diagnostics(y, odd_inv, even_inv, weights, train_weights, eval_weights):
    odd, even = decode(y, odd_inv, even_inv)
    cross = (even * odd.conj()).sum(0)
    angle = torch.angle(cross)
    return dict(phase_cost_all=float(phase_cost(odd, even, weights)),
                phase_cost_train=float(phase_cost(odd, even, train_weights)),
                phase_cost_spatial_holdout=float(phase_cost(odd, even, eval_weights)),
                weighted_phase_rms_rad=float(((weights * angle.square()).sum() / weights.sum()).sqrt()))


def run_case(path, name, args):
    fields = ['spen_original_signal_rofft', 'spen_phase_corrected_signal_rofft']
    mat = scipy.io.loadmat(path, variable_names=fields)
    values = []
    for field in fields:
        raw = np.asarray(mat[field])
        if raw.shape != (96, 96, 1, 4):
            raise ValueError(raw.shape)
        values.append(torch.as_tensor(raw[:, :, 0].transpose(2, 0, 1),
                                      device=args.device, dtype=torch.complex64))
    scale = values[0].abs().square().mean().sqrt()
    raw, scanner = [value / scale for value in values]
    inv, odd_inv, even_inv = load_phase_matrices(path, args.device)
    weights, train_weights, eval_weights = make_weights(raw, odd_inv, even_inv)
    phase, trace, state = fit(raw, odd_inv, even_inv, train_weights, args.steps, args.seed)
    corrected = correct(raw, phase)
    scanner_shape = raw.permute(1, 2, 0).unsqueeze(2)[None]
    with torch.random.fork_rng(devices=[raw.device.index] if raw.is_cuda else []):
        torch.manual_seed(args.seed)
        old_phase = fit_tiny_phase_scanner_batch(scanner_shape, odd_inv, even_inv,
                                                steps=args.steps)[0]
    signals = dict(raw=raw, scanner=scanner, legacy_tiny=correct(raw, old_phase), tiny=corrected)
    methods = {key: diagnostics(y, odd_inv, even_inv, weights, train_weights, eval_weights)
               for key, y in signals.items()}
    error = float((corrected.abs() - raw.abs()).abs().max())
    if error > 2e-6 * float(raw.abs().max()):
        raise AssertionError('Phase-only correction altered signal magnitudes')

    # Known additional acquired-domain phase. The scanner-corrected signal is
    # only a common baseline: its absolute unknown phase is not called truth.
    pair_weights, pair_train, pair_eval = make_weights(scanner, odd_inv, even_inv)
    coords = coords_grid(48, 96, raw.device).reshape(48, 96, 2)
    xx, yy = coords[..., 0], coords[..., 1]
    injected_phase = .6 + .7 * xx - .5 * yy + .25 * xx * yy
    injected = correct(scanner, -injected_phase)
    baseline_phase, _, _ = fit(scanner, odd_inv, even_inv, pair_train, args.steps, args.seed)
    recovered_phase, injected_trace, _ = fit(injected, odd_inv, even_inv, pair_train, args.steps, args.seed)
    recovered_difference = wrap_phase(recovered_phase - baseline_phase)
    phase_error = wrap_phase(recovered_difference - injected_phase)
    # A signal-energy weighting is appropriate in the acquired phase grid.
    phase_weights = scanner[:, 1::2].abs().square().sum(0)
    rms = lambda value: float(((phase_weights * value.square()).sum() / phase_weights.sum()).sqrt())
    restored = correct(injected, recovered_phase)
    baseline_corrected = correct(scanner, baseline_phase)
    before = (injected - scanner).abs().square().sum().sqrt() / scanner.abs().square().sum().sqrt()
    after = (restored - baseline_corrected).abs().square().sum().sqrt() / baseline_corrected.abs().square().sum().sqrt()
    paired = dict(injected_phase_formula_rad='.6 + .7*x - .5*y + .25*x*y; coordinates in [-1,1]',
                  injected_phase_weighted_rms_rad=rms(injected_phase),
                  recovered_difference_error_weighted_rms_rad=rms(phase_error),
                  relative_signal_difference_before=float(before),
                  relative_signal_difference_after=float(after),
                  baseline=diagnostics(baseline_corrected, odd_inv, even_inv, pair_weights, pair_train, pair_eval),
                  recovered=diagnostics(restored, odd_inv, even_inv, pair_weights, pair_train, pair_eval),
                  trace=injected_trace)

    arrays = dict(phase=phase, legacy_phase=old_phase, weights=weights,
                  train_weights=train_weights, spatial_holdout_weights=eval_weights,
                  injected_phase=injected_phase, recovered_phase_difference=recovered_difference,
                  injected_phase_error=phase_error)
    scanner_recon = torch.einsum('ij,cjw->ciw', inv, scanner)
    display_scale = scanner_recon.abs().square().sum(0).sqrt().quantile(.995)
    for key, y in signals.items():
        arrays['signal_' + key] = y
        image = torch.einsum('ij,cjw->ciw', inv, y).abs().square().sum(0).sqrt()
        arrays['image_' + key] = image / display_scale
    np.savez_compressed(args.out / (name + '.npz'),
                        **{k: v.detach().cpu().numpy() for k, v in arrays.items()})
    torch.save(dict(state_dict=state, seed=args.seed, steps=args.steps,
                    architecture='2-16-16-16-16-16-1 Tanh; 1153 parameters'),
               args.out / (name + '_phase_net.pt'))
    result = dict(name=name, source=str(path), source_sha256=digest(path),
                  signal_scale=float(scale), display_scale=float(display_scale),
                  phase_only_amplitude_error=error, methods=methods, trace=trace,
                  paired_injection=paired)
    (args.out / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(name=name, methods=methods, paired_injection={k: v for k, v in paired.items()
                     if k not in ('trace', 'baseline', 'recovered')})), flush=True)
    return result


def render(out, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(results), 6, figsize=(18, 4.4 * len(results)), squeeze=False)
    names = ['raw', 'scanner', 'legacy_tiny', 'tiny']
    titles = ['Uncorrected + InvA', 'Existing MAT correction + InvA',
              'Existing tiny + InvA', 'Acquired-domain tiny + InvA']
    for row, result in enumerate(results):
        with np.load(out / (result['name'] + '.npz')) as arrays:
            for col, (name, title) in enumerate(zip(names, titles)):
                axes[row, col].imshow(np.rot90(arrays['image_' + name], 2), cmap='gray', vmin=0, vmax=1)
                cost = result['methods'][name]['phase_cost_spatial_holdout']
                axes[row, col].set_title(f'{title}\nHeld-out phase cost: {cost:.3f}', fontsize=9)
            phase_plot = axes[row, 4].imshow(np.rot90(arrays['phase'], 2), cmap='twilight', vmin=-np.pi, vmax=np.pi)
            axes[row, 4].set_title('Fitted acquired-even phase [rad]', fontsize=9)
            fig.colorbar(phase_plot, ax=axes[row, 4], orientation='horizontal', fraction=.045, pad=.06)
            error_plot = axes[row, 5].imshow(np.rot90(arrays['injected_phase_error'], 2), cmap='coolwarm', vmin=-1, vmax=1)
            error = result['paired_injection']['recovered_difference_error_weighted_rms_rad']
            axes[row, 5].set_title(f'Known perturbation recovery error\nWeighted RMS: {error:.3f} rad', fontsize=9)
            fig.colorbar(error_plot, ax=axes[row, 5], orientation='horizontal', fraction=.045, pad=.06)
        axes[row, 0].set_ylabel(result['name'], fontsize=10)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle('Test-time relative odd/even phase pilot | 1,153 parameters per acquisition\n'
                 'Fixed MAT-based display scale; spatial holdout is not independent acquired data; no HR truth', fontsize=12)
    fig.subplots_adjust(left=.035, right=.99, bottom=.05, top=.84, wspace=.15, hspace=.45)
    fig.savefig(out / 'comparison.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scan-root', type=Path, default=PROJECT.parent / 'data/spen_acquired_260915/mat')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=260915)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
    config = dict(script_sha256=digest(__file__), steps=args.steps, seed=args.seed,
                  device=args.device, parameter_count=1153,
                  objective='circular odd/even phase alignment after acquired-line correction + .002 wrapped smoothness + .0001 phase energy',
                  selection='Fixed middle examples, FOV16 export22 and FOV24 export11; no score selection',
                  train_test='Per-acquisition fitting, 20% decoded spatial blocks held out from alignment loss; same raw acquisition enters both',
                  scope='Relative odd/even phase only; no diffusion update or joint object-phase estimation; no anatomical ground truth',
                  legacy='Existing tiny uses all eligible pixels and its original optimizer; descriptive comparator, not a controlled optimizer ablation')
    (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    selections = [('fov16_export22', '20240321_lxj_spen_mouse_240321_1_1_1', 22),
                  ('fov24_export11', '20240115_lxj_SPEN_96_240115_1_1_1', 11)]
    results = [run_case(args.scan_root / scan / f'slice_{number}.mat', name, args)
               for name, scan, number in selections]
    (args.out / 'summary.json').write_text(json.dumps(dict(config=config, cases=results), indent=2) + '\n')
    render(args.out, results)


if __name__ == '__main__':
    main()
