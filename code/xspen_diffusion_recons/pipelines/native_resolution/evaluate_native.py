"""Matched protocol simulations and measured xSPEN comparisons for one prior/grid."""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from paths import SCANNER_DATA
from evaluate import metrics
from model import load_strong_prior
from native_scanner import load_case
from operators import NativeGridXSPENOperator, encoding_matrices, synthetic_coils
from solvers import diffpir
from utils import sha256, write_json


class Resampled128Prior(nn.Module):
    """Explicit resolution-transfer baseline; interpolated noise is correlated.

    This adapts the existing 210-mm/128-square model to the current FOV/grid.
    It is not a model trained or independently validated at this new resolution.
    """
    def __init__(self, prior):
        super().__init__()
        self.prior = prior

    def forward(self, x, sigma):
        shape = x.shape[-2:]
        large = F.interpolate(x, (128, 128), mode='bilinear', align_corners=False, antialias=True)
        denoised = self.prior(large, sigma)
        return F.interpolate(denoised, shape, mode='bilinear', align_corners=False, antialias=True)


def resolve_geometry(manifest, data_path):
    protocols = json.loads((HERE / 'protocols.json').read_text())['profiles']
    profile_value = manifest.get('profile', manifest.get('protocol', manifest.get('profile_id')))
    if isinstance(profile_value, dict):
        profile_id = profile_value.get('id', profile_value.get('profile_id'))
    else:
        profile_id = profile_value
    grid_value = manifest.get('grid', manifest.get('grid_id'))
    grid_id = grid_value.get('id') if isinstance(grid_value, dict) else grid_value
    shape = manifest.get('image_shape', manifest.get('shape'))
    if shape is None and isinstance(grid_value, dict):
        shape = grid_value.get('shape')
    names = set(Path(data_path).parts) | set(Path(data_path).name.split('_'))
    if profile_id is None:
        found = [p['id'] for p in protocols if p['id'] in names or p['id'] in str(data_path)]
        if len(found) == 1:
            profile_id = found[0]
    selected = [p for p in protocols if p['id'] == profile_id]
    if len(selected) != 1:
        raise ValueError(f'Cannot uniquely resolve profile from manifest/data path: {profile_id}')
    profile = selected[0]
    if grid_id is None:
        found = [g for g in profile['grids'] if (shape is not None and tuple(g['shape']) == tuple(shape)) or g['id'] in names]
    else:
        found = [g for g in profile['grids'] if g['id'] == grid_id]
    if len(found) != 1:
        raise ValueError('Cannot uniquely resolve image grid from manifest')
    grid = found[0]
    if shape is not None and tuple(shape) != tuple(grid['shape']):
        raise ValueError('Manifest and protocol grid disagree')
    return profile, grid


def select_clean(data_path, manifest, split, count, device):
    """One T2 view per distinct held-out subject, deterministically selected."""
    subjects = manifest['subjects'][split]
    records = manifest['records'][split]
    array = np.load(Path(data_path) / f'{split}.npy', mmap_mode='r')
    chosen_subjects = np.linspace(0, len(subjects) - 1, min(count, len(subjects)), dtype=int)
    indices = []
    for n, j in enumerate(chosen_subjects):
        view = 'axial' if n % 2 == 0 else 'sagittal'
        options = [i for i, r in enumerate(records) if r['subject'] == subjects[j] and r['modality'] == 'T2' and r['view'] == view]
        if not options:
            options = [i for i, r in enumerate(records) if r['subject'] == subjects[j] and r['modality'] == 'T2']
        if not options:
            raise ValueError(f'No T2 image for held-out subject {subjects[j]}')
        indices.append(options[len(options) // 2])
    if not indices:
        raise ValueError(f'No {split} cases')
    raw = array[indices]
    x = raw.astype(np.float32)
    if raw.dtype == np.uint16:
        x /= 65535.
    elif raw.dtype == np.uint8:
        x /= 255.
    if x.min() < 0 or x.max() > 1.00001:
        raise ValueError('Expected native dataset magnitude in [0,1] or integer quantization')
    if x.ndim == 3:
        x = x[:, None]
    return torch.as_tensor(x, device=device) * 2 - 1, [records[i]['key'] for i in indices]


def synthetic_observation(x, profile, acceleration=1, seed=20260913):
    """Same acquired grid, finite-volume basis and coil count as measured data."""
    m, k = profile['native_shape']
    a, f = encoding_matrices(m, k, m, k, profile['r_value'], profile['beta'], device=x.device)
    a = a / torch.linalg.svdvals(a).max()
    mask = torch.ones(m, k, device=x.device)
    mask[torch.arange(m, device=x.device) % acceleration != 0] = 0
    coils = synthetic_coils(m, k, count=32, device=x.device)
    op = NativeGridXSPENOperator(a, f, coils, image_shape=x.shape[-2:], mask=mask)
    clean_y = op.forward(x)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    real = torch.randn(clean_y.shape, generator=generator, device=x.device)
    imag = torch.randn(clean_y.shape, generator=generator, device=x.device)
    y = (clean_y + torch.complex(real, imag) * .01) * mask
    return op, y


def degradation_image(op, y):
    """RO inverse + RSS with explicit half-PE-pixel focus/display alignment."""
    magnitude = (y @ op.f.conj()).abs().square().sum(1, keepdim=True).sqrt()
    h, w = op.image_shape
    m = op.a.shape[0]
    yy = (torch.arange(h, device=y.device, dtype=torch.float32) + .5) * 2 / h - 1 + 1 / m
    xx = (torch.arange(w, device=y.device, dtype=torch.float32) + .5) * 2 / w - 1
    gy, gx = torch.meshgrid(yy, xx, indexing='ij')
    grid = torch.stack([gx, gy], dim=-1)[None].expand(len(y), -1, -1, -1)
    return F.grid_sample(magnitude, grid, align_corners=False) * 2 - 1


def plot_rows(images, titles, labels, path, metric_rows=None, real=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nrow, ncol = len(images), len(titles)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 2.7), squeeze=False)
    for i, row in enumerate(images):
        for j, (name, array) in enumerate(row.items()):
            ax = axes[i, j]
            ax.imshow(array, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_axis_off()
            if i == 0:
                ax.set_title(titles[j], fontsize=9)
            if metric_rows is not None and name in metric_rows[i]:
                value = metric_rows[i][name]
                ax.text(.02, .98, f"PSNR {value['psnr']:.2f} dB\nSSIM {value['ssim']:.3f}",
                        transform=ax.transAxes, va='top', color='white', fontsize=8,
                        bbox=dict(facecolor='black', alpha=.7, edgecolor='none', pad=2))
        axes[i, 0].text(0, -.06, labels[i], transform=axes[i, 0].transAxes, fontsize=7)
    if real:
        fig.suptitle('Measured xSPEN; shared intensity window; no clean ground truth', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def arrays_for_display(outputs, index):
    return {name: ((x[index, 0].detach().float().cpu().numpy() + 1) / 2).clip(0, 1) for name, x in outputs.items()}


def summarize_metrics(rows, names):
    return {name: {metric: float(np.mean([r[name][metric] for r in rows])) for metric in ('psnr', 'ssim')}
            for name in names}


@torch.no_grad()
def controlled(net, old, data_path, manifest, profile, out, steps=40, limit=6, smoke=False):
    device = next(net.parameters()).device
    val, val_keys = select_clean(data_path, manifest, 'val', limit, device)
    test, test_keys = select_clean(data_path, manifest, 'test', limit, device)
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if set(manifest['subjects'][a]) & set(manifest['subjects'][b]):
            raise ValueError('Subject split leakage')
    summary = {}
    for acceleration in ([1] if smoke else [1, 2]):
        selection = dict(validation_keys=val_keys, candidate_lambdas=[.3, 1., 3.], candidate_rhos=[.0003, .003, .03])
        op, y = synthetic_observation(val, profile, acceleration, 20260913)
        selected = {}
        for name, prior in [('diffusion', net), ('baseline128', old)]:
            sweep = []
            for lamb in ([1.] if smoke else selection['candidate_lambdas']):
                pred, _ = diffpir(prior, op, y, steps=steps, sigma_noise=.01, lamb=lamb)
                psnr = float(np.mean([r['psnr'] for r in metrics(pred, val)]))
                sweep.append(dict(value=lamb, mean_psnr=psnr))
            selected[name] = max(sweep, key=lambda r: r['mean_psnr'])['value']
            selection[f'{name}_lambda_sweep'] = sweep
        rho_sweep = []
        for rho in ([.003] if smoke else selection['candidate_rhos']):
            pred = op.proximal(torch.full_like(val, -1), y, rho)
            rho_sweep.append(dict(value=rho, mean_psnr=float(np.mean([r['psnr'] for r in metrics(pred, val)]))))
        selected['tikhonov_rho'] = max(rho_sweep, key=lambda r: r['mean_psnr'])['value']
        selection.update(selected=selected, smoke_fixed_parameters=smoke)
        write_json(out / f'R{acceleration}_selection.json', selection)
        op, y = synthetic_observation(test, profile, acceleration, 20260914)
        outputs = dict(truth=test, degraded=degradation_image(op, y),
                       tikhonov=op.proximal(torch.full_like(test, -1), y, selected['tikhonov_rho']))
        traces = {}
        for name, prior in [('diffusion', net), ('baseline128', old)]:
            outputs[name], traces[name] = diffpir(prior, op, y, steps=steps, sigma_noise=.01, lamb=selected[name], seed=20260913)
        scores = {name: metrics(x, test) for name, x in outputs.items() if name != 'truth'}
        residuals = {name: op.relative_residual(x, y).tolist() for name, x in outputs.items() if name != 'truth'}
        clipped = {name: op.relative_residual(x.clamp(-1, 1), y).tolist() for name, x in outputs.items() if name != 'truth'}
        rows = [dict(key=key, **{name: values[i] for name, values in scores.items()},
                     measurement_residual={name: values[i] for name, values in residuals.items()},
                     clipped_measurement_residual={name: values[i] for name, values in clipped.items()}) for i, key in enumerate(test_keys)]
        report = dict(acceleration=acceleration, cases=rows, averages=summarize_metrics(rows, scores), traces=traces,
                      simulation='Matched finite-volume native-coil reduced model; 32 analytic coils; known phase; complex noise std=.01 per real/imag component. Retrospective parity masking for R2.',
                      intensity='Identical simulator units and [0,1] clipping for image metrics; no GT-derived fitted gain.',
                      limitations='Matched simulation checks inversion under assumed physics, not measured waveform/B0 validity. Baseline128 interpolation changes noise covariance and physical training scale.')
        write_json(out / f'R{acceleration}_metrics.json', report)
        np.savez_compressed(out / f'R{acceleration}_arrays.npz', **{name: x.cpu().numpy() for name, x in outputs.items()},
                            measurement=y.cpu().numpy(), encoding=op.a.cpu().numpy(), readout=op.f.cpu().numpy(),
                            coils=op.coils.cpu().numpy(), pe_projection=op.p.cpu().numpy(), ro_projection=op.q.cpu().numpy(), mask=op.mask.cpu().numpy())
        for start in range(0, len(test_keys), 4):
            indices = range(start, min(start + 4, len(test_keys)))
            plot_rows([arrays_for_display(outputs, i) for i in indices],
                      ['GT', 'RO Fourier + RSS', 'Tikhonov', 'Native-grid EDM + DiffPIR', 'Resampled 128 prior'],
                      test_keys[start:start + 4], out / f'R{acceleration}_page_{start // 4 + 1:02d}.png', rows[start:start + 4])
        summary[f'R{acceleration}'] = report['averages']
        print(json.dumps(dict(event='synthetic_complete', acceleration=acceleration, averages=report['averages'])), flush=True)
    return summary


@torch.no_grad()
def real_scanner(net, old, profile, shape, out, steps=40, smoke=False):
    cases, page_images, page_labels = [], [], []
    device = next(net.parameters()).device
    approved = json.loads((SCANNER_DATA / 'human_qc.json').read_text())
    for scan in profile['scan_ids']:
        if not approved.get(scan, {}).get('accepted_human', False):
            raise ValueError(f'Scan lacks existing human QC approval: {scan}')
        path = SCANNER_DATA / f'{scan}.h5'
        with h5py.File(path) as source:
            nrep, nslice = source['kspace'].shape[:2]
        slices = [nslice // 2] if smoke else np.linspace(int(nslice * .25), int(nslice * .75), 4, dtype=int)
        repeats = [0] if smoke else sorted({0, nrep // 2, nrep - 1})
        for repeat in repeats:
            for sl in slices:
                op, y, baseline, anchor, raw, info = load_case(path, int(sl), repeat, device, image_shape=shape)
                if list(op.a.shape) != [profile['native_shape'][0]] * 2 or y.shape[-1] != profile['native_shape'][1]:
                    raise ValueError('Measured dimensions do not match selected profile')
                if abs(info['r_value'] - profile['r_value']) > 1e-6 or abs(info['beta'] - profile['beta']) > 1e-6:
                    raise ValueError('Measured encoding parameters do not match selected profile')
                outputs = dict(degraded=raw * 2 - 1, native_tikhonov=anchor, tikhonov=baseline)
                traces = {}
                for name, prior in [('diffusion', net), ('baseline128', old)]:
                    outputs[name], traces[name] = diffpir(prior, op, y, steps=steps, sigma_noise=.02, lamb=1., seed=20260913)
                info.update(measurement_residual={name: float(op.relative_residual(x, y)) for name, x in outputs.items()},
                            clipped_measurement_residual={name: float(op.relative_residual(x.clamp(-1, 1), y)) for name, x in outputs.items()},
                            sigma_noise=.02, lambda_fixed=1., traces=traces, steps=steps,
                            metric_note='No real GT: no real PSNR/SSIM. Residual quantifies fitting of this fixed nuisance model only.')
                name = f'{scan}_rep{repeat:02d}_slice{sl:03d}'
                write_json(out / f'{name}.json', info)
                np.savez_compressed(out / f'{name}.npz', **{key: x.cpu().numpy() for key, x in outputs.items()},
                                    measurement=y.cpu().numpy(), original_measurement=op.original_observation.cpu().numpy(),
                                    phase_correction=op.phase_correction.cpu().numpy(), encoding=op.a.cpu().numpy(),
                                    readout=op.f.cpu().numpy(), coils=op.coils.cpu().numpy(),
                                    pe_projection=op.p.cpu().numpy(), ro_projection=op.q.cpu().numpy())
                cases.append(info)
                page_images.append(arrays_for_display(outputs, 0)); page_labels.append(name)
                if len(page_images) == 4:
                    plot_rows(page_images, ['RO Fourier + RSS', 'Native complex Tikhonov', 'Magnitude Tikhonov', 'Native-grid EDM + DiffPIR', 'Resampled 128 prior'],
                              page_labels, out / f'real_page_{math.ceil(len(cases) / 4):02d}.png', real=True)
                    page_images, page_labels = [], []
                print(json.dumps(dict(event='real_case', case=name, residuals=info['measurement_residual'])), flush=True)
    if page_images:
        plot_rows(page_images, ['RO Fourier + RSS', 'Native complex Tikhonov', 'Magnitude Tikhonov', 'Native-grid EDM + DiffPIR', 'Resampled 128 prior'],
                  page_labels, out / f'real_page_{math.ceil(len(cases) / 4):02d}.png', real=True)
    write_json(out / 'real_cases.json', cases)
    return dict(cases=len(cases), scans=profile['scan_ids'], mean_residual={name: float(np.mean([case['measurement_residual'][name] for case in cases])) for name in outputs},
                limitations='Scan/repeat/slice counts are not independent subjects. No real clean GT or achieved-resolution claim.')


def main():
    global SCANNER_DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scanner', type=Path, default=SCANNER_DATA)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--limit', type=int, default=6)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--baseline', type=Path, default=PROJECT / 'runs/human128/model_ema.pt')
    parser.add_argument('--synthetic-only', action='store_true')
    parser.add_argument('--real-only', action='store_true')
    args = parser.parse_args()
    SCANNER_DATA = args.scanner
    if args.steps < 2 or args.limit < 1 or (args.synthetic_only and args.real_only):
        raise ValueError('Require steps >=2, limit >=1 and at most one only-mode')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Use a new evaluation output directory')
    torch.set_num_threads(2)
    from native_model import load_native_prior
    manifest_path = args.data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    profile, grid = resolve_geometry(manifest, args.data)
    net, checkpoint = load_native_prior(args.checkpoint, args.device)
    expected_hash = checkpoint.get('manifest_sha256')
    if expected_hash is not None and expected_hash != sha256(manifest_path):
        raise ValueError('Checkpoint/data manifest mismatch')
    if hasattr(net, 'image_shape') and tuple(net.image_shape) != tuple(grid['shape']):
        raise ValueError('Checkpoint/output-grid mismatch')
    old_net, old_checkpoint = load_strong_prior(args.baseline, args.device)
    old = Resampled128Prior(old_net).eval().requires_grad_(False)
    args.out.mkdir(parents=True, exist_ok=True)
    source_dir = args.out / 'source_snapshot'
    source_dir.mkdir()
    for path in list(HERE.glob('*.py')) + [PROJECT / n for n in ['operators.py', 'solvers.py', 'scanner.py', 'edm.py', 'model.py', 'tiny_unet.py', 'evaluate.py', 'utils.py', 'paths.py']]:
        shutil.copyfile(path, source_dir / path.name)
    shutil.copyfile(HERE / 'protocols.json', args.out / 'protocols.json')
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = dict(config=config, profile=profile, grid=grid, checkpoint_step=checkpoint.get('step'),
                  checkpoint_note=checkpoint.get('note'), exploratory_smoke=args.smoke,
                  checkpoint_sha256=sha256(args.checkpoint), manifest_sha256=sha256(manifest_path),
                  baseline_checkpoint_sha256=sha256(args.baseline), baseline_step=old_checkpoint['step'],
                  source_sha256={p.name: sha256(p) for p in source_dir.glob('*.py')},
                  baseline_limitations='128x128 IXI prior trained at 210x210 mm; bilinear image transfer at current FOV changes physical scale and noise covariance; this is an explicit transfer baseline, not retraining.',
                  interpretation='Four output grids are not four acquired resolutions. Real physics calibration is provisional.')
    write_json(args.out / 'config.json', report)
    if not args.real_only:
        report['synthetic'] = controlled(net, old, args.data, manifest, profile, args.out, args.steps, min(args.limit, 2) if args.smoke else args.limit, args.smoke)
    if not args.synthetic_only:
        report['real'] = real_scanner(net, old, profile, grid['shape'], args.out, args.steps, args.smoke)
    write_json(args.out / 'summary.json', report)
    write_json(args.out / 'completed.json', dict(checkpoint_step=checkpoint.get('step'), summary=str(args.out / 'summary.json')))
    print(json.dumps(dict(event='evaluation_complete', out=str(args.out))), flush=True)


if __name__ == '__main__':
    main()
