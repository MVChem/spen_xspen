"""Held-out synthetic inverse checks and separate exploratory real xSPEN reconstructions."""
import argparse
import json
import math
from pathlib import Path
import shutil
import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from data import PriorData
from model import load_strong_prior
from operators import XSPENOperator, encoding_matrices, synthetic_coils
from scanner import load_case
from solvers import diffpir
from utils import HERE, sha256, save_grid, validate, write_json
from paths import IXI_DATA, SCANNER_DATA

def structural_similarity(a, b):
    """Gaussian SSIM, sigma 1.5 / 11-pixel window, population covariance, range 1."""
    blur = lambda x: gaussian_filter(x, sigma=1.5, truncate=3.5)
    ma, mb = blur(a), blur(b)
    va, vb = blur(a*a)-ma*ma, blur(b*b)-mb*mb
    cov = blur(a*b)-ma*mb
    score = ((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return float(score[5:-5, 5:-5].mean())

def metrics(pred, truth):
    p = ((pred[:, 0].detach().cpu().numpy()+1)/2).clip(0, 1)
    t = ((truth[:, 0].detach().cpu().numpy()+1)/2).clip(0, 1)
    out = []
    for a, b in zip(p, t):
        mse = np.mean((a-b)**2)
        out.append(dict(psnr=float(-10*np.log10(max(mse, 1e-12))),
                        ssim=structural_similarity(a, b)))
    return out

def select_clean(manifest, split, count, device):
    arr = np.load(IXI_DATA/f'{split}.npy', mmap_mode='r')
    subjects = manifest['subjects'][split]
    choose = np.linspace(0, len(subjects)-1, min(count, len(subjects)), dtype=int)
    records = manifest['records'][split]
    indices = []
    for n, j in enumerate(choose):
        view = 'axial' if n % 2 == 0 else 'sagittal'
        choices = [i for i, r in enumerate(records) if r['subject'] == subjects[j] and r['modality'] == 'T2' and r['view'] == view]
        indices.append(choices[len(choices)//2])
    x = torch.from_numpy(arr[indices].astype(np.float32)/65535.)[:, None].to(device)*2-1
    return x, [records[i]['key'] for i in indices]

def synthetic_observation(x, acceleration, seed):
    a, f = encoding_matrices(60, 64, 128, 128, 60., device=x.device)
    an, _ = encoding_matrices(60, 64, 60, 64, 60., device=x.device)
    a = a/torch.linalg.svdvals(an).max()
    mask = torch.ones(60, 64, device=x.device)
    mask[torch.arange(60, device=x.device) % acceleration != 0] = 0
    op = XSPENOperator(a, f, synthetic_coils(128, 128, device=x.device), mask)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    y = op.forward(x)
    noise = torch.complex(torch.randn(y.shape, generator=gen, device=x.device), torch.randn(y.shape, generator=gen, device=x.device))*.01
    return op, (y+noise)*mask

@torch.no_grad()
def controlled(net, manifest, out, steps, limit):
    val, val_keys = select_clean(manifest, 'val', limit, next(net.parameters()).device)
    test, test_keys = select_clean(manifest, 'test', limit, val.device)
    report = {}
    for accel in [1, 2]:
        op, yv = synthetic_observation(val, accel, 20260909)
        lambda_rows, rho_rows = [], []
        for lamb in [.3, 1., 3.]:
            pred, _ = diffpir(net, op, yv, steps=steps, sigma_noise=.01, lamb=lamb)
            rows = metrics(pred, val)
            lambda_rows.append(dict(value=lamb, mean_psnr=float(np.mean([r['psnr'] for r in rows]))))
        for rho in [.0003, .003, .03]:
            pred = op.proximal(torch.full_like(val, -1.), yv, rho)
            rho_rows.append(dict(value=rho, mean_psnr=float(np.mean([r['psnr'] for r in metrics(pred, val)]))))
        selected_lambda = max(lambda_rows, key=lambda r: r['mean_psnr'])['value']
        selected_rho = max(rho_rows, key=lambda r: r['mean_psnr'])['value']
        selection = dict(lambda_sweep=lambda_rows, rho_sweep=rho_rows, selected_lambda=selected_lambda,
                         selected_rho=selected_rho, validation_keys=val_keys, steps=steps)
        write_json(out/f'R{accel}_selection.json', selection)
        op, yt = synthetic_observation(test, accel, 20260910)
        baseline = op.proximal(torch.full_like(test, -1.), yt, selected_rho)
        result, trace = diffpir(net, op, yt, steps=steps, sigma_noise=.01, lamb=selected_lambda)
        rows = []
        bm, dm = metrics(baseline, test), metrics(result, test)
        residual = op.relative_residual(result, yt).tolist()
        clipped_residual = op.relative_residual(result.clamp(-1, 1), yt).tolist()
        for i, key in enumerate(test_keys):
            rows.append(dict(key=key, tikhonov=bm[i], diffusion=dm[i], residual=residual[i], clipped_residual=clipped_residual[i]))
        write_json(out/f'R{accel}_test.json', dict(cases=rows, trace=trace,
                    note='Matched ideal reduced xSPEN simulation with analytic coils and known phase. Retrospective PE masking, no real acceleration claim.'))
        np.savez_compressed(out/f'R{accel}_arrays.npz', truth=test.cpu().numpy(), measurement=yt.cpu().numpy(),
                            tikhonov=baseline.cpu().numpy(), diffusion=result.cpu().numpy(),
                            encoding=op.a.cpu().numpy(), readout=op.f.cpu().numpy(), coils=op.coils.cpu().numpy(), mask=op.mask.cpu().numpy())
        display = torch.stack([test[:6], baseline[:6], result[:6]], dim=1).flatten(0, 1)
        save_grid(display, out/f'R{accel}_comparison.png', columns=3)
        report[f'R{accel}'] = dict(tikhonov_psnr=float(np.mean([r['psnr'] for r in bm])), diffusion_psnr=float(np.mean([r['psnr'] for r in dm])))
        print(json.dumps(dict(event='synthetic_complete', acceleration=accel, **report[f'R{accel}'])), flush=True)
    return report

def plot_real(images, infos, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    titles = ['RO Fourier + RSS', 'Native complex Tikhonov', 'Magnitude Tikhonov', 'EDM + DiffPIR']
    fig, axes = plt.subplots(len(images), 4, figsize=(10, 2.5*len(images)), squeeze=False)
    for i, (row, info) in enumerate(zip(images, infos)):
        for j, im in enumerate(row):
            axes[i, j].imshow(im, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axes[i, j].set_axis_off()
            axes[i, j].set_title(titles[j] if i == 0 else '', fontsize=9)
        axes[i, 0].text(0, -.08, f"{info['scan']} / slice {info['slice_index']} / repeat {info['repeat']}", transform=axes[i, 0].transAxes, fontsize=8)
    fig.suptitle('Real crossed-chirp xSPEN: exploratory reduced-model reconstruction; no clean ground truth', fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

@torch.no_grad()
def real_scanner(net, out, steps, smoke=False):
    cases, images, labels = [], [], []
    approved = json.loads((SCANNER_DATA/'human_qc.json').read_text())
    selected = [mid for mid, row in approved.items() if row['accepted_human']]
    if not selected:
        raise ValueError('No visually verified human scans')
    for mid in selected:
        path = SCANNER_DATA/f'{mid}.h5'
        with h5py.File(path) as f:
            nrep, ns = f['kspace'].shape[:2]
        slices = [ns//2] if smoke else np.linspace(int(ns*.25), int(ns*.75), 4, dtype=int)
        repeats = [0] if smoke else sorted({0, nrep//2, nrep-1})
        for repeat in repeats:
            for sl in slices:
                op, y, baseline, anchor, raw, info = load_case(path, int(sl), repeat, next(net.parameters()).device)
                result, trace = diffpir(net, op, y, steps=steps, sigma_noise=.02, lamb=1., seed=20260909)
                info.update(diffusion_residual=float(op.relative_residual(result, y)),
                            diffusion_clipped_residual=float(op.relative_residual(result.clamp(-1, 1), y)),
                            lambda_fixed=1., trace=trace, steps=steps)
                name = f'{mid}_rep{repeat:02d}_slice{sl:03d}'
                np.savez_compressed(out/f'{name}.npz', measurement=y.cpu().numpy(),
                                    original_measurement=op.original_observation.cpu().numpy(), phase_correction=op.phase_correction.cpu().numpy(),
                                    baseline=baseline.cpu().numpy(), anchor=anchor.cpu().numpy(), raw_rss=raw.cpu().numpy(),
                                    diffusion=result.cpu().numpy(), encoding=op.a.cpu().numpy(), readout=op.f.cpu().numpy(),
                                    coils=op.coils.cpu().numpy(), pe_projection=op.p.cpu().numpy(), ro_projection=op.q.cpu().numpy())
                write_json(out/f'{name}.json', info)
                cases.append(info)
                images.append([raw[0, 0].cpu().numpy()]+[((v[0, 0]+1)/2).clamp(0, 1).cpu().numpy() for v in [anchor, baseline, result]])
                labels.append(info)
                if len(images) == 6:
                    plot_real(images, labels, out/f'real_page_{len(cases)//6:02d}.png')
                    images, labels = [], []
                print(json.dumps(dict(event='real_case', case=name, baseline_residual=info['baseline_residual'], diffusion_residual=info['diffusion_residual'])), flush=True)
    if images:
        plot_real(images, labels, out/f'real_page_{math.ceil(len(cases)/6):02d}.png')
    write_json(out/'real_cases.json', cases)
    return dict(cases=len(cases), scans=selected, interpretation='No real GT; measurement residual and visual comparison only. Sensitivity to provisional sequence mapping remains unresolved.')

def main():
    global IXI_DATA, SCANNER_DATA
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=IXI_DATA)
    p.add_argument('--scanner', type=Path, default=SCANNER_DATA)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--limit', type=int, default=12)
    p.add_argument('--real-only', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    IXI_DATA, SCANNER_DATA = args.data, args.scanner
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Use a new evaluation output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    manifest = json.loads((IXI_DATA/'manifest.json').read_text())
    snapshot = args.out/'prior_ema.pt'
    shutil.copyfile(args.checkpoint, snapshot)
    net, checkpoint = load_strong_prior(snapshot)
    if checkpoint['manifest_sha256'] != sha256(IXI_DATA/'manifest.json'):
        raise ValueError('Checkpoint and dataset manifest mismatch')
    report = dict(checkpoint_step=checkpoint['step'], checkpoint_sha256=sha256(snapshot),
                  manifest_sha256=checkpoint['manifest_sha256'], exploratory_smoke=args.smoke)
    source = args.out/'source_snapshot'
    source.mkdir()
    for path in HERE.glob('*.py'):
        shutil.copyfile(path, source/path.name)
    report['source_sha256'] = {path.name: sha256(path) for path in source.glob('*.py')}
    if not args.real_only:
        data = PriorData(IXI_DATA)
        val, keys = data.validation()
        test, test_keys = select_clean(manifest, 'test', len(manifest['subjects']['test']), 'cuda')
        report['denoising'] = dict(validation_loss=validate(net, val), test_loss=validate(net, test), test_keys=test_keys)
        report['synthetic'] = controlled(net, manifest, args.out, args.steps, args.limit)
    report['real'] = real_scanner(net, args.out, args.steps, args.smoke)
    write_json(args.out/'summary.json', report)
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
