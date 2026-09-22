"""Small reproducible Hybrid SPEN inverse-problem pilot in the native RO domain.

Human/rodent priors use their training grids; exact voxel overlap projects their
magnitudes to the SAME 60x64 measurement model. Report images and metrics there.
The real data constraint is on the legacy decode/correct/re-encode signal, not
the untouched ADC. Coil/object-phase factors are estimated from that signal.
"""
import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model import load_strong_prior
from operators import NativeGridXSPENOperator, pixel_average_matrix, synthetic_coils
from solvers import diffpir
from utils import sha256, write_json


def rss(x):
    return x.abs().square().sum(0).sqrt()


def traditional(a, inv_a, y):
    """Native complex Tikhonov and windowed-adjoint baseline in physical units."""
    ridge = .01 * torch.linalg.matrix_norm(a, ord=2).square()
    ah = a.mH
    coils = torch.linalg.solve(ah @ a + ridge * torch.eye(a.shape[1], dtype=a.dtype), ah @ y)
    inv = inv_a @ y
    predicted = a @ inv
    gain = (predicted.conj() * y).sum() / predicted.abs().square().sum().clamp_min(1e-20)
    # A single complex receiver gain from measurements, no magnitude/GT fitting.
    inv = inv * gain
    residual = (ah @ (a @ coils - y) + ridge * coils).norm() / (ah @ y).norm()
    assert float(residual) < 1e-9
    return coils, rss(inv), dict(inva_gain=[float(gain.real), float(gain.imag)],
                                tikhonov_normal_residual=float(residual))


def save_case(out, name, matrices, ro, corrected, metadata, gt=None, known_coils=None, phase=None):
    a, inv_a = matrices['a'], matrices['inv_a']
    y = corrected.permute(2, 0, 1).to(torch.complex128)
    native_coils, inva, checks = traditional(a, inv_a, y)
    magnitude = rss(native_coils)
    scale = float(magnitude.quantile(.995)) if gt is None else 1.
    factors = native_coils / magnitude.clamp_min(scale * 1e-4)
    if known_coils is not None:
        factors = known_coils.to(torch.complex128)
    norm = float(torch.linalg.matrix_norm(a, ord=2))
    arrays = dict(a=(a / norm).numpy().astype(np.complex64),
                  y=(y / (norm * scale)).numpy().astype(np.complex64),
                  coils=factors.numpy().astype(np.complex64),
                  tikhonov=(magnitude / scale).numpy().astype(np.float32),
                  inva=(inva / scale).numpy().astype(np.float32),
                  input=rss(ro.permute(2, 0, 1)).numpy().astype(np.float32),
                  ro_original=ro.numpy(), ro_corrected=corrected.numpy())
    if gt is not None:
        arrays['gt'] = gt.numpy().astype(np.float32)
    if phase is not None:
        arrays['phase'] = phase.numpy()
    np.savez_compressed(out / f'{name}.npz', **arrays)
    metadata.update(id=name, scale=scale, a_norm=norm, **checks,
                    shape=[60, 64], sigma_noise_assumption=.02,
                    constraint='phase-corrected RO signal; legacy even-row low-resolution projection',
                    nuisance='known synthetic coil/phase' if known_coils is not None else
                    'native complex coil/object-phase factors from same-measurement Tikhonov')
    write_json(out / f'{name}.json', metadata)
    return metadata


def prepare(args):
    from spenpy.io import read_siemens
    from spenpy.recon import calc_hybrid_spen_matrices, estimate_hybrid_spen_shift, reconstruct_hybrid_spen

    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    selection = json.loads(args.selection.read_text())['hybrid_spen']
    raw = Path(selection['source_raw'])
    acquisition = read_siemens(raw)
    assert acquisition.data.shape == (60, 64, 32, 40, 4)
    assert acquisition.metadata['r_value'] == 60
    assert np.allclose([acquisition.metadata['phase_fov_m'], acquisition.metadata['readout_fov_m']], [.18, .192])
    shift = estimate_hybrid_spen_shift(acquisition)
    matrices = calc_hybrid_spen_matrices(60, .18, 60)
    metadata = dict(raw=str(raw), raw_sha256=sha256(raw), acquisition=acquisition.metadata,
                    shift_pixels=shift, legacy_mat=selection['reconstruction_mat'])
    write_json(out / 'acquisition.json', metadata)
    cases = []
    labels = ['b0', 'DWI-RO', 'DWI-PE', 'DWI-SS']
    with h5py.File(selection['reconstruction_mat']) as reference:
        old = reference['InvA'][()].T
        old = old['real'] + 1j * old['imag']
        inva_error = float(np.linalg.norm(matrices['inv_a'].resolve_conj().numpy() - old) / np.linalg.norm(old))
        assert inva_error < 1e-10
        for sl in args.slices:
            for volume in range(4):
                result = reconstruct_hybrid_spen(acquisition.select(slice=sl, volume=volume),
                                                shift_pixels=shift, matrices=matrices)
                assert result.metadata['phase_fit_valid']
                expected = reference['SmatBeforePhaseMapInvA'][volume, sl].T
                actual = rss(result.ro_image.permute(2, 0, 1)).numpy()
                gain = float(np.vdot(actual, expected) / np.vdot(actual, actual))
                before_error = float(np.linalg.norm(actual * gain - expected) / np.linalg.norm(expected))
                assert before_error < .03
                legacy_after = reference['Smat'][volume, sl].T
                current_after = result.magnitude.numpy()
                after_gain = float(np.vdot(current_after, legacy_after) / np.vdot(current_after, current_after))
                after_error = float(np.linalg.norm(current_after * after_gain - legacy_after) / np.linalg.norm(legacy_after))
                name = f'MID253_s{sl:02d}_v{volume}'
                info = dict(kind='real', slice=sl, volume=volume, label=labels[volume],
                            matlab_inva_relative_error=inva_error, legacy_ro_scaled_nrmse=before_error,
                            legacy_inva_scaled_nrmse=after_error, phase=result.metadata,
                            phase_coefficients=result.coefficients.tolist())
                cases.append(save_case(out, name, matrices, result.ro_image,
                                       result.corrected_ro_image, info, phase=result.phase_map_rad))
                print(json.dumps(dict(event='prepared', **info)), flush=True)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    data = np.load(args.data / 'test.npy', mmap_mode='r')
    records = manifest['records']['test']
    subjects = manifest['subjects']['test'][:4]
    p, q = pixel_average_matrix(60, 128), pixel_average_matrix(64, 128)
    for si, subject in enumerate(subjects):
        candidates = [i for i, r in enumerate(records) if r['subject'] == subject
                      and r['modality'] == 'T2' and r['view'] == 'axial']
        index = candidates[len(candidates)//2]
        im = torch.tensor(data[index].astype(np.float32) / 65535.)[None, None]
        theta = torch.tensor([[[192/210, 0., 0.], [0., 180/210, 0.]]])
        im = F.grid_sample(im, F.affine_grid(theta, im.shape, align_corners=False), align_corners=False)
        gt = (p @ im[0, 0] @ q.T).double()
        coils = synthetic_coils(60, 64, count=4, dtype=torch.complex128)
        clean = matrices['a'] @ (coils * gt)
        generator = torch.Generator().manual_seed(917 + si)
        noise_std = float(clean.abs().square().mean().sqrt()) * .02
        y = clean + torch.randn(clean.shape, dtype=clean.dtype, generator=generator) * noise_std
        ro = y.permute(1, 2, 0)
        info = dict(kind='simulation', label=subject, test_index=index, record=records[index],
                    noise_std_complex=noise_std, noise_rms_relative=.02,
                    simulation='native quadratic A, smooth known coils; zero parity phase; no phase fitting',
                    fov_crop_mm=[180, 192])
        cases.append(save_case(out, f'IXI_{subject}', matrices, ro, ro, info, gt=gt, known_coils=coils))
    write_json(out / 'manifest.json', dict(cases=cases, selection='two prespecified slices and all four volumes; first four held-out subjects, middle axial T2 slice',
               limitation='real DWI has no paired truth; IXI T2/PD is an anatomical magnitude prior, not DWI ground truth',
               source_manifest_sha256=sha256(args.data / 'manifest.json')))


def make_operator(case, size, device):
    a = torch.tensor(case['a'], device=device)
    coils = torch.tensor(case['coils'], device=device)
    identity = torch.eye(coils.shape[-1], dtype=a.dtype, device=device)
    return NativeGridXSPENOperator(a, identity, coils, image_shape=(size, size), cg_iterations=100)


def scores(im, case):
    result = dict(min=float(im.min()), max=float(im.max()))
    if 'gt' in case:
        # Native physical grid, fixed data range, no fitted scale or clipping.
        result.update(psnr=float(peak_signal_noise_ratio(case['gt'], im, data_range=1)),
                      ssim=float(structural_similarity(case['gt'], im, data_range=1)))
    return result


@torch.no_grad()
def evaluate(args):
    device = 'cuda'
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    net, checkpoint = load_strong_prior(args.checkpoint, device)
    net.img_resolution = args.size
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    rows = []
    for info in manifest['cases']:
        case = dict(np.load(args.prepared / f'{info["id"]}.npz'))
        op = make_operator(case, args.size, device)
        y = torch.tensor(case['y'], device=device)[None]
        # Simulation noise is known from its generation, so propagate its units
        # through A/measurement normalization (torch complex noise has E|n|²=1).
        # This is an analytic calibration, with no search against test metrics.
        sigma_noise = (info['noise_std_complex'] / (info['a_norm'] * info['scale'] * 2**.5)
                       if info['kind'] == 'simulation' else .02)
        x, trace = diffpir(net, op, y, steps=args.steps, sigma_noise=sigma_noise, lamb=1., xi=0., seed=917)
        native = op.project((x + 1) / 2)[0, 0].cpu().numpy()
        assert np.isfinite(native).all()
        # Evaluate traditional magnitude baselines with the identical native
        # nuisance factors and corrected measurements used by diffusion.
        native_op = make_operator(case, 64, device)
        from operators import XSPENOperator
        native_op = XSPENOperator(native_op.a, native_op.f, native_op.coils)
        metrics = {}
        for key in ('tikhonov', 'inva'):
            value = torch.tensor(case[key], device=device)[None, None] * 2 - 1
            metrics[key] = dict(**scores(case[key], case),
                                residual=float(native_op.relative_residual(value, y)))
        metrics['diffusion'] = dict(**scores(native, case), residual=float(op.relative_residual(x, y)))
        np.savez_compressed(out / f'{info["id"]}.npz', native=native, prior_grid=((x[0, 0]+1)/2).cpu().numpy())
        row = dict(id=info['id'], kind=info['kind'], sigma_noise=sigma_noise, metrics=metrics, trace=trace)
        write_json(out / f'{info["id"]}.json', row)
        rows.append(row)
        print(json.dumps(dict(event='evaluated', id=info['id'], metrics=metrics)), flush=True)
    config = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
                  checkpoint_step=checkpoint['step'], size=args.size, steps=args.steps,
                  real_sigma_noise=.02, simulation_sigma_noise='known per-component noise after operator/measurement normalization',
                  lamb=1., xi=0., seed=917, rows=rows,
                  prepared_manifest_sha256=sha256(args.prepared/'manifest.json'),
                  model='Hybrid quadratic A in RO domain; finite-volume magnitude projection; native fixed complex coils',
                  interpretation='96/128 prior grid is not evidence of super-resolution; all comparisons are native 60x64')
    write_json(out / 'summary.json', config)
    shutil.copyfile(__file__, out / 'hybrid_pilot_source.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--slices', nargs='+', type=int, default=[19, 26])
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('evaluate')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--size', type=int, choices=[96, 128], required=True)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == 'prepare':
        prepare(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
