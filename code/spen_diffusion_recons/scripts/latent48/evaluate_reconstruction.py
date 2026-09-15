"""Matched simulation and acquired-data reconstruction with a frozen latent EMA.

Small calibration uses only the old four calibration images per condition.
Report panels and all traditional results retain the approved figure cases.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT.parent / 'spenpy'))
sys.path.insert(0, str(HERE.parent / 'core'))
from evaluate import metrics
from dit import LatentEDM
from vae_codec import FrozenVAE
from latent_reconstruction import latent_reconstruct
from reference_cases import load_simulation_reference, load_real_reference


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    pending.replace(path)


def stable_seed(key):
    return int(hashlib.sha256(('latent_spen_preview:' + key).encode()).hexdigest()[:8], 16) % (2**31)


def announce(**value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def load_prior(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    root = Path(ckpt['vae_path'])
    for name, digest in ckpt['vae_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError('VAE differs from the training checkpoint')
    norm = ckpt['latent_normalization']
    if (norm['vae_sha256'] != ckpt['vae_sha256'] or norm['manifest_sha256'] != ckpt['manifest_sha256']
            or norm['vae_autocast'] != 'bfloat16' or norm['posterior'] != 'mode'):
        raise ValueError('Invalid latent codec normalization provenance')
    net = LatentEDM(**ckpt['model_config']).to(device)
    net.load_state_dict(ckpt['ema'])
    net.eval().requires_grad_(False)
    codec = FrozenVAE(root, device=device, autocast_dtype=torch.bfloat16,
                      encode_batch_size=1, decode_batch_size=1)
    mean, std = (torch.tensor(norm[k], dtype=torch.float32, device=device).reshape(1, -1, 1, 1)
                 for k in ('mean', 'std'))
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or bool((std <= 0).any()):
        raise ValueError('Invalid normalization moments')
    metadata = {k: v for k, v in ckpt.items() if k != 'ema'}
    metadata['checkpoint_sha256'] = sha(path)
    del ckpt
    return net, codec, mean, std, metadata


def solve_case(case, model, args, lamb, folder):
    net, codec, mean, std, checkpoint = model
    folder.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    parameters = dict(steps=args.steps, inner_steps=args.inner_steps, lamb=lamb,
        sigma_noise=float(case['sigma_noise']), sigma_max=args.sigma_max,
        sigma_min=.02, prox_lr=args.prox_lr, seed=stable_seed(case['key']), decoder_fp32=True)
    def progress(value):
        if value['step'] % 10 == 0 or value['step'] == args.steps-1:
            save_json(folder/'progress.json',dict(step=value['step'],total_steps=args.steps,
                elapsed_sec=value['elapsed_seconds'],objective_before=value['objective_before'],
                objective_after=value['objective_after'],accepted_steps=value['accepted_steps']))
    raw, trace = latent_reconstruct(net, codec, mean, std, case['op'], case['observation'],
        initial_image=case['initial_image'], progress_callback=progress, **parameters)
    if raw.shape != (1, 1, 192, 192) or not torch.isfinite(raw).all():
        raise FloatingPointError('Invalid reconstructed image')
    with torch.no_grad():
        residual = float(case['op'].relative_residual(raw, case['observation']))
        warm_residual = float(case['op'].relative_residual(case['initial_image'], case['observation']))
    row = dict(key=case['key'], case_index=case['case_index'], parameters=parameters,
        checkpoint_step=checkpoint['step'], elapsed_sec=time.monotonic()-start,
        measurement_nrmse=residual, initialization_measurement_nrmse=warm_residual,
        outside_range_fraction=float(((raw < -1) | (raw > 1)).float().mean()),
        metadata=case.get('metadata', {}))
    payload = dict(prediction_raw=raw[0, 0].detach().cpu().numpy(),
        initial_raw=case['initial_image'][0, 0].detach().cpu().numpy(),
        observation=case['observation'].detach().cpu().numpy())
    if case.get('gt') is not None:
        row['metrics'] = metrics(raw, case['gt'])[0]
        row['initial_metrics'] = metrics(case['initial_image'], case['gt'])[0]
        payload['gt'] = case['gt'][0, 0].detach().cpu().numpy()
    np.savez_compressed(folder / 'arrays.npz', **payload)
    save_json(folder / 'metrics.json', row)
    save_json(folder / 'solver_trace.json', trace)
    announce(event='case_completed', key=case['key'], lamb=lamb,
             seconds=row['elapsed_sec'], measurement_nrmse=residual, metrics=row.get('metrics'))
    return payload['prediction_raw'], row


def simulation(args, model):
    root = args.out / f'R{args.condition + 1}'
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'completed.json').exists():
        raise FileExistsError('Condition is already completed')
    reference = load_simulation_reference(args.reference_dir, args.device)
    if reference['provenance']['training_manifest_sha256'] != model[-1]['manifest_sha256']:
        raise ValueError('Reference images differ from the frozen prior training manifest')
    save_json(root/'reference_provenance.json',reference['provenance'])
    calibration = [c for c in reference['calibration'] if c['condition_index'] == args.condition]
    report = [c for c in reference['report'] if c['condition_index'] == args.condition]
    if len(calibration) != 4 or len(report) != 4 or {c['key'] for c in calibration} & {c['key'] for c in report}:
        raise ValueError('Expected four disjoint calibration / report cases per condition')
    config = dict(stage='simulation', condition=args.condition, checkpoint_step=model[-1]['step'],
        checkpoint_sha256=model[-1]['checkpoint_sha256'], reference_dir=str(args.reference_dir),
        reference_npz_sha256=sha(args.reference_dir/'simulation.npz'),
        algorithm='latent DiffPIR-style approximate nonlinear proximal; measurement-derived warm start',
        calibration_keys=[c['key'] for c in calibration], report_keys=[c['key'] for c in report],
        lambda_candidates=args.lambdas, steps=args.steps, inner_steps=args.inner_steps,
        sigma_max=args.sigma_max, prox_lr=args.prox_lr,
        prior_training_scope='All calibration/report images participated in prior training; no holdout',
        code_sha256={p.name: sha(p) for p in [Path(__file__), HERE/'latent_reconstruction.py', HERE/'reference_cases.py']})
    save_json(root/'config.json', config)
    trials = []
    for lamb in args.lambdas:
        rows = []
        for case in calibration:
            _, row = solve_case(case, model, args, lamb,
                root / 'calibration' / f'lambda_{lamb:g}' / f'case_{case["case_index"]}')
            rows.append(row)
        trials.append(dict(lamb=lamb, mean_psnr=float(np.mean([r['metrics']['psnr'] for r in rows])),
            mean_ssim=float(np.mean([r['metrics']['ssim'] for r in rows])), cases=rows))
        save_json(root/'calibration.json', trials)
    selected = max(trials, key=lambda trial: trial['mean_psnr'])
    selection = dict(lamb=selected['lamb'], calibration_mean_psnr=selected['mean_psnr'],
                     rule='Fixed four calibration cases, source-equal mean PSNR; report cases never tune parameters',
                     steps=args.steps, inner_steps=args.inner_steps, sigma_max=args.sigma_max, prox_lr=args.prox_lr)
    save_json(root/'selected.json', selection)
    predictions, rows = [], []
    for case in report:
        image, row = solve_case(case, model, args, selection['lamb'], root/'report'/f'case_{case["case_index"]}')
        predictions.append(image); rows.append(row)
    np.savez_compressed(root/'predictions.npz', prediction_raw=np.stack(predictions),
                        case_keys=np.asarray([c['key'] for c in report]))
    save_json(root/'report.json', rows)
    save_json(root/'completed.json', dict(completed=True, checkpoint_step=model[-1]['step'],
        report_cases=4, calibration_cases=4, selected=selection,
        mean_psnr=float(np.mean([r['metrics']['psnr'] for r in rows])),
        mean_ssim=float(np.mean([r['metrics']['ssim'] for r in rows]))))


def real(args, model):
    root = args.out / 'real'
    root.mkdir(parents=True, exist_ok=True)
    chosen = json.loads((args.out/'R1/selected.json').read_text())
    for key in ('steps', 'inner_steps', 'sigma_max', 'prox_lr'):
        if chosen[key] != getattr(args, key):
            raise ValueError('Real-data solver settings must match simulation calibration')
    reference = load_real_reference(args.reference_dir, args.device)
    save_json(root/f'reference_provenance_{"all" if args.case_indices is None else "_".join(map(str,args.case_indices))}.json',
              reference['provenance'])
    all_cases = reference['cases']
    cases = all_cases if args.case_indices is None else [all_cases[i] for i in args.case_indices]
    save_json(root/f'config_{"all" if args.case_indices is None else "_".join(map(str,args.case_indices))}.json',
        dict(checkpoint_step=model[-1]['step'], checkpoint_sha256=model[-1]['checkpoint_sha256'],
             parameters_from='R1 calibration only; real noise sigma fixed to previous .02', selected=chosen,
             no_ground_truth=True, case_indices=[c['case_index'] for c in cases],
             reference_npz_sha256=sha(args.reference_dir/'real.npz')))
    for case in cases:
        folder = root / f'case_{case["case_index"]:02d}'
        if (folder/'metrics.json').exists():
            raise FileExistsError(f'Case output exists: {folder}')
        solve_case(case, model, args, chosen['lamb'], folder)
    announce(event='real_subset_completed', cases=[c['case_index'] for c in cases])


def assemble(args):
    old_sim = np.load(args.reference_dir/'simulation.npz', allow_pickle=False)
    old_real = np.load(args.reference_dir/'real.npz', allow_pickle=False)
    step = json.loads((args.out/'checkpoint_metadata.json').read_text())['step']
    result = dict(checkpoint_step=np.asarray(step), value_range=np.asarray('display_0_1'),
        real_orientation=np.asarray('reference_display_rot180'),
        simulation_case_keys=old_sim['case_keys'], real_labels=old_real['labels'], real_fov_mm=old_real['fov_mm'])
    summary = dict(checkpoint_step=step, checkpoint_sha256=sha(args.out/'checkpoint.pt'),
        simulation={}, real=[], limitations=[
            'Intermediate checkpoint, not the final 60000-step model.',
            'Latent nonlinear proximal and initialization differ from the pixel DiffPIR reference.',
            'All simulation cases are in prior training; real data have no paired ground truth.'])
    for condition in (0, 1):
        name = f'R{condition+1}'
        folder = args.out/name
        done = json.loads((folder/'completed.json').read_text())
        if done['checkpoint_step'] != step:
            raise ValueError('Mixed checkpoint steps')
        current = np.load(folder/'predictions.npz', allow_pickle=False)
        if not np.array_equal(current['case_keys'], old_sim['case_keys']):
            raise ValueError('Simulation report case order changed')
        raw = current['prediction_raw']
        result[f'simulation_diffusion_{name}'] = ((raw+1)/2).clip(0,1)
        result[f'simulation_diffusion_raw_{name}'] = raw
        old = torch.from_numpy(old_sim['diffusion'][condition])[:,None]*2-1
        target = torch.from_numpy(old_sim['target'])[:,None]*2-1
        old_scores = metrics(old, target)
        summary['simulation'][name] = dict(new=done, old_pixel_prior_mean={
            k:float(np.mean([r[k] for r in old_scores])) for k in ('psnr','ssim')},
            cases=json.loads((folder/'report.json').read_text()))
    real_images = []
    for i in range(10):
        folder=args.out/'real'/f'case_{i:02d}'
        row=json.loads((folder/'metrics.json').read_text())
        if row['checkpoint_step'] != step or row['case_index'] != i:
            raise ValueError('Real case order/checkpoint changed')
        raw=np.load(folder/'arrays.npz')['prediction_raw']
        real_images.append(np.rot90(((raw+1)/2).clip(0,1),2))
        summary['real'].append(row)
    result['real_diffusion']=np.stack(real_images)
    if not all(np.isfinite(a).all() for a in result.values() if a.dtype.kind not in 'US'):
        raise ValueError('Nonfinite display data')
    np.savez_compressed(args.out/'reconstruction_results.npz',**result)
    save_json(args.out/'summary.json',summary)
    save_json(args.out/'real/completed.json',dict(completed=True,cases=10,checkpoint_step=step))
    announce(event='assembled',out=str(args.out))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['simulation','real','assemble'],required=True)
    p.add_argument('--condition',type=int,choices=[0,1])
    p.add_argument('--reference-dir',type=Path,default=PROJECT/'runs/rodent192_spen2x_260914/figures_260915')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--inner-steps',type=int,default=8)
    p.add_argument('--lambdas',type=float,nargs='+',default=[.1,1.,10.])
    p.add_argument('--sigma-max',type=float,default=1.)
    p.add_argument('--prox-lr',type=float,default=.1)
    p.add_argument('--case-indices',type=int,nargs='+')
    args=p.parse_args()
    args.out=args.out.resolve();args.reference_dir=args.reference_dir.resolve()
    args.checkpoint=(args.checkpoint or args.out/'checkpoint.pt').resolve()
    if args.stage=='assemble':
        assemble(args);return
    if args.stage=='simulation' and args.condition is None:
        p.error('Simulation requires --condition 0 or 1')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    model=load_prior(args.checkpoint,args.device)
    if args.stage=='simulation':simulation(args,model)
    else:real(args,model)


if __name__=='__main__':main()
