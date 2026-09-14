"""Controlled 96x96 SPEN observations -> 192x192 reconstruction pilot.

All inverse parameters are selected on the existing validation animals. Source
MRI is resampled directly at 192; the previous 96-pixel arrays are never GT.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / 'prior96'
sys.path.insert(0, str(HERE.parent / 'core'))
sys.path.insert(0, str(V2))
from project_paths import CORE, RUNS, PRIOR96_DATA, PRIOR192_DATA, MOUSE_RAW
from model_v2 import load_strong_prior
from solvers import diffpir
from evaluate import metrics, unit
from operators import scanner_matrices
sys.path.insert(0, str(HERE))
from sr_operator import make_sr_operator, make_low_resolution_operator, SCANS

LABELS = {
    'inva96_up': 'InvA 96 + bicubic',
    'tikh96_up': 'Tikhonov 96 + bicubic',
    'diff96_up': 'Diffusion 96 + bicubic',
    'tikh192': 'Tikhonov 192',
    'diff192_old': 'Diffusion 192 / old prior',
    'diff192_hr': 'Diffusion 192 / HR fine-tune',
}


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_cases(data, part, fov, device, per_subject=2):
    manifest = json.loads((data / 'manifest.json').read_text())
    rows = manifest['records'][part]
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if r['dataset'] == 'ds005236' and float(r['fov_mm']) == fov:
            groups[r['subject']].append(i)
    selected = []
    # Match the old figure's slice 3 and slice 9, uniformly for every animal.
    for subject, ids in sorted(groups.items()):
        for position in (3, 9)[:per_subject]:
            selected.append(min(ids, key=lambda i: abs(rows[i]['slice_index'] - position)))
    if len(set(selected)) != len(selected) or not selected:
        raise ValueError('Missing or duplicate evaluation cases')
    array = np.load(data / f'{part}.npy', mmap_mode='r')
    x = torch.as_tensor(array[selected].astype(np.float32) / 65535., device=device)[:, None]
    assert tuple(x.shape[-2:]) == (192, 192)
    return x * 2 - 1, [dict(rows[i],array_index=i,display_rot180=False) for i in selected]


def upsample(x):
    return F.interpolate(x, size=(192, 192), mode='bicubic', align_corners=False)


def observe(op, x, seed, noise):
    generator = torch.Generator(device=x.device).manual_seed(seed)
    y = op.forward(x)
    return y + noise * (torch.randn(y.shape, device=x.device, generator=generator)
                        + 1j * torch.randn(y.shape, device=x.device, generator=generator))


def quality(pred, target):
    rows = metrics(pred, target)
    for row, p, g in zip(rows, unit(pred), unit(target)):
        mask = g > .05
        row['foreground_psnr'] = float(-10 * np.log10(max(float(np.mean((p[mask]-g[mask])**2)), 1e-12)))
        hp, hg = p-gaussian_filter(p, 1), g-gaussian_filter(g, 1)
        row['highpass_nrmse'] = float(np.linalg.norm(hp-hg) / max(float(np.linalg.norm(hg)), 1e-12))
    return rows


def subject_mean(rows, records):
    grouped = defaultdict(list)
    for row, r in zip(rows, records):
        grouped[r['subject']].append(row)
    return {key: float(np.mean([np.mean([r[key] for r in group]) for group in grouped.values()]))
            for key in rows[0]}


@torch.no_grad()
def reconstruct(method, param, op192, op96, y, net, args, fov):
    out, traces = [], []
    for start in range(0, len(y), args.batch):
        obs = y[start:start+args.batch]
        if method == 'inva96_up':
            inv, a, _ = scanner_matrices(SCANS[fov] / 'slice_7.mat', args.device)
            # InvA is a weighted adjoint. Fit its global receiver gains against
            # acquired measurements, with no target-dependent intensity fit.
            a = a / torch.linalg.svdvals(a).max()
            z = torch.einsum('hm,bcmw->bchw', inv, obs)
            projected = torch.einsum('mh,bchw->bcmw', a, z)
            gain = (projected.conj()*obs).sum((-2,-1)) / projected.abs().square().sum((-2,-1)).clamp_min(1e-12)
            mag = (z*gain[:,:,None,None]).abs().square().sum(1,keepdim=True).sqrt()
            pred = upsample(2*mag-1)
        elif method.startswith('tikh'):
            op = op96 if method == 'tikh96_up' else op192
            z = torch.full((len(obs),1,*op.coils.shape[-2:]), -1., device=args.device)
            pred = op.proximal(z, obs, param)
            if method == 'tikh96_up':
                pred = upsample(pred)
        else:
            op = op96 if method == 'diff96_up' else op192
            pred, trace = diffpir(net, op, obs, steps=args.steps, sigma_noise=args.noise,
                                 lamb=param, seed=2718+start, sigma_max=args.sigma_max,
                                 sigma_min=.02)
            traces.append(dict(batch_start=start,trace=trace))
            if method == 'diff96_up':
                pred = upsample(pred)
        if not torch.isfinite(pred).all():
            raise FloatingPointError(method)
        out.append(pred)
    return torch.cat(out), traces


def evaluate_fov(args, fov, old, hr):
    folder = args.out / f'fov{fov}'
    folder.mkdir(parents=True, exist_ok=True)
    val, vr = load_cases(args.data, 'val', fov, args.device, per_subject=args.val_slices)
    test, tr = load_cases(args.data, 'test', fov, args.device)
    op192 = make_sr_operator(fov=fov,image_size=192,measurement_size=96,noise=args.noise,
                             device=args.device,seed=4527+fov)
    op96 = make_low_resolution_operator(fov=fov,noise=args.noise,
                                        device=args.device,seed=4527+fov)
    yv = observe(op192,val,9100+fov,args.noise)
    yt = observe(op192,test,9200+fov,args.noise)
    assert tuple(yt.shape[1:]) == (4,96,96)
    selected_path = folder/'selection.json'
    selected = json.loads(selected_path.read_text()) if selected_path.exists() else {}
    result_path = folder/'metrics.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else dict(records=tr,methods={})
    npz = folder/'reconstructions.npz'
    arrays = dict(np.load(npz)) if npz.exists() else dict(target=unit(test),observation=yt.cpu().numpy())
    if not np.allclose(arrays['target'],unit(test),atol=0,rtol=0) or not np.allclose(arrays['observation'],yt.cpu().numpy(),atol=1e-6):
        raise ValueError('Existing evaluation differs from current cases/operator')
    methods = list(LABELS) if args.stage=='all' else (['diff192_hr'] if args.stage=='final' else list(LABELS)[:-1])
    config = dict(fov_mm=fov,observation_shape=list(yt.shape),target_shape=list(test.shape),
                  scale_per_axis=2,acquired_pe_rows=96,readout_samples=96,
                  known_coil_and_object_phase=True,noise_per_real_imag_component=args.noise,
                  source='original ds005236 NIfTI; never upsampled 96-pixel arrays',
                  validation_records=vr,test_records=tr,steps=args.steps,sigma_max=args.sigma_max,
                  cg='real-domain conjugate-gradient quadratic proximal',seed=4527+fov)
    if hasattr(op192,'metadata'): config['operator_metadata']=op192.metadata
    save_json(folder/'config.json',config)
    for method in methods:
        candidates = [None] if method=='inva96_up' else ([.0001,.0003,.001,.003,.01] if method.startswith('tikh') else args.lambdas)
        previous_trials=selected.get(method,{}).get('trials',[])
        missing=[v for v in candidates if v not in [r['param'] for r in previous_trials]]
        if method in result['methods'] and not missing:
            print(json.dumps(dict(event='skip_completed',fov=fov,method=method)),flush=True)
            continue
        net = hr if method == 'diff192_hr' else old
        if method=='diff192_hr' and net is None: raise ValueError('HR model missing')
        start = time.monotonic()
        if missing:
            # Expand only the VALIDATION grid. Preserve the initial pilot before
            # replacing any test reconstruction selected by the wider grid.
            if previous_trials and not (folder/'initial_grid_selection.json').exists():
                shutil.copyfile(selected_path,folder/'initial_grid_selection.json')
                shutil.copyfile(result_path,folder/'initial_grid_metrics.json')
                shutil.copyfile(npz,folder/'initial_grid_reconstructions.npz')
            trials=list(previous_trials)
            for param in missing:
                pred,_=reconstruct(method,param,op192,op96,yv,net,args,fov)
                row=dict(param=param,**subject_mean(quality(pred,val),vr))
                trials.append(row)
                print(json.dumps(dict(event='validation',fov=fov,method=method,**row)),flush=True)
            selected[method]=dict(best=max(trials,key=lambda r:r['psnr']),trials=trials,
                                  criterion='validation subject-mean PSNR only')
            save_json(selected_path,selected)
        if method in result['methods'] and result['methods'][method]['selected']['param']==selected[method]['best']['param']:
            print(json.dumps(dict(event='expanded_validation_same_selection',fov=fov,method=method)),flush=True)
            continue
        pred,traces=reconstruct(method,selected[method]['best']['param'],op192,op96,yt,net,args,fov)
        rows=quality(pred,test)
        residual=op192.relative_residual(pred,yt).cpu().tolist()
        clip_residual=op192.relative_residual(pred.clamp(-1,1),yt).cpu().tolist()
        for row, r, rc, p in zip(rows,residual,clip_residual,pred):
            row['measurement_nrmse']=r
            row['displayed_measurement_nrmse']=rc
            row['outside_range_fraction']=float(((p<-1)|(p>1)).float().mean())
        result['methods'][method]=dict(subject_mean=subject_mean(rows,tr),cases=rows,
                                       seconds=time.monotonic()-start,selected=selected[method]['best'])
        arrays[method]=unit(pred)
        arrays[method+'_unclipped']=pred.detach().cpu().numpy()[:,0]
        np.savez_compressed(npz,**arrays)
        save_json(result_path,result)
        save_json(folder/(method+'_trace.json'),traces)
        print(json.dumps(dict(event='test_complete',fov=fov,method=method,**result['methods'][method]['subject_mean'])),flush=True)
    if hasattr(op192,'cg_diagnostics'):
        save_json(folder/f'cg_{args.stage}.json',op192.cg_diagnostics)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=PRIOR192_DATA)
    p.add_argument('--out',type=Path,default=RUNS/'prior192/evaluation')
    p.add_argument('--old-checkpoint',type=Path,default=RUNS/'prior96/strong_mouse96/model_ema.pt')
    p.add_argument('--hr-checkpoint',type=Path,default=RUNS/'prior192/train/model_ema.pt')
    p.add_argument('--device',default='cuda')
    p.add_argument('--stage',choices=['baseline','final','all'],default='all')
    p.add_argument('--fovs',nargs='+',type=int,default=[16,24])
    p.add_argument('--steps',type=int,default=40)
    p.add_argument('--sigma-max',type=float,default=2.)
    p.add_argument('--noise',type=float,default=.01)
    p.add_argument('--batch',type=int,default=3)
    p.add_argument('--val-slices',type=int,choices=[1,2],default=1)
    p.add_argument('--lambdas',type=float,nargs='+',default=[.01,.03,.1,.3,1.,3.])
    args=p.parse_args()
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    args.out.mkdir(parents=True,exist_ok=True)
    old,ock=load_strong_prior(args.old_checkpoint,args.device)
    hr,hck=(None,None) if args.stage=='baseline' else load_strong_prior(args.hr_checkpoint,args.device)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(old_checkpoint_sha256=sha(args.old_checkpoint),data_manifest_sha256=sha(args.data/'manifest.json'),
                  old_checkpoint_step=ock['step'],test_status='Existing held-out animals reused for a new developmental SR experiment; not an untouched confirmatory test cohort.')
    config['operator_sha256']=sha(HERE/'sr_operator.py')
    if hr is not None:
        if hck.get('manifest_sha256')!=config['data_manifest_sha256']:
            raise ValueError('HR checkpoint was trained on a different manifest')
        if not (args.hr_checkpoint.parent/'completed.json').exists():
            raise ValueError('Wait for HR training completion before freezing evaluation')
        config.update(hr_checkpoint_sha256=sha(args.hr_checkpoint),hr_checkpoint_step=hck['step'])
    # Completed methods and validation selections may only be reused with the
    # exact same images, noise, sampler, batch-seeding, and checkpoint bytes.
    shared_keys=['data_manifest_sha256','old_checkpoint_sha256','steps','sigma_max',
                 'noise','batch','val_slices','device']
    for previous_path in args.out.glob('run_*.json'):
        previous=json.loads(previous_path.read_text())
        for key in shared_keys:
            if previous.get(key)!=config.get(key):
                raise ValueError(f'Resume changes {key}; use a fresh output directory')
        if 'operator_sha256' in previous and previous['operator_sha256']!=config['operator_sha256']:
            raise ValueError('Resume changes the physical operator')
        if 'hr_checkpoint_sha256' in previous and hr is not None and previous['hr_checkpoint_sha256']!=config['hr_checkpoint_sha256']:
            raise ValueError('Resume changes the HR checkpoint; use a fresh output directory')
    save_json(args.out/f'run_{args.stage}.json',config)
    for fov in args.fovs:evaluate_fov(args,fov,old,hr)
    save_json(args.out/f'completed_{args.stage}.json',dict(completed=True,fovs=args.fovs))


if __name__=='__main__':main()
