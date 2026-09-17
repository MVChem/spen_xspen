"""Paired prior diagnostic and validation-only lambda selection for 96x96 priors."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT.parent / 'data'
os.environ.setdefault('SPEN_REFERENCE_ROOT', str(DATA / 'prior96_0911_260916/scanner_reference'))
sys.path[:0] = [str(PROJECT/'scripts'/p) for p in ('dit96', 'prior96', 'core')]
from pixel_model import load_pixel_dit
from model_v2 import load_strong_prior
from evaluate_mouse import cases, subject_mean, controlled_operator, observe
from evaluate import metrics, unit
from train import validate
from solvers import diffpir


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sampling_audit():
    manifest = json.loads((DATA/'prior96_0911_260916/mouse_mixed/manifest.json').read_text())
    records = manifest['records']['train']
    total = sum(r['sample_weight'] for r in records)
    groups = {}
    for field in ('species', 'dataset', 'view'):
        weight, count = defaultdict(float), defaultdict(int)
        for r in records:
            weight[r[field]] += r['sample_weight']/total
            count[r[field]] += 1
        groups[field] = dict(old_probability=dict(weight), old_count=dict(count),
                             new_probability_for_old_records={k:v/28160 for k,v in count.items()})
    return dict(groups=groups, old_mixed_steps=30000, new_steps=60000, batch=96,
                old_expected_mouse_draws=30000*96*groups['species']['old_probability']['mouse'],
                new_expected_old_mouse_draws=60000*96*groups['species']['new_probability_for_old_records']['mouse'],
                note='Expected draws, not distinct images; new lab RAT species not inferred from directory name')


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    args.out=args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()): raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False
    device='cuda'
    started=time.monotonic()
    inputs=DATA/'prior96_0911_260916/mouse_mixed'
    checkpoints=dict(unet=PROJECT/'runs/retrain_0911_260916/mouse_mixed/model_ema.pt',
                     dit=PROJECT/'runs/rodent96_dit20m_260917/training/model_ema.pt')
    sources=[Path(__file__),PROJECT/'scripts/core/solvers.py',PROJECT/'scripts/core/operators.py',
             PROJECT/'scripts/core/evaluate.py',PROJECT/'scripts/core/train.py',
             PROJECT/'scripts/prior96/evaluate_mouse.py',PROJECT/'scripts/prior96/model_v2.py',
             PROJECT/'scripts/dit96/pixel_model.py',PROJECT/'scripts/latent48/dit.py']
    grid=[.1,.3,1.,3.,10.]
    config=dict(checkpoints={k:dict(path=str(v),sha256=hash_file(v)) for k,v in checkpoints.items()},
                lambda_grid=grid, selection='maximum subject-mean PSNR on validation only; grid fixed before evaluation',
                steps=60, validation_limit=16, test_limit=30, denoising_limit=128,
                sigma_levels=[.02,.05,.1,.3,1.,2.,5.,20.],
                source_sha256={str(p):hash_file(p) for p in sources},
                runtime=dict(torch=torch.__version__,cudnn_benchmark=True,matmul_tf32=False),
                replay_tolerance='BF16 plus cuDNN autotuning: compare aggregate metrics (0.05 dB / 0.002 SSIM); record pixel max error',
                note='Developmental diagnosis after inspecting prior test results; not a new pristine benchmark')
    save(args.out/'config.json',config)
    save(args.out/'sampling_audit.json',sampling_audit())
    nets={}
    for name,path in checkpoints.items():
        net,state=(load_strong_prior if name=='unet' else load_pixel_dit)(path,device)
        nets[name]=net
        assert state['step']==(30000 if name=='unet' else 60000)
    x,records=cases(inputs,'val',16,128,device)
    denoise=dict(records=records,weighted_edm_loss={},noise_levels={})
    for name,net in nets.items():denoise['weighted_edm_loss'][name]=validate(net,x)
    print(json.dumps(dict(event='paired_edm',**denoise['weighted_edm_loss'])),flush=True)
    for sigma in config['sigma_levels']:
        gen=torch.Generator(device=device).manual_seed(9400+int(100*sigma))
        noisy=x+sigma*torch.randn(x.shape,device=device,generator=gen)
        scores={}
        for name,net in nets.items():
            prediction=torch.cat([net(part,sigma) for part in noisy.split(24)])
            rows=metrics(prediction,x)
            mse=((prediction-x).square().flatten(1).mean(1)/4).cpu().tolist()
            for row,value in zip(rows,mse):row['raw_unit_mse']=value
            scores[name]=dict(subject_mean=subject_mean(rows,records),cases=rows)
        denoise['noise_levels'][str(sigma)]=scores
        print(json.dumps(dict(event='denoising',sigma=sigma,**{k:v['subject_mean'] for k,v in scores.items()})),flush=True)
    save(args.out/'denoising.json',denoise)
    val,vr=cases(inputs,'val',16,16,device)
    test,tr=cases(inputs,'test',16,30,device)
    summary={}
    for acceleration,noise in ((1,.01),(2,.02)):
        key=f'fov16_R{acceleration}'
        op=controlled_operator(16,acceleration,noise,device)
        yval=observe(op,val,noise,9100+16+acceleration)
        old=PROJECT/'runs/rodent96_dit20m_260917/evaluation/simulation'
        old_report=json.loads((old/f'{key}_metrics.json').read_text())
        assert tr==old_report['records']
        with np.load(old/f'{key}.npz') as a:
            np.testing.assert_array_equal(unit(test),a['target'])
            y=torch.tensor(a['observation'],device=device)
            old_arrays={name:a[name].copy() for name in ('unet','dit')}
        trials,selected={},{ }
        for name,net in nets.items():
            trials[name]=[]
            for lamb in grid:
                pred,_=diffpir(net,op,yval,steps=60,sigma_noise=noise,lamb=lamb,seed=71)
                row=dict(lamb=lamb,**subject_mean(metrics(pred,val),vr))
                trials[name].append(row)
                print(json.dumps(dict(event='validation_lambda',condition=key,model=name,**row)),flush=True)
            selected[name]=max(trials[name],key=lambda r:r['psnr'])['lamb']
        save(args.out/f'{key}_validation.json',dict(records=vr,trials=trials,selected=selected))
        result={};arrays=dict(target=unit(test),observation=y.cpu().numpy())
        for name,net in nets.items():
            pred,_=diffpir(net,op,y,steps=60,sigma_noise=noise,lamb=selected[name],seed=72)
            rows=metrics(pred,test)
            result[name]=dict(lamb=selected[name],subject_mean=subject_mean(rows,tr),cases=rows,
                              previous_lambda1=old_report['methods'][name]['subject_mean'])
            arrays[name]=unit(pred)
            if selected[name]==1.:
                error=float(np.max(np.abs(arrays[name]-old_arrays[name])))
                result[name]['frozen_replay_max_abs']=error
                before=result[name]['previous_lambda1'];after=result[name]['subject_mean']
                result[name]['frozen_replay_metric_delta']={k:after[k]-before[k] for k in ('psnr','ssim')}
                if abs(after['psnr']-before['psnr'])>.05 or abs(after['ssim']-before['ssim'])>.002:
                    raise ValueError(f'Frozen replay metrics differ materially: {name} {key}')
        summary[key]=result
        save(args.out/f'{key}_test.json',dict(records=tr,methods=result))
        np.savez_compressed(args.out/f'{key}_test.npz',**arrays)
        print(json.dumps(dict(event='selected_test',condition=key,**{k:{s:v[s] for s in ('lamb','subject_mean','previous_lambda1')} for k,v in result.items()})),flush=True)
    save(args.out/'summary.json',summary)
    save(args.out/'completed.json',dict(status='complete',seconds=time.monotonic()-started))


if __name__=='__main__':main()
