"""Bounded real-data inverse-solver diagnostics; no GT or checkpoint selection.

Compare numerical operator checks and one-factor solver changes on fixed cases.
Measurement fit and differences between predictions are not accuracy metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from evaluate_reconstruction import load_prior, save_json, sha, stable_seed
from reference_cases import load_real_reference
from latent_reconstruction import latent_reconstruct, latent_proximal


def checks(case, codec, mean, std):
    op, obs, initial = case['op'], case['observation'], case['initial_image']
    rng = torch.Generator(device=initial.device).manual_seed(104 + case['case_index'])
    values = []
    for _ in range(3):
        x = torch.randn(initial.shape, device=initial.device, generator=rng)
        y = torch.randn(obs.shape, device=obs.device, dtype=obs.dtype, generator=rng)
        ax, aty = op.linear(x), op.adjoint(y)
        lhs, rhs = (ax.conj() * y).sum().real, (x * aty).sum()
        values.append(float((lhs-rhs).abs() / (lhs.abs()+rhs.abs()).clamp_min(1e-8)))
    with torch.enable_grad():
        x = initial.clone().requires_grad_()
        loss = (op.forward(x)-obs).abs().square().sum()
        automatic, = torch.autograd.grad(loss, x)
        manual = op.gradient(x.detach(), obs)
    gradient_error = float((automatic-manual).norm() / manual.norm().clamp_min(1e-12))
    with torch.no_grad():
        encoded = ((codec.encode(initial)-mean)/std).detach()
    dtype, tf32 = codec.autocast_dtype, torch.backends.cudnn.allow_tf32
    codec.autocast_dtype = None
    torch.backends.cudnn.allow_tf32 = False
    try:
        def objective(u):
            return (op.forward(codec.decode(u*std+mean, clamp=False))-obs).to(torch.complex128).abs().square().sum()
        with torch.enable_grad():
            u = encoded.clone().requires_grad_()
            g, = torch.autograd.grad(objective(u), u)
        directional = []
        random = torch.randn(g.shape, device=g.device, generator=rng)
        for name, direction in [('gradient', g.detach()/g.norm()), ('random', random/random.norm())]:
            analytic = float((g*direction).sum())
            for eps in (.1, .03, .01):
                with torch.no_grad():
                    numeric = float((objective(encoded+eps*direction)-objective(encoded-eps*direction))/(2*eps))
                directional.append(dict(direction=name, epsilon=eps, analytic=analytic, finite_difference=numeric,
                    relative_error=abs(numeric-analytic)/max(abs(numeric),abs(analytic),1e-10)))
    finally:
        codec.autocast_dtype = dtype
        torch.backends.cudnn.allow_tf32 = tf32
    responses = {}
    t = torch.arange(192, device=initial.device, dtype=torch.float32)
    for axis in ('pe','ro'):
        for cycles in (4, 80):
            wave = torch.sin(2*torch.pi*cycles*t/192)
            pattern = wave[:,None].expand(192,192) if axis=='pe' else wave[None,:].expand(192,192)
            pattern = pattern[None,None]
            responses[f'{axis}_{cycles}_cycles'] = float(op.linear(pattern).norm()/pattern.norm())
    result = dict(adjoint_relative_errors=values, image_gradient_relative_error=gradient_error,
                  decoder_directional_derivatives=directional, sinusoid_measurement_gain=responses,
                  note='Four sinusoidal probes do not estimate full resolution or full operator nullspace.')
    if max(values)>2e-4 or gradient_error>2e-5:
        raise ValueError(f'Physical operator consistency failed: {result}')
    for direction in ('gradient','random'):
        if min(v['relative_error'] for v in directional if v['direction']==direction)>.02:
            raise ValueError(f'Decoder directional derivative failed: {result}')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--case-indices',type=int,nargs='+',required=True)
    parser.add_argument('--initialization-probe',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    args.out.mkdir(parents=True,exist_ok=True)
    model=load_prior(args.run/'checkpoint.pt','cuda')
    net,codec,mean,std,metadata=model
    reference=load_real_reference(args.reference,'cuda')
    if args.initialization_probe:
        for index in args.case_indices:
            case=reference['cases'][index]
            folder=args.out/'initialization_probe'/f'case_{index:02d}'
            folder.mkdir(parents=True,exist_ok=False)
            inva=torch.as_tensor(np.rot90(case['reference_arrays']['phase_inva_unclipped'],2).copy(),device='cuda')[None,None]*2-1
            with torch.no_grad():
                encoded=codec.encode(inva.clamp(-1,1))
                dtype=codec.autocast_dtype;codec.autocast_dtype=None
                roundtrip=codec.decode(encoded,clamp=False)
                codec.autocast_dtype=dtype
            raw,trace=latent_reconstruct(net,codec,mean,std,case['op'],case['observation'],
                initial_image=inva.clamp(-1,1),steps=60,inner_steps=8,lamb=.1,sigma_noise=.02,
                sigma_max=1.,sigma_min=.02,prox_lr=.1,seed=stable_seed(case['key']),decoder_fp32=True)
            np.savez_compressed(folder/'arrays.npz',inva_init=raw[0,0].cpu().numpy(),
                                vae_roundtrip_inva=roundtrip[0,0].cpu().numpy())
            save_json(folder/'trace.json',trace)
            with torch.no_grad():
                rows=[dict(method=name,index=index,key=case['key'],
                    measurement_nrmse=float(case['op'].relative_residual(image,case['observation'])))
                    for name,image in [('inva_init',raw),('vae_roundtrip_inva',roundtrip)]]
            save_json(folder/'metrics.json',rows)
            print(json.dumps(dict(event='initialization_probe_completed',index=index,metrics=rows)),flush=True)
        return
    save_json(args.out/('config_'+'_'.join(map(str,args.case_indices))+'.json'),dict(
        checkpoint_sha256=metadata['checkpoint_sha256'],checkpoint_step=metadata['step'],
        cases=args.case_indices,reference_provenance=reference['provenance'],
        source_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('latent_reconstruction.py')]},
        selection='Fixed before new results: FOV16 acquisitions5,13 and FOV24 acquisitions3,15',
        no_ground_truth=True,parameters_not_selected_for_production=True))
    for index in args.case_indices:
        case=reference['cases'][index]
        folder=args.out/f'case_{index:02d}'
        folder.mkdir(exist_ok=False)
        print(json.dumps(dict(event='case_started',index=index,key=case['key'])),flush=True)
        audit=checks(case,codec,mean,std)
        save_json(folder/'operator_checks.json',audit)
        print(json.dumps(dict(event='operator_checks_passed',index=index,
                             adjoint_error=max(audit['adjoint_relative_errors']),
                             gradient_error=audit['image_gradient_relative_error'])),flush=True)
        saved=np.load(args.run/f'real/case_{index:02d}/arrays.npz')
        baseline=torch.as_tensor(saved['prediction_raw'],device='cuda')[None,None]
        inva=torch.as_tensor(np.rot90(case['reference_arrays']['phase_inva_unclipped'],2).copy(),device='cuda')[None,None]*2-1
        initial=case['initial_image']
        images,rows={},[]
        def record(name,raw,seconds=0.,trace=None):
            raw=raw.detach()
            if not torch.isfinite(raw).all():raise FloatingPointError(name)
            with torch.no_grad():
                row=dict(method=name,index=index,key=case['key'],seconds=seconds,
                    measurement_nrmse=float(case['op'].relative_residual(raw,case['observation'])),
                    display_rmse_to_original_diffusion=float((((raw+1)/2).clamp(0,1)-((baseline+1)/2).clamp(0,1)).square().mean().sqrt()),
                    display_rmse_to_inva=float((((raw+1)/2).clamp(0,1)-((inva+1)/2).clamp(0,1)).square().mean().sqrt()))
            if trace is not None:
                save_json(folder/(name+'_trace.json'),trace)
                if 'trace' in trace:
                    row.update(converged_subproblems=trace['converged_subproblems'],
                        median_relative_gradient=float(np.median([t['relative_gradient_norm'] for t in trace['trace']])))
            images[name]=raw[0,0].cpu().numpy()
            rows.append(row)
            save_json(folder/'metrics.json',rows)
            np.savez_compressed(folder/'arrays.npz',**images)
            print(json.dumps(dict(event='method_completed',**row)),flush=True)
        record('phase_inva',inva)
        record('initial_tikh96_up',initial)
        record('original_diffusion',baseline)
        with torch.no_grad():
            encoded=codec.encode(initial)
            original_dtype=codec.autocast_dtype
            codec.autocast_dtype=None
            roundtrip=codec.decode(encoded,clamp=False)
            codec.autocast_dtype=original_dtype
        record('vae_roundtrip_initial',roundtrip)
        parameters=dict(steps=60,inner_steps=8,lamb=.1,sigma_noise=.02,sigma_max=1.,
                        sigma_min=.02,prox_lr=.1,seed=stable_seed(case['key']),decoder_fp32=True)
        for name,overrides in [('baseline_replay',{}),('inner32',dict(inner_steps=32)),
                               ('lambda1',dict(lamb=1.)),('sigma3',dict(sigma_max=3.))]:
            start=time.monotonic()
            raw,trace=latent_reconstruct(net,codec,mean,std,case['op'],case['observation'],
                        initial_image=initial,**dict(parameters,**overrides))
            record(name,raw,time.monotonic()-start,trace)
        start=time.monotonic()
        original_dtype=codec.autocast_dtype
        codec.autocast_dtype=None
        try:
            # No DiT, no injected noise, rho=0: only the fixed VAE manifold and data.
            z,trace=latent_proximal((encoded-mean)/std,codec,mean,std,case['op'],case['observation'],
                                    rho=0.,inner_steps=480,prox_lr=.1,objective_rtol=0.)
            with torch.no_grad():raw=codec.decode(z*std+mean,clamp=False)
        finally:codec.autocast_dtype=original_dtype
        record('data_only_vae480',raw,time.monotonic()-start,trace)
        save_json(folder/'completed.json',dict(completed=True,index=index,methods=len(rows)))


if __name__=='__main__':main()
