"""Exploratory native 96x96 scanner observations reconstructed on a 192 grid.

Fixed coil/object-phase estimates come from the original InvA reconstruction.
There is no paired HR truth, so only measurement residuals are reported.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

from evaluate_sr import (HERE,V2,SCANS,save_json,sha,unit,upsample,
                         load_strong_prior,diffpir,scanner_matrices)
from operators import SpenMagnitudeOperator
from sr_operator import scanner_sr_matrices,SpenSuperResolutionOperator

from project_paths import RUNS


def load_case(path,device):
    inv,a,params=scanner_matrices(path,device)
    raw=np.asarray(scipy.io.loadmat(path,variable_names=['spen_phase_corrected_signal_rofft'])['spen_phase_corrected_signal_rofft'])
    if raw.shape!=(96,96,1,4):raise ValueError(f'Unexpected axes: {raw.shape}')
    signal=torch.as_tensor(raw[:,:,0,:],dtype=torch.complex64,device=device).permute(2,0,1)[None]
    z=torch.einsum('ij,bcjw->bciw',inv,signal)
    projected=torch.einsum('ij,bcjw->bciw',a,z)
    gain=(projected.conj()*signal).sum()/projected.abs().square().sum().clamp_min(1e-20)
    z=z*gain
    rss=z.abs().square().sum(1,keepdim=True).sqrt()
    scale=torch.quantile(rss,.995).clamp_min(1e-8)
    coils=z/rss.clamp_min(scale*1e-8)
    smax=torch.linalg.svdvals(a).max()
    y=signal/(scale*smax)
    op96=SpenMagnitudeOperator(a/smax,coils)
    anchor=2*rss/scale-1
    a192,a96,_,meta=scanner_sr_matrices(path,192,device)
    assert torch.allclose(a96,a/smax,atol=1e-6)
    # Interpolate fixed complex fractions; this does not add sensitivity data.
    c192=F.interpolate(coils.real,size=(192,192),mode='bilinear',align_corners=False)
    c192=c192+1j*F.interpolate(coils.imag,size=(192,192),mode='bilinear',align_corners=False)
    c192=c192/c192.abs().square().sum(1,keepdim=True).sqrt().clamp_min(1e-8)
    op192=SpenSuperResolutionOperator(a192,c192,measurement_size=96,sigma_noise=.02)
    meta.update(path=str(path),source_sha256=sha(path),raw_shape=list(raw.shape),
                magnitude_scale=float(scale),source_encoding_smax=float(smax),
                scalar_gain=[float(gain.real),float(gain.imag)],
                anchor96_measurement_nrmse=float(op96.relative_residual(anchor,y)),
                nuisance='Fixed InvA-derived complex fractions, bilinearly interpolated and RSS-normalized at192. They are not independently acquired coil maps.',
                display='All methods use the SAME original anchor magnitude scale and [0,1] window; rotate180 for display only.')
    return op96,op192,y,anchor,meta


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=RUNS/'prior192/real')
    p.add_argument('--reference-checkpoint',type=Path,default=RUNS/'prior96/strong_mouse96/model_ema.pt')
    p.add_argument('--checkpoint',type=Path,default=RUNS/'prior192/train/model_ema.pt')
    p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=40)
    p.add_argument('--skip-reference',action='store_true',
                   help='Evaluate the new HR prior without requiring a legacy 96 checkpoint')
    p.add_argument('--all-slices',action='store_true',
                   help='Evaluate every exported slice in each scan instead of three positions')
    p.add_argument('--fov',type=int,nargs='+',choices=(16,24),default=[16,24])
    args=p.parse_args()
    if args.steps < 2:
        p.error('--steps must be at least 2')
    selected=[]
    for fov in dict.fromkeys(args.fov):
        paths=sorted(SCANS[fov].glob('slice_*.mat'),key=lambda p:int(p.stem.split('_')[-1]))
        if not paths:
            raise FileNotFoundError(f'No real slices in {SCANS[fov]}')
        positions=[(i/max(len(paths)-1,1),path) for i,path in enumerate(paths)] if args.all_slices else [
            (pos,paths[int(pos*(len(paths)-1))]) for pos in (.3,.5,.7)]
        selected.extend((fov,pos,path) for pos,path in positions)
    if len({str(path) for _,_,path in selected}) != len(selected):
        raise ValueError('Duplicate real slices selected')
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    args.out.mkdir(exist_ok=True,parents=True)
    if any(args.out.glob('fov*_slice_*.npz')) or (args.out/'completed.json').exists():
        raise FileExistsError(f'Use a fresh output directory: {args.out}')
    checkpoint_hash=sha(args.checkpoint)
    save_json(args.out/'run_config.json',dict(checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=checkpoint_hash,script_sha256=sha(Path(__file__)),
        all_slices=args.all_slices,steps=args.steps,noise=.02,lamb=1.,rho=.003,
        sigma_max=2.,seed=73,device=args.device,skip_reference=args.skip_reference,
        cases=[dict(fov_mm=fov,selected_position=pos,path=str(path)) for fov,pos,path in selected]))
    started=time.monotonic()
    save_json(args.out/'status.json',dict(status='running',completed_cases=0,total_cases=len(selected)))
    hr,ck=load_strong_prior(args.checkpoint,args.device)
    old = None
    if not args.skip_reference:
        old,_=load_strong_prior(args.reference_checkpoint,args.device)
    rows=[]
    for fov in dict.fromkeys(args.fov):
        for _,pos,path in [row for row in selected if row[0]==fov]:
            case_started=time.monotonic()
            name=f'fov{fov}_{path.stem}'
            op96,op192,y,anchor,meta=load_case(path,args.device)
            if not torch.isfinite(y).all() or not torch.isfinite(anchor).all():
                raise FloatingPointError(f'Non-finite real input: {name}')
            meta.update(name=name,fov_mm=fov,selected_position=pos,no_reference_truth=True,methods={})
            images={'anchor96_up':upsample(anchor)}
            # Fixed real-data parameters, chosen BEFORE inspecting these scans.
            images['tikh96_up']=upsample(op96.proximal(torch.full_like(anchor,-1),y,.003))
            if old is not None:
                images['diff96_up']=upsample(diffpir(old,op96,y,steps=args.steps,sigma_noise=.02,lamb=1.,seed=73,sigma_max=2.)[0])
            images['tikh192']=op192.proximal(torch.full((1,1,192,192),-1.,device=args.device),y,.003)
            images['diff192_hr'],trace=diffpir(hr,op192,y,steps=args.steps,sigma_noise=.02,lamb=1.,seed=73,sigma_max=2.)
            arrays={'observation':y.cpu().numpy()}
            for key,x in images.items():
                if not torch.isfinite(x).all():
                    raise FloatingPointError(f'Non-finite reconstruction: {name}/{key}')
                arrays[key]=unit(x)[0]
                arrays[key+'_unclipped']=x.cpu().numpy()[0,0]
                meta['methods'][key]=dict(measurement_nrmse=float(op192.relative_residual(x,y)),
                    displayed_measurement_nrmse=float(op192.relative_residual(x.clamp(-1,1),y)),
                    outside_range_fraction=float(((x<-1)|(x>1)).float().mean()))
            np.savez_compressed(args.out/(name+'.npz'),**arrays)
            meta['elapsed_sec']=time.monotonic()-case_started
            save_json(args.out/(name+'.json'),meta)
            save_json(args.out/(name+'_cg.json'),op192.cg_diagnostics)
            save_json(args.out/(name+'_diffpir.json'),trace)
            rows.append(meta)
            save_json(args.out/'status.json',dict(status='running',completed_cases=len(rows),
                total_cases=len(selected),last_case=name,elapsed_sec=time.monotonic()-started))
            print(json.dumps(dict(event='real_complete',name=name,methods=meta['methods'])),flush=True)
    save_json(args.out/'summary.json',dict(cases=rows,checkpoint_sha256=checkpoint_hash,
        checkpoint_step=ck['step'],steps=args.steps,noise=.02,lamb=1.,rho=.003,sigma_max=2.,
        all_slices=args.all_slices,
        reference_checkpoint_sha256=sha(args.reference_checkpoint) if old is not None else None,
        checkpoint_manifest_sha256=ck.get('manifest_sha256'),
        note='Exploratory real scans with no paired HR truth. Measurement residual is not an anatomical accuracy score. These acquisition-derived coil/phase estimates have model error; no claim of doubled true spatial resolution.'))
    render(args.out,rows)
    save_json(args.out/'completed.json',dict(completed=True,cases=len(rows)))
    save_json(args.out/'status.json',dict(status='completed',completed_cases=len(rows),
        total_cases=len(selected),elapsed_sec=time.monotonic()-started))


def render(out,rows):
    if len(rows)<=6:
        render_panel(out,rows,'real_comparison')
        return
    overview=[]
    for fov in sorted({row['fov_mm'] for row in rows}):
        group=[row for row in rows if row['fov_mm']==fov]
        overview.extend(group[i] for i in dict.fromkeys(int(pos*(len(group)-1)) for pos in (.3,.5,.7)))
        for start in range(0,len(group),6):
            render_panel(out,group[start:start+6],f'fov{fov}_page{start//6+1:02d}')
    render_panel(out,overview,'real_comparison')


def render_panel(out,rows,stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'anchor96_up':'Phase + InvA 96 / bicubic','tikh96_up':'Tikhonov 96 / bicubic',
            'diff96_up':'Diffusion 96 / bicubic','tikh192':'Tikhonov 192','diff192_hr':'Diffusion 192 / HR prior'}
    labels={key: label for key,label in labels.items() if key in rows[0]['methods']}
    fig,axes=plt.subplots(len(labels),len(rows),figsize=(max(4,2.2*len(rows)),2.4*len(labels)+1),squeeze=False)
    for col,row in enumerate(rows):
        z=np.load(out/(row['name']+'.npz'))
        for i,(key,label) in enumerate(labels.items()):
            ax=axes[i,col]
            ax.imshow(np.rot90(z[key],2),cmap='gray',vmin=0,vmax=1,interpolation='none')
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(row['name'],fontsize=8)
            if col==0:ax.set_ylabel(label,fontsize=9)
    title = ('Real SPEN: 96 x 96 -> 192 x 192\nSame anchor scale across methods\nNo paired HR ground truth'
             if len(rows)<4 else 'Real SPEN: 96 x 96 observations -> 192 x 192 grid\nSame anchor scale for every method; no paired high-resolution ground truth')
    fig.suptitle(title,fontsize=10 if len(rows)<4 else 12)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(out/(stem+'.png'),dpi=180)
    fig.savefig(out/(stem+'.pdf'))
    plt.close(fig)


if __name__=='__main__':main()
