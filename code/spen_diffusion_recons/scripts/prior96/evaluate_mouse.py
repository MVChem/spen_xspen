"""Paired old/new-prior evaluation, with inverse parameters selected on validation only."""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.io
import torch
from model_v2 import load_strong_prior
from prepare_data import HERE,V1,PROJECT,sha256
from model import load_prior
from train import validate
from operators import SpenMagnitudeOperator,scanner_matrices,synthetic_coils
from solvers import diffpir
from evaluate import metrics,average,unit,plot_synthetic
from project_paths import RUNS, PRIOR96_DATA

SCANS={16:PROJECT/'data/mat/20240321_lxj_spen_mouse_240321_1_1_1',
       24:PROJECT/'data/mat/20240115_lxj_SPEN_96_240115_1_1_1'}


def cases(data,part,fov,limit,device):
    m=json.loads((data/'manifest.json').read_text())
    a=np.load(data/f'{part}.npy',mmap_mode='r');by_subject=defaultdict(list)
    for i,r in enumerate(m['records'][part]):
        if r['view']==f'mouse_fov{fov}':by_subject[r['subject']].append(i)
    # Round-robin subjects: a dense volume cannot dominate by contributing more slices.
    each=math.ceil(limit/len(by_subject));queues=[]
    for subject,ids in sorted(by_subject.items()):
        positions=np.linspace(.2,.8,each) if each>1 else np.array([.5])
        queues.append([ids[int(p*(len(ids)-1))] for p in positions])
    selected=[q[j] for j in range(each) for q in queues][:max(limit,len(queues))]
    records=[dict(m['records'][part][i],rot180=(j%2==0)) for j,i in enumerate(selected)]
    x=torch.tensor(a[selected].astype(np.float32)/65535.,device=device)[:,None]
    x[::2]=torch.rot90(x[::2],2,(-2,-1))
    return x*2-1,records


def subject_mean(rows,records):
    by=defaultdict(list)
    for r,rec in zip(rows,records):by[rec['subject']].append(r)
    return average([average(v) for v in by.values()])


@torch.no_grad()
def denoising(nets,args,out):
    x,records=cases(args.data,args.partition,16,128,args.device)
    report={'records':records,'same_fixed_noise_edm_loss':{},'noise_levels':{}}
    for name,net in nets.items():report['same_fixed_noise_edm_loss'][name]=validate(net,x)
    for sigma in (.05,.1,.3,1.,2.):
        gen=torch.Generator(device=x.device).manual_seed(9400+int(100*sigma))
        noisy=x+sigma*torch.randn(x.shape,device=x.device,generator=gen)
        detail={}
        for name,net in nets.items():
            pred=torch.cat([net(noisy[i:i+24],sigma) for i in range(0,len(x),24)])
            rows=metrics(pred,x)
            detail[name]=dict(subject_mean=subject_mean(rows,records),cases=rows)
        report['noise_levels'][str(sigma)]=detail
    (out/'denoising.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(event='denoising',partition=args.partition,**report['same_fixed_noise_edm_loss'])),flush=True)


def controlled_operator(fov,acceleration,noise,device):
    _,a,_=scanner_matrices(SCANS[fov]/'slice_7.mat',device)
    a=a/torch.linalg.svdvals(a).max()
    gen=torch.Generator(device=device).manual_seed(4500+fov+acceleration)
    c=synthetic_coils(device=device,phase_strength=.7)
    axis=torch.linspace(-1,1,96,device=device);yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    coeff=torch.randn(4,3,device=device,generator=gen)*.5
    phase=coeff[:,0,None,None]+coeff[:,1,None,None]*xx+coeff[:,2,None,None]*yy
    gain=torch.exp(.2*torch.randn(4,1,1,device=device,generator=gen))
    c=c*gain*torch.exp(1j*phase);c=c/c.abs().square().sum(0,keepdim=True).sqrt()
    mask=torch.ones(96,device=device,dtype=torch.bool)
    if acceleration==2:
        # Fixed irregular half sampling; both priors receive exactly the same measurements.
        mask[:]=False;mask[torch.randperm(96,device=device,generator=gen)[:48]]=True
    return SpenMagnitudeOperator(a,c,mask,noise)


def observe(op,x,noise,seed):
    gen=torch.Generator(device=x.device).manual_seed(seed);y=op.forward(x)
    return y+noise*(torch.randn(y.shape,device=x.device,generator=gen)+1j*torch.randn(y.shape,device=x.device,generator=gen))


def reconstruction(nets,args,out):
    summary={}
    for fov in (16,24):
        val,vr=cases(args.data,'val',fov,16,args.device)
        test,tr=cases(args.data,args.partition,fov,args.limit,args.device)
        for acceleration,noise in ((1,.01),(2,.02)):
            key=f'fov{fov}_R{acceleration}';op=controlled_operator(fov,acceleration,noise,args.device)
            yval=observe(op,val,noise,9100+fov+acceleration)
            trials={};selected={}
            for name,net in nets.items():
                trials[name]=[]
                for lamb in (.3,1.,3.):
                    pred,_=diffpir(net,op,yval,steps=args.steps,sigma_noise=noise,lamb=lamb,seed=71)
                    result=dict(lamb=lamb,**subject_mean(metrics(pred,val),vr))
                    trials[name].append(result)
                selected[name]=max(trials[name],key=lambda r:r['psnr'])['lamb']
            trials['tikhonov']=[]
            for rho in (.001,.003,.01,.03,.1):
                pred=op.proximal(torch.full_like(val,-1),yval,rho)
                trials['tikhonov'].append(dict(rho=rho,**subject_mean(metrics(pred,val),vr)))
            selected['tikhonov']=max(trials['tikhonov'],key=lambda r:r['psnr'])['rho']
            (out/f'{key}_selection.json').write_text(json.dumps(dict(trials=trials,selected=selected,validation_records=vr),indent=2)+'\n')
            y=observe(op,test,noise,9200+fov+acceleration)
            outputs={'Target':unit(test)};detail={};arrays={'target':unit(test),'observation':y.cpu().numpy()}
            for name in ['tikhonov',*nets]:
                start=time.monotonic()
                if name=='tikhonov':pred=op.proximal(torch.full_like(test,-1),y,selected[name])
                else:pred,_=diffpir(nets[name],op,y,steps=args.steps,sigma_noise=noise,lamb=selected[name],seed=72)
                rows=metrics(pred,test)
                residual=op.relative_residual(pred,y).cpu().tolist()
                for row,res in zip(rows,residual):row['measurement_nrmse']=res
                detail[name]=dict(subject_mean=subject_mean(rows,tr),cases=rows,seconds=time.monotonic()-start)
                arrays[name]=unit(pred);outputs[name]=unit(pred)
            np.savez_compressed(out/f'{key}.npz',**arrays)
            plot_synthetic(outputs,out/f'{key}.png')
            (out/f'{key}_metrics.json').write_text(json.dumps(dict(records=tr,methods=detail),indent=2)+'\n')
            summary[key]={k:v['subject_mean'] for k,v in detail.items()}
            print(json.dumps(dict(event='inverse_evaluation',case=key,**summary[key])),flush=True)
    (out/'synthetic_summary.json').write_text(json.dumps(dict(note='Controlled known coil/phase model; this is not proof of matching real artifact distributions.',cases=summary),indent=2)+'\n')


def real_case(path,device):
    inv,a,params=scanner_matrices(path,device)
    mat=scipy.io.loadmat(path)
    raw=np.asarray(mat['spen_phase_corrected_signal_rofft'])
    if raw.shape!=(96,96,1,4):raise ValueError(f'Unexpected scanner axes: {raw.shape}')
    signal=torch.tensor(raw[:,:,0,:],device=device,dtype=torch.complex64).permute(2,0,1)[None]
    z=torch.einsum('ij,bcjw->bciw',inv,signal)
    # calcInvA is a scaled regularized reconstruction, not the exact inverse of A.
    # Estimate one complex scanner gain from measurements before defining image units.
    projected=torch.einsum('ij,bcjw->bciw',a,z)
    gain=(projected.conj()*signal).sum()/projected.abs().square().sum().clamp_min(1e-20)
    z=z*gain
    rss=z.abs().square().sum(1,keepdim=True).sqrt();scale=torch.quantile(rss,.995).clamp_min(1e-8)
    coils=z/rss.clamp_min(scale*1e-8);smax=torch.linalg.svdvals(a).max()
    op=SpenMagnitudeOperator(a/smax,coils);y=signal/(scale*smax);anchor=2*rss/scale-1
    native=torch.einsum('ij,bcjw->bciw',a,z)/(scale*smax)
    error=float((op.forward(anchor)-native).abs().max())
    if error>2e-5:raise ValueError(f'Scanner normalization mismatch: {error}')
    return op,y,anchor,dict(path=str(path),sha256=sha256(path),raw_shape=list(raw.shape),full_args=params,
        magnitude_scale=float(scale),normalization_error=error,anchor_residual=float(op.relative_residual(anchor,y)),
        inva_to_measurement_gain_real=float(gain.real),inva_to_measurement_gain_imag=float(gain.imag))


def scanner(nets,args,out):
    images=defaultdict(list);report=[]
    for fov,directory in SCANS.items():
        paths=sorted(directory.glob('slice_*.mat'),key=lambda p:int(p.stem.split('_')[-1]))
        for pos in (.3,.5,.7):
            path=paths[int(pos*(len(paths)-1))];op,y,anchor,meta=real_case(path,args.device)
            row={'Phase + InvA':anchor};meta['fov_mm']=fov
            for name,net in nets.items():
                pred,_=diffpir(net,op,y,steps=args.steps,sigma_noise=.02,lamb=1.,seed=73)
                row[name]=pred
                meta[name]=dict(residual=float(op.relative_residual(pred,y)),
                    displayed_residual=float(op.relative_residual(pred.clamp(-1,1),y)),
                    outside_range_fraction=float(((pred<-1)|(pred>1)).float().mean()))
            for key,x in row.items():images[key].append(np.rot90(unit(x)[0],2))
            np.savez_compressed(out/f'real_fov{fov}_{path.stem}.npz',observation=y.cpu().numpy(),**{k:unit(x)[0] for k,x in row.items()})
            report.append(meta)
    plot_synthetic({k:np.stack(v) for k,v in images.items()},out/'real_mouse_comparison.png')
    (out/'real_mouse_metrics.json').write_text(json.dumps(dict(cases=report,
        note='No clean real SPEN ground truth. Coil/phase estimates are fixed from the existing phase-corrected InvA anchor, not independently measured. Noise=.02 and lambda=1 are assumptions, not tuned against a real target.'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,default=PRIOR96_DATA);p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=60);p.add_argument('--limit',type=int,default=30)
    p.add_argument('--reference-checkpoint',type=Path,default=RUNS/'core/edm_rat96/best.pt')
    p.add_argument('--partition',choices=['val','test'],default='test')
    p.add_argument('--only',choices=['all','denoising','synthetic','scanner'],default='all')
    args=p.parse_args();torch.set_num_threads(3);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False
    if args.out.exists() and any(args.out.iterdir()):raise FileExistsError('Use a fresh evaluation directory')
    args.out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(args.checkpoint,args.out/'evaluated_checkpoint.pt')
    new,ckpt=load_strong_prior(args.out/'evaluated_checkpoint.pt',args.device)
    if ckpt.get('manifest_sha256')!=sha256(args.data/'manifest.json'):
        raise ValueError('Evaluation requires a model trained on this exact mixed dataset')
    old,oc=load_prior(args.reference_checkpoint,args.device)
    nets={'V1 rat prior':old,'V2 mouse prior':new}
    snapshot=args.out/'source_snapshot';snapshot.mkdir()
    sources={}
    for root,names in [(HERE,['evaluate_mouse.py','model_v2.py','tiny_unet_v2.py']),
                       (V1,['operators.py','solvers.py','evaluate.py','model.py','train.py','tiny_unet.py'])]:
        for name in names:
            destination=snapshot/(('v1_' if root==V1 else '')+name)
            shutil.copyfile(root/name,destination);sources[str(destination.name)]=sha256(destination)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    cfg.update(checkpoint_step=ckpt['step'],checkpoint_sha256=sha256(args.out/'evaluated_checkpoint.pt'),
        v1_checkpoint_sha256=sha256(args.reference_checkpoint),dataset_manifest_sha256=sha256(args.data/'manifest.json'),
        sources=sources,test_status='Held out animals/litters; evaluation is developmental if reused for subsequent model design.')
    (args.out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    if args.only in ['all','denoising']:denoising(nets,args,args.out)
    if args.only in ['all','synthetic']:reconstruction(nets,args,args.out)
    if args.only in ['all','scanner']:scanner(nets,args,args.out)
    (args.out/'completed.json').write_text(json.dumps(dict(step=ckpt['step'],status='complete'))+'\n')
    print('EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
