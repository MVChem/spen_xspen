"""Select hyperparameters on validation, then run held-out synthetic and real scans."""
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from data import read_split, load_images, sha256, ROOT
from model import load_prior
from operators import make_synthetic_operator, load_scanner_case
from solvers import diffpir, daps

BOXES={13:(27,30,39,13),17:(6,43,35,19),24:(31,38,33,26),31:(52,37,28,21),51:(34,51,31,22),62:(14,37,49,22)}


def unit(x):
    return ((x.detach().cpu().numpy()[:,0]+1)/2).clip(0,1)


def metrics(pred, gt):
    """Fixed [0,1] image range; no fit or per-output rescaling."""
    pred,gt=unit(pred),unit(gt)
    rows=[]
    for p,g in zip(pred,gt):
        mse=np.mean((p-g)**2)
        mu_p=gaussian_filter(p,1.5,truncate=3.5)
        mu_g=gaussian_filter(g,1.5,truncate=3.5)
        var_p=gaussian_filter(p*p,1.5,truncate=3.5)-mu_p**2
        var_g=gaussian_filter(g*g,1.5,truncate=3.5)-mu_g**2
        cov=gaussian_filter(p*g,1.5,truncate=3.5)-mu_p*mu_g
        ssim=((2*mu_p*mu_g+.01**2)*(2*cov+.03**2))/((mu_p**2+mu_g**2+.01**2)*(var_p+var_g+.03**2))
        rows.append(dict(psnr=float(-10*np.log10(max(float(mse),1e-12))),ssim=float(ssim[5:-5,5:-5].mean()),
                         nrmse=float(np.linalg.norm(p-g)/max(np.linalg.norm(g),1e-12))))
    return rows


def average(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}


def choose_cases(files, limit):
    # Use base images only: augmented flipped duplicates do not count twice.
    base=[n for n in files if '__base.png' in n]
    ordered=sorted(base)
    ids=np.linspace(0,len(ordered)-1,min(limit,len(ordered)),dtype=int)
    return [ordered[i] for i in ids]


def observations(op, x, seed, noise):
    gen=torch.Generator(device=x.device).manual_seed(seed)
    y=op.forward(x)
    # noise is standard deviation of EACH real/imaginary component.
    re=torch.randn(y.shape,device=x.device,generator=gen)
    im=torch.randn(y.shape,device=x.device,generator=gen)
    return y+noise*(re+1j*im)


def plot_synthetic(images, out):
    keys=list(images)
    count=min(6,len(next(iter(images.values()))))
    fig,axes=plt.subplots(len(keys),count,figsize=(count*2,len(keys)*2),squeeze=False)
    for row,key in enumerate(keys):
        a=images[key]
        for col in range(count):
            ax=axes[row,col]
            ax.imshow(a[col],cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
            if col==0:ax.set_ylabel(key,fontsize=9)
    fig.tight_layout(pad=.3);fig.savefig(out,dpi=150);plt.close(fig)


def synthetic(net,out,args):
    split=read_split()
    names={k:choose_cases(split[k],args.limit) for k in ('val','test')}
    val=load_images(names['val']).to(args.device)
    test=load_images(names['test']).to(args.device)
    summary={}
    (out/'evaluation_files.json').write_text(json.dumps(names,indent=2)+'\n')
    for acceleration in (1,2):
        op=make_synthetic_operator(args.device,acceleration,args.noise)
        yval=observations(op,val,110+acceleration,args.noise)
        trials=[]
        for lamb in (.1,.3,1.,3.,10.):
            t=time.monotonic()
            pred,_=diffpir(net,op,yval,steps=args.steps,sigma_noise=args.noise,lamb=lamb,seed=17)
            row=dict(method='diffpir',lamb=lamb,**average(metrics(pred,val)),seconds=time.monotonic()-t)
            trials.append(row);print(json.dumps(dict(event='validation',R=acceleration,**row)),flush=True)
        for rho in (.001,.003,.01,.03,.1):
            pred=op.proximal(torch.full_like(val,-1),yval,rho)
            trials.append(dict(method='tikhonov',rho=rho,**average(metrics(pred,val))))
        chosen={m:max([r for r in trials if r['method']==m],key=lambda r:r['psnr']) for m in ('diffpir','tikhonov')}
        # Persist selection BEFORE test inference. Tests never select hyperparameters.
        (out/f'R{acceleration}_selection.json').write_text(json.dumps(dict(trials=trials,selected=chosen),indent=2)+'\n')
        ytest=observations(op,test,210+acceleration,args.noise)
        t=time.monotonic()
        dp,trace=diffpir(net,op,ytest,steps=args.steps,sigma_noise=args.noise,lamb=chosen['diffpir']['lamb'],seed=27)
        dp_time=time.monotonic()-t
        tik=op.proximal(torch.full_like(test,-1),ytest,chosen['tikhonov']['rho'])
        outputs={'tikhonov':tik,'diffpir':dp}
        detail={}
        for method,pred in outputs.items():
            rows=metrics(pred,test)
            residual=op.relative_residual(pred,ytest).cpu().tolist()
            clipped=op.relative_residual(pred.clamp(-1,1),ytest).cpu().tolist()
            for n,r,res,cl in zip(names['test'],rows,residual,clipped):
                r.update(file=n,measurement_nrmse=res,displayed_image_measurement_nrmse=cl)
            detail[method]=dict(mean=average([{k:v for k,v in r.items() if k!='file'} for r in rows]),cases=rows)
        detail['diffpir']['seconds_total']=dp_time
        if args.daps_cases:
            results=[]
            t=time.monotonic()
            for i in range(min(args.daps_cases,len(test))):
                results.append(daps(net,op,ytest[i:i+1],steps=args.daps_steps,tau=args.noise,seed=37+i))
            da=torch.cat(results)
            rows=metrics(da,test[:len(da)])
            for n,r,res in zip(names['test'],rows,op.relative_residual(da,ytest[:len(da)]).cpu().tolist()):
                r.update(file=n,measurement_nrmse=res)
            detail['daps_fixed_config_subset']=dict(mean=average([{k:v for k,v in r.items() if k!='file'} for r in rows]),cases=rows,seconds_total=time.monotonic()-t)
            np.savez_compressed(out/f'R{acceleration}_daps.npz',reconstruction=unit(da),target=unit(test[:len(da)]))
        np.savez_compressed(out/f'R{acceleration}_test.npz',target=unit(test),tikhonov=unit(tik),diffpir=unit(dp),
                            observation=ytest.cpu().numpy(),prediction_model_range=dp.cpu().numpy())
        plot_synthetic({'Target':unit(test),'Tikhonov':unit(tik),'Diffusion + DiffPIR':unit(dp)},out/f'R{acceleration}_test.png')
        (out/f'R{acceleration}_test_metrics.json').write_text(json.dumps(detail,indent=2)+'\n')
        (out/f'R{acceleration}_diffpir_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
        summary[f'R{acceleration}']={m:r['mean'] for m,r in detail.items()}
    (out/'synthetic_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def scanner(net,out,args):
    images=[]; records=[]
    for slice_id in BOXES:
        op,y,anchor,saved,traditional,meta=load_scanner_case(slice_id,args.device)
        t=time.monotonic()
        dp,trace=diffpir(net,op,y,steps=args.steps,sigma_noise=args.scanner_noise,lamb=args.scanner_lambda,seed=7)
        # rho is fixed in advance; scanner images have no GT for tuning.
        tik=op.proximal(torch.full_like(anchor,-1),y,.01)
        meta.update(diffpir_residual=float(op.relative_residual(dp,y)),
                    diffpir_displayed_residual=float(op.relative_residual(dp.clamp(-1,1),y)),
                    tikhonov_residual=float(op.relative_residual(tik,y)),seconds=time.monotonic()-t,
                    scanner_noise_assumed=args.scanner_noise,scanner_lambda=args.scanner_lambda,
                    diffusion_out_of_range_fraction=float(((dp < -1)|(dp > 1)).float().mean()))
        row={'Input':saved['input'],'Traditional PV360':traditional,'Phase + InvA':np.rot90(saved['phase_inva'],2),
             'Flow (existing)':np.rot90(saved['flow'],2),'Diffusion + DiffPIR':np.rot90(unit(dp)[0],2)}
        arrays=dict(diffpir=unit(dp)[0],diffpir_model_range=dp.cpu().numpy(),tikhonov=unit(tik)[0],observation=y.cpu().numpy())
        if args.scanner_daps:
            da=daps(net,op,y,steps=args.daps_steps,tau=args.scanner_noise,seed=7)
            row['Diffusion + DAPS']=np.rot90(unit(da)[0],2)
            arrays['daps']=unit(da)[0]
            arrays['daps_model_range']=da.cpu().numpy()
            meta['daps_residual']=float(op.relative_residual(da,y))
            meta['daps_displayed_residual']=float(op.relative_residual(da.clamp(-1,1),y))
        np.savez_compressed(out/f'scanner_slice_{slice_id}.npz',**arrays)
        (out/f'scanner_slice_{slice_id}_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
        images.append(row);records.append(meta)
        print(json.dumps(dict(event='scanner',**meta)),flush=True)
    keys=list(images[0]);fig,axes=plt.subplots(len(keys),6,figsize=(12,2*len(keys)),squeeze=False)
    for j,slice_id in enumerate(BOXES):
        x,y,w,h=BOXES[slice_id]
        for i,key in enumerate(keys):
            ax=axes[i,j];ax.imshow(images[j][key],cmap='gray',vmin=0,vmax=1)
            ax.set_xticks([]);ax.set_yticks([])
            if i>0:ax.add_patch(Rectangle((x-.5,y-.5),w,h,fill=False,edgecolor='#ff2020',lw=1.))
            if j==0:ax.set_ylabel(key,fontsize=10)
    fig.tight_layout(pad=.2)
    fig.savefig(out/'scanner_comparison_boxes.png',dpi=180)
    fig.savefig(out/'scanner_comparison_boxes.pdf')
    plt.close(fig)
    (out/'scanner_metrics.json').write_text(json.dumps(dict(note='No clean GT. Residual under estimated, fixed phase/coil model only.',cases=records),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--limit',type=int,default=24)
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--noise',type=float,default=.01)
    p.add_argument('--daps-cases',type=int,default=4)
    p.add_argument('--daps-steps',type=int,default=40)
    p.add_argument('--scanner-daps',action='store_true')
    p.add_argument('--scanner-noise',type=float,default=.02)
    p.add_argument('--scanner-lambda',type=float,default=1.)
    p.add_argument('--only',choices=['all','synthetic','scanner'],default='all')
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {args.out}')
    args.out.mkdir(parents=True,exist_ok=True)
    # Snapshot the immutable evaluated checkpoint; training can keep replacing best.pt.
    shutil.copyfile(args.checkpoint,args.out/'evaluated_checkpoint.pt')
    net,ckpt=load_prior(args.out/'evaluated_checkpoint.pt',args.device)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(checkpoint_step=ckpt['step'],checkpoint_sha256=sha256(args.out/'evaluated_checkpoint.pt'),
                  source_sha256={f:sha256(ROOT/f) for f in ('operators.py','solvers.py','evaluate.py','model.py','tiny_unet.py')},
                  upstream_inversebench_commit='1db5932bc2995504491663d1e2c498e7317c5404',
                  test_status='development evaluation; reuse in later tuning invalidates confirmatory test status')
    (args.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    if args.only in ('all','synthetic'):synthetic(net,args.out,args)
    if args.only in ('all','scanner'):scanner(net,args.out,args)
    (args.out/'completed.json').write_text(json.dumps(dict(checkpoint_step=ckpt['step'],status='complete'))+'\n')
    print('EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
