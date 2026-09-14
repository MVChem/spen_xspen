"""DDP training of a stronger, scanner-FOV-aware unconditional diffusion prior."""
import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from data import PriorData
from model import StrongPrior, sample_prior
from utils import HERE,sha256,save_checkpoint,save_grid,validate
from paths import IXI_DATA
from run_guard import acquire_run_lock


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=IXI_DATA)
    p.add_argument('--out',type=Path,default=HERE/'runs/human128')
    p.add_argument('--steps',type=int,default=50000)
    p.add_argument('--local-batch',type=int,default=16)
    p.add_argument('--base-ch',type=int,default=64)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--seed',type=int,default=19)
    p.add_argument('--backend',choices=['nccl','gloo'],default='gloo')
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--sample-every',type=int,default=5000)
    p.add_argument('--resume',type=Path)
    p.add_argument('--init-from',type=Path,help='Initialize EMA weights from a prior stage; optimizer/validation reset')
    args=p.parse_args()
    if min(args.steps,args.local_batch,args.save_every,args.sample_every)<=0:
        raise ValueError('Steps, batch and checkpoint intervals must be positive')
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    # Acquire before CUDA/DDP initialization, and retain ownership until rank 0 exits.
    run_lock=acquire_run_lock(args.out) if rank==0 else None
    local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    if world>1:
        if args.backend=='nccl':dist.init_process_group('nccl',device_id=device)
        else:dist.init_process_group('gloo')
    barrier=lambda:dist.barrier() if world>1 else None
    torch.manual_seed(args.seed)
    data=PriorData(args.data,device)
    val,val_keys=data.validation(256)
    net=StrongPrior(base_ch=args.base_ch).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=0.)
    ema=copy.deepcopy(net).eval().requires_grad_(False) if rank==0 else None
    step0=0;best=float('inf')
    manifest_hash=sha256(args.data/'manifest.json')
    if args.init_from:
        if args.resume:raise ValueError('Choose init-from OR resume')
        initial=torch.load(args.init_from,map_location=device,weights_only=False)
        if initial['model_config']!=net.config:raise ValueError('Initialization architecture mismatch')
        net.load_state_dict(initial['ema'])
        if rank==0:ema.load_state_dict(initial['ema'])
    if args.resume:
        ckpt=torch.load(args.resume,map_location=device,weights_only=False)
        if ckpt['model_config']!=net.config or ckpt['manifest_sha256']!=manifest_hash:
            raise ValueError('Resume model/data mismatch')
        if ckpt['global_batch']!=args.local_batch*world:
            raise ValueError('Exact resume requires the same global batch')
        if ckpt['total_steps']!=args.steps:
            raise ValueError('Resume must preserve the learning-rate schedule')
        net.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer'])
        if rank==0:ema.load_state_dict(ckpt['ema'])
        step0=ckpt['step'];best=ckpt['best_val']
    model=DDP(net,device_ids=[local]) if world>1 else net
    torch.manual_seed(args.seed+1009*rank)
    if args.resume:
        states=ckpt['rng_states']
        if len(states)!=world:raise ValueError('Exact resume requires the same world size')
        torch.set_rng_state(states[rank]['cpu'].cpu())
        torch.cuda.set_rng_state(states[rank]['cuda'].cpu(),device)
    if rank==0:
        args.out.mkdir(parents=True,exist_ok=True)
        if (args.out/'latest.pt').exists() and not args.resume:raise FileExistsError('Existing run')
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        config.update(world_size=world,global_batch=args.local_batch*world,parameters=sum(p.numel() for p in net.parameters()),
                      model=net.config,manifest_sha256=manifest_hash,validation_keys=val_keys,
                      data_counts=data.manifest['counts'],training_subjects=len(data.manifest['subjects']['train']),
                      objective='Unconditional EDM; no scanner-corrupted image as clean training target',
                      torch=torch.__version__)
        if args.init_from:config['initialization_sha256']=sha256(args.init_from)
        config_path=args.out/'config.json'
        if args.resume and config_path.exists():
            original=json.loads(config_path.read_text())
            if original['validation_keys']!=val_keys:raise ValueError('Resume validation protocol changed')
            with (args.out/'resume_events.jsonl').open('a') as f:
                f.write(json.dumps(dict(resumed_at=time.time(),checkpoint_step=step0,**config))+'\n')
        else:config_path.write_text(json.dumps(config,indent=2)+'\n')
        snapshot=args.out/(f'resume_source_{step0}_{int(time.time())}' if args.resume else 'source_snapshot')
        snapshot.mkdir(exist_ok=True)
        for f in ['train.py','data.py','model.py','tiny_unet.py','prepare_data.py','edm.py','utils.py','run_guard.py','paths.py']:
            shutil.copyfile(HERE/f,snapshot/f)
        print(json.dumps(dict(event='start',**{k:v for k,v in config.items() if k!='validation_keys'})),flush=True)
        if not args.resume:
            initial_loss=validate(ema,val)
            (args.out/'baseline.json').write_text(json.dumps(dict(initial_model=initial_loss))+'\n')
            with torch.random.fork_rng(devices=[local]):
                save_grid(data.sample(24),args.out/'augmented_training_examples.png',columns=6)
                save_grid(val[:24],args.out/'validation_examples.png',columns=6)
            print(json.dumps(dict(event='baseline',initial_model=initial_loss)),flush=True)
    barrier();start=time.monotonic();losses=[]
    for step in range(step0+1,args.steps+1):
        model.train();x=data.sample(args.local_batch)
        sigma=(torch.randn(len(x),1,1,1,device=device)*1.2-1.2).exp()
        noisy=x+sigma*torch.randn_like(x)
        pred=model(noisy,sigma)
        weight=(sigma.square()+net.sigma_data**2)/(sigma*net.sigma_data).square()
        loss=(weight*(pred-x).square()).mean()
        if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss: {rank}/{step}')
        lr=args.lr*min(step/500.,1.)*(.15+.85*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if rank==0:
            decay=min(.9995,(1+step)/(10+step))
            with torch.no_grad():
                for ep,np_ in zip(ema.parameters(),net.parameters()):ep.lerp_(np_,1-decay)
        losses.append(loss.detach())
        if step%50==0 or step==args.steps:
            value=torch.stack(losses).mean()
            if world>1:dist.all_reduce(value);value/=world
            if rank==0:
                row=dict(step=step,loss=float(value),grad_norm=float(grad_norm),lr=lr,
                         images_seen=step*world*args.local_batch,elapsed_sec=time.monotonic()-start,
                         steps_per_sec=(step-step0)/(time.monotonic()-start))
                print(json.dumps(row),flush=True)
                with (args.out/'train_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            losses.clear()
        if step%args.save_every==0 or step==args.steps:
            barrier()
            state=dict(cpu=torch.get_rng_state().cpu(),cuda=torch.cuda.get_rng_state(device).cpu())
            states=[None]*world
            if world>1:dist.all_gather_object(states,state)
            else:states=[state]
            if rank==0:
                vl=validate(ema,val);improved=vl<best;best=min(best,vl)
                checkpoint=dict(step=step,model_config=net.config,model=net.state_dict(),ema=ema.state_dict(),
                                optimizer=optimizer.state_dict(),val_loss=vl,best_val=best,rng_states=states,
                                manifest_sha256=manifest_hash,global_batch=world*args.local_batch)
                checkpoint['total_steps']=args.steps
                save_checkpoint(args.out/'latest.pt',checkpoint)
                if improved:
                    save_checkpoint(args.out/'best.pt',checkpoint)
                    save_checkpoint(args.out/'model_ema.pt',{k:checkpoint[k] for k in ['step','model_config','ema','val_loss','manifest_sha256']})
                row=dict(step=step,val_loss=vl,best_val=best)
                with (args.out/'val_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(dict(event='checkpoint',**row)),flush=True)
                if step%args.sample_every==0 or step==args.steps:
                    save_grid(sample_prior(ema,12,steps=64),args.out/f'samples_{step:06d}.png',columns=6)
            barrier()
    if rank==0:
        (args.out/'completed.json').write_text(json.dumps(dict(step=args.steps,best_val=best,elapsed_sec=time.monotonic()-start))+'\n')
        print('TRAINING_COMPLETE',flush=True)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
