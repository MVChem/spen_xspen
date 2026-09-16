"""Subject-balanced clean-prior batches with conservative MRI appearance variation."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


class PriorData:
    def __init__(self,root,device='cuda',protocol='mouse'):
        if protocol not in ('mouse','rat_pretrain'):
            raise ValueError('Unknown data protocol')
        self.root=Path(root);self.device=device;self.protocol=protocol
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.arrays={p:torch.from_numpy(np.load(self.root/f'{p}.npy').astype(np.float32)/65535.)[:,None].to(device)
                     for p in ('train','val')}
        records=self.manifest['records']['train']
        counts={}
        for r in records:
            key=(r['subject'],r['view']);counts[key]=counts.get(key,0)+1
        probabilities=self.manifest['train_probability']
        weights=[r['sample_weight'] if 'sample_weight' in r else probabilities[r['view']]/counts[(r['subject'],r['view'])] for r in records]
        self.weights=torch.tensor(weights,device=device,dtype=torch.float32)

    def sample(self,batch,augment=True):
        ids=torch.multinomial(self.weights,batch,replacement=True)
        x=self.arrays['train'][ids]
        if augment:x=augment_magnitude(x,rotate180=self.protocol=='mouse')
        return x*2-1

    def validation(self,limit=256):
        records=self.manifest['records']['val']
        if self.protocol=='rat_pretrain':
            # The initial 5k-step run predates mouse-balanced validation/rotation.
            choices=[i for i,r in enumerate(records) if r['view']=='physical_fov35']
            ids=np.linspace(0,len(choices)-1,min(limit,len(choices)),dtype=int)
            selected=[choices[i] for i in ids]
            return self.arrays['val'][selected]*2-1,[records[i]['key'] for i in selected]
        subjects=sorted({r['subject'] for r in records if r['species']=='mouse'})
        if not subjects:subjects=sorted({r['subject'] for r in records})
        selected=[]
        each=max(1,limit//len(subjects))
        for subject in subjects:
            choices=[i for i,r in enumerate(records) if r['subject']==subject and r['view'] in ('mouse_fov16','physical_fov35')]
            ids=np.linspace(0,len(choices)-1,min(each,len(choices)),dtype=int)
            selected.extend(choices[i] for i in ids)
        x=self.arrays['val'][selected].clone()
        x[::2]=torch.rot90(x[::2],2,(-2,-1))
        return x*2-1,[records[i]['key']+(':rot180' if j%2==0 else '') for j,i in enumerate(selected)]


def augment_magnitude(x,rotate180=True):
    """Clean anatomy/contrast augmentation, NOT scanner ghost/noise as clean targets."""
    b=len(x);device=x.device
    # Match both display and native SPEN orientations; native real images may be rot180.
    if rotate180:
        rotate=torch.rand(b,1,1,1,device=device)<.5
        x=torch.where(rotate,torch.rot90(x,2,(-2,-1)),x)
    angle=(torch.rand(b,device=device)*2-1)*(.12) # about +/-7 degrees
    scale=torch.empty(b,device=device).uniform_(.88,1.12)
    flip=torch.where(torch.rand(b,device=device)<.5,-1.,1.)
    theta=torch.zeros(b,2,3,device=device)
    theta[:,0,0]=angle.cos()*scale*flip;theta[:,0,1]=-angle.sin()*scale
    theta[:,1,0]=angle.sin()*scale*flip;theta[:,1,1]=angle.cos()*scale
    theta[:,:,2]=torch.empty(b,2,device=device).uniform_(-.14,.14)
    grid=F.affine_grid(theta,x.shape,align_corners=False)
    out=F.grid_sample(x,grid,align_corners=False,padding_mode='zeros')
    gamma=torch.empty(b,1,1,1,device=device).uniform_(.85,1.15)
    gain=torch.empty(b,1,1,1,device=device).uniform_(.90,1.08)
    field=F.interpolate(torch.randn(b,1,4,4,device=device)*.08,size=(96,96),mode='bicubic',align_corners=False).exp()
    return (out.clamp_min(0).pow(gamma)*gain*field).clamp(0,1)
