"""Read-only CPU check of exact 2x grids, unchanged data and projection limits.

Print JSON to stdout; creates no checkpoint/output arrays and never uses CUDA.
"""
import json,os,sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import h5py
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;PROJECT=EXP.parents[1]
sys.path.insert(0,str(EXP))
from native_scanner import load_case

def rep(x):
 return x.repeat_interleave(2,-2).repeat_interleave(2,-1)

def error(a,b):
 return float((a-b).abs().max())

@torch.no_grad()
def main():
 torch.set_num_threads(2)
 report={'device':'cpu','model_resize':False,'real':[],'simulation':[]}
 paths=[('MID112',PROJECT/'scanner/MID112.h5','p3mm_mm1p5'),('MID27',PROJECT/'scanner/MID27.h5','p4mm_mm2'),
        ('MID613',EXP/'expanded_human/scanner/MID613.h5','p4mm_mm2')]
 for scan,path,model in paths:
  with h5py.File(path) as f:
   meta=json.loads(f.attrs['metadata']);native=tuple(f['kspace'].shape[-2:]);sl=f['kspace'].shape[1]//2
  shape=tuple(2*n for n in native)
  ckpt=torch.load(EXP/'runs'/model/'model_ema.pt',map_location='cpu',weights_only=False,mmap=True)
  assert tuple(ckpt['model_config']['image_shape'])==shape and ckpt['step']==20000
  train=json.loads((EXP/'data'/model/'manifest.json').read_text())
  op,y,base,anchor,rss,info=load_case(path,sl,0,'cpu',image_shape=None)
  high,yh,bh,ah,rh,hi=load_case(path,sl,0,'cpu',image_shape=shape)
  assert y.shape==yh.shape and high.image_shape==shape
  for a,b in [(y,yh),(op.a,high.a),(op.f,high.f),(op.coils,high.coils),(op.phase_correction,high.phase_correction)]:
   torch.testing.assert_close(a,b,rtol=0,atol=0)
  assert info['magnitude_scale']==hi['magnitude_scale'] and info['gain']==hi['gain']
  pp=torch.zeros(native[0],shape[0]);qq=torch.zeros(native[1],shape[1])
  for i in range(native[0]):pp[i,2*i:2*i+2]=.5
  for i in range(native[1]):qq[i,2*i:2*i+2]=.5
  torch.testing.assert_close(high.p,pp,rtol=0,atol=0);torch.testing.assert_close(high.q,qq,rtol=0,atol=0)
  toy=torch.linspace(-.8,.8,native[0]*native[1]).reshape(1,1,*native)
  torch.testing.assert_close(high.project(rep(toy)),toy,rtol=0,atol=0)
  equality=error(high.forward(rep(toy)),op.forward(toy))
  null=torch.ones(1,1,*shape);null[...,1::2,:]*=-1
  nullproject=float(torch.linalg.norm(high.project(null)));nullmeasurement=float(torch.linalg.norm(high.linear(null)))
  assert nullproject==nullmeasurement==0
  # Fixed x-domain rho accumulates four penalties per native cell at exact 2x.
  fine_same_native_penalty=high.proximal(torch.full_like(ah,-1),yh,.003/4)
  native_matching_default_high=op.proximal(torch.full_like(anchor,-1),y,.003*4)
  scaled_rho_error=error(fine_same_native_penalty,rep(base))
  default_rho_error=error(bh,rep(native_matching_default_high))
  torch.testing.assert_close(fine_same_native_penalty,rep(base),rtol=3e-4,atol=3e-4)
  torch.testing.assert_close(bh,rep(native_matching_default_high),rtol=3e-4,atol=3e-4)
  pixel=[f/n for f,n in zip(meta['fov_mm'],shape)]
  item=dict(scan=scan,slice=sl,native_shape=list(native),output_shape=list(shape),checkpoint=model,checkpoint_step=ckpt['step'],
            real_fov_mm=meta['fov_mm'],real_output_pixel_mm=pixel,training_fov_mm=train['fov_mm'],
            fov_fraction_shift_relative_to_training=[f/t-1 for f,t in zip(meta['fov_mm'],train['fov_mm'])],
            actual_r_value=meta['r_value'],same_measurement_operator_coils_phase_scale=True,
            input_dof=int(np.prod(shape)),projected_dof=int(np.prod(native)),guaranteed_null_fraction=.75,
            repeated_native_forward_max_abs_error=equality,null_projected_norm=nullproject,null_measurement_norm=nullmeasurement,
            rho_quarter_matches_repeated_native_max_abs=scaled_rho_error,
            default_high_rho_matches_native_fourfold_rho_max_abs=default_rho_error)
  report['real'].append(item);del ckpt
 for model in ['p3mm_mm1p5','p4mm_mm2']:
  config=json.loads((EXP/'evaluation'/model/'config.json').read_text())
  for rate in ['R1','R2']:
   p=EXP/'evaluation'/model
   metrics=json.loads((p/f'{rate}_metrics.json').read_text())
   with np.load(p/f'{rate}_arrays.npz') as a:
    truthshape=a['truth'].shape;matrixshape=a['encoding'].shape;mask=a['mask']
    assert tuple(truthshape[-2:])==tuple(2*n for n in config['profile']['native_shape'])
    fraction=float(mask.mean())
    assert fraction==(1 if rate=='R1' else .5)
   report['simulation'].append(dict(model=model,label=rate,spatial_factor=[2,2],acceleration=metrics['acceleration'],
      measured_pe_fraction=fraction,truth_shape=list(truthshape),native_encoding_shape=list(matrixshape),
      averages=metrics['averages'],case_count=len(metrics['cases'])))
 print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
