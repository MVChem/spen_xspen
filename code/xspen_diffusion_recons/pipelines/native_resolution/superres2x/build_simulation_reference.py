"""CPU-only interpolation reference from frozen R1 spatial-2x test simulations.

No model inference, retraining or test-set parameter selection. Native L2 rho is
fixed at 4 times the already validation-selected high-grid rho; a fixed .003
native reference is reported separately, never selected by test scores.
"""
import hashlib,json,os,sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;PROJECT=EXP.parents[1]
sys.path.insert(0,str(PROJECT))
from operators import XSPENOperator
from evaluate import metrics
from utils import sha256


def means(rows):
 return {key:float(np.mean([r[key] for r in rows])) for key in ('psnr','ssim')}

@torch.no_grad()
def main():
 torch.set_num_threads(2)
 report=dict(device='cpu',spatial_factor=[2,2],acceleration=1,network_inference_run=False,test_parameter_tuning=False,
             primary_baseline='Native-grid real-magnitude L2/CG, rho=4*existing validation-selected 2x rho, bicubic 2x, align_corners=False, antialias=True',
             extra_baseline='Native-grid real-magnitude L2/CG with fixed rho=.003, same bicubic; not selected by test scores',
             image_metric='Existing evaluate.metrics: shared [-1,1] model units converted to [0,1], clipping only for metric; no fitted gain.',
             sources={str(Path(__file__)):sha256(__file__),str(PROJECT/'operators.py'):sha256(PROJECT/'operators.py'),str(PROJECT/'evaluate.py'):sha256(PROJECT/'evaluate.py')},protocols=[])
 plot_cases=[];all_subjects=set()
 for model in ['p3mm_mm1p5','p4mm_mm2']:
  root=EXP/'evaluation'/model
  config=json.loads((root/'config.json').read_text());selection=json.loads((root/'R1_selection.json').read_text())
  old=json.loads((root/'R1_metrics.json').read_text());manifest=json.loads((EXP/'data'/model/'manifest.json').read_text())
  for left,right in [('train','val'),('train','test'),('val','test')]:assert not set(manifest['subjects'][left])&set(manifest['subjects'][right])
  subjects=[row['key'].split('-')[0] for row in old['cases']]
  valsubjects={key.split('-')[0] for key in selection['validation_keys']}
  assert len(set(subjects))==12 and set(subjects)<=set(manifest['subjects']['test']) and not set(subjects)&valsubjects
  all_subjects.update(subjects)
  with np.load(root/'R1_arrays.npz') as saved:
   gt=torch.from_numpy(saved['truth']);y=torch.from_numpy(saved['measurement'])
   a=torch.from_numpy(saved['encoding']);f=torch.from_numpy(saved['readout']);coils=torch.from_numpy(saved['coils']);mask=torch.from_numpy(saved['mask'])
   edm=torch.from_numpy(saved['diffusion']);old128=torch.from_numpy(saved['baseline128']);tikh2x=torch.from_numpy(saved['tikhonov'])
  shape=tuple(gt.shape[-2:]);native=(a.shape[1],f.shape[1]);assert shape==tuple(2*n for n in native) and torch.all(mask==1)
  op=XSPENOperator(a,f,coils,mask=mask)
  initial=torch.full((len(gt),1,*native),-1.,dtype=gt.dtype)
  selected_high_rho=selection['selected']['tikhonov_rho'];native_rho=4*selected_high_rho
  native_l2=op.proximal(initial,y,native_rho)
  fixed_native=op.proximal(initial,y,.003)
  repeat=native_l2.repeat_interleave(2,-2).repeat_interleave(2,-1)
  equivalence_error=float((repeat-tikh2x).abs().max())
  torch.testing.assert_close(repeat,tikh2x,rtol=3e-4,atol=3e-4)
  interp=lambda x:F.interpolate(x,size=shape,mode='bicubic',align_corners=False,antialias=True)
  baseline=interp(native_l2);fixed_baseline=interp(fixed_native)
  scores={name:metrics(value,gt) for name,value in [('native_l2_bicubic',baseline),('native_l2_fixed003_bicubic',fixed_baseline),('existing_edm2x',edm),('existing_old128',old128),('existing_tikhonov2x',tikh2x)]}
  for i,r in enumerate(scores['existing_edm2x']):
   np.testing.assert_allclose([r['psnr'],r['ssim']],[old['cases'][i]['diffusion']['psnr'],old['cases'][i]['diffusion']['ssim']],rtol=0,atol=1e-6)
  cases=[]
  for i,record in enumerate(old['cases']):
   cases.append(dict(key=record['key'],subject=subjects[i],**{name:values[i] for name,values in scores.items()},
                     edm_minus_bicubic={metric:scores['existing_edm2x'][i][metric]-scores['native_l2_bicubic'][i][metric] for metric in ('psnr','ssim')}))
  averages={name:means(value) for name,value in scores.items()}
  entry=dict(model=model,source_arrays=str(root/'R1_arrays.npz'),source_arrays_sha256=sha256(root/'R1_arrays.npz'),
             selection_sha256=sha256(root/'R1_selection.json'),checkpoint_sha256=config['checkpoint_sha256'],checkpoint_step=config['checkpoint_step'],
             case_count=12,distinct_test_subjects=subjects,subject_split_verified=True,native_shape=list(native),output_shape=list(shape),
             original_validation_selection=selection['selected'],native_rho=native_rho,fixed_alternative_native_rho=.003,
             bicubic_settings=dict(mode='bicubic',align_corners=False,antialias=True),
             cpu_native_repeated_vs_saved_2x_tikhonov_max_abs=equivalence_error,
             existing_edm_settings=dict(steps=config['config']['steps'],sigma_noise=.01,lamb=selection['selected']['diffusion'],xi=0,seed=20260913),
             real_batch_settings=dict(steps=60,sigma_noise=.02,lamb=1,xi=0,seed=20260913),
             averages=averages,cases=cases,
             mean_edm_minus_bicubic={metric:averages['existing_edm2x'][metric]-averages['native_l2_bicubic'][metric] for metric in ('psnr','ssim')},
             cases_edm_better_than_bicubic={metric:sum(x['edm_minus_bicubic'][metric]>0 for x in cases) for metric in ('psnr','ssim')})
  report['protocols'].append(entry)
  for i in [0,1]:
   plot_cases.append(dict(model=model,key=old['cases'][i]['key'],images=[gt[i,0].numpy(),baseline[i,0].numpy(),edm[i,0].numpy()],
                          scores=[None,scores['native_l2_bicubic'][i],scores['existing_edm2x'][i]],fov=config['profile']['fov_mm']))
  print(json.dumps(dict(event='simulation_reference_complete',model=model,averages=averages,delta=entry['mean_edm_minus_bicubic'])),flush=True)
 report['unique_subjects_across_protocols']=len(all_subjects)
 report['limitations']='Same held-out subjects reused across two geometry settings, not 24 independent people. Matched reduced-model simulations with analytic coils/known phase; interpolation comparison does not establish real acquired spatial resolution. Existing EDM lambda=.3/noise=.01 differs from planned real fixed lambda=1/noise=.02.'
 (HERE/'simulation_reference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 fig,axes=plt.subplots(len(plot_cases),3,figsize=(10,3*len(plot_cases)+1),squeeze=False)
 for i,item in enumerate(plot_cases):
  height,width=item['fov']
  for j,(x,score) in enumerate(zip(item['images'],item['scores'])):
   ax=axes[i,j];ax.imshow((x+1)/2,cmap='gray',vmin=0,vmax=1,interpolation='nearest',extent=(0,width,height,0),aspect='equal');ax.set_xticks([]);ax.set_yticks([])
   if i==0:ax.set_title(['GT at spatial 2x grid','Native L2 + bicubic 2x','Existing EMA EDM + DiffPIR 2x'][j],fontsize=10)
   if j==0:ax.set_ylabel(item['model']+'\n'+item['key'],fontsize=8)
   if score:ax.text(.02,.98,f"{score['psnr']:.2f} dB / {score['ssim']:.3f}",transform=ax.transAxes,va='top',color='white',fontsize=9,bbox=dict(facecolor='black',alpha=.7,pad=2))
 fig.suptitle('Held-out matched simulation | full PE (R1) | spatial 2x in both axes',fontsize=12)
 fig.text(.5,.014,'Fixed first two test cases per protocol; shared [0,1] window; native L2 rho from existing validation selection.\nStored EDM: lambda=.3, noise=.01; this is not the real-data fixed-parameter reconstruction.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.05,1,.96));fig.savefig(HERE/'simulation_reference.png',dpi=160);plt.close(fig)
 lines=['# 空间 2× 与单纯插值的受试者留出仿真对照','',
 '本次只读已有 R1 仿真的 GT、复数测量和 2× diffusion 输出，在 CPU 上重新求原生网格幅度 L2/CG 并 bicubic 放大。没有训练、没有网络推理，没有用 test 图挑参数。R1 表示 PE 全采；两种协议均为长宽各两倍的空间输出。','',
 '主基线采用已有独立 validation 选中的高网格 rho=.0003，按精确2×的四倍关系换为 native rho=.0012；再 bicubic(scale2, align_corners=False, antialias=True)。另列固定 native rho=.003 的结果，不根据 test 分数选用其中较好者。','',
 '| 协议 | 原生L2＋bicubic PSNR / SSIM | 固定rho=.003＋bicubic | 已存2×EDM | EDM相对主插值基线 |',
 '|---|---|---|---|---|']
 for p in report['protocols']:
  a=p['averages'];b=a['native_l2_bicubic'];f=a['native_l2_fixed003_bicubic'];d=a['existing_edm2x'];delta=p['mean_edm_minus_bicubic']
  lines.append(f"| {p['model']} | {b['psnr']:.3f} / {b['ssim']:.4f} | {f['psnr']:.3f} / {f['ssim']:.4f} | {d['psnr']:.3f} / {d['ssim']:.4f} | {delta['psnr']:+.3f} dB / {delta['ssim']:+.4f} |")
 lines+=['','各协议12个不同 test 受试者，与 train/validation 的主体划分无交叉；两协议复用了相同的12个 test 受试者，因此不是24个独立人。所有已有 diffusion 指标均从存储数组重新计算并与旧报告一致。原生解的2×2复制也与旧高网格L2结果在浮点容差内一致，确认了纯L2与插值基线的关联。','',
 '这些结果检验了“学习先验参与高网格重建”相对“原生 L2 解的普通 bicubic 放大”在匹配仿真中的增益；不代表每个真实细节都被测量恢复。当前2×投影有75%零空间，解析coil/已知phase和仿真GT也比真实校准条件理想。','',
 '特别是已有模拟 diffusion 采用 validation 选 lambda=.3、sigma_noise=.01、60步；新的真实批次固定 lambda=1、sigma_noise=.02、60步。不能把该仿真提升当成真实固定参数已获得同等提升的证据。真实无干净 GT，不报告其 PSNR/SSIM。','',
 '示例图固定取各协议 test 顺序的前两例，未按效果挑选。完整逐例分数与来源 hash：[simulation_reference.json](simulation_reference.json)；图：[simulation_reference.png](simulation_reference.png)。']
 (HERE/'simulation_reference.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
