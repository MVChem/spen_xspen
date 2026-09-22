"""Extract figure arrays from this repository's actual MRI data and saved run.

Uses CPU only. Rebuilding the PPT later needs only the exported local NPZ,
not the workspace data or checkpoint. Run with --workspace PATH.
"""
from pathlib import Path
import argparse,sys,json,hashlib
import numpy as np

HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args(); root=a.workspace.resolve()
    code=root/'code/spen_diffusion_recons'
    for sub in ['scripts/core','scripts/prior96','scripts/prior192']:sys.path.insert(0,str(code/sub))
    import torch
    from evaluate_real_sr import load_case
    from model_v2 import load_strong_prior
    torch.set_num_threads(3)
    dataset=root/'code/data/rodent192_training_260915/prepared'
    manifest=json.loads((dataset/'manifest.json').read_text())['records']['train']
    source=np.load(dataset/'train.npy',mmap_mode='r'); index=823
    clean=source[index].astype(np.float32)/65535
    checkpoint=code/'runs/rodent192_spen2x_260914/train/model_ema.pt'
    net,ckpt=load_strong_prior(checkpoint,'cpu')
    generator=torch.Generator().manual_seed(20260922)
    sigma=.35
    noisy=(2*torch.from_numpy(clean)-1)[None,None]+sigma*torch.randn((1,1,192,192),generator=generator)
    with torch.no_grad(): denoised=((net(noisy,sigma)+1)/2)[0,0].numpy()
    arrays={'training_clean':clean,'training_noisy':((noisy+1)/2)[0,0].numpy(),'training_denoised':denoised}
    cases=[]
    run=code/'runs/joint_phase_diffpir_260916/gpu1_real_residual'
    for fov,export,scan in [(16,22,'20240321_lxj_spen_mouse_240321_1_1_1'),(24,11,'20240115_lxj_SPEN_96_240115_1_1_1')]:
        mat=root/f'code/data/spen_acquired_260915/mat/{scan}/slice_{export}.mat'
        op96,op,y,anchor,meta=load_case(mat,'cpu')
        saved=run/f'real_residual_fov{fov}_export{export}.npz'
        with np.load(saved) as z: d={k:z[k].copy() for k in z.files}
        assert np.allclose(y[0].numpy(),d['observation'],rtol=3e-5,atol=1e-7)
        case={'fov':fov,'export':export,'mat':str(mat.relative_to(root)),'mat_sha256':sha(mat),'saved_run':str(saved.relative_to(root)),'saved_run_sha256':sha(saved),'metadata':meta,'phase_range_rad':[float(d['phase_joint'].min()),float(d['phase_joint'].max())]}
        cases.append(case)
        for key in ['anchor','image_scanner','image_joint','phase_joint']:arrays[f'case{fov}_{key}']=d[key]
        if fov==16:
            coils=op.coils[0].detach().numpy(); m=d['image_scanner']
            with torch.no_grad():
                predicted=op.forward(torch.from_numpy(m*2-1)[None,None])[0].numpy()
                # Representative denoiser evaluation on the stored baseline reconstruction.
                # Not claimed to be a recovered intermediate of the historical sampler.
                illustrative=((net(torch.from_numpy(m*2-1)[None,None],.02)+1)/2)[0,0].numpy()
            arrays.update(observation=d['observation'],coils=coils,encoding=op.a_full.detach().numpy(),predicted=predicted,sampling_clean_estimate=illustrative,initial_noise=np.random.default_rng(73).standard_normal((192,192)).astype(np.float32))
            # Native InvA complex images exactly from z/RSS and saved native RSS.
            # Use native operator's coils rather than the 192 interpolated factors.
            arrays['coil_images']=op96.coils[0].numpy()*((anchor[0,0].numpy()+1)/2)
    dest=HERE/'assets';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'figure_data.npz',**arrays)
    provenance=dict(training_record=manifest[index],training_array_index=index,training_sigma=sigma,training_noise_seed=20260922,checkpoint=str(checkpoint.relative_to(root)),checkpoint_sha256=sha(checkpoint),checkpoint_step=ckpt['step'],cases=cases,selected_case='FOV16 export22',baseline='image_scanner: saved 60-step diffusion with fixed scanner correction and fixed coil factors',coil_phase='angle of exactly the 192x192 complex coil fractions used by the operator; not independent sensitivity calibration',residual_phase='phase_joint from saved corrected-input joint phase run, shape 48x96; acquired-even row by RO; no anatomical mask',sampling_clean_estimate='Additional single CPU evaluation D_theta(2*image_scanner-1, sigma=0.02), illustrative; not a saved sampler intermediate',orientation='Real image-space thumbnails rotate180 for display; observation, predicted data, encoding and residual phase retain native acquisition array orientation',magnitude_window=[0,1],phase_colormap='coil: twilight_shifted [-pi,pi]; residual: coolwarm with symmetric range rounded up to nearest 0.1 rad',note='No original reference-image crop contributes to the rebuilt figure. Training noise and initialization noise are reproducible synthetic Gaussian arrays.')
    (dest/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    print('Exported local arrays, checkpoint forward passes and exact coil/phase inputs',flush=True)
    print([(c['fov'],c['phase_range_rad']) for c in cases],flush=True)
if __name__=='__main__':main()
