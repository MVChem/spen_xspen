"""Render local numeric arrays. No reference-image crops are used."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps

HERE=Path(__file__).resolve().parent
ASSETS=HERE/'assets'
GENERATED=ASSETS/'generated'

def rgba(a,cmap='gray',lo=0,hi=1):
    return (colormaps[cmap](np.clip((a-lo)/(hi-lo),0,1))*255).round().astype(np.uint8)

def make_assets():
    GENERATED.mkdir(exist_ok=True)
    with np.load(ASSETS/'figure_data.npz') as z: d={k:z[k] for k in z.files}
    def save(name,a,cmap='gray',lo=0,hi=1):
        Image.fromarray(rgba(a,cmap,lo,hi)).save(GENERATED/(name+'.png'))
    for key in ['training_clean','training_noisy','training_denoised']:save(key,d[key])
    save('sampling_clean_estimate',np.rot90(d['sampling_clean_estimate'],2))
    save('reconstruction',np.rot90(d['case16_image_scanner'],2))
    save('initial_noise',d['initial_noise'],lo=-3,hi=3)
    # Complex measurements are shown by log magnitude; identical window for y and F(x).
    scale=float(np.quantile(abs(d['observation']),.995));compression=30.
    for key in ['observation','predicted']:
        view=np.log1p(compression*abs(d[key])/scale)/np.log1p(compression)
        for c in range(4):save(f'{key}_coil{c}',view[c])
    zscale=float(np.quantile(np.sqrt((abs(d['coil_images'])**2).sum(0)),.995))
    for c in range(4):
        save(f'coil_image{c}',np.rot90(abs(d['coil_images'][c])/zscale,2))
        save(f'coil_magnitude{c}',np.rot90(abs(d['coils'][c]),2))
        save(f'coil_phase{c}',np.rot90(np.angle(d['coils'][c]),2),'twilight_shifted',-np.pi,np.pi)
    a=abs(d['encoding']); save('encoding_magnitude',a,lo=0,hi=float(a.max()))
    phase=d['case16_phase_joint'];limit=float(np.ceil(np.max(abs(phase))*10)/10)
    save('residual_phase',phase,'coolwarm',-limit,limit)
    save('residual_colorbar',np.linspace(limit,-limit,256)[:,None].repeat(20,axis=1),'coolwarm',-limit,limit)
    save('coil_colorbar',np.linspace(np.pi,-np.pi,256)[:,None].repeat(20,axis=1),'twilight_shifted',-np.pi,np.pi)
    meta=dict(residual_phase_limit_rad=limit,coil_phase_range_rad=[-float(np.pi),float(np.pi)],measurement_log_compression=compression,measurement_q995_scale=scale,coil_image_rss_q995_scale=zscale,normalization='No phase smoothing or anatomy masking. Colorbars generated from the same colormaps and value ranges as the maps.')
    (GENERATED/'render_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta
if __name__=='__main__':make_assets()
