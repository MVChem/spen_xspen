"""Export additional full-FOV crossed-chirp scans using the verified VB reader.

Original .dat and existing scanner exports are read-only. Incomplete acquisitions
are recorded as failures for review, never repaired by silently reindexing frames.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
PROJECT=EXP.parents[1]
sys.path.insert(0,str(PROJECT))
TARGETS=['MID51','MID613','MID615','MID106','MID108','MID530','MID533','MID535','MID537','MID74','MID76']


def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def worker(mid):
    import prepare_scanner
    # Reuse the exact audited function, change only its derived output root.
    prepare_scanner.HERE=HERE
    entries=json.loads((EXP/'audit/siemens_header_manifest.json').read_text())
    row=next(r for r in entries if r['scan_id']==mid and r['family']=='crossed_chirp_bipolarDiff')
    prepare_scanner.export(Path(row['path']))
    import h5py
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=HERE/'scanner'/f'{mid}.h5'
    with h5py.File(source,'r') as h5:
        meta=json.loads(h5.attrs['metadata'])
        nrep,ns,coils,m,k=h5['kspace'].shape
        slices=sorted(set(np.linspace(0,ns-1,min(ns,9),dtype=int).tolist()))
        fig,axes=plt.subplots(2,len(slices),figsize=(len(slices)*2.2,5.1),squeeze=False)
        for i,rep in enumerate(sorted(set([0,nrep//2]))):
            for j,sl in enumerate(slices):
                raw=h5['kspace'][rep,sl]
                ro=np.fft.fftshift(np.fft.fft(np.fft.ifftshift(raw,axes=-1),axis=-1,norm='ortho'),axes=-1)
                im=np.sqrt(np.sum(np.abs(ro)**2,axis=0))
                assert np.isfinite(im).all() and im.max()>0
                axes[i,j].imshow(im,cmap='gray',vmin=0,vmax=np.quantile(im,.995),interpolation='nearest')
                axes[i,j].set_title(f'slice {sl}, occurrence {rep}',fontsize=9)
                axes[i,j].axis('off')
        if nrep==1:
            for ax in axes[1]:ax.axis('off')
        fig.suptitle(f'{mid}: raw RO FFT + coil RSS; acquired {m} x {k}, {ns} slices x {nrep} occurrences',fontsize=13)
        fig.text(.5,.015,'Inventory / anatomy QC only; per-frame percentile display. No PE inverse or diffusion applied.',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.03,1,.95))
        fig.savefig(HERE/'qc'/f'{mid}_raw.png',dpi=125)
        plt.close(fig)
    write_json(HERE/'qc'/f'{mid}.json',dict(scan=mid,shape=meta['shape_rep_slice_coil_pe_ro'],
               source_sha256=meta['source_sha256'],preview_slices=slices,preview_repeats=sorted(set([0,nrep//2])),
               fov_mm=meta['fov_mm'],thickness_mm=meta['thickness_mm'],r_value=meta['r_value'],
               excluded_original_slice_counters=meta['excluded_original_slice_counters']))


def main():
    for folder in ['scanner','qc','logs']:(HERE/folder).mkdir(parents=True,exist_ok=True)
    if len(sys.argv)>1:
        worker(sys.argv[1]);return
    records=[]
    for mid in TARGETS:
        log=HERE/'logs'/f'{mid}_export.log'
        start=time.time()
        with log.open('a') as handle:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),mid],stdout=handle,stderr=subprocess.STDOUT)
        records.append(dict(scan=mid,exit_code=proc.returncode,seconds=time.time()-start,log=str(log),
                            exported=(HERE/'scanner'/f'{mid}.h5').exists(),qc_ready=(HERE/'qc'/f'{mid}_raw.png').exists()))
        write_json(HERE/'export_status.json',dict(complete=False,records=records,total=len(TARGETS)))
        print(json.dumps(records[-1]),flush=True)
    write_json(HERE/'export_status.json',dict(complete=True,records=records,total=len(TARGETS)))


if __name__=='__main__':main()
