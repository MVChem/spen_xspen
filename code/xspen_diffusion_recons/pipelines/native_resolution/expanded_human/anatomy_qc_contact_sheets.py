"""Retained anatomy audit; execute explicitly after preparing its inputs."""

def main():
    from pathlib import Path
    import json
    import h5py,numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    B=Path(__file__).resolve().parent
    for mid in ['MID51','MID613','MID615','MID106','MID108']:
     with h5py.File(B/'scanner'/f'{mid}.h5','r') as h:
      ds=h['kspace'];nr,ns,*_=ds.shape
      plans=[('all_slices',[(nr//2,s) for s in range(ns)],4,12),('selected',[(r,s) for r in [0,nr//2,nr-1] for s in [17,23,29,35]],3,4)]
      for name,entries,rows,cols in plans:
       fig,axs=plt.subplots(rows,cols,figsize=(cols*1.8,rows*1.85+0.3),squeeze=False)
       for ax,(r,s) in zip(axs.ravel(),entries):
        raw=ds[r,s]
        ro=np.fft.fftshift(np.fft.fft(np.fft.ifftshift(raw,axes=-1),axis=-1,norm='ortho'),axes=-1)
        im=np.sqrt(np.sum(abs(ro)**2,axis=0))
        ax.imshow(im,cmap='gray',vmin=0,vmax=np.quantile(im,.995),interpolation='nearest');ax.set_title(f's{s:02d} o{r:02d}',fontsize=9);ax.axis('off')
       fig.suptitle(f'{mid} raw RO FFT + RSS; {name}; geometry order, no PE inverse / DL',fontsize=12)
       fig.tight_layout(rect=(0,0,1,.96));fig.savefig(B/f'anatomy_qc_{mid}_{name}.png',dpi=120);plt.close(fig)
      print(mid,flush=True)

if __name__ == '__main__':
    main()
