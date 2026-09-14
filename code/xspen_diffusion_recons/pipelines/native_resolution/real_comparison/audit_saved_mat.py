"""Retained analysis recipe; execute explicitly after supplying its input artifacts."""

def main():
    import json,hashlib,gc
    from pathlib import Path
    import scipy.io as sio
    import h5py,numpy as np
    out=Path(__file__).resolve().parent
    project=out.parents[2]
    old=json.loads((project/'runs/visual_review_20260909/real_extended/real_metrics.json').read_text())['cases']
    report=[]
    def sha256(path):
     h=hashlib.sha256()
     with Path(path).open('rb') as f:
      for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
     return h.hexdigest()
    def corr(a,b):
     a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel();a=a-a.mean();b=b-b.mean();return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
    for mid in ['MID112','MID114','MID27']:
     meta=json.loads((project/'scanner'/f'{mid}.json').read_text());path=Path(meta['source']).with_suffix('.mat')
     fields=sio.whosmat(path);data=sio.loadmat(path,variable_names=['Img','CmplxData'])
     im=data['Img'][:,:,0,:];c=np.squeeze(data['CmplxData']) if 'CmplxData' in data else None
     ns=im.shape[-1]//meta['repeated_counter_occurrences'];nr=meta['repeated_counter_occurrences'];records=[]
     print(mid,'loaded',im.shape,None if c is None else c.shape,flush=True)
     with h5py.File(project/'scanner'/f'{mid}.h5','r') as h:
      for entry in [e for e in old if e['scan']==mid]:
       sl,rep=entry['slice_index'],entry['repeat'];counter=meta['slice_order'][sl]
       orig=2*(counter-ns//2) if counter>=ns//2 else 2*counter+1;frame=rep*ns+orig
       a=h['kspace'][rep,sl].transpose(1,2,0)
       width=a.shape[1];w=np.exp(-.001*(np.arange(1,width+1)-width/2)**2)
       filtered=a*w[None,:,None]
       fft=lambda z: np.fft.fftshift(np.fft.fft(np.fft.ifftshift(z,axes=1),axis=1),axes=1)
       rss=np.sqrt(np.square(np.abs(fft(a))).sum(2))
       wrss=np.sqrt(np.square(np.abs(fft(filtered))).sum(2))
       targets=[im[:,:,r*ns+orig] for r in range(nr)]
       scores=[corr(wrss,v) for v in targets]
       item=dict(case=entry['case'],repeat=rep,slice_index=sl,original_slice_counter=counter,mat_slice_zero_based=orig,
        mat_frame_zero_based=frame,mapping_matches_old=frame==entry['original_mat_frame_zero_based'],
        raw_rss_correlation=corr(rss,im[:,:,frame]),windowed_raw_rss_correlation=scores[rep],
        best_windowed_rss_repeat=int(np.argmax(scores)),windowed_rss_repeat_scores=scores)
       if c is not None:
        ci=c[:,:,:,orig,rep]
        expected=np.sqrt(np.square(np.abs(fft(ci))).sum(2))
        item['img_from_cmplxdata_relative_error']=float(np.linalg.norm(expected-im[:,:,frame])/np.linalg.norm(im[:,:,frame]))
        cosine=[]
        for r in range(nr):
         target=c[:,:,:,orig,r]
         cosine.append(float(abs(np.vdot(filtered,target))/(np.linalg.norm(filtered)*np.linalg.norm(target))))
        item['complex_windowed_raw_cosine']=cosine[rep];item['best_complex_repeat']=int(np.argmax(cosine));item['complex_repeat_cosines']=cosine
       records.append(item)
      print(mid,'completed',len(records),'windowcorr',min(x['windowed_raw_rss_correlation'] for x in records),'repeatmatches',sum(x['best_windowed_rss_repeat']==x['repeat'] for x in records),flush=True)
     report.append(dict(scan=mid,mat_path=str(path),mat_sha256=sha256(path),mat_bytes=path.stat().st_size,fields=fields,original_slices=ns,repeats=nr,cases=records,
      all_same_frame_as_old=all(x['mapping_matches_old'] for x in records),
      all_expected_repeat_best_windowed_rss=all(x['best_windowed_rss_repeat']==x['repeat'] for x in records),
      all_expected_repeat_best_complex=None if c is None else all(x['best_complex_repeat']==x['repeat'] for x in records)))
     del data,im,c;gc.collect()
    (out/'original_mat_alignment_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('saved',flush=True)

if __name__ == '__main__':
    main()
