"""Combine verified public mouse structural MRI with subject-separated rat support."""
import json
import re
from pathlib import Path
from collections import Counter
import numpy as np
import nibabel as nib
from prepare_data import HERE,physical_slice,sha256
from project_paths import RUNS, MOUSE_RAW


def main():
    out=RUNS/'prepared/mouse96';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Mixed dataset already exists')
    raw=json.loads((MOUSE_RAW/'download_manifest.json').read_text())
    old=json.loads((RUNS/'prepared/rat96/manifest.json').read_text())
    records={k:[] for k in ('train','val','test')};images={k:[] for k in records}
    assignments={};rng=np.random.default_rng(20260908)
    for ds in raw['datasets']:
        ids=sorted({r['path'].split('/')[0] for r in raw['files'] if r['dataset']==ds})
        group=lambda s:re.match(r'sub-(COMR\d+)',s).group(1) if ds=='ds006663' else s
        groups=sorted({group(s) for s in ids});rng.shuffle(groups)
        n=max(2,round(len(groups)*.10))
        grouped={g:('test' if i<n else 'val' if i<2*n else 'train') for i,g in enumerate(groups)}
        assignments.update({ds+':'+s:dict(part=grouped[group(s)],group=ds+':'+group(s)) for s in ids})
    volume_info=[]
    for file in raw['files']:
        path=Path(file['local_path']);ds=file['dataset'];subject=ds+':'+file['path'].split('/')[0]
        assignment=assignments[subject];part=assignment['part']
        vol=nib.load(path);arr=vol.get_fdata(dtype=np.float32).squeeze()
        if arr.ndim!=3:raise ValueError(f'Not a 3D structural volume: {path} {arr.shape}')
        codes=nib.aff2axcodes(vol.affine);zooms=vol.header.get_zooms()[:3]
        if any(float(x)>2 for x in zooms):raise ValueError(f'Review physical voxel size for {path}: {zooms}')
        ap=next(i for i,c in enumerate(codes) if c in ('A','P'))
        ri=next(i for i,c in enumerate(codes) if c in ('R','L'))
        si=next(i for i,c in enumerate(codes) if c in ('S','I'))
        remaining=[i for i in range(3) if i!=ap];n=arr.shape[ap]
        # Avoid redundant interpolation across thick original slices: use acquired AP planes.
        count=0
        for index in range(max(0,int(n*.08)),min(n,int(np.ceil(n*.92)))):
            plane=np.take(arr,index,axis=ap).transpose(remaining.index(si),remaining.index(ri))
            if codes[si]=='S':plane=plane[::-1]
            if codes[ri]=='R':plane=plane[:,::-1]
            for fov in (16.,24.):
                im=physical_slice(plane,(zooms[si],zooms[ri]),fov)
                if im.std()<.03 or (im>.1).mean()<.035:continue
                records[part].append(dict(key=f'{ds}:{path.name}:{index}:fov{fov:g}',subject=subject,
                    split_group=assignment['group'],species='mouse',dataset=ds,source=str(path),source_sha256=file['sha256'],
                    slice_axis=ap,slice_index=index,view=f'mouse_fov{fov:g}',fov_mm=fov,
                    original_shape=list(vol.shape),original_spacing_mm=list(map(float,zooms))))
                images[part].append(np.round(im*65535).astype(np.uint16));count+=1
        volume_info.append(dict(dataset=ds,subject=subject,file=str(path),shape=list(vol.shape),codes=list(codes),
                                voxel_spacing_mm=list(map(float,zooms)),accepted_views=count,split=part))
    for part in records:
        rats=np.load(HERE/f'dataset/{part}.npy')
        for r,im in zip(old['records'][part],rats):
            r=dict(r,subject='ds002870:'+r['subject'],split_group='ds002870:'+r['subject'],dataset='ds002870')
            records[part].append(r);images[part].append(im)
    # Equal total probability per animal WITHIN species, 80% mouse and 20% rat.
    subjects={species:{r['subject'] for r in records['train'] if r['species']==species} for species in ('mouse','rat')}
    nviews=Counter((r['subject'],r['view']) for r in records['train'])
    view_prob={'mouse_fov16':.60,'mouse_fov24':.40,'physical_fov35':.85,'legacy_crop':.15}
    for r in records['train']:
        species_prob=.8 if r['species']=='mouse' else .2
        r['sample_weight']=species_prob/len(subjects[r['species']])*view_prob[r['view']]/nviews[(r['subject'],r['view'])]
    hashes={}
    for part in records:
        tensor=np.stack(images[part]);np.save(out/f'{part}.npy',tensor)
        hashes[part]={sha256_bytes(x.tobytes()) for x in tensor}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if hashes[a]&hashes[b]:raise ValueError(f'Exact image overlap: {a}/{b}')
        assert not ({r['subject'] for r in records[a]}&{r['subject'] for r in records[b]})
        assert not ({r['split_group'] for r in records[a]}&{r['split_group'] for r in records[b]})
    manifest=dict(version=3,records=records,mouse_download_manifest_sha256=sha256(MOUSE_RAW/'download_manifest.json'),
        rat_manifest_sha256=sha256(RUNS/'prepared/rat96/manifest.json'),assignments=assignments,volume_info=volume_info,
        subjects={k:sorted({r['subject'] for r in v}) for k,v in records.items()},
        counts={k:dict(Counter(r['view'] for r in v)) for k,v in records.items()},
        species_subject_counts={k:dict(Counter(s.split(':')[0] for s in {r['subject'] for r in v})) for k,v in records.items()},
        train_probability={'mouse':.8,'rat':.2},
        note='Longitudinal sessions stay together; ds006663 siblings stay in the same split. FOV views are not new animals.',
        train_npy_sha256=sha256(out/'train.npy'))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(counts=manifest['counts'],subjects=manifest['species_subject_counts'])),flush=True)


def sha256_bytes(b):
    import hashlib
    return hashlib.sha256(b).hexdigest()


if __name__=='__main__':main()
