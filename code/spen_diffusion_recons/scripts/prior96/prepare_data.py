"""Expand raw rat RARE volumes while preserving the original subject split."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates, gaussian_filter
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from project_paths import CORE, PRIOR96, REFERENCE_ROOT, RAT_SPLIT, RUNS, PRIOR96_DATA, MOUSE_RAW

HERE=PRIOR96
V1=CORE
PROJECT=REFERENCE_ROOT
SPLIT=RAT_SPLIT
VOLUMES=PROJECT/'data/0428_extra_rat_data/original_data/camri_ds002870_rare'


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def normalize(x):
    x=np.maximum(np.nan_to_num(x),0).astype(np.float32)
    scale=max(float(np.quantile(x,.995)),1e-6)
    return np.clip(x/scale,0,1)


def physical_slice(plane,spacing,fov_mm=35.,size=96):
    """Resample physical FOV with antialiasing; no anatomy-dependent crop."""
    target_spacing=fov_mm/size
    sigma=[max(target_spacing/float(v)-1,0)*.5 for v in spacing]
    plane=gaussian_filter(plane.astype(np.float32),sigma=sigma)
    coords=[(np.arange(size)-(size-1)/2)*target_spacing/float(s)+(n-1)/2
            for n,s in zip(plane.shape,spacing)]
    grid=np.meshgrid(*coords,indexing='ij')
    result=map_coordinates(plane,grid,order=1,mode='constant',cval=0)
    return normalize(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=RUNS/'prepared/rat96')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'manifest.json').exists():raise FileExistsError('Dataset already exists')
    split=json.loads(SPLIT.read_text())
    subjects={part:set(split[part+'_subjects']) for part in ('train','val','test')}
    assert not any(subjects[a]&subjects[b] for a,b in [('train','val'),('train','test'),('val','test')])
    records={k:[] for k in subjects};images={k:[] for k in subjects};sources={}
    for path in sorted(VOLUMES.rglob('*.nii.gz')):
        subject=path.parts[-4]
        part=next((k for k,s in subjects.items() if subject in s),None)
        if part is None:raise ValueError(f'Unknown subject: {subject}')
        vol=nib.load(path);codes=nib.aff2axcodes(vol.affine)
        ap=next(i for i,c in enumerate(codes) if c in ('A','P'))
        remaining=[i for i in range(3) if i!=ap]
        # Canonical coronal image: inferior increases down, subject-right increases left.
        # This is the same plane convention as the original clockwise rotation.
        ri=next(i for i,c in enumerate(codes) if c in ('R','L'))
        si=next(i for i,c in enumerate(codes) if c in ('S','I'))
        arr=vol.get_fdata(dtype=np.float32)
        n=arr.shape[ap];spacing=vol.header.get_zooms()
        file_hash=sha256(path);sources[str(path)]=file_hash
        for index in range(max(0,int(n*.08)),min(n,int(np.ceil(n*.92)))):
            plane=np.take(arr,index,axis=ap)
            plane=plane.transpose(remaining.index(si),remaining.index(ri))
            if codes[si]=='S':plane=plane[::-1]
            if codes[ri]=='R':plane=plane[:,::-1]
            # FOV=35 mm matches the main 96x96 SPEN scan, not a center crop in voxel units.
            image=physical_slice(plane,(spacing[si],spacing[ri]),35.)
            if image.std()<.025 or (image>.05).mean()<.025:continue
            key=f'rat:{subject}:coronal:{index}'
            records[part].append(dict(key=key,subject=subject,species='rat',source=str(path),
                                      slice_axis=ap,slice_index=index,view='physical_fov35',fov_mm=35.,
                                      voxel_spacing_mm=list(map(float,spacing)),native_codes=list(codes)))
            images[part].append(np.round(image*65535).astype(np.uint16))
        print(json.dumps(dict(subject=subject,split=part,total=len(images[part]))),flush=True)
    # Retain original base images as a separate, lower-probability view. Flips are online.
    for part in subjects:
        for name in split[part+'_files']:
            if not name.endswith('__base.png'):continue
            path=PROJECT/'data/0428_rat/hr'/name
            with Image.open(path) as im:x=np.asarray(im.convert('L'),dtype=np.float32)/255.
            subject=next(s for s in subjects[part] if s+'_' in name)
            records[part].append(dict(key='legacy_view:'+name,subject=subject,species='rat',source=str(path),
                                      view='legacy_crop',source_sha256=sha256(path)))
            images[part].append(np.round(normalize(x)*65535).astype(np.uint16))
        tensor=np.stack(images[part])
        np.save(args.out/f'{part}.npy',tensor)
    # Byte duplicates across partitions are not permitted, including reconstructed views.
    hashes={p:{hashlib.sha256(x.tobytes()).hexdigest() for x in images[p]} for p in images}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if hashes[a]&hashes[b]:raise ValueError(f'Exact cross-split duplicates: {a}/{b}')
    manifest=dict(version=2,source_split=str(SPLIT),source_split_sha256=sha256(SPLIT),
                  records=records,volume_sha256=sources,subjects={k:sorted(v) for k,v in subjects.items()},
                  counts={k:dict(Counter(r['view'] for r in v)) for k,v in records.items()},
                  train_probability=dict(physical_fov35=.85,legacy_crop=.15),
                  note='More slices from the SAME 106 training animals; augmented views are not new subjects.',
                  normalization='q0.995 -> uint16; model range [-1,1]',
                  train_npy_sha256=sha256(args.out/'train.npy'))
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['counts']),flush=True)


if __name__=='__main__':main()
