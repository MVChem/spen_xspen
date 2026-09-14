"""Fetch only individual in-vivo mouse T2/RARE MRI from public OpenNeuro S3.

Pinned git-annex hashes from local OpenNeuro metadata clones verify each file.
No EPI, FLASH, atlas, ex-vivo, or non-mouse images enter this collection.
"""
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from project_paths import RUNS, DATA_ROOT
HERE=RUNS/'prepared'
DATASETS=['ds002868','ds006663','ds005236']


def main():
    root=HERE/'mouse_raw';root.mkdir(parents=True,exist_ok=True)
    tasks=[];datasets={}
    for ds in DATASETS:
        src=DATA_ROOT/'sources'/ds
        description=json.loads((src/'dataset_description.json').read_text())
        if description.get('License')!='CC0':raise ValueError('Review unexpected dataset license')
        datasets[ds]=dict(description=description,commit=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip(),
                          subjects=len(list(src.glob('sub-*'))),source=f'https://openneuro.org/datasets/{ds}')
        files=[p for p in src.rglob('*T2w.nii.gz') if 'derivatives' not in p.parts and ('RARE' in p.name or ds=='ds005236')]
        for path in sorted(files):
            key=Path(os.readlink(path)).name
            match=re.fullmatch(r'(MD5|SHA256)E-s(\d+)--([a-f0-9]+)\.nii\.gz',key)
            if not match:raise ValueError(key)
            tasks.append(dict(dataset=ds,path=str(path.relative_to(src)),algorithm=match[1].lower(),
                              size=int(match[2]),expected_digest=match[3]))
        for f in src.rglob('*'):
            if f.is_symlink() or '.git' in f.parts or not f.is_file():continue
            if f.suffix in ('.json','.tsv') or f.name.startswith(('README','CHANGES')):
                target=root/ds/f.relative_to(src);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    def fetch(task):
        path=root/task['dataset']/task['path'];path.parent.mkdir(parents=True,exist_ok=True)
        url='https://s3.amazonaws.com/openneuro.org/'+urllib.parse.quote(task['dataset']+'/'+task['path'])
        if not path.exists():
            for attempt in range(4):
                try:
                    temp=path.with_suffix('.part')
                    with urllib.request.urlopen(url,timeout=90) as response,open(temp,'wb') as f:
                        shutil.copyfileobj(response,f)
                    os.replace(temp,path);break
                except Exception:
                    if attempt==3:raise
                    time.sleep(attempt+1)
        raw=path.read_bytes();digest=hashlib.new(task['algorithm'],raw).hexdigest()
        if len(raw)!=task['size'] or digest!=task['expected_digest']:
            raise ValueError(f'Annex checksum mismatch: {path}')
        return dict(**task,url=url,local_path=str(path),sha256=hashlib.sha256(raw).hexdigest(),verified=True)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for result in pool.map(fetch,tasks):
            results.append(result)
            if len(results)%20==0:print(json.dumps(dict(downloaded=len(results),total=len(tasks))),flush=True)
    manifest=dict(datasets=datasets,files=results,total_bytes=sum(r['size'] for r in results),
                  subjects=sum(v['subjects'] for v in datasets.values()),volumes=len(results),
                  selection='in vivo mouse RARE/T2 individual magnitude volumes, not coil-resolved raw k-space')
    (root/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('datasets','files')}),flush=True)


if __name__=='__main__':main()
