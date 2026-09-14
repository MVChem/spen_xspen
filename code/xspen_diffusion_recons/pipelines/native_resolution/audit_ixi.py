"""Read-only local IXI inventory; reads gzip NIfTI headers, never expands archives."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    import os
    source = Path(os.environ.get('XSPEN_IXI_SOURCE', str(PROJECT/'data/raw/ixi'))).expanduser().resolve()
    cache = PROJECT / 'data/ixi128'
    manifest = json.loads((cache / 'manifest.json').read_text())
    files = sorted(source.rglob('*.nii.gz'))
    modalities = defaultdict(list)
    failures = []
    for path in files:
        modality = re.search(r'-(T1|T2|PD|MRA|DTI)(?:-|\.)', path.name)[1]
        try:
            with gzip.open(path, 'rb') as f:
                header = f.read(352)
            endian = '<' if struct.unpack('<i', header[:4])[0] == 348 else '>'
            if struct.unpack(endian+'i', header[:4])[0] != 348:
                raise ValueError('Not a NIfTI-1 header')
            dim = struct.unpack(endian+'8h', header[40:56])
            pix = struct.unpack(endian+'8f', header[76:108])
            modalities[modality].append(dict(path=str(path), subject=path.name.split('-')[0],
                shape=list(dim[1:dim[0]+1]), spacing_mm=[round(x,5) for x in pix[1:4]],
                bytes=path.stat().st_size))
        except Exception as e:
            failures.append(dict(path=str(path), error=str(e)))
    raw = {}
    for modality, rows in modalities.items():
        raw[modality] = dict(files=len(rows), subjects=len({r['subject'] for r in rows}),
            compressed_bytes=sum(r['bytes'] for r in rows),
            header_shapes=dict(Counter('x'.join(map(str,r['shape'])) for r in rows)),
            header_spacing_mm=dict(Counter('x'.join(map(str,r['spacing_mm'])) for r in rows)),
            representative=rows[0],
            within_frozen_split={sp:len({r['subject'] for r in rows}&set(ids))
                                 for sp,ids in manifest['subjects'].items()},
            subjects_outside_frozen_split=sorted({r['subject'] for r in rows}-
                {sid for ids in manifest['subjects'].values() for sid in ids}))
    split_sets = {sp:set(ids) for sp,ids in manifest['subjects'].items()}
    split_audit = {}
    hashes = {}
    source_splits = defaultdict(set)
    slice_splits = defaultdict(set)
    invalid_records = []
    for v in manifest['volumes']:
        source_splits[v['sha256']].add(v['split'])
    for sp, records in manifest['records'].items():
        a=np.load(cache/f'{sp}.npy', mmap_mode='r')
        actual = sha256(cache/f'{sp}.npy')
        hashes[sp] = dict(expected=manifest['array_sha256'][sp], actual=actual,
                          match=actual == manifest['array_sha256'][sp])
        for row in records:
            slice_splits[row['slice_sha256']].add(sp)
            if row['subject'] not in split_sets[sp]:
                invalid_records.append(row['key'])
        sample_ids = np.unique(np.linspace(0, len(a)-1, min(64,len(a))).round().astype(int))
        sampled_hashes = all(hashlib.sha256(a[i].tobytes()).hexdigest() == records[i]['slice_sha256'] for i in sample_ids)
        split_audit[sp]=dict(subjects=len(split_sets[sp]), slices=len(records), array_shape=list(a.shape),
            dtype=str(a.dtype), array_count_matches=len(a)==len(records),
            modality_slices=dict(Counter(r['modality'] for r in records)),
            view_slices=dict(Counter(r['view'] for r in records)),
            sampled_row_hashes_match=sampled_hashes, sampled_rows=len(sample_ids))
    overlaps = {a+'_'+b:sorted(split_sets[a]&split_sets[b])
                for a,b in [('train','val'),('train','test'),('val','test')]}
    alternates = {}
    for p in [Path(p).expanduser() for p in os.environ.get('XSPEN_IXI_ALTERNATES', '').split(os.pathsep) if p]:
        alternates[str(p)] = dict(exists=p.exists())
        if p.exists():
            alternates[str(p)]['direct_entries']=len(list(p.iterdir()))
            if p.name=='sessions':
                alternates[str(p)]['zip_archives']=len(list(p.glob('*.zip')))
            if p.name=='ixi':
                alt=list(p.rglob('*.nii.gz'))
                alternates[str(p)]['nifti_files']=len(alt)
                alternates[str(p)]['unique_subjects']=len({p.name.split('-')[0] for p in alt})
    out=dict(created_utc=datetime.now(timezone.utc).isoformat(), source_root=str(source),
        method='Header-only gzip reads, full prepared array SHA256, deterministic sampled row hashes; no GPU and no archive expansion',
        raw_files=len(files), raw_unique_subjects=len({p.name.split('-')[0] for p in files}),
        modalities=raw, header_failures=failures, alternate_locations=alternates,
        prepared_cache=dict(root=str(cache), manifest_sha256=sha256(cache/'manifest.json'),
            resolution=manifest['resolution'], fov_mm=manifest['fov_mm'], pixel_mm=[x/128 for x in manifest['fov_mm']],
            splits=split_audit, array_sha256=hashes,
            all_volume_cache_paths_exist=all(Path(v['cache']).exists() for v in manifest['volumes']),
            all_original_paths_exist=all(Path(v['path']).exists() for v in manifest['volumes'])),
        leakage=dict(subject_overlaps=overlaps, invalid_record_subjects=invalid_records,
            cross_split_source_hashes=sum(len(v)>1 for v in source_splits.values()),
            cross_split_slice_hashes=sum(len(v)>1 for v in slice_splits.values())),
        preparation_recommendation=dict(frozen_subject_manifest=str(cache/'manifest.json'),
            subjects=manifest['subjects'], use_modalities=['T2','PD'],
            native_source='Read original NIfTI once per volume and produce all grids in physical mm',
            other_modalities='T1/MRA/DTI available locally but no demonstrated need for structural xSPEN adaptation; DTI files are directions, not separate subjects',
            isolation='Use identical subject membership for every modality/view/profile/resolution; choose hyperparameters on validation only',
            limitations=['IXI gives structural magnitude images, not paired xSPEN truth or acquired complex phase.',
                        'Original data are anisotropic; higher output grids do not imply newly acquired source detail.',
                        'The old 128 cache uses 210mm FOV and no finite slice-thickness integration; native grids must use original NIfTI.']))
    dest=HERE/'audit';dest.mkdir(exist_ok=True)
    affine_path = dest / 'ixi_affine_audit.json'
    if affine_path.exists():
        out['affine_audit'] = json.loads(affine_path.read_text())
    (dest/'ixi_inventory.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    lines=['本地 IXI 审计（2026-09-13）','',f'原始 NIfTI：`{source}`。共 {out["raw_files"]} 个文件、{out["raw_unique_subjects"]} 位受试者。仅解压读取 gzip 头，没有展开归档或占用 GPU。','',
           '| 模态 | 文件数 | 受试者数 | 最常见原始尺寸 | 最常见 voxel spacing (mm) |','|---|---:|---:|---|---|']
    for modality,r in raw.items():
        shape=max(r['header_shapes'],key=r['header_shapes'].get)
        spacing=max(r['header_spacing_mm'],key=r['header_spacing_mm'].get)
        lines.append(f'| {modality} | {r["files"]} | {r["subjects"]} | {shape} | {spacing} |')
    lines += ['',f'现成缓存：`{cache}`；128×128，210×210 mm FOV，即 1.640625 mm/pixel；T2/PD，axial/sagittal。', '',
              '| split | 受试者 | 切片 |','|---|---:|---:|']
    for sp,r in split_audit.items(): lines.append(f'| {sp} | {r["subjects"]} | {r["slices"]} |')
    lines += ['',f'完整缓存 SHA256 与旧 manifest：{all(r["match"] for r in hashes.values())}；每 split 64 行等距确定位置的像素 hash 与记录一致：{all(r["sampled_row_hashes_match"] for r in split_audit.values())}。',
              f'跨 split subject/source-hash/slice-hash 重复：{sum(map(len,overlaps.values()))}/{out["leakage"]["cross_split_source_hashes"]}/{out["leakage"]["cross_split_slice_hashes"]}。', '',
              '准备方案：固定继承以上 subject split。直接从 T2/PD 原始体数据按目标 FOV、矩阵和层厚积分生成物理切片，每体一次读取同时生成 8 个协议/网格，保留每 view 最多 24 个均匀分布的有效层。每个模态、视图、分辨率都保持同一 subject split，验证用于选择超参数，test 留作最终评价。', '',
              'IXI 是结构幅度先验，不提供配对 xSPEN 真值、原始复相位或对应的扩散加权采集；T1/MRA/DTI 虽在本地，不等价于真实扫描的新协议。原始各向异性限制可恢复细节，不能把细网格输出称为原生高分辨率采集。', '',
              '准备采用 closest-canonical voxel axes，不进行 patient-world 去倾斜，也不匹配真实扫描 slice normal。1156 体中有 226 体至少一轴倾斜超过 10°，18 体超过 20°，最大 23.17°；shear 可忽略（最大归一化轴内积 2.04e-5）。具体见 `ixi_affine_audit.json` 和 `../PREPARATION.md`。', '',
              '复现审计：`python pipelines/native_resolution/audit_ixi.py`。完整数量分布、检查结果和固定受试者列表见同目录 `audit/ixi_inventory.json`。']
    (dest/'ixi_inventory.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(raw_files=out['raw_files'],raw_unique_subjects=out['raw_unique_subjects'],
                         modalities={k:(v['files'],v['subjects']) for k,v in raw.items()},
                         all_cache_hashes_match=all(r['match'] for r in hashes.values()),
                         header_failures=len(failures),output=str(dest)),ensure_ascii=False))


if __name__=='__main__':
    main()
