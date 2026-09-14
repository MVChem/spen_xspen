from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import zlib

BASE = Path('/home/data2/chk/workspace/2026/08/14/01_工作项目/spen_recons/diffusion_inverse_20260908/v2_strong_prior')
EXPANSION = BASE / 'experiments/sr2x_20260914/data_expansion'
DEST = Path('/home/data2/chk/workspace/2026/09/14/spen_xspen/code/data/rodent_mri/public_rodent_mri')
SESSION = '01a09ec8-c005-7852-a0e6-0978c94fc33f'
NOW = datetime.now(timezone(timedelta(hours=8))).isoformat()

configs = [
 dict(id='ds005236', root=BASE/'mouse_raw/ds005236', family='FSE/RARE', description='C58 成年组 TurboRARE T2', version='1.0.0', license='CC0', url='https://openneuro.org/datasets/ds005236/versions/1.0.0', doi='10.18112/openneuro.ds005236.v1.0.0', status='已下载的 T2 子集完整：59 个体积；非整个多序列数据库', data=['sub-*'], metadata=['CHANGES','README','dataset_description.json','participants.json','participants.tsv'], evidence={}, notes=['已有派生训练集保存在旧项目其他目录；此入口只链接原发布文件。']),
 dict(id='ds006663', root=BASE/'mouse_raw/ds006663', family='FSE/RARE', description='C57BL6J 纵向 RARE T2', version='1.0.3', license='CC0', url='https://openneuro.org/datasets/ds006663/versions/1.0.3', doi='10.18112/openneuro.ds006663.v1.0.3', status='已下载的 RARE T2 子集完整：138 个体积、69 只动物；非整个多序列数据库', data=['sub-*'], metadata=['CHANGES','README','*.json','participants.tsv','sessions.tsv'], evidence={}, notes=['其他序列 JSON 是作者发布的继承元数据，不表示这些序列图像已下载。']),
 dict(id='ds002868', root=BASE/'mouse_raw/ds002868', family='FSE/RARE', description='CAMRI Mouse Brain MRI：RARE T2', version='1.0.1', license='CC0', url='https://openneuro.org/datasets/ds002868/versions/1.0.1', doi='10.18112/openneuro.ds002868.v1.0.1', status='已下载的 RARE T2 子集完整：16 个体积；非整个数据库', data=['sub-*'], metadata=['CHANGES','dataset_description.json'], evidence={}, notes=['ses-2 的 EPI JSON 是原发布元数据，未对应下载 EPI 图像。']),
 dict(id='ds005186', root=EXPANSION/'sources/ds005186', family='FSE/RARE', description='C58 幼年组 TurboRARE T2', version='1.1.0', license='CC0', url='https://openneuro.org/datasets/ds005186/versions/1.1.0', doi='10.18112/openneuro.ds005186.v1.1.0', status='已下载 T2：16 个体积', data=['sub-*'], metadata=['CHANGES','README','dataset_description.json','participants.json','participants.tsv'], evidence={'download_provenance.json':'download_provenance.json','source_listing.xml':'source_listing.xml'}, notes=['与 ds005236 存在跨库动物身份重叠，不能把两个库动物数直接相加。']),
 dict(id='figshare_aging_28433102', root=EXPANSION/'sources/figshare_aging_28433102', family='FSE/RARE + DWI', description='Aging C57BL/6：TurboRARE T2 与 4-shot DtiEpi', version='9', license='CC0', url='https://figshare.com/articles/dataset/28433102/9', doi='10.6084/m9.figshare.28433102.v9', status='所选成员完整：30 个 T2；5 只动物的 25 个 4D DWI；未下载整库 ZIP', data=['sub-*'], metadata=['README','dataset_description.json','participant.tsv'], evidence={'record.json':'record.json','download_provenance.json':'download_provenance.json','download_records':'download_records','download_acceptance.json':'download_acceptance.json','epi_sequence_verification.json':'epi_sequence_verification.json','download_notes_260914.md':'README_DOWNLOAD.md'}, notes=['DWI 不是 FSE/RARE；此前用户将其排除于干净先验和 HR 训练，仅作噪声观测，本次照原样保留。','sub-26_T2W.gz 是作者原文件名；不改名、不另加别名。','作者的 dataset_description.json 本身语法无效，原样保留。','下载按 ZIP 成员验证 CRC32；完整 ZIP MD5 未验证。']),
 dict(id='zenodo5834507', root=EXPANSION/'sources/zenodo5834507', family='FSE/RARE', description='C57/Shiverer/WT：RARE T2、MT on/off 三通道', version=None, license='CC0', url='https://zenodo.org/records/5834507', doi='10.5061/dryad.1vhhmgqv8', status='所选成员完整：20 个三通道 Analyze 图像和 20 个作者 mask，共 80 个 .hdr/.img；非整个记录', data=['images'], metadata=['README_source.txt'], evidence={'zenodo_record.json':'zenodo_record.json','download_provenance.json':'download_provenance.json','download_records':'download_records','zip_index_and_selection.json':'zip_index_and_selection.json','download_acceptance.json':'download_acceptance.json','import_notes_260914.md':'README_IMPORT.md'}, notes=['作者发布前已插值并配准；这些是发布原件，不是未处理扫描仪 raw 或 k-space。','原采集 100 μm；发布 header 的单位/间距及通道索引存在未解项，本次均不修改。','仅下载选定 ZIP 成员并验证 CRC32；完整 ZIP 未下载，不能声称完整 ZIP MD5 已通过。']),
 dict(id='figshare_tc1_wt_3258139', root=EXPANSION/'sources/figshare_tc1_wt_3258139', family='GRE', description='Tc1/WT：40 μm 离体 3D GRE', version='C1/C2/metadata article v1', license='CC BY 4.0', url='https://figshare.com/collections/Tc1_and_WT_data/3258139', doi=None, status='本批 55 个发布体积完整', data=['C1','C2'], metadata=['metadata'], evidence={'figshare_3382693.json':'figshare_3382693.json','figshare_3394786.json':'figshare_3394786.json','figshare_3394801.json':'figshare_3394801.json','download_provenance.json':'download_provenance.json','download_records':'download_records','download_acceptance.json':'download_acceptance.json'}, notes=['不属于 FSE/RARE；保留该会话已下载的相关小鼠 MRI 原件。']),
 dict(id='ds004644', root=EXPANSION/'sources/ds004644', family='FLASH + MP2RAGE + UTE', description='Dp1Tyb/WT：40 μm FLASH、在体 MP2RAGE/UTE', version='1.0.0', license='CC0', url='https://openneuro.org/datasets/ds004644/versions/1.0.0', doi='10.18112/openneuro.ds004644.v1.0.0', status='所选影像完整：22 个 FLASH + 132 个 real/imag 组件；24 只动物；未下载 MRS/PRESS', data=['sub-*'], metadata=['CHANGES','README','dataset_description.json','participants.tsv'], evidence={'manifest.json':'manifest.json','download_records':'download_records','expected_inventory.json':'expected_inventory.json','download_acceptance.json':'download_acceptance.json','download_notes_260914.md':'README_DOWNLOAD.md'}, notes=['不属于 FSE/RARE；本机生成的 magnitude/ 和 PNG 不纳入。','154 个文件包括实部/虚部，不能等同为 154 只动物或独立解剖体。']),
 dict(id='zenodo6844489', root=EXPANSION/'sources/zenodo6844489', family='T2WI；FSE/RARE 未确认', description='BEN young adult C57BL6J 纵向 T2WI', version=None, license='CC BY 4.0', url='https://zenodo.org/records/6844489', doi='10.5281/zenodo.6844489', status='src/label 两个发布 ZIP 均已下载并解出；101 个图像、101 个作者标签', data=['src','label'], metadata=[], evidence={'record.json':'record.json','download_provenance.json':'download_provenance.json','source_audit.json':'source_audit.json','import_notes_260914.md':'README_IMPORT.md'}, archives=['src_adult.zip','label_adult.zip'], notes=['仅确认 T2WI；没有充分证据将其归入 FSE/RARE。','archives/ 保存同批图像/标签的完整下载 ZIP 包装链接，不另计图像数量。']),
 dict(id='zenodo6379879', root=EXPANSION/'sources/zenodo6379879', family='T2；序列未在本次确认；部分下载', description='An et al. mouse stroke MRI / Charité 已落盘部分', version=None, license='CC BY 4.0', url='https://zenodo.org/records/6379879', doi='10.5281/zenodo.6379879', status='部分下载：64 个 t2.nii 与 69 个作者 mask；完整 ZIP 未下载且 MD5 未验证', data=['extracted'], metadata=['README.txt','data_set.xlsx'], evidence={'record.json':'record.json','extraction_status.json':'extraction_status.json','zip_index.json':'zip_index.json'}, notes=['只链接之前已经解出的发布成员，本次不解压、不补下载。','保留作者发布的 manual/auto mask，不把 mask 算成 MRI 图像。','data_an_et_al_2022.zip.part 和 range_chunks 是不完整下载缓存，未作为原件入口；历史 extraction_status.json 的缓存字节数可能早于实际残片大小。']),
]

DEST.mkdir(parents=True, exist_ok=True)
links = []
file_jobs = {}
expected = {}

def add_known(value, evidence_path):
    if isinstance(value, dict):
        digest=value.get('sha256')
        raw=value.get('local_path',value.get('path'))
        if isinstance(digest,str) and len(digest)==64 and isinstance(raw,str) and raw.startswith('/'):
            raw=raw.replace('/2026/08/14/spen_recons/','/2026/08/14/01_工作项目/spen_recons/')
            expected.setdefault(raw, []).append({'sha256':digest,'evidence':str(evidence_path)})
        for child in value.values(): add_known(child,evidence_path)
    elif isinstance(value,list):
        for child in value: add_known(child,evidence_path)

def role_of(path, default):
    if default!='published_content': return default
    low=path.name.lower()
    if low.endswith('.img'): return 'published_mask' if 'masked_outline' in low else 'published_mri_image'
    if low.endswith('.hdr'): return 'published_image_or_mask_header'
    if low.endswith(('.nii','.nii.gz')) or low=='sub-26_t2w.gz':
        return 'published_mask' if path.parent.name=='label' or low.startswith('masklesion') else 'published_mri_image'
    return 'source_metadata'

def link_one(dataset, relative, source, default):
    if not source.exists(): raise FileNotFoundError(source)
    dest=(DEST/'_provenance'/relative) if dataset=='_shared_provenance' else (DEST/dataset/relative)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if os.path.lexists(dest):
        if not dest.is_symlink() or dest.resolve()!=source.resolve(): raise RuntimeError(f'Refusing to replace {dest}')
    else: dest.symlink_to(source,target_is_directory=source.is_dir())
    source_files=sorted(p for p in source.rglob('*') if p.is_file()) if source.is_dir() else [source]
    role_counts=Counter()
    for f in source_files:
        rel=dest.relative_to(DEST)/(f.relative_to(source)) if source.is_dir() else dest.relative_to(DEST)
        role=role_of(f,default);role_counts[role]+=1
        file_jobs[str(f)]={'source_path':str(f),'source_resolved_path':str(f.resolve()),'local_path':str(DEST/rel),'local_relative_path':str(rel),'dataset_id':dataset,'role':role,'bytes':f.stat().st_size}
    links.append({'dataset_id':dataset,'local_relative_path':str(dest.relative_to(DEST)),'source_path':str(source),'source_resolved_path':str(source.resolve()),'kind':'directory' if source.is_dir() else 'file','file_count':len(source_files),'bytes':sum(f.stat().st_size for f in source_files),'counts_by_role':dict(role_counts)})

for cfg in configs:
    root=cfg['root']
    for pattern in cfg['data']:
        matches=sorted(root.glob(pattern))
        if not matches: raise RuntimeError(f'Empty data glob {root}/{pattern}')
        for p in matches: link_one(cfg['id'],p.relative_to(root),p,'published_content')
    selected=set()
    for pattern in cfg['metadata']:
        for p in sorted(root.glob(pattern)):
            if p in selected:continue
            selected.add(p);link_one(cfg['id'],p.relative_to(root),p,'source_metadata')
    for name,source in cfg['evidence'].items():
        p=root/source;link_one(cfg['id'],Path('_provenance')/name,p,'prior_provenance')
        if p.is_file() and p.suffix=='.json':
            try:add_known(json.loads(p.read_text()),p)
            except ValueError:pass
    for name in cfg.get('archives',[]):link_one(cfg['id'],Path('archives')/name,root/name,'published_archive')

shared=DEST/'_provenance';shared.mkdir(exist_ok=True)
for name,source in {
    'mouse_raw_download_manifest.json':BASE/'mouse_raw/download_manifest.json',
    'FSE_RARE_INVENTORY_260914.md':EXPANSION/'FSE_RARE_INVENTORY.md',
    'DOWNLOAD_STATUS_260914.md':EXPANSION/'DOWNLOAD_STATUS.md',
    'QUALIFICATION_005236_005186_260914.md':EXPANSION/'QUALIFICATION_005236_005186.md',
    'cross_dataset_subject_mapping.json':EXPANSION/'cross_dataset_subject_mapping.json',
}.items():
    link_one('_shared_provenance',name,source,'prior_provenance')
    if source.suffix=='.json':add_known(json.loads(source.read_text()),source)

charite_index={entry['name']:entry for entry in json.loads((EXPANSION/'sources/zenodo6379879/zip_index.json').read_text())}

def hash_file(item):
    path=Path(item['source_path']);before=path.stat();sha=hashlib.sha256();crc=0
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            sha.update(block)
            if item['dataset_id']=='zenodo6379879' and 'extracted/' in str(path):crc=zlib.crc32(block,crc)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError(f'Source changed during hashing: {path}')
    result={**item,'sha256':sha.hexdigest(),'source_mtime_ns':after.st_mtime_ns}
    known=expected.get(str(path),[])
    if known:
        result['prior_sha256_evidence']=known
        result['matches_prior_sha256']=all(r['sha256']==result['sha256'] for r in known)
        if not result['matches_prior_sha256']:raise RuntimeError(f'Historical SHA256 mismatch: {path}')
    if item['dataset_id']=='zenodo6379879' and '/extracted/' in str(path):
        member=str(path).split('/extracted/',1)[1];record=charite_index.get(member)
        result['archive_member']=member
        if record:
            result['matches_recorded_zip_member_crc32_and_size']=(record['crc32']==crc and record['size']==before.st_size)
            result['crc32']=f'{crc:08x}'
            if not result['matches_recorded_zip_member_crc32_and_size']:raise RuntimeError(f'ZIP member mismatch: {path}')
    return result

print(f'Created {len(links)} links; hashing {len(file_jobs)} files, {sum(j["bytes"] for j in file_jobs.values())} bytes',flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:
    files=list(pool.map(hash_file,file_jobs.values()))

datasets=[]
for cfg in configs:
    fs=[f for f in files if f['dataset_id']==cfg['id']]
    counts=Counter(f['role'] for f in fs)
    row={k:v for k,v in cfg.items() if k not in ('root','data','metadata','evidence','archives')}
    row['original_source_root']=str(cfg['root']);row['local_relative_path']=cfg['id']
    row['counts_by_role']=dict(counts);row['bytes_by_role']={r:sum(f['bytes'] for f in fs if f['role']==r) for r in counts}
    row['symlink_count']=sum(l['dataset_id']==cfg['id'] for l in links)
    row['published_file_count']=sum(n for r,n in counts.items() if r!='prior_provenance')
    row['published_file_bytes']=sum(f['bytes'] for f in fs if f['role']!='prior_provenance')
    row['version_and_license_evidence']=[str(cfg['root']/n) for n in ['dataset_description.json','record.json','zenodo_record.json','figshare_3382693.json','figshare_3394786.json'] if (cfg['root']/n).exists()]
    if cfg['version'] is None:row['version_note']='原记录未声明 version 字段；以固定 Zenodo record ID 标识，未将 revision 当作数据版本。'
    datasets.append(row)

manifest={'schema_version':1,'created_at':NOW,'source_session_id':SESSION,'scope':'本次引用会话已有的公开小鼠 MRI 发布原件和作者元数据；仅创建本机绝对符号链接，无下载或图像处理。','definition_of_original':'原始指下载/此前从发布ZIP解出的原样文件，不保证发布方未作重建、插值、配准或mask生成。','integrity_check':'本次逐文件 SHA256；有旧 SHA256 的文件复核一致；Charité 已解出成员另与历史 ZIP 索引的 CRC32 和大小一致。','source_roots':[str(BASE/'mouse_raw'),str(EXPANSION/'sources')],'datasets':datasets,'links':links,'files':files,'summary':{'dataset_count':len(datasets),'symlink_count':len(links),'hashed_file_count':len(files),'hashed_bytes':sum(f['bytes'] for f in files),'published_mri_image_files':sum(f['role']=='published_mri_image' for f in files),'published_mask_files':sum(f['role']=='published_mask' for f in files),'files_matched_prior_sha256':sum(f.get('matches_prior_sha256',False) for f in files),'charite_members_crc32_verified':sum(f.get('matches_recorded_zip_member_crc32_and_size',False) for f in files)},'excluded_existing_content':['本机派生 magnitude/、sample100、192训练NPY、预览PNG与QC渲染','HTTP Range缓存、ZIP片段、.part文件；仅在来源说明中记录其不完整状态'],'not_downloaded':['DeepBrainIPP：原会话用户取消，无MRI图像，不继续下载。']}
(DEST/'provenance_260914.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')

lines=['# 公开小鼠 MRI 发布原件','',f'整理日期：2026-09-14。用户引用会话：`{SESSION}`。','',
'这里收录该会话在本机已经下载的 10 个来源。数据通过绝对符号链接指向旧工作目录；本次没有下载新来源、解压、裁剪、重采样、归一化、去噪或生成训练集。原目录不能删除或移动，否则链接会失效。','',
'“原始”指下载的发布原件，未声称都是扫描仪 raw/k-space。作者发布时已有的图像重建、配准、插值和 mask 仍保持原样；本机生成的预览、RSS 幅度体和 192×192 导出未纳入。','',
'| 目录 | 发布数据与序列 | MRI 图像文件数 | 状态 | 版本；许可证 |','|---|---|---:|---|---|']
for d in datasets:
    lines.append(f'| [{d["id"]}]({d["id"]}/) | {d["description"]} | {d["counts_by_role"].get("published_mri_image",0)} | {d["status"]} | {d["version"] or "固定 Zenodo 记录，未声明版本"}；{d["license"]} |')
lines+=['','图像文件数不等于独立动物数：其中有纵向扫描、real/imag 组件和三通道发布体；作者 mask、配对 .hdr 与 ZIP 包装未算作 MRI 图像。ds005236 与 ds005186 有跨库身份重叠。','',
'## 来源与边界','']
for d in datasets:
    lines += [f'### {d["id"]}','',f'- 来源：[{d["id"]}]({d["url"]})。'+(f' DOI：`{d["doi"]}`。' if d['doi'] else ''),f'- 旧目录：`{d["original_source_root"]}`。',f'- 纳入发布文件 {d["published_file_count"]} 个、{d["published_file_bytes"]:,} 字节，另有历史溯源证据。']
    for note in d['notes']:lines.append('- '+note)
    lines.append('')
lines+=['## 溯源与校验','',
'[provenance_260914.json](provenance_260914.json) 记录每个数据集的 URL、版本/许可证证据、每条链接的源路径、文件数和字节数，以及每个纳入文件的 SHA256。各来源 `_provenance/` 保留旧下载记录；总目录 `_provenance/` 保留旧公开清单、资格审计及跨库身份映射。旧记录中的路径和时间保持历史原样，当前链接位置以新清单为准。','',
f'本次校验 {len(files)} 个文件、{sum(f["bytes"] for f in files):,} 字节；其中 {manifest["summary"]["files_matched_prior_sha256"]} 个文件与历史 SHA256 一致。Charité 的 133 个已解出成员与旧 ZIP 索引的大小和 CRC32 一致；这不能证明整个 Charité 压缩包下载完整。','',
'Aging 和 Zenodo5834507 仅获取过所选 ZIP 成员，不能把“所选成员完整”写成“整个公开数据库已下载”。Charité 明确为部分下载，残片缓存没有纳入原件入口。DeepBrainIPP 已在原会话取消且没有 MRI 图像，本次未继续下载。','']
(DEST/'README_260914.md').write_text('\n'.join(lines))
broken=[str(DEST/l['local_relative_path']) for l in links if not (DEST/l['local_relative_path']).exists()]
if broken:raise RuntimeError(f'Broken symlinks: {broken}')
print(json.dumps(manifest['summary'],ensure_ascii=False),flush=True)
print('README and provenance saved; all symlink targets exist.',flush=True)
