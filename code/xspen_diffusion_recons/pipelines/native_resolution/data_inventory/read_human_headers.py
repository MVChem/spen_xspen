"""Retained analysis recipe; execute explicitly after supplying its input artifacts."""

def main():
    from pathlib import Path
    from collections import Counter,defaultdict
    import struct,re,json,hashlib,datetime
    import os
    OUT=Path(__file__).resolve().parent
    PROJECT=OUT.parents[2]
    DATA=Path(os.environ.get('XSPEN_RAW_SCANNER', str(PROJECT/'data/raw/siemens'))).expanduser().resolve()
    OLD=json.loads((PROJECT/'scanner/inventory.json').read_text())
    old_paths={str(Path(r['source']).resolve()) for r in OLD}
    old_audit={str(Path(r['path']).resolve()):r for r in json.loads((PROJECT/'pipelines/native_resolution/audit/siemens_header_manifest.json').read_text())}
    
    def values(text,key):
     raw=re.findall('^'+re.escape(key)+r'\s*=\s*(.*?)\s*$',text,re.M)
     unique=list(dict.fromkeys(raw))
     def convert(x):
      if x.startswith('"'):return x.strip('"')
      try:return float(x) if any(c in x for c in '.eE') else int(x,0)
      except ValueError:return x
     return [convert(x) for x in unique]
    
    def read_header(path):
     with path.open('rb') as f:
      size=struct.unpack('<I',f.read(4))[0]
      if not 4<size<16*1024*1024 or size>=path.stat().st_size:raise ValueError(f'Not expected VB raw: {path}')
      f.seek(0);head=f.read(size)
      payload=f.read(4096)
     return head,payload
    
    def priority(mid):
     if mid in ['MID112','MID114','MID27']:return ('current',0,'Already included in current validated adapter/comparison; not an additional scan.')
     if mid=='MID78':return ('known_phantom',9,'Existing raw-data visual QC identifies a uniform phantom; useful for phantom physics QC, excluded from human results.')
     if mid=='MID80':return ('anatomy_pending',5,'Companion of the outer-folder phantom scan; exported diagnostic HDF5 exists, but human anatomy not confirmed.')
     if mid=='MID51':return ('highest_extension_priority',1,'Sagittal counterpart of MID27; same FOV/matrix/thickness/R/beta, same-name saved MAT. Verify anatomy and MDH repeated-row coverage before adding.')
     if mid in ['MID106','MID108']:return ('high_extension_priority',2,'Full-FOV axial/sagittal files under the already accepted named session. Header R=48 and FOV184x192 differ from MID27 R=46/FOV184.958x193, requiring explicit protocol mapping and QC.')
     if mid in ['MID613','MID615']:return ('high_extension_priority',2,'Full-FOV axial/coronal b600 candidates with saved MAT; coronal adds another orientation. Need anatomy/MDH QA and header-matched R48 operator.')
     if mid in ['MID127','MID129']:return ('localized_extension_candidate',3,'36x40, small FOV72.9x81mm, 2mm slice thickness, same-name MAT; useful candidate for localized anatomy, not evidence of 2mm whole-brain effective resolution.')
     if mid in ['MID57','MID59']:return ('localized_extension_candidate',3,'Small FOV90x96mm, 3mm slice thickness; same-name MAT. Header PE32 vs chirp R30 requires checking actual line coverage.')
     if mid in ['MID530','MID533','MID535','MID537']:return ('orientation_extension_candidate',4,'Axial/coronal/oblique full-FOV candidates; each has a corresponding 48-image DICOM-like derived directory, but no same-name MAT. Verify anatomy, repetition structure and reconstruction input.')
     if mid in ['MID74','MID76','MID82','MID84']:return ('anatomy_pending',5,'Outer-folder scans near known phantom MID78: filename is not proof of a human subject. Confirm object and coverage before adding human results.')
     return ('separate_hybrid_family',8,'Quadratic Hybrid SPEN, not crossed-chirp xSPEN; maintain a separate model/adapter and evaluation group.')
    
    rows=[]
    for path in sorted(p for p in DATA.rglob('*.dat') if p.is_file()):
     head,payload=read_header(path);text=head.decode('latin1');conflicts={}
     def get(k):
      found=values(text,k)
      if len(found)>1:conflicts[k]=found
      return found[0] if found else None
     sequence=get('tSequenceFileName');family='crossed_chirp_xSPEN' if sequence and 'esaszz_xSPEN_180c180c_bipolarDiff' in sequence else 'quadratic_Hybrid_SPEN' if sequence and 'esrs_hyb_spen_Diff2' in sequence else 'unclassified'
     mid=re.search(r'MID\d+',path.name)[0]
     pe,ro=get('sKSpace.lPhaseEncodingLines'),get('sKSpace.lBaseResolution')
     fov=[get('sSliceArray.asSlice[0].dPhaseFOV'),get('sSliceArray.asSlice[0].dReadoutFOV')]
     normal=[get('sSliceArray.asSlice[0].sNormal.'+n) or 0 for n in ['dSag','dCor','dTra']]
     view='oblique' if sum(abs(v)>.1 for v in normal)!=1 else ['sagittal','coronal','axial'][max(range(3),key=lambda i:abs(normal[i]))]
     status,prio,reason=priority(mid)
     headhash=hashlib.sha256(head).hexdigest();old=old_audit.get(str(path.resolve()))
     b=get('sWiPMemBlock.adFree[4]') if family=='crossed_chirp_xSPEN' else None
     mat=path.with_suffix('.mat');h5=PROJECT/'scanner'/f'{mid}.h5'
     row=dict(scan_id=mid,path=str(path),relative_path=str(path.relative_to(DATA)),group=path.relative_to(DATA).parts[0],
      bytes=path.stat().st_size,kind='Siemens_VB_raw_file',is_symlink=path.is_symlink(),resolved_path=str(path.resolve()),
      header_bytes=len(head),header_sha256=headhash,read_payload_prefix_bytes=len(payload),payload_prefix_contains_nonzero_bytes=any(payload),
      header_matches_20260913=old is not None and old['header_sha256']==headhash,
      sequence=sequence,encoding_family=family,protocol_name=get('tProtocolName'),
      header_matrix_pe_ro=[pe,ro],fov_mm_pe_ro=fov,header_nominal_pixel_mm_pe_ro=[f/n if f and n else None for f,n in zip(fov,[pe,ro])],
      thickness_mm=get('sSliceArray.asSlice[0].dThickness'),header_slices=get('sSliceArray.lSize'),
      view=view,slice_normal_lps=normal,nominal_b_value=b,
      nominal_b_evidence='sWiPMemBlock.adFree[4], provisional current bipolarDiff WIP mapping; not a verified per-repeat b-table' if family=='crossed_chirp_xSPEN' else 'Not calibrated for Hybrid family in this inventory; see unassigned header adFree[0] and filename separately',
      hybrid_unassigned_adFree0=get('sWiPMemBlock.adFree[0]') if family=='quadratic_Hybrid_SPEN' else None,
      r_value=get('sWiPMemBlock.alFree[14]') if family=='crossed_chirp_xSPEN' else None,
      second_chirp_r=get('sWiPMemBlock.alFree[18]') if family=='crossed_chirp_xSPEN' else None,
      beta=get('sWiPMemBlock.adFree[2]') if family=='crossed_chirp_xSPEN' else None,
      chirp_duration_us=get('sWiPMemBlock.alFree[15]') if family=='crossed_chirp_xSPEN' else None,
      second_chirp_duration_us=get('sWiPMemBlock.alFree[19]') if family=='crossed_chirp_xSPEN' else None,
      echo_spacing_us=get('sFastImaging.lEchoSpacing'),echo_time_us=get('alTE[0]'),
      standard_header_repetitions=get('lRepetitions'),repeat_count_verified=None,
      current_experiment_selected=status=='current',current_status=status,extension_priority=prio,selection_or_extension_reason=reason,
      saved_same_stem_mat=str(mat) if mat.is_file() else None,existing_scanner_h5=str(h5) if h5.is_file() else None,
      conflicting_header_values=conflicts)
     if status=='current':
      export=json.loads((PROJECT/'scanner'/f'{mid}.json').read_text())
      row['repeat_count_verified']=export['repeated_counter_occurrences'];row['current_export_shape_rep_slice_coil_pe_ro']=export['shape_rep_slice_coil_pe_ro'];row['current_excluded_slice_counters']=export['excluded_original_slice_counters']
     rows.append(row)
    folders=[]
    for path in sorted(p for p in DATA.rglob('*.dat') if p.is_dir()):
     files=[p for p in path.iterdir() if p.is_file()]
     markers=[]
     for p in files[:3]:
      with p.open('rb') as f:f.seek(128);markers.append(f.read(4).decode('ascii',errors='replace'))
     folders.append(dict(path=str(path),kind='derived_image_directory_not_raw_file',files=len(files),
      sampled_files=min(3,len(files)),sampled_magic_at_offset128=markers,all_sampled_have_DICM=all(m=='DICM' for m in markers)))
    rawpaths={str(Path(r['path']).resolve()) for r in rows}
    families={}
    for fam in sorted({r['encoding_family'] for r in rows}):
     rr=[r for r in rows if r['encoding_family']==fam];families[fam]=dict(raw_files=len(rr),bytes=sum(r['bytes'] for r in rr),current_selected=sum(r['current_experiment_selected'] for r in rr),same_stem_mat_files=sum(r['saved_same_stem_mat'] is not None for r in rr))
    groups=[]
    for group in sorted({r['group'] for r in rows}):
     rr=[r for r in rows if r['group']==group];groups.append(dict(group=group,raw_files=len(rr),families=dict(Counter(r['encoding_family'] for r in rr)),scan_ids=[r['scan_id'] for r in rr],bytes=sum(r['bytes'] for r in rr)))
    summary=dict(date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(DATA),
      count_definition='Local original-acquisition .dat files, not independently confirmed human subjects or deduplicated raw payload acquisitions.',
      raw_files=len(rows),families=families,groups=groups,dat_named_derived_directories=folders,
      current_crossed_chirp_selected=3,current_crossed_chirp_unselected=sum(r['encoding_family']=='crossed_chirp_xSPEN' and not r['current_experiment_selected'] for r in rows),
      counts_match_previous_inventory=rawpaths==old_paths,new_raw_paths=sorted(rawpaths-old_paths),missing_old_raw_paths=sorted(old_paths-rawpaths),
      all_headers_match_previous_native_audit=all(r['header_matches_20260913'] for r in rows),
      all_header_values_consistent=all(not r['conflicting_header_values'] for r in rows),
      audit_scope='Fresh recursive file listing, full ASCII header reads, up to 4096 payload-prefix bytes per raw, same-name MAT/HDF5 file existence, DICM marker samples; no reconstruction, no full array loading, no GPU.',
      deduplication='No full-file or raw-payload hashing performed. Header hashes are provenance checks only, not payload deduplication. MAT, HDF5, DICOM, registered volumes and source copies are not added as raw scans.',
      interpretation=['The 22 crossed-chirp files include known phantom MID78 and unverified-anatomy files; do not call them 22 verified human scans or subjects.',
       'Header FOV / matrix describes nominal sampling spacing only. It does not prove effective spatial resolution, valid complete MDH line coverage or whole-brain coverage.',
       'Filename 2iso/3iso/4iso and 16xDTI are labels, not independent verification of voxel spacing or repeat identities.',
       'The two previously selected physical geometries were a first adapted comparison subset, not the entire available xSPEN data pool.',
       'For all unselected files, slice/repeat counts beyond header_slices require MDH occurrence validation; no new rejection is implied simply by missing adaptation.'],
      scans=rows)
    (OUT/'human_xspen_inventory.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    
    def fmt(xs):return '×'.join(f'{x:.6g}' if isinstance(x,(int,float)) else str(x) for x in xs)
    md=['# 人脑目录中的全部 Siemens xSPEN / Hybrid SPEN 原始文件','',
     '2026-09-14，只读头部盘点。**本地不止刚才展示的 3 个扫描：此目录共有 34 份 Siemens raw 文件，其中 22 份是 crossed-chirp bipolarDiff xSPEN、12 份是 quadratic Hybrid SPEN。当前人脑适配只使用前者中的 3 份，另有 19 份尚未纳入。**','',
     '这里数的是原始采集 `.dat` 文件，不是独立人次。22 份里含已核实为体模的 MID78，以及尚未核实对象的文件；不能把它们全称作已确认人脑。未纳入多数表示尚未做当前 reader/算子/图像 QC，不表示没有数据。','',
     '## 按来源分组','',
     '| 来源目录 | raw 文件 | 序列家族 | MID |','|---|---:|---|---|']
    for g in groups:md.append(f'| `{g["group"]}` | {g["raw_files"]} | '+', '.join(f'{k}: {v}' for k,v in g['families'].items())+' | '+', '.join(g['scan_ids'])+' |')
    md +=['','## 22 份真正 crossed-chirp xSPEN 的头部','',
     '以下矩阵为头部 `[PE,RO]`，间距为 FOV/头部矩阵；层数也只是头部值。它们不是全部 MDH 有效观测维度的重新验证，更不是有效空间分辨率。标准 header repetition counter 无法替代本序列的 hidden-occurrence 循环检查。','',
     '| MID | 方向 | PE×RO | FOV mm | 名义平面间距 mm | 层厚 mm | 头部层数 | 名义 b | R | 同名 MAT | 当前状态 |','|---|---|---|---|---|---:|---:|---:|---:|---|---|']
    labels={'current':'已纳入','known_phantom':'已知体模','anatomy_pending':'对象待核实','highest_extension_priority':'首选扩展','high_extension_priority':'优先扩展','localized_extension_candidate':'局部 FOV 候选','orientation_extension_candidate':'更多方向候选'}
    for r in sorted((r for r in rows if r['encoding_family']=='crossed_chirp_xSPEN'),key=lambda x:int(x['scan_id'][3:])):
     md.append(f'| {r["scan_id"]} | {r["view"]} | {fmt(r["header_matrix_pe_ro"])} | {fmt(r["fov_mm_pe_ro"])} | {fmt(r["header_nominal_pixel_mm_pe_ro"])} | {r["thickness_mm"]} | {r["header_slices"]} | {r["nominal_b_value"]} | {r["r_value"]} | {"有" if r["saved_same_stem_mat"] else "无"} | {labels[r["current_status"]]} |')
    md +=['','`b` 来自当前 bipolarDiff 协议 `sWiPMemBlock.adFree[4]` 的名义参数，不代表已核实每一个 repeat 的 b0/扩散方向表。R 来自 chirp 编码参数，不是加速率。','',
     '**文件名不能代替几何：** MID82/84 的文件名有 `2iso`，但头部为 32×32、FOV 60×64 mm，即 PE/RO 名义间距 1.875×2 mm，且 chirp R=30；MID127/129 为 36×40、72.9×81 mm，即 2.025×2.025 mm。两组都是小 FOV，不能描述成“已验证的 2 mm 全脑 xSPEN”。MID57/59 则为 32×32、90×96 mm，即 2.8125×3 mm，并非仅凭 `3iso` 就是严格 3 mm 各向同性。','',
     '## 为什么只展示了 3 份，以及先扩展哪些','',
     '当前 MID112/MID114/MID27 已具备保留重复采集的 reader 导出、多线圈 HDF5、对象图像 QC、与简化 cross-term 算子的匹配和同案例方法比较，合计 2,876 个切片×采集组观测。其余文件不能仅凭有 raw 就宣称已经完成这些验证。','',
     '1. **MID51 最接近直接扩展。** 它是 MID27 的 sagittal 对应文件，同为 46×48、184.9583333×193 mm、4 mm 层厚、R=46、名义 b=1000，并有同名 MAT。下一步核实对象和 MDH 各行 occurrence；通过后可增加方向而不先引入一个全新的编码家族。',
     '2. **MID106/MID108、MID613/MID615 为全 FOV 优先候选。** 前两份轴/矢方向、名义 b=1000；后两份轴/冠方向、名义 b=600，且有同名 MAT。它们为 46×48、184×192 mm、R=48，不能直接复用 MID27 的 R=46 和略不同 FOV。冠状位是有价值的新增适配方向。',
     '3. **MID127/MID129、MID57/MID59 可扩展局部采集。** 都有原作者 MAT，但必须先确认实际解剖 coverage、对象和有效采样行数；小 FOV 不应被包装成新增全脑高分辨率数据。',
     '4. **MID530/533/535/537 提供轴位、冠状位和斜切方向。** 有真实 raw，另有每目录 48 幅的 DICOM 图像派生；当前未检查其全部 MDH 循环与对象，不因缺同名 MAT 排除原始数据的存在。',
     '5. **MID74/76/78/80/82/84 先做对象分类。** 它们在外层目录，MID78 已有图像 QC 显示体模；MID80 作为伴随扫描仍待核实，其余也不能仅凭 brain 或 DTI 文件名认成人脑。体模依然可用作物理/噪声/重建测试，只需与人体结果分组。','',
     '以上是适配优先级，不是按图像好坏或算法结果筛选样本；本轮没有做新的重建。','',
     '## 另外 12 份 Hybrid SPEN','',
     '`Trio_brain/full brain` 有 MID201/212/247/253 共 4 份；`Trio_brain/cerebellum` 有 MID1494/1496/1498/1500/1503/1505/1507/1509 共 8 份。头部全部为 `esrs_hyb_spen_Diff2`，属于 quadratic Hybrid SPEN。它们确实是更多真实数据，但需要单独的编码模型，不能凑进 22 份 crossed-chirp xSPEN。','',
     '| MID | 头部 PE×RO | FOV mm | 层厚 mm | 头部层数 | 未标定的 adFree[0] |','|---|---|---|---:|---:|---|']
    for r in sorted((r for r in rows if r['encoding_family']=='quadratic_Hybrid_SPEN'),key=lambda x:int(x['scan_id'][3:])):
     md.append(f'| {r["scan_id"]} | {fmt(r["header_matrix_pe_ro"])} | {fmt(r["fov_mm_pe_ro"])} | {r["thickness_mm"]} | {r["header_slices"]} | {r["hybrid_unassigned_adFree0"] if r["hybrid_unassigned_adFree0"] is not None else "未显式存储"} |')
    md +=['','Hybrid 多 shot 的完整图像矩阵不能仅凭上表推导；本轮未重新标定它们的 b 值映射。特别是 MID253 文件名 `Ortho900`，头部 `adFree[0]=600`，应保留原始字段证据，而不是直接把文件名中的 900 当作已验证 b 值。','',
     '## 原始文件、保存图像和复制的区别','',
     '- 此根目录包含 **34 个实际 `.dat` 文件**，与之前 `scanner/inventory.json` 的路径集合完全一致；本次逐份重新读取完整 VB ASCII 头，序列与此前 native audit 一致。',
     '- 还存在 **4 个以 `.dat` 结尾的目录**（`3_meas_MID530...` 至 `6_meas_MID537...`），每个含 48 个 MR 图像文件。这些不是原始 `.dat` 文件，不计为另外 4 次采集。每目录抽查前 3 个文件的 offset 128 标记，结果记录在 JSON。',
     '- 22 份 crossed-chirp raw 中 **10 份有同主名 `.mat`**：MID613/615/112/114/127/129/57/59/27/51。MAT 保存图像/中间数据、HDF5 当前导出、DICOM、配准后的 Smat/CoregedFull 和 DTI/b-map 均为派生资料，不另计 raw 次数。',
     '- 原始材料本身是本地保存的实验室数据副本；本次不计算完整 raw/payload hash，不声称完成内容级去重。header hash 和文件大小仅用于可复查的文件/头部核对，不能证明是 22 位受试者或 22 个独立生物学样本。','',
     '完整路径、序列、精确原字段、头部哈希、同名 MAT/HDF5 入口及每份扩展理由见 [human_xspen_inventory.json](human_xspen_inventory.json)。所有原始资料保持只读；本轮未加载完整成像数组、未启动重建或训练。']
    (OUT/'human_xspen_inventory.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({k:summary[k] for k in ['raw_files','families','current_crossed_chirp_unselected','counts_match_previous_inventory','all_headers_match_previous_native_audit','all_header_values_consistent']},indent=2))

if __name__ == '__main__':
    main()
