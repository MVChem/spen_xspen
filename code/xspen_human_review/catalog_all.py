"""Read-only catalogue of the acquisition archive, including legacy image exports.

Unknown array axes remain explicit; raw ADC previews are never called PE
reconstructions. Large arrays are exported in small, independently loaded chunks.
"""
import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import traceback
import zipfile

import h5py
import nibabel as nib
import numpy as np
from PIL import Image
import pydicom
from scipy.io import loadmat, whosmat
from build_unified import digest, dump

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/data2/chk/workspace/2026/08/14/xSPEN_项目')
KINDS = {'.mat':'mat', '.fig':'mat', '.nii':'nifti', '.ima':'dicom', '.dcm':'dicom', '.dat':'raw'}


def ident(prefix, key):
    return prefix + '_' + hashlib.sha256(key.encode()).hexdigest()[:16]


def public_entry(entry):
    result=dict(entry)
    result['source_details']={k:v for k,v in entry['source_details'].items()
        if k not in ['frame_metadata','frame_counters','source_files']}
    return result


def candidate(name):
    p=Path(name)
    if p.name.lower().endswith('.nii.gz'):return 'nifti'
    if p.suffix.lower() in KINDS:return KINDS[p.suffix.lower()]
    if p.suffix.startswith('.2016'):return 'dicom'
    return None


def inventory(run):
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and candidate(p.name):
            files.append(dict(path=str(p), source=str(p), kind=candidate(p.name), bytes=p.stat().st_size))
    archives=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in ['.zip','.rar','.7z','.tar','.gz']:continue
        if p.name.lower().endswith('.nii.gz'):continue
        report=dict(source=str(p), status='inspected', members=[])
        folder=run/'archive_sources'/ident('container',str(p))
        try:
            if p.suffix.lower()=='.zip':
                with zipfile.ZipFile(p) as archive:
                    members=[i.filename for i in archive.infolist() if not i.is_dir() and candidate(i.filename)]
                    for name in members:
                        destination=folder/(ident('file',name)+Path(name).suffix)
                        destination.parent.mkdir(parents=True,exist_ok=True)
                        if not destination.exists():
                            with archive.open(name) as src,destination.open('wb') as dst:shutil.copyfileobj(src,dst)
                        files.append(dict(path=str(destination),source=str(p)+'!/'+name,kind=candidate(name),bytes=destination.stat().st_size))
                        report['members'].append(name)
            else:
                result=subprocess.run(['7z','l','-slt',str(p)],capture_output=True,text=True,timeout=45)
                if result.returncode:raise ValueError(result.stderr.strip() or result.stdout[-400:])
                blocks=result.stdout.split('\n\n')
                for block in blocks:
                    fields=dict(line.split(' = ',1) for line in block.splitlines() if ' = ' in line)
                    name=fields.get('Path','')
                    if name==str(p) or fields.get('Folder')=='+' or not candidate(name):continue
                    destination=folder/(ident('file',name)+Path(name).suffix)
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    if not destination.exists() or destination.stat().st_size==0:
                        with destination.open('wb') as stream:
                            extracted=subprocess.run(['7z','x','-so',str(p),name],stdout=stream,stderr=subprocess.PIPE,timeout=180)
                        if extracted.returncode and p.suffix.lower()=='.rar':
                            with destination.open('wb') as stream:
                                extracted=subprocess.run(['unrar','p','-inul','-p-',str(p),name],stdout=stream,stderr=subprocess.PIPE,timeout=180)
                        if extracted.returncode:raise ValueError(extracted.stderr.decode(errors='replace'))
                    files.append(dict(path=str(destination),source=str(p)+'!/'+name,kind=candidate(name),bytes=destination.stat().st_size))
                    report['members'].append(name)
        except Exception as error:
            report.update(status='unreadable_archive',reason=str(error))
        archives.append(report)
    dump(run/'inventory/archive_container_coverage.json',archives)
    unique={};rows=[]
    for f in files:
        key=digest(f['path'])
        f['sha256']=key
        if key in unique:
            f.update(status='duplicate_file',duplicate_of=unique[key]['source'])
        else:
            f['status']='pending';unique[key]=f
        rows.append(f)
    dump(run/'inventory/all_source_files.json',rows)
    return rows


def array_leaves(value, name, skipped, depth=0):
    """Read numeric image/matrix leaves, including MATLAB cells and structs."""
    if depth>12:
        skipped.append(dict(variable=name,reason='Nested container depth exceeds 12'));return
    if hasattr(value,'_fieldnames'):
        for field in value._fieldnames:
            yield from array_leaves(getattr(value,field),name+'.'+field,skipped,depth+1)
        return
    if not isinstance(value,np.ndarray):return
    if value.dtype==object:
        cells=list(value.flat)
        if cells and all(isinstance(a,np.ndarray) and a.dtype.kind in 'biufc' and a.shape==cells[0].shape for a in cells):
            if sum(n>=16 for n in cells[0].shape)>=2:
                yield name+'[cells]',np.stack(cells,axis=-1)
                return
        for i,child in enumerate(cells):yield from array_leaves(child,f'{name}[{i+1}]',skipped,depth+1)
        return
    if value.dtype.names:
        for field in value.dtype.names:yield from array_leaves(value[field],name+'.'+field,skipped,depth+1)
        return
    if value.dtype.kind not in 'biufc':return
    shape=value.shape
    if sum(n>=16 for n in shape)<2:
        skipped.append(dict(variable=name,shape=list(shape),reason='Scalar/vector/small parameter array; no two image-sized axes'))
        return
    yield name,value


def mat_arrays(path, skipped):
    if h5py.is_hdf5(path):
        with h5py.File(path) as f:
            def walk(group,prefix=''):
                for name,v in group.items():
                    if name.startswith('#'):continue
                    key=prefix+name
                    if isinstance(v,h5py.Group):yield from walk(v,key+'.')
                    elif v.dtype.kind in 'biufc' or (v.dtype.names and set(v.dtype.names)=={'real','imag'}):
                        a=v[()]
                        if a.dtype.names:a=a['real']+1j*a['imag']
                        a=a.transpose(tuple(range(a.ndim-1,-1,-1)))
                        yield from array_leaves(a,key,skipped)
                    else:skipped.append(dict(variable=key,reason='Non-numeric HDF5 field/reference; not an image array'))
            yield from walk(f)
    else:
        try:
            headers=whosmat(path)
        except TypeError:
            values=loadmat(path,struct_as_record=False,squeeze_me=False)
            for name,value in values.items():
                if str(name).startswith('__'):continue
                if type(value).__name__=='MatlabOpaque':
                    skipped.append(dict(variable=str(name),reason='MATLAB MCOS/symbolic object; no directly accessible numeric image'))
                else:
                    yield from array_leaves(value,str(name),skipped)
            return
        for name,shape,dtype in headers:
            if dtype not in ['cell','struct'] and sum(n>=16 for n in shape)<2:
                skipped.append(dict(variable=name,shape=list(shape),reason='Metadata, scalar/vector or small parameter array'));continue
            value=loadmat(path,variable_names=[name],struct_as_record=False,squeeze_me=False)[name]
            yield from array_leaves(value,name,skipped)


def frames_of(array, spatial=None):
    a=np.asarray(array)
    if spatial is None:
        spatial=(0,1) if a.ndim>=2 and a.shape[0]>=16 and a.shape[1]>=16 else tuple(sorted(np.argsort(a.shape)[-2:].tolist()))
    extra=[i for i in range(a.ndim) if i not in spatial and a.shape[i]!=1]
    singleton=[i for i in range(a.ndim) if i not in spatial and a.shape[i]==1]
    arranged=a.transpose(*extra,*spatial,*singleton)
    frames=arranged.reshape(-1,a.shape[spatial[0]],a.shape[spatial[1]])
    axes=[dict(label=f'原数组轴 {i+1}',size=int(a.shape[i])) for i in extra] or [dict(label='图像',size=1)]
    return frames,axes,list(spatial)


def export_array(run, record, variable, frames, axes, fov, details, source_kind=None):
    key=record['sha256']+':'+variable
    eid=ident('archive',key)
    folder=run/'entries'/eid
    if (folder/'metadata.json').is_file():return public_entry(json.loads((folder/'metadata.json').read_text()))
    folder.mkdir(parents=True,exist_ok=True)
    complex_values=np.iscomplexobj(frames)
    scientific=np.asarray(np.abs(frames) if complex_values else frames,dtype=np.float32)
    count,h,w=scientific.shape
    np.savez_compressed(folder/'images.npz',image=scientific)
    signed=bool(np.any(scientific<0))
    finite=np.isfinite(scientific)
    invalid_count=int((~finite).sum())
    chunk_size=max(1,min(64,2_000_000//(h*w)))
    chunks=[];min_all=float(np.min(scientific,where=finite,initial=0));max_all=float(np.max(scientific,where=finite,initial=0))
    for start in range(0,count,chunk_size):
        block=scientific[start:start+chunk_size]
        good=np.isfinite(block);clean=np.where(good,block,np.nan)
        flat=clean.reshape(len(block),-1)
        with np.errstate(all='ignore'):
            low=np.nanmin(flat,axis=1) if signed else np.zeros(len(block))
            high=np.nanmax(flat,axis=1)
            p995=np.nanquantile(flat,.995,axis=1)
            p005=np.nanquantile(flat,.005,axis=1) if signed else np.zeros(len(block))
        low,high,p995,p005=[np.nan_to_num(a,nan=0,posinf=0,neginf=0).astype(np.float64) for a in [low,high,p995,p005]]
        span=np.maximum(high-low,1e-30)
        values=np.where(good,block,low[:,None,None])
        encoded=np.rint(np.clip((values-low[:,None,None])/span[:,None,None],0,1)*65535).astype('<u2')
        restored=encoded.astype(float)/65535*span[:,None,None]+low[:,None,None]
        err=np.abs(restored-values)
        tolerance=span[:,None,None]/65535/2+np.maximum(np.abs(values),1)*3e-7
        assert np.all(err<=tolerance)
        payload=dict(start=start,count=len(block),minima=low.tolist(),maxima=high.tolist(),p995=p995.tolist(),p005=p005.tolist(),
            data=base64.b64encode(encoded.tobytes()).decode())
        if not good.all():payload['invalid']=base64.b64encode(np.packbits(~good.ravel(),bitorder='little').tobytes()).decode()
        cname=f'chunk_{start//chunk_size:05d}.js';ckey=f'{eid}:{start//chunk_size}'
        (folder/cname).write_text('window.REVIEW_CHUNKS=window.REVIEW_CHUNKS||{};window.REVIEW_CHUNKS['+json.dumps(ckey)+']='+json.dumps(payload,separators=(',',':'))+';\n')
        chunks.append(f'entries/{eid}/{cname}')
    stage=dict(id='image',label=('复数幅度' if complex_values else '原数组数值'),axes=axes,shape=[count,h,w],fov=fov,
        note=details['display_note'],chunked=True,chunk_size=chunk_size,chunks=chunks,chunk_key=eid,
        signed=signed,global_window=max_all,global_lower=min_all,invalid_count=invalid_count)
    coords=[a['size']//2 if i==0 else 0 for i,a in enumerate(axes)]
    rep=int(np.ravel_multi_index(coords,[a['size'] for a in axes]))
    im=scientific[rep];valid=im[np.isfinite(im)]
    lo=float(np.quantile(valid,.005)) if signed and valid.size else 0
    hi=float(np.quantile(valid,.995)) if valid.size else 1
    pixels=np.nan_to_num(np.clip((im-lo)/max(hi-lo,1e-30),0,1),nan=0)
    preview=Image.fromarray(np.rint(pixels*255).astype('uint8')).convert('RGB')
    fy,fx=fov;width=max(1,round(min(288,236*fx/fy)));height=max(1,round(width*fy/fx))
    preview=preview.resize((width,height),Image.Resampling.NEAREST)
    thumb=Image.new('RGB',(304,260),'#080e17');thumb.paste(preview,((304-width)//2,(260-height)//2));thumb.save(folder/'thumbnail.jpg',quality=90)
    kind=source_kind or record['kind']
    titles={'mat':'MAT 数组','nifti':'NIfTI 影像','dicom':'DICOM 序列','raw':'原始 ADC','reference':'参考示例'}
    source_name=Path(record['source'].split('!/')[-1]).stem
    match=re.search(r'MID\d+',source_name)
    short=(match.group() if match else source_name[:52])
    title=details.get('title') or short+((' · '+variable[:40]) if kind=='mat' else '')
    notes=['这些条目按来源数组展示；不会自动与其他文件或处理阶段配对。',details['display_note']]
    if complex_values:notes.append('显示复数的绝对值；相位保留在来源文件，未冒充幅度图中的信息。')
    if invalid_count:notes.append(f'原数组包含 {invalid_count} 个 NaN/Inf；下载数组保留原值，显示时以灰色标记。')
    entry=dict(id=eid,scan=short,family='archive',group=kind,source_kind=kind,title=title,
        subtitle=details.get('subtitle') or source_name,source=record['source'],source_file=str(record['path']),
        source_mat=record['source'] if kind=='mat' else None,status='archive_'+kind,status_label=titles[kind],
        is_selected=False,frame_count=count,default_coords=coords,stages=[stage],
        geometry=dict(sequence=details.get('sequence','见来源元数据'),fov_mm=fov,fov_unit=details.get('fov_unit','px'),
            header_pe_ro=[h,w],thickness_mm=details.get('thickness_mm'),r_value=None),
        source_details=dict(**details,variable=variable,source_sha256=record['sha256'],complex_magnitude=complex_values,
            signed=signed,invalid_count=invalid_count),warnings=notes,
        payload=f'entries/{eid}/data.js',arrays=f'entries/{eid}/images.npz',metadata=f'entries/{eid}/metadata.json',thumbnail=f'entries/{eid}/thumbnail.jpg')
    (folder/'data.js').write_text('window.REVIEW_PAYLOADS=window.REVIEW_PAYLOADS||{};window.REVIEW_PAYLOADS['+json.dumps(eid)+']='+json.dumps([stage],ensure_ascii=False,separators=(',',':'))+';\n')
    dump(folder/'metadata.json',entry)
    return public_entry(entry)


def file_worker(record, run):
    path=Path(record['path']);kind=record['kind'];entries=[];skipped=[]
    try:
        if kind=='mat':
            for name,a in mat_arrays(path,skipped):
                frames,axes,spatial=frames_of(a)
                details=dict(original_shape=list(a.shape),display_axes_zero_based=spatial,
                    display_note=f'读取变量 {name}，显示原数组轴 {spatial[0]+1} × {spatial[1]+1}；其余非单例维度全部保留。未推断未知轴的解剖/coil/扩散含义。')
                if path.suffix.lower()=='.fig':
                    details['display_note']+=' MATLAB FIG 仅提取底层数值数组，不复刻绘图布局和色标；RGB 的三个分量逐轴浏览。已有彩色导出图另见旧导出图入口。'
                entries.append(export_array(run,record,name,frames,axes,list(frames.shape[1:]),details))
        elif kind=='nifti':
            ni=nib.load(path);a=np.asanyarray(ni.dataobj)
            if a.dtype.names:
                a=np.stack([a[n] for n in a.dtype.names],axis=-1)
            frames,axes,spatial=frames_of(a,spatial=(1,0))
            extra=[i for i in range(a.ndim) if i not in spatial and a.shape[i]!=1]
            for axis,i in zip(axes,extra):axis['label']='切片（原轴3）' if i==2 else f'附加轴 {i+1}'
            zoom=ni.header.get_zooms()
            unit=ni.header.get_xyzt_units()[0]
            scale={'mm':1,'meter':1000,'micron':.001}.get(unit,1)
            details=dict(original_shape=list(a.shape),affine=ni.affine.tolist(),zooms=list(map(float,zoom)),nifti_spatial_unit=unit,fov_unit='mm' if unit!='unknown' else 'header units',
                display_note='NIfTI 原始体素网格：轴2为纵轴、轴1为横轴，保留全部切片和附加维度；未重采样或自动统一解剖方向。')
            entries.append(export_array(run,record,'nifti',frames,axes,[a.shape[1]*zoom[1]*scale,a.shape[0]*zoom[0]*scale],details))
        elif kind=='raw':
            import twixtools
            scans=twixtools.read_twix(str(path),parse_geometry=False,parse_pmu=False,verbose=False)
            for sn,scan in enumerate(scans):
                mdbs=[m for m in scan['mdb'] if m.is_image_scan()]
                if not mdbs:
                    skipped.append(dict(variable=f'measurement {sn}',reason='No imaging MDH blocks (adjustment/calibration/non-image measurement)'));continue
                groups=defaultdict(dict);occurrences=Counter();counters_by_group={}
                counter_names=['Sli','Rep','Set','Ave','Eco','Par','Phs','Ida','Idb','Idc','Idd','Ide']
                for m in mdbs:
                    c=m.mdh.Counter;coords=tuple(int(getattr(c,n)) for n in counter_names);line=int(c.Lin)
                    shape=m.data.shape;polarity=int(m.is_flag_set('REFLECT'))
                    base=(coords,shape[0],shape[1]);occ=occurrences[(base,line)];occurrences[(base,line)]+=1
                    key=base+(occ,)
                    # Raw-domain coil RSS; no spatial reconstruction assumptions.
                    vector=np.sqrt((np.abs(m.data.astype('complex128'))**2).sum(0))
                    groups[key][line]=vector
                    counters_by_group[key]=dict(zip(counter_names,coords),inferred_line_occurrence=occ)
                byshape=defaultdict(list)
                for key,lines in groups.items():byshape[(max(lines)+1,key[2])].append((key,lines))
                seq=scan.get('hdr',{}).get('MeasYaps',{}).get('tSequenceFileName','未读取到序列名')
                for shape,items in byshape.items():
                    stack=[];info=[]
                    for key,lines in sorted(items,key=lambda item:item[0]):
                        frame=np.full(shape,np.nan,np.float32)
                        for line,v in lines.items():frame[line]=v
                        stack.append(frame);info.append(dict(**counters_by_group[key],acquired_lines=sorted(lines)))
                    frames=np.stack(stack);label=f'ADC_m{sn}_{shape[0]}x{shape[1]}'
                    details=dict(original_shape=list(frames.shape),sequence=seq,frame_counters=info,source_coils=sorted({k[1] for k,v in items}),
                        display_note='原始 ADC 的 coil RSS 幅度，纵轴为 Lin、横轴为采样点。保留过采样和记录极性，未做 RO FFT、regrid 或 PE 重建。相同 counter/line 的重复按出现顺序分组，不推断扩散方向；未采集行以灰色标记。')
                    entries.append(export_array(run,record,label,frames,[dict(label='采样组',size=len(frames))],list(shape),details))
        return dict(source=record['source'],status='displayed' if entries else 'no_image_array',entry_ids=[e['id'] for e in entries],skipped_fields=skipped),entries
    except Exception as error:
        return dict(source=record['source'],status='partially_displayed' if entries else 'read_error',entry_ids=[e['id'] for e in entries],reason=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),skipped_fields=skipped),entries


def dicom_worker(records,run):
    combined=dict(records[0]);combined['sha256']=hashlib.sha256(''.join(sorted(r['sha256'] for r in records)).encode()).hexdigest()
    cache=run/'entries'/ident('archive',combined['sha256']+':dicom_series')/'metadata.json'
    if cache.is_file():return public_entry(json.loads(cache.read_text()))
    arrays=[];info=[];reference=None
    for r in sorted(records,key=lambda r:(r.get('instance_number',0),r['source'])):
        ds=pydicom.dcmread(r['path']);a=ds.pixel_array
        if int(getattr(ds,'SamplesPerPixel',1))>1:
            if a.ndim==3:a=a[None]
            a=a.transpose(0,3,1,2).reshape(-1,*a.shape[1:3])
        elif a.ndim==2:a=a[None]
        assert a.ndim==3
        a=a.astype(np.float32)*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0))
        if str(getattr(ds,'PhotometricInterpretation',''))=='MONOCHROME1':
            # Preserve numerical values; viewer displays standard ascending intensity.
            pass
        arrays.extend(a);info.extend([dict(source=r['source'],instance_number=int(getattr(ds,'InstanceNumber',0)),
            image_position=list(map(float,getattr(ds,'ImagePositionPatient',[]))),image_orientation=list(map(float,getattr(ds,'ImageOrientationPatient',[]))))]*len(a))
        if reference is None:reference=ds
    frames=np.stack(arrays);spacing=list(map(float,getattr(reference,'PixelSpacing',[1,1])))
    description=str(getattr(reference,'SeriesDescription','DICOM series'))
    details=dict(title=description,subtitle=f'DICOM · {len(records)} 个文件',source_files=[r['source'] for r in records],
        original_shape=list(frames.shape),frame_metadata=info,series_uid=str(reference.SeriesInstanceUID),
        fov_unit='mm' if hasattr(reference,'PixelSpacing') else 'px',thickness_mm=float(getattr(reference,'SliceThickness',0)) or None,
        display_note='DICOM 像素（应用 RescaleSlope/Intercept），按 InstanceNumber 浏览全部帧；不把重复方向或 mosaic 自动拆成解剖切片。彩色像素按分量展开。')
    e=export_array(run,combined,'dicom_series',frames,[dict(label='序列帧',size=len(frames))],
        [frames.shape[1]*spacing[0],frames.shape[2]*spacing[1]],details,source_kind='dicom')
    return e


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--rescan',action='store_true',help='Refresh archive inventory while retaining successful exports')
    args=parser.parse_args()
    run=args.run.resolve();assert run.is_relative_to(HERE/'runs')
    inventory_file=run/'inventory/all_source_files.json'
    rows=json.loads(inventory_file.read_text()) if args.resume and not args.rescan and inventory_file.exists() else inventory(run)
    old=json.loads((run/'manifest.json').read_text())
    previous_file=run/'inventory/source_coverage.json'
    previous={r['source']:r for r in json.loads(previous_file.read_text())['files']} if args.resume and previous_file.exists() else {}
    old_entries={e['id']:public_entry(e) for e in old['entries'] if e['family']=='archive'}
    reports=[];entries=[];dicoms=defaultdict(list);tasks=[]
    for row in rows:
        if row['status']=='duplicate_file':reports.append(row);continue
        cached=previous.get(row['source'],{})
        ids=cached.get('entry_ids',[])
        if row['kind']!='dicom' and cached.get('status') in ['displayed','no_image_array','non_imaging_file'] and (not cached.get('sha256') or cached['sha256']==row['sha256']) and all(eid in old_entries for eid in ids):
            reports.append(cached);entries.extend(old_entries[eid] for eid in ids);continue
        if row['kind']=='raw':
            with Path(row['path']).open('rb') as f:head=f.read(256)
            # Sequence registration files and compilation assets have no MR image payload.
            if row['bytes']<32768 or Path(row['source']).name.lower()=='seq.dat':
                reports.append(dict(**row,status_new='non_imaging_file',reason='Sequence/code registration or tiny non-image data file'));continue
        if row['kind']=='dicom':
            try:
                ds=pydicom.dcmread(row['path'],stop_before_pixels=True)
                if not hasattr(ds,'Rows') or not hasattr(ds,'Columns'):
                    reports.append(dict(**row,status_new='non_imaging_dicom',reason='No Rows/Columns image matrix'));continue
                key=(str(ds.SeriesInstanceUID),int(ds.Rows),int(ds.Columns),str(getattr(ds,'PhotometricInterpretation','')))
                row['instance_number']=int(getattr(ds,'InstanceNumber',0));dicoms[key].append(row)
            except Exception as error:reports.append(dict(**row,status_new='read_error',reason=str(error)))
        else:tasks.append(row)
    print('Unique file tasks:',len(tasks),'DICOM series:',len(dicoms),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(file_worker,row,run):row for row in tasks}
        for n,future in enumerate(as_completed(futures),1):
            report,added=future.result();reports.append(report);entries.extend(added)
            dump(run/'inventory/catalog_progress.json',dict(completed=n,total=len(tasks),entries=len(entries),reports=reports))
            if n%10==0 or report['status'] in ['read_error','partially_displayed']:
                print(f'{n}/{len(tasks)} files, {len(entries)} entries: {Path(report["source"]).name} {report["status"]}',flush=True)
    for n,(key,series) in enumerate(dicoms.items(),1):
        try:
            entry=dicom_worker(series,run);entries.append(entry)
            reports.extend(dict(source=r['source'],status='displayed',entry_ids=[entry['id']]) for r in series)
        except Exception as error:
            reports.extend(dict(source=r['source'],status='read_error',reason=str(error)) for r in series)
        if n%10==0:print(f'DICOM series {n}/{len(dicoms)}',flush=True)
    # Stable file aliases point to the same displayed arrays; preserve every source path.
    bysource={r['source']:r for r in reports if r.get('status')!='duplicate_file'}
    for r in reports:
        if r.get('status_new'):r['status']=r.pop('status_new')
        if r.get('status')=='duplicate_file':r['entry_ids']=bysource.get(r['duplicate_of'],{}).get('entry_ids',[])
    byrow={r['source']:r for r in rows}
    reports=[dict(byrow[r['source']],**r) for r in reports]
    entries=sorted({e['id']:public_entry(e) for e in entries}.values(),key=lambda e:(['mat','nifti','dicom','raw'].index(e['group']),e['title']))
    base=[e for e in old['entries'] if e['family']!='archive'];all_entries=base+entries
    summary=dict(old['summary'],archive_entries=len(entries),entry_count=len(all_entries),
        dataset_count=sum(not e.get('is_selected') for e in all_entries),archive_frames=sum(e['frame_count'] for e in entries),
        catalog_roots=[str(ROOT)],source_files=len(rows),source_status_counts=dict(Counter(r['status'] for r in reports)))
    manifest=dict(summary=summary,entries=all_entries)
    dump(run/'manifest.json',manifest);(run/'manifest.js').write_text('window.REVIEW_MANIFEST='+json.dumps(manifest,ensure_ascii=False)+';\n')
    dump(run/'inventory/source_coverage.json',dict(scope=str(ROOT),summary=summary,files=reports,
        outside_scope='Separate training caches and repeated algorithm experiment outputs in sibling work projects are not counted as new acquisitions. Current verified reconstruction comparisons remain included.'))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
