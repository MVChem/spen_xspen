#!/usr/bin/env python3
"""Process every inventory record and every decoded slice/volume/echo.

Each scan has an independently recoverable case.json. Failures never hide
other frames, and a resume only reuses cases whose saved arrays still exist.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent/'spenpy'))


def now():
    return datetime.now().astimezone().isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def initialize_worker(data_root, run_dir, lambda_relative):
    global DATA, RUN, LAMBDA, FILES, MATS
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    DATA, RUN, LAMBDA = Path(data_root), Path(run_dir), lambda_relative
    raw = json.loads((DATA/'raw_manifest.json').read_text())
    FILES = {r['relative_path']: r for r in raw['records']}
    MATS = {r['raw_scan_path']: r for r in json.loads((DATA/'manifest.json').read_text())['records']}


class TikhonovCache:
    def __init__(self, lambda_relative):
        self.lambda_relative = lambda_relative
        self.entries = {}

    def solve(self, encoding, observation):
        a = np.asarray(encoding, dtype=np.complex128)
        y = np.asarray(observation, dtype=np.complex128)
        key = hashlib.sha256(a.tobytes()).hexdigest()
        if key not in self.entries:
            u, s, vh = np.linalg.svd(a, full_matrices=False)
            smax = float(s[0])
            if not np.isfinite(smax) or smax <= 0:
                raise ValueError('Invalid encoding spectral norm')
            alpha = self.lambda_relative*smax*smax
            inverse = (vh.conj().T*(s/(s*s+alpha))) @ u.conj().T
            self.entries[key] = (inverse, alpha, smax)
        inverse, alpha, smax = self.entries[key]
        rhs = y.reshape(y.shape[0], -1)
        x = inverse @ rhs
        prediction = a @ x
        residual = a.conj().T @ (prediction-rhs)+alpha*x
        normal_relative = float(np.linalg.norm(residual)/max(np.linalg.norm(a.conj().T@rhs), 1e-30))
        if not np.isfinite(x).all() or normal_relative > 1e-9:
            raise FloatingPointError(f'Tikhonov normal residual {normal_relative}')
        return x.reshape(a.shape[1], *y.shape[1:]), {
            'lambda_relative': self.lambda_relative, 'lambda_absolute': alpha,
            'encoding_spectral_norm': smax, 'solver': 'cached complex128 SVD identity L2 per coil',
            'relative_normal_residual': normal_relative,
            'relative_residual': float(np.linalg.norm(prediction-rhs)/max(np.linalg.norm(rhs), 1e-30)),
        }


def reference_comparison(reference, arrays, indices, counts):
    sl, vol, echo = indices
    # Historical MAT stores only the last processed echo.
    if echo != counts['echoes']-1:
        return {'status': 'not_applicable', 'reason': 'Historical export retains only its final echo'}
    comparisons = {}
    for key, old_key in [('rofft_original','spen_original_signal_rofft'),
                         ('rofft_corrected','spen_phase_corrected_signal_rofft'),
                         ('inva_corrected','traditional_sr_data')]:
        if key not in arrays or old_key not in reference:
            continue
        old = reference[old_key]
        shape = old.shape[:2] + (counts['slices'], counts['volumes'], counts['coils'])
        try:
            expected = old[:,:,0,:].reshape(shape, order='F')[:,:,sl,vol,:]
            if expected.shape != arrays[key].shape:
                raise ValueError('Spatial shape differs from historical export')
            error = float(np.linalg.norm(arrays[key]-expected)/max(np.linalg.norm(expected),1e-30))
            comparisons[old_key] = {'relative_l2_error': error, 'passed': error < 1e-6}
        except ValueError as exc:
            comparisons[old_key] = {'passed': False, 'error': str(exc)}
    return {'status': 'passed' if comparisons and all(x['passed'] for x in comparisons.values()) else 'mismatch', 'arrays': comparisons}


def read_scanner_stack(scan_dir, case_dir, case):
    from spenpy._legacy.bruker.image import read_bruker_2dseq
    pdata = scan_dir/'pdata'
    paths = sorted(pdata.glob('*/2dseq'), key=lambda p:(p.parent.name!='1', p.parent.name)) if pdata.is_dir() else []
    failures = []
    for payload in paths:
        if not payload.stat().st_size or not (payload.parent/'visu_pars').exists():
            continue
        try:
            array = read_bruker_2dseq(str(payload.parent))
            if not np.isfinite(array).all() or array.ndim < 2:
                raise ValueError(f'Invalid scanner preview shape {array.shape}')
            for source in [payload, payload.parent/'visu_pars']:
                rel = str(source.relative_to(DATA))
                h = digest(source)
                if h != FILES[rel]['sha256']:
                    raise ValueError(f'Scanner source hash changed: {rel}')
                case['source_sha256'][rel] = h
            np.save(case_dir/'scanner_2dseq.npy', array)
            case['scanner_preview'] = {'path': str(payload.parent.relative_to(DATA)),
                'shape': list(array.shape), 'frame_mapping': 'Native scanner frame order; no registration or raw-frame equivalence claimed'}
            return array
        except Exception as exc:
            failures.append({'path':str(payload.relative_to(DATA)),'error':str(exc)})
    if failures:
        case['scanner_preview_errors'] = failures
    return None


def append_scanner_only(scanner, case, arrays_dir):
    if scanner is None:
        return
    stack = scanner.reshape(*scanner.shape[:2], -1, order='F')
    for index in range(stack.shape[2]):
        target = arrays_dir/f'scanner_frame_{index:05d}.npz'
        np.savez_compressed(target, scanner_preview=np.abs(stack[:,:,index]))
        case['frames'].append({'id':f"{case['id']}_scanner_{index:05d}",
            'frame_type':'scanner_preview', 'scanner_frame_index':index,
            'slice_index':None, 'volume_index':None, 'echo_index':None,
            'status':'scanner_preview_only', 'reason':case.get('reason','Raw reconstruction not available'),
            'arrays_path':str(target.relative_to(RUN))})


def process_record(record):
    from scipy.io import loadmat
    import raw_frame_core
    case_dir = RUN/'cases'/record['id']
    arrays_dir = case_dir/'arrays'
    arrays_dir.mkdir(parents=True, exist_ok=True)
    scan_dir = DATA/record['raw_scan_path']
    case = {**record, 'label':record.get('label',f"{record['experiment_name']} / scan {record['scan_id']}"),
            'status':'running','frames':[],'source_sha256':{},'started_at':now(),
            'source_core_sha256':digest(Path(raw_frame_core.__file__)),
            'source_collection_sha256':digest(Path(__file__))}
    output = case_dir/'case.json'
    if output.exists():
        old = json.loads(output.read_text())
        case['previous_attempt'] = {k:old.get(k) for k in ['status','reason','started_at','finished_at','source_core_sha256']}
    scanner = None
    reference = None
    try:
        for rel in [record['raw_scan_path']+'/method',record['raw_scan_path']+'/acqp']:
            if rel in FILES and (DATA/rel).exists():
                h = digest(DATA/rel)
                if h != FILES[rel]['sha256']:
                    raise ValueError(f'Changed parameter source: {rel}')
                case['source_sha256'][rel] = h
        for payload in record.get('raw_payloads',[]):
            filename = payload.get('file',payload.get('name'))
            rel = record['raw_scan_path']+'/'+filename
            h = digest(DATA/rel)
            if h != FILES[rel]['sha256']:
                raise ValueError(f'Changed raw source: {rel}')
            case['source_sha256'][rel] = h
        scanner = read_scanner_stack(scan_dir, case_dir, case)
        if record['raw_status'] != 'nonempty':
            case['status'] = 'raw_unavailable'
            case['reason'] = f"Source raw status: {record['raw_status']}"
            append_scanner_only(scanner,case,arrays_dir)
            return finish_case(case,output)
        ref_record = MATS.get(record['raw_scan_path'])
        if ref_record:
            case['reference_mat_path'] = ref_record['mat_path']
            if digest(DATA/ref_record['mat_path']) != ref_record['mat_sha256']:
                raise ValueError('Historical MAT hash changed')
            reference = loadmat(DATA/ref_record['mat_path'])
        trajectory = record.get('trajectory_scan_id')
        trajectory_dir = scan_dir.parent/str(trajectory) if trajectory is not None and int(trajectory)>=0 else scan_dir
        for source in [trajectory_dir/'method',trajectory_dir/'acqp']:
            rel = str(source.relative_to(DATA))
            if source.is_file() and rel in FILES:
                h = digest(source)
                if h != FILES[rel]['sha256']:
                    raise ValueError('Trajectory source hash changed')
                case['source_sha256'][rel] = h
        context = raw_frame_core.prepare_scan(str(scan_dir),
            regrid_flavor=record.get('regrid_flavor') or 'pv360',
            trajectory_scan_dir=str(trajectory_dir))
        counts = context['counts']
        case['decoded_counts'] = plain(counts)
        case['raw_shape'] = list(context['sorted_samples'].shape)
        case['raw_axes'] = ['readout','spen','slice','volume','coil','echo']
        case['parameters'] = {**case.get('parameters',{}),**plain(context.get('parameters',{})),**plain(counts)}
        case['scope'] = plain(context.get('scope',{}))
        case['reason'] = context.get('scope',{}).get('reason','')
        case['raw_reader'] = context.get('reader')
        if context.get('reader_info'):
            case['raw_reader_info'] = plain(context['reader_info'])
        if context.get('dimension_correction'):
            case['dimension_correction'] = plain(context['dimension_correction'])
        case['decoded_frames'] = counts['slices']*counts['volumes']*counts['echoes']
        if record.get('expected_frames') is not None and case['decoded_frames'] != record['expected_frames']:
            case['frame_count_warning'] = {'expected':record['expected_frames'],'decoded':case['decoded_frames']}
        cache = TikhonovCache(LAMBDA)
        for echo in range(counts['echoes']):
            for volume in range(counts['volumes']):
                for sl in range(counts['slices']):
                    frame_id = f"{record['id']}_s{sl:03d}_v{volume:03d}_e{echo:02d}"
                    frame = {'id':frame_id,'slice_index':sl,'volume_index':volume,'echo_index':echo,'frame_type':'raw_reconstruction'}
                    try:
                        arrays, metadata = raw_frame_core.reconstruct_frame(context,sl,volume,echo)
                        arrays = {k:np.asarray(v) for k,v in arrays.items() if isinstance(v,(np.ndarray,list,tuple)) or hasattr(v,'shape')}
                        frame['reconstruction'] = plain(metadata)
                        if 'encoding' in arrays:
                            y = arrays.get('rofft_corrected',arrays.get('rofft_original'))
                            arrays['tikhonov_coils'], frame['tikhonov'] = cache.solve(arrays['encoding'],y)
                        completed = 'inva_corrected' in arrays and 'tikhonov_coils' in arrays
                        frame['status'] = 'completed' if completed else ('partial_reconstruction' if 'tikhonov_coils' in arrays or 'inva_uncorrected'in arrays else 'preview_only')
                        frame['reason'] = metadata.get('reason',context.get('scope',{}).get('reason',''))
                        if reference is not None and completed:
                            frame['mat_regression'] = reference_comparison(reference,arrays,(sl,volume,echo),counts)
                        if scanner is not None and scanner.ndim==3 and scanner.shape[-1]==case['decoded_frames']:
                            index=sl+counts['slices']*(volume+counts['volumes']*echo)
                            arrays['scanner_preview'] = np.abs(scanner[:,:,index])
                            frame['scanner_frame_index'] = index
                        if not arrays or not all(np.isfinite(a).all() for a in arrays.values()):
                            raise FloatingPointError('Nonfinite or empty frame output')
                        filename = arrays_dir/f's{sl:03d}_v{volume:03d}_e{echo:02d}.npz'
                        np.savez_compressed(filename,**arrays)
                        frame['arrays_path']=str(filename.relative_to(RUN))
                        frame['array_shapes']={k:list(v.shape) for k,v in arrays.items()}
                    except Exception as exc:
                        frame.update(status='failed',reason=str(exc),traceback=traceback.format_exc())
                    case['frames'].append(frame)
                    if len(case['frames'])%10==0:
                        write_json(output,plain(case))
        statuses=Counter(f['status'] for f in case['frames'])
        case['status']=('completed' if statuses.get('completed',0)==case['decoded_frames']
                        else 'failed' if statuses.get('failed',0) else 'completed_with_limitations')
        case['frame_status_counts']=dict(statuses)
        return finish_case(case,output)
    except Exception as exc:
        case.update(status='failed',reason=str(exc),traceback=traceback.format_exc())
        if not case['frames']:
            append_scanner_only(scanner,case,arrays_dir)
        return finish_case(case,output)


def finish_case(case,output):
    case['finished_at']=now()
    write_json(output,plain(case))
    return {k:case.get(k) for k in ['id','status','reason','decoded_frames','frame_status_counts']}


def collect_summary(out,inventory,settings,status):
    cases=[]
    for r in inventory['records']:
        path=out/'cases'/r['id']/'case.json'
        if path.exists():
            cases.append(json.loads(path.read_text()))
    frames=[f for c in cases for f in c.get('frames',[])]
    result={'status':status,'updated_at':now(),'data_root':settings['data_root'],
        'run_dir':str(out),'planned_records':len(inventory['records']),
        'inventory_summary':inventory.get('summary',{}),'lambda_relative':settings['lambda_relative'],
        'case_status_counts':dict(Counter(c['status'] for c in cases)),
        'frame_status_counts':dict(Counter(f['status'] for f in frames)),
        'frame_count':len(frames),'cases':cases,
        'failures':[{'id':c['id'],'experiment_name':c['experiment_name'],'scan_id':c['scan_id'],'status':c['status'],'reason':c.get('reason','')} for c in cases if c['status'] in ('failed','raw_unavailable')],
        'scope':'All inventory SPEN/xSPEN records; every decoded slice, volume and echo is processed. Preview-only and incomplete methods remain explicitly labelled.'}
    write_json(out/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=PROJECT.parent/'data/spen_acquired_260915')
    parser.add_argument('--out',type=Path,default=PROJECT/'runs/all_raw_260915')
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--lambda-relative',type=float,default=.01)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--retry-status',nargs='+',default=['failed','worker_failed','running'])
    parser.add_argument('--only-ids-file', type=Path, help='JSON list of scan IDs to recompute while retaining the full inventory')
    args=parser.parse_args()
    if args.workers<1 or not np.isfinite(args.lambda_relative) or args.lambda_relative<=0:
        parser.error('Positive workers and lambda required')
    out=args.out.resolve();data=args.data.resolve()
    if out==data or out.is_relative_to(data):
        parser.error('Output cannot be within input data')
    if out.exists() and any(out.iterdir()) and not args.resume:
        parser.error('Use a fresh output folder or --resume')
    out.mkdir(parents=True,exist_ok=True)
    inventory=json.loads(args.inventory.read_text())
    records=inventory['records']
    if len({r['id'] for r in records})!=len(records):
        parser.error('Duplicate scan IDs')
    settings={'data_root':str(data),'inventory_path':str(args.inventory.resolve()),
        'inventory_sha256':digest(args.inventory),'lambda_relative':args.lambda_relative,
        'workers':args.workers,'started_at':now(),'python':sys.executable,'resume':args.resume}
    prior=out/'settings.json'
    if args.resume and prior.exists():
        saved=json.loads(prior.read_text())
        for key in ['data_root','inventory_sha256','lambda_relative']:
            if saved[key]!=settings[key]:
                parser.error(f'Resume setting differs: {key}')
        settings['previous_started_at']=saved.get('started_at')
    write_json(prior,settings)
    if args.inventory.resolve()!=out/'inventory.json':
        shutil.copy2(args.inventory,out/'inventory.json')
    source=out/'source'/datetime.now().strftime('%Y%m%d_%H%M%S')
    source.mkdir(parents=True,exist_ok=True)
    for name in ['run_collection.py','raw_frame_core.py','segmented_raw_reader.py','batched_gridding.py','inventory_all.py']:
        shutil.copy2(PROJECT/'scripts'/name,source/name)
    pending=[]
    only_ids = None if args.only_ids_file is None else set(json.loads(args.only_ids_file.read_text()))
    if only_ids is not None:
        if not args.resume:
            parser.error('--only-ids-file requires --resume on the complete inventory')
        if only_ids - {r['id'] for r in records}:
            parser.error('--only-ids-file includes unknown IDs')
        settings['selected_case_ids'] = sorted(only_ids)
        write_json(prior, settings)
    for r in records:
        if only_ids is not None and r['id'] not in only_ids:
            continue
        case_path=out/'cases'/r['id']/'case.json'
        reuse=False
        if args.resume and case_path.exists():
            c=json.loads(case_path.read_text())
            reuse=c['status'] not in args.retry_status and all((out/f['arrays_path']).exists() for f in c.get('frames',[]) if 'arrays_path'in f)
        if not reuse:pending.append(r)
    # One-level process parallelism; no GPU and no nested BLAS pools.
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key]='1'
    started=time.monotonic();finished=0
    write_json(out/'status.json',{'status':'running','pending_scans':len(pending),'reused_scans':len(records)-len(pending),'completed_this_invocation':0})
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
        initializer=initialize_worker,initargs=(str(data),str(out),args.lambda_relative)) as pool:
        futures={pool.submit(process_record,r):r for r in pending}
        for future in as_completed(futures):
            r=futures[future]
            try:result=future.result()
            except Exception as exc:
                result={'id':r['id'],'status':'worker_failed','reason':str(exc)}
                folder=out/'cases'/r['id'];folder.mkdir(parents=True,exist_ok=True)
                write_json(folder/'case.json',{**r,**result,'frames':[]})
            finished+=1
            print(json.dumps({'event':'scan_complete','completed':finished,'total':len(pending),**result}),flush=True)
            write_json(out/'status.json',{'status':'running','pending_scans':len(pending),
                'reused_scans':len(records)-len(pending),'completed_this_invocation':finished,
                'last_case':result,'elapsed_seconds':time.monotonic()-started})
            if finished%20==0:
                collect_summary(out,inventory,settings,'running')
    summary=collect_summary(out,inventory,settings,'finished')
    write_json(out/'status.json',{'status':'finished','processed_records':len(summary['cases']),
        'planned_records':len(records),'frame_count':summary['frame_count'],
        'case_status_counts':summary['case_status_counts'],'frame_status_counts':summary['frame_status_counts'],
        'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'event':'collection_finished','cases':len(summary['cases']),
        'frames':summary['frame_count'],'case_status_counts':summary['case_status_counts'],
        'frame_status_counts':summary['frame_status_counts']}),flush=True)


if __name__=='__main__':
    main()
