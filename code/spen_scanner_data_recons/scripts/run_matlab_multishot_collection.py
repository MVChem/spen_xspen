#!/usr/bin/env python3
"""Complete supported multi-shot PhaseMap frames through the archived MATLAB code.

Frames retain their collection IDs and array paths. Later odd-shot echoes use
the automatic mask from echo 1 of the SAME slice and volume. Every MATLAB log,
input, output and source snapshot is retained before updating the collection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from scipy.io import loadmat, savemat

from run_collection import TikhonovCache, collect_summary, write_json, now

PROJECT = Path(__file__).resolve().parents[1]
MATLAB = Path('/usr/local/MATLAB/R2024a/bin/matlab')
LEGACY = Path('/home/data1/musong/workspace/2026/03/17/spen_matlab/spen')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def matlab_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def snapshot_sources(work, archive):
    records = []
    for name, source in [('legacy', LEGACY), ('archive', archive),
                         ('bridge', PROJECT/'scripts/matlab_multishot_bridge')]:
        for path in sorted(source.rglob('*')):
            if not path.is_file() or path.suffix not in ('.m', '.p', '.mexa64'):
                continue
            local = work/'source'/name/path.relative_to(source)
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, local)
            records.append({'source':str(path), 'local':str(local.relative_to(work)), 'sha256':sha(local)})
    runner = work/'source'/Path(__file__).name
    shutil.copy2(__file__, runner)
    records.append({'source':str(Path(__file__).resolve()), 'local':str(runner.relative_to(work)), 'sha256':sha(runner)})
    write_json(work/'source_manifest.json', {'created_at':now(), 'records':records})


def prepare_jobs(run, work, workers):
    groups, skipped = [], []
    for case_path in sorted((run/'cases').glob('*/case.json')):
        case = json.loads(case_path.read_text())
        params = case.get('parameters', {})
        shots = int(params.get('n_segments', 1))
        if shots <= 1 or case.get('classification') != 'spen_imaging':
            continue
        if case.get('raw_status') != 'nonempty':
            continue
        case_jobs = []
        for frame in sorted(case.get('frames', []), key=lambda f:(f.get('slice_index',0), f.get('volume_index',0), f.get('echo_index',0))):
            if frame.get('status') == 'completed':
                continue
            reason = None
            path = run/frame['arrays_path'] if frame.get('arrays_path') else None
            if path is None or not path.exists():
                reason = 'No decoded raw frame array'
            else:
                with np.load(path) as stored:
                    arrays = {k:stored[k] for k in stored.files}
                observation = arrays.get('rofft_original')
                if observation is None or 'encoding' not in arrays:
                    reason = 'No calibrated quadratic-SPEN observation and encoding matrix'
                elif observation.shape[0] % (2*shots):
                    reason = 'Original multi-shot PhaseMap requires an even number of PE samples per shot'
            if reason:
                skipped.append({'case_id':case['id'], 'frame_id':frame['id'], 'reason':reason})
                continue
            sl, vol, echo = (int(frame[k]) for k in ('slice_index','volume_index','echo_index'))
            base = f's{sl:03d}_v{vol:03d}_e{echo:02d}'
            folder = work/'frames'/case['id']
            folder.mkdir(parents=True, exist_ok=True)
            output_mat = folder/(base+'_output.mat')
            input_mat = folder/(base+'_input.mat')
            # Existing collection ROFFT already contains the recorded even-echo
            # PE reversal. Inverse RO FFT only restores the stage expected by
            # the archived scripts; no additional orientation is introduced.
            cmplx = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(observation, axes=1), axis=1), axes=1)
            lpe = float(params['fov_mm'][1])/10
            coefficient = -2*np.pi*4257.4*float(params['spen_gy_gauss_cm'])*float(params['spatial_encoding_duration_ms'])/1000/lpe
            if echo % 2:
                coefficient = -coefficient
            payload = {
                'CmplxData':cmplx[:, :, None, :], 'NumShots':shots,
                'LPE':lpe, 'LRO':float(params['fov_mm'][0])/10,
                'a_rad2cmsqr':coefficient, 'ShiftPE':float(params['phase1_offset_mm']),
                'regrid_flavor':params.get('regrid_flavor',case.get('regrid_flavor','pv360')),
                'source_scan':str(Path(json.loads((run/'settings.json').read_text())['data_root'])/case['raw_scan_path']),
                'slice_index':sl, 'volume_index':vol, 'echo_index':echo,
                'first_echo_output':str(folder/f's{sl:03d}_v{vol:03d}_e00_output.mat'),
            }
            savemat(input_mat, payload)
            job = {'case_id':case['id'], 'case_json':str(case_path), 'frame_id':frame['id'],
                   'arrays_path':frame['arrays_path'], 'arrays_sha256_before':sha(path),
                   'input_mat':str(input_mat), 'input_sha256':sha(input_mat), 'output_mat':str(output_mat),
                   'result_json':str(folder/(base+'_result.json')),
                   'log_path':str(folder/(base+'_matlab.log')),
                   'slice_index':sl,'volume_index':vol,'echo_index':echo,
                   'n_segments':shots,'shape':list(observation.shape)}
            case_jobs.append(job)
        if case_jobs:
            cost = sum(j['shape'][0]**3*j['shape'][2] for j in case_jobs)
            groups.append((cost,case_jobs))
    buckets = [[] for _ in range(workers)]
    loads = [0]*workers
    # A complete case stays on one process, so same-frame echo-1 masks always
    # exist before the corresponding later echo is attempted.
    for cost, group in sorted(groups, key=lambda item:item[0], reverse=True):
        index = int(np.argmin(loads))
        buckets[index].extend(group)
        loads[index] += cost
    return buckets, skipped


def run_worker(index, jobs, work, matlab):
    job_path = work/f'worker_{index:02d}_jobs.json'
    write_json(job_path, jobs)
    expression = ('addpath(' + matlab_quote(work/'source/bridge') + '); run_multishot_batch('
                  + ','.join(map(matlab_quote,[job_path,work/'source/legacy',work/'source/archive'])) + ');')
    log = work/f'worker_{index:02d}.log'
    preferences = work/f'worker_{index:02d}_preferences'
    preferences.mkdir(exist_ok=True)
    started = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.run([str(matlab),'-batch',expression],stdout=stream,stderr=subprocess.STDOUT,
                                 env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                                      'MATLAB_PREFDIR':str(preferences)})
    return {'worker':index,'returncode':process.returncode,'elapsed_seconds':time.monotonic()-started,
            'log':str(log.relative_to(work)),'jobs':len(jobs)}


def matlab_string(value):
    return str(np.asarray(value).reshape(-1)[0]) if np.size(value) else ''


def update_collection(run, work, jobs, lambda_relative):
    cache = TikhonovCache(lambda_relative)
    cases = {}
    updated, failures, verification = [], [], []
    for job in jobs:
        result_path = Path(job['result_json'])
        result = json.loads(result_path.read_text()) if result_path.exists() else {'status':'failed','error':'MATLAB process produced no frame result'}
        if result['status'] != 'completed':
            failures.append({**job,'matlab_result':result})
            continue
        try:
            arrays_path = run/job['arrays_path']
            if sha(arrays_path) != job['arrays_sha256_before']:
                raise RuntimeError('Collection frame changed after MATLAB input preparation; refusing stale replacement')
            with np.load(arrays_path) as stored:
                arrays = {k:stored[k] for k in stored.files}
            matlab = loadmat(job['output_mat'])
            shape = tuple(job['shape'])
            original = matlab['roffted_original'].reshape(shape)
            corrected = matlab['roffted_corrected'].reshape(shape)
            inva = matlab['inva_corrected'].reshape(shape)
            encoding = matlab['encoding']
            weighted_adjoint = matlab['inva_weighted_adjoint']
            input_error = float(np.linalg.norm(original-arrays['rofft_original'])/max(np.linalg.norm(original),1e-30))
            encoding_error = float(np.linalg.norm(encoding-arrays['encoding'])/max(np.linalg.norm(encoding),1e-30))
            check_inva = np.einsum('ij,jrc->irc',weighted_adjoint,corrected)
            equation_error = float(np.linalg.norm(inva-check_inva)/max(np.linalg.norm(inva),1e-30))
            magnitude_error = float(np.linalg.norm(np.abs(corrected)-np.abs(original))/max(np.linalg.norm(original),1e-30))
            if input_error > 1e-6 or encoding_error > 1e-6 or equation_error > 1e-9 or magnitude_error > 1e-5:
                raise ValueError(f'MATLAB numerical invariant failed: input={input_error}, A={encoding_error}, InvA={equation_error}, amplitude={magnitude_error}')
            if 'tikhonov_coils' in arrays:
                arrays['tikhonov_uncorrected'] = arrays['tikhonov_coils']
            arrays.update(rofft_corrected=corrected,inva_corrected=inva,
                          encoding=encoding,inva_weighted_adjoint=weighted_adjoint)
            arrays['inva_uncorrected'] = np.einsum('ij,jrc->irc',weighted_adjoint,arrays['rofft_original'])
            arrays['tikhonov_coils'], tikhonov = cache.solve(encoding,corrected)
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise FloatingPointError('Nonfinite MATLAB or Tikhonov result')
            temporary = arrays_path.with_suffix('.matlab.tmp.npz')
            np.savez_compressed(temporary,**arrays)
            temporary.replace(arrays_path)
            case_path = Path(job['case_json'])
            if case_path not in cases:
                cases[case_path] = json.loads(case_path.read_text())
                backup = work/'before_cases'/case_path.parent.name/'case.json'
                backup.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(case_path,backup)
            case = cases[case_path]
            frame = next(f for f in case['frames'] if f['id'] == job['frame_id'])
            previous_status = frame['status']
            frame.update(status='completed',reason='',tikhonov=tikhonov)
            frame['array_shapes'] = {k:list(v.shape) for k,v in arrays.items()}
            phase_info = {
                'method':matlab_string(matlab['phase_method']),
                'effective_gauss_relative_width':float(matlab['effective_gauss_relative_width'].reshape(-1)[0]),
                'manual_mask_used':False,
                'first_echo_mask_source':(str(Path(job['input_mat']).parent/f"s{job['slice_index']:03d}_v{job['volume_index']:03d}_e00_output.mat")
                                          if job['n_segments']%2 and job['echo_index']>0 else None),
                'optimization_budget_reached':result.get('optimization_budget_reached',False),
                'phase_fit_fallback':result.get('phase_fit_fallback',False),
                'badly_conditioned_polynomial':result.get('badly_conditioned_polynomial',False),
                'log_path':str(Path(job['log_path']).relative_to(run)),
                'output_mat':str(Path(job['output_mat']).relative_to(run)),
                'source_manifest':str((work/'source_manifest.json').relative_to(run)),
                'previous_frame_status':previous_status,
                'input_relative_error':input_error,'encoding_relative_error':encoding_error,
                'inva_equation_relative_error':equation_error,'phase_magnitude_relative_change':magnitude_error,
                'elapsed_seconds':result.get('elapsed_seconds'),
                'matlab_version':result.get('matlab_version'),
                'input_mat_sha256':job['input_sha256'],
                'output_mat_sha256':sha(job['output_mat']),
            }
            frame['matlab_phase'] = phase_info
            warnings = list(frame.get('quality_warnings',[]))
            for flag, message in (
                ('optimization_budget_reached','MATLAB 相位优化达到原始计算预算，使用当前最优结果，未确认收敛'),
                ('phase_fit_fallback','MATLAB 部分相位拟合失败，原始脚本将对应多项式系数置零'),
                ('badly_conditioned_polynomial','MATLAB 相位多项式拟合存在病态条件警告')):
                if phase_info[flag] and message not in warnings:
                    warnings.append(message)
            frame['quality_warnings'] = warnings
            frame['reconstruction'].update(phase_map_status='applied_matlab',
                reconstruction_status='phase_map_inva',scope_reason='',
                phase_backend='archived MATLAB multi-shot scripts')
            verification.append({'frame_id':frame['id'],**phase_info,
                                 'tikhonov_relative_normal_residual':tikhonov['relative_normal_residual']})
            updated.append(job['frame_id'])
        except Exception as exc:
            failures.append({**job,'verification_error':str(exc),'matlab_result':result})
    for path, case in cases.items():
        counts = Counter(f['status'] for f in case['frames'])
        case['frame_status_counts'] = dict(counts)
        case['status'] = 'completed' if set(counts)=={'completed'} else 'completed_with_limitations'
        case['matlab_phase_backend'] = 'archived MATLAB scripts; per-frame source, diagnostics and masks retained'
        if case['status'] == 'completed':
            case['reason'] = ''
            case['scope'].update(phase_map_supported=True,status='phase_map_inva',reason='',phase_backend='matlab')
        case['finished_at'] = now()
        write_json(path,case)
    settings = json.loads((run/'settings.json').read_text())
    inventory = json.loads((run/'inventory.json').read_text())
    collect_summary(run,inventory,settings,'finished')
    return updated,failures,verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=PROJECT/'runs/all_raw_260915')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--matlab',type=Path,default=MATLAB)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    settings = json.loads((run/'settings.json').read_text())
    if (run/'status.json').exists() and json.loads((run/'status.json').read_text()).get('status') != 'finished':
        parser.error('Wait for the Python collection run to finish before starting MATLAB updates')
    if args.workers < 1:
        parser.error('--workers must be positive')
    work = run/'matlab_multishot_260915'
    if work.exists():
        parser.error(f'Use a fresh bridge work directory; already exists: {work}')
    work.mkdir()
    archive = Path(settings['data_root'])/'raw_spectroscopy/20240229_190150_cts_240229_multi_delay_spec_1_1/SPENReco'
    snapshot_sources(work,archive)
    buckets, skipped = prepare_jobs(run,work,args.workers)
    jobs = [job for bucket in buckets for job in bucket]
    summary = {'status':'prepared','created_at':now(),'run_dir':str(run),
               'workers':args.workers,'prepared_frames':len(jobs),'skipped_frames':len(skipped),
               'skipped':skipped,'jobs':jobs,'worker_results':[],
               'scope':'Measured multi-shot SPEN; native original MATLAB phase correction and per-coil complex Tikhonov. No manual masks.'}
    write_json(work/'summary.json',summary)
    print(json.dumps({'event':'prepared','frames':len(jobs),'skipped':len(skipped),'workers':args.workers}),flush=True)
    if args.prepare_only:
        return
    summary['status'] = 'running'
    write_json(work/'summary.json',summary)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_worker,i,bucket,work,args.matlab) for i,bucket in enumerate(buckets) if bucket]
        for future in as_completed(futures):
            result = future.result()
            summary['worker_results'].append(result)
            write_json(work/'summary.json',summary)
            print(json.dumps({'event':'worker_finished',**result}),flush=True)
    updated, failures, checks = update_collection(run,work,jobs,settings['lambda_relative'])
    summary.update(status='finished' if not failures else 'finished_with_failures',
                   updated_frames=len(updated),failed_frames=len(failures),failures=failures,
                   finished_at=now(),optimization_budget_frames=sum(c['optimization_budget_reached'] for c in checks),
                   phase_fit_fallback_frames=sum(c['phase_fit_fallback'] for c in checks))
    write_json(work/'verification.json',{'updated_frames':updated,'checks':checks})
    write_json(work/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','updated_frames','failed_frames','optimization_budget_frames','phase_fit_fallback_frames')}),flush=True)


if __name__ == '__main__':
    main()
