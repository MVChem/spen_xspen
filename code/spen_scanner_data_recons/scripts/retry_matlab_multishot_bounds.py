#!/usr/bin/env python3
"""Retry an observed legacy MATLAB zero-index bug using an isolated source copy.

No masks, optimizer parameters or source acquisitions are changed. The guard
is the same boundary convention already used in the Python phase port: a
first compact sample has no preceding sample and therefore is not a gap.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import json

from run_matlab_multishot_collection import (PROJECT, MATLAB, sha, run_worker,
                                             update_collection, write_json, now)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=PROJECT/'runs/all_raw_260915')
    parser.add_argument('--finite-support',action='store_true',help='Also apply the Python-equivalent empty-correlation/linear-fit guard')
    args = parser.parse_args()
    run = args.run.resolve()
    base = run/'matlab_multishot_260915'
    summary = json.loads((base/'summary.json').read_text())
    if not summary['status'].startswith('finished'):
        parser.error('Wait for the main MATLAB batch to finish before retrying')
    failed = summary.get('failures', [])
    selected = [f for f in failed if 'Array indices must be positive integers' in f.get('matlab_result',{}).get('error','')]
    if not selected:
        print('No observed MATLAB zero-index failures require retry')
        return
    work = base/('retry_phase_fit_support' if args.finite_support else 'retry_compact_phase_bounds')
    if work.exists():
        parser.error(f'Retry directory already exists: {work}')
    shutil.copytree(base/'source',work/'source')
    runtime_records = []
    for runtime in (Path(__file__),PROJECT/'scripts/run_matlab_multishot_collection.py'):
        target = work/'source/runtime_python'/runtime.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(runtime,target)
        runtime_records.append({'path':str(target.relative_to(work)),'sha256':sha(target)})
    patches = []
    needle = 'JumpIdxCompact = find(PassThresholdChangeCompact==1) ;'
    for filename in ('EvenOddFix.m','EvenOddFixPEOddNum.m','EvenOddFixPEOddNum_forInvivo.m'):
        path = work/'source/legacy'/filename
        text = path.read_text()
        if text.count(needle) != 1:
            raise RuntimeError(f'Expected exactly one known compact-phase boundary in {filename}')
        original_sha = sha(path)
        replacement = needle + '\n  % A first retained sample has no previous retained sample.\n  JumpIdxCompact = JumpIdxCompact(JumpIdxCompact > 1);'
        text = text.replace(needle,replacement)
        changes = ['Ignore compact index 1 before accessing JumpIdx-1; MATLAB arrays are 1-based']
        if args.finite_support:
            guard_targets = [
                ('  AmpChangeCorr1DUse = AmpChangeCorr1DUse / AmpChangeCorr1DUse(1) ;',
                 "  if isempty(AmpChangeCorr1DUse) || ~isfinite(AmpChangeCorr1DUse(1)) || AmpChangeCorr1DUse(1)==0\n    warning('SPEN:NoLinearPhaseEstimate','Phase fitting failed: no finite correlation support; zero linear-phase correction');\n    MeanLinPhases = 0; return;\n  end\n  AmpChangeCorr1DUse = AmpChangeCorr1DUse / AmpChangeCorr1DUse(1) ;"),
                ('  NumPointsCorrUse = length(PhaseChangeCorr1DCompact) ;',
                 "  NumPointsCorrUse = length(PhaseChangeCorr1DCompact) ;\n  if NumPointsCorrUse < 2\n    warning('SPEN:NoLinearPhaseEstimate','Phase fitting failed: fewer than two retained correlation samples; zero linear-phase correction');\n    MeanLinPhases = 0; return;\n  end"),
                ('\n  MeanLinPhases = P(3) ;\n',
                 "\n  MeanLinPhases = P(3) ;\n  if ~isfinite(MeanLinPhases)\n    warning('SPEN:NoLinearPhaseEstimate','Phase fitting failed: nonfinite linear fit; zero linear-phase correction');\n    MeanLinPhases = 0;\n  end\n"),
            ]
            for old,new in guard_targets:
                if text.count(old) != 1:
                    raise RuntimeError(f'Expected one finite-support patch location in {filename}: {old}')
                text = text.replace(old,new)
            changes.append('Same empty/zero-correlation, fewer-than-two-samples and nonfinite-linear-fit handling as Python: no estimated linear phase means zero correction for that term, retaining the coil and signal')
        path.write_text(text)
        patches.append({'path':str(path.relative_to(work)),'original_sha256':original_sha,
                        'patched_sha256':sha(path),'changes':changes})
    write_json(work/'source_manifest.json',{
        'created_at':now(),'base_source_manifest':str(base/'source_manifest.json'),
        'base_source_manifest_sha256':sha(base/'source_manifest.json'),'patches':patches,
        'runtime_python':runtime_records,
        'scope':'Boundary guard only; original phase model, data, automatic masks and optimizer budgets retained'})
    jobs = []
    for failed_job in selected:
        job = {k:v for k,v in failed_job.items() if k not in ('matlab_result','verification_error')}
        folder = work/'frames'/job['frame_id']
        folder.mkdir(parents=True)
        job['output_mat'] = str(folder/'output.mat')
        job['result_json'] = str(folder/'result.json')
        job['log_path'] = str(folder/'matlab.log')
        jobs.append(job)
    worker = run_worker(0,jobs,work,MATLAB)
    settings = json.loads((run/'settings.json').read_text())
    updated,remaining,checks = update_collection(run,work,jobs,settings['lambda_relative'])
    if args.finite_support:
        for job in jobs:
            if job['frame_id'] not in updated:
                continue
            log = Path(job['log_path']).read_text(errors='replace')
            triggered = 'zero linear-phase correction' in ' '.join(log.split())
            case_path = Path(job['case_json'])
            case = json.loads(case_path.read_text())
            frame = next(f for f in case['frames'] if f['id']==job['frame_id'])
            frame['matlab_phase']['linear_phase_support_guard'] = {
                'triggered':triggered,'source_manifest':str((work/'source_manifest.json').relative_to(run)),
                'handling':'No available finite linear-phase estimate -> zero correction of that term; all original coils retained'}
            if triggered:
                frame.setdefault('quality_warnings',[]).append('该帧部分线性相位拟合没有可用估计，按已有数值保护采用零相位校正；全部接收通道保留')
            write_json(case_path,case)
            for check in checks:
                if check['frame_id']==job['frame_id']:
                    check['linear_phase_support_guard_triggered'] = triggered
        from run_collection import collect_summary
        collect_summary(run,json.loads((run/'inventory.json').read_text()),settings,'finished')
    write_json(work/'summary.json',{'status':'finished' if not remaining else 'finished_with_failures',
                                  'worker':worker,'updated_frames':updated,'failures':remaining,'checks':checks})
    recovered = set(updated)
    summary['failures'] = [f for f in failed if f['frame_id'] not in recovered]
    summary['updated_frames'] += len(updated)
    summary['failed_frames'] = len(summary['failures'])
    summary['status'] = 'finished' if not summary['failures'] else 'finished_with_failures'
    if summary.get('boundary_retry'):
        summary.setdefault('previous_retries',[]).append(summary['boundary_retry'])
    summary['boundary_retry'] = str(work.relative_to(run))
    summary['finished_at'] = now()
    verification = json.loads((base/'verification.json').read_text())
    verification['updated_frames'].extend(updated)
    verification['checks'].extend(checks)
    summary['optimization_budget_frames'] = sum(c['optimization_budget_reached'] for c in verification['checks'])
    summary['phase_fit_fallback_frames'] = sum(c['phase_fit_fallback'] for c in verification['checks'])
    write_json(base/'verification.json',verification)
    write_json(base/'summary.json',summary)
    print(json.dumps({'recovered_frames':len(updated),'remaining_failed_frames':len(summary['failures'])}))


if __name__ == '__main__':
    main()
