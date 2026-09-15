#!/usr/bin/env python3
"""Make MATLAB native InvA width consistent in both corrected and raw views.

The initial batch was already running when this issue was found: even-shot
MATLAB reconstruction sets its native Gaussian width to .5, while Python's
initial uncorrected image used .8. Recompute the uncorrected image with the
same saved native operator and expose original optimizer/fit warnings.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np
from run_matlab_multishot_collection import PROJECT, sha, write_json, now
from run_collection import collect_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=PROJECT/'runs/all_raw_260915')
    args = parser.parse_args()
    run = args.run.resolve()
    work = run/'matlab_multishot_260915'
    main_summary = json.loads((work/'summary.json').read_text())
    if not main_summary['status'].startswith('finished'):
        parser.error('Wait for MATLAB and its retries to finish')
    source = work/'postprocessing_source'/Path(__file__).name
    source.parent.mkdir(exist_ok=True)
    shutil.copy2(__file__,source)
    source_manifest = work/'postprocessing_source_manifest.json'
    write_json(source_manifest,{'created_at':now(),'script':str(source.relative_to(run)),
                               'sha256':sha(source),'change':'Use the returned MATLAB weighted adjoint for both original and phase-corrected observations; expose phase diagnostics in gallery warnings'})
    widths, before_errors, changed, warning_counts = Counter(), [], [], Counter()
    for path in sorted((run/'cases').glob('*/case.json')):
        case = json.loads(path.read_text())
        touched = False
        for frame in case.get('frames',[]):
            meta = frame.get('matlab_phase')
            if meta is None:
                continue
            array_path = run/frame['arrays_path']
            with np.load(array_path) as stored:
                arrays = {k:stored[k] for k in stored.files}
            expected = np.einsum('ij,jrc->irc',arrays['inva_weighted_adjoint'],arrays['rofft_original'])
            error = float(np.linalg.norm(arrays['inva_uncorrected']-expected)/max(np.linalg.norm(expected),1e-30))
            before_errors.append(error)
            # Odd-shot width .8 already agrees to roundoff; preserve those
            # expensive compressed files when the saved operator is equal.
            if error > 1e-10:
                arrays['inva_uncorrected'] = expected
                temporary = array_path.with_suffix('.finalize.tmp.npz')
                np.savez_compressed(temporary,**arrays)
                temporary.replace(array_path)
            after_error = float(np.linalg.norm(arrays['inva_uncorrected']-expected)/max(np.linalg.norm(expected),1e-30))
            warnings = list(frame.get('quality_warnings',[]))
            if 'linear_phase_support_guard' in meta:
                # MATLAB wraps warning text at the terminal width. Read our
                # explicit new guard marker after joining wrapped whitespace,
                # independently of the old generic P0-fallback classifier.
                log_path = run/meta['log_path']
                guard_log = ' '.join(log_path.read_text(errors='replace').split())
                triggered = 'zero linear-phase correction' in guard_log
                meta['linear_phase_support_guard'].update(triggered=triggered,
                    evidence_log_sha256=sha(log_path),
                    evidence_marker='zero linear-phase correction',
                    marker_is_new_guard_instrumentation=True)
                if triggered:
                    warning = '该帧部分线性相位拟合没有可用估计，按已有数值保护采用零相位校正；全部接收通道保留'
                    if warning not in warnings:
                        warnings.append(warning)
            for flag, message in (
                ('optimization_budget_reached','MATLAB 相位优化达到原始计算预算，使用当前最优结果，未确认收敛'),
                ('phase_fit_fallback','MATLAB 部分相位拟合失败，原始脚本将对应多项式系数置零'),
                ('badly_conditioned_polynomial','MATLAB 相位多项式拟合存在病态条件警告')):
                if meta.get(flag):
                    warning_counts[flag] += 1
                    if message not in warnings:
                        warnings.append(message)
            frame['quality_warnings'] = warnings
            meta['uncorrected_operator_postprocessing'] = {
                'source_manifest':str(source_manifest.relative_to(run)),
                'prior_uncorrected_relative_error':error,
                'equation_relative_error_after':after_error,
                'arrays_sha256_after':sha(array_path),
                'note':'Both InvA views now use the same saved MATLAB native weighted adjoint; Tikhonov still uses the saved forward A'}
            widths[str(meta['effective_gauss_relative_width'])] += 1
            changed.append(frame['id'])
            touched = True
        if touched:
            write_json(path,case)
    report = {'finished_at':now(),'frames':len(changed),'gaussian_width_counts':dict(widths),
              'diagnostic_counts':dict(warning_counts),'maximum_prior_uncorrected_relative_error':max(before_errors,default=0.),
              'source_manifest':str(source_manifest.relative_to(run)),'updated_frame_ids':changed}
    write_json(work/'postprocessing_verification.json',report)
    main_summary['postprocessing_verification'] = str((work/'postprocessing_verification.json').relative_to(run))
    write_json(work/'summary.json',main_summary)
    final_summary = collect_summary(run,json.loads((run/'inventory.json').read_text()),json.loads((run/'settings.json').read_text()),'finished')
    status_path = run/'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if 'elapsed_seconds' in status:
        status['python_batch_elapsed_seconds'] = status.pop('elapsed_seconds')
    status.update(status='finished',stage='matlab_phase_and_postprocessing_finished',
                  processed_records=len(final_summary['cases']),planned_records=final_summary['planned_records'],
                  frame_count=final_summary['frame_count'],case_status_counts=final_summary['case_status_counts'],
                  frame_status_counts=final_summary['frame_status_counts'],matlab_updated_frames=len(changed),
                  matlab_stage_elapsed_seconds=(datetime.now().astimezone()-datetime.fromisoformat(main_summary['created_at'])).total_seconds(),
                  updated_at=now())
    write_json(status_path,status)
    print(json.dumps({k:v for k,v in report.items() if k!='updated_frame_ids'},ensure_ascii=False))


if __name__=='__main__':
    main()
