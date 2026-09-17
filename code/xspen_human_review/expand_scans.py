"""Expose every retained frame of the selected xSPEN scans in the offline gallery."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import h5py
import numpy as np

from build_unified import dump, digest, save_entry, stage
from review import HUMAN, VIEWS, reconstruct, ro_fft, rss

HERE = Path(__file__).resolve().parent


def expand_one(row, run):
    path = Path(row['scanner_export'])
    mid = row['scan_id']
    with h5py.File(path) as f:
        meta = json.loads(f.attrs['metadata'])
        nocc, nslice, coils, pe, ro = f['kspace'].shape
        assert [nocc, nslice, coils, pe, ro] == meta['shape_rep_slice_coil_pe_ro']
        assert np.iscomplexobj(f['kspace'][0, 0])
        names = ['raw_ro_rss', 'legacy_recipe_rss', 'tikhonov_rss', 'phasemap_rss']
        arrays = {name: np.zeros((nslice * nocc, pe, ro), np.float32) for name in names}
        available = {name: np.ones(nslice * nocc, bool) for name in names}
        failures = []
        for s in range(nslice):
            for occurrence in range(nocc):
                index = s * nocc + occurrence
                raw = f['kspace'][occurrence, s]
                assert np.isfinite(raw).all()
                try:
                    output, _ = reconstruct(raw, meta)
                    for name in names:
                        arrays[name][index] = output[name]
                except (ValueError, RuntimeError, FloatingPointError, AssertionError) as error:
                    # Keep the acquired image accessible even if a phase fit fails.
                    arrays['raw_ro_rss'][index] = rss(ro_fft(raw))
                    window = np.exp(-.001 * (np.arange(1, ro + 1) - ro / 2) ** 2)
                    arrays['legacy_recipe_rss'][index] = rss(ro_fft(raw * window[None, None, :]))
                    for name in ['tikhonov_rss', 'phasemap_rss']:
                        available[name][index] = False
                    failures.append(dict(slice=s, occurrence=occurrence, error=str(error)))
            if (s + 1) % 8 == 0:
                print(f'{mid}: {s+1}/{nslice} slices', flush=True)
    axes = [dict(label='切片', size=nslice), dict(label='occurrence', size=nocc, values=list(range(nocc)))]
    definitions = [
        ('raw_ro_rss', '输入', 'RO FFT + 线圈 RSS，尚未做 PE 逆编码。'),
        ('legacy_recipe_rss', '旧流程复算', 'RO Gaussian 滤波 + FFT + RSS；同流程复算，不是逐帧配对的旧 MATLAB Img。'),
        ('tikhonov_rss', 'Tikhonov', '邻行奇偶相位校正 + sinc Tikhonov，alpha=0.01。'),
        ('phasemap_rss', 'PhaseMap + InvA', 'sinc 模型下的 PhaseMap + 加窗 InvA，window=0.8。')]
    stages = [stage(name, label, arrays[name], axes, note, meta['fov_mm'],
                    frame_available=available[name].tolist()) for name, label, note in definitions]
    notes = ['occurrence 为逐层逐行的采集先后次数，尚未核验为具体 b0 / 扩散方向。',
             '完整扫描中的旧流程一列为复算；已核验的旧 MATLAB 图仍保留在 10 个选例中。',
             '重建使用当前简化 sinc 模型；未完成独立波形/B0 标定。']
    excluded = meta['excluded_original_slice_counters']
    if excluded:
        notes.append(f'旧 H5 导出排除了重复采集不完整的原始 slice counter {excluded}；本页覆盖导出中全部保留帧。')
    if failures:
        notes.append(f'{len(failures)} 帧的相位重建失败；输入仍可查看，重建位置明确标为不可用。')
    view = {'axial': '轴位', 'sagittal': '矢状位', 'coronal': '冠状位'}.get(VIEWS.get(mid), '方向见元数据')
    entry = dict(id=mid, scan=mid, family='crossed', group='crossed', title=mid,
        subtitle=f'{view} · {nslice} 层 × {nocc} 次采集', status='full_scan', status_label='全部可用帧',
        scope='all_retained_scanner_frames', is_selected=False, warnings=notes,
        source=meta['source'], scanner_h5=str(path),
        geometry=dict(sequence=meta['sequence'], fov_mm=meta['fov_mm'], header_pe_ro=[pe, ro],
            r_value=meta['r_value'], thickness_mm=meta['thickness_mm'], echo_time_ms=meta['echo_time_s']*1000,
            nominal_b_field=meta['nominal_protocol_b_value']),
        source_details=dict(shape_rep_slice_coil_pe_ro=meta['shape_rep_slice_coil_pe_ro'],
            slice_order=meta['slice_order'], excluded_original_slice_counters=excluded,
            source_raw_sha256=meta['source_sha256'], scanner_h5_sha256=digest(path),
            model='Existing review.reconstruct, identical to the 10 selected cases'),
        reconstruction_failures=failures)
    result = save_entry(run, entry, stages)
    print(f'{mid}: saved {result["frame_count"]} frames, {len(failures)} failures', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--mids', nargs='+', default=HUMAN)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    assert run.is_relative_to(HERE / 'runs')
    inventory = json.loads((run/'inventory/crossed_previous_verification/inventory.json').read_text())
    rows = [next(r for r in inventory if r['scan_id'] == mid and r.get('scanner_export')) for mid in args.mids]
    old = json.loads((run/'manifest.json').read_text())
    results, pending = {}, []
    for row in rows:
        cached = run/'entries'/row['scan_id']/'metadata.json'
        if args.resume and cached.is_file():
            results[row['scan_id']] = json.loads(cached.read_text())
        else:
            pending.append(row)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(expand_one, row, run): row['scan_id'] for row in pending}
        for job in as_completed(jobs):
            result = job.result()
            results[result['id']] = result
    existing = [e for e in old['entries'] if e['id'] not in results]
    for entry in existing:
        entry['is_selected'] = entry['family'] == 'crossed' and entry['frame_count'] == 1
        dump(run/entry['metadata'], entry)
    entries = [results[mid] for mid in args.mids] + existing
    full = [e for e in entries if not e.get('is_selected')]
    summary = dict(entry_count=len(entries), selected_entries=sum(e.get('is_selected', False) for e in entries),
        distinct_scans=len({e['scan'] for e in full}), crossed_scans=sum(e['family']=='crossed' for e in full), hybrid_scans=sum(e['family']=='hybrid' for e in full),
        crossed_primary_frames=sum(e['frame_count'] for e in full if e['family']=='crossed'),
        hybrid_primary_frames=sum(e['frame_count'] for e in full if e['family']=='hybrid'),
        primary_frames=sum(e['frame_count'] for e in full))
    manifest = dict(summary=summary, entries=entries)
    dump(run/'manifest.json', manifest)
    (run/'manifest.js').write_text('window.REVIEW_MANIFEST='+json.dumps(manifest, ensure_ascii=False)+';\n')
    dump(run/'expanded_scans_verification.json', dict(passed=True, summary=summary,
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=digest(Path(__file__)),
        scans=[dict(id=e['id'], frames=e['frame_count'], failures=len(e['reconstruction_failures']),
            excluded_original_slice_counters=e['source_details']['excluded_original_slice_counters']) for e in results.values()]))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
