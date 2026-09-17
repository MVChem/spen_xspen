"""Import the 543 existing Bruker reconstructions without rerunning a method.

The two original MATLAB images retain their complex values in complex_arrays.npz.
The two later numerical solutions retain signed values in images.npz. Display
chunks use the same finite-value uint16/min/max/mask convention as catalog_all.
No shared manifest, source file, or web asset is modified by this importer.
"""
import argparse
import base64
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from build_unified import digest, dump

HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/data2/chk/workspace/2026/08/14/spen_diffusion_recons/results/real_batch')
CANONICAL = (SOURCE/'manifest.json').resolve().parent
METHODS = SOURCE / 'additional_methods_20260911'
LABELS = {
    'bruker_before': '既有输入 · RO FFT + 线圈合并',
    'bruker_after': '既有传统重建 · PhaseMap + InvA / SR',
    'bruker_tikhonov': '既有 Tikhonov · rho=0.01',
    'bruker_diffusion': '既有 Diffusion · EDM + DiffPIR',
}
NOTES = {
    'bruker_before': '读取原 MATLAB before（Imag_origin）；RO FFT 后线圈合并的复数幅度，尚未做 SPEN 逆编码。完整复数保存在 complex_arrays.npz。',
    'bruker_after': '读取原 MATLAB after（images）；奇偶/shot PhaseMap 校正、InvA / SR 和线圈合并的复数幅度。本轮未重新运行重建。',
    'bruker_tikhonov': '读取旧 NPZ 的 tikhonov：同一校正后复数观测、固定线圈/物体相位模型，rho=0.01。保留原始实值解，不截断负值或大于 1 的值。',
    'bruker_diffusion': '读取旧 NPZ 的 diffusion：小鼠为主的 EDM prior（EMA step 30000）和 60 步 DiffPIR；本轮未重新推理。保留未截断的实值解。',
}


def image_frames(value):
    """Old MATLAB batch saved [PE, RO, frame] in MATLAB frame order."""
    array = np.asarray(value)
    arranged = array.reshape(array.shape[0], array.shape[1], -1, order='F')
    return arranged.transpose(2, 0, 1)


def export_stage(folder, eid, key, values, axes, fov):
    values = np.asarray(values, dtype=np.float32)
    n, h, w = values.shape
    assert n == np.prod([a['size'] for a in axes])
    finite = np.isfinite(values)
    signed = bool(np.any(values < 0))
    chunk_size = max(1, min(64, 2_000_000 // (h * w)))
    chunk_key = eid + ':' + key
    chunks = []
    for start in range(0, n, chunk_size):
        block = values[start:start + chunk_size]
        good = np.isfinite(block)
        clean = np.where(good, block, np.nan).reshape(len(block), -1)
        low = np.nanmin(clean, axis=1) if signed else np.zeros(len(block))
        high = np.nanmax(clean, axis=1)
        p995 = np.nanquantile(clean, .995, axis=1)
        p005 = np.nanquantile(clean, .005, axis=1) if signed else np.zeros(len(block))
        low, high, p995, p005 = [np.nan_to_num(a, nan=0, posinf=0, neginf=0).astype(np.float64)
                                for a in (low, high, p995, p005)]
        span = np.maximum(high - low, 1e-30)
        safe = np.where(good, block, low[:, None, None])
        encoded = np.rint(np.clip((safe-low[:, None, None])/span[:, None, None], 0, 1)*65535).astype('<u2')
        decoded = encoded.astype(float)/65535*span[:, None, None]+low[:, None, None]
        bound = span[:, None, None]/65535/2 + np.maximum(np.abs(safe), 1)*3e-7
        assert np.all(np.abs(decoded-safe) <= bound), key
        chunk = dict(start=start, count=len(block), minima=low.tolist(), maxima=high.tolist(),
                     p995=p995.tolist(), p005=p005.tolist(), data=base64.b64encode(encoded.tobytes()).decode())
        if not good.all():
            chunk['invalid'] = base64.b64encode(np.packbits(~good.ravel(), bitorder='little').tobytes()).decode()
        filename = f'{key}_chunk_{start//chunk_size:05d}.js'
        (folder/filename).write_text('window.REVIEW_CHUNKS=window.REVIEW_CHUNKS||{};window.REVIEW_CHUNKS['
            + json.dumps(f'{chunk_key}:{start//chunk_size}') + ']=' + json.dumps(chunk, separators=(',', ':')) + ';\n')
        chunks.append(f'entries/{eid}/{filename}')
    return dict(id=key, label=LABELS[key], note=NOTES[key], axes=axes, shape=[n, h, w], fov=fov,
                chunked=True, chunk_size=chunk_size, chunk_key=chunk_key, chunks=chunks,
                signed=signed, invalid_count=int((~finite).sum()),
                global_lower=float(np.min(values, where=finite, initial=0)),
                global_window=float(np.max(values, where=finite, initial=0)))


def method_evidence():
    paths = [CANONICAL/'code/reconstruct_shard.m', CANONICAL/'code/reconstruct_repetitions.m',
             METHODS/'README.md', METHODS/'code/reconstruct_methods.py', METHODS/'code/build_gallery.py']
    return [dict(source=str(p), sha256=digest(p)) for p in paths]


def import_one(task):
    row, run, evidence, qc = task
    run = Path(run)
    eid = 'bruker_' + hashlib.sha256(row['case_id'].encode()).hexdigest()[:16]
    folder = run/'entries'/eid
    folder.mkdir(parents=True, exist_ok=True)
    path = Path(row['source'])
    assert digest(path) == row['mat_sha256']
    saved = loadmat(path, variable_names=['before', 'after', 'Imag_low', 'fov_mm', 'reconstruction_source'],
                    simplify_cells=True)
    complex_arrays = {key: image_frames(saved[key]) for key in ['before', 'after']}
    if 'Imag_low' in saved:
        complex_arrays['Imag_low'] = image_frames(saved['Imag_low'])
    display = {'bruker_'+key: np.abs(value).astype(np.float32) for key, value in complex_arrays.items()
               if key in ['before', 'after']}
    additional = Path(row['additional_methods_npz'])
    with np.load(additional, allow_pickle=False) as arrays:
        for key in ['tikhonov', 'diffusion']:
            assert arrays[key].dtype == np.float32 and not np.iscomplexobj(arrays[key])
            display['bruker_'+key] = image_frames(arrays[key])
    shape = display['bruker_before'].shape
    assert all(value.shape == shape for value in display.values())
    assert shape[0] == row['expected_frames'] == row['slices']*row['repetitions']
    assert shape[1:] == tuple(reversed(row['matrix']))
    # Existing files have either multiple slices OR multiple repetitions, never both.
    assert row['slices'] == 1 or row['repetitions'] == 1
    if row['repetitions'] > 1:
        axes = [dict(label='重复 / volume（原顺序）', size=row['repetitions'])]
    else:
        axes = [dict(label='切片（原顺序）', size=row['slices'])]
    fov_source = [float(v) for v in np.ravel(saved['fov_mm'])[:2]]
    fov = fov_source[::-1]  # PVM_Fov is [RO, PE]; displayed rows are PE.
    assert min(fov) > 0
    stages = [export_stage(folder, eid, key, value, axes, fov) for key, value in display.items()]
    np.savez_compressed(folder/'images.npz', **display)
    np.savez_compressed(folder/'complex_arrays.npz', **complex_arrays)
    # Reload every exported stage and compare to its source, including phase and signed values.
    with np.load(folder/'images.npz') as check:
        for key, value in display.items():
            np.testing.assert_array_equal(check[key], value)
    with np.load(folder/'complex_arrays.npz') as check:
        for key, value in complex_arrays.items():
            np.testing.assert_array_equal(check[key], value)
    coord = shape[0]//2
    im = display['bruker_after'][coord]
    valid = im[np.isfinite(im)]
    limit = float(np.quantile(valid, .995)) if valid.size else 1
    gray = np.nan_to_num(np.clip(im/max(limit, 1e-30), 0, 1), nan=0)
    preview = Image.fromarray(np.rint(gray*255).astype('uint8')).convert('RGB')
    width = max(1, round(min(288, 236*fov[1]/fov[0])))
    height = max(1, round(width*fov[0]/fov[1]))
    preview = preview.resize((width, height), Image.Resampling.NEAREST)
    thumb = Image.new('RGB', (304, 260), '#080e17')
    thumb.paste(preview, ((304-width)//2, (260-height)//2))
    thumb.save(folder/'thumbnail.jpg', quality=90)
    status_path = METHODS/'status'/(row['case_id']+'_methods.json')
    status = json.loads(status_path.read_text())
    assert status['status'] == 'success' and status['shape'] == [shape[1], shape[2], shape[0]]
    reconstruction = Path(str(saved['reconstruction_source']))
    source_details = dict(
        study=row['study'], scan_id=row['scan_id'], scanner_format=row['format'],
        scanner_subject_type=row['scanner_subject_type'], species_label=row['anatomy_label'],
        species_verified=False, subject_type_warning=row['subject_type_warning'],
        original_shape=row['shape'], original_fov_mm_ro_pe=fov_source,
        display_note='保留原数组方向：PE 纵轴、RO 横轴；未做旧图册的上下/左右同时翻转。MAT 第三维按原顺序逐帧展示。',
        frame_order='Slice index for multi-slice acquisitions; volume index for the single PV5 repetition case. No acquisition has both axes > 1.',
        display_transform='before/after: complex absolute value; tikhonov/diffusion: unchanged signed real values. No spatial flip, transpose, resampling, or numerical clipping.',
        raw_source=row['raw_file'], raw_sha256=row['raw_sha256'], source_mat=str(path),
        source_sha256=row['mat_sha256'], source_npz=str(additional), npz_sha256=digest(additional),
        reconstruction_source=str(reconstruction), reconstruction_source_sha256=digest(reconstruction),
        method_status_source=str(status_path), method_status_sha256=digest(status_path),
        method_settings=status['settings'], method_evidence=evidence,
        checkpoint_sha256=status['checkpoint_sha256'], physics_source=status['physics_source'],
        physics_sha256=status['physics_sha256'], nuisance=status['nuisance'],
        prior_grid_note=status['prior_grid_note'], quality_flags=qc,
        method_frame_records=status['records'], raw_manifest_record=row,
        coordinate_pairing='All four arrays use the same [PE,RO,frame] native grid and frame indices, confirmed from the source export and inference scripts.',
        amplitude_pairing='MATLAB complex magnitude uses native scanner units. Tikhonov/Diffusion use normalized magnitude (pred+1)/2; amplitude scales are not directly comparable.',
    )
    warnings = [
        'Bruker SPEN 旧算法结果；本轮只导入和显示，未运行重建或 Diffusion 推理。',
        '物种标签来自扫描名称和原始档案，未核实为人脑；设备主体 UID 不等于生物学个体数。',
        '四阶段空间网格和帧序一致，但 MATLAB 与 Tikhonov / Diffusion 的幅度单位不同；默认各帧独立设窗。',
        'Tikhonov / Diffusion 保留未截断实值；旧图册曾先裁剪到 [0,1] 并翻转两个空间轴，因此显示可能与旧 PNG 不同。',
        'Diffusion 使用以小鼠为主的旧 EDM prior；真实扫描没有配对干净真值。',
    ]
    if row['subject_type_warning']:
        warnings.append(row['subject_type_warning'])
    if qc.get('flags'):
        warnings.append('旧质量标记：'+'；'.join(str(flag) for flag in qc['flags']))
    study_label = row['study'][:15] if row['study'][0].isdigit() else row['study'].rsplit('_', 1)[-1]
    entry = dict(id=eid, scan=row['case_id'], family='bruker', group='bruker', source_kind='bruker',
                 title=f"{study_label} · scan {row['scan_id']:03d}",
                 subtitle='Bruker SPEN · '+row['anatomy_label']+' · 旧算法结果',
                 source=row['raw_file'], source_file=str(path), source_mat=str(path),
                 status='bruker_existing_comparison', status_label='旧四阶段对照', is_selected=False,
                 frame_count=shape[0], default_coords=[coord], stages=stages,
                 geometry=dict(sequence='Bruker SPEN · '+row['format'], fov_mm=fov, fov_unit='mm',
                               header_pe_ro=list(shape[1:]), thickness_mm=None, r_value=None),
                 source_details=source_details, warnings=warnings,
                 payload=f'entries/{eid}/data.js', arrays=f'entries/{eid}/images.npz',
                 complex_arrays=f'entries/{eid}/complex_arrays.npz',
                 metadata=f'entries/{eid}/metadata.json', thumbnail=f'entries/{eid}/thumbnail.jpg')
    dump(folder/'metadata.json', entry)
    (folder/'data.js').write_text('window.REVIEW_PAYLOADS=window.REVIEW_PAYLOADS||{};window.REVIEW_PAYLOADS['
        + json.dumps(eid) + ']=' + json.dumps(stages, ensure_ascii=False, separators=(',', ':')) + ';\n')
    public = dict(entry)
    public['source_details'] = {key: value for key, value in source_details.items()
                                if key not in ['method_frame_records', 'raw_manifest_record', 'method_evidence']}
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=HERE/'runs/unified_review_260916')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    audit = json.loads((args.run/'inventory/bruker_import_sources.json').read_text())
    rows = audit['records']
    qc = {row['case_id']: row for row in json.loads((SOURCE/'qc_flags.json').read_text())}
    evidence = method_evidence()
    tasks = [(row, str(args.run), evidence, qc.get(row['case_id'], {})) for row in rows]
    entries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, entry in enumerate(pool.map(import_one, tasks), 1):
            entries.append(entry)
            if index % 50 == 0 or index == len(rows):
                print(f'Bruker: {index}/{len(rows)} scans, {sum(e["frame_count"] for e in entries)} frames', flush=True)
    summary = dict(entry_count=len(entries), frame_count=sum(e['frame_count'] for e in entries),
                   stage_count=sum(len(e['stages']) for e in entries),
                   stage_frames=sum(e['frame_count']*len(e['stages']) for e in entries),
                   study_count=len({r['study'] for r in rows}),
                   species_label_counts=dict(Counter(r['anatomy_label'] for r in rows)),
                   method='Existing results imported only; no reconstruction or inference rerun',
                   array_verification='All 543 display NPZ and complex NPZ reloads exactly match source-derived arrays; every uint16 chunk passes its quantization bound.',
                   signed_stages=sum(s['signed'] for e in entries for s in e['stages']),
                   invalid_values=sum(s['invalid_count'] for e in entries for s in e['stages']))
    assert summary['entry_count'] == 543 and summary['frame_count'] == 921 and summary['stage_count'] == 2172
    dump(args.run/'inventory/bruker_entries.json', dict(summary=summary, entries=entries))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
