"""Build an offline gallery of 10 selected xSPEN observations and 16 Hybrid scans.

No reconstruction is rerun: use existing reconstructed MAT images, the verified
xSPEN arrays, and RO-only previews for the three Hybrid files without a MAT.
"""
import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def geometry(path):
    with Path(path).open('rb') as f:
        size = struct.unpack('<I', f.read(4))[0]
        f.seek(0)
        header = f.read(size).decode('latin1')
    def value(key, default=None):
        m = re.search('^' + re.escape(key) + r'\s*=\s*(.*)$', header, re.M)
        if not m:
            return default
        return m.group(1).strip().strip('"')
    def number(key, default=None):
        v = value(key)
        return float(v) if v is not None else default
    return dict(sequence=value('tSequenceFileName'),
        header_pe_ro=[int(number('sKSpace.lPhaseEncodingLines')), int(number('sKSpace.lBaseResolution'))],
        fov_mm=[number('sSliceArray.asSlice[0].dPhaseFOV'), number('sSliceArray.asSlice[0].dReadoutFOV')],
        header_slices=int(number('sSliceArray.lSize')), thickness_mm=number('sSliceArray.asSlice[0].dThickness'),
        r_value=number('sWiPMemBlock.alFree[12]'), echo_time_ms=number('alTE[0]') / 1000,
        header_repetitions=int(number('lRepetitions', 0)) + 1,
        nominal_b_field=number('sWiPMemBlock.adFree[0]'))


def stage(name, label, frames, axes, note, fov, **kwargs):
    frames = np.asarray(frames, dtype=np.float32)
    assert frames.ndim == 3 and frames.shape[0] == np.prod([a['size'] for a in axes])
    assert np.isfinite(frames).all() and np.all(frames >= 0)
    return dict(id=name, label=label, frames=frames, axes=axes, note=note, fov=fov, **kwargs)


def from_mat_array(name, label, array, axis_labels, note, fov):
    """Input has [PE,RO,slice,extra...]; preserve every extra dimension."""
    a = np.asarray(array)
    assert a.ndim >= 3 and len(axis_labels) == a.ndim - 2
    axes = [dict(label=label, size=int(n)) for label, n in zip(axis_labels, a.shape[2:])]
    frames = a.transpose(*range(2, a.ndim), 0, 1).reshape(-1, *a.shape[:2])
    return stage(name, label, frames, axes, note, fov)


def save_entry(out, entry, stages):
    """Store float arrays plus uint16 display payloads that work through file://."""
    folder = out / 'entries' / entry['id']
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'images.npz', **{s['id']: s['frames'] for s in stages})
    payload = []
    for s in stages:
        frames = s['frames']
        # The browser supports independent-frame and scan-wide windows. Full
        # per-frame maxima encode all values; display quantiles do not clip NPZ.
        maxima = frames.max(axis=(1, 2)).astype(np.float64)
        encoded = np.rint(frames / np.maximum(maxima[:, None, None], 1e-30) * 65535).astype('<u2')
        restored = encoded.astype(float) / 65535 * maxima[:, None, None]
        tolerance = maxima.max() / 65535 / 2 + max(maxima.max(), 1) * 2e-7
        assert np.abs(restored - frames).max() <= tolerance
        p995 = np.quantile(frames, .995, axis=(1, 2))
        nonzero = frames[frames > 0]
        global_window = float(np.quantile(nonzero, .995)) if nonzero.size else 1
        public = {k: v for k, v in s.items() if k != 'frames'}
        public.update(shape=list(frames.shape), maxima=maxima.tolist(), p995=p995.tolist(),
            global_window=global_window, data=base64.b64encode(encoded.tobytes()).decode('ascii'))
        payload.append(public)
    representative = 0
    dims = [a['size'] for a in stages[0]['axes']]
    coords = entry.get('default_coords', [n // 2 if i == 0 else 0 for i, n in enumerate(dims)])
    representative = int(np.ravel_multi_index(coords, dims))
    entry['default_coords'] = coords
    entry['stages'] = [{k: v for k, v in s.items() if k not in ['data', 'maxima', 'p995']} for s in payload]
    entry['frame_count'] = int(stages[0]['frames'].shape[0])
    entry['payload'] = f"entries/{entry['id']}/data.js"
    entry['arrays'] = f"entries/{entry['id']}/images.npz"
    entry['metadata'] = f"entries/{entry['id']}/metadata.json"
    # Thumbnails preserve PE/RO physical proportions; no anatomical rotation.
    im = stages[0]['frames'][representative]
    limit = max(float(np.quantile(im, .995)), 1e-20)
    rgb = Image.fromarray((np.clip(im / limit, 0, 1) * 255).astype('uint8')).convert('RGB')
    fy, fx = stages[0]['fov']
    width = max(1, round(min(288, 236 * fx / fy)))
    height = max(1, round(width * fy / fx))
    rgb = rgb.resize((width, height), Image.Resampling.NEAREST)
    thumb = Image.new('RGB', (304, 260), '#080e17')
    thumb.paste(rgb, ((304 - rgb.width) // 2, (260 - rgb.height) // 2))
    thumb.save(folder / 'thumbnail.jpg', quality=90)
    entry['thumbnail'] = f"entries/{entry['id']}/thumbnail.jpg"
    dump(folder / 'metadata.json', entry)
    js = 'window.REVIEW_PAYLOADS=window.REVIEW_PAYLOADS||{};window.REVIEW_PAYLOADS[' + json.dumps(entry['id']) + ']='
    (folder / 'data.js').write_text(js + json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + ';\n')
    return entry


def cross_entries(old_run, out):
    manifest = json.loads((old_run / 'manifest.json').read_text())
    unified = isinstance(manifest, dict)
    if unified:
        records = [json.loads((old_run / e['source_metadata']).read_text())
                   for e in manifest['entries'] if e['family'] == 'crossed' and e.get('source_metadata')]
    else:
        records = manifest
    result = []
    for number, record in enumerate(records, 1):
        ident = record['id']
        folder = out / 'entries' / ident
        folder.mkdir(parents=True, exist_ok=True)
        old_folder = old_run / ('entries' if unified else 'cases') / ident
        shutil.copy2(old_folder / ('complex_arrays.npz' if unified else 'arrays.npz'), folder / 'complex_arrays.npz')
        shutil.copy2(old_folder / ('source_metadata.json' if unified else 'metadata.json'), folder / 'source_metadata.json')
        with np.load(folder / 'complex_arrays.npz') as data:
            old_key = 'legacy_saved_rss' if record['original_mat'] else 'legacy_recipe_rss'
            note = '已配对的旧 MATLAB Img（RO 滤波/FFT/RSS）' if record['original_mat'] else '按旧 RO 滤波/FFT/RSS 流程复算（旧 MAT 未配对或缺失）'
            fov = record['metadata']['fov_mm']
            raw_adc = np.sqrt((np.abs(data['raw_adc']) ** 2).sum(0))
            adc = np.log1p(100 * raw_adc / max(float(np.quantile(raw_adc, .995)), 1e-20))
            definitions = [('raw_ro_rss', '输入 · RO FFT + RSS', '尚未做 PE 逆编码'),
                (old_key, '旧 MATLAB / 同流程复算', note),
                ('tikhonov_rss', '奇偶校正 + Tikhonov', '当前 sinc 模型；alpha=0.01'),
                ('phasemap_rss', 'PhaseMap + 加窗 InvA', '当前 sinc 模型；window=0.8')]
            stages = [stage(key, label, data[key][None], [dict(label='固定选例', size=1)], desc, fov) for key, label, desc in definitions]
            stages.append(stage('raw_adc_log', '原始 ADC · log RSS', adc[None], [dict(label='固定选例', size=1)],
                '完整复数 ADC 保存在 complex_arrays.npz；此列为对数幅度显示，独立灰度窗。', [1, 1], log_display=True))
        meta = record['metadata']
        entry = dict(id=ident, number=number, scan=record['scan'], family='crossed', group='crossed',
            title=f"{record['scan']} · s{record['slice']} / o{record['occurrence']}",
            subtitle=record['selection_reason'], status='verified_comparison', status_label='五阶段对照',
            warnings=['切片 s 与时序 o 从 0 编号；occurrence 未对应到已核验的 b0 / 扩散方向。',
                '旧 MATLAB reader 的幅度尺度与当前流程不同；默认独立窗仅比较结构。'],
            source=record['raw_source'], geometry=dict(sequence=meta['sequence'], fov_mm=fov,
                header_pe_ro=meta['shape_rep_slice_coil_pe_ro'][-2:], r_value=meta['r_value'],
                thickness_mm=meta['thickness_mm'], nominal_b_field=meta['nominal_protocol_b_value']),
            complex_arrays=f'entries/{ident}/complex_arrays.npz',
            source_metadata=f'entries/{ident}/source_metadata.json', slice=record['slice'], occurrence=record['occurrence'],
            validation=record['adc_verification'])
        result.append(save_entry(out, entry, stages))
    return result


def raw_only(path, meta):
    """Read every acquired frame; preserve counters and explicitly flag missing lines."""
    import twixtools
    scan = twixtools.read_twix(str(path), parse_geometry=False, parse_pmu=False, verbose=False)[0]
    mdbs = [m for m in scan['mdb'] if m.is_image_scan()]
    dims = ['Sli', 'Rep', 'Set', 'Ave', 'Eco', 'Par', 'Phs', 'Ida', 'Idb', 'Idc', 'Idd', 'Ide']
    varying = [d for d in dims if d == 'Sli' or len({int(getattr(m.mdh.Counter, d)) for m in mdbs}) > 1]
    groups = defaultdict(list)
    seen = set()
    for m in mdbs:
        key = tuple(int(getattr(m.mdh.Counter, d)) for d in varying)
        line = int(m.mdh.Counter.Lin)
        if (key, line) in seen:
            raise ValueError('Repeated same frame/line; must not average unknowingly')
        seen.add((key, line))
        groups[key].append(line)
    mapped = twixtools.map_twix(scan, verbose=False)['image']
    for d in mapped.flags['average']:
        mapped.flags['average'][d] = False
    mapped.flags['average']['Seg'] = True
    mapped.flags['regrid'] = True
    mapped.flags['remove_os'] = True
    values = [sorted({key[i] for key in groups}) for i in range(len(varying))]
    names = {'Sli': '原始 slice counter', 'Rep': 'Rep', 'Set': 'Set', 'Ave': 'Ave'}
    axes = [dict(label=names.get(d, d), size=len(v), values=v) for d, v in zip(varying, values)]
    shape = tuple(len(v) for v in values)
    pe, ro = meta['header_pe_ro']
    frames, coverage = [], []
    for index in np.ndindex(shape):
        key = tuple(values[i][j] for i, j in enumerate(index))
        indices = [0] * len(mapped.dim_order)
        for d in ['Lin', 'Cha', 'Col']:
            indices[mapped.dim_order.index(d)] = slice(None)
        for d, v in zip(varying, key):
            indices[mapped.dim_order.index(d)] = v
        array = np.asarray(mapped[tuple(indices)])
        line_size, coils, col_size = [mapped.shape[mapped.dim_order.index(d)] for d in ['Lin', 'Cha', 'Col']]
        array = array.reshape(line_size, coils, col_size)
        if line_size != pe:
            padded = np.zeros((pe, coils, col_size), array.dtype)
            padded[:min(pe, line_size)] = array[:pe]
            array = padded
        fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(array, axes=2), axis=2, norm='ortho'), axes=2)
        frames.append(np.sqrt((np.abs(fft) ** 2).sum(1)))
        coverage.append(dict(counters=dict(zip(varying, key)), lines=len(groups.get(key, [])), expected=pe,
            complete=sorted(groups.get(key, [])) == list(range(pe))))
    return stage('raw_ro', '原始输入 · RO FFT + RSS', np.stack(frames), axes,
        '从 .dat 新读取：反向线校正、ramp regrid、去 RO 过采样、RO FFT、线圈 RSS；没有做 PE 重建。',
        meta['fov_mm'], coverage=coverage), dict(image_mdh_count=len(mdbs), frame_coverage=coverage,
        acquired_slice_counters=values[0], header_slices=meta['header_slices'], header_repetitions=meta['header_repetitions'])


def hybrid_entry(row, out, mid253_mat):
    ident = row['scan']
    path = Path(row['source'])
    meta = geometry(path)
    assert 'esrs_hyb_spen_Diff2' in meta['sequence']
    warnings = ['序列为 Hybrid SPEN 二次相位编码；旧文件名中的 xSPEN 不是序列分类依据。',
        '保留旧数组的层序和 PE/RO 方向；未统一解剖左右，不把重复采集计作独立受试者。']
    source_mat = path.with_suffix('.mat')
    details = {}
    if ident == 'MID253':
        source_mat = mid253_mat
        with h5py.File(source_mat) as f:
            # MATLAB v7.3 reverses array axes as exposed to h5py.
            before = f['SmatBeforePhaseMapInvA'][()].transpose(3, 2, 1, 0)
            after = f['Smat'][()].transpose(3, 2, 1, 0)
        stages = [from_mat_array('recon', '已有传统重建 · PhaseMap + InvA', after,
                    ['切片', 'volume'], '读取旧工作项目 Smat；40 层 × 4 volume。', meta['fov_mm']),
            from_mat_array('before', '重建前 · RO FFT + RSS', before, ['切片', 'volume'],
                    '读取同一 MAT 的 SmatBeforePhaseMapInvA，与重建后逐帧配对。', meta['fov_mm'])]
        for s in stages:
            s['axes'][1]['values'] = ['b0', 'DWI-RO', 'DWI-PE/SPEN', 'DWI-SS']
        status, status_label = 'paired', '重建前后配对'
        details = dict(variables=['Smat', 'SmatBeforePhaseMapInvA'], source=str(source_mat))
    elif source_mat.is_file():
        fields = [v[0] for v in row['mat_fields']]
        if row['group'] == '50slice':
            variable = next(v for v in fields if v.startswith('SignalFixedPostROFFTPostSR_'))
            a = loadmat(source_mat, variable_names=[variable])[variable]
            original_shape = list(a.shape)
            assert a.shape[2] == 4 and a.shape[3] == 50
            mag = np.sqrt((np.abs(a.astype(np.complex128)) ** 2).sum(2))
            labels = ['切片', '旧数组轴5'] + (['旧数组轴6'] if mag.ndim == 5 else [])
            warnings.append('额外轴保留原 MAT 编号；轴5/轴6 尚未完整对应到重复、band 或扩散条件，不作推断。')
            warnings.append('该组来自 GoogleDrive 50slice 文件夹；具体解剖部位和受试者身份尚未核实，未合并计入已确认人脑扫描。')
            note = f'从已有复数重建 {variable} 按 coil 轴做 RSS，未重新重建。'
        else:
            variable = 'MagImagesCombined'
            a = loadmat(source_mat, variable_names=[variable])[variable]
            original_shape = list(a.shape)
            # Some legacy magnitude arrays retain the singleton coil dimension.
            if a.ndim >= 4 and a.shape[2] == 1:
                a = np.squeeze(a, axis=2)
            mag = np.abs(a)
            assert mag.shape[2] == meta['header_slices']
            labels = ['切片'] + [f'附加轴{i+1}' for i in range(mag.ndim - 3)]
            warnings.append('附加轴逐个保留；不能仅凭轴长度把每帧称为独立扩散方向。')
            note = f'读取已有幅度重建 {variable}；仅移除残留的 singleton coil 轴。'
        stages = [from_mat_array('recon', '已有传统重建', mag, labels, note, meta['fov_mm'])]
        status, status_label = 'legacy_reconstruction', '已有重建'
        details = dict(variable=variable, original_shape=original_shape, source=str(source_mat))
    else:
        s, details = raw_only(path, meta)
        stages = [s]
        status, status_label = 'raw_only', '仅原始输入'
        warnings.append('没有同名旧重建 MAT，本页显示新生成的 RO-only 输入，不能当作 PE 重建图。')
        if any(not c['complete'] for c in details['frame_coverage']) or len(details['acquired_slice_counters']) < meta['header_slices']:
            status, status_label = 'incomplete_raw', '不完整采集 · 输入预览'
            warnings.append(f"文件头计划 {meta['header_slices']} 层、{meta['header_repetitions']} 次采集；实际仅记录 {len(details['acquired_slice_counters'])} 层、{len({c['counters'].get('Rep', 0) for c in details['frame_coverage']})} 个 Rep。只展示实际采集范围。")
            if any(not c['complete'] for c in details['frame_coverage']):
                warnings.append('部分帧缺失 PE 行，缺行位置为零；不是补采或有效完整图像。')
    groups = {'50slice': '50slice', 'cerebellum': 'cerebellum', 'full_brain_folder': 'brain'}
    subtitles = {'50slice': 'GoogleDrive 50 层数据', 'cerebellum': 'Trio 小脑', 'full_brain_folder': 'Trio 全脑目录'}
    entry = dict(id=ident, scan=ident, family='hybrid', group=groups[row['group']], title=ident,
        subtitle=subtitles[row['group']], status=status, status_label=status_label,
        source=str(path), source_mat=str(source_mat) if source_mat.is_file() else None,
        geometry=meta, warnings=warnings, source_details=details)
    return save_entry(out, entry, stages)


def overview(out, entries):
    font = ImageFont.truetype(FONT, 19)
    small = ImageFont.truetype(FONT, 14)
    crossed = [e for e in entries if e['family'] == 'crossed' and e['frame_count'] > 1]
    selected = [e for e in entries if e['family'] == 'crossed' and e['frame_count'] == 1]
    hybrid = [e for e in entries if e['family'] == 'hybrid']
    collections = [('crossed', f'xSPEN · {len(crossed or selected)} 份扫描 / 选例', crossed or selected),
                   ('hybrid', f'Hybrid SPEN · {len(hybrid)} 份采集', hybrid)]
    if crossed and selected:
        collections.append(('selected', f'xSPEN · {len(selected)} 个选例', selected))
    for family, label, group in collections:
        cols = 4 if family == 'hybrid' else 5
        rows = int(np.ceil(len(group) / cols))
        image = Image.new('RGB', (cols * 320 + 24, rows * 330 + 90), '#ffffff')
        draw = ImageDraw.Draw(image)
        draw.text((18, 16), label + '（每项代表帧；完整数据见 HTML）', font=font, fill='#222222')
        for i, e in enumerate(group):
            x, y = 12 + (i % cols) * 320, 65 + (i // cols) * 330
            image.paste(Image.open(out / e['thumbnail']), (x, y))
            draw.text((x + 6, y + 264), e['title'], font=small, fill='#222222')
            draw.text((x + 6, y + 286), e['status_label'] + ' · ' + str(e['frame_count']) + ' 帧', font=small, fill='#888888')
        image.save(out / f'overview_{family}.jpg', quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--crossed-run', type=Path)
    parser.add_argument('--hybrid-inventory', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--refresh-view', action='store_true', help='Refresh HTML/CSS/JS and preview images from this run, without reading source data')
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(HERE / 'runs')
    if args.refresh_view:
        manifest = json.loads((out / 'manifest.json').read_text())
        for entry in manifest['entries']:
            if entry['family'] in ['archive','bruker']:
                continue
            with np.load(out / entry['arrays']) as arrays:
                stages = [{**s, 'frames': arrays[s['id']]} for s in entry['stages']]
            save_entry(out, entry, stages)
        dump(out / 'manifest.json', manifest)
        (out / 'manifest.js').write_text('window.REVIEW_MANIFEST=' + json.dumps(manifest, ensure_ascii=False) + ';\n')
        overview(out, manifest['entries'])
        for name in ['index.html', 'viewer.js', 'viewer.css', 'archive_pixels.js']:
            shutil.copy2(HERE / 'web' / name, out / name)
        provenance = json.loads((out / 'provenance.json').read_text())
        provenance['view_refreshed_utc'] = datetime.now(timezone.utc).isoformat()
        provenance['source_hashes'] = {str(p): digest(p) for p in [Path(__file__), *sorted((HERE / 'web').glob('*'))]}
        dump(out / 'provenance.json', provenance)
        print('Refreshed offline gallery:', out)
        return
    if args.crossed_run is None or args.hybrid_inventory is None:
        parser.error('--crossed-run and --hybrid-inventory are required unless --refresh-view is used')
    if out == args.crossed_run.resolve():
        raise ValueError('Output must differ from the source cache; build into a new run.')
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads((HERE / 'selection.json').read_text())
    inventory = json.loads(args.hybrid_inventory.read_text())
    dump(out / 'inventory/hybrid_inventory.json', inventory)
    provenance_dir = out / 'inventory/crossed_previous_verification'
    provenance_dir.mkdir(parents=True, exist_ok=True)
    prior = args.crossed_run / 'inventory/crossed_previous_verification'
    prior = prior if prior.is_dir() else args.crossed_run
    for name in ['verification.json', 'provenance.json', 'selection.json', 'inventory.json']:
        if (prior / name).exists():
            shutil.copy2(prior / name, provenance_dir / name)
    for name in ['source_snapshot']:
        if (prior / name).is_dir():
            shutil.copytree(prior / name, provenance_dir / name, dirs_exist_ok=True)
    entries = cross_entries(args.crossed_run, out)
    rows = sorted(inventory['records'], key=lambda r: (['full_brain_folder', 'cerebellum', '50slice'].index(r['group']), int(r['scan'][3:])))
    for row in rows:
        previous = out / 'entries' / row['scan'] / 'metadata.json'
        if args.resume and previous.is_file():
            entry = json.loads(previous.read_text())
        else:
            print(f"Reading {row['scan']} ...", flush=True)
            entry = hybrid_entry(row, out, Path(selection['hybrid_spen']['reconstruction_mat']))
        entries.append(entry)
        print(f"{entry['id']}: {entry['status']}, {entry['frame_count']} frames", flush=True)
    summary = dict(entry_count=len(entries), crossed_entries=10, hybrid_scans=16,
        distinct_scans=len({e['scan'] for e in entries}),
        hybrid_primary_frames=sum(e['frame_count'] for e in entries if e['family'] == 'hybrid'),
        status_counts=dict(Counter(e['status'] for e in entries)))
    dump(out / 'manifest.json', dict(summary=summary, entries=entries))
    (out / 'manifest.js').write_text('window.REVIEW_MANIFEST=' + json.dumps(dict(summary=summary, entries=entries), ensure_ascii=False) + ';\n')
    overview(out, entries)
    for name in ['index.html', 'viewer.js', 'viewer.css', 'archive_pixels.js']:
        shutil.copy2(HERE / 'web' / name, out / name)
    dump(out / 'verification.json', dict(passed=True, **summary,
        checked='Every stored image finite/nonnegative; MAT slice/coil dimensions checked; uint16 display quantization bounded; source arrays preserved in float32 NPZ'))
    dump(out / 'provenance.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes={str(p): digest(p) for p in [Path(__file__), *sorted((HERE / 'web').glob('*'))]},
        note='Existing reconstructions visualized, no new PE inverse reconstruction. Three missing-MAT raw files read for RO-only preview.'))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
