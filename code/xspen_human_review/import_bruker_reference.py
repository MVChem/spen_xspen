"""Import existing Bruker EPI/FLASH/RARE scanner images, without raw reconstruction.

The reference inventory was independently checked against the 543 SPEN scans.
This script only writes separate entries and a reference manifest; it never edits
the shared gallery manifest, web assets, or any acquisition source file.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from build_unified import dump
from catalog_all import export_array


HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jcamp(path):
    """Read literal JCAMP parameters, including Bruker's @N*(value) runs."""
    text = Path(path).read_text(errors='replace')
    text = '\n'.join(line for line in text.splitlines() if not line.startswith('$$'))
    values = {}
    for match in re.finditer(r'^##\$([^=]+)=(.*?)(?=^##|\Z)', text, re.M | re.S):
        key, value = match.group(1), match.group(2).strip()
        # Array declaration is on the first line; parenthesized frame-group
        # records on following lines must remain intact.
        value = re.sub(r'^\(\s*\d+(?:\s*,\s*\d+)*\s*\)\s*\n', '', value)
        values[key] = value
    return values


def numbers(value):
    value = re.sub(r'@(\d+)\*\(([^()]*)\)',
                   lambda m: ' '.join([m.group(2)] * int(m.group(1))), value)
    return np.asarray([float(x) for x in value.replace(',', ' ').split()], dtype=np.float64)


def read_reference(path):
    """Return scanner-coordinate frames and explicit scale/geometry evidence."""
    path = Path(path)
    visu_path, reco_path = path.parent/'visu_pars', path.parent/'reco'
    visu = jcamp(visu_path)
    core_dim = int(visu['VisuCoreDim'])
    size = numbers(visu['VisuCoreSize']).astype(int)
    frame_count = int(visu['VisuCoreFrameCount'])
    assert core_dim == 2 and len(size) == 2, 'Unexpected spatial dimensionality; do not guess an axis.'
    dtype_map = {'_16BIT_SGN_INT': 'i2', '_32BIT_SGN_INT': 'i4',
                 '_8BIT_UNSGN_INT': 'u1', '_32BIT_FLOAT': 'f4'}
    endian = {'littleEndian': '<', 'bigEndian': '>'}[visu['VisuCoreByteOrder']]
    dtype = np.dtype(endian + dtype_map[visu['VisuCoreWordType']])
    stored = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(size)) * frame_count
    assert stored.size == expected, f'Payload length {stored.size} != declared {expected}'
    # Bruker writes X fastest, then Y, then the declared frame groups.
    stored = stored.reshape(frame_count, int(size[1]), int(size[0]))
    slopes = numbers(visu['VisuCoreDataSlope'])
    offsets = numbers(visu['VisuCoreDataOffs'])
    assert slopes.size in (1, frame_count) and offsets.size in (1, frame_count)
    slopes = np.broadcast_to(slopes, (frame_count,))
    offsets = np.broadcast_to(offsets, (frame_count,))
    scaled64 = stored.astype(np.float64) * slopes[:, None, None] + offsets[:, None, None]
    frames = scaled64.astype(np.float32)
    assert np.isfinite(frames).all()
    assert np.all(np.abs(frames.astype(float)-scaled64) <= np.maximum(np.abs(scaled64), 1)*6e-8)
    groups = [(int(count), label) for count, label in
              re.findall(r'\(\s*(\d+)\s*,\s*<([^>]+)>', visu.get('VisuFGOrderDesc', ''))]
    if groups:
        # The audited sources use one FG_SLICE group. Fail loudly if a future
        # input has other groups, instead of relabeling echo/repetition as slice.
        assert groups == [(frame_count, 'FG_SLICE')], f'Unmapped frame groups: {groups}'
        axes = [dict(label='切片', size=frame_count)]
    else:
        assert frame_count == 1, 'Multiple frames without declared groups'
        axes = [dict(label='图像', size=1)]
    extent = numbers(visu['VisuCoreExtent'])
    assert len(extent) == 2 and np.all(extent > 0)
    fov = [float(extent[1]), float(extent[0])]
    thickness = numbers(visu['VisuCoreFrameThickness'])
    scan = path.parents[2]
    study = scan.parent
    subject = jcamp(study/'subject')
    metadata = dict(
        study=study.name, scan_id=scan.name, original_shape=list(stored.shape),
        stored_dtype=dtype.str, frame_count=frame_count, frame_groups=groups,
        scanner_subject_type=subject.get('SUBJECT_type'), species_verified=False,
        subject_label='小鼠（扫描名称标注）' if 'mouse' in study.name.lower() else '物种未核实',
        visu_core_size=size.tolist(), visu_core_extent_mm=extent.tolist(),
        visu_core_data_slope=slopes.tolist(), visu_core_data_offset=offsets.tolist(),
        scale_formula='value = stored * VisuCoreDataSlope + VisuCoreDataOffs',
        visu_core_orientation=numbers(visu['VisuCoreOrientation']).tolist(),
        visu_core_position=numbers(visu['VisuCorePosition']).tolist(),
        frame_thickness_mm=thickness.tolist(), thickness_mm=float(thickness[0]),
        metadata_sources=[dict(path=str(p), sha256=digest(p)) for p in [visu_path, reco_path]],
        no_spatial_reorientation=True, no_raw_reconstruction=True, fov_unit='mm',
        float32_max_abs_rounding_error=float(np.abs(frames.astype(float)-scaled64).max()),
        display_note='扫描仪已保存的参考序列 2dseq；按 visu_pars 的字节序、矩阵、逐帧 slope/offset 读取。'
                     '完整保留帧数和扫描仪方向；本轮未重建 raw、未插值或配准。这是 EPI/FLASH/RARE 参考影像，不是 SPEN 采集。',
    )
    if subject.get('SUBJECT_type') == 'Human':
        metadata['subject_type_warning'] = '旧 PV5 Subject_type=Human 与小FOV/动物实验背景冲突，物种未核实，不能据此归入人脑。'
    return frames, axes, fov, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert run.is_relative_to(HERE/'runs')
    source_list = run/'inventory/bruker_reference_sources.json'
    inventory = json.loads(source_list.read_text())
    entries, records = [], []
    for scan_record in inventory['records']:
        if scan_record['is_spen']:
            continue
        for record in scan_record['reconstructions']:
            path = Path(record['path'])
            if not record['bytes']:
                records.append(dict(source=str(path), status='empty_file', entry_ids=[]))
                continue
            try:
                frames, axes, fov, details = read_reference(path)
                method = scan_record['method'].strip('<>').split(':')[-1]
                details.update(sequence=method,
                               title=f'Bruker 参考 · {method} · {details["study"]} · scan {details["scan_id"]}',
                               subtitle=f'{details["subject_label"]} · {method} · {len(frames)} 帧 · 非 SPEN')
                row = dict(path=str(path), source=str(path), kind='reference', sha256=digest(path))
                variable = f'2dseq:{details["study"]}:scan{details["scan_id"]}'
                entry = export_array(run, row, variable, frames, axes, fov, details, source_kind='reference')
                # Re-read the downloadable image and compare every value with
                # independently decoded/scaled scanner data, including sign.
                with np.load(run/entry['arrays']) as saved:
                    assert np.array_equal(saved['image'], frames)
                entries.append(entry)
                records.append(dict(source=str(path), status='displayed', entry_ids=[entry['id']],
                                    study=details['study'], scan_id=details['scan_id'], method=method,
                                    frames=len(frames), shape=list(frames.shape), finite=True,
                                    source_sha256=row['sha256'], downloaded_array_exact_float32_match=True,
                                    metadata_sources=details['metadata_sources']))
            except Exception as error:
                records.append(dict(source=str(path), status='read_error', reason=f'{type(error).__name__}: {error}', entry_ids=[]))
        if len(entries) and len(entries) % 20 == 0:
            print(f'Imported {len(entries)} reference scans', flush=True)
    summary = dict(reference_scan_count=len(entries), frame_count=sum(e['frame_count'] for e in entries),
                   status_counts=dict(Counter(r['status'] for r in records)),
                   method_counts=dict(Counter(r['method'] for r in records if r['status']=='displayed')),
                   all_displayed_values_verified=all(r.get('downloaded_array_exact_float32_match') for r in records if r['status']=='displayed'),
                   source_inventory_sha256=digest(source_list), importer_sha256=digest(__file__))
    dump(run/'inventory/bruker_reference_entries.json', dict(summary=summary, entries=entries, records=records))
    audit_path = run/'inventory/additional_source_audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        audit['reference_import_result'] = summary
        dump(audit_path, audit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary['status_counts'].get('read_error'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
