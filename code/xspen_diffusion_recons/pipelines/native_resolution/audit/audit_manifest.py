"""Read-only source audit; all generated files remain beside this script.

Run in the project Python environment. Raw Siemens data and prior
scanner exports are opened read-only. No GPU imports are needed.
"""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import struct

import h5py
import numpy as np

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
WORKSPACE = Path(os.environ.get('XSPEN_LEGACY_ROOT', str(PROJECT/'data/legacy'))).expanduser().resolve()
OLD_AUDIT = WORKSPACE / '02_说明文档/数据盘点_2026-09-10'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def headers(path):
    """Read exact VB header or each VD/VE measurement header via offset table."""
    with Path(path).open('rb') as stream:
        first, second = struct.unpack('<II', stream.read(8))
        if first == 0 and 0 < second <= 64:
            offsets = []
            for _ in range(second):
                entry = struct.unpack('<IIQQ64s64s', stream.read(152))
                offsets.append(entry[2])
        else:
            offsets = [0]
        result = []
        for offset in offsets:
            stream.seek(offset)
            size = struct.unpack('<I', stream.read(4))[0]
            if size < 4 or size > 16 * 1024 * 1024:
                raise ValueError(f'Implausible Siemens header size: {path}, {size}')
            stream.seek(offset)
            data = stream.read(size)
            if len(data) != size:
                raise ValueError(f'Truncated header: {path}')
            result.append((offset, data))
    return result


def value(text, key):
    found = re.search('^' + re.escape(key) + r'\s*=\s*(.*?)\s*$', text, re.M)
    if found is None:
        return None
    raw = found.group(1)
    if raw.startswith('"'):
        return raw.strip('"')
    try:
        return float(raw) if any(c in raw for c in '.eE') else int(raw, 0)
    except ValueError:
        return raw


def inventory():
    previous = json.loads((OLD_AUDIT / 'siemens_dat.json').read_text())
    known = {row['path'] for row in previous}
    current = {str(path) for path in (WORKSPACE / 'xSPEN_项目').rglob('*.dat') if path.is_file()}
    save('raw_path_discovery.json', dict(current_dat_paths=len(current), previous_dat_paths=len(known),
                                      new_paths=sorted(current-known), missing_paths=sorted(known-current)))
    if current != known:
        raise ValueError('The raw-file listing changed; classify new or missing paths before reusing the previous inventory')
    rows = []
    for old in previous:
        if not old['is_spen']:
            continue
        path = Path(old['path'])
        parsed = []
        for offset, data in headers(path):
            text = data.decode('latin1')
            sequence = value(text, 'tSequenceFileName')
            if sequence and 'spen' in sequence.lower():
                parsed.append((offset, data, text, sequence))
        if len(parsed) != 1:
            raise ValueError(f'Expected exactly one SPEN measurement: {path}')
        offset, data, text, sequence = parsed[0]
        pe, ro = [value(text, f'sKSpace.{key}') for key in ['lPhaseEncodingLines', 'lBaseResolution']]
        fov = [value(text, f'sSliceArray.asSlice[0].{key}') for key in ['dPhaseFOV', 'dReadoutFOV']]
        normal = [value(text, f'sSliceArray.asSlice[0].sNormal.{key}') or 0 for key in ['dSag', 'dCor', 'dTra']]
        view = ['sagittal', 'coronal', 'axial'][int(np.argmax(np.abs(normal)))] if sum(abs(x) > .1 for x in normal) == 1 else 'oblique'
        mid = re.search(r'MID\d+', path.name).group()
        family = 'crossed_chirp_bipolarDiff' if 'bipolarDiff' in sequence else 'quadratic_hybrid_SPEN' if 'hyb_spen' in sequence.lower() else 'zz_xSPEN_3D_knee'
        selected = mid in ['MID112', 'MID114', 'MID27'] and family == 'crossed_chirp_bipolarDiff'
        status = 'selected_human_existing_adapter' if selected else 'exclude_human_phantom_visual_qc' if mid == 'MID78' else 'exclude_pending_anatomy_qc' if family == 'crossed_chirp_bipolarDiff' else 'separate_encoding_family_or_anatomy'
        r = value(text, 'sWiPMemBlock.alFree[14]')
        beta = value(text, 'sWiPMemBlock.adFree[2]')
        tp = value(text, 'sWiPMemBlock.alFree[15]')
        esp = value(text, 'sFastImaging.lEchoSpacing')
        rows.append(dict(scan_id=mid, path=str(path), source_bytes=path.stat().st_size,
                         source_size_matches_20260910=path.stat().st_size == old['bytes'],
                         source_group=old['group'], sequence=sequence, family=family,
                         header_offset=offset, header_bytes=len(data), header_sha256=hashlib.sha256(data).hexdigest(),
                         native_header_shape_pe_ro=[pe, ro], fov_mm=fov,
                         nominal_pixel_mm=[f/n if f and n else None for f, n in zip(fov, [pe, ro])],
                         thickness_mm=value(text, 'sSliceArray.asSlice[0].dThickness'),
                         header_slices=value(text, 'sSliceArray.lSize'), view=view, slice_normal=normal,
                         r_value=r if family == 'crossed_chirp_bipolarDiff' else None,
                         beta=beta if family == 'crossed_chirp_bipolarDiff' else None,
                         chirp_duration_s=tp*1e-6 if tp and family == 'crossed_chirp_bipolarDiff' else None,
                         echo_spacing_s=esp*1e-6 if esp else None,
                         timing_ratio_4betaTp_over_NpeESP=4*beta*tp/(pe*esp) if beta and tp and pe and esp and family == 'crossed_chirp_bipolarDiff' else None,
                         nominal_protocol_b_value=value(text, 'sWiPMemBlock.adFree[4]') if family == 'crossed_chirp_bipolarDiff' else None,
                         selected=selected, qc_status=status))
    rows.sort(key=lambda row: (row['family'], row['path']))
    save('siemens_header_manifest.json', rows)
    with (OUT / 'siemens_header_manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    return rows


def selected_exports(rows):
    human_qc = json.loads((PROJECT / 'scanner/human_qc.json').read_text())
    checked = []
    for row in rows:
        if not row['selected']:
            continue
        mid = row['scan_id']
        meta_path = PROJECT / 'scanner' / f'{mid}.json'
        meta = json.loads(meta_path.read_text())
        h5path = meta_path.with_suffix('.h5')
        source_hash = sha256(meta['source'])
        samples = []
        with h5py.File(h5path, 'r') as file:
            dataset = file['kspace']
            h5_meta = json.loads(file.attrs['metadata'])
            for repeat in sorted({0, dataset.shape[0]//2, dataset.shape[0]-1}):
                for slice_index in sorted({dataset.shape[1]//4, dataset.shape[1]//2, 3*dataset.shape[1]//4}):
                    array = dataset[repeat, slice_index]
                    samples.append(dict(repeat=repeat, slice_index=slice_index, finite=bool(np.isfinite(array).all()),
                                        nonzero=bool(np.any(array)), rms=float(np.sqrt(np.mean(np.abs(array)**2))),
                                        sha256=hashlib.sha256(array.tobytes()).hexdigest()))
            shape = list(dataset.shape)
            dtype = str(dataset.dtype)
        if shape != meta['shape_rep_slice_coil_pe_ro'] or h5_meta != meta or source_hash != meta['source_sha256']:
            raise AssertionError(f'Export provenance mismatch: {mid}')
        if not all(sample['finite'] and sample['nonzero'] for sample in samples):
            raise AssertionError(f'Invalid sampled observation: {mid}')
        checked.append(dict(scan_id=mid, raw_path=meta['source'], raw_sha256=source_hash,
                            matches_prior_source_sha256=True, h5_path=str(h5path), metadata_path=str(meta_path),
                            metadata_sha256=sha256(meta_path), shape_rep_slice_coil_pe_ro=shape, dtype=dtype,
                            metadata_matches_h5=True, observation_count=shape[0]*shape[1],
                            native_shape=shape[-2:], fov_mm=meta['fov_mm'],
                            native_pixel_mm=[f/n for f,n in zip(meta['fov_mm'],shape[-2:])],
                            thickness_mm=meta['thickness_mm'], r_value=meta['r_value'], beta=meta['beta'],
                            view=human_qc[mid]['view'], nominal_protocol_b_value=meta['nominal_protocol_b_value'],
                            excluded_original_slice_counters=meta['excluded_original_slice_counters'],
                            calibration_status=meta['calibration_status'], samples=samples,
                            sample_check_scope='Nine slice/acquisition observations, not exhaustive payload QC or reconstruction'))
    save('selected_exports.json', checked)
    return checked


def main():
    rows = inventory()
    checked = selected_exports(rows)
    refs = [PROJECT / 'operators.py', PROJECT / 'scanner.py', PROJECT / 'prepare_scanner.py',
            PROJECT / 'diagnostics/operator_alignment_20260909/alignment.json',
            PROJECT / 'test_physics.py', OUT.parent / 'protocols.json',
            WORKSPACE / '01_工作项目/xspen_recons/diagnostics/20260908_model_audit/audit.json',
            OLD_AUDIT / 'README.md', PROJECT / 'scanner/human_qc.json', PROJECT / 'scanner/exclusions.json']
    save('evidence_files.json', [dict(path=str(path), sha256=sha256(path)) for path in refs])
    summary = dict(date='2026-09-13', no_gpu_used=True, original_data_modified=False,
                   header_audit_scans=len(rows), family_counts=dict(Counter(row['family'] for row in rows)),
                   selected_scans=len(checked), selected_observations=sum(row['observation_count'] for row in checked),
                   selected_sequence_families=1, selected_geometry_protocols=2,
                   selected_unique_source_hashes=len({row['raw_sha256'] for row in checked}),
                   unique_subject_count='Not independently verified; MID112/MID114 share a named session; acquisition groups are not independent subjects',
                   selected_export_samples_checked=sum(len(row['samples']) for row in checked),
                   source_sizes_match_previous=all(row['source_size_matches_20260910'] for row in rows),
                   simulation_prior_profiles=2, grids_per_profile=4, independent_priors_planned=8)
    save('summary.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
