"""Preserve repeated VB counters; export human crossed-chirp scanner measurements.

Counters are changed only on twixtools' in-memory objects, never in source .dat.
Occurrence indices identify repeated acquisitions, not independently confirmed b-values.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import struct
import h5py
import numpy as np
import twixtools
from utils import HERE, sha256, write_json

from paths import RAW_SCANNER

ROOT = RAW_SCANNER

def raw_header(path):
    with open(path, 'rb') as f:
        size = struct.unpack('<I', f.read(4))[0]
        f.seek(0)
        return f.read(size).decode('latin1')

def inventory(out=None):
    rows = []
    for p in sorted(ROOT.rglob('*.dat')):
        if not p.is_file():
            continue
        h = raw_header(p)
        def value(k):
            match = re.search('^'+re.escape(k)+r'\s*=\s*(.*)$', h, re.M)
            return match.group(1).strip() if match else None
        seq = value('tSequenceFileName')
        rows.append(dict(source=str(p), bytes=p.stat().st_size, sequence=seq,
                         crossed_chirp=bool(seq and 'esaszz_xSPEN_180c180c_bipolarDiff' in seq)))
    write_json((Path(out) if out is not None else HERE/'scanner')/'inventory.json', rows)
    return rows

def export(path, out_dir=None):
    mid = re.search(r'MID\d+', path.name).group()
    out = (Path(out_dir) if out_dir is not None else HERE/'scanner')/f'{mid}.h5'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return
    scans = twixtools.read_twix(str(path), parse_geometry=False, parse_pmu=False, verbose=False)
    if len(scans) != 1:
        raise ValueError('Expected legacy VB single measurement')
    scan = scans[0]
    yaps = scan['hdr']['MeasYaps']
    if 'esaszz_xSPEN_180c180c_bipolarDiff' not in yaps['tSequenceFileName']:
        raise ValueError('Wrong sequence family')
    image = [m for m in scan['mdb'] if m.is_image_scan()]
    count = Counter()
    positions = {}
    normal = yaps['sSliceArray']['asSlice'][0]['sNormal']
    normal_vec = np.array([normal.get(k, 0) for k in ['dSag', 'dCor', 'dTra']])
    # This legacy sequence does not advance Rep/Set for its outer acquisition loop.
    # Refuse unfamiliar counters instead of silently averaging them.
    for m in image:
        c = m.mdh.Counter
        if any(getattr(c, d) != 0 for d in ['Rep', 'Set', 'Ave', 'Eco', 'Par', 'Phs', 'Ida', 'Idb', 'Idc', 'Idd', 'Ide']):
            raise ValueError('Unexpected varying counter; needs explicit mapping')
        key = (int(c.Sli), int(c.Lin))
        c.Rep = count[key]
        count[key] += 1
        pos = m.mdh.SliceData.SlicePos
        positions[int(c.Sli)] = [float(pos.Sag), float(pos.Cor), float(pos.Tra)]
    expected_repeats = Counter(count.values()).most_common(1)[0][0]
    bad_slices = sorted({s for (s, line), n in count.items() if n != expected_repeats})
    if len(bad_slices) > 4:
        raise ValueError('Too many incomplete repeated acquisition grids')
    if bad_slices:
        # Omit whole affected slices, preserving all repeats on every retained slice.
        # Never slide later diffusion acquisitions into an earlier missing counter.
        scan['mdb'] = [m for m in scan['mdb'] if not m.is_image_scan() or int(m.mdh.Counter.Sli) not in bad_slices]
    nrepeat = {expected_repeats}
    a = twixtools.map_twix(scan, verbose=False)['image']
    for d in a.flags['average']:
        a.flags['average'][d] = False
    a.flags['average']['Seg'] = True
    a.flags['remove_os'] = True
    a.flags['regrid'] = True
    sl = yaps['sSliceArray']['asSlice'][0]
    free = yaps['sWiPMemBlock']['alFree']
    double = yaps['sWiPMemBlock']['adFree']
    # Paired chirp blocks are a provisional mapping for this 2016 revision.
    r1, r2 = float(free[14]), float(free[18])
    tp1, tp2 = float(free[15])*1e-6, float(free[19])*1e-6
    beta = float(double[2])
    m, k = int(yaps['sKSpace']['lPhaseEncodingLines']), int(yaps['sKSpace']['lBaseResolution'])
    esp = float(yaps['sFastImaging']['lEchoSpacing'])*1e-6
    if r1 != r2 or tp1 != tp2 or not 0 < beta < 1 or abs(tp1*4*beta/(m*esp)-1) > .1:
        raise ValueError('Unmatched chirps/timing: reduced model not applicable')
    order = sorted((s for s in positions if s not in bad_slices), key=lambda s: np.dot(normal_vec, positions[s]))
    shape = dict(zip(a.dim_order, a.shape))
    if shape['Rep'] != next(iter(nrepeat)) or sum(s in order for s, line in count) != len(order)*m:
        raise ValueError('Incomplete slice/PE/repeat coverage')
    meta = dict(source=str(path), source_sha256=sha256(path), sequence=yaps['tSequenceFileName'],
                shape_rep_slice_coil_pe_ro=[shape['Rep'], len(order), shape['Cha'], m, k],
                r_value=r1, beta=beta, chirp_duration_s=tp1, echo_spacing_s=esp,
                fov_mm=[float(sl['dPhaseFOV']), float(sl['dReadoutFOV'])], thickness_mm=float(sl['dThickness']),
                echo_time_s=float(yaps['alTE'][0])*1e-6, nominal_protocol_b_value=float(double[4]),
                wip_longs=free, wip_doubles=double,
                slice_order=order, positions_lps_mm=[positions[s] for s in order],
                slice_normal_lps=normal_vec.tolist(), repeated_counter_occurrences=next(iter(nrepeat)),
                excluded_original_slice_counters=bad_slices,
                counter_multiplicity_histogram=dict(Counter(count.values())),
                regrid=True, remove_os=True, reflection_corrected=True, averaged=False,
                calibration_status='provisional_header_informed_reduced_model',
                parameter_evidence='Two equal chirp blocks alFree[14:18]/[18:22]; beta=adFree[2]; 4*beta*Tp approximately Npe*ESP. Exact bipolar revision WIP enum and full waveform not independently verified.',
                repeat_labels='Chronological occurrence for each original (slice,line), NOT verified diffusion direction or b0 labels')
    temp = out.with_suffix('.h5.tmp')
    with h5py.File(temp, 'w') as h5:
        ds = h5.create_dataset('kspace', shape=tuple(meta['shape_rep_slice_coil_pe_ro']), dtype=np.complex64,
                               chunks=(1, 1, shape['Cha'], m, k), compression='lzf')
        for repeat in range(shape['Rep']):
            selection = [0]*len(a.dim_order)
            for dim in ['Sli', 'Lin', 'Cha', 'Col']:
                selection[a.dim_order.index(dim)] = slice(None)
            selection[a.dim_order.index('Rep')] = repeat
            # Array keeps singleton axes; squeeze only known removed dimensions.
            values = np.asarray(a[tuple(selection)])
            values = values.reshape(shape['Sli'], m, shape['Cha'], k)
            values = values[order].transpose(0, 2, 1, 3)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite raw data')
            ds[repeat] = values
            print(json.dumps(dict(event='scanner_export', scan=mid, repeat=repeat, total=shape['Rep'])), flush=True)
        h5.attrs['metadata'] = json.dumps(meta)
    temp.replace(out)
    write_json(out.with_suffix('.json'), meta)

def main():
    global ROOT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT, help='Siemens raw .dat root (or XSPEN_RAW_SCANNER)')
    p.add_argument('--out', type=Path, default=HERE/'scanner', help='New scanner export directory')
    p.add_argument('--mids', nargs='+', default=['MID112', 'MID114', 'MID27'])
    args = p.parse_args()
    ROOT = args.source.expanduser().resolve()
    if not ROOT.is_dir():
        p.error(f'Siemens source directory does not exist: {ROOT}')
    args.out.mkdir(parents=True, exist_ok=True)
    rows = inventory(args.out)
    for mid in args.mids:
        candidates = [Path(r['source']) for r in rows if r['crossed_chirp'] and re.search(mid+r'_', Path(r['source']).name)]
        if len(candidates) != 1:
            raise ValueError(f'Expected one crossed-chirp source for {mid}')
        export(candidates[0], args.out)

if __name__ == '__main__':
    main()
