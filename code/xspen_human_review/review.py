"""Read-only legacy xSPEN inventory, selected ADC extraction and CPU reconstructions."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.io import loadmat, whosmat

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / 'xspen_diffusion_recons'
sys.path[:0] = [str(PROJECT), str(PROJECT / 'pipelines/native_resolution/real_comparison')]
DEFAULT_LEGACY = Path('/home/data2/chk/workspace/2026/08/14')
HUMAN = ['MID112', 'MID114', 'MID27', 'MID51', 'MID613', 'MID615', 'MID106', 'MID108']
VIEWS = dict(zip(HUMAN, ['axial', 'sagittal', 'axial', 'sagittal', 'axial', 'coronal', 'axial', 'sagittal']))
PHANTOM = ['MID78', 'MID530', 'MID533', 'MID535', 'MID537', 'MID74', 'MID76']


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rss(data):
    return np.sqrt(np.sum(np.abs(data) ** 2, axis=0))


def ro_fft(data):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(data, axes=-1), axis=-1, norm='ortho'), axes=-1)


def case_id(case):
    return f"{case['scan']}_sl{case['slice']:03d}_occ{case['occurrence']:02d}"


def inventory(legacy):
    old = legacy / 'xspen_diffusion_recons'
    exp = old / 'experiments/native_20260913'
    rows = json.loads((exp / 'audit/siemens_header_manifest.json').read_text())
    exports = {}
    for folder in [old / 'scanner', exp / 'expanded_human_20260914/scanner']:
        for path in sorted(folder.glob('*.h5')):
            with h5py.File(path, 'r') as data:
                meta = json.loads(data.attrs['metadata'])
                assert list(data['kspace'].shape) == meta['shape_rep_slice_coil_pe_ro']
            exports[path.stem] = (path, meta)
    for row in rows:
        mid = row['scan_id']
        row['classification_for_this_review'] = ('human_brain' if mid in HUMAN else
            'phantom' if mid in PHANTOM else 'unconfirmed_companion' if mid == 'MID80' else
            'different_sequence_family' if row['family'] != 'crossed_chirp_bipolarDiff' else 'not_reviewed')
        row['source_exists_now'] = Path(row['path']).is_file()
        row['legacy_header_inventory_source'] = str(exp / 'audit/siemens_header_manifest.json')
        if mid in exports:
            path, meta = exports[mid]
            row['scanner_export'] = str(path)
            row['export_metadata'] = meta
            mat = Path(meta['source']).with_suffix('.mat')
            row['same_stem_mat'] = str(mat) if mat.is_file() else None
            row['mat_variables'] = whosmat(mat) if mat.is_file() else []
    return rows, exports


def input_preview(cases, exports, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(15, 8.3))
    for case, ax in zip(cases, axes.flat):
        path, meta = exports[case['scan']]
        with h5py.File(path) as data:
            raw = data['kspace'][case['occurrence'], case['slice']]
        im = rss(ro_fft(raw))
        ax.imshow(im, cmap='gray', vmin=0, vmax=np.quantile(im, .995), interpolation='nearest',
                  aspect=(meta['fov_mm'][0] / im.shape[0]) / (meta['fov_mm'][1] / im.shape[1]))
        ax.set_title(f"{case_id(case)}\n{VIEWS[case['scan']]} | {im.shape[0]} x {im.shape[1]}", fontsize=10)
        ax.axis('off')
    fig.suptitle('10 human xSPEN observations: RO FFT + coil RSS, no PE inverse', fontsize=15)
    fig.text(.5, .015, 'Input-only selection; each panel p99.5. Native PE/RO orientation; indices start at 0.', ha='center')
    fig.tight_layout(rect=(0, .04, 1, .95), h_pad=3)
    fig.savefig(out / 'input_selection.png', dpi=150)
    plt.close(fig)


def extract_adc(path, meta, cases, target):
    """Re-read actual ADC, reconstruct the reader output, and compare with legacy H5.

    Identical count policy to prepare_scanner.export; no .dat mutation. Save the
    unreflected, oversampled ADC separately from the regridded/de-oversampled input.
    """
    import twixtools
    scans = twixtools.read_twix(str(path), parse_geometry=False, parse_pmu=False, verbose=False)
    if len(scans) != 1:
        raise ValueError('Expected one VB measurement')
    scan = scans[0]
    wanted = {(meta['slice_order'][c['slice']], c['occurrence']): c for c in cases}
    blocks = defaultdict(dict)
    counts = Counter()
    for mdb in scan['mdb']:
        if not mdb.is_image_scan():
            continue
        counter = mdb.mdh.Counter
        if any(getattr(counter, dim) for dim in ['Rep', 'Set', 'Ave', 'Eco', 'Par', 'Phs', 'Ida', 'Idb', 'Idc', 'Idd', 'Ide']):
            raise ValueError('Unexpected acquisition counter')
        sl, line = int(counter.Sli), int(counter.Lin)
        occurrence = counts[sl, line]
        counts[sl, line] += 1
        counter.Rep = occurrence
        if (sl, occurrence) in wanted:
            blocks[sl, occurrence][line] = mdb
    bad = meta['excluded_original_slice_counters']
    scan['mdb'] = [m for m in scan['mdb'] if not m.is_image_scan() or int(m.mdh.Counter.Sli) not in bad]
    mapped = twixtools.map_twix(scan, verbose=False)['image']
    for dim in mapped.flags['average']:
        mapped.flags['average'][dim] = False
    mapped.flags['average']['Seg'] = True
    mapped.flags['remove_os'] = True
    mapped.flags['regrid'] = True
    result = {}
    for key, case in wanted.items():
        sl, occurrence = key
        lines = blocks[key]
        m, k = meta['shape_rep_slice_coil_pe_ro'][-2:]
        assert sorted(lines) == list(range(m))
        adc = np.stack([lines[i].data for i in range(m)], axis=1)
        selection = [0] * len(mapped.dim_order)
        for dim in ['Cha', 'Lin', 'Col']:
            selection[mapped.dim_order.index(dim)] = slice(None)
        selection[mapped.dim_order.index('Sli')] = sl
        selection[mapped.dim_order.index('Rep')] = occurrence
        fresh = np.asarray(mapped[tuple(selection)]).reshape(m, adc.shape[0], k).transpose(1, 0, 2)
        with h5py.File(target) as data:
            saved = data['kspace'][occurrence, case['slice']]
        relerr = float(np.linalg.norm(fresh - saved) / np.linalg.norm(saved))
        np.testing.assert_allclose(fresh, saved, rtol=2e-6, atol=2e-7)
        result[case_id(case)] = dict(adc=adc, processed=saved,
            reflect=np.array([lines[i].is_flag_set('REFLECT') for i in range(m)]),
            segment=np.array([int(lines[i].mdh.Counter.Seg) for i in range(m)]),
            verification=dict(fresh_reader_relative_error=relerr, adc_shape=list(adc.shape),
                original_slice_counter=sl, occurrence=occurrence, lines=m,
                adc_sha256=hashlib.sha256(adc.tobytes()).hexdigest(),
                raw_stage='Original complex ADC in recorded polarity, including RO oversampling; no filtering/regridding',
                processed_stage='Reflection correction, ramp regrid and RO oversampling removal; no parity correction'))
    return result


def reconstruct(raw, meta):
    import torch
    from traditional_phase import reconstruct_traditional, build_matrices
    from scanner import correct_even_odd
    torch.set_num_threads(2)
    mats = build_matrices(*raw.shape[-2:], meta['r_value'], meta['beta'])
    x = torch.from_numpy(raw)
    ro = x @ mats['readout'].conj()
    np.testing.assert_allclose(ro.numpy(), ro_fft(raw), rtol=3e-4, atol=2e-6)
    corrected, phase, coeff = correct_even_odd(ro[None])
    a = mats['encoding']
    inverse = torch.linalg.solve(a.mH @ a + .01 * torch.eye(a.shape[1]), a.mH)
    tik_coils = (inverse @ corrected)[0]
    traditional = reconstruct_traditional(raw, meta)
    k = raw.shape[-1]
    window = np.exp(-.001 * (np.arange(1, k + 1) - k / 2) ** 2)
    arrays = dict(processed_raw=raw, ro_complex=ro.numpy(), raw_ro_rss=rss(ro.numpy()),
        raw_adc_placeholder=np.empty(0),
        legacy_recipe_rss=rss(ro_fft(raw * window[None, None, :])),
        tikhonov_coils=tik_coils.numpy(), tikhonov_rss=rss(tik_coils.numpy()),
        tikhonov_phase_rad=phase.numpy(), tikhonov_corrected_ro=corrected[0].numpy(),
        phasemap_rss=traditional['magnitude'], phasemap_coils=traditional['coil_images'],
        phasemap_rad=traditional['phase_map_rad'], phasemap_corrected_ro=traditional['corrected_ro_image'],
        windowed_adjoint_rss=traditional['windowed_adjoint_only'], encoding_a=a.numpy())
    arrays.pop('raw_adc_placeholder')
    details = dict(phasemap=traditional['metadata'], tikhonov=dict(alpha=.01,
        phase='scanner.correct_even_odd: circular quadratic fit of neighboring RO rows', coefficients=coeff,
        equation='RSS_coils[(A^H A + 0.01 I)^-1 A^H RO_corrected]; A normalized by spectral norm'),
        legacy_recipe='exp(-0.001*((1:Nro)-Nro/2)^2), RO centered FFT, RSS; normalized FFT units here')
    return arrays, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', type=Path, default=DEFAULT_LEGACY)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection', type=Path, default=HERE / 'selection.json')
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse completed cases from this run after an interrupted batch')
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(HERE / 'runs'):
        parser.error('--out must be inside this project runs/')
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads(args.selection.read_text())
    cases = selection['cases']
    assert len({case_id(c) for c in cases}) == len(cases)
    rows, exports = inventory(args.legacy)
    write_json(out / 'inventory.json', rows)
    write_json(out / 'selection.json', selection)
    input_preview(cases, exports, out)
    if args.preview_only:
        return
    records = []
    group = defaultdict(list)
    for case in cases:
        group[case['scan']].append(case)
    for mid, members in group.items():
        if args.resume and all((out / 'cases' / case_id(c) / 'metadata.json').is_file() and
                               (out / 'cases' / case_id(c) / 'arrays.npz').is_file() for c in members):
            for case in members:
                record = json.loads((out / 'cases' / case_id(case) / 'metadata.json').read_text())
                assert all(record[key] == case[key] for key in ['scan', 'slice', 'occurrence', 'reason'])
                records.append(record)
            print(f'{mid}: resume completed cases', flush=True)
            continue
        h5, meta = exports[mid]
        rawpath = Path(meta['source'])
        print(f'{mid}: read original ADC and verify scanner preprocessing', flush=True)
        rawhash = sha256(rawpath)
        assert rawhash == meta['source_sha256'], f'Source changed: {rawpath}'
        adc_records = extract_adc(rawpath, meta, members, h5)
        mat = rawpath.with_suffix('.mat')
        original_img = loadmat(mat, variable_names=['Img'])['Img'] if mat.is_file() else None
        for case in members:
            ident = case_id(case)
            target = out / 'cases' / ident
            target.mkdir(parents=True, exist_ok=True)
            extracted = adc_records[ident]
            arrays, method = reconstruct(extracted['processed'], meta)
            arrays.update(raw_adc=extracted['adc'], raw_adc_reflect=extracted['reflect'], raw_adc_segment=extracted['segment'])
            record = dict(case, id=ident, view=VIEWS[mid], selection_reason=case['reason'],
                raw_source=str(rawpath), scanner_h5=str(h5), source_sha256=rawhash,
                source_sha256_recomputed=True, metadata=meta, methods=method,
                adc_verification=extracted['verification'], original_mat=None)
            if original_img is not None:
                nrep = meta['repeated_counter_occurrences']
                ns = original_img.shape[-1] // nrep
                counter = meta['slice_order'][case['slice']]
                # Invert the same odd/even anatomical slice permutation used in xSPEN_Siemens.m.
                order = list(range(1, ns, 2)) + list(range(0, ns, 2)) if ns % 2 == 0 else list(range(0, ns, 2)) + list(range(1, ns, 2))
                orig = order[counter]
                frame = case['occurrence'] * ns + orig
                saved = original_img[:, :, 0, frame]
                # Normalize MATLAB's nonunitary FFT by sqrt(Nro), not by image-fitted gain.
                arrays['legacy_saved_rss'] = saved / np.sqrt(saved.shape[1])
                recipe = arrays['legacy_recipe_rss']
                correlations = [float(np.corrcoef(recipe.ravel(), original_img[:, :, 0, r * ns + orig].ravel())[0, 1]) for r in range(nrep)]
                record['original_mat'] = dict(path=str(mat), variable='Img', frame_zero_based=frame,
                    slice_zero_based=orig, occurrence=case['occurrence'], matlab_fft_divisor=float(np.sqrt(saved.shape[1])),
                    recipe_correlation=correlations[case['occurrence']], best_matching_occurrence=int(np.argmax(correlations)),
                    occurrence_correlations=correlations,
                    relative_error=float(np.linalg.norm(recipe-arrays['legacy_saved_rss'])/np.linalg.norm(arrays['legacy_saved_rss'])))
                if (record['original_mat']['recipe_correlation'] <= .98 or
                        record['original_mat']['best_matching_occurrence'] != case['occurrence']):
                    record['unmatched_mat_candidate'] = record.pop('original_mat')
                    record['unmatched_mat_candidate']['status'] = 'Same-stem Img fails same-observation pairing; excluded from matched comparison'
                    record['original_mat'] = None
                    arrays['unmatched_mat_candidate_rss'] = arrays.pop('legacy_saved_rss')
            for name, array in arrays.items():
                if not np.isfinite(array).all():
                    raise ValueError(f'Nonfinite {ident}/{name}')
            np.savez_compressed(target / 'arrays.npz', **arrays)
            write_json(target / 'metadata.json', record)
            records.append(record)
            print(f'{ident}: done; fresh raw/H5 error={extracted["verification"]["fresh_reader_relative_error"]:.2g}', flush=True)
        write_json(out / 'manifest.partial.json', records)
    records.sort(key=lambda c: [case_id(x) for x in cases].index(c['id']))
    write_json(out / 'manifest.json', records)
    code_sources = [Path(__file__), HERE / 'render.py', args.selection, PROJECT / 'operators.py', PROJECT / 'scanner.py',
        PROJECT / 'prepare_scanner.py', PROJECT / 'pipelines/native_resolution/real_comparison/traditional_phase.py',
        HERE.parent / 'spenpy/spenpy/recon/even_odd.py']
    write_json(out / 'provenance.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        python=sys.executable, packages={p: importlib.metadata.version(p) for p in ['numpy','scipy','torch','h5py','matplotlib','twixtools']},
        code_sha256={str(p): sha256(p) for p in code_sources if p.is_file()},
        selection_basis=selection['selection_basis'], case_count=len(records), scan_count=len(group)))
    from render import render
    render(out)


if __name__ == '__main__':
    main()
