"""Independently verify archive file completeness, pixel rendering and source values."""
import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import re
import zipfile

import nibabel as nib
import numpy as np
from playwright.sync_api import sync_playwright
from scipy.io import loadmat


def read_manifest(run):
    p = run / 'manifest.json'
    return json.loads(p.read_text()) if p.exists() else json.loads((run / 'manifest.js').read_text().split('=', 1)[1].strip().rstrip(';'))


def payload_json(path):
    text = path.read_text()
    return json.loads(text.split(']=', 1)[1].strip().rstrip(';'))


def audit_descriptors(run, entries):
    checked = []
    for e in entries:
        for key in ['payload', 'arrays', 'metadata', 'thumbnail', 'complex_arrays', 'source_metadata']:
            if e.get(key):
                p = run / e[key]
                assert p.is_file() and p.stat().st_size > 0, (e['id'], key, p)
        stages = payload_json(run / e['payload'])
        assert len(stages) == len(e['stages']), e['id']
        with zipfile.ZipFile(run / e['arrays']) as z:
            for stage, expected in zip(stages, e['stages']):
                assert stage['id'] == expected['id'] and stage['shape'] == expected['shape'], e['id']
                assert np.prod([a['size'] for a in stage['axes']]) == stage['shape'][0], e['id']
                with z.open(stage['id'] + '.npy') as f:
                    version = np.lib.format.read_magic(f)
                    shape, fortran, dtype = np.lib.format._read_array_header(f, version)
                assert list(shape) == stage['shape'], (e['id'], shape, stage['shape'])
                if stage.get('chunked'):
                    assert len(stage['chunks']) == (shape[0] + stage['chunk_size'] - 1) // stage['chunk_size'], e['id']
                    for path in stage['chunks']:
                        p = run / path
                        assert p.is_file() and p.stat().st_size > 0, (e['id'], p)
        checked.append(e['id'])
    return dict(entries=len(checked), stage_descriptors_load=True, scientific_array_headers_match=True,
                all_referenced_files_exist=True, families=dict(Counter(e['family'] for e in entries)), archive_groups=dict(Counter(e.get('group') for e in entries if e['family'] == 'archive')))


def select_entries(entries, all_old=False):
    selected = [e for e in entries if e['family'] in ['crossed', 'hybrid'] and (all_old or e['id'] in ['MID112', 'MID253', 'MID1496', 'MID54'])]
    bruker = [e for e in entries if e['family'] == 'bruker']
    if bruker:
        selected.append(max(bruker, key=lambda e: e['frame_count']))
        nonsquare = [e for e in bruker if e['stages'][0]['shape'][1] != e['stages'][0]['shape'][2]]
        if nonsquare:
            selected.append(max(nonsquare, key=lambda e: e['frame_count']))
        selected.extend(e for e in bruker if e['id'] in ['bruker_c7908553153822c5', 'bruker_664122b30ec4025c'])
    reference = [e for e in entries if e.get('group') == 'reference' or e['family'] == 'reference']
    if reference:
        selected.append(max(reference, key=lambda e: e['frame_count']))
    archives = [e for e in entries if e['family'] == 'archive']
    selected.extend(e for e in archives if e['id'] == 'archive_913598c2354bfa24')
    for group in ['mat', 'nifti', 'dicom', 'raw']:
        candidates = [e for e in archives if e['group'] == group]
        if candidates:
            selected.append(max(candidates, key=lambda e: e['frame_count']))
    nifti = [e for e in archives if e['group'] == 'nifti']
    if nifti:
        selected.append(max(nifti, key=lambda e: e['stages'][0]['shape'][1] * e['stages'][0]['shape'][2]))
    for condition in [lambda e: e['stages'][0].get('signed'), lambda e: e['stages'][0].get('invalid_count'),
                      lambda e: e['group'] == 'mat' and e['source_details'].get('variable') == 'Img',
                      lambda e: e['group'] == 'nifti' and not e['stages'][0].get('signed'),
                      lambda e: e['group'] == 'nifti' and e['stages'][0].get('signed') and e['stages'][0].get('invalid_count')]:
        candidates = [e for e in archives if condition(e)]
        if candidates:
            selected.append(min(candidates, key=lambda e: e['frame_count']))
    return list({e['id']: e for e in selected}.values())


def wait_render(page, eid):
    page.wait_for_function('(id) => active?.id === id && stages.length > 0 && !renderBusy && !document.getElementById("viewer").hidden && document.querySelector("#stage-grid canvas")', arg=eid, timeout=120000)
    assert page.locator('#loading').is_hidden(), page.locator('#loading').inner_text()


def compare_canvas(page, e, arrays):
    point = page.evaluate('coords')
    observed = page.locator('#stage-grid canvas').evaluate_all('nodes => nodes.map(c=>({id:c.dataset.stage,width:c.width,height:c.height,displayWidth:c.getBoundingClientRect().width,displayHeight:c.getBoundingClientRect().height,styleWidth:parseFloat(c.style.width),styleHeight:parseFloat(c.style.height),pixels:Array.from(c.getContext("2d").getImageData(0,0,c.width,c.height).data).filter((_,i)=>i%4===0)}))')
    stage_map = {s['id']: s for s in e['stages']}
    errors = []
    for canvas in observed:
        stage = stage_map[canvas['id']]
        idx = np.ravel_multi_index(point, [a['size'] for a in stage['axes']])
        frame = arrays[stage['id']][idx]
        assert [canvas['height'], canvas['width']] == list(frame.shape), e['id']
        values = frame[np.isfinite(frame)]
        lo = float(np.quantile(values, .005)) if stage.get('signed') and values.size else 0
        hi = float(np.quantile(values, .995)) if values.size else 0
        normalized = np.clip((frame.astype(float) - lo) / max(hi - lo, 1e-30), 0, 1)
        expected = np.where(np.isfinite(frame), np.floor(normalized * 255 + .5), 128)
        error = float(np.max(np.abs(expected.ravel() - np.asarray(canvas['pixels']))))
        span = float(np.ptp(values)) if stage.get('signed') and values.size else float(values.max()) if values.size else 0
        tolerance = max(2, int(np.ceil(255 * span / 65535 / 2 / max(hi - lo, 1e-30))) + 1)
        assert error <= tolerance, (e['id'], stage['id'], point, error, tolerance)
        ratio = canvas['displayWidth'] / canvas['displayHeight']
        expected_ratio = stage['fov'][1] / stage['fov'][0]
        assert abs(ratio - expected_ratio) < max(.01, expected_ratio * .005), (e['id'], ratio, expected_ratio)
        errors.append(dict(stage=stage['id'], index=int(idx), maximum_grayscale_error=error, tolerance=tolerance))
    return errors


def browser_verify(run, entries, all_old=False):
    checked = []
    errors = []
    selected = select_entries(entries, all_old=all_old)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True)
        context = browser.new_context(viewport={'width': 1512, 'height': 1080}, device_scale_factor=1, offline=True)
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto((run / 'index.html').as_uri())
        page.wait_for_selector('.card')
        assert page.evaluate('getComputedStyle(document.body).backgroundColor') == 'rgb(255, 255, 255)'
        page.screenshot(path=str(run / 'inventory/archive_browser_overview.png'))
        groups = page.locator('[data-filter]').evaluate_all('nodes => nodes.map(n => n.dataset.filter)')
        for group in groups:
            page.locator(f'[data-filter="{group}"]').click()
            def matches(entry):
                selected_entry = bool(entry.get('is_selected', False))
                if group == 'selected':
                    return selected_entry
                return not selected_entry and (group == 'all' or group == entry['family'] or group == entry.get('group') or (group == 'verified' and entry['family'] in ['crossed', 'hybrid']))
            expected = sum(matches(entry) for entry in entries)
            assert page.evaluate('visibleEntries().length') == expected, (group, expected)
            assert page.locator('.card').count() == min(48, expected), group
            if expected > 48:
                first = page.locator('.card').first.get_attribute('data-card')
                page.locator('#catalog-next').click()
                assert page.locator('.card').first.get_attribute('data-card') != first
                page.locator('#catalog-prev').click()
        page.locator('[data-filter="all"]').click()
        page.locator('#search').fill('independent-no-such-entry')
        assert page.locator('#empty-state').is_visible()
        page.locator('#search').fill('')
        for e in selected:
            print('browser', e['id'], e['frame_count'], flush=True)
            page.evaluate('(id) => location.hash = encodeURIComponent(id)', e['id'])
            wait_render(page, e['id'])
            if e['id'] in ['archive_913598c2354bfa24', 'bruker_4af7312db3fac3c8']:
                page.screenshot(path=str(run / ('inventory/archive_browser_' + e['id'] + '.png')))
            if any(s.get('log_display') for s in e['stages']):
                page.locator('#display-settings summary').click()
                page.locator('#show-adc').check()
                wait_render(page, e['id'])
            samples = []
            with np.load(run / e['arrays']) as arrays:
                materialized = {key: arrays[key] for key in arrays.files}
                for boundary in ['default', 'first', 'last']:
                    if boundary != 'default':
                        target = [a['size'] - 1 if boundary == 'last' else 0 for a in e['stages'][0]['axes']]
                        page.evaluate('(target) => {coords = target; filmPage = Math.floor(coords[filmAxis] / FILM_PAGE_SIZE); renderFrames();}', target)
                        wait_render(page, e['id'])
                    samples.extend(compare_canvas(page, e, materialized))
            for i, axis in enumerate(e['stages'][0]['axes']):
                if axis['size'] > 1:
                    page.locator(f'[data-axis="{i}"]').fill(str(axis['size'] // 2))
                    wait_render(page, e['id'])
                    assert page.evaluate('(i) => coords[i]', i) == axis['size'] // 2
                    page.locator('#film-axis').select_option(str(i))
                    wait_render(page, e['id'])
                    assert page.locator('.film-frame').count() <= 64
                    if axis['size'] > 64:
                        assert page.locator('#film-pages').is_visible()
                        page.locator('#film-prev' if page.locator('#film-prev').is_enabled() else '#film-next').click()
                        wait_render(page, e['id'])
                    expected_coord = int(page.locator('.film-frame').first.get_attribute('data-frame'))
                    page.locator('.film-frame').first.click()
                    wait_render(page, e['id'])
                    assert page.evaluate('(i) => coords[i]', i) == expected_coord
            page.locator('#stage-grid .canvas-wrap').first.click()
            assert page.locator('#image-modal').is_visible()
            assert page.locator('#modal-canvas').evaluate('c => c.width') == e['stages'][0]['shape'][2]
            page.locator('#modal-close').click()
            assert page.evaluate('loaded.size') <= 3
            assert page.evaluate('ArchivePixels.cacheSize()') <= 96
            page.set_viewport_size({'width': 390, 'height': 844})
            page.wait_for_timeout(250)
            wait_render(page, e['id'])
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), e['id']
            page.set_viewport_size({'width': 1512, 'height': 1080})
            page.wait_for_timeout(250)
            wait_render(page, e['id'])
            checked.append(dict(id=e['id'], family=e['family'], group=e.get('group'), frame_count=e['frame_count'], samples=samples))
            (run / 'inventory/archive_browser_progress.json').write_text(json.dumps(dict(entries=checked, javascript_errors=errors), ensure_ascii=False, indent=2) + '\n')
        assert not errors, errors
        browser.close()
    return dict(offline=True, entries_checked=len(checked), javascript_errors=errors, entries=checked,
        checks=['All format filters and catalog pagination', 'Default/first/last scientific pixels and signed/invalid values',
                'Every extra axis and paginated contact sheets', 'Native matrix and physical aspect', 'Image enlargement',
                'Search empty state', 'Cache bounds', 'Mobile without horizontal overflow'])


def source_verify(run, entries):
    results = []
    selected = select_entries(entries)
    for e in selected:
        if e['family'] != 'archive' or e['group'] not in ['nifti', 'mat']:
            continue
        detail = json.loads((run / e['metadata']).read_text())['source_details']
        if e['group'] == 'nifti':
            ni = nib.load(e['source_file'])
            arr = np.asanyarray(ni.dataobj)
            spatial = [1, 0]
            units = list(ni.header.get_xyzt_units())
        else:
            name = detail['variable']
            if any(c in name for c in '.[]'):
                continue
            arr = loadmat(e['source_file'], variable_names=[name])[name]
            spatial = detail['display_axes_zero_based']
            units = None
        extra = [i for i in range(arr.ndim) if i not in spatial and arr.shape[i] != 1]
        singleton = [i for i in range(arr.ndim) if i not in spatial and arr.shape[i] == 1]
        source = arr.transpose(*extra, *spatial, *singleton).reshape(-1, arr.shape[spatial[0]], arr.shape[spatial[1]])
        if np.iscomplexobj(source):
            source = np.abs(source)
        with np.load(run / e['arrays']) as z:
            exported = z['image']
        assert np.array_equal(source.astype(np.float32), exported, equal_nan=True), e['id']
        results.append(dict(id=e['id'], original_source_exact_float32_match=True, nifti_units=units))
    return results



def source_raw_dicom_verify(run, entries):
    """Sample original ADC lines and DICOM pixels independently of export helpers."""
    results = []
    archives = [e for e in entries if e['family'] == 'archive']
    inventory_path = run / 'inventory/all_source_files.json'
    records = json.loads(inventory_path.read_text()) if inventory_path.exists() else []
    source_paths = {r['source']: r['path'] for r in records}
    raw = [e for e in archives if e['group'] == 'raw' and e['frame_count'] > 10]
    if raw:
        import twixtools
        e = min(raw, key=lambda e: Path(e['source_file']).stat().st_size)
        metadata = json.loads((run / e['metadata']).read_text())
        counters = metadata['source_details']['frame_counters']
        sn = int(re.search(r'ADC_m(\d+)_', metadata['source_details']['variable']).group(1))
        measurements = twixtools.read_twix(e['source_file'], parse_geometry=False, parse_pmu=False, verbose=False)
        blocks = [block for block in measurements[sn]['mdb'] if block.is_image_scan()]
        names = ['Sli','Rep','Set','Ave','Eco','Par','Phs','Ida','Idb','Idc','Idd','Ide']
        with np.load(run / e['arrays']) as z:
            arr = z['image']
        sampled = []
        for index in sorted({0, e['frame_count']//2, e['frame_count']-1}):
            info = counters[index]
            desired = tuple(info[n] for n in names)
            counts = Counter()
            matches = []
            for block in blocks:
                if tuple(int(getattr(block.mdh.Counter, n)) for n in names) != desired or block.data.shape[1] != arr.shape[2]:
                    continue
                line = int(block.mdh.Counter.Lin)
                key = (line, block.data.shape[0])
                occurrence = counts[key]
                counts[key] += 1
                if occurrence != info['inferred_line_occurrence'] or line not in info['acquired_lines']:
                    continue
                expected = np.sqrt(np.sum(np.abs(block.data.astype(np.complex128))**2, axis=0)).astype(np.float32)
                assert np.array_equal(expected, arr[index, line]), (e['id'], index, line)
                matches.append(line)
            assert sorted(matches) == info['acquired_lines'], (e['id'], index)
            for line in set(range(arr.shape[1])) - set(matches):
                assert np.isnan(arr[index, line]).all(), (e['id'], index, line)
            sampled.append(dict(frame=index, checked_acquired_lines=len(matches)))
        results.append(dict(id=e['id'], raw_coil_rss_exact_source_match=True, frames=sampled))
    dicoms = [e for e in archives if e['group'] == 'dicom' and e['frame_count'] > 2]
    if dicoms:
        import pydicom
        e = min(dicoms, key=lambda e: e['frame_count'])
        metadata = json.loads((run / e['metadata']).read_text())
        frames = metadata['source_details']['frame_metadata']
        with np.load(run / e['arrays']) as z:
            arr = z['image']
        samples = []
        for index in sorted({0, e['frame_count']//2, e['frame_count']-1}):
            info = frames[index]
            path = source_paths.get(info['source'], info['source'])
            ds = pydicom.dcmread(path)
            expected = ds.pixel_array
            if expected.ndim != 2:
                continue
            expected = expected.astype(np.float32) * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0))
            assert np.array_equal(expected, arr[index], equal_nan=True), (e['id'], index)
            samples.append(index)
        assert samples, e['id']
        results.append(dict(id=e['id'], dicom_rescaled_pixels_exact_source_match=True, frames=samples))
    return results

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-old', action='store_true')
    parser.add_argument('--descriptors-only', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = read_manifest(run)
    result = dict(passed=False, manifest_summary=manifest['summary'])
    result['completeness'] = audit_descriptors(run, manifest['entries'])
    if not args.descriptors_only:
        result['browser'] = browser_verify(run, manifest['entries'], args.all_old)
        result['source_samples'] = source_verify(run, manifest['entries'])
        result['source_samples'].extend(source_raw_dicom_verify(run, manifest['entries']))
    result['passed'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['browser', 'source_samples']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
