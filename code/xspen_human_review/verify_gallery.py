"""Exercise the actual offline gallery and compare rendered pixels to saved arrays."""
import argparse
import json
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright


def verify(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    entries = manifest['entries']
    errors, checked = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True)
        context = browser.new_context(viewport={'width': 1512, 'height': 1080}, device_scale_factor=1, offline=True)
        page = context.new_page()
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto((run / 'index.html').as_uri())
        page.wait_for_selector('.card')
        assert page.locator('.card').count() == manifest['summary']['distinct_scans']
        assert page.evaluate('getComputedStyle(document.body).backgroundColor') == 'rgb(255, 255, 255)'
        page.screenshot(path=str(run / 'browser_overview.png'), full_page=True)
        page.locator('[data-filter="hybrid"]').click()
        assert page.locator('.card').count() == 16
        page.locator('[data-filter="crossed"]').click()
        assert page.locator('.card').count() == manifest['summary']['crossed_scans']
        page.locator('[data-filter="selected"]').click()
        assert page.locator('.card').count() == 10
        page.locator('[data-filter="all"]').click()
        page.locator('#search').fill('MID613')
        assert page.locator('.card').count() == 1
        page.locator('#search').fill('unmatched-query')
        assert page.locator('#empty-state').is_visible()
        page.locator('#search').fill('')
        for entry in entries:
            page.evaluate('(id) => location.hash = encodeURIComponent(id)', entry['id'])
            page.wait_for_function('(id) => active?.id === id && stages.length > 0 && !document.getElementById("viewer").hidden', arg=entry['id'])
            assert page.locator('#loading').is_hidden()
            has_adc = any(s.get('log_display') for s in entry['stages'])
            if has_adc:
                page.locator('#display-settings summary').click()
                page.locator('#show-adc').check()
            with np.load(run / entry['arrays']) as arrays:
                for boundary in ['default', 'first', 'last']:
                    if boundary != 'default':
                        for i, axis in enumerate(entry['stages'][0]['axes']):
                            if axis['size'] > 1:
                                page.locator(f'[data-axis="{i}"]').fill(str(axis['size'] - 1 if boundary == 'last' else 0))
                    point = page.evaluate('coords')
                    observed = page.locator('#stage-grid canvas').evaluate_all('nodes => nodes.map(c=>({id:c.dataset.stage,width:c.width,height:c.height,displayWidth:c.getBoundingClientRect().width,displayHeight:c.getBoundingClientRect().height,pixels:Array.from(c.getContext("2d").getImageData(0,0,c.width,c.height).data).filter((_,i)=>i%4===0)}))')
                    assert len(observed) == len(entry['stages'])
                    for canvas, stage in zip(observed, entry['stages']):
                        assert canvas['id'] == stage['id']
                        idx = np.ravel_multi_index(point, [a['size'] for a in stage['axes']])
                        frame = arrays[stage['id']][idx]
                        assert [canvas['height'], canvas['width']] == list(frame.shape)
                        expected = np.rint(np.clip(frame / max(float(np.quantile(frame, .995)), 1e-30), 0, 1) * 255)
                        error = float(np.max(np.abs(expected.ravel() - np.array(canvas['pixels']))))
                        assert error <= 2, (entry['id'], stage['id'], point, error)
                        ratio = canvas['displayWidth'] / canvas['displayHeight']
                        assert abs(ratio - stage['fov'][1] / stage['fov'][0]) < .005, (entry['id'], ratio)
            if entry['id'] == 'MID253':
                page.locator('[data-axis="0"]').fill('19')
                page.locator('[data-axis="1"]').fill('0')
                page.locator('#film-axis').select_option('1')
                assert page.locator('.film-frame').count() == 4
                page.locator('.film-frame').nth(2).click()
                assert page.evaluate('coords[1]') == 2
                page.locator('#display-settings summary').click()
                page.locator('#window-mode').select_option('shared')
                limits = page.locator('.stage-window').all_text_contents()
                assert limits[0].split('窗 ')[1] == limits[1].split('窗 ')[1]
                page.locator('#window-width').fill('0.6')
                page.locator('#gamma').fill('1.4')
                page.locator('#reset-display').click()
                page.locator('[data-axis="1"]').fill('0')
                page.locator('#display-settings summary').click()
                page.screenshot(path=str(run / 'browser_mid253.png'), full_page=True)
            if entry['id'] == 'MID1496':
                assert page.locator('.film-frame').count() == 6
                assert '实际仅记录 6 层' in page.locator('#notice').inner_text()
            if entry['id'] == 'MID54':
                assert len(page.locator('[data-axis]').all()) == 3
                page.screenshot(path=str(run / 'browser_mid54.png'), full_page=True)
            if entry['id'] == entries[0]['id']:
                for i, value in enumerate(entry['default_coords']):
                    if entry['stages'][0]['axes'][i]['size'] > 1:
                        page.locator(f'[data-axis="{i}"]').fill(str(value))
                page.locator('#stage-grid .canvas-wrap').first.click()
                assert page.locator('#image-modal').is_visible()
                assert page.locator('#modal-canvas').evaluate('c => c.width') == entry['stages'][0]['shape'][2]
                page.locator('#modal-close').click()
                page.screenshot(path=str(run / 'browser_crossed.png'), full_page=True)
            assert page.evaluate('loaded.size') <= 3
            for key in ['payload', 'arrays', 'metadata', 'thumbnail', 'complex_arrays', 'source_metadata']:
                if key in entry:
                    assert (run / entry[key]).is_file()
            checked.append({'id': entry['id'], 'stages': len(entry['stages']), 'boundary_pixels_match_npz': True})
        page.evaluate('location.hash="MID112"')
        page.wait_for_function('active?.id === "MID112" && stages.length > 0')
        page.locator('#bookmarks a').first.click()
        page.wait_for_function('active?.is_selected && stages.length > 0')
        assert page.locator('#full-scan-link').is_visible()
        page.locator('#full-scan-link').click()
        page.wait_for_function('active?.id === "MID112" && stages.length > 0')
        page.set_viewport_size({'width': 390, 'height': 844})
        page.evaluate('location.hash="MID253"')
        page.wait_for_function('active?.id === "MID253" && stages.length > 0 && !document.getElementById("viewer").hidden')
        page.wait_for_timeout(200)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        page.screenshot(path=str(run / 'browser_mobile.png'), full_page=True)
        assert not errors, errors
        browser.close()
    result = dict(passed=True, offline=True, entries_checked=len(checked), javascript_errors=errors,
        checks=[f'All {len(entries)} entries open', 'First/default and last frame pixels match NPZ within 2 grayscale levels',
                'Native canvas shape and physical FOV aspect', 'Group filtering', 'Slice/extra-axis controls',
                'Contact sheet navigation', 'Shared window / gamma / reset', 'Incomplete acquisition marked',
                'Search and empty results', 'Enlarged image', 'Selected-case / full-scan navigation',
                'At most three decoded scans cached', 'Local artifact links', 'Mobile layout without horizontal overflow'], entries=checked)
    (run / 'browser_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'entries'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve())
