"""Verify the imported Bruker sources, display chunks and representative UI."""
import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy.io import loadmat
from playwright.sync_api import sync_playwright

from build_unified import dump

HERE = Path(__file__).resolve().parent


def parse_chunk(path):
    return json.loads(path.read_text().split(']=', 1)[1].rstrip(';\n'))


def verify_arrays(run, entries):
    checked_pixels = 0
    for e in entries:
        meta = json.loads((run/e['metadata']).read_text())
        source = loadmat(meta['source_mat'], variable_names=['before', 'after'], squeeze_me=False)
        with np.load(run/e['arrays']) as images, np.load(run/e['complex_arrays']) as complex_values:
            for name in ['before', 'after']:
                native = source[name]
                if native.ndim == 2:
                    native = native[:, :, None]
                expected = np.moveaxis(native, 2, 0)
                np.testing.assert_array_equal(complex_values[name], expected)
                np.testing.assert_array_equal(images['bruker_'+name], np.abs(expected).astype(np.float32))
            with np.load(meta['source_details']['source_npz']) as old:
                for name in ['tikhonov', 'diffusion']:
                    np.testing.assert_array_equal(images['bruker_'+name], np.moveaxis(old[name], 2, 0))
            for stage in e['stages']:
                values = images[stage['id']]
                assert list(values.shape) == stage['shape']
                covered = []
                for filename in stage['chunks']:
                    chunk = parse_chunk(run/filename)
                    count = chunk['count']
                    low = np.asarray(chunk['minima'])[:, None, None]
                    high = np.asarray(chunk['maxima'])[:, None, None]
                    data = np.frombuffer(base64.b64decode(chunk['data']), '<u2').reshape(count, *values.shape[1:])
                    decoded = data.astype(float)/65535*(high-low)+low
                    original = values[chunk['start']:chunk['start']+count]
                    np.testing.assert_allclose(np.asarray(chunk['p995']), np.quantile(original, .995, axis=(1, 2)), rtol=1e-7)
                    if stage['signed']:
                        np.testing.assert_allclose(np.asarray(chunk['p005']), np.quantile(original, .005, axis=(1, 2)), rtol=1e-7)
                    assert np.all(np.abs(decoded-original) <= (high-low)/65535/2 + np.maximum(np.abs(original), 1)*3e-7)
                    covered.extend(range(chunk['start'], chunk['start']+count))
                    checked_pixels += original.size
                assert covered == list(range(values.shape[0]))
                assert len(e['default_coords']) == len(stage['axes'])
    return dict(entries=len(entries), source_mat_arrays=2*len(entries), source_npz_arrays=2*len(entries),
                all_display_chunk_pixels_checked=checked_pixels, source_array_max_error=0,
                chunk_quantization='Every pixel is within half a uint16 step plus float32 roundoff')


def verify_browser(run, entries):
    selected = {}
    selectors = [
        ('repetitions_13', lambda e: e['frame_count'] == 13),
        ('slices_11', lambda e: e['frame_count'] == 11),
        ('slices_10', lambda e: e['frame_count'] == 10),
        ('non_square_80_96', lambda e: e['stages'][0]['shape'][1:] == [80, 96]),
        ('non_square_96_128', lambda e: e['stages'][0]['shape'][1:] == [96, 128]),
        ('matrix_200', lambda e: e['stages'][0]['shape'][1:] == [200, 200]),
    ]
    for label, match in selectors:
        selected[label] = next(e for e in entries if match(e))
    errors, results = [], []
    with tempfile.TemporaryDirectory(prefix='.bruker_browser_', dir=run) as temp:
        preview = Path(temp)
        (preview/'entries').symlink_to((run/'entries').resolve(), target_is_directory=True)
        (preview/'inventory').symlink_to((run/'inventory').resolve(), target_is_directory=True)
        for name in ['index.html', 'viewer.css', 'viewer.js', 'archive_pixels.js']:
            shutil.copy2(HERE/'web'/name, preview/name)
        manifest = dict(summary=dict(distinct_scans=24), entries=list(selected.values()))
        (preview/'manifest.js').write_text('window.REVIEW_MANIFEST='+json.dumps(manifest, ensure_ascii=False)+';')
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True,
                                        args=['--no-sandbox'])
            context = browser.new_context(viewport=dict(width=1440, height=1100), offline=True)
            page = context.new_page()
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto((preview/'index.html').resolve().as_uri())
            for label, e in selected.items():
                page.evaluate('(id)=>{location.hash=id}', e['id'])
                page.wait_for_function('(id)=>active?.id===id && stages.length===4 && !renderBusy && document.getElementById("loading").hidden', arg=e['id'])
                indices = sorted({0, e['frame_count']-1})
                with np.load(run/e['arrays']) as arrays:
                    for index in indices:
                        page.evaluate('(n)=>setCoord(0,n)', index)
                        page.wait_for_function('!renderBusy')
                        for stage in e['stages']:
                            native = arrays[stage['id']][index]
                            lo = float(np.quantile(native, .005)) if stage['signed'] else 0
                            hi = float(np.quantile(native, .995))
                            expected = np.rint(255*np.clip((native.astype(float)-lo)/max(hi-lo, 1e-30), 0, 1)).astype(int).ravel()
                            samples = np.linspace(0, len(expected)-1, 101, dtype=int)
                            pixels = page.evaluate('''({key,indices})=>{const canvas=document.querySelector(`[data-stage="${key}"]`),data=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;return {width:canvas.width,height:canvas.height,pixels:indices.map(i=>data[i*4])}}''', dict(key=stage['id'], indices=samples.tolist()))
                            assert [pixels['height'], pixels['width']] == list(native.shape)
                            assert np.max(np.abs(np.asarray(pixels['pixels'])-expected[samples])) <= 1
                results.append(dict(label=label, id=e['id'], tested_frames=indices,
                                    stages=4, pixels_match_within_one_gray_level=True))
                if label == 'non_square_96_128':
                    page.screenshot(path=str(run/'inventory/bruker_browser_non_square.png'), full_page=True)
            context.close()
            browser.close()
    assert not errors, errors
    return dict(offline=True, representative_entries=results, browser_errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=HERE/'runs/unified_review_260916')
    args = parser.parse_args()
    catalog = json.loads((args.run/'inventory/bruker_entries.json').read_text())
    result = dict(arrays=verify_arrays(args.run, catalog['entries']),
                  browser=verify_browser(args.run, catalog['entries']))
    dump(args.run/'inventory/bruker_verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
