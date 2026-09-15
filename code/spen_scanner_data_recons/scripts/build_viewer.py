#!/usr/bin/env python3
"""Build an offline experiment gallery from already-rendered MRI frames.

No reconstruction or image conversion is performed. All frames are embedded
as a compact index so the viewer also works through file:// without fetch.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import quote


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = '20231207_150817_lxj_spen_231207_1_1_1_scan024'
VIEWER_MARKER = '<!-- SPEN_INTERACTIVE_VIEWER_V1 -->'
PANEL_KEYS = ('sorted_samples', 'rofft_original', 'inva_corrected',
              'tikhonov_coils', 'scanner_preview', 'inva_uncorrected',
              'tikhonov_uncorrected')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)


def natural(value):
    import re
    return tuple((0, int(x)) if x.isdigit() else (1, x.lower())
                 for x in re.split(r'(\d+)', str(value)))


def prepare_payload(run):
    """Join current case metadata with the exact existing native PNG index."""
    run = Path(run).resolve()
    indexed = {}
    for line in (run/'frame_index.jsonl').read_text().splitlines():
        if line.strip():
            frame = json.loads(line)
            if frame['frame_id'] in indexed:
                raise ValueError('Duplicate frame ID in gallery index')
            indexed[frame['frame_id']] = frame
    summary = json.loads((run/'summary.json').read_text())
    inventory = json.loads((run/'inventory.json').read_text())
    cases, used_ids, assets, case_sources = [], set(), set(), []

    def local_asset(value):
        source = Path(value)
        path = (source if source.is_absolute() else run/source).resolve()
        if not path.is_relative_to(run) or not path.is_file():
            raise ValueError(f'Missing or external gallery asset: {value}')
        relative = path.relative_to(run).as_posix()
        assets.add(relative)
        return quote(relative, safe='/'), path

    for entry in inventory['records']:
        path = run/'cases'/entry['id']/'case.json'
        case = json.loads(path.read_text())
        case_sources.append({'path':str(path.relative_to(run)), 'sha256':sha(path)})
        params = case.get('parameters', {})
        item = {'id':case['id'], 'experiment':case['experiment_name'],
                'scan_id':case['scan_id'], 'label':case.get('label', ''),
                'status':case['status'], 'reason':case.get('reason', ''),
                'raw_status':case.get('raw_status'),
                'matrix':params.get('matrix_ro_pe'),
                'declared_matrix':params.get('matrix_ro_pe'), 'fov_mm':params.get('fov_mm'),
                'counts':case.get('decoded_counts', {}),
                'warnings':case.get('quality_warnings', []), 'frames':[]}
        for original in case.get('frames', []):
            fid = original['id']
            if fid in used_ids or fid not in indexed:
                raise ValueError(f'Duplicate or unrendered source frame: {fid}')
            used_ids.add(fid)
            record = indexed[fid]
            for key in ('slice_index', 'volume_index', 'echo_index'):
                if record.get(key) != original.get(key):
                    raise ValueError(f'Stale gallery coordinate for {fid}: {key}')
            if record['case_id'] != case['id'] or record['input_status'] != original['status']:
                raise ValueError(f'Stale gallery case/status for {fid}; run render_collection.py')
            npz_url, npz_file = local_asset(original['arrays_path'])
            panels = {}
            for key in PANEL_KEYS:
                if key not in record.get('panels', {}):
                    continue
                if key in ('inva_corrected', 'tikhonov_coils') and original['status'] != 'completed':
                    raise ValueError(f'Uncorrected reconstruction is mislabeled as corrected: {fid}/{key}')
                panel = record['panels'][key]
                url, png_file = local_asset(panel['png_path'])
                if npz_file.stat().st_mtime_ns > png_file.stat().st_mtime_ns:
                    raise ValueError(f'PNG predates the source reconstruction: {fid}; run render_collection.py')
                panels[key] = {'url':url,
                               'shape':panel['display_window']['native_png_shape'],
                               'source_key':panel.get('source_key',key)}
            detail_url, _ = local_asset(record['html_path'])
            detail_url += '#' + quote(record['html_anchor'], safe='')
            item['frames'].append({'id':fid, 's':original.get('slice_index'),
                                   'v':original.get('volume_index'), 'e':original.get('echo_index'),
                                   'status':original['status'],
                                   'reason':original.get('reason') or record.get('status_note', ''),
                                   'warnings':original.get('quality_warnings', []),
                                   'npz':npz_url, 'detail_url':detail_url, 'panels':panels})
        item['frames'].sort(key=lambda f:tuple(-1 if f[k] is None else f[k] for k in ('e','v','s')))
        if item['frames']:
            # Display the native output geometry, including the 95 measured
            # PE lines. Keep the method's nominal image matrix separately.
            first_panels = item['frames'][0]['panels']
            for key in ('inva_corrected','inva_uncorrected','rofft_original','sorted_samples'):
                if key in first_panels:
                    item['matrix'] = list(reversed(first_panels[key]['shape']))
                    break
        cases.append(item)
    if used_ids != set(indexed):
        raise ValueError('The PNG index has extra frames absent from current case manifests')
    cases.sort(key=lambda c:(natural(c['experiment']),natural(c['scan_id'])))
    counts = Counter(f['status'] for c in cases for f in c['frames'])
    if len(used_ids) != summary['frame_count']:
        raise ValueError('Current summary and source frame count disagree')
    default = next((c['id'] for c in cases if c['id']==DEFAULT_CASE and c['frames']),None)
    if default is None:
        default = next((c['id'] for c in cases if c['frames']),cases[0]['id'] if cases else None)
    payload = {'version':1, 'stats':{'experiments':len({c['experiment'] for c in cases}),
               'cases':len(cases), 'frames':len(used_ids), 'completed':counts['completed'],
               'partial':counts['partial_reconstruction'], 'preview':counts['preview_only'],
               'unavailable':sum(c['raw_status']!='nonempty' for c in cases)},
               'default_case_id':default, 'archive_url':'archive.html', 'cases':cases}
    return payload, assets, case_sources


def build_viewer(run, template=None):
    run = Path(run).resolve()
    template = Path(template or PROJECT/'scripts/viewer_template.html')
    payload, assets, case_sources = prepare_payload(run)
    html = template.read_text()
    if html.count('__VIEWER_DATA__') != 1:
        raise ValueError('Viewer template must contain exactly one data placeholder')
    # Escape HTML script termination as well as JS line separator characters.
    data = json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    data = data.replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    data = data.replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    html = html.replace('__VIEWER_DATA__',data) + '\n' + VIEWER_MARKER + '\n'
    index, archive = run/'index.html', run/'archive.html'
    if index.exists() and VIEWER_MARKER not in index.read_text():
        shutil.copy2(index, archive)
    if not archive.exists():
        raise ValueError('Render the paginated gallery before building its interactive viewer')
    atomic_text(index,html)
    source = run/'source'/('viewer_'+sha(template)[:12])
    source.mkdir(parents=True,exist_ok=True)
    for original in (template,Path(__file__)):
        shutil.copy2(original,source/original.name)
    report = {'created_at':datetime.now(timezone.utc).isoformat(),
              'stats':payload['stats'], 'default_case_id':payload['default_case_id'],
              'index':'index.html', 'archive':'archive.html', 'html_bytes':index.stat().st_size,
              'frame_ids':sorted(f['id'] for c in payload['cases'] for f in c['frames']),
              'local_assets_checked':len(assets), 'all_frames_included':True,
              'file_protocol_supported':True, 'external_dependencies':[],
              'reconstruction_arrays_modified':False,
              'source':{str(p.relative_to(run)):sha(p) for p in source.iterdir() if p.is_file()},
              'input_case_manifests':case_sources,
              'input_frame_index_sha256':sha(run/'frame_index.jsonl'),
              'output_html_sha256':sha(index)}
    atomic_text(run/'viewer_manifest.json',json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args = parser.parse_args()
    report = build_viewer(args.run)
    print(json.dumps({k:report[k] for k in ('stats','html_bytes','local_assets_checked','all_frames_included')},ensure_ascii=False))


if __name__=='__main__':
    main()
