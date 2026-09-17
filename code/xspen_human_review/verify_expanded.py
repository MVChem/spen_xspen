"""Check full-scan indexing against source H5 and preserved selected reconstructions."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from build_unified import dump
from review import ro_fft, rss


def relative_error(actual, reference):
    return float(np.linalg.norm(actual-reference) / max(float(np.linalg.norm(reference)), 1e-30))


def verify(run):
    manifest = json.loads((run/'manifest.json').read_text())
    entries = manifest['entries']
    full = [e for e in entries if e['family']=='crossed' and not e.get('is_selected')]
    checked = []
    for entry in full:
        axes = entry['stages'][0]['axes']
        ns, no = [a['size'] for a in axes]
        assert entry['frame_count'] == ns * no
        error_max = 0.
        with np.load(run/entry['arrays']) as arrays, h5py.File(entry['scanner_h5']) as h5:
            assert h5['kspace'].shape[:2] == (no, ns)
            for stage in entry['stages']:
                a = arrays[stage['id']]
                assert a.shape == tuple(stage['shape']) and np.isfinite(a).all() and np.all(a>=0)
                assert len(stage['frame_available']) == ns * no
            for s, o in [(0, 0), (ns//2, no//2), (ns-1, no-1)]:
                reference = rss(ro_fft(h5['kspace'][o,s]))
                err = relative_error(arrays['raw_ro_rss'][s*no+o], reference)
                assert err < 3e-6
                error_max = max(error_max, err)
            selected_checks = []
            for selected in [e for e in entries if e.get('is_selected') and e['scan']==entry['scan']]:
                index = selected['slice']*no + selected['occurrence']
                with np.load(run/selected['complex_arrays']) as reference:
                    errors = {name:relative_error(arrays[name][index], reference[name]) for name in arrays.files}
                assert max(errors.values()) < 3e-5, (selected['id'], errors)
                selected_checks.append(dict(id=selected['id'], relative_errors=errors))
        checked.append(dict(id=entry['id'], frames=ns*no, source_ro_error_max=error_max,
            selected_checks=selected_checks, reconstruction_failures=len(entry['reconstruction_failures'])))
    assert sum(c['frames'] for c in checked) == manifest['summary']['crossed_primary_frames']
    assert sum(len(c['selected_checks']) for c in checked) == 10
    result = dict(passed=True, scans=len(checked), frames=sum(c['frames'] for c in checked),
        checked='All arrays finite, nonnegative, complete in retained H5 dimensions. Boundary/middle RO images match H5; all 10 previous reconstructions match full-scan extraction.',
        checks=checked)
    dump(run/'full_scan_source_verification.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    verify(parser.parse_args().run.resolve())
