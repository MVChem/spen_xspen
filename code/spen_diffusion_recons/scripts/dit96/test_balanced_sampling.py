import json

import numpy as np
import pytest
import torch

from balanced_sampling import GROUP_MASS, balanced_weights, safe


def fixture_records(tmp_path):
    old = []
    # Different slice counts must not change per-subject or FOV probabilities.
    for subject, count in [('a', 1), ('b', 5)]:
        for view, mass in [('mouse_fov16', .6), ('mouse_fov24', .4)]:
            for _ in range(count):
                old.append(dict(dataset='mouse', subject=subject, species='mouse', view=view,
                                sample_weight=.8/2*mass/count))
    old.append(dict(dataset='rat', subject='r', species='rat', view='physical_fov35', sample_weight=.2))
    records = [dict(filename=f"old__{r['dataset']}__{r['subject']}__{i:06d}__{safe(r['view'])}.png",
                    batch='old', old_index=i, source=r['dataset'], subject=r['subject'], view=safe(r['view']))
               for i, r in enumerate(old)]
    # A second subject has more acquisitions, echoes, and slices.
    for subject, scans in [('l1', [('s1-e1', 1)]),
                           ('l2', [('s2-e1', 1), ('s2-e2', 4), ('s3-e1', 7)])]:
        for scan, count in scans:
            for i in range(count):
                records.append(dict(filename=f'lab-{subject}-{scan}-{i}',batch='lab', source='lab-RAT',
                                    subject=subject, scan=scan, view='RARE'))
    for source in ('ds005186', 'figshare-aging-28433102'):
        records.append(dict(filename=source,batch='public',source=source,subject='p',scan='scan',view='T2'))
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps({'records': {'train': old}}))
    return records, path


def test_target_mass_and_hierarchical_balance(tmp_path):
    records,path=fixture_records(tmp_path)
    weights,audit=balanced_weights(records,path)
    assert np.all(weights>0)
    assert weights.sum()==pytest.approx(1.)
    for name,mass in GROUP_MASS.items():
        assert audit['groups'][name]['probability']==pytest.approx(mass)
    mouse=audit['groups']['old_mouse']
    assert mouse['per_subject_probability']==pytest.approx({'a':.4,'b':.4})
    assert mouse['per_view_probability']==pytest.approx({'mouse-fov16':.48,'mouse-fov24':.32})
    assert audit['groups']['lab']['per_subject_probability']==pytest.approx({'l1':.04,'l2':.04})
    mass=lambda s:sum(w for r,w in zip(records,weights) if r.get('scan')==s)
    assert mass('s2-e1')==pytest.approx(.01)
    assert mass('s2-e2')==pytest.approx(.01)
    assert mass('s3-e1')==pytest.approx(.02)
    # Exercise the actual multinomial sampler rather than only inspecting metadata.
    draws=torch.multinomial(torch.tensor(weights),100000,replacement=True,
                            generator=torch.Generator().manual_seed(91)).numpy()
    old_mouse=np.array([r.get('source')=='mouse' for r in records])
    assert abs(old_mouse[draws].mean()-.8)<.006


def test_reject_wrong_identity_or_missing_old_image(tmp_path):
    records,path=fixture_records(tmp_path)
    with pytest.raises(ValueError,match='every legacy'):
        balanced_weights(records[1:],path)
    records[0]=dict(records[0],filename='mislabeled.png')
    with pytest.raises(ValueError,match='identity mismatch'):
        balanced_weights(records,path)


def test_reject_unknown_source(tmp_path):
    records,path=fixture_records(tmp_path)
    records[-1]=dict(records[-1],source='unknown')
    with pytest.raises(ValueError,match='Unrecognized new source'):
        balanced_weights(records,path)
