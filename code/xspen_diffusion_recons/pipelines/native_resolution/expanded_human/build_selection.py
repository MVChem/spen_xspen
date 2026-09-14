"""Freeze extra human cases using input anatomy, before diffusion evaluation."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
SCANS=['MID51','MID613','MID615','MID106','MID108']
SLICES=[17,23,29,35]
CHECKPOINT=EXP/'runs/p4mm_mm4/model_ema.pt'


def main():
    checkpoint_meta=json.loads((CHECKPOINT.parent/'config.json').read_text())
    assert checkpoint_meta['image_shape']==[46,48]
    header={r['scan_id']:r for r in json.loads((EXP/'audit/siemens_header_manifest.json').read_text())}
    records=[]
    for mid in SCANS:
        h5=HERE/'scanner'/f'{mid}.h5'
        meta=json.loads(h5.with_suffix('.json').read_text())
        nrep,ns,ncoil,m,k=meta['shape_rep_slice_coil_pe_ro']
        assert [m,k]==[46,48] and max(SLICES)<ns
        repetitions=sorted(set([0,nrep//2,nrep-1]))
        exact=meta['fov_mm']==[184.95833333333334,193.0]
        deviation=max(abs(a/b-1) for a,b in zip(meta['fov_mm'],[184.95833333333334,193.0]))
        note=(f"Native 46x48 inference without image resizing; current raw R={meta['r_value']}, "
              f"FOV={meta['fov_mm']} mm, thickness={meta['thickness_mm']} mm. "
              f"Prior training FOV=[184.95833333333334,193.0] mm; max relative FOV difference={deviation:.6%}. "
              "Prior trained on axial/sagittal IXI views; coronal or oblique raw is view transfer, not independently retrained. "
              "Forward operator uses this scan's own R/FOV metadata; image matrix is acquired native, not upsampled resolution.")
        records.append(dict(scan=mid,h5=str(h5.resolve()),checkpoint=str(CHECKPOINT.resolve()),
                            cases=[dict(slice_index=s,repeat=r) for r in repetitions for s in SLICES],
                            geometry_note=note,header_view=header[mid]['view'],
                            qc_note='Input RO+RSS inspected by root and independent reviewer: visible brain anatomy. Four interior planes selected before diffusion; all first/middle/last occurrence cases retained.',
                            slice_selection_basis='Fixed reviewed interior slices 17,23,29,35; no output/metric-based selection',
                            no_clean_ground_truth=True))
    output=dict(scans=records,method='Native-grid human EDM EMA + DiffPIR',
                intended_cases=sum(len(r['cases']) for r in records),
                parameter_policy='60 steps; sigma_noise=.02; sigma_min=.02; sigma_max=80; lambda=1; xi=0',
                initialization_policy='Use completed IXI human native-grid EMA; do not substitute mouse weights',
                selection_phase='Frozen before expanded diffusion inference',
                remaining_candidates='Phantom and unresolved small-FOV datasets remain separately documented')
    (HERE/'selection.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(f'Frozen {len(records)} scans, {output["intended_cases"]} observations')


if __name__=='__main__':main()
