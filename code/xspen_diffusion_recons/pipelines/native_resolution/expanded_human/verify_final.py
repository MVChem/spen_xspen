"""Independent CPU artifact audit; reads frozen inference outputs without changes."""
import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from native_scanner import load_case

METHODS = ['degraded','native_tikhonov','tikhonov','phasemap_inva','baseline128','diffusion']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def compare(actual, expected, rtol=3e-5, atol=2e-6):
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    return dict(max_abs=float(np.max(np.abs(actual-expected))),
                relative_l2=float(np.linalg.norm((actual-expected).ravel())/max(np.linalg.norm(expected.ravel()),1e-20)),
                rtol=rtol, atol=atol)


def arr(x):
    return x.detach().cpu().resolve_conj().numpy()


def finite_json(value):
    if isinstance(value, dict):
        assert 'psnr' not in {k.lower() for k in value} and 'ssim' not in {k.lower() for k in value}, 'Unexpected real GT image metric'
        return all(finite_json(x) for x in value.values())
    if isinstance(value, list):
        return all(finite_json(x) for x in value)
    return not isinstance(value, float) or bool(np.isfinite(value))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,default=HERE/'evaluation')
    parser.add_argument('--selection',type=Path,default=HERE/'selection.json')
    parser.add_argument('--out',type=Path,default=HERE/'final_verification.json')
    args=parser.parse_args()
    torch.set_num_threads(2)
    summary=json.loads((args.evaluation/'summary.json').read_text())
    completed=json.loads((args.evaluation/'completed.json').read_text())
    selection=json.loads(args.selection.read_text())
    config=json.loads((args.evaluation/'config.json').read_text())
    assert summary['status']=='complete' and completed['status']=='complete'
    assert summary['case_count']==completed['case_count']==60
    assert not summary['smoke'] and not completed['smoke']
    assert config['selection_sha256']==digest(args.selection)
    parameters=config['parameters']
    assert parameters['steps']==60 and parameters['sigma_noise']==.02 and parameters['lamb']==1 and parameters['xi']==0
    assert not parameters['smoke'] and parameters['native_image_resize'] is False
    assert config['selected_scans']==selection['scans']
    assert finite_json(summary)
    current_source_hash={path:digest(path) for path in config['source_sha256']}
    assert current_source_hash==config['source_sha256']
    baseline_hash=digest(config['baseline_checkpoint'])
    assert baseline_hash==config['baseline_checkpoint_sha256']
    base_ckpt=torch.load(config['baseline_checkpoint'],map_location='cpu',weights_only=False)
    assert base_ckpt['step']==config['baseline_step'] and 'ema' in base_ckpt
    del base_ckpt
    model_records={};scans=[];case_records=[];expected_ids=set();actual_ids=set()
    for entry in selection['scans']:
        scan=entry['scan'];scanner_path=Path(entry['h5']);checkpoint_path=Path(entry['checkpoint'])
        scan_summary=json.loads((args.evaluation/scan/'summary.json').read_text())
        assert scan_summary['status']=='complete' and scan_summary['case_count']==len(entry['cases'])==12
        assert scan_summary['parameters']==parameters
        assert len(scan_summary['pages'])==3 and all(Path(p).is_file() and Path(p).stat().st_size>1000 for p in scan_summary['pages'])
        model_path=str(checkpoint_path)
        if model_path not in model_records:
            ckpt=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            assert ckpt['step']==20000 and tuple(ckpt['model_config']['image_shape'])==(46,48) and 'ema' in ckpt
            model_records[model_path]=dict(sha256=digest(checkpoint_path),step=ckpt['step'],shape=ckpt['model_config']['image_shape'],weights='ema',manifest_sha256=ckpt.get('manifest_sha256'))
            del ckpt
        model=model_records[model_path]
        h5hash=digest(scanner_path)
        with h5py.File(scanner_path) as h5:
            meta=json.loads(h5.attrs['metadata']);shape=h5['kspace'].shape
            assert tuple(shape[-2:])==(46,48)
            source_path=Path(meta['source']);raw_hash=digest(source_path)
            assert raw_hash==meta['source_sha256']
            cpu_case=entry['cases'][0]
            for chosen in entry['cases']:
                sl,rep=chosen['slice_index'],chosen['repeat'];name=f'{scan}_rep{rep:02d}_slice{sl:03d}'
                assert name not in expected_ids
                expected_ids.add(name)
                jp=args.evaluation/scan/(name+'.json');npz=args.evaluation/scan/(name+'.npz')
                info=json.loads(jp.read_text());prov=info['provenance']
                assert info['status']=='complete' and not info['smoke'] and info['steps']==60
                assert info['scan']==scan and info['slice_index']==sl and info['repeat']==rep
                assert info['native_shape']==info['output_shape']==[46,48]
                assert prov['parameters']==parameters and prov['checkpoint_weights']=='ema' and prov['checkpoint_step']==20000
                assert prov['checkpoint_sha256']==model['sha256'] and prov['checkpoint_manifest_sha256']==model['manifest_sha256']
                assert prov['scanner_h5_sha256']==h5hash and prov['source_raw_sha256']==raw_hash
                assert prov['baseline_checkpoint_sha256']==baseline_hash and prov['source_code_sha256']==current_source_hash
                assert prov['source_metadata']==meta and prov['geometry_note']==entry.get('geometry_note','') and prov['qc_note']==entry.get('qc_note','')
                assert prov['case_selection']==entry['cases']
                assert info['computation_sha256']==canonical(dict(provenance=prov,case=chosen))
                assert info['npz_sha256']==digest(npz) and finite_json(info)
                for method in ('diffusion','baseline128'):
                    assert info['traces'][method][-1]['step']==59
                with np.load(npz) as values:
                    shapes={key:list(value.shape) for key,value in values.items()}
                    assert all(np.isfinite(value).all() for value in values.values())
                    for method in METHODS:
                        assert values[method].shape==(1,1,46,48)
                    checks=dict(raw_receiver=compare(values['raw_receiver'],h5['kspace'][rep,sl],rtol=0,atol=0),
                                normalized_original=compare(values['original_measurement'][0],values['raw_receiver']/info['magnitude_scale']),
                                phase_modelx=compare(values['phase_modelx'][0,0],2*values['phase_magnitude']-1),
                                phase_alias=compare(values['phase_modelx'],values['phasemap_inva'],rtol=0,atol=0),
                                native_pe_projection=compare(values['pe_projection'],np.eye(46),rtol=0,atol=0),
                                native_ro_projection=compare(values['ro_projection'],np.eye(48),rtol=0,atol=0))
                    reconstructed_ro=values['original_measurement']@values['readout'].conj()
                    reconstructed_y=(reconstructed_ro*np.exp(-1j*values['phase_correction']))@values['readout'].T
                    checks['corrected_observation']=compare(values['measurement'],reconstructed_y,rtol=2e-4,atol=5e-6)
                    raw_rss=np.sqrt(np.sum(np.abs(reconstructed_ro)**2,axis=1,keepdims=True))
                    checks['raw_ro_rss']=compare(values['degraded'],raw_rss*2-1,rtol=2e-4,atol=5e-6)
                    record=dict(case=name,json=str(jp.resolve()),npz_sha256=info['npz_sha256'],array_count=len(values.files),checks=checks)
                    if chosen==cpu_case:
                        op,y,l2,anchor,rss,cpu_info=load_case(scanner_path,sl,rep,'cpu',image_shape=None)
                        cpu_checks={key:compare(values[key],arr(value),rtol=8e-4,atol=3e-4) for key,value in
                                    [('native_tikhonov',anchor),('tikhonov',l2),('degraded',rss*2-1),('measurement',y)]}
                        cpu_checks['scale_relative_difference']=abs(cpu_info['magnitude_scale']-info['magnitude_scale'])/info['magnitude_scale']
                        record['cpu_recompute']=cpu_checks
                    case_records.append(record)
        scans.append(dict(scan=scan,case_count=12,shape=list(shape),scanner_h5_sha256=h5hash,source_raw_sha256=raw_hash,
                          source_raw_hash_recomputed=True,cpu_recompute_case=cpu_case,pages=scan_summary['pages']))
        print(json.dumps(dict(event='scan_audit_complete',scan=scan,cases=12,cpu_recompute=True)),flush=True)
    for jp in args.evaluation.glob('MID*/*_rep*_slice*.json'):
        assert jp.stem not in actual_ids
        actual_ids.add(jp.stem)
    assert len(actual_ids)==60 and actual_ids==expected_ids
    assert len(list(args.evaluation.glob('MID*/*_rep*_slice*.npz')))==60
    report=dict(status='pass',date=datetime.datetime.now().astimezone().isoformat(),device='cpu',
                evaluation=str(args.evaluation.resolve()),selection_sha256=digest(args.selection),
                case_count=60,unique_case_count=60,scan_count=5,model_checkpoints=model_records,
                formal_parameters=parameters,source_code_hashes_verified=True,baseline_hash_verified=True,
                no_real_gt_metrics=True,no_native_resampling=True,scans=scans,cases=case_records,
                limitations='This audit checks artifact integrity, fixed-method reproducibility and provenance. It does not validate anatomical accuracy, independent coil/waveform calibration, achieved resolution or independent subject counts.')
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    rows=['# 正式扩展人脑 xSPEN 输出核验','',
          'CPU 独立核验通过：5 个扫描、60 个唯一 case，均为原生 46×48；没有修改推理源码或结果，也没有使用 GPU。','',
          '- 60 份 NPZ 全部数组有限；六幅方法图均为 `[1,1,46,48]`。PhaseMap magnitude/modelx 一致，native P/Q 为精确单位矩阵，无网格重采样。',
          '- 保存的 raw receiver 与对应 H5 帧逐元素完全一致；归一化原观测、奇偶校正后观测和 RO+RSS 可由存储算子复现。',
          '- 实际重新计算全部 5 份 H5 与源 `.dat` 的 SHA256，匹配导出和 case 记录；模型、baseline、冻结源码和每例 NPZ SHA256 匹配。',
          '- 所有 case 使用 step 20000 EMA、正式 60 步、sigma_noise=0.02、lambda=1、xi=0。DiffPIR trace 最后一步为 59；无 smoke、无 native resize。',
          '- 固定抽取每扫描 selection 中第一例，在 CPU 重算 load_case 的 complex Tikhonov、幅度 L2、RO+RSS 和观测，均在 GPU/CPU 浮点容差内一致。','',
          '| 扫描 | case 数 | CPU 抽验 slice / occurrence | anchor 最大绝对差 | L2 最大绝对差 | RO+RSS 最大绝对差 |',
          '|---|---:|---|---:|---:|---:|']
    for scan in scans:
        cr=next(x for x in case_records if x['case'].startswith(scan['scan']+'_') and 'cpu_recompute' in x)
        t=cr['cpu_recompute'];ch=scan['cpu_recompute_case']
        rows.append(f"| {scan['scan']} | 12 | {ch['slice_index']} / {ch['repeat']} | {t['native_tikhonov']['max_abs']:.3g} | {t['tikhonov']['max_abs']:.3g} | {t['degraded']['max_abs']:.3g} |")
    rows+=['','各扫描 3 页、每页 4 层的六列 PNG 均存在。上述核验不构成无伪影或优于所有基线的结论；真实图没有干净 GT，因此未计算 PSNR/SSIM。残差仅反映当前固定 nuisance 模型的数据拟合，不证明解剖真实性或达到更高分辨率。','',f'详细结果：[final_verification.json]({args.out.name})。']
    args.out.with_suffix('.md').write_text('\n'.join(rows)+'\n')
    print(json.dumps(dict(event='final_audit_complete',status='pass',cases=60,out=str(args.out))),flush=True)


if __name__=='__main__':
    main()
