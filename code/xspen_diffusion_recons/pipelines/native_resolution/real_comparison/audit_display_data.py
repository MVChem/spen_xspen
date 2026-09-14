"""Read-only CPU audit of saved real xSPEN arrays and their HDF5 measurements."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
PROJECT = EXPERIMENT.parents[1]
METHODS = ['degraded', 'native_tikhonov', 'tikhonov', 'diffusion', 'baseline128']


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a).ravel()-np.asarray(b).ravel()) /
                 max(np.linalg.norm(np.asarray(b).ravel()), 1e-20))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cases = []
    h5_meta = {}
    configs = {}
    for dataset in ['p3mm_mm3', 'p4mm_mm4']:
        root = EXPERIMENT / 'evaluation' / dataset
        cfg = json.loads((root/'config.json').read_text())
        configs[dataset] = dict(checkpoint=cfg['config']['checkpoint'], checkpoint_step=cfg['checkpoint_step'],
            checkpoint_sha256=cfg['checkpoint_sha256'], baseline_step=cfg['baseline_step'],
            baseline_checkpoint_sha256=cfg['baseline_checkpoint_sha256'], steps=cfg['config']['steps'],
            source_snapshot_hash_matches={name:digest(root/'source_snapshot'/name)==cfg['source_sha256'][name]
                for name in ['native_scanner.py','evaluate_native.py','operators.py','solvers.py']})
        for path in sorted(root.glob('MID*.npz')):
            info = json.loads(path.with_suffix('.json').read_text())
            with np.load(path) as f:
                arrays = {k:f[k] for k in f.files}
            scan = info['scan']
            with h5py.File(PROJECT/'scanner'/f'{scan}.h5') as h5:
                metadata = json.loads(h5.attrs['metadata'])
                h5_meta[scan] = {k:metadata[k] for k in [
                    'shape_rep_slice_coil_pe_ro','source','source_sha256','sequence',
                    'fov_mm','thickness_mm','slice_normal_lps','regrid','remove_os',
                    'reflection_corrected','averaged','repeat_labels','calibration_status']}
                raw_h5 = h5['kspace'][info['repeat'],info['slice_index']][None]
            original = arrays['original_measurement'].astype(np.complex128)
            y = arrays['measurement'].astype(np.complex128)
            a = arrays['encoding'].astype(np.complex128)
            f = arrays['readout'].astype(np.complex128)
            coils = arrays['coils'].astype(np.complex128)
            phase = arrays['phase_correction'].astype(np.float64)
            ro = original @ f.conj()
            corrected_ro = ro * np.exp(-1j*phase)
            rss = np.sqrt(np.sum(np.abs(ro)**2, axis=1, keepdims=True))
            alpha = info['native_regularization']
            inverse = np.linalg.solve(a.conj().T@a + alpha*np.eye(len(a)), a.conj().T)
            complex_anchor = inverse @ corrected_ro
            anchor = np.sqrt(np.sum(np.abs(complex_anchor)**2, axis=1, keepdims=True))
            expected_coils = complex_anchor / np.maximum(anchor, 1e-4)
            expected_coils *= complex(*info['gain'])
            magnitude = {k:(arrays[k].astype(np.float64)+1)/2 for k in METHODS}
            checks = dict(all_arrays_finite=all(np.isfinite(v).all() for v in arrays.values()),
                source_hash_matches=info['source_sha256']==metadata['source_sha256'],
                original_equals_h5_div_scale_relerr=rel(original,raw_h5/info['magnitude_scale']),
                corrected_measurement_reproduction_relerr=rel(y,corrected_ro@f.T),
                degraded_equals_ro_rss_relerr=rel(magnitude['degraded'],rss),
                native_tikhonov_equals_regularized_inverse_rss_relerr=rel(magnitude['native_tikhonov'],anchor),
                calibrated_coils_reproduction_relerr=rel(coils,expected_coils),
                native_anchor_p995=float(np.quantile(magnitude['native_tikhonov'],.995)),
                readout_unitarity_max_abs=float(np.max(np.abs(f.conj().T@f-np.eye(len(f))))),
                phase_even_rows_exact_zero=bool(np.all(phase[::2]==0)),
                phase_multiplier_unit_modulus_max_abs=float(np.max(np.abs(np.abs(np.exp(-1j*phase))-1))),
                pe_projection_identity=bool(np.array_equal(arrays['pe_projection'],np.eye(len(a)))),
                ro_projection_identity=bool(np.array_equal(arrays['ro_projection'],np.eye(len(f)))))
            stats = {}
            for method,m in magnitude.items():
                pred = (a @ (coils*m)) @ f.T
                residual = float(np.linalg.norm((pred-y).ravel()) / np.linalg.norm(y.ravel()))
                stats[method] = dict(model_x_shape=list(arrays[method].shape),
                    model_x_dtype=str(arrays[method].dtype), normalized_magnitude_min=float(m.min()),
                    normalized_magnitude_max=float(m.max()),
                    normalized_magnitude_percentiles={str(q):float(np.percentile(m,q)) for q in [50,95,99,99.5]},
                    display_clipped_fraction_below_zero=float(np.mean(m<0)),
                    display_clipped_fraction_above_one=float(np.mean(m>1)),
                    saved_measurement_residual=info['measurement_residual'][method],
                    recomputed_measurement_residual=residual,
                    residual_absolute_error=abs(residual-info['measurement_residual'][method]))
            # Magnitude-Tikhonov is a real linear least-squares problem. In
            # magnitude coordinates its coefficient is 4*rho since x=2*m-1.
            m = magnitude['tikhonov']
            forward = lambda z:(a@(coils*z))@f.T
            adjoint = lambda z:np.sum(coils.conj()*(a.conj().T@(z@f.conj())),axis=1,keepdims=True).real
            normal_rhs = adjoint(y)
            normal_left = adjoint(forward(m)) + 4*info['magnitude_regularization']*m
            checks['magnitude_tikhonov_normal_equation_relerr'] = rel(normal_left,normal_rhs)
            assert checks['all_arrays_finite'] and checks['source_hash_matches']
            for name,value in checks.items():
                if name.endswith('_relerr'):
                    assert value < 1e-4, (path.name,name,value)
            assert checks['pe_projection_identity'] and checks['ro_projection_identity']
            assert all(s['residual_absolute_error']<1e-5 for s in stats.values())
            cases.append(dict(case=path.stem, dataset=dataset, npz=str(path), npz_sha256=digest(path),
                json=str(path.with_suffix('.json')), repeat=info['repeat'], slice_index=info['slice_index'],
                native_shape=info['native_shape'], fov_mm=info['fov_mm'], thickness_mm=info['thickness_mm'],
                magnitude_scale=info['magnitude_scale'], gain=info['gain'],
                stored_array_schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},
                checks=checks, methods=stats))
    assert len(cases)==36
    representative_ids = ['MID112_rep00_slice037','MID114_rep00_slice026','MID27_rep00_slice026']
    representatives=[]
    for key in representative_ids:
        row=next(c for c in cases if c['case']==key)
        representatives.append(dict(case=key,dataset=row['dataset'],shape=row['native_shape'],
            fov_mm=row['fov_mm'],thickness_mm=row['thickness_mm'],window_normalized_magnitude=[0,1],
            scale_in_receiver_arbitrary_units=row['magnitude_scale'],
            residuals={k:row['methods'][k]['saved_measurement_residual'] for k in METHODS}))
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True, cases_audited=len(cases),
        method='Read-only NumPy/HDF5 CPU audit, no GPU allocation, no source data modification',
        cases_per_dataset=dict(Counter(c['dataset'] for c in cases)),scanner_h5=h5_meta,evaluation_configs=configs,
        fields={
            'original_measurement':'Complex pre-parity-correction HDF5 data divided by per-case magnitude_scale; shape [1,32,PE,RO]. HDF5 is an exported/regridded/RO-oversampling-removed/reflection-corrected measurement, not byte-level scanner ADC.',
            'measurement':'Complex phase-corrected observation in the same normalized measurement units: ((original_measurement @ conj(F))*exp(-i*phase)) @ F.T.',
            'phase_correction':'Real radians on [PE,RO-image] grid, six-coefficient polynomial applied only to odd PE rows; unit-modulus multiplier exp(-i*phase); identical across coils.',
            'degraded':'Stored x=2*m-1. m=RSS over 32 coils of original_measurement @ conj(F); PE xSPEN encoding is not inverted. No coil phase is retained in this display.',
            'native_tikhonov':'Stored x=2*m-1. m=RSS of complex per-coil (A^H A+0.01 I)^-1 A^H applied to phase-corrected RO images. This is the regularized InvA reconstruction; an additional InvA label would duplicate this same method.',
            'tikhonov':'Stored x=2*m-1. Real-magnitude multicoil least squares with fixed measured coil/object phase and gain; op.proximal(z=-1,rho=0.003). Equivalent magnitude-domain objective ||B*m-y||^2+0.012||m||^2. No positivity constraint is imposed by the solver.',
            'diffusion':'Stored x=2*m-1. Native-grid EDM magnitude prior + DiffPIR, 60 steps, fixed complex multicoil measurement consistency, sigma_noise=0.02 and lambda=1.0.',
            'baseline128':'Stored x=2*m-1. Same DiffPIR/data-consistency operator and settings, existing 128x128 prior trained at 210x210 mm. Each denoising call bilinearly upsamples to 128x128 and downsamples; physical scale/noise covariance differ from its training domain.',
            'coils':'32 complex coil/object-phase factors inferred from the same native regularized inverse; normalized by anchor RSS and multiplied by one fitted complex gain. They are not independently acquired sensitivity maps.',
            'encoding':'Spectral-norm-normalized native PE reduced encoding matrix A; provisional header-informed model, not independently verified full waveform/B0 operator.',
            'readout':'Native RO Fourier matrix F; unitary on these acquired square RO matrix dimensions.',
            'pe_projection_and_ro_projection':'Identity for p3mm_mm3 and p4mm_mm4; these displayed reconstructions use acquired native grids.'},
        scaling=dict(display_transform='m=(x+1)/2; never abs(x), never display x directly as [0,1].',
            receiver_units='Arbitrary receiver/image units, not calibrated magnetization or physical MRI units.',
            scalar_definition='magnitude_scale=quantile_0.995(RSS(native complex regularized inverse in original HDF5 units)); shared by every method for a given case.',
            restore_original_arbitrary_units='m_original_units=((x+1)/2)*magnitude_scale; no second method-specific rescaling.',
            recommended_window=[0,1],window_interpretation='Same normalized magnitude limits for all methods/crops in each case, anchored only to its native complex Tikhonov p99.5. Across cases the same normalized limits do not establish physical intensity comparability.',
            clipping='Keep NPZ arrays and numerical residuals unmodified. Clip only for display. Negative real-magnitude estimates are black; values above 1 saturate white.',
            interpolation='Use native arrays and nearest/no image interpolation; do not create apparent high resolution through smoothing.',
            raw_measurement_panel='A complex measurement or its log-magnitude panel is in a different domain from an RO+RSS image; label it separately and do not present it as an anatomical reconstruction.'),
        spatial_display=dict(reconstruction_pixel_centers='(i+0.5)*FOV_PE/PE; array extent from 0 to FOV_PE.',
            ro_rss_pixel_centers='i*FOV_PE/PE, inherited PE focus grid; extent -0.5*dy to FOV_PE-0.5*dy.',
            raw_relative_offset_pe_pixels=-0.5,
            instruction='For a physically aligned comparison, retain stored arrays and shift only the RO+RSS plot extent by minus half a PE pixel; do not resample it and call it a separate method.'),
        method_count='Five stored displays: one undecoded RO+RSS input, two distinct classical reconstructions, and two prior-based reconstructions. Native complex Tikhonov and regularized InvA are the same method.',
        representatives=representatives,
        representative_selection='Chosen after viewing all four rep0 slices of each scan, for central brain anatomy and axial/sagittal coverage; no method-score ranking. MID112 slice26 includes very bright eyes; MID27 slice18 is closer to skull base. All 36 saved cases remain available.',
        quantitative_claims=dict(clean_ground_truth=False,real_psnr_ssim='Not defined or reported.',
            allowed_measure='Relative complex measurement residual ||B*m-y||/||y|| under the same fixed nuisance model.',
            residual_limitation='Data-fit diagnostic, not image fidelity. Same-measurement coil/phase estimation favors the native anchor; lower residual alone cannot establish better reconstruction.',
            sample_independence='36 scan/repeat/slice cases are not 36 independent participants. Repeat 0 is a chronological occurrence, not verified b0 or diffusion direction.'),
        cases=cases)
    (HERE/'data_display_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# 真实 xSPEN 比较图数据审计','',
        '已在 CPU 只读核验 p3mm_mm3 的 24 例与 p4mm_mm4 的 12 例，共 36 组 NPZ/JSON，并逐例对应 scanner HDF5 切片。没有修改源数据或调用 GPU。所有观测缩放、相位变换、RO+RSS、正则逆、coil 因子及已保存 residual 的数值重算均通过。','',
        '| 存储字段 | 准确含义与建议标签 |','|---|---|',
        '| original_measurement | 32 通道复数测量 / magnitude_scale；来自已 regrid、去 RO oversampling、reflection correction 的 HDF5，尚未 parity-phase correction。 |',
        '| measurement | RO 图域施加奇偶行相位校正后变回测量域的复数数据。 |',
        '| degraded | RO Fourier 逆变换 + 32 coil RSS；PE xSPEN 编码尚未反演。是可视化输入，非原始复数据本身。 |',
        '| native_tikhonov | 逐 coil 复数正则 InvA，alpha=0.01，之后 RSS。这就是“正则 InvA”，不可用两个标签重复计作两种方法。 |',
        '| tikhonov | 固定测得 coil/object phase/gain 的实幅度 L2 重建；x 域 rho=0.003，m 幅度域系数 0.012。 |',
        '| diffusion | 当前原生网格 EDM + 60 步 DiffPIR，复数多 coil 数据一致性。 |',
        '| baseline128 | 相同 DiffPIR 与物理算子，原 210 mm / 128² 先验通过双线性上下采样调用；是迁移基线，非原生重训。 |','',
        '所有 5 个图像字段均存为 x=2m−1，显示必须使用 m=(x+1)/2。不要 abs(x)，不要逐方法 min-max 或逐方法 percentile 归一化。推荐固定 [0,1] 窗宽；同一例所有方法和局部放大共用该窗。magnitude_scale 是原始单位下 native complex Tikhonov RSS 的 99.5 percentile，所有方法已共用它。需要恢复任意接收机单位时乘回此标量；跨例不据此比较绝对物理信号。只对显示作截断，不修改 NPZ 或数值 residual。','',
        'native_tikhonov 与 tikhonov 的区别在未知量和约束：前者每个 coil 独立求复图像后 RSS；后者在固定复数 coil/object phase 和 gain 下求一个共同实幅度。两者不是同一个结果。coil 相位来自同一次测量的 native anchor，没有独立 sensitivity calibration。','',
        'RO+RSS 的 PE 行中心位于 i·dy，重建中心位于 (i+0.5)·dy；前者比后者偏半个 PE 像素。主图保留存储像素，RO+RSS 的 PE extent 使用 [−0.5dy, FOV_PE−0.5dy]，重建使用 [0,FOV_PE]；显示采用 nearest/no interpolation。原始复测量的幅度/log 幅度应独立标注测量域，不能称为完整 MRI。','',
        '推荐代表例：MID112_rep00_slice037（中央轴位脑室）、MID114_rep00_slice026（中央矢状位）、MID27_rep00_slice026（轴位脑室）。依据解剖覆盖选择，不按方法指标筛选。MID112 slice26 的明亮双眼会使同一 p99.5 窗下脑实质偏暗；MID27 slice18 接近颅底。','',
        '真实数据没有干净 GT，不计算或展示真实 PSNR/SSIM。可列相同固定算子下的相对复测量 residual，但它只反映该模型的数据拟合；尤其 native anchor 同时参与 coil/phase 拟合，residual 小不能证明重建更真实。36 例不是 36 位独立受试者；rep0 仅为时间顺序 occurrence，不标为已核实 b0。','',
        '完整逐例数值、缩放、尺寸、哈希、源快照核验和窗宽截断比例见 `data_display_audit.json`。']
    (HERE/'data_display_audit.md').write_text('\n'.join(lines)+'\n')
    summary=dict(passed=True,cases=len(cases),
        max_checks={name:max(c['checks'][name] for c in cases) for name in cases[0]['checks'] if name.endswith('_relerr')},
        max_residual_absolute_error=max(s['residual_absolute_error'] for c in cases for s in c['methods'].values()),
        representatives=representative_ids,output=str(HERE))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
