"""CPU verification of saved 2x results against their frozen native references."""
import collections
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reconstruct_2x import METHODS, PARAMETERS, make_operator, sha256, write_json


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    summary = json.loads((HERE / 'evaluation/summary.json').read_text())
    records = summary['records']
    assert summary['status'] == 'complete' and summary['full_selection'] and not summary['smoke']
    assert len(records) == 96 and len({(r['scan'], r['repeat'], r['slice_index']) for r in records}) == 96
    expected = json.loads((HERE / 'selection.json').read_text())['records']
    assert {(r['scan'], r['repeat'], r['slice_index']) for r in expected} == {(r['scan'], r['repeat'], r['slice_index']) for r in records}
    counts = collections.Counter(r['inference_origin'] for r in records)
    assert counts == {'verified_existing': 36, 'new_inference': 60}
    checkpoint_hashes = {}
    verified = []
    for info in records:
        assert info['status'] == 'complete' and not info['smoke'] and info['parameters'] == PARAMETERS
        assert info['sr_checkpoint_step'] == 20000
        checkpoint = info['sr_checkpoint']
        if checkpoint not in checkpoint_hashes:
            checkpoint_hashes[checkpoint] = sha256(checkpoint)
        assert checkpoint_hashes[checkpoint] == info['sr_checkpoint_sha256']
        assert sha256(info['npz']) == info['npz_sha256']
        assert sha256(info['native_reference_npz']) == info['native_reference_sha256']
        assert sha256(Path(info['native_reference_npz']).with_suffix('.json')) == info['native_reference_json_sha256']
        native_info = json.loads(Path(info['native_reference_npz']).with_suffix('.json').read_text())
        for key in ('scan', 'repeat', 'slice_index', 'native_shape', 'fov_mm', 'thickness_mm', 'magnitude_scale', 'r_value', 'beta', 'position_lps_mm'):
            assert info[key] == native_info[key], (info['npz'], key)
        m, k = info['native_shape']
        h, w = info['output_shape']
        assert (h, w) == (m*2, k*2)
        np.testing.assert_allclose(np.array(info['native_pixel_mm']) / 2, info['output_pixel_mm'], atol=1e-12, rtol=0)
        with np.load(info['npz']) as f:
            values = {key: f[key] for key in f.files}
        with np.load(info['native_reference_npz']) as f:
            native = {key: f[key] for key in f.files}
        assert all(np.isfinite(v).all() for v in values.values())
        for key, source_key in [('degraded', 'degraded'), ('native_tikhonov', 'native_tikhonov'), ('native_diffusion', 'diffusion')]:
            assert values[key].shape == (1, 1, m, k)
            assert np.array_equal(values[key], native[source_key]), (info['npz'], key)
        for key in METHODS[3:]:
            assert values[key].shape == (1, 1, h, w)
        expected_p = np.repeat(np.eye(m, dtype=np.float32), 2, axis=1) * .5
        expected_q = np.repeat(np.eye(k, dtype=np.float32), 2, axis=1) * .5
        assert np.array_equal(values['pe_projection'], expected_p)
        assert np.array_equal(values['ro_projection'], expected_q)
        projected = values['diffusion2x'].reshape(1, 1, m, 2, k, 2).mean(axis=(3, 5))
        np.testing.assert_allclose(projected, values['diffusion2x_projected_native'], atol=5e-7, rtol=0)
        interpolation_errors = []
        for key, low in [('tikhonov_bicubic2x', 'native_tikhonov'), ('native_diffusion_bicubic2x', 'native_diffusion')]:
            expected_image = F.interpolate(torch.from_numpy(values[low]), size=(h, w), mode='bicubic', align_corners=False).numpy()
            error = float(np.max(np.abs(expected_image-values[key])))
            assert error < 3e-6, (info['npz'], key, error)
            interpolation_errors.append(error)
        if info['inference_origin'] == 'verified_existing':
            prior = info['reused_result']
            assert sha256(prior['source_npz']) == prior['source_npz_sha256']
            with np.load(prior['source_npz']) as f:
                assert np.array_equal(values['diffusion2x'], f['diffusion'])
        op, y = make_operator(native, (h, w), 'cpu')
        native_op, _ = make_operator(native, (m, k), 'cpu')
        recalculated = {key: float((native_op if values[key].shape[-2:] == (m, k) else op).relative_residual(torch.from_numpy(values[key]), y)) for key in METHODS}
        max_residual_difference = max(abs(recalculated[key]-info['measurement_residual'][key]) for key in METHODS)
        assert max_residual_difference < 2e-5, (info['npz'], max_residual_difference)
        reconstructed_y = op.forward(torch.from_numpy(values['diffusion2x']))
        projected_y = native_op.forward(torch.from_numpy(values['diffusion2x_projected_native']))
        forward_error = float((reconstructed_y-projected_y).abs().max())
        assert forward_error < 2e-6
        verified.append(dict(scan=info['scan'], repeat=info['repeat'], slice_index=info['slice_index'],
                             origin=info['inference_origin'], residuals=recalculated,
                             max_residual_difference=max_residual_difference, max_forward_projection_error=forward_error,
                             max_bicubic_cpu_gpu_difference=max(interpolation_errors),
                             relative_image_change_from_bicubic=info['relative_image_change_from_bicubic'],
                             sr_fraction_below_display=float(np.mean(values['diffusion2x'] < -1)),
                             sr_fraction_above_display=float(np.mean(values['diffusion2x'] > 1))))
    groups = collections.defaultdict(list)
    for info in verified:
        groups[info['scan']].append(info)
    grouped = {scan: dict(case_count=len(rows),
                         median_residual={key:float(np.median([r['residuals'][key] for r in rows])) for key in METHODS},
                         median_relative_change_from_bicubic=float(np.median([r['relative_image_change_from_bicubic'] for r in rows])))
               for scan, rows in groups.items()}
    report = dict(status='passed', case_count=96, scan_count=8, source_summary_sha256=sha256(HERE/'evaluation/summary.json'),
                  origins=dict(counts), checkpoint_hashes=checkpoint_hashes, by_scan=grouped, cases=verified,
                  max_residual_cpu_gpu_difference=max(r['max_residual_difference'] for r in verified),
                  max_forward_projection_error=max(r['max_forward_projection_error'] for r in verified),
                  note='CPU checks; image changes and provisional-model residuals are not real-image accuracy. No real PSNR/SSIM computed.')
    write_json(HERE / 'verification.json', report)
    lines = ['# 两倍网格真实结果核验', '',
             '全部 96 个案例通过 CPU 核验：8 份扫描，每份 12 个固定案例；36 个已验证复用，60 个本次新推理。', '',
             '- 长宽分别恰好 2 倍，原生输入与原生 Tikhonov/EDM 图像逐元素保持不变；FOV、层厚、接收强度尺度及扫描编码参数保持一致。',
             '- 2 个最佳 EMA 的哈希和 20000-step 来源一致；所有结果使用正式 60-step 参数，没有混入烟雾测试。',
             '- 新输出和原生来源文件哈希一致；36 个复用的 SR 图像与旧正式输出逐元素一致。',
             '- 双三次插值独立 CPU 重算通过，P/Q 为精确邻像素平均，SR 投影回原生网格与直接计算复数观测的结果一致。',
             f"- 全部观测残差独立 CPU 重算通过，最大差 {report['max_residual_cpu_gpu_difference']:.3g}。", '',
             '| 扫描 | 原生 EDM 残差中位 | 原生 EDM 后 bicubic 残差中位 | 2× EDM 残差中位 | SR 相对 bicubic 图像变化中位 |',
             '|---|---:|---:|---:|---:|']
    for scan, g in grouped.items():
        rr = g['median_residual']
        lines.append(f"| {scan} | {rr['native_diffusion']:.2%} | {rr['native_diffusion_bicubic2x']:.2%} | {rr['diffusion2x']:.2%} | {g['median_relative_change_from_bicubic']:.2%} |")
    lines += ['', '残差仅表示对同一个固定简化观测模型的拟合，coil/物体相位由 Tikhonov 锚点估计；不能据此宣称真实图像更准确。图像变化比例是输出差异，不是精度提升。真实数据没有计算 PSNR/SSIM。', '',
              '逐例数据见 [verification.json](verification.json)。']
    (HERE / 'verification.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ('cases','by_scan')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
