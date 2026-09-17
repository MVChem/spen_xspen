"""Compare pixel DiT to the 260916 U-Net on the exact same frozen SPEN cases."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
PROJECT = SCRIPTS.parent
DATA_ROOT = PROJECT.parent / 'data'
# Import the new loader before registering the existing physics module paths.
from pixel_model import load_pixel_dit
from png_data import parse_name, read_png

os.environ.setdefault('SPEN_REFERENCE_ROOT', str(DATA_ROOT/'prior96_0911_260916/scanner_reference'))
sys.path[:0] = [str(SCRIPTS/'prior96'), str(SCRIPTS/'core')]
import numpy as np
import torch
import spenpy
from evaluate_mouse import cases, controlled_operator, observe, subject_mean
from evaluate import metrics, unit
from evaluate_reconstruction import evaluate_real
from solvers import diffpir
from render_comparison import CASE_IDS, draw_comparison


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


@torch.no_grad()
def select_lambda(net, op, args, acceleration, noise, output):
    """Only validation images/observations enter this hyperparameter search."""
    clean, records = cases(args.inputs/'mouse_mixed', 'val', 16, 16, args.device)
    observation = observe(op, clean, noise, 9100+16+acceleration)
    trials = []
    for lamb in args.lambda_grid:
        pred, _ = diffpir(net, op, observation, steps=60, sigma_noise=noise, lamb=lamb, seed=71)
        trials.append(dict(lamb=lamb, **subject_mean(metrics(pred, clean), records)))
    chosen = max(trials, key=lambda row: row['psnr'])['lamb']
    write_json(output/f'fov16_R{acceleration}_validation_selection.json',
        dict(records=records, trials=trials, selected_lambda=chosen,
             rule='maximum subject-mean validation PSNR; no test/real images used', seed=71))
    print(json.dumps(dict(event='validation_lambda_selected', acceleration=acceleration,
                          selected=chosen, trials=trials)), flush=True)
    return chosen


def verify_png_holdout(data, run, checkpoint, inputs):
    """Verify all bytes against the training audit and every old test PNG's pixels."""
    audit = json.loads((run/'data_audit.json').read_text())
    if audit['fingerprint'] != checkpoint['data_fingerprint']:
        raise ValueError('Training audit and checkpoint differ')
    split_fingerprints = {}
    for part, split in audit['splits'].items():
        expected = {r['filename']: r for r in split['images']}
        actual = {p.name for p in (data/part).glob('*.png')}
        if actual != set(expected): raise ValueError(f'Changed {part} filenames')
        for name, row in expected.items():
            if sha256(data/part/name) != row['sha256']: raise ValueError(f'Changed PNG: {part}/{name}')
        fingerprint = hashlib.sha256(json.dumps(split['images'], sort_keys=True).encode()).hexdigest()
        if fingerprint != split['fingerprint']: raise ValueError(f'Changed {part} audit')
        split_fingerprints[part] = fingerprint
    overall = hashlib.sha256(json.dumps(split_fingerprints, sort_keys=True).encode()).hexdigest()
    if overall != checkpoint['data_fingerprint']: raise ValueError('Changed overall audit')
    original = np.load(inputs/'mouse_mixed/test.npy', mmap_mode='r')
    seen = set()
    for path in sorted((data/'test').glob('*.png')):
        record = parse_name(path)
        if record['batch'] != 'old': raise ValueError('Expected unchanged old held-out test split')
        index = record['old_index']
        if index in seen: raise ValueError('Repeated original test index')
        seen.add(index)
        np.testing.assert_array_equal(read_png(path)[0], original[index])
    if seen != set(range(len(original))): raise ValueError('Missing original test image')
    return dict(data_fingerprint=overall, original_test_images_verified=len(seen),
                all_png_sha256_verified=True, same_test_pixels=True,
                lab_holdout=False)


def simulation(net, args, checkpoint):
    output = args.out/'simulation'; output.mkdir(parents=True, exist_ok=True)
    old = args.reference/'evaluation'
    old_config = json.loads((old/'config.json').read_text())
    if old_config['checkpoint_sha256'] != sha256(args.reference/'mouse_mixed/model_ema.pt'):
        raise ValueError('Reference U-Net checkpoint changed')
    all_arrays, reports, metric_rows, summary = {}, {}, [], {}
    for acceleration, noise in ((1, .01), (2, .02)):
        key = f'fov16_R{acceleration}'
        report = json.loads((old/f'{key}_metrics.json').read_text())
        target, records = cases(args.inputs/'mouse_mixed', 'test', 16, 30, args.device)
        if records != report['records']: raise ValueError('Frozen case identities changed')
        with np.load(old/f'{key}.npz', allow_pickle=False) as frozen:
            np.testing.assert_array_equal(unit(target), frozen['target'])
            arrays = {name: frozen[name].copy() for name in ('target', 'observation', 'raw_rss', 'phase_inva', 'tikhonov')}
            arrays['unet'] = frozen['diffusion'].copy()
        y = torch.tensor(arrays['observation'], device=args.device)
        op = controlled_operator(16, acceleration, noise, args.device)
        regenerated = observe(op, target, noise, 9200+16+acceleration)
        error = float((regenerated-y).abs().max())
        if error > 2e-6: raise ValueError(f'Frozen observation mismatch: {error}')
        selected = dict(report['selected'])
        selected['pixel_dit'] = (select_lambda(net, op, args, acceleration, noise, output)
                                 if args.select_lambda else selected['V2 mouse prior'])
        start = time.monotonic()
        with torch.no_grad():
            prediction, trace = diffpir(net, op, y, steps=60, sigma_noise=noise,
                                       lamb=selected['pixel_dit'], seed=72)
        arrays['dit'] = unit(prediction)
        arrays['dit_model_range'] = prediction.cpu().numpy()
        methods = {}
        for name in ('raw_rss', 'tikhonov', 'phase_inva', 'unet', 'dit'):
            scores = metrics(torch.tensor(arrays[name], device=args.device)[:, None]*2-1, target)
            methods[name] = dict(subject_mean=subject_mean(scores, records), cases=scores)
            for index, (rec, score) in enumerate(zip(records, scores)):
                metric_rows.append(dict(condition=key, method=name, index=index, subject=rec['subject'], **score))
        reports[key] = dict(records=records, methods=methods, selected=selected, steps=60, seed=72,
            regenerated_observation_max_error=error, inference_seconds=time.monotonic()-start,
            reference_arrays_sha256=sha256(old/f'{key}.npz'), checkpoint_step=checkpoint['step'])
        summary[key] = {name: row['subject_mean'] for name, row in methods.items()}
        all_arrays[key] = arrays
        np.savez_compressed(output/f'{key}.npz', **arrays)
        write_json(output/f'{key}_metrics.json', reports[key])
        write_json(output/f'{key}_trace.json', trace)
        print(json.dumps(dict(event='simulation_complete', condition=key, **summary[key])), flush=True)
    with (output/'all_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader(); writer.writerows(metric_rows)
    images = {name: [] for name in ('target', 'raw_rss', 'tikhonov', 'phase_inva', 'unet', 'dit')}
    annotations = {name: [] for name in images if name != 'target'}
    for key in ('fov16_R1', 'fov16_R2'):
        for index in CASE_IDS:
            record = reports[key]['records'][index]
            for name in images:
                value = all_arrays[key][name][index]
                images[name].append(np.rot90(value, 2) if record['rot180'] else value)
                if name in annotations:
                    score = reports[key]['methods'][name]['cases'][index]
                    annotations[name].append(f"{score['psnr']:.2f} / {score['ssim']:.3f}")
    draw_comparison({k: np.stack(v) for k, v in images.items()}, output/'comparison',
        [f'Mouse {i+1}' for i in range(len(CASE_IDS))]*2,
        [(0, 4, 'Full PE · σ = 0.01'), (4, 8, 'Random 50% PE · σ = 0.02')],
        annotations, {'unet': 'Previous U-Net', 'dit': args.model_label})
    write_json(output/'selection.json', dict(case_indices=list(CASE_IDS), conditions=list(reports)))
    write_json(output/'summary.json', summary)
    return summary


def real(net, args, checkpoint):
    output = args.out/'real'; output.mkdir(parents=True, exist_ok=True)
    real_args = SimpleNamespace(inputs=args.inputs, scan_root=DATA_ROOT/'spen_acquired_260915/mat',
        real16_ids=[5, 13, 22, 30, 38], real24_ids=[3, 7, 11, 15, 19], reuse_real=None,
        out=output, device=args.device, checkpoint=args.checkpoint)
    evaluate_real(net, checkpoint, real_args)
    old = args.reference/'evaluation_real_expanded_reuse_260916'
    original = json.loads((old/'real.json').read_text())
    current = json.loads((output/'real.json').read_text())
    if original['checkpoint_sha256'] != sha256(args.reference/'mouse_mixed/model_ema.pt'):
        raise ValueError('Reference real-data checkpoint changed')
    if [(r['fov_mm'], r['export_index'], r['sha256']) for r in original['cases']] != [
        (r['fov_mm'], r['export_index'], r['sha256']) for r in current['cases']]:
        raise ValueError('Real comparison cases differ')
    with np.load(old/'real.npz', allow_pickle=False) as previous, np.load(output/'real.npz', allow_pickle=False) as new:
        for name in ('raw_rss', 'tikhonov', 'phase_inva'):
            np.testing.assert_allclose(new[name], previous[name], atol=2e-5, rtol=2e-5)
        images = {name: new[name].copy() for name in ('raw_rss', 'tikhonov', 'phase_inva')}
        images.update(unet=previous['diffusion'].copy(), dit=new['diffusion'].copy())
        labels = new['labels'].tolist()
    residuals = [dict(fov_mm=b['fov_mm'], export_index=b['export_index'],
                     unet=a['methods']['diffusion'], dit=b['methods']['diffusion'])
                 for a, b in zip(original['cases'], current['cases'])]
    draw_comparison(images, output/'unet_dit_comparison', labels,
                    [(0, 5, 'Real acquired SPEN · FOV 16 mm'), (5, 10, 'Real acquired SPEN · FOV 24 mm')],
                    row_labels={'unet': 'Previous U-Net', 'dit': args.model_label})
    np.savez_compressed(output/'unet_dit_comparison.npz', **images)
    write_json(output/'unet_dit_residuals.json', residuals)
    return residuals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, default=DATA_ROOT/'rodent96_expanded_260917')
    p.add_argument('--inputs', type=Path, default=DATA_ROOT/'prior96_0911_260916')
    p.add_argument('--reference', type=Path, default=PROJECT/'runs/retrain_0911_260916')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--expected-step', type=int, default=60000)
    p.add_argument('--select-lambda', action='store_true', help='Select DiT lambda on validation only')
    p.add_argument('--lambda-grid', type=float, nargs='+', default=[.1, .3, 1., 3., 10.])
    p.add_argument('--model-label', default='DiT · expanded data')
    args = p.parse_args()
    if not all(np.isfinite(x) and x > 0 for x in args.lambda_grid):
        p.error('Lambda grid must contain finite positive values')
    torch.set_num_threads(4)
    # Physics checks retain FP32 matmul behavior from the reference evaluation.
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    net, checkpoint = load_pixel_dit(args.checkpoint, args.device)
    if checkpoint['step'] != args.expected_step:
        raise ValueError(f"Expected step {args.expected_step}; got {checkpoint['step']}")
    holdout = verify_png_holdout(args.data, args.checkpoint.parent, checkpoint, args.inputs)
    config = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
        checkpoint_step=checkpoint['step'], reference=str(args.reference), holdout=holdout,
        runtime=dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
                     spenpy=spenpy.__version__, spenpy_file=spenpy.__file__),
        purpose='FINAL EVALUATION' if args.expected_step == 60000 else 'PIPELINE SMOKE TEST ONLY',
        simulation='30 held-out images / condition; frozen observations, seed 72, DiffPIR 60 steps',
        lambda_selection='per-condition validation PSNR' if args.select_lambda else 'inherited U-Net parameter',
        lambda_grid=args.lambda_grid if args.select_lambda else None,
        model_label=args.model_label,
        real='10 same MAT cases; seed 73, DiffPIR 60 steps, lambda 1, sigma 0.02; no paired GT',
        caveat='Both architecture and training data/budget differ; not an isolated architecture ablation')
    write_json(args.out/'config.json', config)
    summary = simulation(net, args, checkpoint)
    real(net, args, checkpoint)
    lines = [f"# DiT 96×96 SPEN 逆问题测试（第 {checkpoint['step']:,} 步）", '',
        '| 条件 | 原 U-Net PSNR / SSIM | 新 DiT PSNR / SSIM |', '| --- | --- | --- |']
    for condition, result in summary.items():
        a, b = result['unet'], result['dit']
        lines.append(f"| {condition} | {a['psnr']:.4f} / {a['ssim']:.6f} | {b['psnr']:.4f} / {b['ssim']:.6f} |")
    lines += ['', '每种条件 30 张原留出图；指标先按个体平均，再对个体等权平均。',
        '测试输入、原始复数观测、显示病例与旧实验一致；DiffPIR 60 步，无测试集调参。',
        ('DiT 各条件 λ 单独在验证集选择，搜索记录保存在 simulation/；真实采集固定 λ=1。'
         if args.select_lambda else 'DiT 沿用旧 U-Net 的仿真 λ；真实采集固定 λ=1。'),
        '10 例真实采集没有配对 GT，仅报告测量残差。新旧模型的架构、数据及训练预算均不同。', '',
        '[仿真对比](simulation/comparison.png) · [真实采集对比](real/unet_dit_comparison.png)', '']
    (args.out/'RESULTS.md').write_text('\n'.join(lines))
    write_json(args.out/'completed.json', dict(status='complete', checkpoint_step=checkpoint['step'],
        checkpoint_sha256=config['checkpoint_sha256'], purpose=config['purpose']))


if __name__ == '__main__': main()
