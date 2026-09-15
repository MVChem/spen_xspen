"""Post-training latent Gaussian denoising diagnostics on the fixed 18 cases.

These are in-sample prior diagnostics, not SPEN reconstructions, validation,
checkpoint selection, or a held-out generalization assessment. Gaussian sigma
is measured in standardized latent coordinates and is not pixel-noise sigma.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from dit import LatentEDM, sample_latent
from vae_codec import FrozenVAE

# Reuse the previous experiment's exact metric implementation and local spenpy.
sys.path.insert(0, str(PROJECT.parent / 'spenpy'))
sys.path.insert(0, str(HERE.parent / 'core'))
from evaluate import metrics, unit

SIGMAS = (.1, .3, 1., 3.)
METHODS = {
    'clean': 'Clean native 192 reference',
    'vae_roundtrip': 'VAE round trip',
    'noisy_decoded': 'Decoded noisy latent',
    'denoised_decoded': 'Decoded DiT estimate',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stable_seed(seed, key):
    return int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:8], 16) % (2**31)


def verify_inputs(checkpoint, data, cases_path, training_config=None):
    """Bind the codec, full training array, and frozen report images to the EMA."""
    manifest_path = data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = sha256(manifest_path)
    if checkpoint['manifest_sha256'] != manifest_hash:
        raise ValueError('Checkpoint and training manifest differ')
    if checkpoint.get('training_mode') != 'all_data_no_holdout':
        raise ValueError('Expected an all-image prior with no holdout')
    if manifest['dataset'] != 'rodent_native192_png_all' or any(
            manifest['records'].get(split) for split in ('val', 'test')):
        raise ValueError('Expected the all-image training dataset with empty val/test')
    if checkpoint.get('img_resolution') != 192 or checkpoint.get('latent_resolution') != 48:
        raise ValueError('Expected the native 192 / latent 48 checkpoint')
    normalization = checkpoint['latent_normalization']
    if normalization.get('manifest_sha256') != manifest_hash:
        raise ValueError('Latent normalization belongs to different data')
    if normalization.get('posterior') != 'mode':
        raise ValueError('This diagnostic requires the training posterior-mode codec')
    if normalization.get('vae_sha256') != checkpoint['vae_sha256']:
        raise ValueError('Latent normalization belongs to a different VAE')
    if (normalization.get('vae_autocast') != 'bfloat16'
            or checkpoint.get('vae_autocast', 'bfloat16') != 'bfloat16'):
        raise ValueError('Latent normalization must use the training BF16 codec precision')
    training_augmentation = checkpoint.get('augmentation', (training_config or {}).get('augmentation'))
    if training_augmentation is None or normalization.get('augmentation') != training_augmentation:
        raise ValueError('Normalization augmentation differs or lacks an independent training setting')
    channels = checkpoint['model_config']['in_channels']
    for field in ('mean', 'std'):
        values = np.asarray(normalization[field], dtype=np.float64)
        if values.shape != (channels,) or not np.isfinite(values).all():
            raise ValueError(f'Invalid latent normalization {field}')
        if field == 'std' and np.any(values <= 0):
            raise ValueError('Latent standard deviations must be positive')
    if checkpoint['model_config']['input_size'] != 48:
        raise ValueError('Model configuration does not describe a 48 x 48 latent')

    vae = Path(checkpoint['vae_path']).resolve()
    vae_hashes = {str(path.relative_to(vae)): sha256(path) for path in sorted(vae.rglob('*'))
                  if path.is_file() and (path.name == 'config.json' or path.suffix in ('.safetensors', '.bin', '.ckpt'))
                  and '.cache' not in path.parts}
    if not vae_hashes or vae_hashes != checkpoint['vae_sha256']:
        raise ValueError('Local VAE config/weights differ from the training checkpoint')
    array_path = data / 'train.npy'
    array_hash = sha256(array_path)
    if array_hash != manifest['npy_sha256']['train']:
        raise ValueError('Training array differs from its manifest')
    array = np.load(array_path, mmap_mode='r', allow_pickle=False)
    training = manifest['records']['train']
    if array.dtype != np.uint16 or array.shape != (len(training), 192, 192):
        raise ValueError('Expected native uint16 [N,192,192] training images')

    saved_cases = json.loads(cases_path.read_text())
    if set(saved_cases) != {'val', 'test'} or tuple(map(len, (saved_cases['val'], saved_cases['test']))) != (8, 10):
        raise ValueError('Expected the frozen eight calibration and ten report cases')
    records, pixels, keys = [], [], set()
    fields = ('key', 'pixel_sha256', 'physical_plane_key', 'source_sha256', 'dataset', 'subject_group')
    for role in ('val', 'test'):
        for record in saved_cases[role]:
            index = record['prior_training_array_index']
            if type(index) is not int or not 0 <= index < len(training):
                raise ValueError('Invalid fixed-case index into the full training array')
            if record.get('seen_in_prior_training') is not True or record.get('no_holdout') is not True:
                raise ValueError('Fixed case must declare prior-training membership')
            if any(record.get(field) != training[index].get(field) for field in fields):
                raise ValueError(f"Fixed-case provenance differs: {record.get('key')}")
            if record['key'] in keys:
                raise ValueError('Repeated fixed diagnostic image')
            keys.add(record['key'])
            pixels.append(array[index].astype(np.float32) / 65535.)
            records.append(dict(record, diagnostic_role='in_sample_prior_diagnostic',
                                original_case_role='calibration' if role == 'val' else 'report'))
    clean = torch.from_numpy(np.stack(pixels))[:, None] * 2 - 1
    if not torch.isfinite(clean).all() or bool((clean.amax((1, 2, 3)) <= -1).any()):
        raise ValueError('Nonfinite or empty fixed diagnostic images')
    audit = dict(manifest_sha256=manifest_hash, training_npy_sha256=array_hash,
                 cases_sha256=sha256(cases_path), vae_path=str(vae), vae_sha256=vae_hashes,
                 images=len(training), diagnostic_images=len(records), provenance_verified=True,
                 normalization=normalization, all_cases_seen_in_prior_training=True)
    return clean, records, audit


def summarize(rows, records):
    """As before: average slices within animals, animals within sources, sources equally."""
    animals = defaultdict(list)
    for row, record in zip(rows, records):
        animals[(record['dataset'], record['subject_group'])].append(row)
    average = lambda values: {key: float(np.mean([row[key] for row in values])) for key in rows[0]}
    per_animal = {key: average(values) for key, values in animals.items()}
    datasets = defaultdict(list)
    for (dataset, _), values in per_animal.items():
        datasets[dataset].append(values)
    per_source = {key: average(values) for key, values in datasets.items()}
    return dict(image_mean=average(rows), subject_mean=average(list(per_animal.values())),
                source_mean=average(list(per_source.values())), per_source=per_source,
                per_subject={f'{dataset}|{animal}': values for (dataset, animal), values in per_animal.items()})


def image_metrics(images, clean, records):
    rows = metrics(images, clean)
    for row, image in zip(rows, images):
        row['outside_range_fraction'] = float(((image < -1) | (image > 1)).float().mean())
    roles = {}
    for role in ('calibration', 'report'):
        indices = [i for i, record in enumerate(records) if record['original_case_role'] == role]
        roles[role] = summarize([rows[i] for i in indices], [records[i] for i in indices])
    return dict(cases=rows, summary=summarize(rows, records), legacy_case_roles=roles)


def plot_comparisons(folder, images, results, records, sigma):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rendered = {name: unit(value) for name, value in images.items()}
    filenames = []
    for start in range(0, len(records), 6):
        count = min(6, len(records) - start)
        fig, axes = plt.subplots(4, count, figsize=(2.5 * count, 9.5), squeeze=False)
        for row, (name, values) in enumerate(rendered.items()):
            for col, index in enumerate(range(start, start + count)):
                axis = axes[row, col]
                axis.imshow(values[index], cmap='gray', vmin=0, vmax=1, interpolation='none')
                axis.set_xticks([]); axis.set_yticks([])
                if col == 0:
                    axis.set_ylabel(METHODS[name], fontsize=9)
                if name == 'clean':
                    axis.set_title(f"{index + 1}: {records[index]['dataset']}\n{records[index]['subject_group'][-24:]}", fontsize=8)
                else:
                    values_for_image = results[name]['cases'][index]
                    axis.set_title(f"{values_for_image['psnr']:.2f} dB / {values_for_image['ssim']:.3f}", fontsize=9)
        fig.suptitle(f'Latent Gaussian denoising, standardized latent sigma = {sigma:g}\n'
                     'All 18 cases seen in prior training; no SPEN observations or checkpoint selection', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .95))
        stem = f'comparison_{start // 6 + 1:02d}'
        fig.savefig(folder / f'{stem}.png', dpi=150)
        fig.savefig(folder / f'{stem}.pdf')
        plt.close(fig)
        filenames.append(f'{stem}.png')
    return filenames


def plot_samples(images, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.8))
    for index, (axis, image) in enumerate(zip(axes.flat, unit(images))):
        axis.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='none')
        axis.set_title(f'Sample {index + 1}', fontsize=9)
        axis.axis('off')
    fig.suptitle('Unconditional latent DiT samples, decoded to 192 x 192', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_report(out, config, results):
    lines = ['# Latent DiT 训练后诊断', '',
             f"采用 step {config['checkpoint_step']:,} 的 EMA；18 幅固定图像均已参与 prior 训练。", '',
             '**这是 latent 高斯去噪诊断，不是 SPEN 反演、留出测试或验证集选权重。** '
             '没有生成或使用 SPEN 测量。原先 8 幅校准/10 幅报告的身份只保留为来源记录，本诊断不调参数。', '',
             '先按训练时相同的 BF16 VAE posterior mode 编码，再用检查点的逐通道 mean/std 标准化。'
             '仅在标准化 latent 中添加独立高斯噪声，做一次 DiT 去噪，然后反标准化并解码。'
             '四个 σ 共用同一逐图固定噪声方向。这里的 σ 不能与旧像素 diffusion 的 σ 直接比较。', '',
             'PSNR/SSIM 使用此前的固定 [0,1] 指标，显示与指标统一截断，不做逐图亮度拟合。'
             '按动物内切片平均、来源内动物等权、再来源等权汇总。未截断图像、latent 和超范围比例另存。'
             'VAE 往返用于记录压缩误差，不是严格的重建性能上界。', '',
             '| latent σ | VAE 往返 PSNR / SSIM | noisy latent 解码 PSNR / SSIM | DiT 去噪后解码 PSNR / SSIM |',
             '| ---: | ---: | ---: | ---: |']
    for sigma in SIGMAS:
        result = results[f'{sigma:g}']
        values = [result['methods'][method]['summary']['source_mean']
                  for method in ('vae_roundtrip', 'noisy_decoded', 'denoised_decoded')]
        lines.append(f"| {sigma:g} | " + ' | '.join(f"{row['psnr']:.3f} / {row['ssim']:.4f}" for row in values) + ' |')
    lines += ['', '## 逐图对照', '']
    for sigma in SIGMAS:
        result = results[f'{sigma:g}']
        lines += [f'### 标准化 latent σ = {sigma:g}', '']
        for image in result['figures']:
            lines += [f"![固定病例对照]({result['directory']}/{image})", '']
    lines += ['## 无条件生成', '',
              '固定种子生成 8 张，采用 64 步 EDM Heun。无配对目标，不报告生成图 PSNR/SSIM。', '',
              '![无条件生成](unconditional.png)', '',
              '逐图指标及各来源/个体汇总见 `metrics.json`，输入核验和全部配置见 `config.json`。', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260915)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    for key in ('checkpoint', 'data', 'cases', 'out'):
        setattr(args, key, getattr(args, key).resolve())
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Use a fresh diagnostic output directory: {args.out}')
    started = time.monotonic()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    training_config_path = args.checkpoint.parent / 'config.json'
    training_config = json.loads(training_config_path.read_text()) if training_config_path.exists() else None
    clean, records, audit = verify_inputs(checkpoint, args.data, args.cases, training_config)
    codec = FrozenVAE(audit['vae_path'], device=device, autocast_dtype=torch.bfloat16,
                      encode_batch_size=args.batch_size, decode_batch_size=args.batch_size)
    net = LatentEDM(**checkpoint['model_config']).to(device)
    net.load_state_dict(checkpoint['ema'])
    net.eval().requires_grad_(False)
    if codec.latent_channels != net.img_channels or codec.downsample_factor != 4:
        raise ValueError('Checkpoint latent channels or scale differ from the VAE')
    normalization = checkpoint['latent_normalization']
    mean, std = (torch.tensor(normalization[key], dtype=torch.float32, device=device).reshape(1, -1, 1, 1)
                 for key in ('mean', 'std'))
    clean = clean.to(device)
    latent = (codec.encode(clean, sample=False) - mean) / std
    roundtrip = codec.decode(latent * std + mean, clamp=False)
    if latent.shape != (18, net.img_channels, 48, 48) or roundtrip.shape != clean.shape:
        raise ValueError('Unexpected codec latent/image shape')
    if not torch.isfinite(latent).all() or not torch.isfinite(roundtrip).all():
        raise FloatingPointError('Nonfinite clean latent or VAE reconstruction')
    seeds = [stable_seed(args.seed, record['key']) for record in records]
    noise = torch.stack([torch.randn(z.shape, dtype=torch.float32, device=device,
                                    generator=torch.Generator(device=device).manual_seed(seed))
                         for z, seed in zip(latent, seeds)])
    args.out.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(**audit, checkpoint_sha256=sha256(args.checkpoint), checkpoint_step=checkpoint['step'],
                  model_config=checkpoint['model_config'], latent_sigmas=list(SIGMAS), noise_seeds=seeds,
                  noise_coupling='same standard-normal latent noise per image, rescaled across sigmas',
                  codec_precision='FP32 parameters; BF16 autocast encode/decode; posterior mode',
                  pixel_range='2*uint16/65535-1; fixed [0,1] metric/display clipping; no intensity fitting',
                  metric_implementation=str(HERE.parent / 'core/evaluate.py'),
                  metric_source_sha256=sha256(HERE.parent / 'core/evaluate.py'),
                  source_sha256={str(path): sha256(path) for path in (Path(__file__), HERE / 'dit.py', HERE / 'vae_codec.py')},
                  diagnostic_only=True, spen_reconstruction=False, no_holdout=True,
                  checkpoint_selection=False, unconditional_samples=8, sample_steps=64,
                  sample_seed=args.seed + 1000)
    save_json(args.out / 'config.json', config)
    save_json(args.out / 'cases.json', records)
    del checkpoint
    fixed_results = {name: image_metrics(image, clean, records)
                     for name, image in (('clean', clean), ('vae_roundtrip', roundtrip))}
    results = {}
    for sigma in SIGMAS:
        folder = args.out / f'sigma_{sigma:g}'
        folder.mkdir()
        noisy = latent + sigma * noise
        denoised = torch.cat([net(batch, sigma) for batch in noisy.split(args.batch_size)])
        noisy_image = codec.decode(noisy * std + mean, clamp=False)
        denoised_image = codec.decode(denoised * std + mean, clamp=False)
        for value in (noisy, denoised, noisy_image, denoised_image):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f'Nonfinite denoising output at latent sigma={sigma:g}')
        images = dict(clean=clean, vae_roundtrip=roundtrip, noisy_decoded=noisy_image,
                      denoised_decoded=denoised_image)
        methods = dict(fixed_results, **{name: image_metrics(images[name], clean, records)
                                        for name in ('noisy_decoded', 'denoised_decoded')})
        latent_mse = {name: (values - latent).square().flatten(1).mean(1).cpu().tolist()
                      for name, values in (('noisy', noisy), ('denoised', denoised))}
        result = dict(sigma=sigma, directory=folder.name, methods=methods, latent_mse=latent_mse)
        np.savez_compressed(folder / 'arrays.npz',
                            **{name + '_raw': value.cpu().numpy() for name, value in images.items()},
                            clean_latent=latent.cpu().numpy(), noisy_latent=noisy.cpu().numpy(),
                            denoised_latent=denoised.cpu().numpy())
        result['figures'] = plot_comparisons(folder, images, methods, records, sigma)
        save_json(folder / 'metrics.json', result)
        results[f'{sigma:g}'] = result
        save_json(args.out / 'metrics.json', results)
        print(json.dumps(dict(event='latent_denoising_diagnostic', sigma=sigma,
                              **methods['denoised_decoded']['summary']['source_mean'])), flush=True)
    sampled = sample_latent(net, count=8, steps=64, seed=config['sample_seed'])
    generated = codec.decode(sampled * std + mean, clamp=False)
    if not torch.isfinite(sampled).all() or not torch.isfinite(generated).all():
        raise FloatingPointError('Nonfinite unconditional samples')
    np.savez_compressed(args.out / 'unconditional.npz', standardized_latent=sampled.cpu().numpy(),
                        decoded_raw=generated.cpu().numpy(), decoded_unit=unit(generated))
    plot_samples(generated, args.out / 'unconditional.png')
    write_report(args.out, config, results)
    save_json(args.out / 'completed.json', dict(completed=True, checkpoint_step=config['checkpoint_step'],
              diagnostic_only=True, spen_reconstruction=False, fixed_images=18, sigmas=list(SIGMAS),
              unconditional_samples=8, elapsed_seconds=time.monotonic() - started))


if __name__ == '__main__':
    main()
