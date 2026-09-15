"""Evaluate a frozen f4 VAE on fixed simulation references and balanced MRI data.

Selection is by fixed subject hashes, before reconstruction. The same VAE is
used for every image; no hyperparameters are fit on simulation report cases.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity
import torch

from vae_codec import FrozenVAE


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_records(manifest, cases, per_source, seed):
    records = manifest["records"]["train"]
    by_key = {row["key"]: (i, row) for i, row in enumerate(records)}
    selected = {}
    for split, rows in cases.items():
        for case in rows:
            index, row = by_key[case["key"]]
            if row["png_sha256"] != case["png_sha256"]:
                raise ValueError("Simulation reference and training PNG identity differ")
            selected.setdefault(index, {"index": index, "record": row, "groups": []})
            selected[index]["groups"].append("simulation_" + split)
    sources = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(records):
        sources[row["dataset"]][row["subject_group"]].append(i)
    stable_hash = lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    for source, subjects in sorted(sources.items()):
        subject_order = sorted(subjects, key=stable_hash)
        ordered = {s: sorted(subjects[s], key=lambda i: stable_hash(records[i]["key"]))
                   for s in subject_order}
        choices = []
        depth = 0
        while len(choices) < min(per_source, sum(map(len, ordered.values()))):
            for subject in subject_order:
                if depth < len(ordered[subject]):
                    choices.append(ordered[subject][depth])
                    if len(choices) == per_source:
                        break
            depth += 1
        for index in choices:
            selected.setdefault(index, {"index": index, "record": records[index], "groups": []})
            selected[index]["groups"].append("source_balanced")
    return list(selected.values())


def metrics(reference, reconstruction):
    error = (reference.astype(np.float64) - reconstruction.astype(np.float64)) ** 2
    # Match scripts/core/evaluate.py exactly, including population covariance.
    p, g = reconstruction, reference
    mu_p = gaussian_filter(p, 1.5, truncate=3.5)
    mu_g = gaussian_filter(g, 1.5, truncate=3.5)
    var_p = gaussian_filter(p*p, 1.5, truncate=3.5) - mu_p**2
    var_g = gaussian_filter(g*g, 1.5, truncate=3.5) - mu_g**2
    cov = gaussian_filter(p*g, 1.5, truncate=3.5) - mu_p*mu_g
    ssim_map = ((2*mu_p*mu_g+.01**2)*(2*cov+.03**2))/((mu_p**2+mu_g**2+.01**2)*(var_p+var_g+.03**2))
    ssim = ssim_map[5:-5, 5:-5].mean()
    mask = reference > 0.05
    interior = mask.copy()
    interior[:5] = interior[-5:] = False
    interior[:, :5] = interior[:, -5:] = False
    return dict(psnr=float(-10 * np.log10(max(error.mean(), 1e-15))),
                ssim=float(ssim),
                ssim_skimage_default=float(structural_similarity(reference, reconstruction, data_range=1.0)),
                foreground_psnr=float(-10 * np.log10(max(error[mask].mean(), 1e-15))),
                foreground_ssim=float(ssim_map[interior].mean()),
                foreground_fraction=float(mask.mean()),
                mae=float(np.sqrt(error).mean()),
                maximum_absolute_error=float(np.sqrt(error).max()))


def summarize(rows):
    keys = ("psnr", "ssim", "foreground_psnr", "foreground_ssim")
    summary = {"count": len(rows)}
    for key in keys:
        values = np.array([row[key] for row in rows])
        summary[key] = dict(mean=float(values.mean()), median=float(np.median(values)),
                            minimum=float(values.min()), p05=float(np.quantile(values, .05)),
                            maximum=float(values.max()), std=float(values.std()))
    return summary


def make_preview(path, ids, rows, references, reconstructions):
    fig, axes = plt.subplots(len(ids), 3, figsize=(9, 2.75 * len(ids)), squeeze=False)
    for rr, ii in enumerate(ids):
        row = rows[ii]
        axes[rr, 0].imshow(references[ii], cmap="gray", vmin=0, vmax=1)
        axes[rr, 0].set_title(f"{row['dataset']} / {row['index']}\nReference", fontsize=9)
        axes[rr, 1].imshow(reconstructions[ii], cmap="gray", vmin=0, vmax=1)
        axes[rr, 1].set_title(f"VAE: {row['psnr']:.2f} dB / {row['ssim']:.4f}", fontsize=9)
        im = axes[rr, 2].imshow(np.abs(reconstructions[ii]-references[ii]), cmap="magma", vmin=0, vmax=.1)
        axes[rr, 2].set_title(f"|error| [0, 0.1]; FG {row['foreground_psnr']:.2f} dB", fontsize=9)
        for ax in axes[rr]:
            ax.axis("off")
    fig.colorbar(im, ax=axes[:, 2], fraction=.02, pad=.02)
    fig.subplots_adjust(top=.98, bottom=.02, left=.02, right=.91, hspace=.25, wspace=.05)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=.145)
    parser.add_argument("--model-label", default="stabilityai/stable-diffusion-x4-upscaler")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    manifest_path = args.data / "manifest.json"
    cases_path = args.simulation / "cases.json"
    manifest = json.loads(manifest_path.read_text())
    selected = select_records(manifest, json.loads(cases_path.read_text()), args.per_source, args.seed)
    image_data = np.load(args.data / "train.npy", mmap_mode="r")
    compact = [dict(index=s["index"], key=s["record"]["key"], groups=s["groups"],
                    dataset=s["record"]["dataset"], subject=s["record"]["subject_group"],
                    png_sha256=s["record"]["png_sha256"]) for s in selected]
    protocol = dict(selection="all 18 fixed previous simulation references, plus fixed-hash subject-round-robin 32/source",
                    seed=args.seed, per_source=args.per_source, image_shape=[192, 192],
                    normalization="uint16 / 65535; VAE input 2*x01-1; no per-image scale adjustment",
                    codec="frozen pretrained RGB VAE; grayscale repeated 3x; deterministic posterior mode; RGB output mean; clip [0,1]",
                    image_quality="Legacy core/evaluate.py SSIM: Gaussian sigma1.5 truncate3.5 population covariance crop5; full-image PSNR on [0,1]; skimage default SSIM provided separately",
                    foreground="reference > 0.05; PSNR over masked pixels; mean legacy SSIM map excluding the outer 5px",
                    foreground_is_brain_segmentation=False,
                    baseline="clean simulated ground-truth images; no SPEN inverse reconstruction or prior model involved",
                    prior_holdout=False,
                    note="Previous simulation references were in the prior's full-data training; this tests only the frozen external VAE.",
                    screening_thresholds=dict(each_source_psnr_mean_min=35, each_source_ssim_mean_min=.95),
                    screening_note="Practical engineering screen fixed before VAE evaluation; not proof that downstream reconstruction improves.",
                    hashes={"data_manifest": sha256(manifest_path), "simulation_cases": sha256(cases_path),
                            "vae_codec.py": sha256(Path(__file__).with_name("vae_codec.py")),
                            "evaluate_vae.py": sha256(__file__)},
                    selected=compact)
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    codec = FrozenVAE(args.vae, args.device)
    rows, references, reconstructions, latent_sum, latent_sq = [], [], [], None, None
    latent_count = 0
    started = time.time()
    for n, item in enumerate(selected):
        ref = image_data[item["index"]].astype(np.float32) / 65535
        x = torch.from_numpy(ref)[None, None].to(args.device) * 2 - 1
        with torch.no_grad():
            z = codec.encode(x)
            raw = codec.decode(z, clamp=False)
            pred = ((raw.clamp(-1, 1) + 1) / 2)[0, 0].cpu().numpy()
        row = compact[n] | metrics(ref, pred)
        row["raw_psnr"] = metrics(ref, ((raw + 1) / 2)[0, 0].cpu().numpy())["psnr"]
        row["clipped_fraction"] = float(((raw < -1) | (raw > 1)).float().mean())
        values = z.double().sum((0, 2, 3)).cpu().numpy()
        squares = z.double().square().sum((0, 2, 3)).cpu().numpy()
        latent_sum = values if latent_sum is None else latent_sum + values
        latent_sq = squares if latent_sq is None else latent_sq + squares
        latent_count += z.shape[0] * z.shape[2] * z.shape[3]
        rows.append(row); references.append(ref); reconstructions.append(pred)
        if n == 0 or (n + 1) % 16 == 0 or n + 1 == len(selected):
            print(json.dumps(dict(evaluated=n+1, total=len(selected), seconds=time.time()-started,
                                  last_psnr=row["psnr"], last_ssim=row["ssim"])), flush=True)
    groups = {group: summarize([r for r in rows if group in r["groups"]])
              for group in ("simulation_val", "simulation_test", "source_balanced")}
    by_source = {source: summarize([r for r in rows if r["dataset"] == source and "source_balanced" in r["groups"]])
                 for source in sorted(set(r["dataset"] for r in rows))}
    source_equal = {k: float(np.mean([v[k]["mean"] for v in by_source.values()]))
                    for k in ("psnr", "ssim", "foreground_psnr", "foreground_ssim")}
    bf16_rows = []
    if args.device.startswith("cuda") and torch.cuda.is_bf16_supported():
        codec.autocast_dtype = torch.bfloat16
        for ii, item in enumerate(selected):
            if not any(g.startswith("simulation_") for g in item["groups"]):
                continue
            x = torch.from_numpy(references[ii])[None, None].to(args.device) * 2 - 1
            with torch.no_grad():
                pred = ((codec(x) + 1) / 2)[0, 0].cpu().numpy()
            bf = compact[ii] | metrics(references[ii], pred)
            bf.update(psnr_delta=bf["psnr"] - rows[ii]["psnr"],
                      ssim_delta=bf["ssim"] - rows[ii]["ssim"],
                      vs_fp32_rmse=float(np.sqrt(np.mean((pred-reconstructions[ii])**2))),
                      vs_fp32_max_absolute=float(np.max(np.abs(pred-reconstructions[ii]))))
            bf16_rows.append(bf)
        codec.autocast_dtype = None
    with torch.no_grad():
        x = torch.from_numpy(references[0])[None, None].to(args.device) * 2 - 1
        z = codec.encode(x)
        z_second = codec.encode(x)
    z_grad = z.detach().requires_grad_(True)
    decoded = codec.decode(z_grad)
    decoded.square().mean().backward()
    checks = dict(latent_shape=list(z.shape), reconstruction_shape=list(decoded.shape),
                  frozen=all(not p.requires_grad for p in codec.parameters()),
                  posterior_mode_repeat_max_difference=float((z-z_second).abs().max()),
                  decode_gradient_finite=bool(torch.isfinite(z_grad.grad).all()),
                  decode_gradient_nonzero=bool(z_grad.grad.abs().max() > 0),
                  decode_gradient_max=float(z_grad.grad.abs().max()),
                  vae_parameter_gradients_absent=all(p.grad is None for p in codec.parameters()))
    del decoded, z_grad
    all_summary = summarize(rows)
    screening = {source: dict(psnr_mean=values["psnr"]["mean"] >= 35,
                              ssim_mean=values["ssim"]["mean"] >= .95)
                 for source, values in by_source.items()}
    worst = sorted(range(len(rows)), key=lambda i: rows[i]["psnr"])[:8]
    mean = latent_sum/latent_count
    summary = dict(model_path=str(args.vae.resolve()), latent_channels=codec.latent_channels,
                   downsample_factor=codec.downsample_factor, scaling_factor=codec.scaling_factor,
                   all=all_summary, groups=groups, balanced_by_source=by_source,
                   source_equal_balanced=source_equal,
                   latent_stats_unaugmented_diagnostic_only=dict(channel_mean=mean.tolist(),
                       channel_std=np.sqrt(latent_sq/latent_count-mean**2).tolist(), observations_per_channel=latent_count),
                   screening=screening, screening_pass=all(all(v.values()) for v in screening.values()),
                   worst_psnr=[rows[i] for i in worst], checks=checks,
                   elapsed_seconds=time.time()-started, torch_version=torch.__version__)
    if bf16_rows:
        summary["bf16"] = summarize(bf16_rows) | dict(
            mean_psnr_delta=float(np.mean([r["psnr_delta"] for r in bf16_rows])),
            mean_ssim_delta=float(np.mean([r["ssim_delta"] for r in bf16_rows])),
            maximum_rmse_vs_fp32=max(r["vs_fp32_rmse"] for r in bf16_rows),
            maximum_absolute_vs_fp32=max(r["vs_fp32_max_absolute"] for r in bf16_rows))
    if args.device.startswith("cuda"):
        summary["peak_allocated_mb"] = torch.cuda.max_memory_allocated()/1024**2
        summary["peak_reserved_mb"] = torch.cuda.max_memory_reserved()/1024**2
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "metrics.json").write_text(json.dumps(rows, indent=2))
    (args.out / "bf16_metrics.json").write_text(json.dumps(bf16_rows, indent=2))
    with (args.out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.out / "reconstructions.npz", reference=np.array(references),
                        reconstruction=np.array(reconstructions), indices=np.array([r["index"] for r in rows]))
    make_preview(args.out / "worst_cases.png", worst, rows, references, reconstructions)
    representative = []
    for source in by_source:
        source_ids = [i for i, r in enumerate(rows) if r["dataset"] == source and "source_balanced" in r["groups"]]
        source_ids.sort(key=lambda i: rows[i]["psnr"])
        representative += [source_ids[0], source_ids[len(source_ids)//2]]
    make_preview(args.out / "source_comparison.png", representative, rows, references, reconstructions)
    table = ["# 冻结预训练 VAE 重建检查", "", f"模型：`{args.model_label}` 的 f4 VAE。192×192 单通道复制为 RGB，编码到 {codec.latent_channels}×48×48，再解码取 RGB 均值。", "",
             "VAE 固定，无拟合；输入归一化沿用原 uint16/65535。以下评估干净仿真参考图通过 VAE 的损失，不是 DiT 或 SPEN 重建结果。", "",
             "| 样本 | 张数 | PSNR (dB) | SSIM | 前景 PSNR | 前景 SSIM |", "|---|---:|---:|---:|---:|---:|"]
    for label, values in list(groups.items()) + list(by_source.items()):
        table.append(f"| {label} | {values['count']} | {values['psnr']['mean']:.3f} | {values['ssim']['mean']:.4f} | {values['foreground_psnr']['mean']:.3f} | {values['foreground_ssim']['mean']:.4f} |")
    table += ["", f"来源等权平均：PSNR {source_equal['psnr']:.3f} dB；SSIM {source_equal['ssim']:.4f}。",
              f"最差全图 PSNR {all_summary['psnr']['minimum']:.3f} dB；最差 SSIM {all_summary['ssim']['minimum']:.4f}（可能不是同一张）。",
              "", "SSIM 精确沿用旧实验的 Gaussian sigma=1.5、truncate=3.5、总体协方差、裁去边缘 5 像素定义；额外保存 skimage 默认 SSIM。前景定义为参考图 > 0.05，非脑分割；前景 SSIM 为 SSIM 图在该掩膜内、排除边缘 5 像素的均值。所有结果使用固定动态范围 1，不做逐图强度配准。",
              "", "旧仿真病例共 18 张，全部已经用于旧 prior 训练。这里只检验独立公开、未微调的 VAE 编解码误差，不作为新 prior 的泛化测试。",
              "", "抽样规则、阈值和逐图结果分别见 protocol.json、metrics.csv；最差病例及逐来源比较见 PNG。"]
    if bf16_rows:
        table += ["", f"BF16 对 18 张旧仿真病例：平均 PSNR 变化 {summary['bf16']['mean_psnr_delta']:.4f} dB，SSIM 变化 {summary['bf16']['mean_ssim_delta']:.6f}；最大单图 FP32/BF16 RMSE {summary['bf16']['maximum_rmse_vs_fp32']:.6f}。"]
    (args.out / "REPORT.md").write_text("\n".join(table) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
