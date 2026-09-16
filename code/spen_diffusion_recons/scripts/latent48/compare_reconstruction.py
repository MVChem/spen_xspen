"""Compare verified latent reconstruction against the frozen pixel-prior figures.

Reuses the reference display and metric convention, including the real-data
orientation. Creates two figures with both priors, numeric CSV/JSON tables,
and a short Chinese report. No inference or intensity fitting is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prior192"))
import render_rebuilt as renderer
from verify_reconstruction import metrics, read_json, read_npz, sha256


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(reference: Path, run: Path, preview: Path | None, out: Path):
    verification = read_json(run / "verification/final.json")
    if verification["status"] != "passed":
        raise ValueError("Final reconstruction must pass artifact verification first")
    summary = read_json(run / "summary.json")
    step = summary["checkpoint_step"]
    old = {kind: read_npz(reference / f"{kind}.npz") for kind in ("simulation", "real")}
    new = {kind: read_npz(run / f"{kind}.npz") for kind in old}
    for kind in old:
        if sha256(reference / f"{kind}.npz") != read_json(run / "render_metadata.json")["reference_sha256"][f"{kind}.npz"]:
            raise ValueError(f"Reference changed after inference: {kind}")
    previous = read_json(preview / "summary.json") if preview else None
    out.mkdir(parents=True, exist_ok=True)
    for name in ("comparison_simulation.png", "comparison_real.png", "comparison.json", "RESULTS_260916.md"):
        if (out / name).exists():
            raise FileExistsError(out / name)

    renderer.ROW_LABELS.update(pixel="Pixel diffusion\nstep 60,000",
                               latent=f"VAE + DiT\nstep {step:,}")
    sim = old["simulation"]
    row_keys = ["target", "degraded", "phase_inva", "tikhonov", "pixel", "latent"]
    arrays = {k: sim[k] for k in ("degraded", "phase_inva", "tikhonov")}
    arrays.update(pixel=sim["diffusion"], latent=new["simulation"]["diffusion"])
    images = [np.concatenate([sim["target"]] * 2)]
    images += [np.concatenate(arrays[k]) for k in row_keys[1:]]
    target = images[0]
    annotations = {}
    for key, values in zip(row_keys[1:], images[1:]):
        if key == "degraded":
            values = np.repeat(np.repeat(values, 2, -2), 2, -1)
        annotations[key] = renderer._metric_labels(values, target)
    renderer._draw_grid(images, row_keys, sim["labels"].tolist() * 2,
        [(0, 4, f"Full PE · σ = {sim['noise_sigma'][0]:g}"),
         (4, 8, f"Random 50% PE · σ = {sim['noise_sigma'][1]:g}")],
        out / "comparison_simulation", annotations)

    real = old["real"]
    renderer._draw_grid([real[k] for k in ("degraded", "phase_inva", "tikhonov")]
        + [real["diffusion"], new["real"]["diffusion"]], row_keys[1:], real["labels"].tolist(),
        [(0, 5, "FOV 16 mm"), (5, 10, "FOV 24 mm")], out / "comparison_real")

    simulation_rows, aggregates = [], {}
    for ci, condition in enumerate(("R1", "R2")):
        aggregates[condition] = {}
        for method, values in arrays.items():
            rows = []
            for index, value in enumerate(values[ci]):
                if method == "degraded":
                    value = np.repeat(np.repeat(value, 2, -2), 2, -1)
                score = metrics(2 * value - 1, 2 * sim["target"][index] - 1)
                row = dict(condition=condition, source=str(sim["labels"][index]),
                           case_key=str(sim["case_keys"][index]), method=method, **score)
                rows.append(row)
                simulation_rows.append(row)
            aggregates[condition][method] = {
                k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim")}
        if previous:
            prior = previous["simulation"][condition]["new"]
            aggregates[condition]["latent_previous"] = dict(psnr=prior["mean_psnr"], ssim=prior["mean_ssim"])
        aggregates[condition]["delta_vs_pixel"] = {
            k: aggregates[condition]["latent"][k] - aggregates[condition]["pixel"][k]
            for k in ("psnr", "ssim")}
        for key in ("psnr", "ssim"):
            if abs(aggregates[condition]["latent"][key] - summary["simulation"][condition]["new"][f"mean_{key}"]) > 1e-6:
                raise ValueError("Recomputed metrics differ from verified inference output")

    real_rows = []
    for row in summary["real"]:
        meta = row["metadata"]
        real_rows.append(dict(fov_mm=meta["fov_mm"], acquisition=meta["export_index"],
            phase_inva_nrmse=meta["methods"]["phase_inva"]["measurement_nrmse"],
            tikhonov_nrmse=meta["methods"]["tikhonov"]["measurement_nrmse"],
            pixel_nrmse=meta["methods"]["diffusion"]["measurement_nrmse"],
            latent_nrmse=row["measurement_nrmse"],
            latent_outside_range_fraction=row["outside_range_fraction"]))
    real_means = {str(fov): {k: float(np.mean([r[k] for r in real_rows if r["fov_mm"] == fov]))
                            for k in real_rows[0] if k.endswith("nrmse")} for fov in (16, 24)}
    report = dict(checkpoint_step=step, checkpoint_sha256=summary["checkpoint_sha256"],
        previous_step=previous["checkpoint_step"] if previous else None,
        reference=str(reference), run=str(run), simulation=aggregates, real_measurement_nrmse=real_means,
        verification_status=verification["status"], limitations=summary["limitations"],
        display="Fixed [0,1]; no per-image rescaling; real data share reference rot180",
        metric="Mean per-case PSNR/SSIM; clip prediction and GT to [0,1]; Gaussian SSIM sigma=1.5, crop=5",
        source_sha256=sha256(Path(__file__)))
    (out / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_csv(out / "simulation_metrics.csv", simulation_rows)
    write_csv(out / "real_measurement_metrics.csv", real_rows)

    lines = ["# VAE + DiT 最终权重重建对比 · 260916", "",
        f"使用已完成的 **{step:,} 步 EMA**，对照 `experiments/SPEN_Reconstruction_Comparison_260915` 的同一组数据。",
        "仿真 4 个展示病例 × 2 种采样条件；实采 FOV 16/24 mm 各 5 个病例。", "",
        "## 仿真：四例平均 PSNR / SSIM", "",
        "| 方法 | Full PE，σ=0.01 | Random 50% PE，σ=0.02 |",
        "| --- | ---: | ---: |"]
    methods = [("degraded", "Degraded input"), ("phase_inva", "Phase map + InvA"),
               ("tikhonov", "Tikhonov"), ("pixel", "像素扩散 · 60,000 步")]
    if previous:
        methods.append(("latent_previous", f"VAE + DiT · {previous['checkpoint_step']:,} 步"))
    methods.append(("latent", f"VAE + DiT · {step:,} 步"))
    for method, label in methods:
        values = [aggregates[c][method] for c in ("R1", "R2")]
        lines.append(f"| {label} | " + " | ".join(f"{r['psnr']:.3f} / {r['ssim']:.4f}" for r in values) + " |")
    lines += ["", "![仿真对比](comparison_simulation.png)", "", "## 实采", "",
        "![实采对比](comparison_real.png)", "",
        "实采没有配对高分辨率 GT，不计算 PSNR/SSIM。下表为同一前向模型下的测量 NRMSE（越低表示拟合观测越好），不能单独据此判断图像质量。", "",
        "| FOV | Phase map + InvA | Tikhonov | 像素扩散 | VAE + DiT |", "| --- | ---: | ---: | ---: | ---: |"]
    for fov, values in real_means.items():
        lines.append(f"| {fov} mm | " + " | ".join(f"{values[k]:.4f}" for k in
                     ("phase_inva_nrmse", "tikhonov_nrmse", "pixel_nrmse", "latent_nrmse")) + " |")
    selected = {c: summary["simulation"][c]["new"]["selected"]["lamb"] for c in ("R1", "R2")}
    lines += ["", "## 比较条件与核验", "",
        f"- 最终模型：`{run / 'checkpoint.pt'}`；SHA256：`{summary['checkpoint_sha256']}`。",
        f"- 两种仿真条件分别在原 4 个调参病例上从 λ=0.1、1、10 中选取，得到 Full PE λ={selected['R1']:g}、50% PE λ={selected['R2']:g}。展示病例不参与调参。",
        "- 实采沿用 Full PE 选出的 λ，噪声参数 σ=0.02。60 个外层步，每步最多 8 个非线性近端更新。",
        "- 与旧图共用观测、PE 掩码、线圈/相位模型、传统基线、灰度窗和指标定义。潜空间求解器与初始化不同，因此是完整重建方案比较，不是仅替换网络的消融。",
        "- 仿真调参和展示病例均参与过先验训练，本次不能作为独立测试集泛化结论；只有一个固定重建随机种子。",
        "- VAE 保持冻结；编码器和 DiT 用 BF16，解码器关闭 autocast，测量约束直接通过解码器反传。",
        "- 独立核验通过：权重、病例顺序、观测、传统基线、显示方向、重算指标及所有 42 次重建的优化轨迹。有限内迭代不代表近端子问题已精确收敛。",
        "- 192×192 是输出网格，不能据此认定实采空间分辨率翻倍。", "",
        f"完整运行、参数和逐例数组：`{run}`。", "",
        "[逐例仿真指标](simulation_metrics.csv) · [实采测量指标](real_measurement_metrics.csv) · [结构化汇总](comparison.json)", ""]
    (out / "RESULTS_260916.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compare(args.reference_dir.resolve(), args.run.resolve(),
            args.preview.resolve() if args.preview else None, args.out.resolve())


if __name__ == "__main__":
    main()
