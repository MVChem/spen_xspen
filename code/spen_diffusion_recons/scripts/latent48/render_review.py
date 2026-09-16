"""Render repeated VAE + DiT inference using the existing 260915 figure layout."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prior192"))
import render_rebuilt as renderer
from review_reconstruction import CAMPAIGN, REFERENCE, read_json
from verify_reconstruction import metrics, read_npz, sha256, verify_case


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    frozen = read_json(args.run / "frozen_artifact_verification.json")
    assert frozen["status"] == "passed"
    checkpoint_digest = sha256(args.source_run / "checkpoint.pt")
    reference = {kind: read_npz(args.reference_dir / f"{kind}.npz") for kind in ("simulation", "real")}
    groups = {key: read_json(args.run / key / "completed.json") for key in ("R1", "R2", "real16", "real24")}
    sim_new, real_new, repeats = [], [], []
    for key, completed in groups.items():
        assert completed["status"] == "completed" and completed["fresh_inference"]
        prov = read_json(args.run / key / "provenance.json")
        assert prov["checkpoint_sha256"] == checkpoint_digest and prov["checkpoint_step"] == 60000
        kind = "simulation" if key.startswith("R") else "real"
        assert prov["reference"][f"{kind}_npz_sha256"] == sha256(args.reference_dir / f"{kind}.npz")
        assert len(completed["cases"]) == (4 if kind == "simulation" else 5)
        images = []
        for row in completed["cases"]:
            i = row["case_index"]
            folder = args.run / key / f"case_{i:02d}"
            arrays, checked_row, _ = verify_case(folder, 60000, 60, real=kind == "real")
            assert checked_row["key"] == row["key"]
            image = ((arrays["prediction_raw"] + 1) / 2).clip(0, 1)
            if kind == "simulation":
                assert row["key"] == reference[kind]["case_keys"][i]
                assert np.array_equal(((arrays["gt"] + 1) / 2).clip(0, 1), reference[kind]["target"][i])
            else:
                assert row["metadata"]["fov_mm"] == reference[kind]["fov_mm"][i]
                assert f'Acquisition #{row["metadata"]["export_index"]}' == reference[kind]["labels"][i]
                image = np.rot90(image, 2)
            images.append(image)
            repeats.append(dict(group=key, case_key=row["key"], **row["repeat_check"]))
        if kind == "simulation":
            sim_new.append(np.stack(images))
        else:
            real_new.extend(images)
    sim_new, real_new = np.stack(sim_new), np.stack(real_new)
    assert sim_new.shape == (2, 4, 192, 192) and real_new.shape == (10, 192, 192)
    np.savez_compressed(args.run / "review_display_arrays.npz", simulation=sim_new, real=real_new)

    renderer.ROW_LABELS.update(pixel="Pixel diffusion", latent="VAE + DiT",
                              target="Ground truth (GT)\nSeen in prior training")
    sim, real = reference["simulation"], reference["real"]
    keys = ["degraded", "tikhonov", "phase_inva", "pixel", "latent"]
    sim_arrays = {k: sim[k] for k in keys[:3]} | dict(pixel=sim["diffusion"], latent=sim_new)
    real_arrays = {k: real[k] for k in keys[:3]} | dict(pixel=real["diffusion"], latent=real_new)
    target = np.concatenate([sim["target"]] * 2)
    annotations = {}
    for key, array in sim_arrays.items():
        values = np.concatenate(array)
        if key == "degraded":
            values = np.repeat(np.repeat(values, 2, -2), 2, -1)
        annotations[key] = renderer._metric_labels(values, target)
    sim_groups = [(0, 4, "Full PE · σ = 0.01"), (4, 8, "Random 50% PE · σ = 0.02")]
    renderer._draw_grid([target] + [np.concatenate(sim_arrays[k]) for k in keys],
        ["target"] + keys, sim["labels"].tolist() * 2, sim_groups,
        out / "figure1_simulation", annotations)
    renderer._draw_grid([real_arrays[k] for k in keys], keys, real["labels"].tolist(),
        [(0, 5, "FOV 16 mm"), (5, 10, "FOV 24 mm")], out / "figure2_real")

    sim_rows, averages = [], {}
    for ci, group in enumerate(("R1", "R2")):
        averages[group] = {}
        for key in keys:
            scores = []
            for i in range(4):
                value = sim_arrays[key][ci, i]
                if key == "degraded":
                    value = np.repeat(np.repeat(value, 2, -2), 2, -1)
                score = metrics(2 * value - 1, 2 * sim["target"][i] - 1)
                scores.append(score)
                sim_rows.append(dict(condition=group, method=key, case_key=str(sim["case_keys"][i]), **score))
                if key == "latent":
                    stored = groups[group]["cases"][i]["metrics"]
                    assert all(abs(score[m] - stored[m]) <= 1e-6 for m in score)
            averages[group][key] = {m: float(np.mean([s[m] for s in scores])) for m in ("psnr", "ssim")}
    real_rows = []
    for group in ("real16", "real24"):
        for row in groups[group]["cases"]:
            meta = row["metadata"]
            real_rows.append(dict(fov_mm=meta["fov_mm"], acquisition=meta["export_index"],
                tikhonov_nrmse=meta["methods"]["tikhonov"]["measurement_nrmse"],
                phase_inva_nrmse=meta["methods"]["phase_inva"]["measurement_nrmse"],
                pixel_nrmse=meta["methods"]["diffusion"]["measurement_nrmse"],
                latent_nrmse=row["measurement_nrmse"]))
    real_means = {str(fov): {m: float(np.mean([r[m] for r in real_rows if r["fov_mm"] == fov]))
                            for m in real_rows[0] if m.endswith("nrmse")} for fov in (16, 24)}
    write_csv(out / "simulation_metrics.csv", sim_rows)
    write_csv(out / "real_measurement_metrics.csv", real_rows)
    report = dict(date=datetime.now().isoformat(), checkpoint_step=60000,
        checkpoint=str(args.source_run / "checkpoint.pt"), checkpoint_sha256=checkpoint_digest,
        fresh_inference_cases=18, simulation=averages, real_measurement_nrmse=real_means,
        repeat_checks=repeats, run=str(args.run), reference=str(args.reference_dir),
        template="scripts/prior192/render_rebuilt.py::_draw_grid", display_window=[0, 1],
        row_order={"simulation": ["target"] + keys, "real": keys},
        no_per_image_intensity_fitting=True, source_sha256=sha256(Path(__file__)),
        limitations=["Simulation cases participated in prior training; not a held-out generalization test.",
                     "No paired real GT; measurement NRMSE is not an image-quality metric.",
                     "192x192 output grid does not establish doubled physical resolution."])
    (out / "review.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# VAE + DiT 真实与仿真复测 · 260916", "",
        "VAE 为冻结的 stable-diffusion-x4-upscaler 预训练编码器；DiT 已完成 60,000 步训练，使用最终 EMA。",
        "本轮重新推理 18 次：4 个仿真病例 × 2 种采样条件，以及 10 个真实采集病例。", "",
        "沿用 SPEN_Reconstruction_Comparison_260915 的绘图函数、病例顺序、灰度窗和原始传统基线，并加入旧像素扩散与 VAE + DiT 两行。实采方向与原图一致。", "",
        "![仿真对比](figure1_simulation.png)", "", "![真实数据对比](figure2_real.png)", "",
        "## 仿真平均 PSNR / SSIM", "", "| 方法 | 完整 PE，σ=0.01 | 随机 50% PE，σ=0.02 |", "| --- | ---: | ---: |"]
    for key in keys:
        values = [averages[g][key] for g in ("R1", "R2")]
        lines.append(f'| {renderer.ROW_LABELS[key].replace(chr(10), " ")} | ' +
                     " | ".join(f'{v["psnr"]:.3f} / {v["ssim"]:.4f}' for v in values) + " |")
    lines += ["", "仿真病例均参与过先验训练；这些结果用于查看已训练模型，不能作为独立测试集泛化结论。", "",
        "## 实采平均测量 NRMSE", "", "| FOV | Tikhonov | Phase map + InvA | Pixel diffusion | VAE + DiT |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for fov, scores in real_means.items():
        lines.append(f'| {fov} mm | ' + " | ".join(f'{scores[m]:.4f}' for m in
            ("tikhonov_nrmse", "phase_inva_nrmse", "pixel_nrmse", "latent_nrmse")) + " |")
    max_delta = max(r["prediction_raw_max_abs"] for r in repeats)
    max_rmse = max(r["prediction_raw_rmse"] for r in repeats)
    max_psnr_delta = max(abs(r["psnr_delta"]) for r in repeats if "psnr_delta" in r)
    lines += ["", "实采没有配对 GT，不计算 PSNR/SSIM。测量残差更低仅表示当前前向模型下更符合观测，不能单独确定图像质量。192×192 为输出网格。", "",
        "## 复测与来源", "",
        f"- 原始最终产物及 42 条优化轨迹已重新核验；本轮 18 次推理的指标和优化轨迹也已核验。",
        f"- 相同观测与参数下，本轮与原结果在原始 [-1,1] 数组上的最大绝对差为 {max_delta:.8g}。",
        f"- 本轮不是逐像素完全一致的复现：原始数组逐例 RMSE 最大 {max_rmse:.6f}，仿真逐例 PSNR 最大变化 {max_psnr_delta:.4f} dB。局部像素有差异，平均指标接近；此处未定位数值差异的具体来源。",
        "- 固定采用之前校准选出的 λ：Full PE 为 0.1，50% PE 为 1；实采采用 0.1。此次未重新调参。",
        "- 60 个外层步，每步最多 8 次近端更新；编码器与 DiT 使用 BF16，解码器使用 FP32。",
        f'- 权重：`{args.source_run / "checkpoint.pt"}`；SHA256：`{checkpoint_digest}`。',
        f'- 训练脚本：`{HERE / "train_latent_ddp.py"}`。',
        f'- 训练数据：`{args.reference_dir.parent / "data_all"}`。',
        f'- 实采数据：`{HERE.parents[2] / "data/spen_acquired_260915/mat"}`。',
        f'- 本轮推理与逐例数组：`{args.run}`。',
        f'- 推理入口：`{HERE / "review_reconstruction.py"}`；绘图入口：`{Path(__file__).resolve()}`。',
        '- Python 环境：`/home/data2/chk/workspace/2026/.venv/bin/python`。', "",
        "[逐例仿真指标](simulation_metrics.csv) · [逐例实采指标](real_measurement_metrics.csv) · [完整复测摘要](review.json)", ""]
    (out / "复测说明_260916.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(status="rendered", out=str(out), cases=18,
                         maximum_repeat_raw_difference=max_delta, simulation=averages,
                         real_measurement_nrmse=real_means), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=CAMPAIGN / "reconstruction_final_260916")
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for key in ("run", "source_run", "reference_dir", "out"):
        setattr(args, key, getattr(args, key).resolve())
    render(args)


if __name__ == "__main__":
    main()
