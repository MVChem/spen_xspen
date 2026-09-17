"""Audit native96 versus latent192 report metrics and align shared real cases.

Uses saved predictions only. Pooled metrics and foreground metrics are labeled
diagnostics, not a matched reconstruction benchmark or image-quality ground truth.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "prior96"))
from render_comparison import draw_comparison


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def score(pred, target):
    pred, target = pred.astype(np.float32), target.astype(np.float32)
    err = (pred-target)**2
    mu, mg = [gaussian_filter(x, 1.5, truncate=3.5) for x in (pred, target)]
    var = gaussian_filter(pred*pred, 1.5, truncate=3.5)-mu**2
    vg = gaussian_filter(target*target, 1.5, truncate=3.5)-mg**2
    cov = gaussian_filter(pred*target, 1.5, truncate=3.5)-mu*mg
    smap = ((2*mu*mg+.01**2)*(2*cov+.03**2)
            / ((mu**2+mg**2+.01**2)*(var+vg+.03**2)))
    mask = target > .05
    crop = np.s_[5:-5, 5:-5]
    return dict(psnr=float(-10*np.log10(max(float(err.mean()), 1e-12))),
        ssim=float(smap[crop].mean()), foreground_psnr=float(-10*np.log10(max(float(err[mask].mean()), 1e-12))),
        foreground_ssim=float(smap[crop][mask[crop]].mean()), foreground_fraction=float(mask.mean()))


def mean(rows):
    return {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}


def pool(x):
    return x.reshape(*x.shape[:-2], 96, 2, 96, 2).mean((-3, -1))


def audit_simulation(out):
    native = PROJECT / "runs/retrain_0911_260916/evaluation"
    data = PROJECT.parent / "data/prior96_0911_260916/mouse_mixed"
    manifest = read(data / "manifest.json")
    report = dict(native96={}, latent192={}, partition_checks={})
    for field in ("subject", "split_group", "source_sha256"):
        sets = {part: {r[field] for r in rows if field in r} for part, rows in manifest["records"].items()}
        report["partition_checks"][field] = {f"{a}_vs_{b}": len(sets[a] & sets[b])
            for a, b in (("train", "test"), ("train", "val"), ("val", "test"))}
        assert all(v == 0 for v in report["partition_checks"][field].values())
    training = np.load(data / "train.npy", mmap_mode="r")
    training_hashes = {hashlib.sha256(im.tobytes()).hexdigest() for im in training}
    test = np.load(data / "test.npy", mmap_mode="r")
    test_index = {r["key"]: i for i, r in enumerate(manifest["records"]["test"])}
    for key in ("fov16_R1", "fov16_R2"):
        values = arrays(native / f"{key}.npz")
        saved = read(native / f"{key}_metrics.json")
        records = saved["records"]
        max_gt_difference, duplicated = 0., []
        for i, record in enumerate(records):
            im = test[test_index[record["key"]]]
            if hashlib.sha256(im.tobytes()).hexdigest() in training_hashes:
                duplicated.append(record["key"])
            im = im.astype(np.float32)/65535
            if record["rot180"]:
                im = np.rot90(im, 2)
            max_gt_difference = max(max_gt_difference, float(np.abs(im-values["target"][i]).max()))
        assert not duplicated and max_gt_difference < 1e-7
        methods = {}
        max_metric_difference = 0.
        for method in ("raw_rss", "phase_inva", "tikhonov", "diffusion"):
            rows = [score(p, g) for p, g in zip(values[method], values["target"])]
            by_subject = defaultdict(list)
            for i, (row, record) in enumerate(zip(rows, records)):
                for name in ("psnr", "ssim"):
                    max_metric_difference = max(max_metric_difference,
                        abs(row[name]-saved["methods"][method]["cases"][i][name]))
                by_subject[record["subject"]].append(row)
            average = mean([mean(v) for v in by_subject.values()])
            methods[method] = dict(subject_mean=average, cases=rows,
                displayed_cases=[dict(index=i, dataset=records[i]["dataset"], **rows[i]) for i in (3, 8, 17)])
        assert max_metric_difference < 2e-5
        raw = (values["diffusion_model_range"][:, 0]+1)/2
        unclipped_psnr = [-10*np.log10(np.mean((p-g)**2)) for p, g in zip(raw, values["target"])]
        report["native96"][key] = dict(methods=methods, images=len(records),
            subjects=len({r["subject"] for r in records}), sources=dict(Counter(r["dataset"] for r in records)),
            max_metric_difference=max_metric_difference, max_manifest_gt_difference=max_gt_difference,
            exact_training_image_duplicates=duplicated, unclipped_diffusion_psnr_mean=float(np.mean(unclipped_psnr)))
    sr = PROJECT / "runs/rodent192_spen2x_260914/figures_260915"
    gt = arrays(sr / "simulation.npz")["target"]
    review = PROJECT / "runs/rodent192_latent_dit_260915/review_260916"
    prediction = arrays(review / "review_display_arrays.npz")["simulation"]
    for i, condition in enumerate(("R1", "R2")):
        report["latent192"][condition] = dict(native_grid=mean([score(p, g) for p,g in zip(prediction[i],gt)]),
            pooled96_diagnostic=mean([score(p,g) for p,g in zip(pool(prediction[i]),pool(gt))]),
            pooling_note="2x2 average both existing prediction and GT; same SR observations and prediction. Not the old96 test.")
    return report


def compare_real(out):
    native = PROJECT / "runs/retrain_0911_260916/evaluation_real"
    campaign = PROJECT / "runs/rodent192_latent_dit_260915"
    reference = PROJECT / "runs/rodent192_spen2x_260914/figures_260915"
    old = arrays(native / "real.npz")
    old_meta = read(native / "real.json")["cases"]
    new_meta = read(reference / "real.json")["cases"]
    new = arrays(campaign / "review_260916/review_display_arrays.npz")["real"]
    rois = read(campaign / "reconstruction_final_260916/real_zoom_260916/roi_config.json")["rois"]
    lookup = {(r["fov_mm"],r["export_index"]): i for i,r in enumerate(new_meta)}
    selected = [lookup[(r["fov_mm"],r["export_index"])] for r in old_meta]
    comparisons = []
    for i,j in enumerate(selected):
        a,b = old_meta[i],new_meta[j]
        assert a["sha256"] == b["source_sha256"]
        assert a["magnitude_scale"] == b["magnitude_scale"]
        assert a["encoding_smax"] == b["source_encoding_smax"]
        old_raw = arrays(native / f'real_fov{a["fov_mm"]}_slice_{a["export_index"]}.npz')
        fresh = arrays(campaign / "review_260916" / f'real{a["fov_mm"]}' / f'case_{j:02d}' / "arrays.npz")
        observation_difference = float(np.abs(old_raw["observation"]-fresh["observation"]).max())
        assert observation_difference < 2e-6
        comparisons.append(dict(fov_mm=a["fov_mm"], export_index=a["export_index"], source_sha256=a["sha256"],
            native_index=i, latent_index=j, observation_max_difference=observation_difference,
            same_magnitude_scale=True, same_encoding_smax=True,
            native96_measurement_nrmse=a["methods"]["diffusion"]["measurement_nrmse"],
            roi=rois[j]))
    native_images=old["diffusion"]
    # Bicubic display control isolates the effect of displaying more pixels.
    up = F.interpolate(torch.from_numpy(native_images)[:,None], (192,192),
                       mode="bicubic", align_corners=False)[:,0].numpy().clip(0,1)
    latent=new[selected]
    images=dict(native96=native_images, native96_bicubic=up, latent192=latent)
    labels={"native96":"96 Diffusion", "native96_bicubic":"96 Diffusion\nbicubic display", "latent192":"VAE + DiT"}
    groups=[(0,3,"FOV 16 mm"),(3,6,"FOV 24 mm")]
    draw_comparison(images,out/"real_same_cases",old["labels"].tolist(),groups,row_labels=labels)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,6,figsize=(15,6.2),layout="constrained")
    for row,(method,values) in enumerate(images.items()):
        for col,j in enumerate(selected):
            roi=rois[j]
            x,y,w,h=[roi[k] for k in ("x","y","width","height")]
            if method=="native96": x,y,w,h=[v//2 for v in (x,y,w,h)]
            ax=axes[row,col]
            ax.imshow(values[col,y:y+h,x:x+w],cmap="gray",vmin=0,vmax=1,interpolation="nearest")
            ax.set_xticks([]);ax.set_yticks([])
            if row==0: ax.set_title(f'{comparisons[col]["fov_mm"]} mm · #{comparisons[col]["export_index"]}',fontsize=11)
            if col==0: ax.set_ylabel(labels[method],fontsize=11)
    fig.suptitle("Same brain ROI · fixed grayscale [0,1] · no paired real GT",fontsize=14)
    fig.savefig(out/"real_same_roi.png",dpi=220)
    plt.close(fig)
    np.savez_compressed(out/"real_display_arrays.npz",**images)
    return dict(cases=comparisons, display="Identical cases, FOV, orientation and [0,1] window; existing predictions only",
        caveat="Native96 and latent192 have different forward-model image grids, priors, initialization and solvers. Per-pipeline measurement NRMSE is not a common image-quality score.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(simulation=audit_simulation(args.out),real=compare_real(args.out))
    (args.out/"audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    compact={"native96":{k:{m:r["subject_mean"] for m,r in v["methods"].items()} for k,v in report["simulation"]["native96"].items()},
        "latent192":report["simulation"]["latent192"], "partition_checks":report["simulation"]["partition_checks"],
        "matched_real_cases":len(report["real"]["cases"])}
    print(json.dumps(compact,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
