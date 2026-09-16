"""Render matched full-image / ROI pairs from saved real reconstruction arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


METHODS = [
    ("degraded", "Degraded input", "96 × 96"),
    ("phase_inva", "Phase map\n+ InvA", "192 × 192"),
    ("tikhonov", "Tikhonov", "192 × 192"),
    ("pixel", "Pixel diffusion", "60,000 steps · 192 × 192"),
    ("latent", "VAE + DiT", "60,000 steps · 192 × 192"),
]
ACCENT = "#20B9C5"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def render(arrays, rois, indices, out, name):
    # Physical panel sizes preserve a true 3x linear zoom: 192/64 = 3.
    side, col_gap, left, right = 1.8, .11, 1.95, .15
    zoom_gap, row_gap, top, bottom = .075, .31, 1.10, .36
    zoom_height = side * 40 / 64
    block_height = side + zoom_gap + zoom_height
    xs, cursor = [], left
    for column, index in enumerate(indices):
        if column and rois[index]["fov_mm"] != rois[indices[column - 1]]["fov_mm"]:
            cursor += .22
        xs.append(cursor)
        cursor += side + col_gap
    width = cursor - col_gap + right
    height = top + len(METHODS) * block_height + (len(METHODS) - 1) * row_gap + bottom
    fig = plt.figure(figsize=(width, height), facecolor="white")

    def axes(x, y, w, h):
        return fig.add_axes([x / width, 1 - (y + h) / height, w / width, h / height])

    fig.text(left / width, 1 - .21 / height, "Acquired SPEN · Regional comparison",
             ha="left", va="center", fontsize=16, weight="bold", color="#202933")
    fig.text(left / width, 1 - .49 / height,
             "Each panel: full image above · 3× ROI below     |     Same ROI and grayscale window across methods",
             ha="left", va="center", fontsize=8.8, color="#56616D")
    for fov in dict.fromkeys(rois[i]["fov_mm"] for i in indices):
        columns = [j for j, i in enumerate(indices) if rois[i]["fov_mm"] == fov]
        center = (xs[columns[0]] + xs[columns[-1]] + side) / 2
        fig.text(center / width, 1 - .77 / height, f"FOV {fov} mm", ha="center",
                 va="center", fontsize=12, weight="bold", color="#202933")
    for column, index in enumerate(indices):
        fig.text((xs[column] + side / 2) / width, 1 - .99 / height,
                 rois[index]["label"], ha="center", va="center", fontsize=10)

    for row, (key, label, grid) in enumerate(METHODS):
        y = top + row * (block_height + row_gap)
        center_y = y + block_height / 2
        fig.text((left - .17) / width, 1 - (center_y - .14) / height, label,
                 ha="right", va="center", fontsize=11.3, weight="bold", color="#202933")
        fig.text((left - .17) / width, 1 - (center_y + .24) / height, grid,
                 ha="right", va="center", fontsize=8.1, color="#64707D")
        if row:
            divider = 1 - (y - row_gap / 2) / height
            fig.add_artist(Line2D([.18 / width, (width - right) / width], [divider, divider],
                                 transform=fig.transFigure, color="#DDE2E7", lw=.6))
        for column, index in enumerate(indices):
            roi = rois[index]
            x0, y0, w, h = (roi[k] for k in ("x", "y", "width", "height"))
            image = arrays[key][index]
            full = axes(xs[column], y, side, side)
            full.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            full.set_axis_off()
            full.add_patch(Rectangle((x0 - .5, y0 - .5), w, h, fill=False, ec=ACCENT, lw=.9))
            zoom = axes(xs[column], y + side + zoom_gap, side, zoom_height)
            zoom.imshow(image[y0:y0+h, x0:x0+w], cmap="gray", vmin=0, vmax=1,
                        interpolation="nearest")
            zoom.set_xticks([])
            zoom.set_yticks([])
            for spine in zoom.spines.values():
                spine.set_color(ACCENT)
                spine.set_linewidth(.9)
    fig.text(left / width, .14 / height,
             "ROI: 64 × 40 on the 192 × 192 display grid · nearest-neighbor display · grayscale [0, 1]",
             fontsize=8, color="#64707D", va="center")
    for extension in ("png", "pdf"):
        fig.savefig(out / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def render_pairs(arrays, rois, indices, out, name):
    """Keep the two methods adjacent within each acquisition; one band per FOV."""
    groups = [[i for i in indices if rois[i]["fov_mm"] == fov]
              for fov in dict.fromkeys(rois[i]["fov_mm"] for i in indices)]
    side, method_gap, case_gap = 1.72, .075, .29
    left, right, top, bottom = .22, .22, .77, .34
    zoom_gap, zoom_height, band_header, band_gap = .07, side * 40 / 64, .81, .40
    pair_width = 2 * side + method_gap
    band_height = band_header + side + zoom_gap + zoom_height
    width = left + max(map(len, groups)) * (pair_width + case_gap) - case_gap + right
    height = top + len(groups) * band_height + (len(groups) - 1) * band_gap + bottom
    fig = plt.figure(figsize=(width, height), facecolor="white")

    def axes(x, y, w, h):
        return fig.add_axes([x / width, 1 - (y + h) / height, w / width, h / height])

    fig.text(left / width, 1 - .23 / height,
             "Real acquisitions · PhaseMap + InvA vs diffusion",
             ha="left", va="center", fontsize=16, weight="bold", color="#202933")
    fig.text(left / width, 1 - .52 / height,
             "Each acquisition: PhaseMap + InvA on the left, VAE + DiT on the right   |   Full image above, 3× ROI below",
             ha="left", va="center", fontsize=9, color="#56616D")
    for band, cases in enumerate(groups):
        start_y = top + band * (band_height + band_gap)
        fig.text(left / width, 1 - (start_y + .08) / height,
                 f"FOV {rois[cases[0]]['fov_mm']} mm", fontsize=12,
                 weight="bold", color="#202933", ha="left", va="center")
        if band:
            divider = 1 - (start_y - band_gap / 2) / height
            fig.add_artist(Line2D([left / width, 1 - right / width], [divider, divider],
                                 transform=fig.transFigure, color="#DDE2E7", lw=.7))
        for column, index in enumerate(cases):
            start_x = left + column * (pair_width + case_gap)
            roi = rois[index]
            x0, y0, w, h = (roi[k] for k in ("x", "y", "width", "height"))
            fig.text((start_x + pair_width / 2) / width,
                     1 - (start_y + .29) / height, roi["label"], fontsize=11,
                     ha="center", va="center", weight="bold", color="#202933")
            for method, (key, label) in enumerate((("phase_inva", "PhaseMap + InvA"),
                                                   ("latent", "Ours · VAE + DiT"))):
                x = start_x + method * (side + method_gap)
                y = start_y + band_header
                fig.text((x + side / 2) / width, 1 - (start_y + .52) / height,
                         label, fontsize=9.5, ha="center", va="center", color="#394550")
                grid_label = "96 → 192 bicubic" if key == "phase_inva" else "48 latent → 192 decoded"
                fig.text((x + side / 2) / width, 1 - (start_y + .69) / height,
                         grid_label, fontsize=7.7, ha="center", va="center", color="#64707D")
                image = arrays[key][index]
                full = axes(x, y, side, side)
                full.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                full.set_axis_off()
                full.add_patch(Rectangle((x0 - .5, y0 - .5), w, h,
                                         fill=False, ec=ACCENT, lw=.9))
                zoom = axes(x, y + side + zoom_gap, side, zoom_height)
                zoom.imshow(image[y0:y0+h, x0:x0+w], cmap="gray", vmin=0, vmax=1,
                            interpolation="nearest")
                zoom.set_xticks([])
                zoom.set_yticks([])
                for spine in zoom.spines.values():
                    spine.set_color(ACCENT)
                    spine.set_linewidth(.9)
    fig.text(left / width, .14 / height,
             "Same 64 × 40 ROI per acquisition · grayscale [0, 1] · nearest-neighbor display · VAE + DiT: 60,000-step EMA",
             fontsize=8, color="#64707D", va="center")
    for extension in ("png", "pdf"):
        fig.savefig(out / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Original pixel-prior real.npz")
    parser.add_argument("--latent", type=Path, required=True, help="Verified VAE+DiT real.npz")
    parser.add_argument("--rois", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--focus-pair", action="store_true",
                        help="Only PhaseMap + InvA and VAE+DiT, adjacent within each acquisition")
    args = parser.parse_args()
    old, new = read_npz(args.reference), read_npz(args.latent)
    config = json.loads(args.rois.read_text())
    rois = config["rois"]
    if int(new["checkpoint_step"]) != 60000:
        raise ValueError("Expected final 60,000-step latent reconstruction")
    if len(rois) != 10 or not np.array_equal(old["labels"], new["labels"]):
        raise ValueError("Case count or ordering differs")
    for key in ("degraded", "phase_inva", "tikhonov", "fov_mm"):
        if not np.array_equal(old[key], new[key]):
            raise ValueError(f"Reference changed: {key}")
    arrays = {k: old[k] for k in ("phase_inva", "tikhonov")}
    arrays.update(degraded=np.repeat(np.repeat(old["degraded"], 2, -2), 2, -1),
                  pixel=old["diffusion"], latent=new["diffusion"])
    if args.focus_pair:
        arrays = {key: arrays[key] for key in ("phase_inva", "latent")}
    for key, values in arrays.items():
        if values.shape != (10, 192, 192) or not np.isfinite(values).all():
            raise ValueError(f"Invalid image array: {key}")
    crops = {}
    for index, roi in enumerate(rois):
        if (roi["index"] != index or roi["label"] != old["labels"][index]
                or roi["fov_mm"] != old["fov_mm"][index]):
            raise ValueError("ROI case identity differs")
        x, y, w, h = (roi[k] for k in ("x", "y", "width", "height"))
        if w != 64 or h != 40 or not (0 <= x <= 192-w and 0 <= y <= 192-h):
            raise ValueError("Expected in-bounds 64 x 40 ROIs")
        if any(v % 2 for v in (x, y, w, h)):
            raise ValueError("ROIs must align to native96 pixels")
    for key, values in arrays.items():
        crops[key] = np.stack([values[i, r["y"]:r["y"]+r["height"],
                                        r["x"]:r["x"]+r["width"]] for i, r in enumerate(rois)])
    args.out.mkdir(parents=True, exist_ok=True)
    prefix = "real_inva_vs_diffusion" if args.focus_pair else "comparison_real_zoom"
    outputs = [prefix, prefix + "_fov16", prefix + "_fov24"]
    if any((args.out / f"{name}.png").exists() for name in outputs):
        raise FileExistsError("Use a fresh output directory")
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        for indices, name in zip((list(range(10)), list(range(5)), list(range(5, 10))), outputs):
            render_fn = render_pairs if args.focus_pair else render
            render_fn(arrays, rois, indices, args.out, name)
    np.savez_compressed(args.out / "roi_arrays.npz", **crops,
                        labels=old["labels"], fov_mm=old["fov_mm"])
    metadata = dict(reference=str(args.reference.resolve()), latent=str(args.latent.resolve()),
                    input_sha256={str(p.resolve()): sha(p) for p in (args.reference, args.latent, args.rois)},
                    rois=config, methods=list(arrays), panels=10 * len(arrays), zoom=3,
                    layout="adjacent method pairs by acquisition" if args.focus_pair else "method rows",
                    intensity_window=[0, 1], interpolation="nearest", generated_detail=False,
                    native_input="96x96 repeated 2x on each axis for aligned display; ROI is 32x20 native pixels",
                    orientation="Existing reference display orientation; no additional rotation",
                    source_sha256=sha(Path(__file__)), outputs=outputs)
    (args.out / "render_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(out=str(args.out.resolve()), panels=10 * len(arrays), outputs=outputs), indent=2))


if __name__ == "__main__":
    main()
