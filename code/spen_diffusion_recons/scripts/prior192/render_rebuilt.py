"""Render GT and reconstruction comparisons with equal-sized image panels.

Simulation NPZ files contain ``degraded``, ``phase_inva``, ``tikhonov`` and
``diffusion`` arrays with shape (2, N, H, W), ``labels`` with shape (N,),
``noise_sigma`` with shape (2,), and ``mask`` with shape (2, PE).  The first
condition uses full PE and the second uses random 50% PE.  Spatial dimensions
are 96 x 96 for input and 192 x 192 for reconstruction. Simulation also
requires ``target`` with shape (N, 192, 192), displayed as the first row.

Real-data NPZ files use the same image keys with shape (N, H, W), ``labels``
with shape (N,), and optionally ``fov_mm`` with shape (N,).  Consecutive cases
with the same field of view share a heading.  The column order is preserved.

Images are already display intensities: neither renderer normalizes them.
Every panel uses the same grayscale window [0, 1].
All panels have the same displayed size; labels retain the native grid sizes.

Run without arguments to redraw the approved two PNGs in the default run's
figures_260915 directory. All plotting inputs are read from that directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import patheffects
import numpy as np
from scipy.ndimage import gaussian_filter


IMAGE_KEYS = ("degraded", "phase_inva", "tikhonov", "diffusion")
DEFAULT_FIGURES = (Path(__file__).resolve().parents[2]
                   / "runs/rodent192_spen2x_260914/figures_260915")
ROW_LABELS = {
    "target": "Ground truth (GT)",
    "degraded": "Degraded input",
    "phase_inva": "Phase map\n+ InvA",
    "tikhonov": "Tikhonov",
    "diffusion": "Diffusion prior",
}
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 11,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def _metric_labels(predictions: np.ndarray, targets: np.ndarray) -> list[str]:
    """Use the saved evaluation convention: unit range, Gaussian SSIM, 5px crop."""
    labels = []
    for pred, target in zip(predictions, targets):
        pred = np.asarray(pred, dtype=np.float32).clip(0, 1)
        target = np.asarray(target, dtype=np.float32).clip(0, 1)
        mse = float(np.mean((pred - target) ** 2))
        mu_p = gaussian_filter(pred, 1.5, truncate=3.5)
        mu_g = gaussian_filter(target, 1.5, truncate=3.5)
        var_p = gaussian_filter(pred * pred, 1.5, truncate=3.5) - mu_p ** 2
        var_g = gaussian_filter(target * target, 1.5, truncate=3.5) - mu_g ** 2
        cov = gaussian_filter(pred * target, 1.5, truncate=3.5) - mu_p * mu_g
        ssim_map = ((2 * mu_p * mu_g + .01 ** 2) * (2 * cov + .03 ** 2)
                    / ((mu_p ** 2 + mu_g ** 2 + .01 ** 2) * (var_p + var_g + .03 ** 2)))
        psnr = -10 * np.log10(max(mse, 1e-12))
        labels.append(f"{psnr:.2f} / {ssim_map[5:-5, 5:-5].mean():.3f}")
    return labels


def _load_images(path: str | Path, simulation: bool) -> dict[str, np.ndarray]:
    """Load only required arrays and reject malformed display data."""
    with np.load(path, allow_pickle=False) as archive:
        required = (*IMAGE_KEYS, "labels")
        if simulation:
            required += ("noise_sigma", "mask", "target")
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"Missing NPZ keys: {', '.join(missing)}")
        arrays = {key: np.asarray(archive[key]) for key in required}
        if "fov_mm" in archive:
            arrays["fov_mm"] = np.asarray(archive["fov_mm"])

    labels = arrays["labels"]
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a nonempty one-dimensional array")
    expected_prefix = (2, labels.size) if simulation else (labels.size,)
    for key in IMAGE_KEYS:
        values = arrays[key]
        if values.ndim != len(expected_prefix) + 2:
            raise ValueError(f"{key} must have shape {expected_prefix} + (H, W)")
        if values.shape[:-2] != expected_prefix or min(values.shape[-2:]) < 1:
            raise ValueError(f"{key} has an incompatible shape: {values.shape}")
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise ValueError(f"{key} must contain real-valued display intensities")
        if not np.isfinite(values).all():
            raise ValueError(f"{key} contains non-finite intensities")
        size = 96 if key == "degraded" else 192
        if values.shape[-2:] != (size, size):
            raise ValueError(f"{key} must retain its {size} x {size} grid")

    if simulation:
        target = arrays["target"]
        if (target.shape != (labels.size, 192, 192)
                or not np.issubdtype(target.dtype, np.number)
                or np.iscomplexobj(target) or not np.isfinite(target).all()):
            raise ValueError("target must contain finite real GT images of shape (N, 192, 192)")
        sigma = arrays["noise_sigma"]
        if sigma.shape != (2,) or not np.isfinite(sigma).all() or (sigma < 0).any():
            raise ValueError("noise_sigma must contain two finite nonnegative values")
        mask = arrays["mask"]
        if mask.ndim != 2 or mask.shape[0] != 2 or mask.shape[1] == 0:
            raise ValueError("mask must have shape (2, PE)")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("mask must be binary")
        if not mask[0].all() or 2 * int(mask[1].sum()) != mask.shape[1]:
            raise ValueError("Expected full PE first and exactly 50% PE second")
    elif "fov_mm" in arrays and arrays["fov_mm"].shape != labels.shape:
        raise ValueError("fov_mm must have one entry per case")
    return arrays


def _draw_grid(
    images: list[np.ndarray],
    row_keys: list[str],
    labels: list[str],
    groups: list[tuple[int, int, str]],
    output_stem: str | Path,
    annotations: dict[str, list[str]] | None = None,
) -> dict[str, Path]:
    """Use equal panel sizes while preserving each input array's native grid."""
    n_columns = len(labels)
    group_starts = {start for start, _, _ in groups if start > 0}
    side, gap, group_gap, left = 1.92, 0.055, 0.32, 1.90
    positions = []
    cursor = left
    for column in range(n_columns):
        if column in group_starts:
            cursor += group_gap - gap
        positions.append(cursor)
        cursor += side + gap

    output_stem = Path(output_stem)
    if output_stem.suffix.lower() in {".png", ".pdf", ".svg"}:
        output_stem = output_stem.with_suffix("")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure_width = cursor - gap + 0.12
    top_margin = 0.78 if groups else 0.42
    row_sides = [side for _ in images]
    row_tops, cursor = [], top_margin
    for row_side in row_sides:
        row_tops.append(cursor)
        cursor += row_side + 0.075
    figure_height = cursor + 0.02
    annotations = annotations or {}
    metric_font = None
    if annotations:
        metric_font = font_manager.FontProperties(fname=font_manager.findfont(
            font_manager.FontProperties(family="Times New Roman", weight="bold"),
            fallback_to_default=False), size=13, weight="bold")
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(figure_width, figure_height), layout=None)
        try:
            for row, values in enumerate(images):
                row_side, top = row_sides[row], row_tops[row]
                for column in range(n_columns):
                    panel_left = positions[column] + (side - row_side) / 2
                    ax = fig.add_axes([panel_left / figure_width,
                        1 - (top + row_side) / figure_height,
                        row_side / figure_width, row_side / figure_height])
                    ax.imshow(
                        values[column],
                        cmap="gray",
                        vmin=0,
                        vmax=1,
                        interpolation="nearest",
                        aspect="equal",
                    )
                    ax.set_axis_off()
                    if row_keys[row] in annotations:
                        ax.text(0.035, 0.965, annotations[row_keys[row]][column],
                                transform=ax.transAxes, ha="left", va="top",
                                color="white", fontproperties=metric_font,
                                path_effects=[patheffects.withStroke(linewidth=0.8,
                                                                    foreground="black")])
                    if row == 0:
                        ax.set_title(labels[column], pad=7)
                size = values.shape[-1]
                fig.text(
                    (left - 0.14) / figure_width,
                    1 - (top + row_side / 2) / figure_height,
                    f"{ROW_LABELS[row_keys[row]]}\n{size} × {size}",
                    ha="right",
                    va="center",
                    fontsize=11.5,
                    linespacing=1.35,
                )
            for start, end, heading in groups:
                fig.text(
                    (positions[start] + positions[end - 1] + side) / 2 / figure_width,
                    1 - 0.12 / figure_height,
                    heading,
                    ha="center",
                    va="top",
                    fontsize=13,
                    fontweight="bold",
                )
            paths = {}
            for extension in ("png",):
                path = output_stem.parent / f"{output_stem.name}.{extension}"
                fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
                paths[extension] = path
        finally:
            plt.close(fig)
    return paths


def render_simulation(npz_path: str | Path, output_stem: str | Path) -> dict[str, Path]:
    """Save PNG with full PE on the left and random 50% PE on the right."""
    arrays = _load_images(npz_path, simulation=True)
    labels = [str(label) for label in arrays["labels"]]
    n_cases = len(labels)
    images = [
        np.concatenate([arrays[key][0], arrays[key][1]], axis=0)
        for key in IMAGE_KEYS
    ]
    images.insert(0, np.concatenate([arrays["target"], arrays["target"]], axis=0))
    sigma_full, sigma_half = arrays["noise_sigma"]
    groups = [
        (0, n_cases, f"Full PE · σ = {sigma_full:g}"),
        (n_cases, 2 * n_cases, f"Random 50% PE · σ = {sigma_half:g}"),
    ]
    target = np.concatenate([arrays["target"], arrays["target"]], axis=0)
    annotations = {key: _metric_labels(
        np.concatenate([arrays[key][0], arrays[key][1]], axis=0), target)
        for key in ("phase_inva", "tikhonov", "diffusion")}
    # Match the nearest-neighbor display enlargement; retain native 96 arrays.
    degraded = np.concatenate([arrays["degraded"][0], arrays["degraded"][1]], axis=0)
    degraded192 = np.repeat(np.repeat(degraded, 2, axis=-2), 2, axis=-1)
    annotations["degraded"] = _metric_labels(degraded192, target)
    return _draw_grid(images, ["target", *IMAGE_KEYS], labels + labels,
                      groups, output_stem, annotations=annotations)


def render_real(npz_path: str | Path, output_stem: str | Path) -> dict[str, Path]:
    """Save the same four reconstruction rows for real-data cases."""
    arrays = _load_images(npz_path, simulation=False)
    labels = [str(label) for label in arrays["labels"]]
    groups = []
    if "fov_mm" in arrays:
        fov = arrays["fov_mm"]
        start = 0
        for stop in range(1, len(labels) + 1):
            if stop == len(labels) or fov[stop] != fov[start]:
                value = (
                    f"{fov[start]:g}"
                    if np.issubdtype(fov.dtype, np.number)
                    else str(fov[start])
                )
                groups.append((start, stop, f"FOV {value} mm"))
                start = stop
    return _draw_grid([arrays[key] for key in IMAGE_KEYS], list(IMAGE_KEYS),
                      labels, groups, output_stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", nargs="?", choices=("all", "simulation", "real"), default="all")
    parser.add_argument("npz_path", nargs="?", type=Path)
    parser.add_argument("output_stem", nargs="?", type=Path, help="Output path without an extension")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES,
                        help="Directory containing saved arrays and the output PNGs")
    args = parser.parse_args()
    if (args.npz_path is None) != (args.output_stem is None):
        parser.error("Pass both npz_path and output_stem, or omit both")
    if args.kind == "all" and args.npz_path is not None:
        parser.error("For all figures use --figures-dir instead of single-file paths")
    kinds = ("simulation", "real") if args.kind == "all" else (args.kind,)
    for kind in kinds:
        render = render_simulation if kind == "simulation" else render_real
        stem = "figure1_simulation" if kind == "simulation" else "figure2_real"
        source = args.npz_path or args.figures_dir / f"{kind}.npz"
        output = args.output_stem or args.figures_dir / stem
        for path in render(source, output).values():
            print(path)


if __name__ == "__main__":
    main()
