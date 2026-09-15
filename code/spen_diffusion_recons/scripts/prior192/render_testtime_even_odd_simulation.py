"""Render the paired synthetic odd/even phase correction pilot.

Read ``full/summary.json``, ``half/summary.json`` and their per-case NPZs.
Reconstruction panels use the same [0, 1] display window and saved metrics;
the renderer does not renormalize images or select cases by performance.
Only the smooth2d injection is plotted; aggregates include all injections.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np


METHODS = ("uncorrected", "legacy_tiny", "tiny", "oracle")
METHOD_LABELS = {
    "uncorrected": "Uncorrected",
    "legacy_tiny": "Previous tiny MLP",
    "tiny": "Acquired-line tiny MLP",
    "oracle": "Known phase (oracle)",
}
DATASETS = ("ds002870", "ds005186", "ds005236", "lab_mouse")
SAMPLINGS = ("full", "half")
STYLE = {"font.family": "DejaVu Sans", "font.size": 10,
         "figure.facecolor": "white", "savefig.facecolor": "white"}


def load_results(run_dir: Path) -> dict[str, dict]:
    """Keep all paired cases and load arrays without allowing pickle data."""
    results = {}
    for sampling in SAMPLINGS:
        directory = run_dir / sampling
        summary = json.loads((directory / "summary.json").read_text())
        if summary["config"]["sampling"] != sampling:
            raise ValueError(f"Sampling mismatch in {directory / 'summary.json'}")
        identities = set()
        cases = []
        for record in summary["cases"]:
            record = dict(record)
            record["noise_sigma"] = float(summary["config"]["noise_sigma"])
            identity = (record["dataset"], record["case_index"], record["phase_kind"])
            if identity in identities:
                raise ValueError(f"Duplicate case in {sampling}: {identity}")
            identities.add(identity)
            path = directory / f"{record['name']}.npz"
            with np.load(path, allow_pickle=False) as archive:
                record["arrays"] = {key: archive[key] for key in archive.files}
            arrays = record["arrays"]
            for key, shape in (("target", (192, 192)),
                               ("phase_true", (48, 96)),
                               ("phase_weights", (48, 96)), ("mask", (96,))):
                if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                    raise ValueError(f"Invalid {key} array in {path}")
            if not np.isin(arrays["mask"], [0, 1]).all():
                raise ValueError(f"Nonbinary acquisition mask in {path}")
            for method in METHODS:
                if method not in record["methods"]:
                    raise ValueError(f"Missing method {method} in {path}")
                for prefix, shape in (("phase", (48, 96)),
                                      ("tikhonov", (192, 192)),
                                      ("diffusion", (192, 192))):
                    key = f"{prefix}_{method}"
                    required = prefix != "diffusion" or "diffusion" in record["methods"][method]
                    if not required and key not in arrays:
                        continue
                    if key not in arrays or arrays[key].shape != shape:
                        raise ValueError(f"Missing or invalid {key} in {path}")
                    if np.iscomplexobj(arrays[key]) or not np.isfinite(arrays[key]).all():
                        raise ValueError(f"Nonfinite or complex {key} in {path}")
            cases.append(record)
        cases.sort(key=lambda record: (
            DATASETS.index(record["dataset"]) if record["dataset"] in DATASETS else len(DATASETS),
            record["dataset"], record["case_index"], record["phase_kind"]))
        results[sampling] = {"config": summary["config"], "cases": cases,
                             "identities": identities}
    if results["full"]["identities"] != results["half"]["identities"]:
        raise ValueError("Full/half summaries must contain identical case/injection pairs")
    return results


def aggregate_results(results: dict[str, dict]) -> dict:
    """Average each scalar equally across cases, preserving method pairing."""
    groups = []
    for sampling in SAMPLINGS:
        cases = results[sampling]["cases"]
        phase_kinds = sorted({record["phase_kind"] for record in cases})
        for phase_kind in phase_kinds:
            paired = [record for record in cases if record["phase_kind"] == phase_kind]
            for method in METHODS:
                measurements = [record["methods"][method] for record in paired]
                group = {"sampling": sampling, "phase_kind": phase_kind,
                         "method": method, "n_cases": len(paired), "mean": {}}
                for key in ("phase_error_rms_rad", "measurement_difference_to_oracle"):
                    values = np.asarray([item[key] for item in measurements], dtype=float)
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite {key} in {sampling}/{phase_kind}/{method}")
                    group["mean"][key] = float(values.mean())
                for solver in ("tikhonov", "diffusion"):
                    available = [solver in item for item in measurements]
                    if any(available) and not all(available):
                        raise ValueError(f"Partial {solver} coverage for {sampling}/{phase_kind}/{method}")
                    if not all(available):
                        continue
                    group["mean"][solver] = {}
                    for metric in ("psnr", "ssim", "nrmse"):
                        values = np.asarray([item[solver][metric] for item in measurements], dtype=float)
                        if not np.isfinite(values).all():
                            raise ValueError(f"Nonfinite {solver}/{metric} in {sampling}/{phase_kind}/{method}")
                        group["mean"][solver][metric] = float(values.mean())
                groups.append(group)
    return {"aggregation": "Unweighted arithmetic mean across all paired cases per condition; no image rescaling.",
            "groups": groups}


def smooth_columns(results: dict[str, dict]) -> list[tuple[str, dict]]:
    columns = [(sampling, record) for sampling in SAMPLINGS
               for record in results[sampling]["cases"]
               if record["phase_kind"] == "smooth2d"]
    if not columns:
        raise ValueError("No smooth2d cases to render")
    return columns


def _make_grid(columns: list[tuple[str, dict]], nrows: int, *, phase: bool = False):
    """Explicit panel positions keep every reconstruction panel equal-sized."""
    side = 1.8
    panel_height = side * (0.60 if phase else 1.0)
    left, top, bottom = 1.82, 1.0, 0.54 if phase else 0.27
    row_gap, col_gap, group_gap = 0.09, 0.035, 0.25
    positions, cursor, previous = [], left, None
    for sampling, _ in columns:
        if previous is not None and sampling != previous:
            cursor += group_gap
        positions.append(cursor)
        cursor += side + col_gap
        previous = sampling
    width = cursor + 0.08
    height = top + nrows * panel_height + (nrows - 1) * row_gap + bottom
    fig = plt.figure(figsize=(width, height))
    axes = np.empty((nrows, len(columns)), dtype=object)
    for row in range(nrows):
        panel_top = top + row * (panel_height + row_gap)
        for column, (sampling, record) in enumerate(columns):
            ax = fig.add_axes([positions[column] / width,
                               1 - (panel_top + panel_height) / height,
                               side / width, panel_height / height])
            ax.set_axis_off()
            axes[row, column] = ax
            if row == 0:
                ax.set_title(f"{record['dataset']}\ncase {record['case_index']}", fontsize=9, pad=5)
    for sampling in SAMPLINGS:
        selected = [i for i, (value, _) in enumerate(columns) if value == sampling]
        if selected:
            center = (positions[selected[0]] + positions[selected[-1]] + side) / 2
            title = "Full PE (96/96)" if sampling == "full" else "Random 50% PE (48/96)"
            title += f", σ={columns[selected[0]][1]['noise_sigma']:.2f}"
            fig.text(center / width, 1 - 0.37 / height, title, ha="center", va="top",
                     fontsize=12, weight="bold")
    return fig, axes


def _row_label(fig, ax, label: str) -> None:
    position = ax.get_position()
    fig.text(position.x0 - 0.008, position.y0 + position.height / 2,
             label, ha="right", va="center", fontsize=10, linespacing=1.3)


def render_reconstruction(columns: list[tuple[str, dict]], solver: str, run_dir: Path) -> Path | None:
    available = {}
    for method in METHODS:
        present = [f"{solver}_{method}" in record["arrays"]
                   and solver in record["methods"][method] for _, record in columns]
        if any(present) and not all(present):
            raise ValueError(f"Cannot plot partial paired coverage for {solver}/{method}")
        available[method] = all(present)
    methods = [method for method in METHODS if available[method]]
    if not methods:
        return None
    with plt.rc_context(STYLE):
        fig, axes = _make_grid(columns, len(methods) + 1)
        try:
            fig.text(0.5, 0.991, f"Smooth 2D odd/even phase injection | {solver.capitalize()}",
                     ha="center", va="top", fontsize=13, weight="bold")
            for column, (_, record) in enumerate(columns):
                arrays = record["arrays"]
                axes[0, column].imshow(arrays["target"], cmap="gray", vmin=0, vmax=1,
                                      interpolation="nearest")
                for row, method in enumerate(methods, start=1):
                    ax = axes[row, column]
                    ax.imshow(arrays[f"{solver}_{method}"], cmap="gray", vmin=0, vmax=1,
                              interpolation="nearest")
                    metrics = record["methods"][method][solver]
                    ax.text(0.025, 0.965, f"{metrics['psnr']:.2f} / {metrics['ssim']:.3f}",
                            color="white", fontsize=9, weight="bold", ha="left", va="top",
                            transform=ax.transAxes,
                            path_effects=[patheffects.withStroke(linewidth=1.4, foreground="black")])
            _row_label(fig, axes[0, 0], "Ground truth")
            for row, method in enumerate(methods, start=1):
                _row_label(fig, axes[row, 0], METHOD_LABELS[method])
            fig.text(0.5, 0.01, "All images: 192 × 192; display range [0, 1]. Labels: PSNR (dB) / SSIM.",
                     ha="center", va="bottom", fontsize=9)
            output = run_dir / f"comparison_{solver}.png"
            fig.savefig(output, dpi=180, facecolor="white")
        finally:
            plt.close(fig)
    return output


def render_phase(columns: list[tuple[str, dict]], run_dir: Path) -> Path:
    """Use wrapped phases and hide errors on unacquired even observations."""
    rows = (("true", None), ("estimate", "legacy_tiny"), ("error", "legacy_tiny"),
            ("estimate", "tiny"), ("error", "tiny"))
    labels = ("Injected phase", "Previous tiny MLP\nEstimated phase",
              "Previous tiny MLP\nWrapped error", "Acquired-line tiny MLP\nEstimated phase",
              "Acquired-line tiny MLP\nWrapped error")
    with plt.rc_context(STYLE):
        fig, axes = _make_grid(columns, len(rows), phase=True)
        try:
            fig.text(0.5, 0.991, "Smooth 2D injection | Phase on even acquired-line coordinates",
                     ha="center", va="top", fontsize=13, weight="bold")
            cmap = plt.get_cmap("twilight_shifted").copy()
            cmap.set_bad("#d9d9d9")
            error_cmap = plt.get_cmap("RdBu_r").copy()
            error_cmap.set_bad("#d9d9d9")
            phase_image = error_image = None
            for column, (_, record) in enumerate(columns):
                arrays = record["arrays"]
                truth = arrays["phase_true"]
                acquired = np.broadcast_to(arrays["mask"][1::2, None].astype(bool), truth.shape)
                for row, (kind, method) in enumerate(rows):
                    values = truth if kind == "true" else arrays[f"phase_{method}"]
                    if kind == "error":
                        values = np.ma.array(np.angle(np.exp(1j * (values - truth))), mask=~acquired)
                        error_image = axes[row, column].imshow(values, cmap=error_cmap, vmin=-np.pi,
                            vmax=np.pi, aspect="auto", interpolation="nearest")
                        error = record["methods"][method]["phase_error_rms_rad"]
                        axes[row, column].text(0.025, 0.04, f"RMS {error:.3f} rad", fontsize=8,
                            color="black", va="bottom", transform=axes[row, column].transAxes,
                            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1})
                    else:
                        values = np.angle(np.exp(1j * values))
                        phase_image = axes[row, column].imshow(values, cmap=cmap, vmin=-np.pi,
                            vmax=np.pi, aspect="auto", interpolation="nearest")
            for row, label in enumerate(labels):
                _row_label(fig, axes[row, 0], label)
            for left, mappable, title in ((0.22, phase_image, "Phase [rad]"),
                                          (0.64, error_image, "Wrapped error [rad]")):
                cax = fig.add_axes([left, 0.045, 0.16, 0.016])
                colorbar = fig.colorbar(mappable, cax=cax, orientation="horizontal",
                                        ticks=[-np.pi, 0, np.pi])
                colorbar.ax.set_xticklabels(["−π", "0", "π"])
                colorbar.ax.tick_params(labelsize=8, pad=1)
                colorbar.set_label(title, fontsize=8, labelpad=1)
            fig.text(0.5, 0.001, "Gray: unacquired even rows; RMS labels use saved signal-weighted phase error.",
                     ha="center", va="bottom", fontsize=8)
            output = run_dir / "phase_recovery.png"
            fig.savefig(output, dpi=180, facecolor="white")
        finally:
            plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    results = load_results(run_dir)
    aggregates = aggregate_results(results)
    columns = smooth_columns(results)
    outputs = [render_reconstruction(columns, solver, run_dir)
               for solver in ("tikhonov", "diffusion")]
    outputs.append(render_phase(columns, run_dir))
    aggregate_path = run_dir / "aggregates.json"
    aggregate_path.write_text(json.dumps(aggregates, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    outputs.append(aggregate_path)
    print(json.dumps({"outputs": [str(path) for path in outputs if path is not None]}, indent=2))


if __name__ == "__main__":
    main()
