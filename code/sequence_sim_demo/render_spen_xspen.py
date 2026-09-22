"""Render the two completed waveform-based brain demos on a common M0 scale."""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from demo_epi import ROOT

# isort: split

import matplotlib.pyplot as plt
import numpy as np
from demo_brain_spen_xspen import plot_family


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=ROOT / "runs/brain_spen_xspen_260917"
    )
    args = parser.parse_args()
    root = args.run.resolve()
    command = json.loads((root / "metadata.json").read_text())["command"]
    offset_hz = (
        float(command[command.index("--offset-hz") + 1])
        if "--offset-hz" in command
        else 80.0
    )
    fig, axes = plt.subplots(2, 5, figsize=(17, 8), layout="constrained")
    records = {}
    for row, family in enumerate(["spen", "xspen"]):
        meta = json.loads((root / family / "metadata.json").read_text())
        with np.load(root / family / "data.npz") as data:
            n = meta["matrix"][0]
            ss = meta["spatial_oversampling"]
            target = data["effective_m0"].reshape(n, ss, n, ss).mean(axis=(1, 3))
            ideal, offset = np.abs(data["image_ideal"]), np.abs(data["image_offset"])
            ro = np.abs(data["ro_only_ideal"])
            plot_family(
                root / family,
                data["effective_m0"],
                {
                    condition: {
                        "image": data[f"image_{condition}"],
                        "ro_only": data[f"ro_only_{condition}"],
                    }
                    for condition in ["ideal", "offset"]
                },
                data["kernel_ideal"],
                data["y_m"],
                meta,
                SimpleNamespace(matrix=n, oversampling=ss, offset_hz=offset_hz),
            )
            magnitude_change = float(
                np.linalg.norm(offset - ideal) / np.linalg.norm(ideal)
            )
            shifts = np.arange(-8, 9)
            errors = [
                float(
                    np.linalg.norm(offset - np.roll(ideal, int(s), axis=0))
                    / np.linalg.norm(ideal)
                )
                for s in shifts
            ]
            records[family] = {
                "b0_magnitude_relative_change": magnitude_change,
                "best_integer_y_shift_relative_to_ideal": int(
                    shifts[np.argmin(errors)]
                ),
                "residual_after_best_integer_shift": min(errors),
                "caution": "image diagnostics in this simplified phantom; not a general robustness or resolution comparison",
            }
            vmax = float(np.quantile(target, 0.995))
            extent = [-meta["fov_m"] * 500, meta["fov_m"] * 500] * 2
            panels = [
                (target, "Brain template (M0)", "gray", 0, vmax),
                (
                    ro,
                    "RO-only signal / acquisition order\n(independent signal scale)",
                    "gray",
                    0,
                    float(np.quantile(ro, 0.995)),
                ),
                (ideal, "Reconstruction, B0 = 0", "gray", 0, vmax),
                (
                    offset,
                    f"Reconstruction, B0 = {offset_hz:+g} Hz\n(B0=0 operator)",
                    "gray",
                    0,
                    vmax,
                ),
                (
                    offset - ideal,
                    "B0-induced difference",
                    "RdBu_r",
                    -vmax / 2,
                    vmax / 2,
                ),
            ]
            for col, (arr, title, cmap, lo, hi) in enumerate(panels):
                ax = axes[row, col]
                panel_extent = [*extent[:2], 0, n] if col == 1 else extent
                im = ax.imshow(
                    arr,
                    origin="lower",
                    extent=panel_extent,
                    cmap=cmap,
                    vmin=lo,
                    vmax=hi,
                    interpolation="nearest",
                )
                if row == 0:
                    ax.set_title(title, fontsize=11)
                ax.set_xlabel("Readout x (mm)")
                if col == 0:
                    ax.set_ylabel(f"{family.upper()}\nEncoding y (mm)", fontsize=12)
                elif col == 1:
                    ax.set_ylabel("Acquisition row")
                    ax.set_aspect("auto")
                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(
        f"Brain SPEN180 / crossed-chirp xSPEN | {n} x {n} finite-volume phantom | R = {meta['r_value']:g}\n"
        "Adapted from local legacy sources; finite RF Bloch encoding, actual ADC times, Tikhonov reconstruction",
        fontsize=14,
    )
    fig.savefig(root / "brain_spen_xspen_comparison.png", dpi=180)
    plt.close(fig)
    (root / "image_diagnostics.json").write_text(json.dumps(records, indent=2) + "\n")
    shutil.copy2(Path(__file__), root / "source" / Path(__file__).name)
    shutil.copy2(
        ROOT / "demo_brain_spen_xspen.py",
        root / "source" / "demo_brain_spen_xspen_rendering.py",
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
