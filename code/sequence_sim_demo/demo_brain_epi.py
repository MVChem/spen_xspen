"""Use local brain anatomy as an effective M0 template for sequence-level EPI.

The input is a structural magnitude image, not quantitative PD/T1/T2/B0 maps.
All outputs, source provenance and physical checks are saved under --output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path

# Configure Julia and matplotlib before loading either backend.
from demo_epi import ROOT, analytic_reference, build_epi, plot_sequence, reconstruct

# isort: split

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

DEFAULT_SOURCE = Path(
    "/home/data2/chk/data/medical_pretraining/vjepa2_medical_full_v1/volumes/ixi/"
    "NITRC_IR_E10460/20_Guys/scans/T2-T2/resources/NIfTI/files/"
    "IXI020-Guys-0700-T2.nii.gz"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_anatomy(path, slices, size):
    original = nib.load(path)
    canonical = nib.as_closest_canonical(original)
    if len(canonical.shape) != 3:
        raise ValueError("Expected a single 3D structural magnitude volume")
    if any(s < 0 or s >= canonical.shape[2] for s in slices):
        raise ValueError(f"Slice indices must be within [0, {canonical.shape[2] - 1}]")
    volume = canonical.get_fdata(dtype=np.float64)
    if not np.isfinite(volume).all() or np.any(volume < 0):
        raise ValueError("Input must contain finite, nonnegative magnitude intensities")
    positive = volume[volume > 0]
    if positive.size == 0:
        raise ValueError("Input has no positive signal")
    scale = float(np.quantile(positive, 0.995))
    spacing = np.asarray(canonical.header.get_zooms()[:2], dtype=float) * 1e-3
    fov = float(np.max(np.asarray(canonical.shape[:2]) * spacing))
    axis = (np.arange(size) + 0.5 - size / 2) * fov / size
    x, y = np.meshgrid(axis, axis)
    coordinates = np.array(
        [
            y / spacing[1] + (canonical.shape[1] - 1) / 2,
            x / spacing[0] + (canonical.shape[0] - 1) / 2,
        ]
    )
    step_yx = fov / size / spacing[::-1]
    sigma = np.maximum((step_yx - 1) / 2, 0)
    objects = {}
    for s in slices:
        # Convert NIfTI [x,y,z] into simulation [y,x]. Preserve the acquired
        # near-axial plane; canonicalization only permutes/flips native axes.
        plane = volume[:, :, s].T / scale
        filtered = gaussian_filter(plane, sigma=sigma) if np.any(sigma > 0) else plane
        objects[s] = map_coordinates(
            filtered, coordinates, order=1, mode="nearest", prefilter=False
        )
        inside = (np.abs(x) <= canonical.shape[0] * spacing[0] / 2) & (
            np.abs(y) <= canonical.shape[1] * spacing[1] / 2
        )
        objects[s] *= inside
    provenance = {
        "source_path": str(path.resolve()),
        "source_sha256": sha256(path),
        "source_shape": list(original.shape),
        "source_affine_mm": original.affine.tolist(),
        "source_axis_codes": list(nib.aff2axcodes(original.affine)),
        "canonical_shape": list(canonical.shape),
        "canonical_affine_mm": canonical.affine.tolist(),
        "canonical_axis_codes": list(nib.aff2axcodes(canonical.affine)),
        "canonical_spacing_mm": [float(v) for v in canonical.header.get_zooms()],
        "canonical_slice_indices_zero_based": slices,
        "canonical_slice_centers_world_mm": {
            str(s): nib.affines.apply_affine(
                canonical.affine,
                [(canonical.shape[0] - 1) / 2, (canonical.shape[1] - 1) / 2, s],
            ).tolist()
            for s in slices
        },
        "intensity_normalization": "divide by whole-volume positive-voxel q99.5; no clipping",
        "normalization_value": scale,
        "simulation_grid_shape": [size, size],
        "resampling": "linear interpolation at physical voxel centers; nearest boundary extension within acquired FOV only",
        "antialias_sigma_native_yx": sigma.tolist(),
        "plane": "native near-axial plane after axis permutation/flips; no oblique-to-axial reslicing",
        "interpretation": "structural magnitude used as effective M0; not measured proton density",
    }
    return x, y, objects, fov, provenance


def validate(raw, images, weight, k_adc, t_adc, n, ss, meta, grid_error):
    reference = analytic_reference(
        weight,
        k_adc,
        t_adc,
        meta["fov_m"],
        ss,
        meta["excitation_center_s"],
        meta["t2_s"],
    )
    gain = np.vdot(reference, raw["ideal"]) / np.vdot(reference, reference)
    error = float(
        np.linalg.norm(raw["ideal"] - gain * reference) / np.linalg.norm(raw["ideal"])
    )
    ideal = np.abs(images["ideal"])
    offset = np.abs(images["uniform_b0"])
    shifts = np.arange(-8, 9)
    shift_errors = [
        float(
            np.linalg.norm(offset - np.roll(ideal, int(s), axis=0))
            / np.linalg.norm(ideal)
        )
        for s in shifts
    ]
    measured = int(shifts[np.argmin(shift_errors)])
    checks = {
        "timing_passed": meta["timing_check_passed"],
        "cartesian_grid_max_error_cycles_per_fov": grid_error,
        "ideal_signal_relative_error_vs_independent_fourier": error,
        "reference_global_complex_gain": [float(gain.real), float(gain.imag)],
        "uniform_b0_expected_y_shift_pixels": 3,
        "uniform_b0_measured_y_shift_pixels": measured,
        "uniform_b0_magnitude_error_after_expected_shift": shift_errors[
            list(shifts).index(3)
        ],
        "passed": bool(error < 0.005 and measured == 3),
    }
    return reference, checks


def plot_overview(objects, images, b0, n, ss, meta, output):
    extent = np.array([-0.5, 0.5, -0.5, 0.5]) * meta["fov_m"] * 1e3
    fig, axes = plt.subplots(
        len(objects),
        5,
        figsize=(18, 3.8 * len(objects)),
        squeeze=False,
        layout="constrained",
    )
    image_scale = max(
        float(np.quantile(np.abs(v["ideal"]), 0.995)) for v in images.values()
    )
    object_scale = max(float(np.quantile(v, 0.995)) for v in objects.values())
    for row, (sl, weight) in enumerate(objects.items()):
        target = weight.reshape(n, ss, n, ss).mean(axis=(1, 3))
        delta = np.abs(images[sl]["spatial_b0"]) - np.abs(images[sl]["ideal"])
        entries = [
            (target, "Input anatomy / effective M0", "gray", 0, object_scale),
            (np.abs(images[sl]["ideal"]), "Bloch EPI: B0 = 0", "gray", 0, image_scale),
            (
                np.abs(images[sl]["uniform_b0"]),
                f"Uniform B0: +{meta['uniform_b0_hz']:.1f} Hz\nPE shift: +3 pixels",
                "gray",
                0,
                image_scale,
            ),
            (
                np.abs(images[sl]["spatial_b0"]),
                "Bloch EPI: spatial B0",
                "gray",
                0,
                image_scale,
            ),
            (
                delta,
                "Difference: spatial B0 - ideal",
                "RdBu_r",
                -image_scale / 2,
                image_scale / 2,
            ),
        ]
        for col, (arr, title, cmap, lo, hi) in enumerate(entries):
            ax = axes[row, col]
            im = ax.imshow(
                arr,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=lo,
                vmax=hi,
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(title, fontsize=11)
            ax.set_xlabel("Local readout x (mm)")
            if col == 0:
                ax.set_ylabel(f"Slice {sl}\nLocal PE y (mm)")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(
        f"Brain anatomy -> PyPulseq -> Bloch EPI | {n} x {n} | "
        f"FOV {meta['fov_m'] * 1e3:.1f} mm | TE {meta['te_s'] * 1e3:.2f} ms\n"
        "Structural-image M0 template; assumed uniform T1/T2; prescribed B0; same scale across all EPI images",
        fontsize=14,
    )
    fig.savefig(output / "brain_comparison.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.7), layout="constrained")
    im = axs[0].imshow(
        b0, origin="lower", extent=extent, cmap="RdBu_r", vmin=-110, vmax=110
    )
    fig.colorbar(im, ax=axs[0], label="Offset (Hz)")
    axs[0].set_title("Prescribed spatial B0 (same for all slices)")
    im = axs[1].imshow(
        b0 * n * meta["echo_spacing_s"],
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-14,
        vmax=14,
    )
    fig.colorbar(im, ax=axs[1], label="PE pixels")
    axs[1].set_title("Local PE displacement approximation")
    for ax in axs:
        ax.set(xlabel="Local readout x (mm)", ylabel="Local PE y (mm)")
    fig.savefig(output / "b0_map.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--slices", type=int, nargs="+", default=[40, 58, 78])
    parser.add_argument("--matrix", type=int, default=96)
    parser.add_argument("--oversampling", type=int, default=3)
    parser.add_argument(
        "--t1", type=float, default=1.0, help="Assumed uniform T1, seconds"
    )
    parser.add_argument(
        "--t2", type=float, default=0.10, help="Assumed uniform T2, seconds"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs/brain_epi_260917")
    args = parser.parse_args()
    n, ss = args.matrix, args.oversampling
    if n < 16 or n % 2 or ss < 1 or args.t1 <= 0 or args.t2 <= 0:
        parser.error("Use an even matrix >=16, oversampling >=1, and positive T1/T2")
    if len(set(args.slices)) != len(args.slices):
        parser.error("Slice indices must be unique")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs"):
        parser.error("Output must be under this project's runs/")
    x, y, objects, fov, provenance = load_anatomy(args.input, args.slices, n * ss)
    seq, k_adc, t_adc, meta = build_epi(n, fov)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source"
    snapshot.mkdir()
    for path in [
        Path(__file__),
        ROOT / "demo_epi.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]:
        shutil.copy2(path, snapshot / path.name)
    seq_file = output / "brain_epi.seq"
    seq.write(str(seq_file))
    plot_sequence(seq, k_adc, t_adc, output)
    xx, yy = x / (fov / 2), y / (fov / 2)
    b0 = 105 * np.exp(-((xx - 0.15) ** 2 / 0.28 + (yy + 0.55) ** 2 / 0.12))
    b0 -= 55 * np.exp(-((xx + 0.30) ** 2 / 0.22 + (yy - 0.30) ** 2 / 0.20))
    meta.update(
        t1_s=args.t1,
        t2_s=args.t2,
        oversampling_per_axis=ss,
        uniform_b0_hz=3 / (n * meta["echo_spacing_s"]),
        b0_range_hz=[float(b0.min()), float(b0.max())],
        source=provenance,
        command=[sys.executable, *sys.argv],
        source_code_sha256={
            p.name: sha256(p) for p in [Path(__file__), ROOT / "demo_epi.py"]
        },
        versions={
            p: importlib.metadata.version(p)
            for p in ["numpy", "scipy", "nibabel", "pypulseq", "komamripy", "juliacall"]
        },
        assumptions=[
            "Structural T2-weighted intensity serves as effective M0, not measured PD; original tissue contrast remains embedded",
            "Uniform chosen T1 and T2, no tissue segmentation or fitted relaxation maps",
            "Prescribed identical B0 field in each native plane; no measured field map or susceptibility solver",
            "2D stationary spins; non-slice-selective RF; each slice simulated independently",
            "Single uniform receive coil, no added noise, diffusion, motion or hardware imperfections",
            "No empirical T2*; explicit T2 decay and position-dependent B0 phase evolution",
            "Same Cartesian reconstruction for all cases, no distortion correction",
        ],
        scanner_limits={
            "b0_t": 3.0,
            "b1_max_t": 40e-6,
            "gmax_t_m": 0.032,
            "smax_t_m_s": 130.0,
        },
    )
    (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"Loaded {args.input.name}; {len(objects)} slices; {n}x{n}; TE {meta['te_s'] * 1000:.2f} ms",
        flush=True,
    )
    import komamripy as km
    from juliacall import Main as jl

    scanner = km.Scanner(
        limits=km.HardwareLimits(B0=3.0, B1=40e-6, Gmax=32e-3, Smax=130.0)
    )
    koma_seq = km.read_seq(str(seq_file))
    make_obj = jl.seval("""(x,y,rho,b0,t1,t2) -> KomaMRI.Phantom(
        name="brain_anatomical_M0", x=collect(x), y=collect(y), ρ=collect(rho),
        Δw=2π .* collect(b0), T1=fill(t1,length(x)), T2=fill(t2,length(x)))""")
    fields = {
        "ideal": np.zeros_like(b0),
        "uniform_b0": np.full_like(b0, meta["uniform_b0_hz"]),
        "spatial_b0": b0,
    }
    all_images, validations, timings, spins = {}, {}, {}, {}
    for sl, weight in objects.items():
        mask = weight > 0
        raw, images, kspaces = {}, {}, {}
        spins[str(sl)] = int(mask.sum())
        for name, field in fields.items():
            obj = make_obj(
                x[mask], y[mask], weight[mask] / ss**2, field[mask], args.t1, args.t2
            )
            params = km.core.default_sim_params()
            params["return_type"], params["gpu"], params["precision"] = (
                "mat",
                False,
                "f64",
            )
            params["Δt_rf"] = 1e-6
            start = time.perf_counter()
            print(f"Simulating slice {sl}, {name}: {mask.sum()} spins", flush=True)
            result = km.simulate(obj, koma_seq, scanner, sim_params=params)
            raw[name] = np.asarray(result).reshape(-1).copy()
            timings[f"slice_{sl}/{name}"] = time.perf_counter() - start
            if raw[name].size != n * n or not np.isfinite(raw[name]).all():
                raise RuntimeError("Invalid Bloch output")
            images[name], kspaces[name], grid_error = reconstruct(
                raw[name], k_adc, n, fov
            )
        reference, checks = validate(
            raw, images, weight, k_adc, t_adc, n, ss, meta, grid_error
        )
        validations[str(sl)] = checks
        all_images[sl] = images
        np.savez_compressed(
            output / f"slice_{sl:03d}.npz",
            x_m=x,
            y_m=y,
            effective_m0=weight,
            b0_hz=b0,
            adc_times_s=t_adc,
            k_adc_cycles_m=k_adc,
            reference_signal=reference,
            **{f"signal_{k}": v for k, v in raw.items()},
            **{f"image_{k}": v for k, v in images.items()},
            **{f"kspace_{k}": v for k, v in kspaces.items()},
        )
        print(json.dumps({"slice": sl, **checks}), flush=True)
    meta.update(
        simulation_seconds=timings,
        spin_count_per_slice=spins,
        julia_version=str(jl.seval("VERSION")),
        koma_version=str(jl.seval("pkgversion(KomaMRI)")),
    )
    checks = {
        "slices": validations,
        "passed": all(v["passed"] for v in validations.values()),
    }
    (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (output / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    plot_overview(objects, all_images, b0, n, ss, meta, output)
    print(f"Results: {output}; validation passed: {checks['passed']}", flush=True)
    if not checks["passed"]:
        raise RuntimeError("Physics validation failed; inspect validation.json")


if __name__ == "__main__":
    main()
