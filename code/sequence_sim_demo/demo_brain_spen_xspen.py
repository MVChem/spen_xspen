"""Brain SPEN180 and crossed-chirp xSPEN driven by finite RF/gradient waveforms.

RF encoding is integrated with KomaMRI. RF-free readout is evaluated by exact
Bloch free precession, retaining the actual time of EVERY ADC sample. Reusing
the RF response along x is exact for this x-independent RF/B0/T1/T2 setup.
An independent full-sequence KomaMRI run verifies this factorization.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from demo_epi import ROOT

# isort: split

import matplotlib.pyplot as plt
import numpy as np
from demo_brain_epi import DEFAULT_SOURCE, load_anatomy, sha256
from spen_waveforms import SOURCES, build_sequence, plot_waveforms


class Bloch:
    def __init__(self):
        import komamripy as km
        from juliacall import Main as jl

        self.km, self.jl = km, jl
        self.scanner = km.Scanner(
            limits=km.HardwareLimits(B0=3.0, B1=40e-6, Gmax=32e-3, Smax=130.0)
        )
        self.make = jl.seval("""(x,y,z,rho,b0,t1,t2) -> KomaMRI.Phantom(
            name="source_informed_brain", x=collect(x), y=collect(y), z=collect(z),
            ρ=collect(rho), Δw=2π .* collect(b0),
            T1=fill(t1,length(x)), T2=fill(t2,length(x)))""")

    def simulate(self, seq, x, y, z, rho, b0, t1, t2, step, state=False):
        obj = self.make(x, y, z, rho, b0, float(t1), float(t2))
        params = self.km.core.default_sim_params()
        params["gpu"], params["precision"] = False, "f64"
        params["return_type"] = "state" if state else "mat"
        params["Δt_rf"] = step
        result = self.km.simulate(obj, seq, self.scanner, sim_params=params)
        return np.asarray(result.xy if state else result).reshape(-1).copy()


def readout_kernel(state, y, z, zw, field, u, meta, t2):
    # No RF after prefix: this is the exact analytic solution of Bloch, not an
    # ideal quadratic/sinc encoding substitute. Finite-RF state is retained.
    decay = np.exp(-u / t2)
    if meta["family"] == "spen":
        frequency = meta["read_gradient_hz_m"] * y + field
        return (
            np.exp(-2j * np.pi * u[:, None] * frequency[None])
            * state[:, 0][None]
            * decay[:, None]
        )
    z_phase = np.exp(-2j * np.pi * meta["background_gz_hz_m"] * u[:, None] * z[None])
    kernel = (z_phase * zw[None]) @ state.T
    return kernel * np.exp(-2j * np.pi * u[:, None] * field[None]) * decay[:, None]


def forward(weight, kernel, x, kx, q, ss, pixel_weights=None):
    # Group by validated lattice indices, not decimal rounding: tiny moment
    # accumulation differences can otherwise split one frequency into two.
    frequencies = np.array([np.mean(kx[q == k]) for k in range(len(x) // ss)])
    fourier = np.exp(-2j * np.pi * frequencies[:, None] * x[None])
    pixel_weights = np.ones(ss) / ss if pixel_weights is None else pixel_weights
    spatial_weights = np.tile(pixel_weights, len(x) // ss)
    ro_object = weight @ (fourier * spatial_weights[None]).T
    signal = np.sum(kernel * ro_object[:, q].T * spatial_weights[None], axis=1)
    return signal, fourier


def reconstruct(
    signal, kernel_zero, fourier, q, n, ss, regularization, pixel_weights=None
):
    # For each actual kx, solve the y encoding using its own ADC times.
    # This includes the focus motion DURING each readout, not just line centres.
    grid = signal.reshape(n, n)
    indices = q.reshape(n, n)
    ro = np.empty((n, n), dtype=complex)
    naive_grid = np.empty((n, n), dtype=complex)
    predicted = np.empty_like(grid)
    conditions, normal_residuals = [], []
    pixel_weights = np.ones(ss) / ss if pixel_weights is None else pixel_weights
    for k in range(n):
        col = np.argmax(indices == k, axis=1)
        idx = np.arange(n) * n + col
        a = kernel_zero[idx].reshape(n, n, ss) @ pixel_weights
        b = grid[np.arange(n), col]
        u, s, vh = np.linalg.svd(a, full_matrices=False)
        solution = vh.conj().T @ (
            (s / (s * s + regularization * s[0] ** 2)) * (u.conj().T @ b)
        )
        ro[:, k] = solution
        naive_grid[:, k] = b
        predicted[np.arange(n), col] = a @ solution
        conditions.append(float(s[0] / max(s[-1], 1e-30)))
        rhs = a.conj().T @ b
        gradient = (
            a.conj().T @ (a @ solution - b) + regularization * s[0] ** 2 * solution
        )
        normal_residuals.append(
            float(np.linalg.norm(gradient) / max(np.linalg.norm(rhs), 1e-30))
        )
    f_coarse = fourier.reshape(n, n, ss) @ pixel_weights
    image = np.linalg.solve(f_coarse, ro.T).T
    ro_only = np.linalg.solve(f_coarse, naive_grid.T).T
    return (
        image,
        ro_only,
        {
            "relative_signal_residual": float(
                np.linalg.norm(predicted.ravel() - signal) / np.linalg.norm(signal)
            ),
            "regularization_relative_to_max_singular_value_squared": regularization,
            "encoding_condition_number_min_max": [min(conditions), max(conditions)],
            "maximum_normal_equation_relative_residual": max(normal_residuals),
        },
    )


def phase_check(state, y, z, meta):
    fov, thick = meta["fov_m"], meta["slice_thickness_m"]
    iy = np.abs(y) < 0.30 * fov
    if meta["family"] == "spen":
        # Remove the known post-chirp prephaser before unwrapping.
        phase = np.unwrap(
            np.angle(state[iy, 0] * np.exp(2j * np.pi * meta["r_value"] / fov * y[iy]))
        )
        coef = np.polyfit(y[iy], phase, 2)
        fit = np.polyval(coef, y[iy])
        expected = meta["expected_quadratic_rad_m2"]
        value = float(coef[0])
        detail = {"quadratic_rad_m2": value, "expected_quadratic_rad_m2": expected}
    else:
        iz = np.abs(z) < 0.20 * thick
        yy, zz = np.meshgrid(y[iy], z[iz], indexing="ij")
        demod = state[np.ix_(iy, iz)] * np.exp(
            -2j
            * np.pi
            * meta["background_gz_hz_m"]
            * meta["acquisition_window_s"]
            / 2
            * zz
        )
        phase = np.unwrap(np.unwrap(np.angle(demod), axis=0), axis=1)
        yn, zn = yy / fov, zz / thick
        design = np.column_stack(
            [
                np.ones(yy.size),
                yn.ravel(),
                zn.ravel(),
                (yn * yn).ravel(),
                (yn * zn).ravel(),
                (zn * zn).ravel(),
            ]
        )
        coef = np.linalg.lstsq(design, phase.ravel(), rcond=None)[0]
        fit = (design @ coef).reshape(phase.shape)
        expected = meta["expected_cross_term_rad_m2"]
        value = float(coef[4] / (fov * thick))
        detail = {
            "cross_term_rad_m2": value,
            "expected_cross_term_rad_m2": expected,
            "fit_coefficients_normalized_coordinates": coef.tolist(),
        }
    error = abs(value - expected) / abs(expected)
    residual = float(np.sqrt(np.mean((phase - fit) ** 2)))
    return {
        **detail,
        "coefficient_relative_error": error,
        "phase_fit_residual_rms_rad": residual,
        "passed": bool(error < 0.12 and residual < 1.0),
    }


def validate_direct(
    engine, full_seq, prefix_seq, state, y, z, field, x, u, kx, meta, args
):
    rng = np.random.default_rng(20260917)
    count = 192
    yi = rng.integers(len(y) // 5, 4 * len(y) // 5, count)
    zi = rng.integers(len(z), size=count)
    xi = rng.integers(len(x), size=count)
    rho = rng.uniform(0.2, 1, count)
    sampled = state[yi, zi]
    finer = engine.simulate(
        prefix_seq,
        np.zeros(count),
        y[yi],
        z[zi],
        np.ones(count),
        field[yi],
        args.t1,
        args.t2,
        args.rf_step_us * 0.5e-6,
        state=True,
    )
    rf_error = float(np.linalg.norm(finer - sampled) / np.linalg.norm(finer))
    raw = engine.simulate(
        full_seq,
        x[xi],
        y[yi],
        z[zi],
        rho,
        field[yi],
        args.t1,
        args.t2,
        args.rf_step_us * 1e-6,
    )
    read_freq = (
        meta["read_gradient_hz_m"] * y[yi]
        if meta["family"] == "spen"
        else meta["background_gz_hz_m"] * z[zi]
    ) + field[yi]
    expected = (
        np.exp(-2j * np.pi * (kx[:, None] * x[xi][None] + u[:, None] * read_freq[None]))
        @ (sampled * rho)
    ) * np.exp(-u / args.t2)
    direct_error = float(np.linalg.norm(raw - expected) / np.linalg.norm(raw))
    return {
        "random_spin_count": count,
        "rf_step_halving_relative_mxy_error": rf_error,
        "factorized_signal_relative_error_vs_full_sequence_bloch": direct_error,
        "no_fitted_gain_or_phase": True,
        "passed": bool(rf_error < 0.005 and direct_error < 0.005),
    }


def validate_z_integration(engine, prefix, y, u, kernels, meta, args):
    """Double z quadrature on 13 y positions, including the field-of-view edges."""
    if meta["family"] != "xspen":
        return {"applicable": False, "passed": True}
    chosen = np.linspace(0, len(y) - 1, 13).round().astype(int)
    ys = y[chosen]
    nodes, weights = np.polynomial.legendre.leggauss(args.z_nodes * 2)
    z, zw = nodes * meta["slice_thickness_m"] * 0.75, weights * 0.75
    yy, zz = np.meshgrid(ys, z, indexing="ij")
    errors = {}
    for name, df in [("ideal", 0.0), ("offset", args.offset_hz)]:
        state = engine.simulate(
            prefix,
            np.zeros(yy.size),
            yy.ravel(),
            zz.ravel(),
            np.ones(yy.size),
            np.full(yy.size, df),
            args.t1,
            args.t2,
            args.rf_step_us * 1e-6,
            state=True,
        ).reshape(yy.shape)
        fine = readout_kernel(state, ys, z, zw, np.full_like(ys, df), u, meta, args.t2)
        errors[name] = float(
            np.linalg.norm(kernels[name][:, chosen] - fine) / np.linalg.norm(fine)
        )
    return {
        "applicable": True,
        "base_z_nodes": args.z_nodes,
        "fine_z_nodes": args.z_nodes * 2,
        "y_probe_count": len(chosen),
        "relative_kernel_errors": errors,
        "passed": max(errors.values()) < 0.01,
    }


def plot_family(out, weight, results, kernel, y, meta, args):
    n, ss = args.matrix, args.oversampling
    target = weight.reshape(n, ss, n, ss).mean(axis=(1, 3))
    extent = [-meta["fov_m"] * 500, meta["fov_m"] * 500] * 2
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    vmax = float(np.quantile(target, 0.995))
    ideal, offset = results["ideal"], results["offset"]
    ro = np.abs(ideal["ro_only"])
    entries = [
        (target, "Input brain / effective M0", "gray", 0, vmax),
        (
            ro,
            "RO-only signal magnitude\n(separate signal scale)",
            "gray",
            0,
            np.quantile(ro, 0.995),
        ),
        (np.abs(ideal["image"]), "Finite-RF reconstruction: B0 = 0", "gray", 0, vmax),
        (
            np.abs(offset["image"]),
            f"Same B0=0 reconstruction: +{args.offset_hz:g} Hz",
            "gray",
            0,
            vmax,
        ),
        (
            np.abs(offset["image"]) - np.abs(ideal["image"]),
            "B0-induced magnitude difference",
            "RdBu_r",
            -vmax / 2,
            vmax / 2,
        ),
        (
            np.abs(ideal["image"]) - target,
            "Ideal reconstruction - input",
            "RdBu_r",
            -vmax / 2,
            vmax / 2,
        ),
    ]
    for index, (ax, (arr, title, cmap, lo, hi)) in enumerate(zip(axes.flat, entries)):
        panel_extent = [*extent[:2], 0, n] if index == 1 else extent
        im = ax.imshow(
            arr,
            origin="lower",
            extent=panel_extent,
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            interpolation="nearest",
        )
        ax.set(
            title=title,
            xlabel="Readout x (mm)",
            ylabel="Acquisition row" if index == 1 else "Encoding y (mm)",
        )
        if index == 1:
            ax.set_aspect("auto")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(
        f"{meta['family'].upper()} | local-source waveform adaptation | {n} x {n} | R={meta['r_value']:g}\n"
        "Bloch RF + exact RF-free acquisition; reconstruction includes assumed T2 and finite RF",
        fontsize=12,
    )
    fig.savefig(out / "brain_comparison.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    centres = np.arange(n) * n + n // 2
    ax.imshow(
        np.abs(kernel[centres]),
        origin="lower",
        aspect="auto",
        extent=[y[0] * 1000, y[-1] * 1000, 0, n],
        cmap="magma",
    )
    ax.set(
        xlabel="Object y (mm)",
        ylabel="Acquisition row",
        title="Finite-RF encoding response |K(row,y)|",
    )
    fig.savefig(out / "encoding_kernel.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--slice", type=int, default=58)
    parser.add_argument("--matrix", type=int, default=64)
    parser.add_argument(
        "--oversampling",
        type=int,
        default=8,
        help="Gauss-Legendre points per in-plane voxel axis",
    )
    parser.add_argument("--r-value", type=float, default=64)
    parser.add_argument("--z-nodes", type=int, default=384)
    parser.add_argument("--offset-hz", type=float, default=80)
    parser.add_argument("--t1", type=float, default=1)
    parser.add_argument("--t2", type=float, default=0.1)
    parser.add_argument("--rf-step-us", type=float, default=1)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument(
        "--families", nargs="+", choices=["spen", "xspen"], default=["spen", "xspen"]
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/brain_spen_xspen_260917"
    )
    args = parser.parse_args()
    n, ss = args.matrix, args.oversampling
    if n < 16 or n % 2 or ss < 1 or args.z_nodes < 16:
        parser.error("Use an even matrix >=16, oversampling >=1, z-nodes >=16")
    if min(args.t1, args.t2, args.rf_step_us, args.r_value, args.regularization) <= 0:
        parser.error("T1/T2, RF step, R and regularization must be positive")
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / "runs"):
        parser.error("Output must be in this project's runs/")
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "source"
    snapshot.mkdir()
    for path in [
        Path(__file__),
        ROOT / "spen_waveforms.py",
        ROOT / "demo_brain_epi.py",
        ROOT / "demo_epi.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]:
        shutil.copy2(path, snapshot / path.name)
    source_info = {}
    for name, path in SOURCES.items():
        source_info[name] = {"path": str(path), "sha256": sha256(path)}
        shutil.copy2(path, snapshot / path.name)
    _, _, objects, fov, provenance = load_anatomy(args.input, [args.slice], n * ss)
    original_template = objects[args.slice]
    template = original_template.reshape(n, ss, n, ss).mean(axis=(1, 3))
    # A finite-volume phantom: one M0 per target voxel, with within-voxel
    # quadrature for the oscillatory encoding. This is explicitly a matched
    # discretization demo, not a claim of recovering independent HR anatomy.
    weight = np.repeat(np.repeat(template, ss, axis=0), ss, axis=1)
    nodes, node_weights = np.polynomial.legendre.leggauss(ss)
    pixel_weights = node_weights / 2
    axis = ((np.arange(n)[:, None] + 0.5 - n / 2 + nodes[None] / 2) * fov / n).ravel()
    x, y = axis.copy(), axis.copy()
    experiment = {
        "command": [sys.executable, *sys.argv],
        "input": provenance,
        "legacy_sources": source_info,
        "assumptions": [
            "Source-informed SPEN180 and crossed-chirp families, not exact replay of any scanner binary or subject acquisition",
            "Human FOV and demonstration timing/R/amplitude chosen explicitly; no PGSE diffusion gradients",
            "Structural image intensity is effective M0; spatially uniform assumed T1/T2 and single uniform receive coil",
            "M0 is averaged onto the reconstruction grid and treated as piecewise constant; within-voxel Gauss-Legendre integration resolves encoding phase",
            "Matched forward/inverse voxel representation; image errors here are not independent high-resolution recovery evidence",
            "xSPEN uses a sinc slice excitation and an object constant through 1.5 nominal slice thicknesses",
            "B0 is spatially uniform in each condition; no object/coil/B0 variation along x in RF dynamics",
            "RF prefix integrated by KomaMRI; RF-free readout uses exact Bloch solution at actual ADC times",
            "Factorization verified against direct full-sequence KomaMRI with random 3D spins",
            "Both conditions reconstructed with B0=0 finite-RF operator; no ground truth fitting or learned prior",
        ],
    }
    (out / "metadata.json").write_text(json.dumps(experiment, indent=2) + "\n")
    engine = Bloch()
    all_checks = {}
    for kind in args.families:
        folder = out / kind
        folder.mkdir()
        seq, prefix, t_adc, u, kx, q, meta = build_sequence(kind, n, fov, args.r_value)
        seq.write(str(folder / f"{kind}.seq"))
        prefix.write(str(folder / "encoding_prefix.seq"))
        plot_waveforms(seq, t_adc, folder / "sequence.png")
        full_koma = engine.km.read_seq(str(folder / f"{kind}.seq"))
        prefix_koma = engine.km.read_seq(str(folder / "encoding_prefix.seq"))
        if kind == "xspen":
            nodes, weights = np.polynomial.legendre.leggauss(args.z_nodes)
            z, zw = nodes * meta["slice_thickness_m"] * 0.75, weights * 0.75
        else:
            z, zw = np.zeros(1), np.ones(1)
        yy, zz = np.meshgrid(y, z, indexing="ij")
        raw_results, states, kernels, timings = {}, {}, {}, {}
        for condition, df in [("ideal", 0.0), ("offset", args.offset_hz)]:
            field = np.full_like(y, df)
            print(
                f"{kind}/{condition}: RF Bloch, {yy.size} unit spins, dt={args.rf_step_us} us",
                flush=True,
            )
            start = time.perf_counter()
            state = engine.simulate(
                prefix_koma,
                np.zeros(yy.size),
                yy.ravel(),
                zz.ravel(),
                np.ones(yy.size),
                np.full(yy.size, df),
                args.t1,
                args.t2,
                args.rf_step_us * 1e-6,
                state=True,
            ).reshape(yy.shape)
            kernel = readout_kernel(state, y, z, zw, field, u, meta, args.t2)
            signal, fourier = forward(weight, kernel, x, kx, q, ss, pixel_weights)
            states[condition], kernels[condition] = state, kernel
            timings[condition] = time.perf_counter() - start
            image, ro_only, reconstruction = reconstruct(
                signal,
                kernels["ideal"],
                fourier,
                q,
                n,
                ss,
                args.regularization,
                pixel_weights,
            )
            target = weight.reshape(n, ss, n, ss).mean(axis=(1, 3))
            reconstruction["magnitude_nrmse_vs_effective_m0"] = float(
                np.linalg.norm(np.abs(image) - target) / np.linalg.norm(target)
            )
            raw_results[condition] = {
                "signal": signal,
                "image": image,
                "ro_only": ro_only,
                "metrics": reconstruction,
            }
            print(
                json.dumps(
                    {
                        "family": kind,
                        "condition": condition,
                        "seconds": timings[condition],
                        **reconstruction,
                    }
                ),
                flush=True,
            )
        print(f"{kind}: checking RF step and direct full-sequence ADC...", flush=True)
        direct = validate_direct(
            engine,
            full_koma,
            prefix_koma,
            states["ideal"],
            y,
            z,
            np.zeros_like(y),
            x,
            u,
            kx,
            meta,
            args,
        )
        phase = phase_check(states["ideal"], y, z, meta)
        direct_offset = validate_direct(
            engine,
            full_koma,
            prefix_koma,
            states["offset"],
            y,
            z,
            np.full_like(y, args.offset_hz),
            x,
            u,
            kx,
            meta,
            args,
        )
        print(f"{kind}: checking layer integration...", flush=True)
        z_check = validate_z_integration(engine, prefix_koma, y, u, kernels, meta, args)
        normal_ok = all(
            v["metrics"]["maximum_normal_equation_relative_residual"] < 1e-8
            for v in raw_results.values()
        )
        checks = {
            "direct_bloch": direct,
            "direct_bloch_offset": direct_offset,
            "phase_encoding": phase,
            "z_quadrature": z_check,
            "normal_equation_passed": normal_ok,
            "passed": direct["passed"]
            and direct_offset["passed"]
            and z_check["passed"]
            and normal_ok
            and phase["passed"]
            and abs(meta["full_refocusing_timing_error_s"]) < 1e-6,
        }
        all_checks[kind] = checks
        meta.update(
            t1_s=args.t1,
            t2_s=args.t2,
            simulation_seconds=timings,
            spatial_oversampling=ss,
            z_nodes=len(z),
            z_integration="Gauss-Legendre, total support 1.5 slice thickness"
            if kind == "xspen"
            else "z=0",
            reconstruction={k: v["metrics"] for k, v in raw_results.items()},
        )
        np.savez_compressed(
            folder / "data.npz",
            effective_m0=weight,
            x_m=x,
            y_m=y,
            z_m=z,
            z_weights=zw,
            inplane_quadrature_weights=pixel_weights,
            original_resampled_anatomy=original_template,
            adc_times_s=t_adc,
            readout_times_s=u,
            kx_cycles_m=kx,
            **{f"state_{k}": v for k, v in states.items()},
            **{f"kernel_{k}": v for k, v in kernels.items()},
            **{
                f"{a}_{k}": v[a]
                for k, v in raw_results.items()
                for a in ["signal", "image", "ro_only"]
            },
        )
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        (folder / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
        plot_family(folder, weight, raw_results, kernels["ideal"], y, meta, args)
        print(json.dumps({"family": kind, **checks}), flush=True)
    report = {
        "families": all_checks,
        "passed": all(v["passed"] for v in all_checks.values()),
    }
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results: {out}; all checks passed: {report['passed']}", flush=True)
    if not report["passed"]:
        raise RuntimeError("Validation failed; inspect saved diagnostics")


if __name__ == "__main__":
    main()
