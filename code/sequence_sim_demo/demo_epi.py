"""PyPulseq waveforms -> KomaMRI Bloch simulation -> EPI reconstruction.

Run with: uv run python demo_epi.py --output runs/epi_260917
The phantom is synthetic. No scanner data or downloaded phantom is required.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("JULIA_DEPOT_PATH", str(ROOT / ".julia"))
os.environ.setdefault("JULIA_NUM_THREADS", "8")
os.environ.setdefault("JULIA_NUM_PRECOMPILE_TASKS", "4")
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp


def build_epi(n: int = 64, fov: float = 0.22):
    """Single-shot, non-slice-selective GE-EPI, with flat-top ADC sampling."""
    system = pp.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=20e-6,
        adc_dead_time=10e-6,
        grad_raster_time=10e-6,
    )
    seq = pp.Sequence(system)
    rf = pp.make_block_pulse(
        np.pi / 2,
        duration=200e-6,
        system=system,
        use="excitation",
        delay=system.rf_dead_time,
    )
    # Choose a dwell on the ADC raster and a flat time on the gradient raster.
    # 10 us also keeps all tested even matrix sizes on the gradient raster.
    dwell = 10e-6
    gx = pp.make_trapezoid(
        "x",
        amplitude=1 / (fov * dwell),
        flat_time=n * dwell,
        rise_time=100e-6,
        system=system,
    )
    adc = pp.make_adc(n, dwell=dwell, delay=gx.rise_time, system=system)
    pre_x = pp.make_trapezoid("x", area=-gx.area / 2, duration=1e-3, system=system)
    pre_y = pp.make_trapezoid("y", area=-n / (2 * fov), duration=1e-3, system=system)
    blip = pp.make_trapezoid("y", area=1 / fov, duration=100e-6, system=system)
    seq.add_block(rf)
    seq.add_block(pre_x, pre_y)
    for row in range(n):
        read = pp.make_trapezoid(
            "x",
            amplitude=(-1) ** row * gx.amplitude,
            flat_time=gx.flat_time,
            rise_time=gx.rise_time,
            system=system,
        )
        seq.add_block(read, adc)
        if row < n - 1:
            seq.add_block(blip)
    seq.set_definition("Name", "demo_ge_epi")
    seq.set_definition("FOV", [fov, fov, 0.005])
    seq.set_definition("Nx", n)
    seq.set_definition("Ny", n)
    ok, errors = seq.check_timing()
    if not ok:
        raise RuntimeError(f"Pulseq timing check failed: {errors}")
    k_adc, _, t_exc, _, t_adc = seq.calculate_kspace()
    echo_spacing = pp.calc_duration(gx) + pp.calc_duration(blip)
    return (
        seq,
        k_adc,
        np.asarray(t_adc),
        {
            "matrix": [n, n],
            "fov_m": fov,
            "adc_dwell_s": dwell,
            "echo_spacing_s": echo_spacing,
            "excitation_center_s": float(t_exc[0]),
            "te_s": float(t_adc.reshape(n, n)[n // 2].mean() - t_exc[0]),
            "duration_s": float(seq.duration()[0]),
            "rf_duration_s": 200e-6,
            "rf_flip_deg": 90.0,
            "timing_check_passed": bool(ok),
            "sequence_type": "single-shot non-slice-selective gradient-echo EPI",
            "sampling": "flat-top ADC, bipolar x readout, y blips; no ramp sampling",
        },
    )


def phantom(n: int, oversampling: int, fov: float):
    hi = n * oversampling
    axis = (np.arange(hi) + 0.5 - hi / 2) * fov / hi
    x, y = np.meshgrid(axis, axis)
    xx, yy = x / (fov / 2), y / (fov / 2)
    pd = np.zeros_like(x)
    pd[(xx / 0.78) ** 2 + (yy / 0.88) ** 2 < 1] = 0.65
    pd[(xx / 0.69) ** 2 + ((yy + 0.025) / 0.79) ** 2 < 1] = 0.85
    pd[((xx + 0.27) / 0.18) ** 2 + ((yy - 0.13) / 0.30) ** 2 < 1] = 0.35
    pd[((xx - 0.25) / 0.14) ** 2 + ((yy - 0.08) / 0.24) ** 2 < 1] = 0.25
    pd[((xx + 0.12) / 0.12) ** 2 + ((yy + 0.48) / 0.10) ** 2 < 1] = 1.25
    pd[((xx - 0.32) / 0.09) ** 2 + ((yy + 0.37) / 0.12) ** 2 < 1] = 1.05
    # A prescribed smooth frequency map; no susceptibility field solver is used.
    b0 = 105 * np.exp(-((xx - 0.15) ** 2 / 0.28 + (yy + 0.55) ** 2 / 0.12))
    b0 -= 55 * np.exp(-((xx + 0.30) ** 2 / 0.22 + (yy - 0.30) ** 2 / 0.20))
    return x, y, pd, b0


def reconstruct(raw, k_adc, n, fov):
    """Map samples by their actual coordinates, including reverse readouts."""
    qx, qy = k_adc[:2] * fov
    ix = np.rint(qx - 0.5 + n / 2).astype(int)
    iy = np.rint(qy + n / 2).astype(int)
    if np.any(ix < 0) or np.any(ix >= n) or np.any(iy < 0) or np.any(iy >= n):
        raise ValueError("ADC trajectory lies outside the expected Cartesian grid")
    if np.unique(iy * n + ix).size != n * n:
        raise ValueError("ADC trajectory does not cover every Cartesian sample once")
    grid_error = max(
        np.max(np.abs(qx - (ix - n / 2 + 0.5))), np.max(np.abs(qy - (iy - n / 2)))
    )
    if grid_error > 1e-6:
        raise ValueError(f"ADC samples are off the expected lattice: {grid_error}")
    kspace = np.empty((n, n), dtype=np.complex128)
    kspace[iy, ix] = np.asarray(raw).reshape(-1)
    ky, kx = np.meshgrid(
        np.arange(n) - n / 2, np.arange(n) - n / 2 + 0.5, indexing="ij"
    )
    # Reconstruct at voxel centers, x,y = (j + 1/2 - N/2)*FOV/N.
    centered = kspace * np.exp(2j * np.pi * (kx + ky) * 0.5 / n)
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(centered)))
    image *= np.exp(2j * np.pi * 0.5 * (np.arange(n) - n / 2) / n)[None, :]
    return image, kspace, float(grid_error)


def analytic_reference(pd, k_adc, t_adc, fov, oversampling, excitation_center, t2):
    """Independent hard-pulse Fourier reference for the B0=0 validation case.

    Spins sit at subvoxel centers. Fractional frequency and spatial origins are
    handled explicitly; this reference does not call KomaMRI's signal operator.
    """
    hi = pd.shape[0]
    fractional_x = np.exp(-2j * np.pi * 0.5 * np.arange(hi) / hi)
    spectrum = np.fft.fft2(pd / oversampling**2 * fractional_x[None, :])
    qx, qy = k_adc[:2] * fov
    ix = np.rint(qx - 0.5).astype(int) % hi
    iy = np.rint(qy).astype(int) % hi
    origin = (0.5 - hi / 2) / hi
    reference = spectrum[iy, ix] * np.exp(-2j * np.pi * (qx + qy) * origin)
    return reference * np.exp(-(t_adc - excitation_center) / t2)


def plot_sequence(seq, k_adc, t_adc, output):
    waves, *_ = seq.waveforms_and_times(append_RF=True)
    fig, axs = plt.subplots(5, 1, figsize=(11, 8), sharex=True, layout="constrained")
    for ax, w, label in zip(
        axs[:4],
        [waves[3], *waves[:3]],
        ["RF amplitude (Hz)", "Gx (mT/m)", "Gy (mT/m)", "Gz (mT/m)"],
    ):
        if np.size(w):
            values = (
                np.abs(w[1])
                if label.startswith("RF")
                else w[1].real / seq.system.gamma * 1e3
            )
            ax.plot(w[0].real * 1e3, values, lw=0.9)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axs[4].plot(t_adc * 1e3, np.ones_like(t_adc), "|", markersize=4)
    axs[4].set_ylabel("ADC")
    axs[4].set_xlabel("Time (ms)")
    axs[0].set_title("PyPulseq GE-EPI: RF / gradients / acquisition times")
    fig.savefig(output / "sequence.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.plot(k_adc[0], k_adc[1], lw=0.5, alpha=0.6)
    p = ax.scatter(k_adc[0], k_adc[1], c=t_adc * 1e3, s=2, cmap="viridis")
    fig.colorbar(p, ax=ax, label="ADC time (ms)")
    ax.set(xlabel="kx (cycles/m)", ylabel="ky (cycles/m)", title="EPI ADC trajectory")
    ax.set_aspect("equal")
    fig.savefig(output / "trajectory.png", dpi=160)
    plt.close(fig)


def plot_results(pd, b0, images, raw, t_adc, n, oversampling, meta, output):
    target = pd.reshape(n, oversampling, n, oversampling).mean(axis=(1, 3))
    extent = np.array([-0.5, 0.5, -0.5, 0.5]) * meta["fov_m"] * 1e3
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 8.5), layout="constrained")
    scale = np.max(np.abs(images["ideal"]))
    panels = [
        (target, "Synthetic object (proton density)", "gray", 0, pd.max()),
        (np.abs(images["ideal"]), "Bloch EPI: B0 = 0", "gray", 0, scale),
        (
            np.abs(images["spatial_b0"]),
            "Bloch EPI: spatial B0 offset",
            "gray",
            0,
            scale,
        ),
        (b0, "Prescribed B0 offset (Hz)", "RdBu_r", -110, 110),
        (
            np.abs(images["uniform_b0"]),
            f"Uniform B0: {meta['uniform_b0_hz']:.1f} Hz\nexpected PE shift: +3 pixels",
            "gray",
            0,
            scale,
        ),
        (
            np.abs(images["spatial_b0"]) - np.abs(images["ideal"]),
            "Magnitude difference: spatial B0 minus ideal",
            "RdBu_r",
            -scale / 2,
            scale / 2,
        ),
    ]
    for ax, (arr, title, cmap, vmin, vmax) in zip(axs.flat, panels):
        im = ax.imshow(
            arr,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set(title=title, xlabel="Readout x (mm)", ylabel="Phase encode y (mm)")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(
        f"Sequence-level EPI simulation | {n} x {n} | TE {meta['te_s'] * 1e3:.2f} ms | "
        f"echo spacing {meta['echo_spacing_s'] * 1e3:.2f} ms\n"
        "PyPulseq waveforms + KomaMRI Bloch evolution; common scale for all reconstructions",
        fontsize=13,
    )
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    fig, axs = plt.subplots(2, 1, figsize=(11, 6), layout="constrained")
    for name, sig in raw.items():
        axs[0].plot(t_adc * 1e3, np.abs(sig), lw=0.8, label=name)
        sel = np.arange(n * (n // 2 - 1), n * (n // 2 + 2))
        axs[1].plot(t_adc[sel] * 1e3, sig[sel].real, lw=1, label=name)
    axs[0].set(title="Raw ADC magnitude", ylabel="Signal (a.u.)")
    axs[1].set(
        title="Raw ADC real part near k-space center",
        ylabel="Signal (a.u.)",
        xlabel="Time (ms)",
    )
    for ax in axs:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(output / "signals.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/epi_260917"))
    parser.add_argument("--matrix", type=int, default=64)
    parser.add_argument("--oversampling", type=int, default=4)
    parser.add_argument(
        "--gpu", action="store_true", help="Requires a working Koma CUDA backend"
    )
    args = parser.parse_args()
    n, ss = args.matrix, args.oversampling
    if n < 16 or n % 2 or ss < 1:
        parser.error("matrix must be even and >=16; oversampling must be >=1")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seq, k_adc, t_adc, meta = build_epi(n)
    seq_file = output / "epi.seq"
    seq.write(str(seq_file))
    plot_sequence(seq, k_adc, t_adc, output)
    x, y, pd, b0 = phantom(n, ss, meta["fov_m"])
    meta.update(
        oversampling_per_axis=ss,
        t1_s=1.0,
        t2_s=0.10,
        uniform_b0_hz=3 / (n * meta["echo_spacing_s"]),
        phantom="procedural asymmetric ellipses; uniform T1 and T2",
        coil="single uniform receive sensitivity",
        noise="none",
        gpu=args.gpu,
    )
    print("Sequence generated; loading KomaMRI...", flush=True)
    import komamripy as km
    from juliacall import Main as jl

    if args.gpu:
        km.load_cuda()
    scanner = km.Scanner(
        limits=km.HardwareLimits(B0=3.0, B1=40e-6, Gmax=32e-3, Smax=130.0)
    )
    koma_seq = km.read_seq(str(seq_file))
    # Materialize native Julia vectors; do not rely on keyword conversion of PyArrays.
    make_obj = jl.seval("""(x,y,rho,b0,t1,t2) -> KomaMRI.Phantom(
        name="synthetic_ellipses", x=collect(x), y=collect(y), ρ=collect(rho),
        Δw=2π .* collect(b0), T1=fill(t1,length(x)), T2=fill(t2,length(x)))""")
    mask = pd > 0
    fields = {
        "ideal": np.zeros_like(pd),
        "uniform_b0": np.full_like(pd, meta["uniform_b0_hz"]),
        "spatial_b0": b0,
    }
    raw, images, kspaces, timings = {}, {}, {}, {}
    for name, field in fields.items():
        obj = make_obj(
            x[mask], y[mask], pd[mask] / ss**2, field[mask], meta["t1_s"], meta["t2_s"]
        )
        params = km.core.default_sim_params()
        params["return_type"] = "mat"
        params["gpu"] = args.gpu
        params["precision"] = "f64"
        params["Δt_rf"] = 1e-6
        start = time.perf_counter()
        print(f"Simulating {name}: {mask.sum()} spins...", flush=True)
        result = km.simulate(obj, koma_seq, scanner, sim_params=params)
        raw[name] = np.asarray(result).reshape(-1).copy()
        timings[name] = time.perf_counter() - start
        if raw[name].size != n * n or not np.isfinite(raw[name]).all():
            raise RuntimeError(
                f"Invalid output from KomaMRI: {name}, {raw[name].shape}"
            )
        images[name], kspaces[name], grid_error = reconstruct(
            raw[name], k_adc, n, meta["fov_m"]
        )
        print(f"Finished {name} in {timings[name]:.2f} s", flush=True)

    reference = analytic_reference(
        pd, k_adc, t_adc, meta["fov_m"], ss, meta["excitation_center_s"], meta["t2_s"]
    )
    gain = np.vdot(reference, raw["ideal"]) / np.vdot(reference, reference)
    reference_error = np.linalg.norm(raw["ideal"] - gain * reference) / np.linalg.norm(
        raw["ideal"]
    )
    # For increasing ky and exp(-i*2*pi*df*t), a positive df shifts +y.
    shifts = np.arange(-8, 9)
    ideal_mag = np.abs(images["ideal"])
    offset_mag = np.abs(images["uniform_b0"])
    shift_errors = [
        float(
            np.linalg.norm(offset_mag - np.roll(ideal_mag, int(s), axis=0))
            / np.linalg.norm(ideal_mag)
        )
        for s in shifts
    ]
    measured_shift = int(shifts[np.argmin(shift_errors)])
    checks = {
        "timing_passed": True,
        "cartesian_grid_max_error_cycles_per_fov": grid_error,
        "ideal_signal_relative_error_vs_independent_fourier": float(reference_error),
        "reference_global_complex_gain": [float(gain.real), float(gain.imag)],
        "uniform_b0_expected_y_shift_pixels": 3,
        "uniform_b0_measured_y_shift_pixels": measured_shift,
        "uniform_b0_magnitude_error_after_expected_shift": shift_errors[
            list(shifts).index(3)
        ],
        "passed": bool(reference_error < 0.005 and measured_shift == 3),
    }
    meta.update(
        spin_count=int(mask.sum()),
        simulation_seconds=timings,
        scanner_limits={
            "b0_t": 3.0,
            "b1_max_t": 40e-6,
            "gmax_t_m": 0.032,
            "smax_t_m_s": 130.0,
        },
        versions={
            p: importlib.metadata.version(p)
            for p in ["numpy", "pypulseq", "komamripy", "juliacall"]
        },
        julia_version=str(jl.seval("VERSION")),
        koma_version=str(jl.seval("pkgversion(KomaMRI)")),
        limitations=[
            "2D static spins, non-slice-selective RF",
            "uniform T1/T2 and one receive coil",
            "prescribed B0 map, no susceptibility solver",
            "no diffusion, motion, eddy currents, or gradient errors",
            "same B0=0 reconstruction used for every case; no distortion correction",
        ],
    )
    np.savez_compressed(
        output / "data.npz",
        x_m=x,
        y_m=y,
        proton_density=pd,
        b0_hz=b0,
        adc_times_s=t_adc,
        k_adc_cycles_m=k_adc,
        reference_signal=reference,
        **{f"signal_{k}": v for k, v in raw.items()},
        **{f"image_{k}": v for k, v in images.items()},
        **{f"kspace_{k}": v for k, v in kspaces.items()},
    )
    (output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output / "validation.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    plot_results(pd, b0, images, raw, t_adc, n, ss, meta, output)
    print(json.dumps(checks, indent=2), flush=True)
    print(f"Results: {output}", flush=True)
    if not checks["passed"]:
        raise RuntimeError("Physics validation failed; inspect validation.json")


if __name__ == "__main__":
    main()
