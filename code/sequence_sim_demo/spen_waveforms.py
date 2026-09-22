"""Finite-RF SPEN180 / crossed-chirp xSPEN, adapted from local archived code.

This is a source-informed Pulseq implementation, not a scanner binary replay.
All gradient amplitudes are in Hz/m and RF amplitudes are in Hz.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pypulseq as pp

LEGACY = Path("/home/data2/chk/workspace/2026/08/14/xSPEN_项目")
SOURCES = {
    "cross_term_equation": LEGACY
    / "00_xSPEN_仿真与训练准备/前向物理模型与旧理论仿真/xSPEN1D.m",
    "chirp_waveform": LEGACY
    / "04_PE_xSPEN_7T_and_Multiband/7T_VE12U_sequences_30052022/mo_SPEN_xSPEN/SBBChirp.cpp",
    "crossed_gradient_order": LEGACY
    / "09_Bruker_PV6.0.1_Custom_Method_Snapshots/xSPEN_scan18/pulseprogram",
    "spen_full_refocusing": LEGACY
    / "00_xSPEN_仿真与训练准备/前向物理模型与旧理论仿真/ss90_chirp180_diffMode1.m",
}


def constant(channel, amplitude, duration, system):
    return pp.make_extended_trapezoid(
        channel, times=[0, duration], amplitudes=[amplitude, amplitude], system=system
    )


def build_sequence(kind, n=64, fov=0.24, r_value=64, thickness=0.006):
    if kind not in ("spen", "xspen"):
        raise ValueError(kind)
    system = pp.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=20e-6,
        adc_dead_time=10e-6,
        rf_raster_time=1e-6,
        grad_raster_time=10e-6,
    )
    seq = pp.Sequence(system)
    rise, dwell, pre_time = 100e-6, 10e-6, 1e-3
    gx = pp.make_trapezoid(
        "x",
        amplitude=1 / (fov * dwell),
        flat_time=n * dwell,
        rise_time=rise,
        system=system,
    )
    esp = pp.calc_duration(gx)
    ta = n * esp
    tp = ta / 2  # beta=0.5 in xSPEN; full-refocusing 180 chirp in SPEN.
    bw = r_value / tp
    gy = bw / fov * (0.5 if kind == "xspen" else 1)
    gz = 0.5 * bw / thickness if kind == "xspen" else 0.0
    # Adiabaticity 2*pi*B1^2/(BW/Tp) = 6.4 at the sweep centre.
    peak = np.sqrt(6.4 * bw / (2 * np.pi * tp))
    t = (np.arange(round(tp / system.rf_raster_time)) + 0.5) * system.rf_raster_time
    envelope = 1 - np.abs(np.cos(np.pi * t / tp)) ** 40
    waveform = peak * envelope * np.exp(1j * np.pi * bw / tp * (t - tp / 2) ** 2)
    chirp = pp.make_arbitrary_rf(
        waveform,
        np.pi,
        dwell=system.rf_raster_time,
        no_signal_scaling=True,
        delay=rise,
        system=system,
        use="refocusing",
    )
    chirp_duration = tp + 2 * rise
    chirp_centers = []

    def add_background(*events):
        duration = pp.calc_duration(*events)
        seq.add_block(*events, constant("z", gz, duration, system))

    def add_chirp(sign):
        chirp_centers.append(seq.duration()[0] + rise + tp / 2)
        grad = pp.make_trapezoid(
            "y", amplitude=sign * gy, flat_time=tp, rise_time=rise, system=system
        )
        if kind == "xspen":
            add_background(chirp, grad)
        else:
            seq.add_block(chirp, grad)

    pre_x = pp.make_trapezoid("x", area=-gx.area / 2, duration=pre_time, system=system)
    if kind == "spen":
        rf90 = pp.make_block_pulse(
            np.pi / 2, duration=200e-6, delay=rise, system=system, use="excitation"
        )
        seq.add_block(rf90)
        excitation_center = rise + 100e-6
        delay = tp + pre_time + rise + excitation_center - seq.duration()[0]
        seq.add_block(pp.make_delay(round(delay / 1e-5) * 1e-5))
        add_chirp(1)
        pre_y = pp.make_trapezoid(
            "y", area=r_value / fov + gy * rise / 2, duration=pre_time, system=system
        )
        seq.add_block(pre_x, pre_y)
        seq.add_block(
            pp.make_extended_trapezoid(
                "y", times=[0, rise], amplitudes=[0, -gy], system=system
            )
        )
        read_axis, read_gradient = "y", -gy
    else:
        # Slice-selective 90 followed by two equal sweeps under -Gy,+Gy.
        # Gz is constant from excitation through both sweeps and acquisition.
        rf90_duration = round((4 / (0.5 * bw)) / 1e-5) * 1e-5
        rf90 = pp.make_sinc_pulse(
            np.pi / 2,
            duration=rf90_duration,
            time_bw_product=0.5 * bw * rf90_duration,
            apodization=0.5,
            delay=rise,
            system=system,
            use="excitation",
        )
        first_duration = pp.calc_duration(rf90)
        bg_start = pp.make_extended_trapezoid(
            "z", times=[0, rise, first_duration], amplitudes=[0, gz, gz], system=system
        )
        seq.add_block(rf90, bg_start)
        excitation_center = rise + rf90_duration / 2
        first_end = seq.duration()[0]
        add_chirp(-1)
        gap = first_end - excitation_center + pre_time + ta / 2
        add_background(pp.make_delay(round(gap / 1e-5) * 1e-5))
        add_chirp(1)
        add_background(pre_x)
        read_axis, read_gradient = "z", gz

    read_start = float(seq.duration()[0])
    prefix = deepcopy(seq)
    adc = pp.make_adc(n, dwell=dwell, delay=rise, system=system)
    for row in range(n):
        read = pp.make_trapezoid(
            "x",
            amplitude=(-1) ** row * gx.amplitude,
            flat_time=gx.flat_time,
            rise_time=rise,
            system=system,
        )
        seq.add_block(read, adc, constant(read_axis, read_gradient, esp, system))
    seq.add_block(
        pp.make_extended_trapezoid(
            read_axis, times=[0, rise], amplitudes=[read_gradient, 0], system=system
        )
    )
    seq.set_definition("Name", f"local_source_{kind}_brain_demo")
    seq.set_definition("FOV", [fov, fov, thickness])
    seq.set_definition("R", r_value)
    seq.set_definition("WaveformScope", "source-informed demo; not scanner replay")
    for name, item in [("full", seq), ("prefix", prefix)]:
        ok, errors = item.check_timing()
        if not ok:
            raise RuntimeError(f"{kind} {name} timing check failed: {errors}")
    # Readout x moment is affected only by the post-RF prephaser and Gx train.
    k_adc, _, _, _, t_adc = seq.calculate_kspace()
    t_adc = np.asarray(t_adc)
    u = t_adc - read_start
    kx = k_adc[0]
    q = np.rint(kx * fov + n / 2 - 0.5).astype(int)
    if np.max(np.abs(kx * fov - (q - n / 2 + 0.5))) > 1e-6:
        raise RuntimeError("Readout samples not on expected Cartesian x lattice")
    if any(np.unique(row).size != n for row in q.reshape(n, n)):
        raise RuntimeError("Readout rows must cover all kx samples")
    centers = np.asarray(chirp_centers)
    b0_time_start = (
        read_start - 2 * centers[0] + excitation_center
        if kind == "spen"
        else read_start - 2 * centers[1] + 2 * centers[0] - excitation_center
    )
    meta = {
        "family": kind,
        "matrix": [n, n],
        "fov_m": fov,
        "r_value": r_value,
        "echo_spacing_s": esp,
        "acquisition_window_s": ta,
        "chirp_duration_s": tp,
        "chirp_bandwidth_hz": bw,
        "rf_peak_hz": float(peak),
        "rf_peak_uT": float(peak / system.gamma * 1e6),
        "rf_raster_s": system.rf_raster_time,
        "wurst_exponent": 40,
        "beta": 0.5 if kind == "xspen" else None,
        "slice_thickness_m": thickness,
        "encoding_gy_hz_m": gy,
        "background_gz_hz_m": gz,
        "read_gradient_hz_m": read_gradient,
        "read_gradient_axis": read_axis,
        "excitation_center_s": excitation_center,
        "chirp_centers_s": centers.tolist(),
        "readout_start_s": read_start,
        "readout_end_s": read_start + ta,
        "center_acquisition_time_after_excitation_s": float(
            np.mean(t_adc) - excitation_center
        ),
        "b0_effective_time_at_readout_start_s": float(b0_time_start),
        "full_refocusing_timing_error_s": float(b0_time_start + ta / 2),
        "expected_quadratic_rad_m2": -2 * np.pi * r_value / fov**2,
        "expected_cross_term_rad_m2": -8 * np.pi * tp / bw * gy * gz,
        "timing_passed": True,
        "chirp_block_duration_s": chirp_duration,
    }
    return seq, prefix, t_adc, u, kx, q, meta


def plot_waveforms(seq, t_adc, output):
    import matplotlib.pyplot as plt

    waves, *_ = seq.waveforms_and_times(append_RF=True)
    fig, axes = plt.subplots(5, 1, figsize=(11, 8), sharex=True, layout="constrained")
    for ax, wave, name in zip(
        axes[:4],
        [waves[3], *waves[:3]],
        ["RF (Hz)", "Gx (mT/m)", "Gy (mT/m)", "Gz (mT/m)"],
    ):
        if np.size(wave):
            values = (
                np.abs(wave[1])
                if name.startswith("RF")
                else wave[1].real / seq.system.gamma * 1e3
            )
            ax.plot(wave[0].real * 1e3, values, lw=0.8)
        ax.set_ylabel(name)
        ax.grid(alpha=0.2)
    axes[-1].plot(t_adc * 1e3, np.ones_like(t_adc), "|", ms=3)
    axes[-1].set(xlabel="Time (ms)", ylabel="ADC")
    axes[0].set_title(seq.get_definition("Name"))
    fig.savefig(output, dpi=160)
    plt.close(fig)
