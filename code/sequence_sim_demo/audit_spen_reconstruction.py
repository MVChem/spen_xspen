"""Compare sequence and encoding simulations without changing archived results.

Run with the workspace .venv (torch/nibabel), not this project's Julia .venv.
Reuses saved finite-RF responses; no new Bloch integration or invented artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "code/spenpy"))
sys.path.insert(0, str(REPO / "code/spen_diffusion_recons/scripts/prior192"))
from spenpy._legacy.core import calcInvA
from phase_inva import fit_tiny_phase_scanner_batch, apply_even_phase


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normerr(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def matrices(n, fov, r_value):
    """Legacy matrix convention adapted to the known sequence geometry.

    Negative quadratic phase; increasing y focus; first line at pixel centre.
    Divide cm voxel integrals by voxel width to match saved voxel-average units.
    """
    length = fov * 100
    a = -2 * np.pi * r_value / length**2
    inv, forward = calcInvA(a, length, n, 0., 1, .5, .8)
    odd, _ = calcInvA(a, length, n // 2, 0., 1, .25, .8)
    even, _ = calcInvA(a, length, n // 2, 0., 1, .75, .8)
    return tuple(v.resolve_conj().numpy() / width for v, width in (
        (forward, length/n), (inv, length/n),
        (odd, 2*length/n), (even, 2*length/n)))


def solve_y(data, a, relative_lambda):
    # [kx, acquired y, object y]; all kx matrices use actual ADC times.
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    filt = s / (s*s + relative_lambda*s[:, :1]**2)
    coeff = np.einsum("kij,ki->kj", u.conj(), data.T)
    return np.einsum("kji,kj->ik", vh.conj(), filt*coeff)


def ro_inverse(data, fourier):
    return np.linalg.solve(fourier, data.T).T


def project(image, a, fourier):
    encoded_x = image @ fourier.T
    return np.einsum("kij,jk->ik", a, encoded_x)


def gain_from_observation(image, observation, a, fourier):
    predicted = project(image, a, fourier)
    gain = np.vdot(predicted, observation) / np.vdot(predicted, predicted)
    return image * gain, [float(gain.real), float(gain.imag)]


def phase_legacy(ro, inv, odd, even):
    data = torch.from_numpy(ro.astype(np.complex64))[:, :, None, None]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260917)
        phase = fit_tiny_phase_scanner_batch(
            data[None], torch.tensor(odd, dtype=torch.complex64),
            torch.tensor(even, dtype=torch.complex64), steps=300)[0]
    corrected = apply_even_phase(data, phase)[:, :, 0, 0].numpy()
    return inv @ corrected, phase.numpy(), corrected


def native_phantom(info, x, y):
    """Sample original anatomy at the SAME Gauss nodes used by saved RF states."""
    source = Path(info["source_path"])
    if sha(source) != info["source_sha256"]:
        raise ValueError("Anatomy source hash changed")
    nii = nib.as_closest_canonical(nib.load(source))
    plane = nii.get_fdata()[:, :, info["canonical_slice_indices_zero_based"][0]].T
    plane /= info["normalization_value"]
    dx, dy = np.array(nii.header.get_zooms()[:2], dtype=float) * 1e-3
    xx, yy = np.meshgrid(x, y)
    coords = np.array([yy/dy + (plane.shape[0]-1)/2,
                       xx/dx + (plane.shape[1]-1)/2])
    return map_coordinates(plane, coords, order=1, mode="nearest", prefilter=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=ROOT/"runs/brain_spen_xspen_260917")
    parser.add_argument("--output", type=Path,
                        default=ROOT/"runs/spen_reconstruction_audit_260917")
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT/"runs"):
        parser.error("Output must be within this project's runs/")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    folder = args.run / "spen"
    d = np.load(folder/"data.npz")
    meta = json.loads((folder/"metadata.json").read_text())
    provenance = json.loads((args.run/"metadata.json").read_text())
    n, ss = meta["matrix"][0], meta["spatial_oversampling"]
    fov, r_value = meta["fov_m"], meta["r_value"]
    pw = d["inplane_quadrature_weights"]
    x, y, kx = d["x_m"], d["y_m"], d["kx_cycles_m"]
    q = np.rint(kx*fov + n/2 - .5).astype(int)
    order = np.argsort(q.reshape(n, n), axis=1)
    idx = np.arange(n)[:, None]*n + order
    frequencies = np.array([np.mean(kx[q == k]) for k in range(n)])
    fine_fourier = np.exp(-2j*np.pi*frequencies[:, None]*x[None])
    fourier = fine_fourier.reshape(n, n, ss) @ pw
    kernel = d["kernel_ideal"]
    a_seq = (kernel[idx].reshape(n, n, n, ss) @ pw).transpose(1, 0, 2)
    data_seq = d["signal_ideal"][idx]
    target = d["effective_m0"].reshape(n, ss, n, ss).mean(axis=(1, 3))

    fine = native_phantom(provenance["input"], x, y)
    spatial_weights = np.tile(pw, n)
    fine_ro = fine @ (fine_fourier*spatial_weights[None]).T
    signal_fine = np.sum(kernel*fine_ro[:, q].T*spatial_weights[None], axis=1)
    target_fine = np.einsum("iajb,a,b->ij", fine.reshape(n, ss, n, ss), pw, pw)
    a_legacy, inv, odd, even = matrices(n, fov, r_value)
    a_enc = np.broadcast_to(a_legacy, (n, n, n)).copy()
    data_enc = project(target, a_enc, fourier)

    # Check independent legacy voxel integral approximation against quadrature.
    t_center = (np.arange(n)+.5)*meta["echo_spacing_s"]
    phase = (-2*np.pi*r_value/fov**2*y[None]**2
             -2*np.pi*r_value/fov*y[None]
             -2*np.pi*meta["read_gradient_hz_m"]*t_center[:, None]*y[None])
    exact_ideal = np.exp(1j*phase).reshape(n, n, ss) @ pw
    legacy_quadrature_error = normerr(a_legacy, exact_ideal)
    if legacy_quadrature_error > .01:
        raise AssertionError("Legacy and sequence ideal geometry do not agree")

    cases = [
        ("encoding_matched", "Encoding A (same 64 x 64 brain)", data_enc, a_enc, target),
        ("sequence_matched", "Sequence: averaged voxel phantom", data_seq, a_seq, target),
        ("sequence_native", "Sequence: original anatomy within voxels", signal_fine[idx], a_seq, target_fine),
    ]
    metrics, arrays, images = {}, {}, {}
    wavelengths = .8*n*n/(2*r_value)
    yc = y.reshape(n, ss).mean(axis=1)
    focus_seq = -fov/2 + fov*d["readout_times_s"][idx].T/meta["acquisition_window_s"]
    focus_enc = np.broadcast_to(yc[None], (n, n))
    for key, label, data, a, truth in cases:
        print(f"Reconstructing {key}", flush=True)
        ro = ro_inverse(data, fourier)
        rec = {"target": truth.astype(complex)}
        rec["inverse"] = ro_inverse(solve_y(data, a, 0.), fourier)
        rec["tikhonov"] = ro_inverse(solve_y(data, a, .01), fourier)
        focus = focus_enc if key == "encoding_matched" else focus_seq
        weights = np.exp(-((focus[:, :, None]-yc[None, None])*n/fov)**2/(2*wavelengths**2))
        weighted = np.einsum("kij,ik->jk", (a*weights).conj(), data)
        rec["matched_windowadj"], gain_window = gain_from_observation(
            ro_inverse(weighted, fourier), data, a, fourier)
        rec["legacy_inva"], gain_inv = gain_from_observation(inv@ro, data, a_enc, fourier)
        phase_image, phase_map, corrected = phase_legacy(ro, inv, odd, even)
        corrected_data = corrected @ fourier.T
        rec["legacy_phase"], gain_phase = gain_from_observation(
            phase_image, corrected_data, a_enc, fourier)
        # The old ideal encoding has no T2. Remove the KNOWN simulation's
        # uniform free-transverse decay before comparing image artifacts.
        # Finite RF rotates magnetization during the pulse, so this conventional
        # scalar-time compensation is approximate, unlike the finite-RF A.
        compensation = (np.ones_like(data.real) if key == "encoding_matched" else
                        np.exp((d["adc_times_s"][idx]-meta["excitation_center_s"])/meta["t2_s"]))
        t2_data = data*compensation
        t2_ro = ro_inverse(t2_data, fourier)
        rec["legacy_inva_t2"], gain_t2 = gain_from_observation(inv@t2_ro, t2_data, a_enc, fourier)
        t2_phase_image, t2_phase, t2_corrected = phase_legacy(t2_ro, inv, odd, even)
        rec["legacy_phase_t2"], gain_t2_phase = gain_from_observation(
            t2_phase_image, t2_corrected@fourier.T, a_enc, fourier)
        metrics[key] = {name: {"magnitude_nrmse": normerr(np.abs(value), truth),
                               "relative_signal_residual": normerr(project(value, a, fourier), data)}
                        for name, value in rec.items() if name != "target"}
        # Phase-corrected reconstructions refer to corrected, not original data.
        metrics[key]["legacy_phase"]["relative_corrected_signal_residual_legacy_model"] = normerr(
            project(rec["legacy_phase"], a_enc, fourier), corrected_data)
        metrics[key]["gains"] = dict(window=gain_window, legacy=gain_inv, phase=gain_phase,
                                     legacy_t2=gain_t2, phase_t2=gain_t2_phase)
        metrics[key]["estimated_phase_rms_rad"] = float(np.sqrt(np.mean(phase_map**2)))
        metrics[key]["phase_magnitude_preservation_error"] = float(np.max(np.abs(np.abs(corrected)-np.abs(ro))))
        images[key] = rec
        arrays.update({f"{key}_{name}": value for name, value in rec.items()})
        arrays.update({f"{key}_phase_rad": phase_map, f"{key}_data": data,
                       f"{key}_t2_phase_rad": t2_phase})
        print(json.dumps(metrics[key]), flush=True)

    checks = {
        "saved_signal_reproduction_relative_error": normerr(project(target, a_seq, fourier), data_seq),
        "saved_reconstruction_reproduction_relative_error": normerr(images["sequence_matched"]["tikhonov"], d["image_ideal"]),
        "legacy_matrix_vs_ideal_quadrature_relative_error": legacy_quadrature_error,
        "strict_inverse_matched_encoding_error": normerr(images["encoding_matched"]["inverse"], target),
        "strict_inverse_matched_sequence_error": normerr(images["sequence_matched"]["inverse"], target),
        "native_vs_averaged_signal_relative_difference": normerr(signal_fine[idx], data_seq),
        "native_vs_averaged_target_relative_difference": normerr(target_fine, target),
    }
    checks["passed"] = bool(all(checks[k] < 1e-9 for k in (
        "saved_signal_reproduction_relative_error", "saved_reconstruction_reproduction_relative_error",
        "strict_inverse_matched_encoding_error", "strict_inverse_matched_sequence_error")))
    if not checks["passed"]:
        raise AssertionError(checks)

    names = ["target", "tikhonov", "matched_windowadj", "legacy_inva_t2", "legacy_phase_t2"]
    titles = ["Input voxel mean", "Tikhonov (matched A)", "Windowed adjoint (matched A)",
              "Legacy InvA + known T2 correction", "PhaseMap + InvA + known T2 correction"]
    vmax = float(np.quantile(target[target > 0], .995))
    fig, axes = plt.subplots(3, 5, figsize=(18, 11), layout="constrained")
    for row, (key, label, *_rest) in enumerate(cases):
        for col, (name, title) in enumerate(zip(names, titles)):
            value = np.abs(images[key][name])
            axes[row, col].imshow(value, origin="lower", cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
            error = "" if name == "target" else f"\nNRMSE {metrics[key][name]['magnitude_nrmse']:.3f}"
            axes[row, col].set_title(title+error, fontsize=10)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(label, fontsize=10)
    fig.suptitle("SPEN reconstruction audit | identical 64 x 64 geometry, B0=0, no noise, one coil\n"
                 "Same display scale; InvA gains fitted from observations only; no planted artifacts", fontsize=12)
    fig.savefig(out/"comparison.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(15, 10), layout="constrained")
    for row, (key, label, *_rest) in enumerate(cases):
        for col, name in enumerate(names[1:]):
            error = np.abs(images[key][name])-np.abs(images[key]["target"])
            axes[row, col].imshow(error, origin="lower", cmap="RdBu_r", vmin=-.2*vmax, vmax=.2*vmax, interpolation="nearest")
            axes[row, col].set_title(titles[col+1], fontsize=10)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if col == 0: axes[row, col].set_ylabel(label, fontsize=10)
    fig.suptitle("Signed magnitude errors | common range +/-20% of image window\nRed: excess signal; blue: missing signal", fontsize=12)
    fig.savefig(out/"errors.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    for col, (key, label, data, a, truth) in enumerate(cases):
        xp = n//2+n//8
        for name in ["target", "tikhonov", "legacy_inva_t2", "legacy_phase_t2"]:
            axes[0, col].plot(np.abs(images[key][name])[:, xp], label=name, lw=1.3)
        axes[0, col].set(title=label, xlabel="Encoding y pixel", ylabel=f"Magnitude at x={xp}")
        axes[0, col].legend(fontsize=8)
        u, s, vh = np.linalg.svd(a[n//2], full_matrices=False)
        axes[1, col].semilogy(s/s[0], ".-", label="singular values / max")
        axes[1, col].axhline(.1, color="red", ls="--", label="sqrt(relative lambda)=0.1")
        axes[1, col].set(xlabel="Mode index", ylabel="Relative singular value")
        axes[1, col].legend(fontsize=8)
    fig.savefig(out/"profiles_spectrum.png", dpi=170)
    plt.close(fig)

    # Explicitly show the response of the old approximate inverse to impulses.
    response = inv @ a_legacy
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for j in [n//4, n//2, 3*n//4]:
        profile = response[:, j]/response[j, j]
        axes[0].plot(np.arange(n)-j, np.abs(profile), label=f"pixel {j}")
        axes[1].plot(np.arange(n)-j, profile.real, label=f"pixel {j}")
    for ax in axes:
        ax.set(xlim=(-14, 14), xlabel="Encoding y displacement (pixels)")
        ax.legend(); ax.grid(alpha=.2)
    axes[0].set(title="Legacy InvA @ A: magnitude", ylabel="Normalized response")
    axes[1].set(title="Legacy InvA @ A: real component", ylabel="Normalized response")
    fig.savefig(out/"inva_point_response.png", dpi=170)
    plt.close(fig)

    report = {
        "source_run": str(args.run.resolve()), "checks": checks, "metrics": metrics,
        "parameters": dict(n=n, fov_m=fov, chirp_R=r_value, relative_lambda=.01, gauss_width=.8,
                           noise=0, injected_even_odd_phase=0, phase_steps=300),
        "scope": [
            "SPEN180 audit only; xSPEN uses a different physical kernel",
            "Encoding control uses legacy calcInvA but matches current human geometry, not old mouse protocol",
            "Sequence states/kernel reused from completed Bloch run; native anatomy sampled directly at saved Gauss nodes",
            "Native anatomy is not preaveraged to reconstruction grid; finite quadrature is still an approximation",
            "Tikhonov/matched-windowadj use finite-RF/actual ADC A for sequence rows; legacy uses ideal line-centre A",
            "InvA gains are fitted to measured/corrected observations with no target fitting",
            "Main display uses known uniform T2 timing compensation for legacy; uncorrected results are also saved",
            "T2 timing compensation is approximate during finite RF and is not claimed to match finite-RF amplitude exactly",
            "PhaseMap is the existing prior192 scanner batch fitting function, generalized through its native arbitrary-size interface",
            "Residual against original data includes intentional PhaseMap change; corrected-data legacy residual also supplied",
            "No scanner calibration or exact old mouse-protocol replay claimed",
        ],
        "hashes": {str(path.resolve()): sha(path) for path in [Path(__file__), folder/"data.npz",
            folder/"metadata.json", args.run/"metadata.json", ROOT/"demo_brain_spen_xspen.py",
            REPO/"code/spenpy/spenpy/_legacy/core/matrix.py",
            REPO/"code/spen_diffusion_recons/scripts/prior192/phase_inva.py"]},
    }
    (out/"audit.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(out/"arrays.npz", **arrays, native_anatomy=fine,
                        a_sequence=a_seq, a_legacy=a_legacy, inva_response=response)
    source = out/"source"; source.mkdir()
    for path in [Path(__file__), ROOT/"demo_brain_spen_xspen.py",
                 REPO/"code/spenpy/spenpy/_legacy/core/matrix.py",
                 REPO/"code/spen_diffusion_recons/scripts/prior192/phase_inva.py"]:
        shutil.copy2(path, source/path.name)
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
