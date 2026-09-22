"""Re-run SPEN RF Bloch on finer Gauss nodes for the independent-anatomy audit."""
import argparse
import json
from pathlib import Path
import shutil

from demo_brain_spen_xspen import Bloch, forward, readout_kernel, reconstruct
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT/"runs/brain_spen_xspen_260917")
    p.add_argument("--audit", type=Path, default=ROOT/"runs/spen_reconstruction_audit_t2_260917")
    args = p.parse_args()
    folder = args.run/"spen"
    meta = json.loads((folder/"metadata.json").read_text())
    info = json.loads((args.run/"metadata.json").read_text())["input"]
    audit = np.load(args.audit/"arrays.npz")
    saved = np.load(folder/"data.npz")
    n, fov = meta["matrix"][0], meta["fov_m"]
    q = np.rint(saved["kx_cycles_m"]*fov+n/2-.5).astype(int)
    idx = np.arange(n)[:, None]*n+np.argsort(q.reshape(n, n), axis=1)
    nii = nib.as_closest_canonical(nib.load(info["source_path"]))
    plane = nii.get_fdata()[:, :, info["canonical_slice_indices_zero_based"][0]].T
    plane /= info["normalization_value"]
    dx, dy = np.asarray(nii.header.get_zooms()[:2], dtype=float)*1e-3
    engine = Bloch()
    prefix = engine.km.read_seq(str(folder/"encoding_prefix.seq"))
    previous_signal = audit["sequence_native_data"]
    previous_image = audit["sequence_native_tikhonov"]
    previous_ss = meta["spatial_oversampling"]
    reports, arrays = {}, {}
    for ss in [16, 24]:
        nodes, weights = np.polynomial.legendre.leggauss(ss)
        pw = weights/2
        axis = ((np.arange(n)[:, None]+.5-n/2+nodes[None]/2)*fov/n).ravel()
        xx, yy = np.meshgrid(axis, axis)
        phantom = map_coordinates(plane, np.array([yy/dy+(plane.shape[0]-1)/2,
                                                   xx/dx+(plane.shape[1]-1)/2]),
                                  order=1, mode="nearest", prefilter=False)
        state = engine.simulate(prefix, np.zeros_like(axis), axis,
                                np.zeros_like(axis), np.ones_like(axis),
                                np.zeros_like(axis), meta["t1_s"], meta["t2_s"],
                                meta["rf_raster_s"], state=True)[:, None]
        kernel = readout_kernel(state, axis, np.zeros(1), np.ones(1),
                                np.zeros_like(axis), saved["readout_times_s"], meta, meta["t2_s"])
        signal, fourier = forward(phantom, kernel, axis, saved["kx_cycles_m"], q, ss, pw)
        image, _, _ = reconstruct(signal, kernel, fourier, q, n, ss, .01, pw)
        truth = np.einsum("iajb,a,b->ij", phantom.reshape(n, ss, n, ss), pw, pw)
        error = lambda a, b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
        reports[str(ss)] = {
            "previous_nodes_per_voxel": previous_ss,
            "signal_relative_change": error(signal[idx], previous_signal),
            "complex_reconstruction_relative_change": error(image, previous_image),
            "magnitude_nrmse": error(np.abs(image), truth),
        }
        arrays.update({f"signal_{ss}": signal, f"image_{ss}": image, f"truth_{ss}": truth})
        previous_signal, previous_image, previous_ss = signal[idx], image, ss
        print(json.dumps(reports[str(ss)]), flush=True)
    result = {"quadrature": reports,
              "passed": reports["24"]["signal_relative_change"] < .002
              and reports["24"]["complex_reconstruction_relative_change"] < .01}
    destination = args.audit/"quadrature_validation.json"
    with destination.open("x") as f:
        f.write(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(args.audit/"quadrature_validation.npz", **arrays)
    shutil.copy2(__file__, args.audit/"source"/Path(__file__).name)
    if not result["passed"]:
        raise AssertionError(result)


if __name__ == "__main__":
    main()
